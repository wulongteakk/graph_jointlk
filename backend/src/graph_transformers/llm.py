import json
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.graphs import Neo4jGraph

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.graph_document import GraphDocument

from src.diffbot_transformer import get_graph_from_diffbot
from src.openAI_llm import get_graph_from_OpenAI # 替换
from src.gemini_llm import get_graph_from_Gemini # 替换
from src.shared.constants import *
from src.evidence_store.sqlite_store import EvidenceStore
from src.evidence_store.text_span import find_best_span, split_sentences_with_offsets, pick_span_sentence
# from src.llm import get_graph_from_llm as old_get_graph_from_llm
from src.llm import get_graph_from_llm, get_llm


logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")





def _strip_json_fence(content: str) -> str:
    if "```json" in content:
        return content.split("```json")[1].split("```")[0]
    if "```" in content:
        return content.replace("```", "").strip()
    return content


def _parse_classification_payload(raw_content: str) -> Optional[Any]:
    """兼容 LLM 返回的 ```json / 前后夹带解释文本 / 单对象或对象数组。"""
    content = _strip_json_fence(raw_content).strip()
    if not content:
        return None

    try:
        return json.loads(content)
    except Exception:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", content, flags=re.S)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def classify_risk_factor_hfcsa_with_llm(
    risk_factor: str,
    context: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    try:
        llm, _ = get_llm(model)
        prompt = ChatPromptTemplate.from_template(
            HFCSA_CONTROLLED_CLASSIFICATION_PROMPT_TEMPLATE
            + "\n\n待分类 RiskFactor: \"{risk_factor}\"\n\n原文片段:\n{context}\n"
        )
        chain = prompt | llm
        response = chain.invoke({"risk_factor": risk_factor, "context": context})
        data = _parse_classification_payload(response.content)
        if data is None:
            raise ValueError(f"无法从返回中解析 JSON: {response.content[:200]}")
    except Exception as e:
        logging.warning(f"LLM 受控分类失败，RiskFactor: {risk_factor}, 错误: {e}")
        return None

    if isinstance(data, dict):
        data = [data]
    if not data:
        return None

    result = data[0]
    return {
        "layer_code": result.get("layer_code"),
        "category_code": result.get("category_code"),
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "evidence": result.get("evidence"),
    }


def _is_risk_factor_node(node: Any) -> bool:
    """扩大候选匹配范围，避免受控分类覆盖为 0。"""
    node_type = (node.type or "").strip()
    node_type_lower = node_type.lower()
    risk_types = {
        "riskfactor", "风险因子", "风险", "隐患", "不安全行为", "管理缺陷", "前提条件", "人因", "cause",
        "hazardsource", "hazard_source", "hazard", "hazardouscondition", "unsafeact", "unsafecondition",
        # Stage1 升级 schema 的常见类型
        "barrier", "resourcecondition", "humanstate", "managementaction", "accidentevent", "loss",
        "unsafeact", "standardclause",
    }
    if node_type_lower in risk_types or node_type in risk_types:
        return True

    text = f"{_get_node_display_name(node)} {(node.properties or {}).get('description', '')}".lower()
    risk_keywords = [
        "风险", "隐患", "违规", "违章", "未", "缺失", "失效", "故障", "坠落", "打击", "坍塌", "触电", "起重",
        "unsafe", "hazard", "incident", "accident", "injury", "death",
    ]
    return any(kw in text for kw in risk_keywords)


def _get_node_display_name(node: Any) -> str:
    if node.properties:
        return node.properties.get("name") or node.properties.get("id") or node.id
    return node.id




LAYER_FALLBACK_BY_TYPE = {
    "managementaction": ("S", "S2"),
    "standardclause": ("O", "O1"),
    "resourcecondition": ("C", "C2"),
    "barrier": ("C", "C3"),
    "humanstate": ("C", "C6"),
    "hazardsource": ("C", "C2"),
    "unsafeact": ("A", "A2"),
    "accidentevent": ("E", None),
    "loss": ("L", None),
}


def _fallback_classification(node: Any) -> Optional[Dict[str, Any]]:
    node_type = (node.type or "").strip().lower()
    if node_type in LAYER_FALLBACK_BY_TYPE:
        layer_code, category_code = LAYER_FALLBACK_BY_TYPE[node_type]
        return {
            "layer_code": layer_code,
            "category_code": category_code,
            "confidence": 0.45,
            "reason": f"fallback_by_node_type:{node.type}",
            "evidence": _get_node_display_name(node),
        }

    name = _get_node_display_name(node)
    lower = (name or "").lower()
    if any(k in lower for k in ["坠落", "打击", "坍塌", "触电", "起重", "事故", "事件"]):
        return {"layer_code": "E", "category_code": None, "confidence": 0.4, "reason": "fallback_by_event_keyword", "evidence": name}
    if any(k in lower for k in ["死亡", "重伤", "轻伤", "损失", "伤亡"]):
        return {"layer_code": "L", "category_code": None, "confidence": 0.4, "reason": "fallback_by_loss_keyword", "evidence": name}
    if any(k in lower for k in ["未", "违章", "违规", "冒险", "错误操作"]):
        return {"layer_code": "A", "category_code": "A2", "confidence": 0.4, "reason": "fallback_by_action_keyword", "evidence": name}
    return None


def _preview_causal_chain(graph_doc: GraphDocument) -> str:
    """在控制台展示每个 chunk 的简易因果链预览。"""
    node_map = {}
    for n in graph_doc.nodes:
        props = n.properties or {}
        node_map[n.id] = {
            "name": props.get("name") or n.id,
            "layer": props.get("layer_code"),
            "category": props.get("category_code"),
        }

    allowed = set(CTP_ALLOWED_TRANSITIONS)
    chain_edges: List[Tuple[str, str]] = []

    for rel in graph_doc.relationships:
        s = getattr(rel.source, 'id', None)
        t = getattr(rel.target, 'id', None)
        if not s or not t or s not in node_map or t not in node_map:
            continue
        sl = node_map[s]["layer"]
        tl = node_map[t]["layer"]
        if sl and tl and (sl, tl) in allowed:
            chain_edges.append((s, t))

    if not chain_edges:
        per_layer = {k: [] for k in ["O", "S", "C", "A", "E", "L"]}
        for nid, meta in node_map.items():
            layer = meta.get("layer")
            if layer in per_layer:
                per_layer[layer].append(nid)

        prev = None
        for layer in ["O", "S", "C", "A", "E", "L"]:
            if not per_layer[layer]:
                continue
            cur = per_layer[layer][0]
            if prev and (node_map[prev]["layer"], layer) in allowed:
                chain_edges.append((prev, cur))
            prev = cur

    if not chain_edges:
        return "[CTP预览] 当前 chunk 无法生成有效因果链（缺少可匹配层级节点）。"

    segments = []
    for s, t in chain_edges:
        sm = node_map[s]
        tm = node_map[t]
        segments.append(f"{sm['layer']}:{sm['name']} -> {tm['layer']}:{tm['name']}")
    return "[CTP预览] " + " | ".join(segments)


def _extract_chain_edges(graph_doc: GraphDocument) -> List[Tuple[str, str]]:
    node_layers = {n.id: (n.properties or {}).get("layer_code") for n in graph_doc.nodes}
    allowed = set(CTP_ALLOWED_TRANSITIONS)
    chain_edges: List[Tuple[str, str]] = []

    for rel in graph_doc.relationships:
        s = getattr(rel.source, "id", None)
        t = getattr(rel.target, "id", None)
        if not s or not t:
            continue
        sl = node_layers.get(s)
        tl = node_layers.get(t)
        if sl and tl and (sl, tl) in allowed:
            chain_edges.append((s, t))
    return chain_edges


def _persist_causal_chain(graph: Neo4jGraph, evidence_store: EvidenceStore, graph_doc: GraphDocument) -> None:
    md = graph_doc.source.metadata if (graph_doc.source and graph_doc.source.metadata) else {}
    chunk_id = md.get("combined_chunk_ids", [None])[0] if isinstance(md.get("combined_chunk_ids"), list) else md.get("combined_chunk_ids")
    file_name = md.get("fileName")
    if not chunk_id:
        return

    edges = _extract_chain_edges(graph_doc)
    if not edges:
        return

    node_map = {n.id: n for n in graph_doc.nodes}
    chain_digest = hashlib.sha1((chunk_id + "|" + "|".join([f"{s}->{t}" for s, t in edges])).encode()).hexdigest()
    chain_id = f"{chunk_id}::ctp::{chain_digest[:12]}"

    chain_text_segments = []
    edge_records = []
    for idx, (s, t) in enumerate(edges, start=1):
        s_node = node_map.get(s)
        t_node = node_map.get(t)
        s_props = (s_node.properties or {}) if s_node else {}
        t_props = (t_node.properties or {}) if t_node else {}
        sl = s_props.get("layer_code")
        tl = t_props.get("layer_code")
        sn = s_props.get("name") or s
        tn = t_props.get("name") or t
        chain_text_segments.append(f"{sl}:{sn}->{tl}:{tn}")
        edge_records.append(
            {
                "edge_id": f"{chain_id}::edge::{idx}",
                "seq": idx,
                "source_node_id": s,
                "source_layer": sl,
                "target_node_id": t,
                "target_layer": tl,
                "evidence_unit_id": f"{chunk_id}::chunk",
                "evidence_start": 0,
                "evidence_end": len(graph_doc.source.page_content) if graph_doc.source else None,
                "meta": {"file_name": file_name},
            }
        )

    chain_text = " | ".join(chain_text_segments)
    evidence_store.upsert_causal_chain(
        chain_id=chain_id,
        file_name=file_name,
        parent_evidence_id=chunk_id,
        chain_text=chain_text,
        chain_json={"edges": edge_records},
    )
    evidence_store.upsert_causal_chain_edges(chain_id=chain_id, edges=edge_records)

    query = """
    MERGE (c:Chunk {id: $chunk_id})
    MERGE (cc:CausalChain {id: $chain_id})
    SET cc.fileName = $file_name,
        cc.parent_evidence_id = $parent_evidence_id,
        cc.chain_text = $chain_text
    MERGE (c)-[:HAS_CAUSAL_CHAIN]->(cc)
    """
    graph.query(
        query,
        params={
            "chunk_id": chunk_id,
            "chain_id": chain_id,
            "file_name": file_name,
            "parent_evidence_id": chunk_id,
            "chain_text": chain_text,
        },
    )


def _build_classification_evidence_payload(node: Any, graph_doc: GraphDocument, evidence_text: str, evidence_store: EvidenceStore) -> Dict[str, Any]:
    md = graph_doc.source.metadata if (graph_doc.source and graph_doc.source.metadata) else {}
    chunk_id = md.get("combined_chunk_ids", [None])[0] if isinstance(md.get("combined_chunk_ids"), list) else md.get("combined_chunk_ids")
    context = graph_doc.source.page_content if graph_doc.source else ""

    start, end = find_best_span(context, evidence_text)
    if start is None:
        start, end = find_best_span(context, _get_node_display_name(node))

    sentence_unit_id = None
    if start is not None and end is not None and chunk_id:
        sents = split_sentences_with_offsets(context, max_len=400)
        if sents:
            sent_idx = pick_span_sentence(sents, start, end)
            sent = sents[sent_idx]
            sentence_unit_id = f"{chunk_id}::hfcsa::{hashlib.sha1(str(node.id).encode()).hexdigest()[:10]}"
            evidence_store.upsert_unit(
                unit_id=sentence_unit_id,
                parent_evidence_id=chunk_id,
                unit_kind="hfcsa_classification",
                content=sent.text,
                start_char=sent.start,
                end_char=sent.end,
                meta={
                    "node_id": node.id,
                    "node_type": node.type,
                    "file_name": md.get("fileName"),
                },
            )

    return {
        "evidence": evidence_text,
        "evidence_id": chunk_id,
        "evidence_start": start,
        "evidence_end": end,
        "evidence_unit_id": sentence_unit_id,
    }


def generate_graphDocuments(model: str, graph: Neo4jGraph, chunkId_chunkDoc_list: List, allowedNodes=None,
                            allowedRelationship=None) -> List[GraphDocument]:

    if not allowedNodes:
        allowedNodes = CONSTRUCTION_NODE_LABELS
    if not allowedRelationship:
        allowedRelationship = CONSTRUCTION_REL_TYPES

    if isinstance(allowedNodes, str) and allowedNodes:
        allowedNodes = allowedNodes.split(',')
    if isinstance(allowedRelationship, str) and allowedRelationship:
        allowedRelationship = allowedRelationship.split(',')

    logging.info(f"使用 Schema - 节点: {allowedNodes}, 关系: {allowedRelationship}")

    graph_documents = []
    if model == "diffbot":
        graph_documents = get_graph_from_diffbot(graph, chunkId_chunkDoc_list)

    elif model in OPENAI_MODELS:
        graph_documents = get_graph_from_OpenAI(model, graph, chunkId_chunkDoc_list, allowedNodes, allowedRelationship)

    elif model in GEMINI_MODELS:
        graph_documents = get_graph_from_Gemini(model, graph, chunkId_chunkDoc_list, allowedNodes, allowedRelationship)

    # elif model in GROQ_MODELS :
    #     graph_documents = get_graph_from_Groq_Llama3(MODEL_VERSIONS[model], graph, chunkId_chunkDoc_list, allowedNodes, allowedRelationship)

    else:
        graph_documents = get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship)


    logging.info(f"提取出来的graph_documents: {graph_documents}")

    logging.info("开始进行阶段二：风险因子受控分类...")
    total_hfcsa_enriched = 0
    evidence_store = EvidenceStore()

    for graph_doc in graph_documents:
        for node in graph_doc.nodes:
            if not _is_risk_factor_node(node):
                continue

            existing_props = node.properties or {}
            if existing_props.get("layer_code") and existing_props.get("category_code"):
                continue

            context = graph_doc.source.page_content if graph_doc.source else ""
            classification = classify_risk_factor_hfcsa_with_llm(
                _get_node_display_name(node),
                context,
                model,
            )
            if not classification:
                classification = _fallback_classification(node)

            if classification:
                evidence_payload = _build_classification_evidence_payload(
                    node,
                    graph_doc,
                    classification.get("evidence") or _get_node_display_name(node),
                    evidence_store,
                )
                classification.update({k: v for k, v in evidence_payload.items() if v is not None})
                existing_props.update(
                    {key: value for key, value in classification.items() if value is not None}
                )
                node.properties = existing_props
                total_hfcsa_enriched += 1

        logging.info(
            "风险因子受控分类完成。当前累计覆盖 %s 个节点。",
            total_hfcsa_enriched,
        )
        preview = _preview_causal_chain(graph_doc)
        logging.info(preview)
        print(preview)
        _persist_causal_chain(graph, evidence_store, graph_doc)
    logging.info(f"风险因子匹配后得到的graph_documents: {graph_documents}")
    return graph_documents