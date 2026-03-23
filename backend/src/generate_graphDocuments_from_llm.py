import logging
from typing import List, Tuple,Optional,Dict,Any

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument

from src.diffbot_transformer import get_graph_from_diffbot
from src.domain_packs.registry import get_domain_pack
from src.gemini_llm import get_graph_from_Gemini
from src.llm import get_graph_from_llm
from src.openAI_llm import get_graph_from_OpenAI
from src.overlay.hfsca_overlay import apply_hfsca_overlay
from src.shared.constants import GEMINI_MODELS, OPENAI_MODELS

logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")


def _preview_causal_chain(graph_doc: GraphDocument) -> str:
    """在控制台展示每个 chunk 的简易 HFCSA 投影视图。"""
    node_map = {}
    for n in graph_doc.nodes:
        props = n.properties or {}
        if not props.get("is_main_chain_candidate"):
            continue
        node_map[n.id] = {
            "name": props.get("name") or props.get("text") or n.id,
            "layer": props.get("hfsca_layer") or props.get("layer_code"),
            "category": props.get("hfsca_category") or props.get("category_code"),
        }

    chain_edges: List[Tuple[str, str]] = []
    for rel in graph_doc.relationships:
        source_id = getattr(getattr(rel, "source", None), "id", None)
        target_id = getattr(getattr(rel, "target", None), "id", None)
        rel_props = getattr(rel, "properties", None) or {}
        projection_type = rel_props.get("projection_type")
        if projection_type != "NEXT_LEVEL_CAUSES":
            continue
        if source_id in node_map and target_id in node_map:
            chain_edges.append((source_id, target_id))

    if not chain_edges:
        return "[HFCSA预览] 当前 chunk 无 NEXT_LEVEL_CAUSES 投影关系。"

    segments = []
    for source_id, target_id in chain_edges:
        source_meta = node_map[source_id]
        target_meta = node_map[target_id]
        segments.append(
            f"{source_meta['layer']}:{source_meta['name']} -> {target_meta['layer']}:{target_meta['name']}"
        )
    return "[HFCSA预览] " + " | ".join(segments)

def _filter_stage1_relationships(allowed_relationships: List[str]) -> List[str]:
    banned = {"NEXT_LEVEL_CAUSES", "SUPPORTED_BY", "CO_OCCURS_IN_EVIDENCE"}
    return [rel for rel in allowed_relationships if rel not in banned]


def _build_evidence_context(evidence_units: Optional[List[Dict[str, Any]]]) -> str:
    if not evidence_units:
        return ""
    lines = []
    for idx, unit in enumerate(evidence_units[:8], start=1):
        payload = unit.__dict__ if hasattr(unit, "__dict__") else unit
        lines.append(
            f"[EU{idx}] id={payload.get('unit_id')} kind={payload.get('unit_kind')} "
            f"triggers={payload.get('trigger_words') or []} text={payload.get('text') or payload.get('content')}"
        )
    return "\n候选证据片段:\n" + "\n".join(lines)


def _enrich_graph_document_metadata(graph_doc: GraphDocument, evidence_units: Optional[List[Dict[str, Any]]]) -> None:
    source_md = graph_doc.source.metadata if graph_doc.source and graph_doc.source.metadata else {}
    evidence_ids = []
    if evidence_units:
        for unit in evidence_units:
            payload = unit.__dict__ if hasattr(unit, "__dict__") else unit
            evidence_ids.append(payload.get("unit_id"))
    for node in graph_doc.nodes:
        props = node.properties or {}
        props.setdefault("core_type", node.type)
        props.setdefault("doc_id", source_md.get("doc_id"))
        props.setdefault("kg_scope", source_md.get("kg_scope"))
        props.setdefault("kg_id", source_md.get("kg_id"))
        props.setdefault("source_chunk_id", source_md.get("id") or (source_md.get("combined_chunk_ids") or [None])[0])
        props.setdefault("evidence_unit_ids", evidence_ids)
        props.setdefault("extraction_stage", "stage1_core")
        node.properties = props
    for rel in graph_doc.relationships:
        props = rel.properties or {}
        props.setdefault("doc_id", source_md.get("doc_id"))
        props.setdefault("kg_scope", source_md.get("kg_scope"))
        props.setdefault("kg_id", source_md.get("kg_id"))
        props.setdefault("source_chunk_id", source_md.get("id") or (source_md.get("combined_chunk_ids") or [None])[0])
        props.setdefault("evidence_unit_ids", evidence_ids)
        props.setdefault("extraction_stage", "stage1_core")
        rel.properties = props


def generate_graphDocuments(
    model: str,
    graph: Neo4jGraph,
    chunkId_chunkDoc_list: List,
    allowedNodes=None,
    allowedRelationship=None,
    domain_pack_id: str = "construction",
    evidence_units_by_chunk: dict | None = None
) -> List[GraphDocument]:
    pack = get_domain_pack(domain_pack_id)

    if not allowedNodes:
        allowedNodes = pack.allowed_core_node_types()
    if not allowedRelationship:
        allowedRelationship = pack.allowed_rel_types()

    if isinstance(allowedNodes, str) and allowedNodes:
        allowedNodes = allowedNodes.split(',')
    if isinstance(allowedRelationship, str) and allowedRelationship:
        allowedRelationship = allowedRelationship.split(',')
    allowedRelationship = _filter_stage1_relationships(allowedRelationship)

    evidence_aware_chunks = []
    for item in chunkId_chunkDoc_list:
        chunk_doc = item["chunk_doc"]
        metadata = dict(chunk_doc.metadata or {})
        chunk_id = item.get("chunk_id")
        units = evidence_units_by_chunk.get(chunk_id, []) if evidence_units_by_chunk else []
        evidence_context = _build_evidence_context(units)
        if evidence_context:
            chunk_doc = type(chunk_doc)(
                page_content=f"{chunk_doc.page_content}\n\n{evidence_context}",
                metadata=metadata,
            )
        evidence_aware_chunks.append({"chunk_id": chunk_id, "chunk_doc": chunk_doc})

    logging.info(f"使用 Schema - 节点: {allowedNodes}, 关系: {allowedRelationship}")

    graph_documents = []
    if model == "diffbot":
        graph_documents = get_graph_from_diffbot(graph, evidence_aware_chunks)

    elif model in OPENAI_MODELS:
        graph_documents = get_graph_from_OpenAI(model, graph, evidence_aware_chunks, allowedNodes, allowedRelationship)

    elif model in GEMINI_MODELS:
        graph_documents = get_graph_from_Gemini(model, graph, evidence_aware_chunks, allowedNodes, allowedRelationship)
    else:
        graph_documents = get_graph_from_llm(model, evidence_aware_chunks, allowedNodes, allowedRelationship)

    logging.info(
        "使用 Domain Pack[%s] Schema - 节点: %s, 关系: %s",
        domain_pack_id,
        allowedNodes,
        allowedRelationship,
    )



    logging.info("提取出来的graph_documents: %s", graph_documents)
    logging.info("开始应用 HFCSA overlay...")

    overlaid_documents = []
    for graph_doc in graph_documents:
        graph_doc = apply_hfsca_overlay(graph_doc, pack)
        preview = _preview_causal_chain(graph_doc)
        logging.info(preview)
        print(preview)
        overlaid_documents.append(graph_doc)

    logging.info("HFCSA overlay 后得到的graph_documents: %s", overlaid_documents)
    return overlaid_documents