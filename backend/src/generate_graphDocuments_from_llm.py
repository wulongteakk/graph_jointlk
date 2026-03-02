import json
import logging
from typing import List, Dict, Any, Optional
from langchain_community.graphs import Neo4jGraph

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.graph_document import GraphDocument

from src.diffbot_transformer import get_graph_from_diffbot
from src.openAI_llm import get_graph_from_OpenAI # 替换
from src.gemini_llm import get_graph_from_Gemini # 替换
from src.shared.constants import *
# from src.llm import get_graph_from_llm as old_get_graph_from_llm
from src.llm import get_graph_from_llm, get_llm


logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")





def _strip_json_fence(content: str) -> str:
    if "```json" in content:
        return content.split("```json")[1].split("```")[0]
    if "```" in content:
        return content.replace("```", "").strip()
    return content


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
        content = _strip_json_fence(response.content)
        data = json.loads(content)
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
    node_type = (node.type or "").strip()
    node_type_lower = node_type.lower()
    risk_types = {
        "riskfactor",
        "风险因子",
        "风险",
        "隐患",
        "不安全行为",
        "管理缺陷",
        "前提条件",
        "人因",
        "cause",
        "hazardsource",
        "hazard_source",
        "hazard",
        "hazardouscondition",
        "unsafeact",
        "unsafecondition",
    }
    if node_type_lower in risk_types or node_type in risk_types:
        return True
    return False


def _get_node_display_name(node: Any) -> str:
    if node.properties:
        return node.properties.get("name") or node.properties.get("id") or node.id
    return node.id




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

    for graph_doc in graph_documents:
        for node in graph_doc.nodes:
            if _is_risk_factor_node(node):
                existing_props = node.properties or {}
                if existing_props.get("layer_code") and existing_props.get("category_code"):
                    continue
                context = graph_doc.source.page_content if graph_doc.source else ""
                classification = classify_risk_factor_hfcsa_with_llm(
                    _get_node_display_name(node),
                    context,
                    model,
                )
                if classification:
                    print(node)
                    existing_props.update(
                        {key: value for key, value in classification.items() if value is not None}
                    )
                    node.properties = existing_props
                    total_hfcsa_enriched += 1
                    print(node)

            logging.info(
                "风险因子受控分类完成。HFCSA 受控分类覆盖 %s 个节点。",
                total_hfcsa_enriched,
            )



    logging.info(f"风险因子匹配后得到的graph_documents: {graph_documents}")
    return graph_documents