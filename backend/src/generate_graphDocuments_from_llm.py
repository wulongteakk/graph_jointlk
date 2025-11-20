import logging
import jieba
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from langchain_community.graphs import Neo4jGraph

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.graph_document import GraphDocument

from src.diffbot_transformer import get_graph_from_diffbot
from src.openAI_llm import get_graph_from_OpenAI # 替换
from src.gemini_llm import get_graph_from_Gemini # 替换
from src.shared.constants import *
# from src.llm import get_graph_from_llm as old_get_graph_from_llm
from src.graph_transformers.llm import LLMGraphTransformer
from src.llm import get_graph_from_llm
from src.HFACS_Factors import HFACS_RISK_FACTORS,HFACS_SIMILARITY_THRESHOLD



logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")





def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", "", text.lower())
    words = jieba.lcut(text)
    return words


def _calc_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def match_hfacs_risk_factor(entity_name: str) -> Optional[Dict[str, Any]]:
    """
    将实体名称与 HFACS 风险因子关键词进行模糊匹配，返回最佳命中项。
    """
    best_match: Optional[Dict[str, Any]] = None
    best_score = 0.0
    matched_keyword = ""

    for factor in HFACS_RISK_FACTORS:
        for keyword in factor.get("keywords", []):
            score_all=[]
            for i in keyword:
                score_single = _calc_similarity(i, entity_name)
                score_all.append(score_single)
            score=max(score_all)
            if score > best_score and score >= HFACS_SIMILARITY_THRESHOLD:
                best_score = score
                matched_keyword = keyword
                best_match = {
                    "risk_name": factor.get("risk_name"),
                    "risk_domain": factor.get("domain"),
                    "hfacs_level_1": factor.get("hfacs_level_1"),
                    "hfacs_level_2": factor.get("hfacs_level_2"),
                    "severity_level": factor.get("severity_level"),
                    "severity_description": factor.get("severity_description"),
                    "hfacs_match_keyword": keyword,
                    "hfacs_match_score": round(score, 3),
                }

    if best_match:
        logging.info(
            f"HFACS 匹配: 实体[{entity_name}] -> {best_match['risk_name']} (keyword: {matched_keyword}, score: {best_score:.3f})"
        )
    return best_match





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


    extraction_prompt = ChatPromptTemplate.from_template(CONSTRUCTION_EXPERT_PROMPT_TEMPLATE)

    logging.info(f"提取出来的graph_documents: {graph_documents}")

    logging.info("开始进行阶段三：隐患严重性打分与风险因子映射...")
    total_hazards_scored = 0
    total_hfacs_enriched = 0

    for graph_doc in graph_documents:
        for node in graph_doc.nodes:
            node_type = node.type.lower()
            node_name = (
                node.properties.get("id") if node.properties else None
            ) or node.id


            # HFACS 风险因子匹配（隐患/风险/作业/任务等场景）

            match = match_hfacs_risk_factor(node_name)
            if match:
                if node.properties:
                    node.properties.update(match)
                else:
                    node.properties = match
                total_hfacs_enriched += 1

    logging.info(
        f"隐患打分完成。总共对 {total_hazards_scored} 个隐患节点进行了打分，HFACS 风险因子属性覆盖 {total_hfacs_enriched} 个节点。"
    )


    logging.info(f"风险因子匹配后得到的graph_documents: {graph_documents}")
    return graph_documents