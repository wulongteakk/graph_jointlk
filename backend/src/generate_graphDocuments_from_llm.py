import logging
import json
from typing import List, Dict, Any
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_community.graphs.graph_document import Node, GraphDocument
from langchain_core.documents import Document

from src.diffbot_transformer import get_graph_from_diffbot
# from src.openAI_llm import get_graph_from_OpenAI # 替换
# from src.gemini_llm import get_graph_from_Gemini # 替换
from src.shared.constants import *
# from src.llm import get_graph_from_llm as old_get_graph_from_llm
from src.graph_transformers.llm import LLMGraphTransformer
from src.llm import get_llm  # 假设 src.llm 中有这个函数

# ==============================================================================
# START: 解决循环导入问题
# 1. 导致错误的 import 语句已被删除 (原第25行)
# ==============================================================================


logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")


# ==============================================================================
# 2. 确保 get_hazard_severity_score 函数定义在这里
# ==============================================================================
def get_hazard_severity_score(llm: BaseLanguageModel, hazard_description: str) -> Dict[str, Any]:
    """
    (阶段三: 隐患打分)
    调用 LLM (Analyst角色) 为隐患描述进行严重性打分 (1-10分)。
    """
    try:
        scoring_prompt = ChatPromptTemplate.from_template(HAZARD_SCORING_PROMPT_TEMPLATE)
        scoring_chain = scoring_prompt | llm

        response = scoring_chain.invoke({"hazard_description": hazard_description})
        content = response.content

        # 清理并解析 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.replace("```", "").strip()

        score_data = json.loads(content)

        score = int(score_data.get("score", 0))
        reason = score_data.get("reason", "")

        logging.info(f"隐患打分: [分数: {score}/10] [描述: {hazard_description}]")
        return {"severity_score": score, "severity_reason": reason}

    except Exception as e:
        logging.error(f"隐患打分失败: {e}. 描述: {hazard_description}")
        return {"severity_score": 0, "severity_reason": "打分失败"}


# ==============================================================================
# END: 函数定义
# ==============================================================================


def generate_graphDocuments(model: str, graph: Neo4jGraph, chunkId_chunkDoc_list: List, allowedNodes=None,
                            allowedRelationship=None) -> List[GraphDocument]:
    """
    (阶段二 + 阶段三)
    使用新的建筑安全 Prompt 提取实体和关系，并对 "隐患" 节点进行打分。
    """

    # 1. (阶段一) 确定 Schema
    if not allowedNodes:
        allowedNodes = CONSTRUCTION_NODE_LABELS
    if not allowedRelationship:
        allowedRelationship = CONSTRUCTION_REL_TYPES

    if isinstance(allowedNodes, str) and allowedNodes:
        allowedNodes = allowedNodes.split(',')
    if isinstance(allowedRelationship, str) and allowedRelationship:
        allowedRelationship = allowedRelationship.split(',')

    logging.info(f"使用 Schema - 节点: {allowedNodes}, 关系: {allowedRelationship}")

    # 2. (阶段二) 准备 LLM 和 Prompt
    try:
        llm, _ = get_llm(model)
    except Exception as e:
        logging.error(f"无法加载模型 '{model}'. 错误: {e}")
        return []

    extraction_prompt = ChatPromptTemplate.from_template(CONSTRUCTION_EXPERT_PROMPT_TEMPLATE)

    transformer = LLMGraphTransformer(
        llm=llm,
        prompt=extraction_prompt,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
        strict_mode=True,
        use_function_call=False
    )

    # 3. (阶段二) 提取知识
    documents = [doc for chunk_id, doc in chunkId_chunkDoc_list]

    logging.info(f"开始从 {len(documents)} 个文档中提取图谱...")
    try:
        langchain_documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            if hasattr(doc, 'page_content') else Document(page_content=str(doc))
            for doc in documents
        ]
        graph_documents = transformer.convert_to_graph_documents(langchain_documents)
    except Exception as e:
        logging.error(f"图谱转换失败: {e}", exc_info=True)
        return []

    logging.info(f"已提取 {len(graph_documents)} 个图文档。")

    # 4. (阶段三) 隐患打分
    logging.info("开始进行阶段三：隐患严重性打分...")
    total_hazards_scored = 0
    for graph_doc in graph_documents:
        for node in graph_doc.nodes:
            if node.type.lower() in ["隐患", "risk", "hazard", "风险"]:

                hazard_description = node.id

                if hazard_description:
                    # (现在调用本地定义的函数)
                    score_properties = get_hazard_severity_score(llm, hazard_description)

                    if node.properties:
                        node.properties.update(score_properties)
                    else:
                        node.properties = score_properties

                    total_hazards_scored += 1

    logging.info(f"隐患打分完成。总共对 {total_hazards_scored} 个隐患节点进行了打分。")

    logging.info(f"graph_documents = {len(graph_documents)}")
    return graph_documents