import logging
import json
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

# 假设 LLM 实例从 src.llm 获取
from src.llm import get_llm
from src.shared.constants import (
    RISK_ASSESSMENT_PROMPT_TEMPLATE,
    HAZARD_SCORING_PROMPT_TEMPLATE  # (保留导入以防万一，但函数已被移动)
)

logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")


# ==============================================================================
# get_hazard_severity_score 函数已于 2025-11-07 移至
# generate_graphDocuments_from_llm.py 以解决循环导入问题。
# ==============================================================================


def get_risk_metrics(graph: Neo4jGraph) -> Dict[str, Any]:
    """
    (阶段三: 指标提取)
    从知识图谱查询结构化的量化指标。
    """
    metrics = {}
    try:
        # 1. 隐患总数和闭环率
        hazard_query = """
        MATCH (h:隐患)
        RETURN 
            count(h) AS total_hazards,
            count(h WHERE h.status = '已整改') AS closed_hazards
        """
        result = graph.query(hazard_query)
        if result:
            total = result[0]['total_hazards']
            closed = result[0]['closed_hazards']
            metrics['total_hazards'] = total
            metrics['closed_hazards'] = closed
            if total > 0:
                metrics['closure_rate_percent'] = round((closed / total) * 100, 2)
            else:
                metrics['closure_rate_percent'] = 100

        # 2. 按严重性评分统计未整改隐患 (假设 severity_score 存在)
        unresolved_by_severity_query = """
        MATCH (h:隐患)
        WHERE (h.status IS NULL OR h.status <> '已整改') AND h.severity_score IS NOT NULL
        RETURN h.severity_score AS severity, count(h) AS count
        ORDER BY severity DESC
        """
        result = graph.query(unresolved_by_severity_query)
        if result:
            metrics['unresolved_hazards_by_severity'] = {
                item['severity']: item['count'] for item in result
            }

        # 3. 高频违章人员 (示例)
        high_risk_personnel_query = """
        MATCH (p:人员)-[:存在]->(h:隐患)
        WHERE h.status IS NULL OR h.status <> '已整改'
        RETURN p.name AS person, count(h) AS open_hazards
        ORDER BY open_hazards DESC
        LIMIT 5
        """
        result = graph.query(high_risk_personnel_query)
        if result:
            metrics['top_5_high_risk_personnel'] = {
                item['person']: item['open_hazards'] for item in result
            }

        logging.info(f"成功提取风险指标: {metrics}")
        return {"status": "success", "metrics": metrics}

    except Exception as e:
        logging.error(f"提取风险指标失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def get_risk_probability_assessment(graph: Neo4jGraph, query: str, model: str) -> Dict[str, Any]:
    """
    (阶段四: 动态概率量化 - RAG)
    1. 从 KG 检索与查询相关的风险因子。
    2. 将因子注入 Prompt，让 LLM 进行推理。
    """
    try:
        # 1. 检索 (Retrieval)
        core_query = query

        rag_query = """
        // 1. 查找与查询 $query 匹配的作业或地点
        MATCH (n)
        WHERE (n:作业 OR n:地点 OR n:设备) AND (n.id CONTAINS $query OR n.name CONTAINS $query)
        WITH collect(n) as nodes
        UNWIND nodes as n

        // 2. 收集关联的人员
        OPTIONAL MATCH (p:人员)-[:执行|使用|关联于*1..2]-(n)
        WITH n, collect(distinct p.name) AS personnel

        // 3. 收集关联的设备
        OPTIONAL MATCH (e:设备)-[:使用|关联于*1..2]-(n)
        WITH n, personnel, collect(distinct {name: e.name, status: e.status}) AS equipment

        // 4. 收集关联的环境因素
        OPTIONAL MATCH (env:环境因素)-[:关联于*1..2]-(n)
        WITH n, personnel, equipment, collect(distinct env.name) AS environment

        // 5. 收集最关键的：未整改的隐患及其严重性评分
        OPTIONAL MATCH (h:隐患)-[:关联于|发生在*1..2]-(n)
        WHERE (h.status IS NULL OR h.status <> '已整改') AND h.severity_score IS NOT NULL
        WITH n, personnel, equipment, environment, 
             collect(distinct {
                 hazard: h.name, 
                 severity: h.severity_score, 
                 reason: h.severity_reason
             }) AS unresolved_hazards

        RETURN {
            target: n.name,
            personnel: personnel,
            equipment: equipment,
            environment: environment,
            unresolved_hazards: unresolved_hazards
        } AS context_data
        LIMIT 10  // 限制返回的上下文数量
        """

        result = graph.query(rag_query, params={"query": core_query})

        if not result:
            logging.warning(f"RAG 检索：未找到与 '{core_query}' 相关的风险因子。")
            return {"status": "error", "message": f"未在知识图谱中找到与 '{core_query}' 相关的明确信息。"}

        # 2. 上下文构建 (Context Building)
        context_str = ""
        all_hazards = []
        all_personnel = set()
        all_equipment = set()
        all_env = set()
        all_targets = set()

        for record in result:
            data = record['context_data']
            if data['target']:
                all_targets.add(data['target'])
            if data['unresolved_hazards']:
                all_hazards.extend(data['unresolved_hazards'])
            if data['personnel']:
                all_personnel.update(data['personnel'])
            if data['equipment']:
                all_equipment.update([f"{e['name']}({e.get('status', 'N/A')})" for e in data['equipment']])
            if data['environment']:
                all_env.update(data['environment'])

        # 去重隐患
        unique_hazards = {h['hazard']: h for h in all_hazards}.values()

        context_str += f"- 评估目标: {', '.join(all_targets) if all_targets else query}\n"
        context_str += f"- 关联人员/班组: {', '.join(all_personnel) if all_personnel else '无'}\n"
        context_str += f"- 关联设备: {', '.join(all_equipment) if all_equipment else '无'}\n"
        context_str += f"- 关联环境因素: {', '.join(all_env) if all_env else '无'}\n"
        context_str += "- 关联的【未整改隐患】(最关键):\n"

        if unique_hazards:
            for hazard in unique_hazards:
                context_str += f"  - 描述: {hazard['hazard']} (严重性: {hazard['severity']}/10, 理由: {hazard.get('reason', 'N/A')})\n"
        else:
            context_str += "  - 无未整改隐患记录\n"

        logging.info(f"RAG 检索到的上下文:\n{context_str}")

        # 3. 推理 (Reasoning)
        llm = get_llm(model)
        assessment_prompt = ChatPromptTemplate.from_template(RISK_ASSESSMENT_PROMPT_TEMPLATE)
        assessment_chain = assessment_prompt | llm

        response = assessment_chain.invoke({
            "query": query,  # 使用用户原始查询
            "context": context_str
        })

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.replace("```", "").strip()

        assessment_result = json.loads(content)

        return {"status": "success", "assessment": assessment_result}

    except Exception as e:
        logging.error(f"风险概率评估失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}