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




def get_risk_metrics(graph: Neo4jGraph) -> Dict[str, Any]:
    """
    (阶段三: 指标提取)
    从知识图谱查询结构化的量化指标。

    注意：本项目的实体已经在节点属性中加入了 HFACS 风险因子信息
    （例如 risk_domain、hfacs_level_1、severity_level 等），因此这里
    直接基于这些属性做统计，而不是依赖特定的 "隐患" 标签。
    """
    metrics = {}
    try:
        # 1. 风险因子总量（任一节点只要包含风险领域即可视作风险因子）
        total_risk_query = """
        MATCH (n)
        WHERE n.risk_domain IS NOT NULL
        RETURN count(n) AS total_risk_factors
        """
        result = graph.query(total_risk_query)
        if result:
            metrics['total_risk_factors'] = result[0]['total_risk_factors']

        # 2. 按严重度分布（severity_level 由 HFACS 补全）
        severity_distribution_query = """
        MATCH (n)
        WHERE n.risk_domain IS NOT NULL AND n.severity_level IS NOT NULL
        RETURN n.severity_level AS severity, count(n) AS count
        ORDER BY severity DESC
        """
        result = graph.query(severity_distribution_query)
        if result:
            metrics['risk_factors_by_severity'] = {
                item['severity']: item['count'] for item in result
            }

        # 3. 按 HFACS 一级分类汇总
        hfacs_level_query = """
        MATCH (n)
        WHERE n.risk_domain IS NOT NULL AND n.hfacs_level_1 IS NOT NULL
        RETURN n.hfacs_level_1 AS level_1, count(n) AS count
        ORDER BY count DESC
        """
        result = graph.query(hfacs_level_query)
        if result:
            metrics['risk_factors_by_level_1'] = {
                item['level_1']: item['count'] for item in result
            }

        # 4. 最高严重度的风险因子 Top5（用于快速关注重点）
        top_severity_query = """
        MATCH (n)
        WHERE n.risk_domain IS NOT NULL AND n.severity_level IS NOT NULL
        RETURN coalesce(n.risk_name, n.name, n.id) AS risk_name,
               n.severity_level AS severity,
               n.risk_domain AS domain,
               n.hfacs_level_1 AS level_1,
               n.hfacs_level_2 AS level_2
        ORDER BY severity DESC
        LIMIT 5
        """
        result = graph.query(top_severity_query)
        if result:
            metrics['top_risk_factors'] = result

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
        // 1. 查找与查询 $query 匹配的作业 / 地点 / 设备 / 任务 / 风险节点
        MATCH (n)
        WHERE (n:作业 OR n:地点 OR n:设备 OR n:任务 OR n:风险 OR n:隐患)
              AND (coalesce(n.id, "") CONTAINS $query OR coalesce(n.name, "") CONTAINS $query OR coalesce(n.risk_name, "") CONTAINS $query)
        WITH collect(n) AS nodes
        UNWIND nodes AS n

        // 2. 收集关联的人员
        OPTIONAL MATCH (p:人员)-[:执行|使用|关联于*1..2]-(n)
        WITH n, collect(distinct coalesce(p.name, p.id)) AS personnel

        // 3. 收集关联的设备
        OPTIONAL MATCH (e:设备)-[:使用|关联于*1..2]-(n)
        WITH n, personnel, collect(distinct {name: coalesce(e.name, e.id), status: e.status}) AS equipment

        // 4. 收集关联的环境因素
        OPTIONAL MATCH (env:环境因素)-[:关联于*1..2]-(n)
        WITH n, personnel, equipment, collect(distinct coalesce(env.name, env.id)) AS environment

        // 5. 收集最关键的：关联的风险因子（基于 HFACS 属性）
        OPTIONAL MATCH (r)-[:关联于|发生在|存在*1..2]-(n)
        WHERE r.risk_domain IS NOT NULL
        WITH n, personnel, equipment, environment,
             collect(distinct {
                 risk: coalesce(r.risk_name, r.name, r.id),
                 severity: r.severity_level,
                 domain: r.risk_domain,
                 level_1: r.hfacs_level_1,
                 level_2: r.hfacs_level_2,
                 reason: r.severity_description
             }) AS related_risks

        RETURN {
            target: coalesce(n.name, n.id),
            target_risk: {
                risk: coalesce(n.risk_name, n.name, n.id),
                severity: n.severity_level,
                domain: n.risk_domain,
                level_1: n.hfacs_level_1,
                level_2: n.hfacs_level_2,
                reason: n.severity_description
            },
            personnel: personnel,
            equipment: equipment,
            environment: environment,
            related_risks: related_risks
        } AS context_data
        LIMIT 10  // 限制返回的上下文数量
        """


        result = graph.query(rag_query, params={"query": core_query})

        if not result:
            logging.warning(f"RAG 检索：未找到与 '{core_query}' 相关的风险因子。")
            return {"status": "error", "message": f"未在知识图谱中找到与 '{core_query}' 相关的明确信息。"}

        # 2. 上下文构建 (Context Building)
        context_str = ""
        all_risk_factors = []
        all_personnel = set()
        all_equipment = set()
        all_env = set()
        all_targets = set()

        for record in result:
            data = record['context_data']
            if data['target']:
                all_targets.add(data['target'])
            if data.get('target_risk'):
                all_risk_factors.append(data['target_risk'])
            if data.get('related_risks'):
                all_risk_factors.extend(data['related_risks'])
            if data['personnel']:
                all_personnel.update(data['personnel'])
            if data['equipment']:
                all_equipment.update([f"{e['name']}({e.get('status', 'N/A')})" for e in data['equipment']])
            if data['environment']:
                all_env.update(data['environment'])

        # 去重风险因子
        unique_risks = {r['risk']: r for r in all_risk_factors if r.get('risk')}.values()

        context_str += f"- 评估目标: {', '.join(all_targets) if all_targets else query}\n"
        context_str += f"- 关联人员/班组: {', '.join(all_personnel) if all_personnel else '无'}\n"
        context_str += f"- 关联设备: {', '.join(all_equipment) if all_equipment else '无'}\n"
        context_str += f"- 关联环境因素: {', '.join(all_env) if all_env else '无'}\n"
        context_str += "- 关联的【风险因子】(基于 HFACS 分类):\n"

        if unique_risks:
            for risk in unique_risks:
                context_str += (
                    f"  - {risk.get('risk', '未知')} | 领域: {risk.get('domain', 'N/A')} | "
                    f"一级: {risk.get('level_1', 'N/A')} | 二级: {risk.get('level_2', 'N/A')} | "
                    f"严重度: {risk.get('severity', 'N/A')} | 理由: {risk.get('reason', 'N/A')}\n"
                )
        else:
            context_str += "  - 未找到风险因子记录\n"

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