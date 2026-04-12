from __future__ import annotations

from typing import Any, Dict, List

from .tokenizer import join_non_empty


_FIELD_ALIAS = {
    "直接原因": ["direct_cause", "直接原因"],
    "初始异常事件": ["initial_abnormal_event", "初始异常事件", "初始异常"],
    "事件时序": ["timeline", "事件时序"],
    "致害节点": ["harm_event", "致害节点", "致害事件"],
    "事故经过": ["accident_process", "事故经过"],
    "死亡事实": ["death_fact", "死亡事实", "是否死亡"],
    "伤情诊断结果": ["diagnosis", "伤情诊断结果"],
    "受伤部位": ["injured_parts", "受伤部位"],
    "医疗诊断时间": ["diagnosis_time", "医疗诊断时间"],
    "是否ICU": ["icu", "是否ICU", "是否进入ICU"],
    "损伤分级/面积/阈值": ["injury_grade", "伤害面积", "阈值", "损伤分级/面积/阈值"],
    "企业主营业务": ["main_business", "企业主营业务"],
    "项目名称": ["project_name", "项目名称"],
    "工程性质": ["project_nature", "工程性质"],
    "装置/产线名称": ["facility_name", "装置名称", "产线名称"],
    "作业场景": ["scene", "作业场景"],
}


class QueryBuilder:
    @staticmethod
    def build_query(task: str, raw_query: str, retrieval_profiles: Dict[str, Any], fact_basis: Dict[str, Any], candidate_types: List[str]) -> str:
        profile = retrieval_profiles.get(task, {})
        evidence_lines: List[str] = []
        for field_name in profile.get("evidence_fields", []):
            value = QueryBuilder._pick_fact_value(fact_basis, field_name)
            if value not in (None, "", [], {}):
                evidence_lines.append(f"{field_name}: {value}")

        focus = profile.get("query_focus", [])
        target_dimension = profile.get("target_dimension", task)
        candidate_text = f"候选类型: {'、'.join(candidate_types)}" if candidate_types else ""
        evidence_block = ("证据锚点:\n" + "\n".join(evidence_lines)) if evidence_lines else ""

        return join_non_empty(
            [
                f"检索任务: {task}",
                f"目标维度: {target_dimension}",
                f"原始查询: {raw_query}",
                candidate_text,
                evidence_block,
                f"检索焦点: {'；'.join(focus)}" if focus else "",
            ]
        )

    @staticmethod
    def _pick_fact_value(fact_basis: Dict[str, Any], field_name: str) -> Any:
        candidates = _FIELD_ALIAS.get(field_name, [field_name])
        for key in candidates:
            if key in fact_basis:
                return fact_basis[key]
        return None
