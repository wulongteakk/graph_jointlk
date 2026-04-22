from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PrototypeMatch:
    node_proto: str
    node_proto_conf: float
    node_proto_family: str
    canonical_surface: str
    is_accident_title: bool
    is_consequence: bool
    is_cause_factor: bool


_RULES: List[tuple[str, str, str, str]] = [
    (r"(事故|受伤事故|坠落事故|高坠事故)$", "ACCIDENT_TITLE_ANCHOR", "ANCHOR", "事故标题"),
    (r"安全意识(淡薄|不强|不足)", "SAFETY_AWARENESS_WEAK", "CAUSE", "安全意识淡薄"),
    (r"(违规|违章).{0,6}(作业|操作)", "VIOLATION_OPERATION", "CAUSE", "违规作业"),
    (r"(行为异常|操作不当|不安全行为)", "UNSAFE_OPERATION", "CAUSE", "不安全作业"),
    (r"(未.*佩戴.*(安全带|防护用品)|未戴.*防护)", "PPE_NOT_USED", "CAUSE", "未正确佩戴防护用品"),
    (r"(未系挂安全绳|未挂生命绳|未使用防坠)", "FALL_PROTECTION_NOT_USED", "CAUSE", "高处防坠措施未使用"),
    (r"(临边防护缺失|防护不到位|防护缺失)", "PROTECTION_DEFECT", "CAUSE", "防护缺失"),
    (r"(教育培训不到位|培训不足|未培训)", "TRAINING_DEFECT", "CAUSE", "安全培训不足"),
    (r"(监管不到位|监督不到位|现场监管)", "SUPERVISION_DEFECT", "CAUSE", "现场监管不到位"),
    (r"(管理不到位|管理缺陷|制度缺陷)", "MANAGEMENT_DEFECT", "CAUSE", "管理缺陷"),
    (r"(风险排查不到位|隐患排查不到位|风险辨识不足)", "RISK_INSPECTION_DEFECT", "CAUSE", "风险排查不足"),
    (r"(高处坠落|高坠|坠落)", "FALL_FROM_HEIGHT_EVENT", "EVENT", "高处坠落"),
    (r"(物体打击)", "OBJECT_STRIKE_EVENT", "EVENT", "物体打击"),
    (r"(坍塌|坍塌事故)", "COLLAPSE_EVENT", "EVENT", "坍塌"),
    (r"(触电|电击)", "ELECTRIC_SHOCK_EVENT", "EVENT", "触电"),
    (r"(机械伤害)", "MECHANICAL_INJURY_EVENT", "EVENT", "机械伤害"),
    (r"(受伤|伤者|伤情)", "INJURY_OUTCOME", "OUTCOME", "人员受伤"),
    (r"(死亡|致死|身亡)", "DEATH_OUTCOME", "OUTCOME", "人员死亡"),
    (r"(经济损失|损失\d+|损失金额)", "ECONOMIC_LOSS_OUTCOME", "OUTCOME", "经济损失"),
]


def map_node_to_prototype(
    node_text: str,
    *,
    node_layer: Optional[str] = None,
    section_role: Optional[str] = None,
    relation_type: Optional[str] = None,
    evidence_text: Optional[str] = None,
    file_name: Optional[str] = None,
    accident_type: Optional[str] = None,
) -> Dict[str, Any]:
    text = str(node_text or "").strip()
    for pattern, proto, family, canonical in _RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            is_outcome = family == "OUTCOME"
            is_cause = family == "CAUSE"
            is_anchor = proto == "ACCIDENT_TITLE_ANCHOR"
            return PrototypeMatch(
                node_proto=proto,
                node_proto_conf=0.95,
                node_proto_family=family,
                canonical_surface=canonical,
                is_accident_title=is_anchor,
                is_consequence=is_outcome,
                is_cause_factor=is_cause,
            ).__dict__

    family = str(node_layer or "UNK").upper()
    return PrototypeMatch(
        node_proto=f"LOCAL::{text[:64] or 'UNKNOWN'}",
        node_proto_conf=0.35,
        node_proto_family=family,
        canonical_surface=text or "UNKNOWN",
        is_accident_title=False,
        is_consequence=family in {"OUTCOME", "CONSEQUENCE"},
        is_cause_factor=family in {"CAUSE", "FACTOR", "RISK"},
    ).__dict__