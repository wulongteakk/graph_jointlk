from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


SUPPLEMENT_CUES = ("另查明", "经调查", "补充说明", "事后发现", "进一步查明")
ROLE_RULES = {
    "investigation_conclusion": ("调查认定", "调查结论", "经调查"),
    "consequence_description": ("造成", "死亡", "受伤", "损失"),
    "disposal_process": ("处置", "救援", "抢险", "整改"),
    "cause_supplement": SUPPLEMENT_CUES,
}


@dataclass
class DiscourseDecision:
    unit_role: str
    main_event_anchor_id: Optional[str]
    supplement_to_unit_id: Optional[str]


class EvidenceDiscourseLinker:
    """事故叙事层链接：角色识别 + 跨段补充回挂。"""

    def assign_roles_and_links(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        last_main_event_id: Optional[str] = None

        for seg in segments:
            role = self._infer_role(seg.get("text") or "")
            seg["unit_role"] = role

            if role == "main_event_narrative":
                last_main_event_id = seg.get("unit_id")
                seg["main_event_anchor_id"] = seg.get("unit_id")
                seg["supplement_to_unit_id"] = None
            elif role in {"cause_supplement", "investigation_conclusion"}:
                seg["main_event_anchor_id"] = last_main_event_id
                seg["supplement_to_unit_id"] = last_main_event_id
            else:
                seg["main_event_anchor_id"] = last_main_event_id
                seg["supplement_to_unit_id"] = None
            enriched.append(seg)
        return enriched

    def _infer_role(self, text: str) -> str:
        norm = text.strip()
        if not norm:
            return "main_event_narrative"
        for role, cues in ROLE_RULES.items():
            if any(cue in norm for cue in cues):
                return role
        return "main_event_narrative"