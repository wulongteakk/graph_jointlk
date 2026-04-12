from __future__ import annotations

import itertools
import re
from typing import Iterable, List, Optional

from .schemas import EvidenceUnit

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；;!?])")
_SECTION_RE = re.compile(r"^\s*(?:[一二三四五六七八九十]+、|\d+[.、])\s*(.+)$")

_TEMPORAL_CUES = ["首先", "随后", "之后", "当日", "当天", "次日", "过程中", "作业时", "后续", "最终"]
_CAUSAL_CUES = ["导致", "造成", "引发", "由于", "因", "致", "使得", "从而"]
_INITIAL_PAT = re.compile(r"失效|故障|异常|断裂|违章|违规|未设置|未按|泄漏|松脱|脱落|缺失")
_HARM_PAT = re.compile(r"坠落|爆炸|起火|火灾|触电|碰撞|打击|中毒|窒息|灼烫|淹溺|滑坡|坍塌")
_OUTCOME_PAT = re.compile(r"死亡|身亡|受伤|重伤|轻伤|骨折|送医|icu|抢救")


def _split_sections(text: str) -> List[tuple[str, str]]:
    current_title = "正文"
    current_lines: List[str] = []
    sections: List[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SECTION_RE.match(line)
        if m:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = m.group(1).strip() or "正文"
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines)))
    return sections or [("正文", text)]


def _event_role(text: str) -> str:
    if _OUTCOME_PAT.search(text):
        return "injury_outcome"
    if _HARM_PAT.search(text):
        return "harm_event"
    if _INITIAL_PAT.search(text):
        return "initial_abnormal"
    if any(c in text for c in _CAUSAL_CUES + _TEMPORAL_CUES):
        return "intermediate_event"
    return "other"


def _extract_cues(text: str, lexicon: Iterable[str]) -> List[str]:
    return [cue for cue in lexicon if cue in text]


def segment_report(text: str, report_id: Optional[str] = None) -> List[EvidenceUnit]:
    """按 PPT/文档要求进行“因果表达 + 时间线索 + 事件完整性”切分。"""
    evidence_units: List[EvidenceUnit] = []
    sections = _split_sections(text)
    counter = itertools.count(1)

    for section_name, section_text in sections:
        chunks = [x.strip() for x in _SENT_SPLIT_RE.split(section_text) if x.strip()]
        merged: List[str] = []
        buffer = ""
        for sent in chunks:
            if not buffer:
                buffer = sent
                continue
            if any(cue in sent for cue in _CAUSAL_CUES + _TEMPORAL_CUES) or any(cue in buffer for cue in _CAUSAL_CUES):
                buffer = f"{buffer} {sent}".strip()
            elif len(buffer) < 25:
                buffer = f"{buffer} {sent}".strip()
            else:
                merged.append(buffer)
                buffer = sent
        if buffer:
            merged.append(buffer)

        for chunk in merged:
            idx = next(counter)
            evidence_units.append(
                EvidenceUnit(
                    evidence_id=f"EV-{idx:05d}",
                    report_id=report_id,
                    text=chunk,
                    section_name=section_name,
                    event_role=_event_role(chunk),
                    temporal_cues=_extract_cues(chunk, _TEMPORAL_CUES),
                    causal_cues=_extract_cues(chunk, _CAUSAL_CUES),
                )
            )
    return evidence_units
