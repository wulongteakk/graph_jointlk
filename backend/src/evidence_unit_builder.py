from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Any

from langchain.docstore.document import Document
from src.evidence_discourse_linker import EvidenceDiscourseLinker


CAUSAL_TRIGGER_WORDS = [
    "由于", "导致", "致使", "从而", "最终", "另查明", "经调查", "补充说明",
    "因此", "所以", "进而", "造成", "引发", "使得",
]
TEMPORAL_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{1,2}月\d{1,2}日",
        r"事故发生后",
        r"随后",
        r"之后",
        r"当日",
        r"次日",
        r"当班",
        r"此前",
        r"最终",
    ]
]
SECTION_PATTERN = re.compile(r"^\s*(第[一二三四五六七八九十0-9]+[章节部分]|[一二三四五六七八九十]+[、.]|\d+[、.）)])\s*(.+)$")
SPLIT_PATTERN = re.compile(r"(?<=[。；;!?])\s+|\n{2,}|(?=^\s*(?:第[一二三四五六七八九十0-9]+[章节部分]|[一二三四五六七八九十]+[、.]|\d+[、.）)]))", re.M)
ACTOR_PATTERN = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,}(?:公司|单位|部门|项目部|班组|人员|司机|工人|员工|负责人|调查组|管理人员))")


@dataclass
class EvidenceUnitRecord:
    unit_id: str
    parent_chunk_id: str
    file_name: str
    report_id: Optional[str]
    page_range: Optional[tuple[int, int]]
    text: str
    unit_kind: str
    start_char: Optional[int]
    end_char: Optional[int]
    trigger_words: list[str]
    temporal_cues: list[str]
    actors: list[str]
    section_name: Optional[str]
    prev_unit_id: Optional[str]
    next_unit_id: Optional[str]
    unit_role: str
    main_event_anchor_id: Optional[str]
    supplement_to_unit_id: Optional[str]
    unit_group_id: Optional[str]
    event_order_hint: Optional[int]
    actor_anchor_ids: list[str]
    equipment_anchor_ids: list[str]
    location_anchor_ids: list[str]
    meta: dict[str, Any]


class EvidenceUnitBuilder:
    def __init__(self, trigger_words: Optional[list[str]] = None):
        self.trigger_words = trigger_words or list(CAUSAL_TRIGGER_WORDS)
        self.discourse_linker = EvidenceDiscourseLinker()

    def build_from_chunks(self, chunk_records: list[dict]) -> list[EvidenceUnitRecord]:
        units: list[EvidenceUnitRecord] = []
        for chunk_record in chunk_records:
            chunk_id = chunk_record["chunk_id"]
            chunk_doc = chunk_record["chunk_doc"]
            if isinstance(chunk_doc, Document):
                chunk_text = chunk_doc.page_content or ""
                metadata = dict(chunk_doc.metadata or {})
            else:
                chunk_text = str(chunk_record.get("text") or "")
                metadata = dict(chunk_record.get("metadata") or {})

            raw_segments = self._split_by_structure(chunk_text)
            merged_segments = self._merge_by_causal_triggers(raw_segments)
            page_range = self._normalise_page_range(metadata)
            file_name = metadata.get("fileName") or metadata.get("file_name") or ""
            report_id = metadata.get("doc_id") or metadata.get("report_id")

            row_segments: list[dict[str, Any]] = []
            for idx, segment in enumerate(merged_segments):
                text = segment["text"].strip()
                if not text:
                    continue
                features = self._extract_local_features(text=text, default_section_name=metadata.get("section_title") or metadata.get("section_name"))
                unit_id = f"{chunk_id}::evu::{idx}"
                row_segments.append({
                    "unit_id": unit_id,
                    "text": text,
                    "features": features,
                    "unit_kind": "causal_unit" if features["trigger_words"] else segment.get("unit_kind", "sentence"),
                    "start_char": segment.get("start_char"),
                    "end_char": segment.get("end_char"),
                })

            row_segments = self.discourse_linker.assign_roles_and_links(row_segments)
            unit_group_id = f"{chunk_id}::group::0"
            for idx, seg in enumerate(row_segments):
                features = seg["features"]
                unit_role = seg.get("unit_role") or "main_event_narrative"
                units.append(EvidenceUnitRecord(
                    unit_id=seg["unit_id"],
                    parent_chunk_id=chunk_id,
                    file_name=file_name,
                    report_id=report_id,
                    page_range=page_range,
                    text=seg["text"],
                    unit_kind=seg["unit_kind"],
                    start_char=seg["start_char"],
                    end_char=seg["end_char"],
                    trigger_words=features["trigger_words"],
                    temporal_cues=features["temporal_cues"],
                    actors=features["actors"],
                    section_name=features["section_name"],
                    prev_unit_id=None,
                    next_unit_id=None,
                    unit_role=unit_role,
                    main_event_anchor_id=seg.get("main_event_anchor_id"),
                    supplement_to_unit_id=seg.get("supplement_to_unit_id"),
                    unit_group_id=unit_group_id,
                    event_order_hint=int(metadata.get("narrative_index") or 0) + idx,
                    actor_anchor_ids=[f"actor::{a}" for a in features["actors"]],
                    equipment_anchor_ids=[],
                    location_anchor_ids=[],
                    meta={
                        "chunk_index": metadata.get("chunk_index"),
                        "page": metadata.get("page") or metadata.get("page_number"),
                        "kg_scope": metadata.get("kg_scope"),
                        "kg_id": metadata.get("kg_id"),
                        "doc_id": metadata.get("doc_id"),
                        "chunk_source": metadata.get("chunk_source"),
                        "section_title": metadata.get("section_title"),
                        "section_path": metadata.get("section_path"),
                        "paragraph_id": metadata.get("paragraph_id"),
                        "block_id": metadata.get("block_id"),
                        "chunk_char_start": metadata.get("chunk_char_start"),
                        "chunk_char_end": metadata.get("chunk_char_end"),
                        "narrative_index": metadata.get("narrative_index"),
                    },
                ))
        return self._link_adjacent_units(units)

    def _split_by_structure(self, text: str) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if not text.strip():
            return parts
        cursor = 0
        for piece in SPLIT_PATTERN.split(text):
            piece = piece.strip()
            if not piece:
                continue
            start = text.find(piece, cursor)
            if start < 0:
                start = cursor
            end = start + len(piece)
            cursor = end
            unit_kind = "paragraph"
            if len(piece) <= 80 and SECTION_PATTERN.match(piece):
                unit_kind = "heading"
            elif any(mark in piece for mark in ["。", "；", ";"]):
                unit_kind = "sentence"
            parts.append({"text": piece, "start_char": start, "end_char": end, "unit_kind": unit_kind})
        return parts

    def _merge_by_causal_triggers(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not segments:
            return []
        merged: list[dict[str, Any]] = []
        for segment in segments:
            text = segment["text"]
            has_trigger = any(trigger in text for trigger in self.trigger_words)
            if not merged:
                merged.append(dict(segment))
                continue

            prev = merged[-1]
            should_attach = has_trigger or prev.get("unit_kind") == "heading"
            if should_attach:
                prev["text"] = f"{prev['text']} {text}".strip()
                prev["end_char"] = segment.get("end_char")
                prev["unit_kind"] = "paragraph" if prev.get("unit_kind") == "heading" else "causal_unit"
            else:
                merged.append(dict(segment))
        return merged

    def _extract_local_features(self, text: str, default_section_name: Optional[str]) -> dict[str, Any]:
        trigger_words = [word for word in self.trigger_words if word in text]
        temporal_cues: list[str] = []
        for pattern in TEMPORAL_PATTERNS:
            temporal_cues.extend(pattern.findall(text))
        actors = list(dict.fromkeys(match.group(1) for match in ACTOR_PATTERN.finditer(text)))
        section_match = SECTION_PATTERN.match(text)
        section_name = default_section_name
        if section_match:
            section_name = section_match.group(2).strip() or default_section_name
        return {
            "trigger_words": list(dict.fromkeys(trigger_words)),
            "temporal_cues": list(dict.fromkeys(temporal_cues)),
            "actors": actors,
            "section_name": section_name,
        }

    def _link_adjacent_units(self, units: list[EvidenceUnitRecord]) -> list[EvidenceUnitRecord]:
        for idx, unit in enumerate(units):
            unit.prev_unit_id = units[idx - 1].unit_id if idx > 0 else None
            unit.next_unit_id = units[idx + 1].unit_id if idx < len(units) - 1 else None
        return units

    def _normalise_page_range(self, metadata: dict[str, Any]) -> Optional[tuple[int, int]]:
        page = metadata.get("page") or metadata.get("page_number")
        if isinstance(page, int):
            return (page, page)
        page_range = metadata.get("page_range")
        if isinstance(page_range, (list, tuple)) and len(page_range) == 2:
            try:
                return (int(page_range[0]), int(page_range[1]))
            except Exception:
                return None
        return None