from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

PROFILE_PATTERNS = [
    re.compile(r"(主营业务|主要从事|经营范围|承建|承包|施工内容|项目概况|单位概况)[:：]?(.{0,80})"),
]

TITLE_HINTS = [
    "公司",
    "集团",
    "项目部",
    "工程",
    "运输",
    "化工",
    "矿",
    "厂",
]

PROCESS_TERMS = [
    "施工", "吊装", "运输", "检修", "掘进", "生产", "装卸", "安装", "拆除", "浇筑", "爆破"
]


def collect_profile_spans(text_blocks: Sequence[str]) -> List[Tuple[str, float, str]]:
    rows: List[Tuple[str, float, str]] = []
    for block in text_blocks:
        for pattern in PROFILE_PATTERNS:
            for m in pattern.finditer(block):
                rows.append((m.group(0), 1.5, "profile_hit"))
    return rows


def collect_title_spans(title: str) -> List[Tuple[str, float, str]]:
    title = title or ""
    rows: List[Tuple[str, float, str]] = []
    if any(k in title for k in TITLE_HINTS):
        rows.append((title, 1.8, "title_hit"))
    return rows


def collect_process_terms(text_blocks: Sequence[str]) -> List[str]:
    hit_terms: List[str] = []
    for block in text_blocks:
        for term in PROCESS_TERMS:
            if term in block:
                hit_terms.append(term)
    return list(dict.fromkeys(hit_terms))


def rank_repeated_activity_terms(text_blocks: Sequence[str]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for block in text_blocks:
        for term in PROCESS_TERMS:
            if term in block:
                counter[term] += 1
    return dict(counter)


def normalize_evidence_texts(items: Iterable[object]) -> List[str]:
    rows: List[str] = []
    for item in items:
        if isinstance(item, str):
            rows.append(item)
        elif isinstance(item, dict):
            text = item.get("content") or item.get("text") or item.get("evidence_text")
            if text:
                rows.append(str(text))
    return rows
