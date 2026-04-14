from __future__ import annotations

import re
from typing import List

try:
    import jieba
except Exception:  # pragma: no cover
    jieba = None

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """中英文混合分词：中文优先 jieba，英文/数字走正则。"""
    if not text:
        return []
    text = str(text).strip()
    if not text:
        return []

    tokens: List[str] = []
    if _CJK_RE.search(text):
        if jieba is not None:
            tokens.extend(t.strip().lower() for t in jieba.cut(text) if t and t.strip())
        else:
            tokens.extend(ch for ch in text if _CJK_RE.match(ch))

    tokens.extend(t.lower() for t in _TOKEN_RE.findall(text))
    # 去重保序
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def keyword_overlap(a: str, b: str) -> float:
    """计算关键词重叠率（Jaccard）。"""
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0


def join_non_empty(parts: List[str], sep: str = "\n") -> str:
    """拼接非空文本片段。"""
    return sep.join([p for p in parts if p and str(p).strip()])