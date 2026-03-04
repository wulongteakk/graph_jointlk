"""证据库（Evidence Store）

与 Neo4j 的 KG 结构分离存储原文 chunk 内容，避免图谱膨胀。

- EvidenceChunk: chunk 级证据（全文/窗口）
- EvidenceUnit : 更细粒度证据（段落/句子），用于 evidence_offset 与精细 SUPPORTED_BY

正文只存证据库；Neo4j 里只保存指针（evidence_id/unit_id）与少量元数据。
"""

from .sqlite_store import EvidenceStore, EvidenceChunk, EvidenceUnit
from .text_span import (
    TextSpan,
    split_paragraphs_with_offsets,
    split_sentences_with_offsets,
    find_best_span,
    pick_span_paragraph,
    pick_span_sentence,
)

__all__ = [
    "EvidenceStore",
    "EvidenceChunk",
    "EvidenceUnit",
    "TextSpan",
    "split_paragraphs_with_offsets",
    "split_sentences_with_offsets",
    "find_best_span",
    "pick_span_paragraph",
    "pick_span_sentence",
]
