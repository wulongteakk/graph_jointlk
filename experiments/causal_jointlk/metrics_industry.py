from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def industry_exact_acc(preds: Sequence[str], golds: Sequence[str]) -> float:
    total = max(len(golds), 1)
    return sum(1 for p, g in zip(preds, golds) if p == g) / total


def industry_section_acc(preds: Sequence[str], golds: Sequence[str]) -> float:
    total = max(len(golds), 1)
    return sum(1 for p, g in zip(preds, golds) if (p or "")[:1] == (g or "")[:1]) / total


def industry_recall_at_k(pred_topk: Sequence[Sequence[str]], golds: Sequence[str], k: int = 5) -> float:
    total = max(len(golds), 1)
    return sum(1 for topk, gold in zip(pred_topk, golds) if gold in list(topk)[:k]) / total


def industry_coverage(preds: Sequence[str]) -> float:
    total = max(len(preds), 1)
    return sum(1 for p in preds if p) / total


def summarize_industry_metrics(pred_codes: Sequence[str], gold_codes: Sequence[str], pred_topk: Sequence[Sequence[str]]) -> Dict[str, float]:
    return {
        "industry_exact_acc": industry_exact_acc(pred_codes, gold_codes),
        "industry_section_acc": industry_section_acc(pred_codes, gold_codes),
        "industry_recall_at_5": industry_recall_at_k(pred_topk, gold_codes, 5),
        "industry_coverage": industry_coverage(pred_codes),
    }
