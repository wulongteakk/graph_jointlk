from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_binary_edge_metrics(
    gold: Sequence[int],
    prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    assert len(gold) == len(prob), "gold and prob must have same length"
    pred = [1 if p >= threshold else 0 for p in prob]

    tp = sum(1 for y, yhat in zip(gold, pred) if y == 1 and yhat == 1)
    fp = sum(1 for y, yhat in zip(gold, pred) if y == 0 and yhat == 1)
    fn = sum(1 for y, yhat in zip(gold, pred) if y == 1 and yhat == 0)
    tn = sum(1 for y, yhat in zip(gold, pred) if y == 0 and yhat == 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, len(gold))

    metrics = {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_accuracy": acc,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "threshold": float(threshold),
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(gold)) > 1:
            metrics["edge_auroc"] = float(roc_auc_score(gold, prob))
            metrics["edge_aupr"] = float(average_precision_score(gold, prob))
    except Exception:
        pass

    return metrics


def tune_binary_threshold(
    gold: Sequence[int],
    prob: Sequence[float],
    candidates: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    if not gold:
        return {"best_threshold": 0.5, "best_edge_f1": 0.0}
    if candidates is None:
        candidates = [x / 100.0 for x in range(10, 91, 2)]

    best_threshold = 0.5
    best_metrics = compute_binary_edge_metrics(gold, prob, threshold=best_threshold)
    best_score = best_metrics.get("edge_f1", 0.0)

    for threshold in candidates:
        metrics = compute_binary_edge_metrics(gold, prob, threshold=float(threshold))
        score = metrics.get("edge_f1", 0.0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    best_metrics = dict(best_metrics)
    best_metrics["best_threshold"] = best_threshold
    best_metrics["best_edge_f1"] = best_score
    return best_metrics


def compute_relation_metrics(
    gold: Sequence[int],
    pred: Sequence[int],
    ignore_label: int = 0,
) -> Dict[str, float]:
    labels = sorted(set(gold) | set(pred))
    labels = [x for x in labels if x != ignore_label]
    if not labels:
        return {"rel_micro_f1": 0.0, "rel_macro_f1": 0.0}

    total_tp = total_fp = total_fn = 0
    per_label_f1: List[float] = []
    for label in labels:
        tp = sum(1 for y, yhat in zip(gold, pred) if y == label and yhat == label)
        fp = sum(1 for y, yhat in zip(gold, pred) if y != label and yhat == label)
        fn = sum(1 for y, yhat in zip(gold, pred) if y == label and yhat != label)
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        per_label_f1.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = _safe_div(total_tp, total_tp + total_fp)
    micro_r = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)
    macro_f1 = sum(per_label_f1) / len(per_label_f1)
    return {"rel_micro_f1": micro_f1, "rel_macro_f1": macro_f1}


def compute_ranking_metrics(
    ranked_gold_flags: Sequence[Sequence[int]],
) -> Dict[str, float]:
    mrr = 0.0
    hits1 = 0.0
    hits3 = 0.0
    hits5 = 0.0
    ndcg5 = 0.0
    n = len(ranked_gold_flags)
    if n == 0:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@5": 0.0, "ndcg@5": 0.0}

    for flags in ranked_gold_flags:
        first_rank = None
        dcg = 0.0
        ideal = sorted(flags, reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(flags[:5], start=1):
            if rel and first_rank is None:
                first_rank = rank
            if rel:
                dcg += 1.0 / math.log2(rank + 1)
        for rank, rel in enumerate(ideal[:5], start=1):
            if rel:
                idcg += 1.0 / math.log2(rank + 1)

        if first_rank is not None:
            mrr += 1.0 / first_rank
            hits1 += 1.0 if first_rank <= 1 else 0.0
            hits3 += 1.0 if first_rank <= 3 else 0.0
            hits5 += 1.0 if first_rank <= 5 else 0.0
        ndcg5 += _safe_div(dcg, idcg)

    return {
        "mrr": mrr / n,
        "hits@1": hits1 / n,
        "hits@3": hits3 / n,
        "hits@5": hits5 / n,
        "ndcg@5": ndcg5 / n,
    }


def compute_grouped_ranking_metrics(
    rows: Sequence[Dict[str, Any]],
    *,
    group_keys: Sequence[str],
    score_field: str = "support_prob",
    label_field: str = "gold_label",
) -> Dict[str, float]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups[key].append(dict(row))

    ranked_gold_flags: List[List[int]] = []
    for _, group_rows in groups.items():
        sorted_rows = sorted(group_rows, key=lambda x: float(x.get(score_field, 0.0)), reverse=True)
        flags = [int(r.get(label_field, 0)) for r in sorted_rows]
        if any(flags):
            ranked_gold_flags.append(flags)

    return compute_ranking_metrics(ranked_gold_flags)


def compute_breakdown_by_field(
    rows: Sequence[Dict[str, Any]],
    *,
    field: str,
    threshold: float,
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field) or "<NONE>")
        buckets[key].append(dict(row))

    result: Dict[str, Dict[str, float]] = {}
    for key, bucket_rows in buckets.items():
        gold = [int(r.get("gold_label", 0)) for r in bucket_rows]
        prob = [float(r.get("support_prob", 0.0)) for r in bucket_rows]
        metrics = compute_binary_edge_metrics(gold, prob, threshold=threshold)
        metrics["count"] = float(len(bucket_rows))
        result[key] = metrics
    return result


def summarize_for_paper(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = set()
    for row in rows:
        keys.update(row.keys())
    summary = {}
    for key in sorted(keys):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[key] = sum(values) / len(values)
    return summary
