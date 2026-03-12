
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(set(gold)) > 1:
            metrics["edge_auroc"] = float(roc_auc_score(gold, prob))
            metrics["edge_aupr"] = float(average_precision_score(gold, prob))
    except Exception:
        pass

    return metrics


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


def _chain_to_node_set(chain: Sequence[Tuple[str, str, str]]) -> set:
    nodes = set()
    for head, _, tail in chain:
        nodes.add(head)
        nodes.add(tail)
    return nodes


def _set_f1(gold_set: set, pred_set: set) -> float:
    if not gold_set and not pred_set:
        return 1.0
    inter = len(gold_set & pred_set)
    p = _safe_div(inter, len(pred_set))
    r = _safe_div(inter, len(gold_set))
    return _safe_div(2 * p * r, p + r)


def compute_chain_metrics(
    gold_chains: Sequence[Sequence[Tuple[str, str, str]]],
    pred_chains: Sequence[Sequence[Tuple[str, str, str]]],
) -> Dict[str, float]:
    assert len(gold_chains) == len(pred_chains), "gold/pred chain list length mismatch"
    n = len(gold_chains)
    if n == 0:
        return {
            "chain_exact_match": 0.0,
            "chain_edge_f1": 0.0,
            "chain_node_f1": 0.0,
        }

    exact = 0.0
    edge_f1 = 0.0
    node_f1 = 0.0
    for gold_chain, pred_chain in zip(gold_chains, pred_chains):
        gold_edge_set = set(tuple(x) for x in gold_chain)
        pred_edge_set = set(tuple(x) for x in pred_chain)
        gold_node_set = _chain_to_node_set(gold_chain)
        pred_node_set = _chain_to_node_set(pred_chain)
        if list(gold_chain) == list(pred_chain):
            exact += 1.0
        edge_f1 += _set_f1(gold_edge_set, pred_edge_set)
        node_f1 += _set_f1(gold_node_set, pred_node_set)

    return {
        "chain_exact_match": exact / n,
        "chain_edge_f1": edge_f1 / n,
        "chain_node_f1": node_f1 / n,
    }


def compute_ranking_metrics(
    ranked_gold_flags: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """Each row is ranked candidate list, value 1 means relevant chain."""
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
