import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class SoftPathScore:
    chain_id: str
    weight: float
    psi_forward: float
    psi_conflict: float


def _norm_path(edges: Iterable[Tuple[str, str]]) -> set:
    return {(str(a), str(b)) for a, b in edges}


def compute_path_consistency(
    step_nodes: Sequence[str],
    bn_edges: Iterable[Tuple[str, str]],
    align_map: Mapping[str, str],
) -> Tuple[float, float]:
    """计算定义4.3的 psi_M 与 \bar{psi}_M。"""
    if len(step_nodes) <= 1:
        return 0.0, 0.0
    edge_set = _norm_path(bn_edges)
    ok = 0
    bad = 0
    total = max(1, len(step_nodes) - 1)
    for i in range(len(step_nodes) - 1):
        s = align_map.get(step_nodes[i], step_nodes[i])
        t = align_map.get(step_nodes[i + 1], step_nodes[i + 1])
        if (s, t) in edge_set:
            ok += 1
        if (t, s) in edge_set or s == t:
            bad += 1
    return ok / total, bad / total


def build_soft_path_prior(
    *,
    chains: Sequence[Mapping[str, object]],
    bn_edges: Iterable[Tuple[str, str]],
    align_map: Mapping[str, str],
    lambda_ctp: float,
    phi_ctp: float,
    lambda_pos: float,
    lambda_neg: float,
) -> Dict[str, object]:
    """输出 BN 结构软路径先验的对数势分解。"""
    chain_scores: List[SoftPathScore] = []
    pos = 0.0
    neg = 0.0
    for c in chains:
        nodes = [str(x) for x in (c.get("node_path") or [])]
        w = float(c.get("weight", 0.0))
        psi_f, psi_b = compute_path_consistency(nodes, bn_edges, align_map)
        pos += w * psi_f
        neg += w * psi_b
        chain_scores.append(SoftPathScore(chain_id=str(c.get("chain_id", "")), weight=w, psi_forward=psi_f, psi_conflict=psi_b))

    log_prior = float(lambda_ctp) * float(phi_ctp) + float(lambda_pos) * pos - float(lambda_neg) * neg
    return {
        "log_prior": log_prior,
        "unnormalized_prior": math.exp(max(min(log_prior, 50.0), -50.0)),
        "components": {
            "ctp": float(lambda_ctp) * float(phi_ctp),
            "positive_path": float(lambda_pos) * pos,
            "conflict_path": float(lambda_neg) * neg,
        },
        "chain_scores": [s.__dict__ for s in chain_scores],
    }