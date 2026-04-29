from typing import Dict, Mapping


def compute_equivalent_sample_size(
    *,
    max_ess: float,
    chain_weight: float,
    align_rate: float,
    conflict_rate: float,
) -> float:
    """定义4.4: N_eq = N_max * w * rho_align * (1-rho_conf)."""
    return float(max_ess) * max(0.0, float(chain_weight)) * max(0.0, min(1.0, float(align_rate))) * max(0.0, 1.0 - float(conflict_rate))


def transport_chain_to_cpt(
    *,
    neq: float,
    assignment: Mapping[str, float],
) -> Dict[str, float]:
    """T_{y|pi}(c,x)=N_eq*A_{y|pi}(c,x)"""
    total = sum(max(0.0, float(v)) for v in assignment.values())
    if total <= 0.0:
        return {k: 0.0 for k in assignment.keys()}
    return {k: float(neq) * (max(0.0, float(v)) / total) for k, v in assignment.items()}