from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Mapping


def build_dirichlet_hyperparams(
    *,
    base_eta: Mapping[str, float],
    transported_counts: Iterable[Mapping[str, float]],
) -> Dict[str, float]:
    """η = η0 + Σ T"""
    out: DefaultDict[str, float] = defaultdict(float)
    for k, v in base_eta.items():
        out[str(k)] += float(v)
    for row in transported_counts:
        for k, v in row.items():
            out[str(k)] += float(v)
    return dict(out)