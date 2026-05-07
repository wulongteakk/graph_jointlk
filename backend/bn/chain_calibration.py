from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence


@dataclass
class CalibrationResult:
    temperature: float
    tau_alpha: float
    alpha: float


def _softmax(scores: Sequence[float], temperature: float) -> List[float]:
    if not scores:
        return []
    t = max(1e-6, float(temperature))
    scaled = [s / t for s in scores]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def fit_temperature(cal_rows: Iterable[Mapping[str, object]], candidates_key: str = "candidate_scores") -> float:
    """网格搜索温度，最小化主链NLL。"""
    rows = list(cal_rows)
    if not rows:
        return 1.0
    best_t = 1.0
    best_nll = float("inf")
    for t in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        nll = 0.0
        n = 0
        for row in rows:
            scores = [float(x) for x in (row.get(candidates_key) or [])]
            gold_index = int(row.get("gold_index", 0))
            probs = _softmax(scores, t)
            if not probs or gold_index >= len(probs):
                continue
            nll += -math.log(max(1e-12, probs[gold_index]))
            n += 1
        if n and nll / n < best_nll:
            best_nll = nll / n
            best_t = t
    return float(best_t)


def quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    idx = min(len(vals) - 1, max(0, int(math.ceil(q * len(vals)) - 1)))
    return vals[idx]


def fit_chain_calibration(cal_rows: Iterable[Mapping[str, object]], alpha: float = 0.1) -> CalibrationResult:
    rows = list(cal_rows)
    if not rows:
        return CalibrationResult(temperature=1.0, tau_alpha=0.0, alpha=float(alpha))

    t = fit_temperature(rows)
    nonconformity = [float(r.get("nonconformity", 0.0)) for r in rows]
    tau = quantile(nonconformity, 1.0 - float(alpha))
    return CalibrationResult(temperature=t, tau_alpha=float(tau), alpha=float(alpha))