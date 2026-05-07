from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass
class PosteriorRow:
    key: str
    prior_count: float
    obs_count: float
    posterior_mean: float


def update_beta_binomial(
    *,
    prior_alpha: float,
    prior_beta: float,
    obs_positive: int,
    obs_negative: int,
) -> Dict[str, float]:
    a_post = float(prior_alpha) + int(obs_positive)
    b_post = float(prior_beta) + int(obs_negative)
    total = max(1e-12, a_post + b_post)
    return {
        "alpha_post": a_post,
        "beta_post": b_post,
        "posterior_mean": a_post / total,
        "prior_ess": float(prior_alpha) + float(prior_beta),
        "obs_n": int(obs_positive) + int(obs_negative),
    }


def update_dirichlet_multinomial(
    *,
    prior_eta: Mapping[str, float],
    obs_counts: Mapping[str, int],
) -> Dict[str, object]:
    posterior: Dict[str, float] = {}
    all_keys = set(prior_eta.keys()) | set(obs_counts.keys())
    for k in all_keys:
        posterior[str(k)] = float(prior_eta.get(k, 0.0)) + float(obs_counts.get(k, 0))

    total = sum(posterior.values()) or 1.0
    means = {k: v / total for k, v in posterior.items()}
    return {
        "posterior_eta": posterior,
        "posterior_mean": means,
        "prior_ess": float(sum(float(v) for v in prior_eta.values())),
        "obs_n": int(sum(int(v) for v in obs_counts.values())),
    }