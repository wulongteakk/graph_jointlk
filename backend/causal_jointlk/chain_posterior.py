import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .schemas import CandidateChain


@dataclass
class ChainPosteriorResult:
    chain_id: str
    posterior: float
    log_posterior: float
    chain_energy: float
    penalty_total: float


@dataclass
class PosteriorBundle:
    posteriors: List[ChainPosteriorResult]
    entropy: float
    top1_chain_id: str
    top1_mass: float


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    max_v = max(values)
    if math.isinf(max_v):
        return max_v
    return max_v + math.log(sum(math.exp(v - max_v) for v in values))


def build_chain_posterior(chains: Sequence[CandidateChain]) -> PosteriorBundle:
    if not chains:
        return PosteriorBundle(posteriors=[], entropy=0.0, top1_chain_id="", top1_mass=0.0)

    energies = []
    for c in chains:
        penalties = (c.meta or {}).get("penalties", {})
        penalty_total = float(sum(float(v) for k, v in penalties.items() if k.endswith("penalty")))
        chain_energy = float((c.meta or {}).get("F_theta", c.score))
        energies.append((c, chain_energy, penalty_total))

    log_z = _logsumexp([x[1] for x in energies])
    posterior_rows: List[ChainPosteriorResult] = []
    for chain, energy, penalty_total in energies:
        log_q = energy - log_z
        posterior_rows.append(
            ChainPosteriorResult(
                chain_id=chain.chain_id,
                posterior=float(math.exp(log_q)),
                log_posterior=log_q,
                chain_energy=energy,
                penalty_total=penalty_total,
            )
        )

    posterior_rows.sort(key=lambda r: r.posterior, reverse=True)
    entropy = -sum(r.posterior * r.log_posterior for r in posterior_rows if r.posterior > 0.0)
    return PosteriorBundle(
        posteriors=posterior_rows,
        entropy=float(entropy),
        top1_chain_id=posterior_rows[0].chain_id,
        top1_mass=float(posterior_rows[0].posterior),
    )


def posterior_to_dict(bundle: PosteriorBundle) -> Dict[str, object]:
    return {
        "posterior": [r.__dict__ for r in bundle.posteriors],
        "posterior_entropy": bundle.entropy,
        "posterior_top1_chain_id": bundle.top1_chain_id,
        "posterior_top1_mass": bundle.top1_mass,
    }