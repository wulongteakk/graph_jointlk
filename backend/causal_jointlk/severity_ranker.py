from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .schemas import CandidateBranch


@dataclass
class SeverityDecisionTrace:
    triggered: bool
    reason: str
    ranking: List[Dict[str, Any]]
    selected_branch_id: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "selected_branch_id": self.selected_branch_id,
            "ranking": self.ranking,
        }


class SeverityRanker:
    def __init__(
        self,
        death_weight: float = 5.0,
        serious_injury_weight: float = 3.0,
        light_injury_weight: float = 1.0,
        energy_weight: float = 1.0,
        toxicity_weight: float = 1.0,
    ) -> None:
        self.death_weight = float(death_weight)
        self.serious_injury_weight = float(serious_injury_weight)
        self.light_injury_weight = float(light_injury_weight)
        self.energy_weight = float(energy_weight)
        self.toxicity_weight = float(toxicity_weight)

    def rerank(self, branches: Sequence[CandidateBranch]) -> Sequence[CandidateBranch]:
        ranked, _ = self.rerank_with_trace(branches, triggered=True, reason="severity_rerank")
        return ranked

    def rerank_with_trace(
        self,
        branches: Sequence[CandidateBranch],
        *,
        triggered: bool,
        reason: str,
    ) -> tuple[Sequence[CandidateBranch], SeverityDecisionTrace]:
        ranked = []
        trace_rows = []
        for branch in branches:
            death = float(branch.meta.get("death_count", 0.0))
            serious = float(branch.meta.get("serious_injury_count", 0.0))
            light = float(branch.meta.get("light_injury_count", 0.0))
            energy = float(branch.meta.get("energy_level", 0.0))
            toxicity = float(branch.meta.get("toxicity_level", 0.0))
            branch.severity_score = (
                self.death_weight * death
                + self.serious_injury_weight * serious
                + self.light_injury_weight * light
                + self.energy_weight * energy
                + self.toxicity_weight * toxicity
            )
            ranked.append(branch)
        ranked = sorted(
            ranked,
            key=lambda b: (
                float(b.severity_score or 0.0),
                float(b.p_branch_first or 0.0),
                float(b.score or 0.0),
            ),
            reverse=True,
        )
        for branch in ranked:
            trace_rows.append(
                {
                    "branch_id": branch.branch_id,
                    "severity_score": float(branch.severity_score or 0.0),
                    "death_count": float(branch.meta.get("death_count", 0.0) or 0.0),
                    "serious_injury_count": float(branch.meta.get("serious_injury_count", 0.0) or 0.0),
                    "light_injury_count": float(branch.meta.get("light_injury_count", 0.0) or 0.0),
                    "energy_level": float(branch.meta.get("energy_level", 0.0) or 0.0),
                    "toxicity_level": float(branch.meta.get("toxicity_level", 0.0) or 0.0),
                    "tie_break_p_branch_first": float(branch.p_branch_first or 0.0),
                    "tie_break_branch_score": float(branch.score or 0.0),
                }
            )
        severity_trace = SeverityDecisionTrace(
            triggered=triggered,
            reason=reason,
            ranking=trace_rows,
            selected_branch_id=trace_rows[0]["branch_id"] if trace_rows else None,
        )
        return ranked, severity_trace