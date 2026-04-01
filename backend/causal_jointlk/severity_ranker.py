from __future__ import annotations

from typing import Sequence

from .schemas import CandidateBranch


class SeverityRanker:
    def rerank(self, branches: Sequence[CandidateBranch]) -> Sequence[CandidateBranch]:
        ranked = []
        for branch in branches:
            death = float(branch.meta.get("death_count", 0.0))
            serious = float(branch.meta.get("serious_injury_count", 0.0))
            light = float(branch.meta.get("light_injury_count", 0.0))
            energy = float(branch.meta.get("energy_level", 0.0))
            toxicity = float(branch.meta.get("toxicity_level", 0.0))
            branch.severity_score = 5.0 * death + 3.0 * serious + 1.0 * light + 1.0 * energy + 1.0 * toxicity
            ranked.append(branch)
        return sorted(ranked, key=lambda b: (b.severity_score, b.score), reverse=True)