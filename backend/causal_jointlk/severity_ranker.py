from __future__ import annotations

from typing import Sequence

from .schemas import CandidateBranch


class SeverityRanker:
    def rerank(self, branches: Sequence[CandidateBranch]) -> Sequence[CandidateBranch]:
        return sorted(branches, key=lambda b: (len(b.evidence_unit_ids), b.score), reverse=True)