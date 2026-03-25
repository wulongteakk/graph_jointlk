from __future__ import annotations

from typing import Sequence

from .schemas import BranchDecision, CandidateBranch


class BranchRuleDecoder:
    def decide(self, branches: Sequence[CandidateBranch]) -> BranchDecision:
        if not branches:
            return BranchDecision(None, 0.0, False, "no_candidate")
        best = branches[0]
        second = branches[1] if len(branches) > 1 else None
        gap = best.score - (second.score if second else 0.0)
        return BranchDecision(
            selected_branch_id=best.branch_id,
            decision_gap=float(gap),
            needs_severity_fallback=gap < 0.2,
            reason="gap_threshold" if gap >= 0.2 else "needs_severity_fallback",
        )