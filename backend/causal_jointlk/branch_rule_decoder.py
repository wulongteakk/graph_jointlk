from __future__ import annotations

import math
from typing import Sequence

from .schemas import BranchDecision, CandidateBranch


class BranchRuleDecoder:
    def __init__(
        self,
        gap_threshold: float = 0.15,
        temporal_violation_penalty: float = 0.8,
        cycle_penalty: float = 1.2,
        downstream_enable_penalty: float = 0.6,
    ):
        self.gap_threshold = float(gap_threshold)
        self.temporal_violation_penalty = float(temporal_violation_penalty)
        self.cycle_penalty = float(cycle_penalty)
        self.downstream_enable_penalty = float(downstream_enable_penalty)

    def decide(self, branches: Sequence[CandidateBranch]) -> BranchDecision:
        if not branches:
            return BranchDecision(None, 0.0, False, "no_candidate", ranking=[], trace={})

        ranking = []
        for branch in branches:
            first_raw = self._compute_first_induced_raw(branch)
            penalty = self._compute_rule_penalty(branch)
            branch.p_branch_first = self._sigmoid(first_raw - penalty)
            branch.decision_gap = 0.0
            ranking.append(
                {
                    "branch_id": branch.branch_id,
                    "p_branch_first": float(branch.p_branch_first),
                    "raw_score": float(first_raw),
                    "penalty": float(penalty),
                    "rule_hits": list(branch.rule_hits or []),
                }
            )

        ranking = sorted(ranking, key=lambda x: x["p_branch_first"], reverse=True)
        best = ranking[0]
        second = ranking[1] if len(ranking) > 1 else None
        gap = float(best["p_branch_first"] - (second["p_branch_first"] if second else 0.0))
        selected_id = best["branch_id"]

        for branch in branches:
            if branch.branch_id == selected_id:
                branch.decision_gap = gap

        fallback = gap < self.gap_threshold
        reason = "first_induced_gap_ok" if not fallback else "decision_gap_below_threshold"
        return BranchDecision(
            selected_branch_id=selected_id,
            decision_gap=gap,
            needs_severity_fallback=fallback,
            reason=reason,
            ranking=ranking,
            trace={
                "gap_threshold": self.gap_threshold,
                "selected_branch_id": selected_id,
            },
        )

    def _compute_first_induced_raw(self, branch: CandidateBranch) -> float:
        mean_edge_score = float(branch.score or 0.0)
        avg_enable = self._avg_prob(branch, "p_enable")
        avg_temporal = self._avg_prob(branch, "p_temporal_before")
        first_cue_count = float(len(branch.meta.get("first_cue_hits") or []))
        shared_downstream = float(branch.meta.get("shared_downstream_count", 0.0) or 0.0)
        return (
            0.45 * mean_edge_score
            + 0.20 * avg_enable
            + 0.20 * avg_temporal
            + 0.10 * min(first_cue_count, 3.0)
            + 0.05 * shared_downstream
        )

    def _compute_rule_penalty(self, branch: CandidateBranch) -> float:
        temporal_viol = float(branch.meta.get("temporal_violation_count", 0.0) or 0.0)
        cycle_flag = 1.0 if branch.meta.get("cycle_flag") else 0.0
        downstream_enable_flag = 1.0 if branch.meta.get("downstream_enable_flag") else 0.0
        return (
            self.temporal_violation_penalty * temporal_viol
            + self.cycle_penalty * cycle_flag
            + self.downstream_enable_penalty * downstream_enable_flag
        )

    @staticmethod
    def _avg_prob(branch: CandidateBranch, attr: str) -> float:
        if not branch.path_edges:
            return 0.0
        vals = [float(getattr(edge, attr, 0.0) or 0.0) for edge in branch.path_edges]
        return sum(vals) / max(len(vals), 1)

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(min(value, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-value))
