from __future__ import annotations

from typing import List, Sequence

from .schemas import CandidateBranch, CandidateChain


class BranchBuilder:
    def build(self, chains: Sequence[CandidateChain]) -> List[CandidateBranch]:
        branches: List[CandidateBranch] = []
        for idx, c in enumerate(chains):
            evidence_ids = [e.evidence_unit_id for e in c.edges if e.evidence_unit_id]
            branches.append(
                CandidateBranch(
                    branch_id=f"branch::{idx}",
                    chain_ids=[c.chain_id],
                    first_node_id=c.nodes[0] if c.nodes else None,
                    score=c.score,
                    evidence_unit_ids=evidence_ids,
                    rule_hits=["chain_score"],
                )
            )
        return sorted(branches, key=lambda x: x.score, reverse=True)