from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class CounterfactualSamplerConfig:
    same_target: bool = True
    same_layer: bool = True
    max_twin_per_positive: int = 2
    actor_overlap_min: float = 0.2


class CounterfactualSampler:
    """Twin-negative sampler using simple structural constraints."""

    def __init__(self, config: Dict[str, Any] | None = None):
        raw = config or {}
        self.cfg = CounterfactualSamplerConfig(
            same_target=bool(raw.get("same_target", True)),
            same_layer=bool(raw.get("same_layer", True)),
            max_twin_per_positive=int(raw.get("max_twin_per_positive", 2)),
            actor_overlap_min=float(raw.get("actor_overlap_min", 0.2)),
        )

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        sa = {x for x in str(a).lower().split() if x}
        sb = {x for x in str(b).lower().split() if x}
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def build_pairs(self, edge_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        positives = [r for r in edge_rows if int(r.get("silver_edge_causal", -1)) == 1]
        candidates = [r for r in edge_rows if int(r.get("silver_edge_causal", -1)) in {0, -1}]
        pairs: List[Dict[str, Any]] = []

        for pos in positives:
            pos_target = pos.get("target_node_id")
            pos_layer = pos.get("source_layer")
            pos_ev = str(pos.get("evidence_unit_id") or "")
            twins = []

            for neg in candidates:
                if self.cfg.same_target and neg.get("target_node_id") != pos_target:
                    continue
                if self.cfg.same_layer and neg.get("source_layer") != pos_layer:
                    continue
                if str(neg.get("evidence_unit_id") or "") == pos_ev and pos_ev:
                    continue
                overlap = self._token_overlap(pos.get("source_text") or "", neg.get("source_text") or "")
                if overlap < self.cfg.actor_overlap_min:
                    continue
                twins.append((overlap, neg))

            twins.sort(key=lambda x: x[0], reverse=True)
            for rank, (_, neg) in enumerate(twins[: self.cfg.max_twin_per_positive], start=1):
                pairs.append(
                    {
                        "positive_pseudo_label_id": pos.get("pseudo_label_id"),
                        "twin_pseudo_label_id": neg.get("pseudo_label_id"),
                        "target_node_id": pos_target,
                        "rank": rank,
                        "overlap": _,
                    }
                )
        return pairs