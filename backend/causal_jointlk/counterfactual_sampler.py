from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class CounterfactualSamplerConfig:
    same_target: bool = True
    same_layer: bool = True
    same_module: bool = False
    same_scenario_tag: bool = False
    max_twin_per_positive: int = 2
    actor_overlap_min: float = 0.2
    same_module_bonus: float = 0.1
    same_target_bonus: float = 0.1


class CounterfactualSampler:
    """Twin-negative sampler using simple structural constraints."""

    def __init__(self, config: Dict[str, Any] | None = None):
        raw = config or {}
        self.cfg = CounterfactualSamplerConfig(
            same_target=bool(raw.get("same_target", True)),
            same_layer=bool(raw.get("same_layer", True)),
            same_module=bool(raw.get("same_module", False)),
            same_scenario_tag=bool(raw.get("same_scenario_tag", False)),
            max_twin_per_positive=int(raw.get("max_twin_per_positive", 2)),
            actor_overlap_min=float(raw.get("actor_overlap_min", 0.2)),
            same_module_bonus=float(raw.get("same_module_bonus", 0.1)),
            same_target_bonus=float(raw.get("same_target_bonus", 0.1)),
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
            pos_label_id = str(pos.get("pseudo_label_id") or "")
            if not pos_label_id:
                continue
            pos_target = pos.get("target_node_id")
            pos_layer = pos.get("source_layer")
            pos_module = pos.get("module_id")
            pos_tags = set(pos.get("scenario_tags") or [])
            pos_ev = str(pos.get("evidence_unit_id") or "")
            twins: List[Dict[str, Any]] = []

            for neg in candidates:
                neg_label_id = str(neg.get("pseudo_label_id") or "")
                if not neg_label_id or neg_label_id == pos_label_id:
                    continue
                if self.cfg.same_target and neg.get("target_node_id") != pos_target:
                    continue
                if self.cfg.same_layer and neg.get("source_layer") != pos_layer:
                    continue
                module_matched = bool(pos_module and neg.get("module_id") == pos_module)
                if self.cfg.same_module and not module_matched:
                    continue
                neg_tags = set(neg.get("scenario_tags") or [])
                scenario_matched = bool(pos_tags and neg_tags and (pos_tags & neg_tags))
                if self.cfg.same_scenario_tag and not scenario_matched:
                    continue
                if str(neg.get("evidence_unit_id") or "") == pos_ev and pos_ev:
                    continue
                overlap = self._token_overlap(pos.get("source_text") or "", neg.get("source_text") or "")
                if overlap < self.cfg.actor_overlap_min:
                    continue
                same_target_flag = neg.get("target_node_id") == pos_target
                hardness_score = (
                    overlap
                    + (self.cfg.same_module_bonus if module_matched else 0.0)
                    + (self.cfg.same_target_bonus if same_target_flag else 0.0)
                )
                twins.append(
                    {
                        "neg": neg,
                        "overlap": overlap,
                        "hardness_score": hardness_score,
                        "same_module": module_matched,
                        "same_scenario_tag": scenario_matched,
                    }
                )

            twins.sort(key=lambda x: x["hardness_score"], reverse=True)
            for rank, twin in enumerate(twins[: self.cfg.max_twin_per_positive], start=1):
                neg = twin["neg"]
                neg_label_id = str(neg.get("pseudo_label_id") or "")
                twin_group_id = f"twin::{pos_label_id}::{rank}"
                pairs.append(
                    {
                        "cf_pair_id": f"{twin_group_id}::{neg_label_id}",
                        "twin_group_id": twin_group_id,
                        "positive_pseudo_label_id": pos_label_id,
                        "negative_pseudo_label_id": neg_label_id,
                        "twin_pseudo_label_id": neg_label_id,
                        "target_node_id": pos_target,
                        "rank": rank,
                        "overlap": twin["overlap"],
                        "hardness_score": twin["hardness_score"],
                        "same_module": twin["same_module"],
                        "same_scenario_tag": twin["same_scenario_tag"],
                    }
                )
        return pairs