from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .evidence_gate import choose_best_evidence
from .prior import CausalPrior
from .schemas import CausalEdge


class EdgePostProcessor:
    """Fuse neural multi-head probabilities with evidence gate signals."""

    def __init__(self, prior: CausalPrior):
        self.prior = prior
        beam_cfg = prior.as_beam_params()
        self.weights = {
            "w_causal": float(beam_cfg.get("w_causal", 0.45)),
            "w_enable": float(beam_cfg.get("w_enable", 0.15)),
            "w_dir": float(beam_cfg.get("w_dir", 0.10)),
            "w_temp": float(beam_cfg.get("w_temp", 0.10)),
            "w_evidence": float(beam_cfg.get("w_evidence", 0.10)),
            "w_first": float(beam_cfg.get("w_first", 0.10)),
        }

    def apply_evidence_gate(
        self,
        edge: CausalEdge,
        evidence_by_edge_id: Dict[str, List[Dict[str, Any]]],
    ) -> CausalEdge:
        relation = self.prior.normalize_relation(edge.relation, edge.evidence_text)
        edge.relation = relation
        candidate_units = evidence_by_edge_id.get(edge.edge_id) or []
        if edge.evidence_text:
            candidate_units = [{"unit_id": edge.evidence_unit_id, "content": edge.evidence_text}] + candidate_units
        best = choose_best_evidence(
            prior=self.prior,
            source_text=edge.source_text,
            target_text=edge.target_text,
            relation=relation,
            evidence_units=candidate_units,
            relation_text=edge.relation,
        )
        edge.evidence_unit_id = best.get("unit_id") or edge.evidence_unit_id
        edge.evidence_text = best.get("content") or edge.evidence_text
        edge.supported = bool(best.get("supported"))
        edge.support_score = float(best.get("support_score") or 0.0)
        edge.p_evidence = (
                0.40 * float(best.get("support_score", 0.0))
                + 0.20 * float(best.get("temporal_score", 0.0))
                + 0.20 * float(best.get("direction_score", 0.0))
                + 0.20 * float(best.get("first_induced_score", 0.0))
        )
        edge.meta = dict(edge.meta or {})
        edge.meta["trigger_hits"] = best.get("trigger_hits") or []
        edge.meta["source_span"] = best.get("source_span")
        edge.meta["target_span"] = best.get("target_span")
        edge.meta["evidence_trace"] = {
            "support_score": float(best.get("support_score", 0.0)),
            "temporal_score": float(best.get("temporal_score", 0.0)),
            "direction_score": float(best.get("direction_score", 0.0)),
            "first_induced_score": float(best.get("first_induced_score", 0.0)),
            "severity_signals": best.get("severity_signals", {}),
        }
        return edge

    def fuse_edge_score(self, edge: CausalEdge) -> CausalEdge:
        p_causal = float(edge.p_causal or 0.0)
        p_enable = float(edge.p_enable or 0.0)
        p_dir = float(edge.p_dir or 0.0)
        p_temp = float(edge.p_temporal_before or 0.0)
        p_evidence = float(edge.p_evidence or 0.0)
        p_first = float(edge.p_node_first or 0.0)
        edge.score = (
            self.weights["w_causal"] * p_causal
            + self.weights["w_enable"] * p_enable
            + self.weights["w_dir"] * p_dir
            + self.weights["w_temp"] * p_temp
            + self.weights["w_evidence"] * p_evidence
            + self.weights["w_first"] * p_first
        )
        return edge

    def apply(
        self,
        edges: Sequence[CausalEdge],
        evidence_by_edge_id: Dict[str, List[Dict[str, Any]]],
    ) -> List[CausalEdge]:
        out: List[CausalEdge] = []
        for edge in edges:
            self.apply_evidence_gate(edge, evidence_by_edge_id)
            self.fuse_edge_score(edge)
            out.append(edge)
        return out