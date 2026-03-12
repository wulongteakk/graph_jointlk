from typing import Any, Dict, Iterable, List, Sequence

from .evidence_gate import choose_best_evidence
from .prior import CausalPrior
from .schemas import CausalEdge


class BaselineCausalExtractor:
    """Rule-based / prior-based local edge scorer.

    这个 baseline 很适合做论文实验中的“普通抽取方式”：
    - 只用关系白名单
    - 只用 CTP 约束
    - 只用 evidence gate
    - 不使用神经网络 reranking
    """

    def __init__(self, prior: CausalPrior):
        self.prior = prior

    def score_edges(
        self,
        edges: Sequence[CausalEdge],
        evidence_by_edge_id: Dict[str, List[Dict[str, Any]]],
    ) -> List[CausalEdge]:
        out: List[CausalEdge] = []
        for edge in edges:
            relation = self.prior.normalize_relation(edge.relation, edge.evidence_text)
            edge.relation = relation

            if not self.prior.is_relation_allowed(relation):
                edge.score = -1.0
                out.append(edge)
                continue

            score = 0.0
            if self.prior.allowed_transition(edge.source_layer, edge.target_layer):
                score += 0.30
            else:
                score -= 0.50

            distance = self.prior.layer_distance(edge.source_layer, edge.target_layer)
            score -= 0.05 * max(distance - 1, 0)

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
            edge.meta = dict(edge.meta or {})
            edge.meta["trigger_hits"] = best.get("trigger_hits") or []
            edge.meta["source_span"] = best.get("source_span")
            edge.meta["target_span"] = best.get("target_span")

            score += edge.support_score
            if edge.supported:
                score += 0.25

            edge.score = score
            out.append(edge)
        return out
