
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .evidence_gate import choose_best_evidence
from .prior import CausalPrior
from .schemas import CausalEdge, CausalNode


class InstanceCausalKGBuilder:
    """将普通抽取结果后处理成“因果 Instance-KG”。

    输入可以是：
    - 已经抽出来的 node/edge dict
    - EvidenceStore 中的 evidence_units

    输出：
    - 仅保留/规范化因果关系白名单
    - 在边上补充 doc_id / kg_scope / kg_id / evidence_unit_id / span
    """

    def __init__(self, graph: Any, prior: CausalPrior, evidence_store: Any):
        self.graph = graph
        self.prior = prior
        self.evidence_store = evidence_store

    def canonicalize_edges(
        self,
        nodes: Sequence[CausalNode],
        raw_edges: Sequence[Dict[str, Any]],
        evidence_units: Sequence[Dict[str, Any]],
        doc_id: Optional[str],
        kg_scope: str = "instance",
        kg_id: Optional[str] = None,
    ) -> List[CausalEdge]:
        node_map = {n.node_id: n for n in nodes}
        out: List[CausalEdge] = []
        for row in raw_edges:
            source_id = row.get("source_id")
            target_id = row.get("target_id")
            if source_id not in node_map or target_id not in node_map:
                continue

            raw_relation = row.get("relation") or row.get("type") or row.get("rel_type")
            edge_text = row.get("text") or row.get("evidence_text") or ""
            relation = self.prior.normalize_relation(raw_relation, edge_text)
            if not self.prior.is_relation_allowed(relation):
                continue

            source_node = node_map[source_id]
            target_node = node_map[target_id]
            best = choose_best_evidence(
                prior=self.prior,
                source_text=source_node.text,
                target_text=target_node.text,
                relation=relation,
                evidence_units=evidence_units,
                relation_text=edge_text,
            )
            edge = CausalEdge(
                edge_id=row.get("edge_id") or str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                source_text=source_node.text,
                target_text=target_node.text,
                source_layer=source_node.layer,
                target_layer=target_node.layer,
                evidence_unit_id=best.get("unit_id"),
                evidence_text=best.get("content"),
                doc_id=doc_id,
                kg_scope=kg_scope,
                kg_id=kg_id,
                supported=bool(best.get("supported")),
                support_score=float(best.get("support_score") or 0.0),
                meta={
                    "source_span": best.get("source_span"),
                    "target_span": best.get("target_span"),
                    "trigger_hits": best.get("trigger_hits") or [],
                },
            )
            out.append(edge)
        return out

    def persist_edges(self, edges: Sequence[CausalEdge]) -> None:
        for edge in edges:
            cypher = f"""
            MATCH (s) WHERE elementId(s) = $source_id
            MATCH (t) WHERE elementId(t) = $target_id
            MERGE (s)-[r:{edge.relation} {{
                doc_id: $doc_id,
                kg_scope: $kg_scope,
                kg_id: $kg_id,
                evidence_unit_id: $evidence_unit_id
            }}]->(t)
            SET r.evidence_text = $evidence_text,
                r.source_span = $source_span,
                r.target_span = $target_span,
                r.support_score = $support_score,
                r.supported = $supported,
                r.trigger_hits = $trigger_hits
            """
            self.graph.query(
                cypher,
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "doc_id": edge.doc_id,
                    "kg_scope": edge.kg_scope,
                    "kg_id": edge.kg_id,
                    "evidence_unit_id": edge.evidence_unit_id,
                    "evidence_text": edge.evidence_text,
                    "source_span": edge.meta.get("source_span") if edge.meta else None,
                    "target_span": edge.meta.get("target_span") if edge.meta else None,
                    "support_score": edge.support_score,
                    "supported": edge.supported,
                    "trigger_hits": edge.meta.get("trigger_hits") if edge.meta else [],
                },
            )
