
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .prior import CausalPrior
from .schemas import CausalEdge, CausalNode


class InstanceKGAccessor:
    """Queries only Instance-KG, never BG-KG."""

    def __init__(self, graph: Any):
        self.graph = graph

    def find_seed_nodes(
        self,
        query_text: str,
        doc_id: Optional[str] = None,
        kg_scope: Optional[str] = "instance",
        kg_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[CausalNode]:
        cypher = """
        MATCH (n)
        WHERE NOT n:Chunk AND NOT n:Document
          AND ($doc_id IS NULL OR coalesce(n.doc_id, '') = $doc_id)
          AND ($kg_scope IS NULL OR coalesce(n.kg_scope, '') = $kg_scope)
          AND ($kg_id IS NULL OR coalesce(n.kg_id, '') = $kg_id)
          AND (
                toLower(coalesce(n.name, '')) CONTAINS toLower($query_text)
             OR toLower(coalesce(n.id, '')) CONTAINS toLower($query_text)
             OR toLower(coalesce(n.text, '')) CONTAINS toLower($query_text)
          )
        RETURN elementId(n) AS node_id,
               labels(n) AS labels,
               properties(n) AS props
        LIMIT $limit
        """
        rows = self.graph.query(
            cypher,
            {
                "query_text": query_text,
                "doc_id": doc_id,
                "kg_scope": kg_scope,
                "kg_id": kg_id,
                "limit": int(limit),
            },
        )
        return [self._row_to_node(x) for x in rows]

    def get_k_hop_subgraph(
        self,
        seed_node_ids: Sequence[str],
        prior: CausalPrior,
        k_hop: int = 2,
        doc_id: Optional[str] = None,
        kg_scope: Optional[str] = "instance",
        kg_id: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        if not seed_node_ids:
            return {"nodes": [], "edges": []}

        rel_types = list(prior.rel_whitelist)
        if not rel_types:
            return {"nodes": [], "edges": []}

        cypher = """
        MATCH (s)
        WHERE elementId(s) IN $seed_node_ids
        CALL apoc.path.subgraphAll(
          s,
          {
            maxLevel: $k_hop,
            relationshipFilter: $relationship_filter,
            labelFilter: '-Chunk|-Document'
          }
        ) YIELD nodes, relationships
        WITH apoc.coll.toSet(apoc.coll.flatten(collect(nodes))) AS ns,
             apoc.coll.toSet(apoc.coll.flatten(collect(relationships))) AS rs
        UNWIND ns AS n
        WITH collect(DISTINCT n) AS nodes, rs
        UNWIND rs AS r
        WITH nodes, collect(DISTINCT r) AS relationships
        RETURN
          [n IN nodes
             WHERE ($doc_id IS NULL OR coalesce(n.doc_id, '') = $doc_id)
               AND ($kg_scope IS NULL OR coalesce(n.kg_scope, '') = $kg_scope)
               AND ($kg_id IS NULL OR coalesce(n.kg_id, '') = $kg_id)
           | {
               node_id: elementId(n),
               labels: labels(n),
               props: properties(n)
             }] AS nodes,
          [r IN relationships
             WHERE type(r) IN $rel_types
               AND ($doc_id IS NULL OR coalesce(r.doc_id, '') = $doc_id)
               AND ($kg_scope IS NULL OR coalesce(r.kg_scope, '') = $kg_scope)
               AND ($kg_id IS NULL OR coalesce(r.kg_id, '') = $kg_id)
           | {
               edge_id: elementId(r),
               rel_type: type(r),
               props: properties(r),
               source_id: elementId(startNode(r)),
               target_id: elementId(endNode(r))
             }] AS edges
        """
        rows = self.graph.query(
            cypher,
            {
                "seed_node_ids": list(seed_node_ids),
                "k_hop": int(k_hop),
                "relationship_filter": "|".join([f"{rel}>|<{rel}" for rel in rel_types]),
                "rel_types": rel_types,
                "doc_id": doc_id,
                "kg_scope": kg_scope,
                "kg_id": kg_id,
            },
        )
        if not rows:
            return {"nodes": [], "edges": []}
        row = rows[0]
        nodes = [self._row_to_node(x) for x in row.get("nodes", [])]
        node_map = {n.node_id: n for n in nodes}
        edges = [self._row_to_edge(x, node_map) for x in row.get("edges", [])]
        return {"nodes": nodes, "edges": edges}

    def hydrate_edge_evidence(
        self,
        evidence_store: Any,
        edges: Sequence[CausalEdge],
    ) -> List[CausalEdge]:
        hydrated: List[CausalEdge] = []
        for edge in edges:
            evidence_text = edge.evidence_text
            unit_id = edge.evidence_unit_id or edge.meta.get("evidence_unit_id")
            if not evidence_text and unit_id:
                unit = evidence_store.get_unit(unit_id)
                if unit is not None:
                    edge.evidence_text = unit.content
            hydrated.append(edge)
        return hydrated

    @staticmethod
    def _row_to_node(row: Dict[str, Any]) -> CausalNode:
        props = dict(row.get("props") or {})
        text = props.get("name") or props.get("text") or props.get("id") or row.get("node_id")
        layer = props.get("layer") or props.get("ctp_layer") or "UNK"
        return CausalNode(
            node_id=row.get("node_id"),
            text=text,
            layer=str(layer).upper(),
            doc_id=props.get("doc_id"),
            kg_scope=props.get("kg_scope"),
            kg_id=props.get("kg_id"),
            labels=list(row.get("labels") or []),
            properties=props,
        )

    @staticmethod
    def _row_to_edge(row: Dict[str, Any], node_map: Dict[str, CausalNode]) -> CausalEdge:
        props = dict(row.get("props") or {})
        source = node_map.get(row.get("source_id"))
        target = node_map.get(row.get("target_id"))
        return CausalEdge(
            edge_id=row.get("edge_id"),
            source_id=row.get("source_id"),
            target_id=row.get("target_id"),
            relation=str(row.get("rel_type") or "UNK").upper(),
            source_text=source.text if source else row.get("source_id"),
            target_text=target.text if target else row.get("target_id"),
            source_layer=source.layer if source else "UNK",
            target_layer=target.layer if target else "UNK",
            evidence_unit_id=props.get("evidence_unit_id"),
            evidence_text=props.get("evidence_text"),
            doc_id=props.get("doc_id"),
            kg_scope=props.get("kg_scope"),
            kg_id=props.get("kg_id"),
            meta=props,
        )
