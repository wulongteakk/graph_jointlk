from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .schemas import CausalEdge, CausalNode

STRUCTURAL_REL_TYPES = {"PART_OF", "NEXT_CHUNK", "HAS_ENTITY", "_Bloom_Perspective_"}


@dataclass
class CandidateEdge:
    doc_id: Optional[str]
    file_name: Optional[str]
    kg_scope: Optional[str]
    kg_id: Optional[str]
    source_node_id: str
    source_text: str
    source_layer: Optional[str]
    source_labels: List[str]
    source_props: Dict[str, Any]
    target_node_id: str
    target_text: str
    target_layer: Optional[str]
    target_labels: List[str]
    target_props: Dict[str, Any]
    relation_type: str
    rel_props: Dict[str, Any]
    evidence_text: Optional[str]
    source_chunk_id: Optional[str]
    source_chunk_pos: Optional[int]
    target_chunk_id: Optional[str]
    target_chunk_pos: Optional[int]


class Neo4jAccessor:
    """
    只面向 Instance-KG 的轻量访问器：
    - 列出文档
    - 拉取文档内实体边
    具体 pair-centered 子图在 Python 里构造，避免对 APOC / 可变长路径做强依赖。
    """

    def __init__(self, graph: Any):
        self.graph = graph

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        if hasattr(self.graph, "query"):
            return self.graph.query(query, params=params)
        if hasattr(self.graph, "execute"):
            return self.graph.execute(query, params)
        raise TypeError("Unsupported graph client: expected LangChain Neo4jGraph-like object with .query().")

    def get_k_hop_subgraph(
            self,
            *,
            seed_node_ids: Sequence[str],
            prior: Any,
            k_hop: int = 2,
            doc_id: Optional[str] = None,
            kg_scope: str = "instance",
            kg_id: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        if not seed_node_ids:
            return {"nodes": [], "edges": []}

        relation_whitelist = sorted(set(str(x).upper() for x in (prior.rel_whitelist if prior else []) if str(x)))
        raw_edges = self.get_doc_candidate_edges(
            doc_id=doc_id,
            kg_scope=kg_scope,
            kg_id=kg_id,
            relation_types=relation_whitelist if relation_whitelist else None,
            limit=None,
        )
        if not raw_edges:
            return {"nodes": [], "edges": []}

        adjacency: Dict[str, List[CandidateEdge]] = {}
        for edge in raw_edges:
            adjacency.setdefault(edge.source_node_id, []).append(edge)
            adjacency.setdefault(edge.target_node_id, []).append(edge)

        seed_set = set(str(x) for x in seed_node_ids if x)
        visited = set(seed_set)
        frontier = set(seed_set)
        selected_edge_keys = set()
        selected_edges: List[CandidateEdge] = []

        for _ in range(max(1, int(k_hop))):
            if not frontier:
                break
            next_frontier = set()
            for nid in frontier:
                for edge in adjacency.get(nid, []):
                    key = (
                        edge.source_node_id,
                        edge.target_node_id,
                        edge.relation_type,
                        edge.source_chunk_id,
                        edge.target_chunk_id,
                    )
                    if key not in selected_edge_keys:
                        selected_edge_keys.add(key)
                        selected_edges.append(edge)
                    for neigh in (edge.source_node_id, edge.target_node_id):
                        if neigh not in visited:
                            visited.add(neigh)
                            next_frontier.add(neigh)
            frontier = next_frontier

        node_map: Dict[str, CausalNode] = {}
        edge_rows: List[CausalEdge] = []
        for edge in selected_edges:
            src_layer = (prior.canonical_layer(edge.source_layer) if prior else (edge.source_layer or "UNK"))
            tgt_layer = (prior.canonical_layer(edge.target_layer) if prior else (edge.target_layer or "UNK"))
            src_layer = str(src_layer or "UNK").upper()
            tgt_layer = str(tgt_layer or "UNK").upper()

            if edge.source_node_id not in node_map:
                node_map[edge.source_node_id] = CausalNode(
                    node_id=edge.source_node_id,
                    text=edge.source_text,
                    layer=src_layer,
                    doc_id=edge.doc_id,
                    kg_scope=edge.kg_scope,
                    kg_id=edge.kg_id,
                    labels=edge.source_labels,
                    properties=edge.source_props,
                )
            if edge.target_node_id not in node_map:
                node_map[edge.target_node_id] = CausalNode(
                    node_id=edge.target_node_id,
                    text=edge.target_text,
                    layer=tgt_layer,
                    doc_id=edge.doc_id,
                    kg_scope=edge.kg_scope,
                    kg_id=edge.kg_id,
                    labels=edge.target_labels,
                    properties=edge.target_props,
                )

            rel = prior.normalize_relation(edge.relation_type, edge.evidence_text) if prior else edge.relation_type
            edge_id = f"{edge.source_node_id}|{rel}|{edge.target_node_id}"
            edge_rows.append(
                CausalEdge(
                    edge_id=edge_id,
                    source_id=edge.source_node_id,
                    target_id=edge.target_node_id,
                    relation=rel,
                    source_text=edge.source_text,
                    target_text=edge.target_text,
                    source_layer=src_layer,
                    target_layer=tgt_layer,
                    evidence_text=edge.evidence_text,
                    doc_id=edge.doc_id,
                    kg_scope=edge.kg_scope,
                    kg_id=edge.kg_id,
                    label_source=edge.rel_props.get("label_source"),
                    sample_weight=float(edge.rel_props.get("sample_weight", 1.0)),
                    meta={
                        "source_chunk_id": edge.source_chunk_id,
                        "source_chunk_pos": edge.source_chunk_pos,
                        "target_chunk_id": edge.target_chunk_id,
                        "target_chunk_pos": edge.target_chunk_pos,
                    },
                )
            )

        return {"nodes": list(node_map.values()), "edges": edge_rows}

    def hydrate_edge_evidence(self, evidence_store: Any, edges: Sequence[CausalEdge]) -> List[CausalEdge]:
        out: List[CausalEdge] = []
        for edge in edges:
            if edge.evidence_unit_id:
                unit = evidence_store.get_unit(edge.evidence_unit_id) if evidence_store else None
                if unit is not None:
                    edge.evidence_text = edge.evidence_text or getattr(unit, "content", None)
            out.append(edge)
        return out

    def list_documents(
        self,
        *,
        kg_scope: str = "instance",
        kg_id: Optional[str] = None,
        doc_ids: Optional[Sequence[str]] = None,
        file_names: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = """
        MATCH (d:SourceDocument:Document)
        WITH d, properties(d) AS dp
        WHERE ($kg_scope IS NULL OR coalesce(d.kg_scope, 'instance') = $kg_scope)
          AND ($kg_id IS NULL OR d.kg_id = $kg_id)
          AND ($doc_ids IS NULL OR coalesce(d.doc_id, d.id, d.fileName, dp['file_name']) IN $doc_ids)
          AND ($file_names IS NULL OR coalesce(d.fileName, dp['file_name'], d.name) IN $file_names)
        RETURN DISTINCT
          coalesce(d.doc_id, d.id, d.fileName, dp['file_name']) AS doc_id,
          coalesce(d.fileName, dp['file_name'], d.name) AS file_name,
          coalesce(d.kg_scope, 'instance') AS kg_scope,
          d.kg_id AS kg_id
        ORDER BY file_name
        """
        rows = self._run(
            query,
            {
                "kg_scope": kg_scope,
                "kg_id": kg_id,
                "doc_ids": list(doc_ids) if doc_ids else None,
                "file_names": list(file_names) if file_names else None,
            },
        )
        if limit is not None:
            rows = rows[: int(limit)]
        return rows

    def get_doc_candidate_edges(
        self,
        *,
        doc_id: Optional[str] = None,
        file_name: Optional[str] = None,
        kg_scope: str = "instance",
        kg_id: Optional[str] = None,
        relation_types: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[CandidateEdge]:
        relation_types = [str(x).strip().upper() for x in (relation_types or []) if str(x).strip()]
        query = """
        MATCH (d:SourceDocument:Document)
        WITH d, properties(d) AS dp
        WHERE ($kg_scope IS NULL OR coalesce(d.kg_scope, 'instance') = $kg_scope)
          AND ($kg_id IS NULL OR d.kg_id = $kg_id)
          AND ($doc_id IS NULL OR coalesce(d.doc_id, d.id, d.fileName, dp['file_name']) = $doc_id)
          AND ($file_name IS NULL OR coalesce(d.fileName, dp['file_name'], d.name) = $file_name)
        MATCH (d)<-[:PART_OF]-(c1:Chunk)-[:HAS_ENTITY]->(s)-[r]->(t)<-[:HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d)
        WITH d, dp, c1, c2, s, t, r, properties(s) AS sp, properties(t) AS tp
        WHERE NOT type(r) IN $structural_rel_types
          AND ($relation_types = [] OR toUpper(type(r)) IN $relation_types)
        RETURN DISTINCT
          coalesce(d.doc_id, d.id, d.fileName, dp['file_name']) AS doc_id,
          coalesce(d.fileName, dp['file_name'], d.name) AS file_name,
          coalesce(d.kg_scope, 'instance') AS kg_scope,
          d.kg_id AS kg_id,
          coalesce(sp['node_id'], s.id, sp['uuid'], s.name, s.text, sp['value'], elementId(s)) AS source_node_id,
          coalesce(s.name, s.text, sp['value'], s.id, sp['node_id'], elementId(s)) AS source_text,
          coalesce(sp['layer'], sp['layer_type'], sp['ctp_layer'], sp['stage'], sp['role'], head(labels(s))) AS source_layer,
          labels(s) AS source_labels,
          sp AS source_props,
          coalesce(tp['node_id'], t.id, tp['uuid'], t.name, t.text, tp['value'], elementId(t)) AS target_node_id,
          coalesce(t.name, t.text, tp['value'], t.id, tp['node_id'], elementId(t)) AS target_text,
          coalesce(tp['layer'], tp['layer_type'], tp['ctp_layer'], tp['stage'], tp['role'], head(labels(t))) AS target_layer,
          labels(t) AS target_labels,
          tp AS target_props,
          type(r) AS relation_type,
          properties(r) AS rel_props,
          coalesce(
            properties(r)['evidence_text'],
            properties(r)['support_text'],
            properties(r)['text'],
            properties(r)['evidence'],
            properties(r)['summary']
          ) AS evidence_text,
          coalesce(properties(c1)['chunk_id'], c1.id, properties(c1)['uuid'], elementId(c1)) AS source_chunk_id,
          c1.position AS source_chunk_pos,
          coalesce(properties(c2)['chunk_id'], c2.id, properties(c2)['uuid'], elementId(c2)) AS target_chunk_id,
          c2.position AS target_chunk_pos
        ORDER BY file_name, relation_type, source_text, target_text
        """
        rows = self._run(
            query,
            {
                "kg_scope": kg_scope,
                "kg_id": kg_id,
                "doc_id": doc_id,
                "file_name": file_name,
                "relation_types": relation_types,
                "structural_rel_types": sorted(STRUCTURAL_REL_TYPES),
            },
        )
        if limit is not None:
            rows = rows[: int(limit)]

        out: List[CandidateEdge] = []
        for row in rows:
            out.append(
                CandidateEdge(
                    doc_id=row.get("doc_id"),
                    file_name=row.get("file_name"),
                    kg_scope=row.get("kg_scope"),
                    kg_id=row.get("kg_id"),
                    source_node_id=str(row.get("source_node_id")),
                    source_text=str(row.get("source_text") or row.get("source_node_id") or ""),
                    source_layer=row.get("source_layer"),
                    source_labels=list(row.get("source_labels") or []),
                    source_props=dict(row.get("source_props") or {}),
                    target_node_id=str(row.get("target_node_id")),
                    target_text=str(row.get("target_text") or row.get("target_node_id") or ""),
                    target_layer=row.get("target_layer"),
                    target_labels=list(row.get("target_labels") or []),
                    target_props=dict(row.get("target_props") or {}),
                    relation_type=str(row.get("relation_type") or ""),
                    rel_props=dict(row.get("rel_props") or {}),
                    evidence_text=row.get("evidence_text"),
                    source_chunk_id=row.get("source_chunk_id"),
                    source_chunk_pos=row.get("source_chunk_pos"),
                    target_chunk_id=row.get("target_chunk_id"),
                    target_chunk_pos=row.get("target_chunk_pos"),
                )
            )
        return out

    def get_doc_entity_mentions(
        self,
        *,
        doc_id: Optional[str] = None,
        file_name: Optional[str] = None,
        kg_scope: str = "instance",
        kg_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = """
        MATCH (d:SourceDocument:Document)
        WITH d, properties(d) AS dp
        WHERE ($kg_scope IS NULL OR coalesce(d.kg_scope, 'instance') = $kg_scope)
          AND ($kg_id IS NULL OR d.kg_id = $kg_id)
          AND ($doc_id IS NULL OR coalesce(d.doc_id, d.id, d.fileName, dp['file_name']) = $doc_id)
          AND ($file_name IS NULL OR coalesce(d.fileName, dp['file_name'], d.name) = $file_name)
        MATCH (d)<-[:PART_OF]-(c:Chunk)-[:HAS_ENTITY]->(n)
        WITH d, dp, c, n, properties(n) AS np
        RETURN DISTINCT
          coalesce(d.doc_id, d.id, d.fileName, dp['file_name']) AS doc_id,
          coalesce(d.fileName, dp['file_name'], d.name) AS file_name,
          coalesce(d.kg_scope, 'instance') AS kg_scope,
          d.kg_id AS kg_id,
          coalesce(np['node_id'], n.id, np['uuid'], n.name, n.text, np['value'], elementId(n)) AS node_id,
          coalesce(n.name, n.text, np['value'], n.id, np['node_id'], elementId(n)) AS node_text,
          coalesce(np['layer'], np['layer_type'], np['ctp_layer'], np['stage'], np['role'], head(labels(n))) AS node_layer,
          labels(n) AS node_labels,
          np AS node_props,
          coalesce(properties(c)['chunk_id'], c.id, properties(c)['uuid'], elementId(c)) AS chunk_id,
          c.position AS chunk_pos
        ORDER BY file_name, chunk_pos, node_text
        """
        rows = self._run(
            query,
            {
                "kg_scope": kg_scope,
                "kg_id": kg_id,
                "doc_id": doc_id,
                "file_name": file_name,
            },
        )
        if limit is not None:
            rows = rows[: int(limit)]
        return [dict(row) for row in rows]

    def upsert_implicit_candidate_edges(
        self,
        *,
        edges: Sequence[CandidateEdge],
        doc_id: Optional[str],
        file_name: Optional[str],
        kg_scope: str = "instance",
        kg_id: Optional[str] = None,
    ) -> int:
        payload: List[Dict[str, Any]] = []
        for edge in edges:
            rel = str(edge.relation_type or "").upper()
            if rel not in {"POTENTIAL_CAUSE", "POTENTIAL_ENABLE", "PRECEDES"}:
                continue
            payload.append(
                {
                    "source_node_id": str(edge.source_node_id),
                    "target_node_id": str(edge.target_node_id),
                    "relation_type": rel,
                    "props": dict(edge.rel_props or {}),
                }
            )
        if not payload:
            return 0

        query = """
        MATCH (d:SourceDocument:Document)
        WITH d, properties(d) AS dp
        WHERE ($kg_scope IS NULL OR coalesce(d.kg_scope, 'instance') = $kg_scope)
          AND ($kg_id IS NULL OR d.kg_id = $kg_id)
          AND ($doc_id IS NULL OR coalesce(d.doc_id, d.id, d.fileName, dp['file_name']) = $doc_id)
          AND ($file_name IS NULL OR coalesce(d.fileName, dp['file_name'], d.name) = $file_name)
        UNWIND $rows AS row
        MATCH (d)<-[:PART_OF]-(:Chunk)-[:HAS_ENTITY]->(s)
        WHERE coalesce(properties(s)['node_id'], s.id, properties(s)['uuid'], s.name, s.text, properties(s)['value'], elementId(s)) = row.source_node_id
        MATCH (d)<-[:PART_OF]-(:Chunk)-[:HAS_ENTITY]->(t)
        WHERE coalesce(properties(t)['node_id'], t.id, properties(t)['uuid'], t.name, t.text, properties(t)['value'], elementId(t)) = row.target_node_id
        FOREACH (_ IN CASE WHEN row.relation_type = 'POTENTIAL_CAUSE' THEN [1] ELSE [] END |
            MERGE (s)-[r:POTENTIAL_CAUSE]->(t)
            SET r += row.props, r.candidate_source='implicit', r.candidate_persisted=true
        )
        FOREACH (_ IN CASE WHEN row.relation_type = 'POTENTIAL_ENABLE' THEN [1] ELSE [] END |
            MERGE (s)-[r:POTENTIAL_ENABLE]->(t)
            SET r += row.props, r.candidate_source='implicit', r.candidate_persisted=true
        )
        FOREACH (_ IN CASE WHEN row.relation_type = 'PRECEDES' THEN [1] ELSE [] END |
            MERGE (s)-[r:PRECEDES]->(t)
            SET r += row.props, r.candidate_source='implicit', r.candidate_persisted=true
        )
        RETURN count(*) AS persisted
        """
        rows = self._run(
            query,
            {
                "kg_scope": kg_scope,
                "kg_id": kg_id,
                "doc_id": doc_id,
                "file_name": file_name,
                "rows": payload,
            },
        )
        return int((rows[0] or {}).get("persisted") or 0) if rows else 0