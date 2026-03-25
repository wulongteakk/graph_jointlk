
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .neo4j_accessor import CandidateEdge, Neo4jAccessor
from .pseudo_labeler import lexical_overlap


@dataclass
class CandidateGeneratorConfig:
    enabled: bool = True
    max_implicit_edges_per_doc: int = 300
    lexical_match_threshold: float = 0.24
    adjacent_unit_window: int = 1
    rules: Dict[str, bool] = None  # type: ignore[assignment]
    relation_mapping: Dict[str, str] = None  # type: ignore[assignment]
    trigger_words: Dict[str, List[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rules is None:
            self.rules = {
                "same_unit": True,
                "adjacent_unit": True,
                "temporal_precedes": True,
            }
        if self.relation_mapping is None:
            self.relation_mapping = {
                "same_unit": "POTENTIAL_CAUSE",
                "adjacent_unit": "POTENTIAL_ENABLE",
                "temporal_precedes": "PRECEDES",
            }
        if self.trigger_words is None:
            self.trigger_words = {
                "cause": ["导致", "引发", "造成", "致使", "because", "caused", "leads to", "result in"],
                "enable": ["促使", "使得", "加剧", "在.*情况下", "enable", "enables", "allows"],
                "precedes": ["随后", "之后", "先", "后", "then", "after", "followed by"],
            }


class CandidateGenerator:
    """Generate explicit + implicit candidate edges for pseudo-label and JointLK."""

    def __init__(self, accessor: Neo4jAccessor, cfg: Optional[CandidateGeneratorConfig] = None):
        self.accessor = accessor
        self.cfg = cfg or CandidateGeneratorConfig()

    def generate_for_doc(
        self,
        *,
        doc_id: Optional[str],
        file_name: Optional[str],
        kg_scope: str,
        kg_id: Optional[str],
        relation_types: Optional[Sequence[str]],
        limit: Optional[int],
        unit_rows: Sequence[Dict[str, Any]],
    ) -> Tuple[List[CandidateEdge], Dict[str, Any]]:
        explicit_edges = self.accessor.get_doc_candidate_edges(
            doc_id=doc_id,
            file_name=file_name,
            kg_scope=kg_scope,
            kg_id=kg_id,
            relation_types=relation_types,
            limit=limit,
        )
        for edge in explicit_edges:
            edge.rel_props = dict(edge.rel_props or {})
            edge.rel_props.setdefault("candidate_source", "explicit")
            edge.rel_props.setdefault("candidate_rule", "graph_explicit_relation")

        if not self.cfg.enabled:
            return explicit_edges, {
                "num_explicit_edges": len(explicit_edges),
                "num_implicit_edges": 0,
                "implicit_rule_stats": {},
                "dedup_dropped": 0,
            }

        mentions = self.accessor.get_doc_entity_mentions(
            doc_id=doc_id,
            file_name=file_name,
            kg_scope=kg_scope,
            kg_id=kg_id,
        )
        unit_nodes = self._index_nodes_by_unit(mentions=mentions, unit_rows=unit_rows)

        implicit_edges: List[CandidateEdge] = []
        rule_stats = {"same_unit": 0, "adjacent_unit": 0, "temporal_precedes": 0}

        if self.cfg.rules.get("same_unit", True):
            produced = self._generate_same_unit_edges(unit_nodes, doc_id, file_name, kg_scope, kg_id)
            implicit_edges.extend(produced)
            rule_stats["same_unit"] = len(produced)

        if self.cfg.rules.get("adjacent_unit", True):
            produced = self._generate_adjacent_unit_edges(unit_rows, unit_nodes, doc_id, file_name, kg_scope, kg_id)
            implicit_edges.extend(produced)
            rule_stats["adjacent_unit"] = len(produced)

        if self.cfg.rules.get("temporal_precedes", True):
            produced = self._generate_temporal_edges(unit_rows, unit_nodes, doc_id, file_name, kg_scope, kg_id)
            implicit_edges.extend(produced)
            rule_stats["temporal_precedes"] = len(produced)

        merged, dedup_dropped = self._merge_and_dedup(explicit_edges, implicit_edges)
        return merged, {
            "num_explicit_edges": len(explicit_edges),
            "num_implicit_edges": len(merged) - len(explicit_edges),
            "implicit_rule_stats": rule_stats,
            "dedup_dropped": dedup_dropped,
        }

    def _contains_any(self, text: str, words: Sequence[str]) -> bool:
        t = str(text or "").lower()
        return any(w.lower() in t for w in words)

    def _infer_relation(self, text: str, fallback_rule: str) -> str:
        if self._contains_any(text, self.cfg.trigger_words.get("cause", [])):
            return "POTENTIAL_CAUSE"
        if self._contains_any(text, self.cfg.trigger_words.get("enable", [])):
            return "POTENTIAL_ENABLE"
        if self._contains_any(text, self.cfg.trigger_words.get("precedes", [])):
            return "PRECEDES"
        return self.cfg.relation_mapping.get(fallback_rule, "POTENTIAL_ENABLE")

    def _index_nodes_by_unit(
        self,
        *,
        mentions: Sequence[Dict[str, Any]],
        unit_rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        indexed: Dict[str, List[Dict[str, Any]]] = {}
        by_chunk: Dict[str, List[Dict[str, Any]]] = {}
        for mention in mentions:
            chunk_id = str(mention.get("chunk_id") or "")
            by_chunk.setdefault(chunk_id, []).append(mention)

        for unit in unit_rows:
            uid = str(unit.get("unit_id") or "")
            parent = str(unit.get("parent_evidence_id") or "")
            text = str(unit.get("content") or "")
            node_rows: List[Dict[str, Any]] = []
            for m in by_chunk.get(parent, []):
                node_text = str(m.get("node_text") or "")
                if not node_text.strip():
                    continue
                overlap = lexical_overlap(node_text, text)
                if overlap >= self.cfg.lexical_match_threshold or node_text.lower() in text.lower():
                    mm = dict(m)
                    mm["match_score"] = overlap
                    node_rows.append(mm)
            if node_rows:
                uniq: Dict[str, Dict[str, Any]] = {}
                for r in sorted(node_rows, key=lambda x: float(x.get("match_score", 0.0)), reverse=True):
                    nid = str(r.get("node_id") or "")
                    if nid and nid not in uniq:
                        uniq[nid] = r
                indexed[uid] = list(uniq.values())
        return indexed

    def _build_implicit_edge(
        self,
        *,
        source: Dict[str, Any],
        target: Dict[str, Any],
        relation_type: str,
        rule: str,
        doc_id: Optional[str],
        file_name: Optional[str],
        kg_scope: Optional[str],
        kg_id: Optional[str],
        evidence_unit_ids: Sequence[str],
        score: float,
        evidence_text: Optional[str] = None,
    ) -> CandidateEdge:
        return CandidateEdge(
            doc_id=doc_id,
            file_name=file_name,
            kg_scope=kg_scope,
            kg_id=kg_id,
            source_node_id=str(source.get("node_id") or ""),
            source_text=str(source.get("node_text") or source.get("node_id") or ""),
            source_layer=source.get("node_layer"),
            source_labels=list(source.get("node_labels") or []),
            source_props=dict(source.get("node_props") or {}),
            target_node_id=str(target.get("node_id") or ""),
            target_text=str(target.get("node_text") or target.get("node_id") or ""),
            target_layer=target.get("node_layer"),
            target_labels=list(target.get("node_labels") or []),
            target_props=dict(target.get("node_props") or {}),
            relation_type=relation_type,
            rel_props={
                "candidate_source": "implicit",
                "candidate_rule": rule,
                "candidate_score": round(float(score), 4),
                "evidence_unit_ids": list(dict.fromkeys([str(x) for x in evidence_unit_ids if str(x).strip()])),
            },
            evidence_text=evidence_text,
            source_chunk_id=source.get("chunk_id"),
            source_chunk_pos=source.get("chunk_pos"),
            target_chunk_id=target.get("chunk_id"),
            target_chunk_pos=target.get("chunk_pos"),
        )

    def _generate_same_unit_edges(self, unit_nodes: Dict[str, List[Dict[str, Any]]], doc_id: Optional[str], file_name: Optional[str], kg_scope: Optional[str], kg_id: Optional[str]) -> List[CandidateEdge]:
        out: List[CandidateEdge] = []
        for uid, nodes in unit_nodes.items():
            if len(nodes) < 2:
                continue
            evidence_text = " ".join([str(n.get("node_text") or "") for n in nodes])
            relation = self._infer_relation(evidence_text, "same_unit")
            for a, b in combinations(nodes, 2):
                out.append(
                    self._build_implicit_edge(
                        source=a,
                        target=b,
                        relation_type=relation,
                        rule="same_unit",
                        doc_id=doc_id,
                        file_name=file_name,
                        kg_scope=kg_scope,
                        kg_id=kg_id,
                        evidence_unit_ids=[uid],
                        score=max(float(a.get("match_score", 0.0)), float(b.get("match_score", 0.0))),
                    )
                )
                if relation != "PRECEDES":
                    out.append(
                        self._build_implicit_edge(
                            source=b,
                            target=a,
                            relation_type=relation,
                            rule="same_unit",
                            doc_id=doc_id,
                            file_name=file_name,
                            kg_scope=kg_scope,
                            kg_id=kg_id,
                            evidence_unit_ids=[uid],
                            evidence_text=evidence_text,
                            score=max(float(a.get("match_score", 0.0)), float(b.get("match_score", 0.0))),
                        )
                    )
        return out

    def _generate_adjacent_unit_edges(self, unit_rows: Sequence[Dict[str, Any]], unit_nodes: Dict[str, List[Dict[str, Any]]], doc_id: Optional[str], file_name: Optional[str], kg_scope: Optional[str], kg_id: Optional[str]) -> List[CandidateEdge]:
        out: List[CandidateEdge] = []
        by_parent: Dict[str, List[Dict[str, Any]]] = {}
        for unit in unit_rows:
            by_parent.setdefault(str(unit.get("parent_evidence_id") or ""), []).append(unit)

        window = max(1, int(self.cfg.adjacent_unit_window or 1))
        for parent, units in by_parent.items():
            ordered = sorted(units, key=lambda x: (x.get("start_char") is None, x.get("start_char") or 10**12))
            for i, cur in enumerate(ordered):
                cur_id = str(cur.get("unit_id") or "")
                cur_nodes = unit_nodes.get(cur_id) or []
                if not cur_nodes:
                    continue
                for j in range(i + 1, min(len(ordered), i + 1 + window)):
                    nxt = ordered[j]
                    nxt_id = str(nxt.get("unit_id") or "")
                    nxt_nodes = unit_nodes.get(nxt_id) or []
                    if not nxt_nodes:
                        continue
                    rel = self._infer_relation(str(nxt.get("content") or ""), "adjacent_unit")
                    for a in cur_nodes:
                        for b in nxt_nodes:
                            if str(a.get("node_id")) == str(b.get("node_id")):
                                continue
                            out.append(
                                self._build_implicit_edge(
                                    source=a,
                                    target=b,
                                    relation_type=rel,
                                    rule="adjacent_unit",
                                    doc_id=doc_id,
                                    file_name=file_name,
                                    kg_scope=kg_scope,
                                    kg_id=kg_id,
                                    evidence_unit_ids=[cur_id, nxt_id],
                                    score=(float(a.get("match_score", 0.0)) + float(b.get("match_score", 0.0))) / 2.0,
                                )
                            )
        return out

    def _generate_temporal_edges(self, unit_rows: Sequence[Dict[str, Any]], unit_nodes: Dict[str, List[Dict[str, Any]]], doc_id: Optional[str], file_name: Optional[str], kg_scope: Optional[str], kg_id: Optional[str]) -> List[CandidateEdge]:
        out: List[CandidateEdge] = []
        units = sorted(unit_rows, key=lambda x: (
            str(x.get("parent_evidence_id") or ""),
            x.get("start_char") is None,
            x.get("start_char") or 10**12,
        ))
        for idx, left in enumerate(units[:-1]):
            right = units[idx + 1]
            if str(left.get("parent_evidence_id") or "") != str(right.get("parent_evidence_id") or ""):
                continue
            left_nodes = unit_nodes.get(str(left.get("unit_id") or "")) or []
            right_nodes = unit_nodes.get(str(right.get("unit_id") or "")) or []
            if not left_nodes or not right_nodes:
                continue
            for a in left_nodes:
                for b in right_nodes:
                    if str(a.get("node_id")) == str(b.get("node_id")):
                        continue
                    out.append(
                        self._build_implicit_edge(
                            source=a,
                            target=b,
                            relation_type="PRECEDES",
                            rule="temporal_precedes",
                            doc_id=doc_id,
                            file_name=file_name,
                            kg_scope=kg_scope,
                            kg_id=kg_id,
                            evidence_unit_ids=[str(left.get("unit_id") or ""), str(right.get("unit_id") or "")],
                            score=(float(a.get("match_score", 0.0)) + float(b.get("match_score", 0.0))) / 2.0,
                        )
                    )
        return out

    def _merge_and_dedup(self, explicit_edges: Sequence[CandidateEdge], implicit_edges: Sequence[CandidateEdge]) -> Tuple[List[CandidateEdge], int]:
        out: List[CandidateEdge] = []
        seen: set = set()

        def _k(e: CandidateEdge) -> Tuple[str, str, str]:
            return (str(e.source_node_id), str(e.relation_type).upper(), str(e.target_node_id))

        for edge in explicit_edges:
            key = _k(edge)
            seen.add(key)
            out.append(edge)

        dropped = 0
        max_implicit = max(0, int(self.cfg.max_implicit_edges_per_doc or 0))
        kept_implicit = 0
        for edge in sorted(
            implicit_edges,
            key=lambda x: float((x.rel_props or {}).get("candidate_score") or 0.0),
            reverse=True,
        ):
            key = _k(edge)
            if key in seen:
                dropped += 1
                continue
            if max_implicit and kept_implicit >= max_implicit:
                dropped += 1
                continue
            seen.add(key)
            out.append(edge)
            kept_implicit += 1

        return out, dropped