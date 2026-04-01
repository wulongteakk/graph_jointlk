from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set

from .schemas import CausalEdge, CausalNode


class NodePriorBuilder:
    FIRST_CUE_KEYWORDS = ("先", "首先", "起初", "初始", "最初", "initial", "first", "upstream")

    @staticmethod
    def _build_graph(edges: Sequence[CausalEdge]) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        succ: Dict[str, Set[str]] = defaultdict(set)
        pred: Dict[str, Set[str]] = defaultdict(set)
        for e in edges:
            succ[e.source_id].add(e.target_id)
            pred[e.target_id].add(e.source_id)
        return succ, pred

    @staticmethod
    def _descendants(node_id: str, succ: Dict[str, Set[str]]) -> Set[str]:
        visited: Set[str] = set()
        stack = list(succ.get(node_id, set()))
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(succ.get(cur, set()))
        return visited

    def build(self, nodes: Sequence[CausalNode], edges: Sequence[CausalEdge]) -> List[Dict]:
        indeg = {n.node_id: 0 for n in nodes}
        outdeg = {n.node_id: 0 for n in nodes}
        node_map = {n.node_id: n for n in nodes}
        succ, pred = self._build_graph(edges)
        desc_map = {nid: self._descendants(nid, succ) for nid in node_map.keys()}
        temporal_vals = [float(n.temporal_rank) for n in nodes if n.temporal_rank is not None]
        min_t = min(temporal_vals) if temporal_vals else None
        max_t = max(temporal_vals) if temporal_vals else None

        enable_fanout_map = {n.node_id: 0 for n in nodes}
        for e in edges:
            indeg[e.target_id] = indeg.get(e.target_id, 0) + 1
            outdeg[e.source_id] = outdeg.get(e.source_id, 0) + 1
            if "enable" in str(e.relation).lower() or float(e.p_enable or 0.0) >= 0.5:
                enable_fanout_map[e.source_id] = enable_fanout_map.get(e.source_id, 0) + 1

        rows = []
        for n in nodes:
            upstream_depth = indeg.get(n.node_id, 0)
            fanout = outdeg.get(n.node_id, 0)
            enable_fanout = enable_fanout_map.get(n.node_id, 0)
            my_desc = desc_map.get(n.node_id, set())
            shared_downstream_count = 0
            for d in my_desc:
                has_peer = any(d in ds for other_id, ds in desc_map.items() if other_id != n.node_id)
                if has_peer:
                    shared_downstream_count += 1
            if min_t is not None and max_t is not None and max_t > min_t and n.temporal_rank is not None:
                temporal_earliness = 1.0 - ((float(n.temporal_rank) - min_t) / (max_t - min_t))
            else:
                temporal_earliness = 0.5
            role_cues = [tag for tag in (n.role_tags or []) if "first" in tag.lower() or "upstream" in tag.lower() or "先" in tag]
            keyword_cues = [kw for kw in self.FIRST_CUE_KEYWORDS if kw in (n.text or "").lower()]
            first_cue_count = len(set(role_cues + keyword_cues))
            common_ancestor_bonus = min(1.0, len(pred.get(n.node_id, set())) / 3.0)
            cue_boost = 0.2 if n.canonical_type in {"SourceState", "SourceEvent"} else 0.0
            p_prior = min(
                0.99,
                0.2
                + 0.15 * (fanout > 0)
                + 0.15 * (upstream_depth == 0)
                + 0.12 * min(enable_fanout, 2)
                + 0.10 * min(shared_downstream_count, 3)
                + 0.12 * temporal_earliness
                + 0.10 * min(first_cue_count, 2)
                + 0.06 * common_ancestor_bonus
                + cue_boost,
            )
            rows.append(
                {
                    "node_id": n.node_id,
                    "canonical_type": n.canonical_type,
                    "domain_id": n.domain_id,
                    "module_id": n.module_id,
                    "scenario_tags": n.scenario_tags,
                    "temporal_rank": n.temporal_rank,
                    "upstream_depth": upstream_depth,
                    "enable_fanout": enable_fanout,
                    "shared_downstream_count": shared_downstream_count,
                    "temporal_earliness": temporal_earliness,
                    "first_cue_count": first_cue_count,
                    "first_cue_hits": sorted(set(role_cues + keyword_cues)),
                    "common_ancestor_bonus": common_ancestor_bonus,
                    "severity_signals": n.severity_signals,
                    "p_node_first_prior": p_prior,
                    "prior_sources": ["node_first_prior_builder_v2_branch_aware"],
                }
            )
        return rows