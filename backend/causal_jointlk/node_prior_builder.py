from __future__ import annotations

from typing import Dict, List, Sequence

from .schemas import CausalEdge, CausalNode


class NodePriorBuilder:
    def build(self, nodes: Sequence[CausalNode], edges: Sequence[CausalEdge]) -> List[Dict]:
        indeg = {n.node_id: 0 for n in nodes}
        outdeg = {n.node_id: 0 for n in nodes}
        for e in edges:
            indeg[e.target_id] = indeg.get(e.target_id, 0) + 1
            outdeg[e.source_id] = outdeg.get(e.source_id, 0) + 1

        rows = []
        for n in nodes:
            upstream_depth = indeg.get(n.node_id, 0)
            fanout = outdeg.get(n.node_id, 0)
            cue_boost = 0.2 if n.canonical_type in {"SourceState", "SourceEvent"} else 0.0
            p_prior = min(0.99, 0.3 + 0.2 * (fanout > 0) + 0.2 * (upstream_depth == 0) + cue_boost)
            rows.append(
                {
                    "node_id": n.node_id,
                    "canonical_type": n.canonical_type,
                    "domain_id": n.domain_id,
                    "module_id": n.module_id,
                    "scenario_tags": n.scenario_tags,
                    "temporal_rank": n.temporal_rank,
                    "upstream_depth": upstream_depth,
                    "enable_fanout": fanout,
                    "shared_downstream_count": 0,
                    "first_cue_hits": n.role_tags,
                    "severity_signals": n.severity_signals,
                    "p_node_first_prior": p_prior,
                    "prior_sources": ["node_first_prior_builder_v1"],
                }
            )
        return rows