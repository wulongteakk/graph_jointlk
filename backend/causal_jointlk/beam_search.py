import heapq
import uuid
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .prior import CausalPrior
from .schemas import CandidateChain, CausalEdge


class BeamSearchChainBuilder:
    def __init__(self, prior: CausalPrior):
        self.prior = prior
        self.params = prior.as_beam_params()

    def build(
        self,
        edges: Sequence[CausalEdge],
        seed_node_ids: Sequence[str],
        target_node_id: Optional[str] = None,
        top_k: Optional[int] = None,
        max_hops: Optional[int] = None,
    ) -> List[CandidateChain]:
        top_k = int(top_k or self.params.get("top_k", 8))
        max_hops = int(max_hops or self.params.get("max_hops", 6))

        adjacency: Dict[str, List[CausalEdge]] = defaultdict(list)
        for edge in edges:
            if edge.score > -1e8:
                adjacency[edge.source_id].append(edge)

        beam: List[Tuple[float, List[str], List[CausalEdge], List[Dict[str, float]], Dict[str, float]]] = []
        for seed in seed_node_ids:
            heapq.heappush(
                beam,
                (
                    -0.0,
                    [seed],
                    [],
                    [],
                    {
                        "hop_penalty": 0.0,
                        "transition_penalty": 0.0,
                        "unsupported_penalty": 0.0,
                        "duplicate_penalty": 0.0,
                        "direction_penalty": 0.0,
                        "temporal_penalty": 0.0,
                    },
                ),
            )

        finished: List[CandidateChain] = []
        while beam and len(finished) < max(top_k * 5, top_k):
            neg_score, node_path, edge_path, step_trace, penalty_sums = heapq.heappop(beam)
            current_score = -neg_score
            current_node = node_path[-1]

            if edge_path:
                last_edge = edge_path[-1]
                should_finish = target_node_id is not None and current_node == target_node_id
                should_finish = should_finish or (
                    target_node_id is None and str(last_edge.target_layer).upper() == "OUTCOME"
                )
                if should_finish:
                    finished.append(self._build_chain(node_path, edge_path, current_score, step_trace, penalty_sums))

            if len(edge_path) >= max_hops:
                continue

            for next_edge in adjacency.get(current_node, []):
                hop_depth = len(edge_path) + 1
                hop_penalty = float(self.params.get("hop_penalty", 0.05)) * max(hop_depth - 1, 0)
                if next_edge.target_id in node_path:
                    duplicate_penalty = float(self.params.get("duplicate_node_penalty", 0.40))
                else:
                    duplicate_penalty = 0.0

                transition_penalty = 0.0
                if not self.prior.allowed_transition(next_edge.source_layer, next_edge.target_layer):
                    transition_penalty += float(self.params.get("skip_layer_penalty", 0.75))

                distance = self.prior.layer_distance(next_edge.source_layer, next_edge.target_layer)
                transition_penalty += float(self.params.get("distance_penalty", 0.15)) * max(distance - 1, 0)

                unsupported_penalty = 0.0
                if not next_edge.supported:
                    unsupported_penalty = float(self.params.get("unsupported_edge_penalty", 1.0))

                direction_penalty = 0.0
                temporal_penalty = 0.0
                if (next_edge.p_dir or 0.0) < 0.5:
                    direction_penalty += float(self.params.get("direction_penalty", 0.40))
                if (next_edge.p_temporal_before or 0.0) < 0.5:
                    temporal_penalty += float(self.params.get("temporal_penalty", 0.40))

                delta = (
                    float(self.params.get("w_causal", 0.45)) * float(next_edge.p_causal or 0.0)
                    + float(self.params.get("w_enable", 0.15)) * float(next_edge.p_enable or 0.0)
                    + float(self.params.get("w_dir", 0.10)) * float(next_edge.p_dir or 0.0)
                    + float(self.params.get("w_temp", 0.10)) * float(next_edge.p_temporal_before or 0.0)
                    + float(self.params.get("w_evidence", 0.10)) * float(next_edge.p_evidence or 0.0)
                    + float(self.params.get("w_first", 0.10)) * float(next_edge.p_node_first or 0.0)
                    - hop_penalty
                    - transition_penalty
                    - unsupported_penalty
                    - duplicate_penalty
                    - direction_penalty
                    - temporal_penalty
                )
                step_delta = {
                    "edge_id": next_edge.edge_id,
                    "source_id": next_edge.source_id,
                    "target_id": next_edge.target_id,
                    "w_causal": float(self.params.get("w_causal", 0.45)) * float(next_edge.p_causal or 0.0),
                    "w_enable": float(self.params.get("w_enable", 0.15)) * float(next_edge.p_enable or 0.0),
                    "w_dir": float(self.params.get("w_dir", 0.10)) * float(next_edge.p_dir or 0.0),
                    "w_temp": float(self.params.get("w_temp", 0.10)) * float(next_edge.p_temporal_before or 0.0),
                    "w_evidence": float(self.params.get("w_evidence", 0.10)) * float(next_edge.p_evidence or 0.0),
                    "w_first": float(self.params.get("w_first", 0.10)) * float(next_edge.p_node_first or 0.0),
                    "hop_penalty": hop_penalty,
                    "transition_penalty": transition_penalty,
                    "unsupported_penalty": unsupported_penalty,
                    "duplicate_penalty": duplicate_penalty,
                    "direction_penalty": direction_penalty,
                    "temporal_penalty": temporal_penalty,
                    "delta_total": delta,
                }
                new_node_path = node_path + [next_edge.target_id]
                new_edge_path = edge_path + [next_edge]
                new_step_trace = step_trace + [step_delta]
                new_penalty_sums = dict(penalty_sums)
                new_penalty_sums["hop_penalty"] += hop_penalty
                new_penalty_sums["transition_penalty"] += transition_penalty
                new_penalty_sums["unsupported_penalty"] += unsupported_penalty
                new_penalty_sums["duplicate_penalty"] += duplicate_penalty
                new_penalty_sums["direction_penalty"] += direction_penalty
                new_penalty_sums["temporal_penalty"] += temporal_penalty
                new_score = current_score + delta
                heapq.heappush(beam, (-new_score, new_node_path, new_edge_path, new_step_trace, new_penalty_sums))

        finished = sorted(finished, key=lambda x: x.score, reverse=True)
        dedup: List[CandidateChain] = []
        seen = set()
        for chain in finished:
            signature = tuple((e.source_id, e.relation, e.target_id) for e in chain.edges)
            if signature in seen:
                continue
            seen.add(signature)
            dedup.append(chain)
            if len(dedup) >= top_k:
                break
        return dedup

    def _build_chain(
        self,
        node_path: List[str],
        edge_path: List[CausalEdge],
        score: float,
        step_trace: List[Dict[str, float]],
        penalty_sums: Dict[str, float],
    ) -> CandidateChain:
        layers = [edge.source_layer for edge in edge_path] + ([edge_path[-1].target_layer] if edge_path else [])
        missing_layers = self.prior.count_missing_layers(layers)
        score -= float(self.params.get("missing_layer_penalty", 0.50)) * len(missing_layers)

        unsupported_edges = [edge.edge_id for edge in edge_path if not edge.supported]
        chain_id = str(uuid.uuid4())
        chain = CandidateChain(
            chain_id=chain_id,
            nodes=node_path,
            edges=list(edge_path),
            score=score,
            missing_layers=missing_layers,
            unsupported_edges=unsupported_edges,
        )
        chain.meta = {
            "beam_rank_score": score,
            "step_scores": [float(e.score) for e in edge_path],
            "step_trace": step_trace,
            "penalties": {
                "hop_penalty": float(penalty_sums.get("hop_penalty", 0.0)),
                "transition_penalty": float(penalty_sums.get("transition_penalty", 0.0)),
                "unsupported_penalty": float(penalty_sums.get("unsupported_penalty", 0.0)),
                "duplicate_penalty": float(penalty_sums.get("duplicate_penalty", 0.0)),
                "direction_penalty": float(penalty_sums.get("direction_penalty", 0.0)),
                "temporal_penalty": float(penalty_sums.get("temporal_penalty", 0.0)),
                "missing_layer_penalty": float(self.params.get("missing_layer_penalty", 0.50)) * len(missing_layers),
                "unsupported_count": len(unsupported_edges),
            },
        }
        return chain