
import uuid
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from .baseline_extractor import BaselineCausalExtractor
from .beam_search import BeamSearchChainBuilder
from .jointlk_edge_scorer import CausalJointLKEdgeScorer
from .neo4j_accessor import InstanceKGAccessor
from .prior import CausalPrior, load_prior_config
from .schemas import CausalEdge, CausalNode, ExtractionResult


class CausalJointLKService:
    """End-to-end causal chain extraction on Instance-KG."""

    def __init__(
        self,
        graph: Any,
        evidence_store: Any,
        prior_config_path: str = "configs/causal_prior.yaml",
        jointlk_checkpoint_path: Optional[str] = None,
    ):
        self.graph = graph
        self.evidence_store = evidence_store
        self.prior = CausalPrior(load_prior_config(prior_config_path))
        self.accessor = InstanceKGAccessor(graph)
        self.baseline = BaselineCausalExtractor(self.prior)
        self.chain_builder = BeamSearchChainBuilder(self.prior)
        self.neural_scorer = None
        if jointlk_checkpoint_path:
            self.neural_scorer = CausalJointLKEdgeScorer(
                checkpoint_path=jointlk_checkpoint_path,
                prior_config_path=prior_config_path,
            )

    def extract(
        self,
        query: Optional[str] = None,
        target_text: Optional[str] = None,
        target_node_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        kg_scope: str = "instance",
        kg_id: Optional[str] = None,
        mode: str = "jointlk",
        k_hop: int = 2,
        top_k: int = 5,
        persist: bool = False,
    ) -> ExtractionResult:
        if not target_node_id and not target_text and not query:
            raise ValueError("one of target_node_id / target_text / query must be provided")

        if target_node_id:
            seed_nodes = [CausalNode(node_id=target_node_id, text=target_text or target_node_id)]
        else:
            seed_nodes = self.accessor.find_seed_nodes(
                query_text=target_text or query or "",
                doc_id=doc_id,
                kg_scope=kg_scope,
                kg_id=kg_id,
                limit=10,
            )
        seed_node_ids = [n.node_id for n in seed_nodes]
        subgraph = self.accessor.get_k_hop_subgraph(
            seed_node_ids=seed_node_ids,
            prior=self.prior,
            k_hop=k_hop,
            doc_id=doc_id,
            kg_scope=kg_scope,
            kg_id=kg_id,
        )
        nodes: List[CausalNode] = subgraph["nodes"]
        edges: List[CausalEdge] = self.accessor.hydrate_edge_evidence(self.evidence_store, subgraph["edges"])

        evidence_by_edge_id = self._build_evidence_map(edges)

        if mode == "jointlk" and self.neural_scorer is not None:
            scored_edges = self.neural_scorer.score(
                query=query or target_text,
                nodes=nodes,
                edges=edges,
                evidence_by_edge_id=evidence_by_edge_id,
                doc_title=doc_id,
            )
            # 神经分数出来之后，仍走 evidence gate 做二次校验
            scored_edges = self.baseline.score_edges(scored_edges, evidence_by_edge_id)
        else:
            scored_edges = self.baseline.score_edges(edges, evidence_by_edge_id)

        if target_node_id:
            beam_seeds = self._find_root_like_seeds(nodes, scored_edges)
        else:
            beam_seeds = seed_node_ids or self._find_root_like_seeds(nodes, scored_edges)

        chains = self.chain_builder.build(
            edges=scored_edges,
            seed_node_ids=beam_seeds,
            target_node_id=target_node_id,
            top_k=top_k,
            max_hops=k_hop + 2,
        )

        result = ExtractionResult(
            mode=mode if (mode != "jointlk" or self.neural_scorer is not None) else "baseline",
            query=query,
            doc_id=doc_id,
            target_node_id=target_node_id,
            chains=chains,
            subgraph_nodes=nodes,
            subgraph_edges=scored_edges,
            meta={
                "seed_node_ids": seed_node_ids,
                "beam_seed_ids": beam_seeds,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "num_supported_edges": sum(1 for e in scored_edges if e.supported),
                "relation_hist": dict(Counter(e.relation for e in scored_edges)),
            },
        )

        if persist:
            self._persist_chains(result)

        return result

    def _build_evidence_map(self, edges: Sequence[CausalEdge]) -> Dict[str, List[Dict[str, Any]]]:
        evidence_by_edge_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            if edge.evidence_unit_id:
                unit = self.evidence_store.get_unit(edge.evidence_unit_id)
                if unit is not None:
                    evidence_by_edge_id[edge.edge_id].append(
                        {
                            "unit_id": unit.unit_id,
                            "content": unit.content,
                        }
                    )
            elif edge.evidence_text:
                evidence_by_edge_id[edge.edge_id].append(
                    {
                        "unit_id": None,
                        "content": edge.evidence_text,
                    }
                )
        return evidence_by_edge_id

    @staticmethod
    def _find_root_like_seeds(nodes: Sequence[CausalNode], edges: Sequence[CausalEdge]) -> List[str]:
        indeg = Counter()
        for edge in edges:
            indeg[edge.target_id] += 1
        root_layers = {"ROOT", "CAUSE", "FACTOR", "CONDITION", "ACTION"}
        seeds = [n.node_id for n in nodes if indeg[n.node_id] == 0 or n.layer in root_layers]
        if not seeds and nodes:
            seeds = [nodes[0].node_id]
        return seeds

    def _persist_chains(self, result: ExtractionResult) -> None:
        for chain in result.chains:
            self.evidence_store.upsert_causal_chain(
                chain_id=chain.chain_id,
                file_name=result.doc_id,
                parent_evidence_id=None,
                chain_text=" -> ".join(chain.nodes),
                chain_json=chain.to_dict(),
            )
            edge_rows = []
            for seq, edge in enumerate(chain.edges):
                edge_rows.append(
                    {
                        "edge_id": edge.edge_id or str(uuid.uuid4()),
                        "seq": seq,
                        "source_node_id": edge.source_id,
                        "source_layer": edge.source_layer,
                        "target_node_id": edge.target_id,
                        "target_layer": edge.target_layer,
                        "evidence_unit_id": edge.evidence_unit_id,
                        "evidence_start": edge.meta.get("source_span", [None, None])[0] if edge.meta else None,
                        "evidence_end": edge.meta.get("target_span", [None, None])[1] if edge.meta else None,
                        "meta": {
                            "relation": edge.relation,
                            "score": edge.score,
                            "support_score": edge.support_score,
                            "supported": edge.supported,
                        },
                    }
                )
            self.evidence_store.upsert_causal_chain_edges(chain.chain_id, edge_rows)
