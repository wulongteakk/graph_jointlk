from __future__ import annotations

import os
import uuid
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from .baseline_extractor import BaselineCausalExtractor
from .beam_search import BeamSearchChainBuilder
from .branch_builder import BranchBuilder
from .branch_rule_decoder import BranchRuleDecoder
from .candidate_generator import CandidateGenerator, CandidateGeneratorConfig
from .gb6441_decoder import GB6441Decoder
from .gbt4754_lookup import GBT4754Lookup
from .industry_classifier import IndustryClassifier
from .joint_edge_scorer import CausalJointLKEdgeScorer
from .neo4j_accessor import Neo4jAccessor
from .node_prior_builder import NodePriorBuilder
from .edge_postprocessor import EdgePostProcessor
from .prior import CausalPrior, load_prior_config
from .runtime_trace import RuntimeTracer
from .schemas import CausalEdge, CausalNode, ExtractionResult
from .severity_ranker import SeverityRanker


class CausalJointLKService:
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
        self.accessor = Neo4jAccessor(graph)
        self.baseline = BaselineCausalExtractor(self.prior)
        self.chain_builder = BeamSearchChainBuilder(self.prior)
        branch_cfg = dict((self.prior.config.get("branch_decision") or {}))
        severity_cfg = dict((self.prior.config.get("severity") or {}))
        trace_cfg = dict((self.prior.config.get("trace") or {}))

        self.branch_builder = BranchBuilder()
        self.branch_decoder = BranchRuleDecoder(
            prior=self.prior,
            gap_threshold=float(branch_cfg.get("gap_threshold", 0.15)),
            temporal_violation_penalty=float(branch_cfg.get("temporal_violation_penalty", 0.8)),
            cycle_penalty=float(branch_cfg.get("cycle_penalty", 1.2)),
            downstream_enable_penalty=float(branch_cfg.get("downstream_enable_penalty", 0.6)),
        )
        self.severity_ranker = SeverityRanker(
            death_weight=float(severity_cfg.get("death_weight", 5.0)),
            serious_injury_weight=float(severity_cfg.get("serious_injury_weight", 3.0)),
            light_injury_weight=float(severity_cfg.get("light_injury_weight", 1.0)),
            energy_weight=float(severity_cfg.get("energy_weight", 1.0)),
            toxicity_weight=float(severity_cfg.get("toxicity_weight", 1.0)),
        )
        self.node_prior_builder = NodePriorBuilder()
        self.gbt4754_lookup = GBT4754Lookup()
        self.industry_classifier = IndustryClassifier(self.gbt4754_lookup)
        self.gb6441_decoder = GB6441Decoder()
        self.postprocessor = EdgePostProcessor(self.prior)
        self.trace_defaults = {
            "enabled": bool(trace_cfg.get("enabled", True)),
            "top_edges": int(trace_cfg.get("top_edges", 10)),
            "top_chains": int(trace_cfg.get("top_chains", 5)),
            "top_branches": int(trace_cfg.get("top_branches", 5)),
        }
        self.tracer = RuntimeTracer(enabled=self.trace_defaults["enabled"])
        self.neural_scorer = None

        cfg_path = os.getenv("AUTO_PSEUDO_LABEL_CANDIDATE_GENERATOR_CONFIG", "configs/causal_candidate_generator.yaml")
        cfg_raw = {}
        if cfg_path and os.path.exists(cfg_path):
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg_raw = yaml.safe_load(fh) or {}
        self.candidate_generator = CandidateGenerator(self.accessor, CandidateGeneratorConfig(**cfg_raw))
        self.persist_implicit_candidates = str(os.getenv("JOINTLK_PERSIST_IMPLICIT_CANDIDATES", "true")).strip().lower() not in {"0", "false", "no", "off"}

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
        trace:Optional[bool] = None,
        trace_top_edges: Optional[int]  = None,
        trace_top_chains: Optional[int]  = None,
        trace_top_branches: Optional[int]  = None,
    ) -> ExtractionResult:
        if not target_node_id and not target_text and not query:
            raise ValueError("one of target_node_id / target_text / query must be provided")
        self.tracer.log_stage_console(
            "service-request",
            {
                "mode": mode,
                "doc_id": doc_id,
                "query": query,
                "target_text": target_text,
                "target_node_id": target_node_id,
                "k_hop": k_hop,
                "top_k": top_k,
            },
        )

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
        node_prior_rows: List[Dict[str, Any]] = []
        node_prior_map: Dict[str, Dict[str, Any]] = {}

        if mode.startswith("jointlk") and self.neural_scorer is not None:
            scored_edges = self.neural_scorer.score(
                query=query or target_text,
                nodes=nodes,
                edges=edges,
                evidence_by_edge_id=evidence_by_edge_id,
                doc_title=doc_id,
                node_prior_map=None,
            )
            node_prior_rows = self.node_prior_builder.build(nodes, scored_edges)
            node_prior_map = {row.get("node_id"): row for row in node_prior_rows if row.get("node_id")}
            scored_edges = self.neural_scorer.score(
                query=query or target_text,
                nodes=nodes,
                edges=scored_edges,
                evidence_by_edge_id=evidence_by_edge_id,
                doc_title=doc_id,
                node_prior_map=node_prior_map,
            )
            if "+gate" in mode:
                scored_edges = self.postprocessor.apply(scored_edges, evidence_by_edge_id)
        else:
            scored_edges = self.baseline.score_edges(edges, evidence_by_edge_id) if mode != "baseline" else list(edges)
            node_prior_rows = self.node_prior_builder.build(nodes, scored_edges)
            node_prior_map = {row.get("node_id"): row for row in node_prior_rows if row.get("node_id")}

        beam_seeds = self._find_root_like_seeds(nodes, scored_edges)
        chains = self.chain_builder.build(
            edges=scored_edges,
            seed_node_ids=beam_seeds,
            target_node_id=target_node_id,
            top_k=top_k,
            max_hops=k_hop + 2,
        )

        self.tracer.log_stage_console(
            "causal-chain-summary",
            {
                "num_chains": len(chains),
                "top_chain_ids": [c.chain_id for c in sorted(chains, key=lambda x: x.score, reverse=True)[:min(3, len(chains))]],
            },
        )
        self.tracer.log_stage_console(
            "causal-chain-candidates",
            {
                "chains": [
                    {
                        "chain_id": c.chain_id,
                        "score": float(c.score),
                        "nodes": list(c.nodes),
                        "edge_ids": [e.edge_id for e in c.edges],
                    }
                    for c in sorted(chains, key=lambda x: x.score, reverse=True)[: min(top_k, len(chains))]
                ]
            },
        )

        enable_branch = mode.endswith("+branch") or mode in {"jointlk", "jointlk+gate", "jointlk+gate+branch", "baseline+gate+branch"}
        if enable_branch:
            branches = self.branch_builder.build(
                nodes=nodes,
                edges=scored_edges,
                chains=chains,
                node_prior_map=node_prior_map,
                evidence_store=self.evidence_store,
            )
            decision = self.branch_decoder.decide(branches)
        else:
            branches = []
            decision = None
        severity_trace: Dict[str, Any] = {"triggered": False, "reason": "not_triggered", "ranking": []}
        if decision is not None and decision.needs_severity_fallback:
            branches, severity_obj = self.severity_ranker.rerank_with_trace(
                branches,
                triggered=True,
                reason=f"{decision.reason}:decision_gap<{self.branch_decoder.gap_threshold}",
            )
            severity_trace = severity_obj.to_dict()
            decision = self.branch_decoder.decide(branches)
            decision.trace["severity_fallback"] = severity_trace

        selected = next((b for b in branches if decision and b.branch_id == decision.selected_branch_id), None)
        self.tracer.log_stage_console(
            "induced-branch-decision",
            {
                "selected_branch_id": decision.selected_branch_id if decision else None,
                "decision_gap": decision.decision_gap if decision else 0.0,
                "needs_severity_fallback": decision.needs_severity_fallback if decision else False,
                "reason": decision.reason if decision else "branch_disabled",
            },
        )

        evidence_units = []
        for branch in branches:
            for unit_id in branch.evidence_unit_ids:
                unit = self.evidence_store.get_unit(unit_id) if unit_id else None
                if unit is not None:
                    evidence_units.append(unit)

        doc_meta = {
            "doc_id": doc_id,
            "doc_title": doc_id,
            "title": query or target_text or doc_id,
        }
        industry_prediction = self.industry_classifier.classify(
            doc_meta=doc_meta,
            selected_branch=selected,
            all_branches=branches,
            evidence_units=evidence_units,
            module_candidates=(selected.basic_type_candidates if selected else []),
        )

        severity_signals = {}
        if selected:
            severity_signals = {
                "death_count": selected.meta.get("death_count", 0),
                "serious_injury_count": selected.meta.get("serious_injury_count", 0),
                "light_injury_count": selected.meta.get("light_injury_count", 0),
            }

        decoded = self.gb6441_decoder.decode(
            {
                "winner_branch": selected.branch_id if selected else None,
                "selected_branch": selected,
                "all_branches": [b.branch_id for b in branches],
                "doc_meta": doc_meta,
                "module_candidates": selected.basic_type_candidates if selected else [],
                "severity_signals": severity_signals,
                "industry_prediction": industry_prediction,
            }
        )
        self.tracer.log_stage_console(
            "final-code-classification",
            {
                "basic_type": decoded.basic_type if decoded else None,
                "basic_code": decoded.basic_code if decoded else None,
                "injury_severity": decoded.injury_severity if decoded else None,
                "injury_code": decoded.injury_code if decoded else None,
                "industry_type": decoded.industry_type if decoded else None,
                "industry_code": decoded.industry_code if decoded else None,
                "decode_confidence": decoded.decode_confidence if decoded else None,
            },
        )

        use_trace = self.trace_defaults["enabled"] if trace is None else bool(trace)
        edge_top_k = int(trace_top_edges if trace_top_edges is not None else self.trace_defaults["top_edges"])
        chain_top_k = int(trace_top_chains if trace_top_chains is not None else self.trace_defaults["top_chains"])
        branch_top_k = int(trace_top_branches if trace_top_branches is not None else self.trace_defaults["top_branches"])

        trace_payload: Dict[str, Any] = {}
        if use_trace:
            self.tracer.enabled = True
            edge_trace = self.tracer.log_edge_scores(scored_edges, top_k=edge_top_k)
            beam_trace = self.tracer.log_beam_chains(chains, top_k=chain_top_k)
            branch_trace = self.tracer.log_branch_decision(branches, decision, top_k=branch_top_k)
            severity_trace_payload = self.tracer.log_severity_fallback(severity_trace, top_k=branch_top_k)
            decode_trace = self.tracer.log_decode(decoded)
            trace_payload = {
                "edge_topk": edge_trace.get("edge_topk", []),
                "beam_topk": beam_trace.get("beam_topk", []),
                "branch_ranking": branch_trace.get("branch_ranking", []),
                "severity_trace": severity_trace_payload.get("severity_trace", {}),
                "decode_trace": decode_trace.get("decode_trace", {}),
            }
            self.tracer.log_edge_scores_console(scored_edges, top_k=edge_top_k)
            self.tracer.log_beam_console(chains, top_k=chain_top_k)
            self.tracer.log_branch_console(branches, decision, top_k=branch_top_k)
            self.tracer.log_severity_console(severity_trace, top_k=branch_top_k)
            self.tracer.log_decode_console(decoded)
            self.tracer.log_stage_console(
                "service-summary",
                {
                    "num_nodes": len(nodes),
                    "num_edges": len(edges),
                    "num_scored_edges": len(scored_edges),
                    "num_chains": len(chains),
                    "num_branches": len(branches),
                    "selected_branch_id": decision.selected_branch_id if decision else None,
                    "needs_severity_fallback": bool(decision.needs_severity_fallback) if decision else False,
                    "decoded_full": {
                        "basic_type": decoded.basic_type if decoded else None,
                        "injury_code": decoded.injury_code if decoded else None,
                        "industry_code": decoded.industry_code if decoded else None,
                        "decode_confidence": decoded.decode_confidence if decoded else None,
                    },
                },
            )

        result = ExtractionResult(
            mode=mode,
            query=query,
            doc_id=doc_id,
            target_node_id=target_node_id,
            chains=chains,
            subgraph_nodes=nodes,
            subgraph_edges=scored_edges,
            branches=branches,
            branch_decision=decision,
            selected_branch=selected,
            rejected_branches=[b for b in branches if selected and b.branch_id != selected.branch_id],
            decoded_result=decoded,
            meta={
                "seed_node_ids": seed_node_ids,
                "beam_seed_ids": beam_seeds,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "num_supported_edges": sum(1 for e in scored_edges if e.supported),
                "relation_hist": dict(Counter(e.relation for e in scored_edges)),
                "candidate_branch_table_size": len(branches),
                "candidate_node_prior_table": node_prior_rows,
                "industry_prediction": industry_prediction.meta,
                "severity_trace": severity_trace,
                "trace": trace_payload,
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
                    evidence_by_edge_id[edge.edge_id].append({"unit_id": unit.unit_id, "content": unit.content})
            elif edge.evidence_text:
                evidence_by_edge_id[edge.edge_id].append({"unit_id": None, "content": edge.evidence_text})
        return evidence_by_edge_id

    @staticmethod
    def _find_root_like_seeds(nodes: Sequence[CausalNode], edges: Sequence[CausalEdge]) -> List[str]:
        indeg = Counter()
        for edge in edges:
            indeg[edge.target_id] += 1
        root_layers = {"ROOT", "CAUSE", "FACTOR", "CONDITION", "ACTION", "STATE"}
        seeds = [n.node_id for n in nodes if indeg[n.node_id] == 0 or n.layer in root_layers]
        return seeds or ([nodes[0].node_id] if nodes else [])

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
