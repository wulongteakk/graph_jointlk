from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .schemas import CandidateBranch, CandidateChain, CausalEdge, DecodedAccidentResult


class ConsoleTracer:
    def log_edge_scores(self, edges: Sequence[CausalEdge], top_k: int = 10) -> Dict[str, Any]:
        ranked = sorted(edges, key=lambda e: e.score, reverse=True)[:top_k]
        rows: List[Dict[str, Any]] = []
        for edge in ranked:
            rows.append(
                {
                    "edge_id": edge.edge_id,
                    "source": edge.source_text,
                    "target": edge.target_text,
                    "score": round(float(edge.score), 4),
                    "p_causal": round(float(edge.p_causal or 0.0), 4),
                    "p_enable": round(float(edge.p_enable or 0.0), 4),
                    "p_dir": round(float(edge.p_dir or 0.0), 4),
                    "p_temporal_before": round(float(edge.p_temporal_before or 0.0), 4),
                    "p_evidence": round(float(edge.p_evidence or 0.0), 4),
                    "p_node_first": round(float(edge.p_node_first or 0.0), 4),
                }
            )
        return {"edge_topk": rows}

    def log_beam_chains(self, chains: Sequence[CandidateChain], top_k: int = 5) -> Dict[str, Any]:
        ranked = sorted(chains, key=lambda c: c.score, reverse=True)[:top_k]
        rows = []
        for chain in ranked:
            rows.append(
                {
                    "chain_id": chain.chain_id,
                    "score": round(float(chain.score), 4),
                    "nodes": chain.nodes,
                    "missing_layers": chain.missing_layers,
                    "unsupported_edges": chain.unsupported_edges,
                    "meta": chain.__dict__.get("meta", {}),
                }
            )
        return {"beam_topk": rows}

    def log_branch_decision(self, branches: Sequence[CandidateBranch], decision: Any, top_k: int = 5) -> Dict[str, Any]:
        ranked = sorted(branches, key=lambda b: b.score, reverse=True)[:top_k]
        return {
            "branch_ranking": [
                {
                    "branch_id": b.branch_id,
                    "score": round(float(b.score), 4),
                    "p_branch_first": round(float(b.p_branch_first), 4),
                    "severity_score": round(float(b.severity_score), 4),
                    "decision_gap": round(float(b.decision_gap), 4),
                    "rule_hits": b.rule_hits,
                }
                for b in ranked
            ],
            "branch_decision": {
                "selected_branch_id": getattr(decision, "selected_branch_id", None),
                "decision_gap": getattr(decision, "decision_gap", 0.0),
                "needs_severity_fallback": getattr(decision, "needs_severity_fallback", False),
                "reason": getattr(decision, "reason", ""),
            },
        }

    def log_decode(self, decoded_result: DecodedAccidentResult) -> Dict[str, Any]:
        if decoded_result is None:
            return {"decode_trace": {}}
        return {
            "decode_trace": {
                "basic_type": decoded_result.basic_type,
                "injury_severity": decoded_result.injury_severity,
                "industry_type": decoded_result.industry_type,
                "basic_code": decoded_result.basic_code,
                "injury_code": decoded_result.injury_code,
                "industry_code": decoded_result.industry_code,
                "decode_confidence": decoded_result.decode_confidence,
                "decode_rule_hits": decoded_result.decode_rule_hits,
            }
        }