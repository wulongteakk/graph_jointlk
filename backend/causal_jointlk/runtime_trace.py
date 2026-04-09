from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .schemas import CandidateBranch, CandidateChain, CausalEdge, DecodedAccidentResult


class ConsoleTracer:
    @staticmethod
    def _metric(metrics: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            if key in metrics and metrics[key] is not None:
                try:
                    return float(metrics[key])
                except Exception:
                    continue
        return float(default)

    def log_stage(self, stage: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"stage": stage, "payload": payload}

    def log_train_epoch(self, metrics: Dict[str, Any], epoch: int | None = None) -> Dict[str, Any]:
        ranking_by_doc = metrics.get("ranking_by_doc", {}) or {}
        payload = {
            "epoch": epoch,
            "loss": self._metric(metrics, "train_loss", "loss"),
            "causal_loss": self._metric(metrics, "train_causal_loss", "causal_loss"),
            "relation_loss": self._metric(metrics, "train_relation_loss", "relation_loss"),
            "aux_loss": self._metric(metrics, "train_aux_loss", "aux_loss"),
            "cf_loss": self._metric(metrics, "train_cf_loss", "cf_loss"),
            "edge_f1": self._metric(metrics, "edge_f1"),
            "enable_f1": self._metric(metrics, "enable_f1"),
            "dir_f1": self._metric(metrics, "dir_f1"),
            "temp_f1": self._metric(metrics, "temp_f1"),
            "src_first_f1": self._metric(metrics, "src_first_f1"),
            "dst_first_f1": self._metric(metrics, "dst_first_f1"),
            "rel_micro_f1": self._metric(metrics, "rel_micro_f1"),
            "rel_macro_f1": self._metric(metrics, "rel_macro_f1"),
            "mrr": float(ranking_by_doc.get("mrr", metrics.get("mrr", 0.0))),
            "hits@1": float(ranking_by_doc.get("hits@1", metrics.get("hits@1", 0.0))),
            "hits@3": float(ranking_by_doc.get("hits@3", metrics.get("hits@3", 0.0))),
            "hits@5": float(ranking_by_doc.get("hits@5", metrics.get("hits@5", 0.0))),
            "ndcg@5": float(ranking_by_doc.get("ndcg@5", metrics.get("ndcg@5", 0.0))),
            "joint_score": self._metric(metrics, "joint_score"),
            "best_threshold": self._metric(metrics, "best_threshold", default=0.5),
        }
        return {"train_epoch": payload}

    def log_train_epoch_console(self, metrics: Dict[str, Any], epoch: int | None = None) -> None:
        payload = self.log_train_epoch(metrics=metrics, epoch=epoch).get("train_epoch", {})
        print("[JointLK][epoch-dashboard]", payload)

    def log_jointlk_input_preview(self, rows: Sequence[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        preview: List[Dict[str, Any]] = []
        for row in list(rows)[:max(int(top_k), 0)]:
            preview.append(
                {
                    "sample_id": row.get("sample_id"),
                    "doc_id": row.get("doc_id"),
                    "source_text": row.get("source_text"),
                    "candidate_relation": row.get("candidate_relation") or row.get("relation_type"),
                    "target_text": row.get("target_text"),
                    "causal_labels": row.get("causal_labels", row.get("label")),
                    "enable_labels": row.get("enable_labels"),
                    "dir_labels": row.get("dir_labels"),
                    "temp_labels": row.get("temp_labels"),
                    "src_first_labels": row.get("src_first_labels"),
                    "dst_first_labels": row.get("dst_first_labels"),
                    "twin_group_id": row.get("twin_group_id"),
                    "cf_role": row.get("cf_role"),
                }
            )
        return {"jointlk_input_preview": preview}

    def log_jointlk_input_preview_console(self, rows: Sequence[Dict[str, Any]], top_k: int = 3) -> None:
        payload = self.log_jointlk_input_preview(rows, top_k=top_k)
        print("[JointLK][input-preview]", payload.get("jointlk_input_preview", []))

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

    def log_edge_scores_console(self, edges: Sequence[CausalEdge], top_k: int = 10) -> None:
        payload = self.log_edge_scores(edges, top_k=top_k)
        print("[JointLK][edge-topk]", payload.get("edge_topk", []))

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

    def log_beam_console(self, chains: Sequence[CandidateChain], top_k: int = 5) -> None:
        payload = self.log_beam_chains(chains, top_k=top_k)
        print("[JointLK][beam-topk]", payload.get("beam_topk", []))

    def log_branch_decision(
        self,
        branches: Sequence[CandidateBranch],
        decision: Any,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        ranked = sorted(
            branches,
            key=lambda b: (float(getattr(b, "p_branch_first", 0.0)), float(getattr(b, "score", 0.0))),
            reverse=True,
        )[:top_k]
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
                "ranking": getattr(decision, "ranking", []),
                "trace": getattr(decision, "trace", {}),
            },
        }

    def log_branch_console(self, branches: Sequence[CandidateBranch], decision: Any, top_k: int = 5) -> None:
        payload = self.log_branch_decision(branches, decision, top_k=top_k)
        print("[JointLK][branch-ranking]", payload.get("branch_ranking", []))
        print("[JointLK][branch-decision]", payload.get("branch_decision", {}))

    def log_severity_fallback(self, severity_trace: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        ranking = severity_trace.get("ranking", []) if isinstance(severity_trace, dict) else []
        rows = []
        for row in ranking[:top_k]:
            rows.append(
                {
                    "branch_id": row.get("branch_id"),
                    "severity_score": round(float(row.get("severity_score", 0.0)), 4),
                    "death_count": row.get("death_count", 0),
                    "serious_injury_count": row.get("serious_injury_count", 0),
                    "light_injury_count": row.get("light_injury_count", 0),
                    "energy_level": row.get("energy_level", 0.0),
                    "toxicity_level": row.get("toxicity_level", 0.0),
                }
            )
        return {
            "severity_trace": {
                "triggered": bool(severity_trace.get("triggered", False)) if isinstance(severity_trace, dict) else False,
                "reason": severity_trace.get("reason", "") if isinstance(severity_trace, dict) else "",
                "ranking": rows,
            }
        }

    def log_severity_console(self, severity_trace: Dict[str, Any], top_k: int = 5) -> None:
        payload = self.log_severity_fallback(severity_trace, top_k=top_k)
        print("[JointLK][severity-trace]", payload.get("severity_trace", {}))

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

    def log_decode_console(self, decoded_result: DecodedAccidentResult) -> None:
        payload = self.log_decode(decoded_result)
        print("[JointLK][decode-trace]", payload.get("decode_trace", {}))
class RuntimeTracer(ConsoleTracer):
    """Runtime tracer with switchable output and unified key aliases."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    def log_edge_scores(self, edges: Sequence[CausalEdge], top_k: int = 10) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return super().log_edge_scores(edges, top_k=top_k)

    def log_beam_chains(self, chains: Sequence[CandidateChain], top_k: int = 5) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return super().log_beam_chains(chains, top_k=top_k)

    def log_branch_decision(self, branches: Sequence[CandidateBranch], decision: Any, top_k: int = 5) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return super().log_branch_decision(branches, decision, top_k=top_k)

    def log_severity_fallback(self, severity_trace: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return super().log_severity_fallback(severity_trace, top_k=top_k)

    def log_train_epoch(self, metrics: Dict[str, Any], epoch: int | None = None) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return super().log_train_epoch(metrics=metrics, epoch=epoch)

    def collect(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "edge_topk": kwargs.get("edge_topk", []),
            "beam_topk": kwargs.get("beam_topk", []),
            "branch_ranking": kwargs.get("branch_ranking", []),
            "severity_trace": kwargs.get("severity_trace", {}),
            "decode_trace": kwargs.get("decode_trace", {}),
        }