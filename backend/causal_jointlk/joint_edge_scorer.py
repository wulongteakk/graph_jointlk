
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoTokenizer

from modeling.causal_jointlk_io import (
    batchify_examples,
    build_example_text,
    build_node_type_to_id,
    build_relation_to_id,
    load_prior_config,
)
from modeling.modeling_causal_jointlk import CausalJointLKModel

from .prior import CausalPrior
from .schemas import CausalEdge, CausalNode


class CausalJointLKEdgeScorer:
    def __init__(
        self,
        checkpoint_path: str,
        prior_config_path: str = "configs/causal_prior.yaml",
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.prior_config = load_prior_config(prior_config_path)
        self.prior = CausalPrior(self.prior_config)
        self.beam_params = self.prior.as_beam_params()
        self.score_weights = {
            "w_causal": float(self.beam_params.get("w_causal", 0.45)),
            "w_enable": float(self.beam_params.get("w_enable", 0.15)),
            "w_dir": float(self.beam_params.get("w_dir", 0.10)),
            "w_temp": float(self.beam_params.get("w_temp", 0.10)),
            "w_evidence": float(self.beam_params.get("w_evidence", 0.10)),
            "w_first": float(self.beam_params.get("w_first", 0.10)),
        }
        self.relation_to_id = build_relation_to_id(self.prior_config)
        self.node_type_to_id = build_node_type_to_id(self.prior_config)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = self.checkpoint.get("args", {})
        self.model_name = args.get("model_name", "roberta-large")
        self.best_threshold = float(self.checkpoint.get("best_threshold", 0.5))

        # 推理阶段优先使用 checkpoint 里保存的映射，避免训练后 prior 配置变更导致 id 漂移。
        self.relation_to_id = dict(self.checkpoint.get("relation_to_id") or self.relation_to_id)
        self.node_type_to_id = dict(self.checkpoint.get("node_type_to_id") or self.node_type_to_id)

        tokenizer_dir = Path(checkpoint_path).parent / "tokenizer"
        if tokenizer_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = CausalJointLKModel(
            model_name=self.model_name,
            num_relations=len(self.relation_to_id),
            num_node_types=len(self.node_type_to_id),
            hidden_size=int(args.get("hidden_size", 256)),
            num_gnn_layers=int(args.get("num_gnn_layers", 3)),
            dropout=float(args.get("dropout", 0.2)),
            freeze_lm=bool(args.get("freeze_lm", False)),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["state_dict"], strict=True)
        self.model.eval()

        self.max_text_length = int(args.get("max_text_length", 320))
        self.max_node_text_length = int(args.get("max_node_text_length", 24))

    @torch.no_grad()
    def score(
        self,
        query: Optional[str],
        nodes: Sequence[CausalNode],
        edges: Sequence[CausalEdge],
        evidence_by_edge_id: Dict[str, List[Dict[str, Any]]],
        doc_title: Optional[str] = None,
    ) -> List[CausalEdge]:
        if not edges:
            return []

        node_map = {n.node_id: n for n in nodes}
        node_ids = list(node_map.keys())
        node_texts = [node_map[nid].text for nid in node_ids]
        node_layers = [node_map[nid].layer for nid in node_ids]
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        examples = []
        valid_edges: List[CausalEdge] = []
        for edge in edges:
            if edge.source_id not in node_id_to_idx or edge.target_id not in node_id_to_idx:
                continue

            valid_graph_edges = [
                e
                for e in edges
                if e.source_id in node_id_to_idx and e.target_id in node_id_to_idx
            ]
            candidate_units = evidence_by_edge_id.get(edge.edge_id) or []
            if edge.evidence_text:
                candidate_units = [{"unit_id": edge.evidence_unit_id, "content": edge.evidence_text}] + candidate_units

            evidence_texts = [x.get("content", "") for x in candidate_units][:3]
            edge_rel = self.prior.normalize_relation(edge.relation, " ".join(evidence_texts))
            text = build_example_text(
                query=query,
                source_text=edge.source_text,
                relation_text=edge_rel,
                target_text=edge.target_text,
                evidence_texts=evidence_texts,
                doc_title=doc_title,
            )
            examples.append(
                {
                    "sample_id": edge.edge_id,
                    "text": text,
                    "source_text": edge.source_text,
                    "target_text": edge.target_text,
                    "candidate_relation": edge_rel,
                    "node_texts": node_texts,
                    "node_layer_types": node_layers,
                    "node_scores": [0.0] * len(node_texts),
                    "edge_index": [
                        [node_id_to_idx[e.source_id] for e in valid_graph_edges],
                        [node_id_to_idx[e.target_id] for e in valid_graph_edges],
                    ],
                    "edge_types": [
                        self.prior.normalize_relation(e.relation, e.evidence_text)
                        for e in valid_graph_edges
                    ],
                    "source_idx": node_id_to_idx[edge.source_id],
                    "target_idx": node_id_to_idx[edge.target_id],
                    "label": 0,
                }
            )
            valid_edges.append(edge)

        if not examples:
            return list(edges)

        batch = batchify_examples(
            examples=examples,
            tokenizer=self.tokenizer,
            relation_to_id=self.relation_to_id,
            node_type_to_id=self.node_type_to_id,
            max_text_length=self.max_text_length,
            max_node_text_length=self.max_node_text_length,
            device=self.device,
        )
        outputs = self.model(batch)
        probs = outputs["support_prob"].detach().cpu().tolist()
        enable_probs = outputs["enable_prob"].detach().cpu().tolist()
        dir_probs = outputs["dir_prob"].detach().cpu().tolist()
        temporal_probs = outputs["temporal_prob"].detach().cpu().tolist()
        node_first_probs = outputs["node_first_prob"].detach().cpu().tolist()

        out: List[CausalEdge] = []
        for edge, p, pe, pd, pt, pn in zip(valid_edges, probs, enable_probs, dir_probs, temporal_probs,
                                           node_first_probs):
            p_evidence = float(edge.p_evidence or edge.support_score or 0.0)
            fused_score = (
                    self.score_weights["w_causal"] * float(p)
                    + self.score_weights["w_enable"] * float(pe)
                    + self.score_weights["w_dir"] * float(pd)
                    + self.score_weights["w_temp"] * float(pt)
                    + self.score_weights["w_evidence"] * p_evidence
                    + self.score_weights["w_first"] * float(pn)
            )
            edge.score = fused_score
            edge.support_score = float(p)
            edge.supported = bool(p >= self.best_threshold)
            edge.p_causal = float(p)
            edge.p_enable = float(pe)
            edge.p_dir = float(pd)
            edge.p_temporal_before = float(pt)
            edge.p_node_first = float(pn)
            edge.p_evidence = p_evidence
            edge.meta = dict(edge.meta or {})
            edge.meta["jointlk_trace"] = {
                "p_causal": edge.p_causal,
                "p_enable": edge.p_enable,
                "p_dir": edge.p_dir,
                "p_temporal_before": edge.p_temporal_before,
                "p_node_first": edge.p_node_first,
                "p_evidence": edge.p_evidence,
                "score_weights": self.score_weights,
                "best_threshold": self.best_threshold,
            }
            source_node = node_map.get(edge.source_id)
            if source_node is not None:
                source_node.p_node_first = max(float(source_node.p_node_first or 0.0), float(edge.p_node_first or 0.0))
            target_node = node_map.get(edge.target_id)
            if target_node is not None:
                target_node.p_node_first = max(float(target_node.p_node_first or 0.0), float(edge.p_node_first or 0.0))
            out.append(edge)
        return out
