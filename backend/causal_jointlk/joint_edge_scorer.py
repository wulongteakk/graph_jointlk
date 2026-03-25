
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
        self.relation_to_id = build_relation_to_id(self.prior_config)
        self.node_type_to_id = build_node_type_to_id(self.prior_config)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = self.checkpoint.get("args", {})
        self.model_name = args.get("model_name", "roberta-large")

        tokenizer_dir = Path(checkpoint_path).parent / "tokenizer"
        if tokenizer_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = CausalJointLKModel(
            model_name=self.model_name,
            num_relations=len(self.checkpoint["relation_to_id"]),
            num_node_types=len(self.checkpoint["node_type_to_id"]),
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
                        [node_id_to_idx[e.source_id] for e in edges if e.source_id in node_id_to_idx and e.target_id in node_id_to_idx],
                        [node_id_to_idx[e.target_id] for e in edges if e.source_id in node_id_to_idx and e.target_id in node_id_to_idx],
                    ],
                    "edge_types": [self.prior.normalize_relation(e.relation, e.evidence_text) for e in edges],
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
            edge.score = float(p)
            edge.support_score = float(p)
            edge.supported = bool(p >= 0.5)
            edge.p_causal = float(p)
            edge.p_enable = float(pe)
            edge.p_dir = float(pd)
            edge.p_temporal_before = float(pt)
            edge.p_node_first = float(pn)
            out.append(edge)
        return out
