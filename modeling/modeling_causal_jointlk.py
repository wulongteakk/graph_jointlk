from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

try:
    from torch_geometric.nn import RGCNConv as _PyGRGCNConv  # type: ignore
    from torch_geometric.utils import softmax as pyg_softmax  # type: ignore
    _HAS_PYG = True
except Exception:
    _PyGRGCNConv = None
    pyg_softmax = None
    _HAS_PYG = False


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleRelationalConv(nn.Module):
    """PyG 不可用时的轻量 fallback。"""

    def __init__(self, hidden_size: int, num_relations: int):
        super().__init__()
        self.self_proj = nn.Linear(hidden_size, hidden_size)
        self.msg_proj = nn.Linear(hidden_size, hidden_size)
        self.rel_emb = nn.Embedding(max(int(num_relations), 1), hidden_size)

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_type_ids: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.self_proj(node_states)

        src = edge_index[0]
        dst = edge_index[1]
        rel_vec = self.rel_emb(edge_type_ids.clamp(min=0, max=self.rel_emb.num_embeddings - 1))
        msg = self.msg_proj(node_states[src] + rel_vec)
        agg = node_states.new_zeros(node_states.shape)
        agg.index_add_(0, dst, msg)

        deg = node_states.new_zeros((node_states.size(0), 1))
        deg.index_add_(0, dst, torch.ones((dst.size(0), 1), device=node_states.device, dtype=node_states.dtype))
        agg = agg / deg.clamp_min(1.0)
        return self.self_proj(node_states) + agg


class ResidualRGCNLayer(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int, dropout: float = 0.2):
        super().__init__()
        if _HAS_PYG and _PyGRGCNConv is not None:
            self.conv = _PyGRGCNConv(hidden_size, hidden_size, num_relations=max(int(num_relations), 1))
        else:
            self.conv = SimpleRelationalConv(hidden_size, num_relations=max(int(num_relations), 1))
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = node_states
        updated = self.conv(node_states, edge_index, edge_type_ids)
        updated = F.gelu(updated)
        updated = self.dropout(updated)
        return self.norm(updated + residual)


def segment_softmax(scores: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    if pyg_softmax is not None:
        return pyg_softmax(scores, group_ids)
    out = torch.zeros_like(scores)
    unique_groups = torch.unique(group_ids)
    for gid in unique_groups.tolist():
        mask = group_ids == gid
        out[mask] = torch.softmax(scores[mask], dim=0)
    return out


class CausalJointLKModel(nn.Module):
    """JointLK-style causal edge scorer for final pseudo-label schema."""

    def __init__(
        self,
        model_name: str,
        num_relations: int,
        num_node_types: int,
        hidden_size: int = 256,
        num_gnn_layers: int = 3,
        dropout: float = 0.2,
        freeze_lm: bool = False,
    ):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        self.lm_hidden_size = int(self.lm.config.hidden_size)
        self.hidden_size = int(hidden_size)
        self.num_relations = int(num_relations)
        self.num_node_types = int(num_node_types)

        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        self.sent_proj = nn.Linear(self.lm_hidden_size, self.hidden_size)
        self.node_text_proj = nn.Linear(self.lm_hidden_size, self.hidden_size)
        self.node_type_emb = nn.Embedding(self.num_node_types, self.hidden_size)
        self.node_score_proj = nn.Linear(1, self.hidden_size)
        self.rel_emb = nn.Embedding(self.num_relations, self.hidden_size)

        self.text_to_graph = nn.Linear(self.hidden_size, self.hidden_size)
        self.node_key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.node_query_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.gnn_layers = nn.ModuleList(
            [ResidualRGCNLayer(self.hidden_size, self.num_relations, dropout=dropout) for _ in range(num_gnn_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.support_head = FeedForward(self.hidden_size * 7, self.hidden_size * 2, 1, dropout=dropout)
        self.enable_head = FeedForward(self.hidden_size * 7, self.hidden_size * 2, 1, dropout=dropout)
        self.dir_head = FeedForward(self.hidden_size * 7, self.hidden_size * 2, 1, dropout=dropout)
        self.temporal_head = FeedForward(self.hidden_size * 7, self.hidden_size * 2, 1, dropout=dropout)
        self.node_first_head = FeedForward(self.hidden_size, self.hidden_size * 2, 1, dropout=dropout)
        self.relation_head = FeedForward(self.hidden_size * 4, self.hidden_size * 2, self.num_relations,
                                         dropout=dropout)

    def _get_word_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb_layer = self.lm.get_input_embeddings()
        return emb_layer(token_ids)

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        cls_vec = last_hidden_state[:, 0]
        sent_vec = self.sent_proj(cls_vec)
        return {"last_hidden_state": last_hidden_state, "sent_vec": sent_vec}

    def _encode_nodes(
        self,
        node_input_ids: torch.Tensor,
        node_attention_mask: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_scores: torch.Tensor,
    ) -> torch.Tensor:
        node_tok_emb = self._get_word_embeddings(node_input_ids)
        node_mask = node_attention_mask.unsqueeze(-1).float()
        node_text_vec = (node_tok_emb * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp_min(1.0)
        node_text_vec = self.node_text_proj(node_text_vec)
        node_type_vec = self.node_type_emb(node_type_ids)
        node_score_vec = self.node_score_proj(node_scores)
        node_states = node_text_vec + node_type_vec + node_score_vec
        return self.dropout(F.gelu(node_states))

    def _pool_graph(
        self,
        node_states: torch.Tensor,
        graph_batch: torch.Tensor,
        sent_vec: torch.Tensor,
    ) -> torch.Tensor:
        query = self.node_query_proj(sent_vec)[graph_batch]
        key = self.node_key_proj(node_states)
        scores = (query * key).sum(dim=-1) / (float(self.hidden_size) ** 0.5)
        attn = segment_softmax(scores, graph_batch)
        graph_vec = node_states.new_zeros((sent_vec.size(0), node_states.size(-1)))
        graph_vec.index_add_(0, graph_batch, attn.unsqueeze(-1) * node_states)
        return graph_vec

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        text_outputs = self._encode_text(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        sent_vec = text_outputs["sent_vec"]

        node_states = self._encode_nodes(
            node_input_ids=batch["node_input_ids"],
            node_attention_mask=batch["node_attention_mask"],
            node_type_ids=batch["node_type_ids"],
            node_scores=batch["node_scores"],
        )
        graph_batch = batch["graph_batch"]
        edge_index = batch["edge_index"]
        edge_type_ids = batch["edge_type_ids"].clamp(min=0, max=max(self.num_relations - 1, 0))

        text_bias = self.text_to_graph(sent_vec)[graph_batch]
        node_states = node_states + text_bias

        for layer in self.gnn_layers:
            node_states = layer(node_states, edge_index, edge_type_ids)
            node_states = node_states + text_bias

        graph_vec = self._pool_graph(node_states, graph_batch, sent_vec)

        src_vec = node_states[batch["source_node_index"]]
        tgt_vec = node_states[batch["target_node_index"]]
        rel_vec = self.rel_emb(batch["relation_ids"].clamp(min=0, max=max(self.num_relations - 1, 0)))

        support_features = torch.cat(
            [
                src_vec,
                tgt_vec,
                torch.abs(src_vec - tgt_vec),
                src_vec * tgt_vec,
                graph_vec,
                sent_vec,
                rel_vec,
            ],
            dim=-1,
        )
        relation_features = torch.cat([src_vec, tgt_vec, graph_vec, sent_vec], dim=-1)

        support_logits = self.support_head(support_features).squeeze(-1)
        enable_logits = self.enable_head(support_features).squeeze(-1)
        dir_logits = self.dir_head(support_features).squeeze(-1)
        temporal_logits = self.temporal_head(support_features).squeeze(-1)
        node_first_logits_all = self.node_first_head(node_states).squeeze(-1)
        src_first_logits = node_first_logits_all[batch["source_node_index"]]
        dst_first_logits = node_first_logits_all[batch["target_node_index"]]
        # 兼容：保留 node_first_*（按边级输出）
        node_first_logits = src_first_logits
        relation_logits = self.relation_head(relation_features)

        return {
            "support_logits": support_logits,
            "causal_logits": support_logits,
            "enable_logits": enable_logits,
            "dir_logits": dir_logits,
            "temporal_logits": temporal_logits,
            "node_first_logits": node_first_logits,
            "src_first_logits": src_first_logits,
            "dst_first_logits": dst_first_logits,
            "relation_logits": relation_logits,
            "support_prob": torch.sigmoid(support_logits),
            "causal_prob": torch.sigmoid(support_logits),
            "enable_prob": torch.sigmoid(enable_logits),
            "dir_prob": torch.sigmoid(dir_logits),
            "temporal_prob": torch.sigmoid(temporal_logits),
            "node_first_prob": torch.sigmoid(node_first_logits),
            "src_first_prob": torch.sigmoid(src_first_logits),
            "dst_first_prob": torch.sigmoid(dst_first_logits),
            "graph_vec": graph_vec,
            "sent_vec": sent_vec,
            "node_states": node_states,
        }

    @torch.no_grad()
    def score_edges(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.eval()
        outputs = self.forward(batch)
        return {
            "sample_ids": batch.get("sample_ids", []),
            "support_prob": outputs["support_prob"].detach().cpu().tolist(),
            "pred_relation": outputs["relation_logits"].argmax(dim=-1).detach().cpu().tolist(),
        }


def weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    if sample_weights is not None:
        losses = losses * sample_weights
        return losses.sum() / sample_weights.sum().clamp_min(1.0)
    return losses.mean()



def compute_training_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    relation_labels: Optional[torch.Tensor] = None,
    multitask_labels: Optional[Dict[str, torch.Tensor]] = None,
    relation_loss_weight: float = 0.2,
    sample_weights: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    cf_group_ids: Optional[list[str]] = None,
    cf_roles: Optional[torch.Tensor] = None,
    cf_margin: float = 0.2,
    cf_loss_weight: float = 0.4,
) -> Dict[str, torch.Tensor]:
    labels = labels.float()
    causal_logits = outputs.get("causal_logits", outputs["support_logits"])
    support_loss = weighted_bce_with_logits(
        causal_logits,
        labels,
        sample_weights=sample_weights,
        pos_weight=pos_weight,
    )

    total_loss = support_loss
    relation_loss = torch.zeros_like(support_loss)
    aux_loss = torch.zeros_like(support_loss)
    cf_loss = torch.zeros_like(support_loss)

    multitask_labels = multitask_labels or {}

    def _masked_bce(logits: torch.Tensor, labels_local: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        if labels_local is None or mask is None:
            return torch.zeros_like(support_loss)
        losses = F.binary_cross_entropy_with_logits(logits, labels_local.float(), reduction="none")
        losses = losses * mask.float()
        denom = mask.sum().clamp_min(1.0)
        return losses.sum() / denom

    aux_loss = (
        _masked_bce(outputs["enable_logits"], multitask_labels.get("enable_labels"), multitask_labels.get("enable_mask"))
        + _masked_bce(outputs["dir_logits"], multitask_labels.get("dir_labels"), multitask_labels.get("dir_mask"))
        + _masked_bce(outputs["temporal_logits"], multitask_labels.get("temp_labels"), multitask_labels.get("temp_mask"))
        + _masked_bce(outputs.get("src_first_logits", outputs["node_first_logits"]), multitask_labels.get("src_first_labels"), multitask_labels.get("src_first_mask"))
        + _masked_bce(outputs.get("dst_first_logits", outputs["node_first_logits"]), multitask_labels.get("dst_first_labels"), multitask_labels.get("dst_first_mask"))
    ) / 5.0

    if relation_labels is not None:
        positive_mask = labels > 0.5
        if positive_mask.any():
            relation_losses = F.cross_entropy(
                outputs["relation_logits"][positive_mask],
                relation_labels[positive_mask],
                reduction="none",
            )
            if sample_weights is not None:
                pos_weights = sample_weights[positive_mask]
                relation_loss = (relation_losses * pos_weights).sum() / pos_weights.sum().clamp_min(1.0)
            else:
                relation_loss = relation_losses.mean()
            total_loss = total_loss + relation_loss_weight * relation_loss

    if cf_group_ids is not None and cf_roles is not None and len(cf_group_ids) == int(cf_roles.shape[0]):
        causal_score = outputs.get("causal_prob", outputs["support_prob"])
        dir_score = outputs.get("dir_prob", torch.sigmoid(outputs["dir_logits"]))
        temp_score = outputs.get("temporal_prob", torch.sigmoid(outputs["temporal_logits"]))
        cf_score = causal_score + 0.5 * dir_score + 0.5 * temp_score

        per_group_losses = []
        cf_roles = cf_roles.long()
        unique_group_ids = [gid for gid in sorted(set(cf_group_ids)) if gid]
        for gid in unique_group_ids:
            idx = [i for i, g in enumerate(cf_group_ids) if g == gid]
            if not idx:
                continue
            idx_tensor = torch.tensor(idx, device=cf_score.device, dtype=torch.long)
            group_roles = cf_roles[idx_tensor]
            group_scores = cf_score[idx_tensor]
            pos_mask = group_roles == 1
            neg_mask = group_roles == 0
            if pos_mask.any() and neg_mask.any():
                score_pos = group_scores[pos_mask].mean()
                score_neg = group_scores[neg_mask].mean()
                per_group_losses.append(F.relu(float(cf_margin) - score_pos + score_neg))
        if per_group_losses:
            cf_loss = torch.stack(per_group_losses).mean()
            total_loss = total_loss + float(cf_loss_weight) * cf_loss

    total_loss = total_loss + 0.2 * aux_loss



    return {
        "loss": total_loss,
        "support_loss": support_loss,
        "relation_loss": relation_loss,
        "aux_loss": aux_loss,
        "cf_loss": cf_loss,
    }
