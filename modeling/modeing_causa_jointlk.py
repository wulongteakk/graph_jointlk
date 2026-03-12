
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import softmax
from transformers import AutoModel


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


class ResidualRGCNLayer(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int, dropout: float = 0.2):
        super().__init__()
        self.conv = RGCNConv(hidden_size, hidden_size, num_relations=max(int(num_relations), 1))
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


class CausalJointLKModel(nn.Module):
    """JointLK-style causal edge scorer for Instance-KG.

    关键思想：
    1. 文本编码器处理 query + target + evidence；
    2. 节点初始化直接来自“实例节点文本”的 token embedding，不依赖 BG-KG；
    3. RGCN 在实例子图上做消息传递；
    4. 文本向量以 residual 方式注入图表示；
    5. 输出边支持分数（binary）+ 关系类型辅助分类（optional）。
    """

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
        self.relation_head = FeedForward(self.hidden_size * 4, self.hidden_size * 2, self.num_relations, dropout=dropout)

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
        attn = softmax(scores, graph_batch)
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
        relation_logits = self.relation_head(relation_features)

        return {
            "support_logits": support_logits,
            "relation_logits": relation_logits,
            "support_prob": torch.sigmoid(support_logits),
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


def compute_training_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    relation_labels: Optional[torch.Tensor] = None,
    relation_loss_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    labels = labels.float()
    support_loss = F.binary_cross_entropy_with_logits(outputs["support_logits"], labels)

    total_loss = support_loss
    relation_loss = torch.zeros_like(support_loss)

    if relation_labels is not None:
        positive_mask = labels > 0.5
        if positive_mask.any():
            relation_loss = F.cross_entropy(
                outputs["relation_logits"][positive_mask],
                relation_labels[positive_mask],
            )
            total_loss = total_loss + relation_loss_weight * relation_loss

    return {
        "loss": total_loss,
        "support_loss": support_loss,
        "relation_loss": relation_loss,
    }
