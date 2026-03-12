# causal_jointlk patch code bundle


## configs/causal_prior.yaml
```yaml
# 轻量先验配置（不是背景图谱）
# 目标：把 CTP / 层间约束 / 触发词规则 / beam penalty 外置成可调参数
version: 1

relations:
  whitelist:
    - CAUSES
    - LEADS_TO
    - RESULTS_IN
    - TRIGGERS
    - INDUCES
    - ENABLES
    - WORSENS
    - PREVENTS
    - MITIGATES
    - INHIBITS

# CTP = Causal Transition Prior
# 可按你的因果层级体系继续扩展
ctp_allowed_transitions:
  ROOT: ["CAUSE", "FACTOR", "CONDITION", "ACTION"]
  CAUSE: ["MECHANISM", "INTERMEDIATE", "STATE", "EVENT", "OUTCOME"]
  FACTOR: ["MECHANISM", "STATE", "EVENT", "OUTCOME"]
  CONDITION: ["STATE", "EVENT", "OUTCOME"]
  ACTION: ["STATE", "EVENT", "OUTCOME"]
  MECHANISM: ["INTERMEDIATE", "STATE", "EVENT", "OUTCOME"]
  INTERMEDIATE: ["INTERMEDIATE", "STATE", "EVENT", "OUTCOME"]
  STATE: ["STATE", "EVENT", "OUTCOME"]
  EVENT: ["EVENT", "OUTCOME"]
  OUTCOME: []

# 用于 beam search 的层优先级，可选
layer_order:
  - ROOT
  - CAUSE
  - FACTOR
  - CONDITION
  - ACTION
  - MECHANISM
  - INTERMEDIATE
  - STATE
  - EVENT
  - OUTCOME

trigger_words:
  zh:
    - 导致
    - 引发
    - 引起
    - 从而
    - 因此
    - 使得
    - 由于
    - 致使
    - 造成
    - 诱发
    - 促使
    - 以致
    - 继而
    - 结果
    - 所以
    - 因为
    - 进而
    - 带来
    - 促成
    - 形成
  en:
    - cause
    - causes
    - caused
    - lead to
    - leads to
    - led to
    - result in
    - results in
    - resulted in
    - trigger
    - triggers
    - triggered
    - induce
    - induces
    - induced
    - due to
    - because of
    - thereby
    - thus
    - hence
    - so that

relation_aliases:
  CAUSES:
    - 导致
    - 引发
    - 引起
    - 造成
    - 致使
    - cause
    - causes
    - caused
  LEADS_TO:
    - 从而
    - 进而
    - 继而
    - lead to
    - leads to
    - led to
  RESULTS_IN:
    - 因此
    - 结果
    - resulting in
    - result in
    - results in
  TRIGGERS:
    - 触发
    - trigger
    - triggers
    - triggered
  INDUCES:
    - 诱发
    - induce
    - induces
    - induced
  ENABLES:
    - 使得
    - 促成
    - enable
    - enables
    - enabled
  WORSENS:
    - 恶化
    - worsen
    - worsens
    - worsened
  PREVENTS:
    - 防止
    - 预防
    - prevent
    - prevents
    - prevented
  MITIGATES:
    - 缓解
    - 降低
    - mitigate
    - mitigates
    - mitigated
  INHIBITS:
    - 抑制
    - inhibit
    - inhibits
    - inhibited

beam_search:
  max_hops: 6
  top_k: 8
  hop_penalty: 0.25
  skip_layer_penalty: 0.75
  distance_penalty: 0.15
  missing_layer_penalty: 0.50
  unsupported_edge_penalty: 1.00
  duplicate_node_penalty: 0.40

evidence_gate:
  min_support_score: 0.55
  min_lexical_overlap: 0.18
  require_trigger_or_reltype: true
  trigger_bonus: 0.25
  two_sided_entity_bonus: 0.20
  ordered_bonus: 0.10
  negative_cues:
    zh: ["尚未证实", "不一定", "未导致", "未引起", "并非由于"]
    en: ["not necessarily", "not cause", "did not cause", "not lead to", "unconfirmed"]

training:
  max_text_length: 320
  max_node_text_length: 24
  hidden_size: 256
  num_gnn_layers: 3
  dropout: 0.20
  relation_loss_weight: 0.20

```


## modeling/causal_jointlk_io.py
```python

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase


DEFAULT_NODE_TYPE_TO_ID = {
    "UNK": 0,
    "ROOT": 1,
    "CAUSE": 2,
    "FACTOR": 3,
    "CONDITION": 4,
    "ACTION": 5,
    "MECHANISM": 6,
    "INTERMEDIATE": 7,
    "STATE": 8,
    "EVENT": 9,
    "OUTCOME": 10,
}


def load_prior_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"prior config not found: {config_path}")

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read YAML prior config.") from exc

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_relation_to_id(prior_config: Dict[str, Any]) -> Dict[str, int]:
    whitelist = prior_config.get("relations", {}).get("whitelist", [])
    rels = ["UNK"] + list(whitelist)
    return {rel: idx for idx, rel in enumerate(rels)}


def build_node_type_to_id(prior_config: Dict[str, Any]) -> Dict[str, int]:
    ctp = prior_config.get("ctp_allowed_transitions", {})
    node_type_to_id = dict(DEFAULT_NODE_TYPE_TO_ID)
    for key in ctp:
        if key not in node_type_to_id:
            node_type_to_id[key] = len(node_type_to_id)
        for value in ctp[key]:
            if value not in node_type_to_id:
                node_type_to_id[value] = len(node_type_to_id)
    return node_type_to_id


def normalize_text(value: Optional[str]) -> str:
    value = (value or "").strip()
    return " ".join(value.split())


def build_example_text(
    query: Optional[str],
    source_text: str,
    relation_text: str,
    target_text: str,
    evidence_texts: Sequence[str],
    doc_title: Optional[str] = None,
    max_evidence: int = 3,
) -> str:
    query = normalize_text(query)
    source_text = normalize_text(source_text)
    relation_text = normalize_text(relation_text)
    target_text = normalize_text(target_text)
    doc_title = normalize_text(doc_title)

    evidence_texts = [normalize_text(x) for x in evidence_texts if normalize_text(x)]
    evidence_texts = evidence_texts[:max_evidence]

    parts: List[str] = []
    if doc_title:
        parts.append(f"[DOC] {doc_title}")
    if query:
        parts.append(f"[QUERY] {query}")
    parts.append(f"[SOURCE] {source_text}")
    parts.append(f"[RELATION] {relation_text}")
    parts.append(f"[TARGET] {target_text}")
    if evidence_texts:
        parts.append("[EVIDENCE] " + " [SEP] ".join(evidence_texts))
    return " ".join(parts)


def _mean_pool_embeddings(
    emb: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    summed = (emb * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _ensure_non_empty_edges(
    edge_index: List[List[int]],
    num_nodes: int,
    edge_type_ids: Optional[List[int]] = None,
) -> Tuple[List[List[int]], List[int]]:
    edge_type_ids = list(edge_type_ids or [])
    if edge_index and len(edge_index[0]) > 0:
        return edge_index, edge_type_ids

    srcs = list(range(num_nodes))
    dsts = list(range(num_nodes))
    rels = [0 for _ in range(num_nodes)]
    return [srcs, dsts], rels


def batchify_examples(
    examples: Sequence[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    relation_to_id: Dict[str, int],
    node_type_to_id: Dict[str, int],
    max_text_length: int = 320,
    max_node_text_length: int = 24,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("examples must not be empty")

    text_inputs = tokenizer(
        [ex["text"] for ex in examples],
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )

    flat_node_texts: List[str] = []
    flat_node_type_ids: List[int] = []
    flat_node_scores: List[float] = []
    flat_graph_batch: List[int] = []
    global_source_indices: List[int] = []
    global_target_indices: List[int] = []
    relation_ids: List[int] = []
    labels: List[float] = []
    relation_labels: List[int] = []
    sample_ids: List[str] = []
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_types: List[int] = []

    node_offset = 0
    for graph_idx, ex in enumerate(examples):
        node_texts = list(ex.get("node_texts") or [])
        if not node_texts:
            source_text = ex.get("source_text") or "source"
            target_text = ex.get("target_text") or "target"
            node_texts = [source_text, target_text]

        node_layers = list(ex.get("node_layer_types") or [])
        node_scores = list(ex.get("node_scores") or [])
        if len(node_layers) < len(node_texts):
            node_layers.extend(["UNK"] * (len(node_texts) - len(node_layers)))
        if len(node_scores) < len(node_texts):
            node_scores.extend([0.0] * (len(node_texts) - len(node_scores)))

        local_num_nodes = len(node_texts)
        local_source_idx = int(ex.get("source_idx", 0))
        local_target_idx = int(ex.get("target_idx", min(1, local_num_nodes - 1)))

        flat_node_texts.extend([normalize_text(x) or "<EMPTY>" for x in node_texts])
        flat_node_type_ids.extend([node_type_to_id.get(str(x).upper(), 0) for x in node_layers])
        flat_node_scores.extend([float(x) for x in node_scores])
        flat_graph_batch.extend([graph_idx] * local_num_nodes)

        global_source_indices.append(node_offset + local_source_idx)
        global_target_indices.append(node_offset + local_target_idx)

        rel_name = str(ex.get("candidate_relation") or ex.get("relation_text") or "UNK").upper()
        relation_ids.append(relation_to_id.get(rel_name, 0))
        relation_labels.append(relation_to_id.get(str(ex.get("gold_relation") or rel_name).upper(), 0))
        labels.append(float(ex.get("label", 0.0)))
        sample_ids.append(str(ex.get("sample_id") or f"sample-{graph_idx}"))

        ex_edge_index = ex.get("edge_index") or [[], []]
        ex_edge_type = ex.get("edge_types") or ex.get("edge_type_labels") or []
        ex_edge_type_ids = [relation_to_id.get(str(rel).upper(), 0) for rel in ex_edge_type]
        ex_edge_index, ex_edge_type_ids = _ensure_non_empty_edges(ex_edge_index, local_num_nodes, ex_edge_type_ids)
        for src, dst, rel_id in zip(ex_edge_index[0], ex_edge_index[1], ex_edge_type_ids):
            edge_src.append(node_offset + int(src))
            edge_dst.append(node_offset + int(dst))
            edge_types.append(int(rel_id))

        node_offset += local_num_nodes

    node_inputs = tokenizer(
        flat_node_texts,
        padding=True,
        truncation=True,
        max_length=max_node_text_length,
        return_tensors="pt",
    )

    batch = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "node_input_ids": node_inputs["input_ids"],
        "node_attention_mask": node_inputs["attention_mask"],
        "node_type_ids": torch.tensor(flat_node_type_ids, dtype=torch.long),
        "node_scores": torch.tensor(flat_node_scores, dtype=torch.float).unsqueeze(-1),
        "graph_batch": torch.tensor(flat_graph_batch, dtype=torch.long),
        "edge_index": torch.tensor([edge_src, edge_dst], dtype=torch.long),
        "edge_type_ids": torch.tensor(edge_types, dtype=torch.long),
        "source_node_index": torch.tensor(global_source_indices, dtype=torch.long),
        "target_node_index": torch.tensor(global_target_indices, dtype=torch.long),
        "relation_ids": torch.tensor(relation_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float),
        "relation_labels": torch.tensor(relation_labels, dtype=torch.long),
    }

    if "token_type_ids" in text_inputs:
        batch["token_type_ids"] = text_inputs["token_type_ids"]
    if "token_type_ids" in node_inputs:
        batch["node_token_type_ids"] = node_inputs["token_type_ids"]

    if device is not None:
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)

    batch["sample_ids"] = sample_ids
    return batch


def sigmoid_to_labels(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (prob >= threshold).long()


def detach_to_list(x: torch.Tensor) -> List[Any]:
    return x.detach().cpu().tolist()

```


## modeling/modeling_causal_jointlk.py
```python

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

```


## experiments/causal_jointlk/dataset.py
```python

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from torch.utils.data import Dataset

from modeling.causal_jointlk_io import (
    build_example_text,
    build_node_type_to_id,
    build_relation_to_id,
    load_prior_config,
)


class CausalEdgeDataset(Dataset):
    """JSONL dataset for edge-level causal support training.

    每条样本建议包含：
    {
      "sample_id": "...",
      "query": "...",
      "doc_title": "...",
      "source_text": "...",
      "target_text": "...",
      "candidate_relation": "CAUSES",
      "gold_relation": "CAUSES",
      "evidence_texts": ["...", "..."],
      "node_texts": ["...", "...", "..."],
      "node_layer_types": ["CAUSE", "INTERMEDIATE", "OUTCOME"],
      "node_scores": [0.9, 0.7, 0.8],
      "edge_index": [[0,1],[1,2]],
      "edge_types": ["LEADS_TO", "RESULTS_IN"],
      "source_idx": 0,
      "target_idx": 2,
      "label": 1
    }
    """

    def __init__(self, jsonl_path: str, prior_config_path: str):
        self.jsonl_path = Path(jsonl_path)
        self.prior_config = load_prior_config(prior_config_path)
        self.relation_to_id = build_relation_to_id(self.prior_config)
        self.node_type_to_id = build_node_type_to_id(self.prior_config)
        self.records = self._load_records()

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"dataset not found: {self.jsonl_path}")

        rows: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as fin:
            for line_idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj.setdefault("sample_id", f"{self.jsonl_path.stem}-{line_idx}")
                obj["text"] = build_example_text(
                    query=obj.get("query"),
                    source_text=obj.get("source_text") or _safe_pick_text(obj, obj.get("source_idx", 0), "source"),
                    relation_text=obj.get("candidate_relation") or obj.get("relation_text") or "UNK",
                    target_text=obj.get("target_text") or _safe_pick_text(obj, obj.get("target_idx", 1), "target"),
                    evidence_texts=obj.get("evidence_texts") or [],
                    doc_title=obj.get("doc_title"),
                )
                rows.append(obj)
        return rows

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def _safe_pick_text(record: Dict[str, Any], idx: int, default_value: str) -> str:
    node_texts: Sequence[str] = record.get("node_texts") or []
    if node_texts and 0 <= int(idx) < len(node_texts):
        return str(node_texts[int(idx)])
    return default_value

```


## experiments/causal_jointlk/metrics.py
```python

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_binary_edge_metrics(
    gold: Sequence[int],
    prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    assert len(gold) == len(prob), "gold and prob must have same length"
    pred = [1 if p >= threshold else 0 for p in prob]

    tp = sum(1 for y, yhat in zip(gold, pred) if y == 1 and yhat == 1)
    fp = sum(1 for y, yhat in zip(gold, pred) if y == 0 and yhat == 1)
    fn = sum(1 for y, yhat in zip(gold, pred) if y == 1 and yhat == 0)
    tn = sum(1 for y, yhat in zip(gold, pred) if y == 0 and yhat == 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, len(gold))

    metrics = {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_accuracy": acc,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(set(gold)) > 1:
            metrics["edge_auroc"] = float(roc_auc_score(gold, prob))
            metrics["edge_aupr"] = float(average_precision_score(gold, prob))
    except Exception:
        pass

    return metrics


def compute_relation_metrics(
    gold: Sequence[int],
    pred: Sequence[int],
    ignore_label: int = 0,
) -> Dict[str, float]:
    labels = sorted(set(gold) | set(pred))
    labels = [x for x in labels if x != ignore_label]
    if not labels:
        return {"rel_micro_f1": 0.0, "rel_macro_f1": 0.0}

    total_tp = total_fp = total_fn = 0
    per_label_f1: List[float] = []
    for label in labels:
        tp = sum(1 for y, yhat in zip(gold, pred) if y == label and yhat == label)
        fp = sum(1 for y, yhat in zip(gold, pred) if y != label and yhat == label)
        fn = sum(1 for y, yhat in zip(gold, pred) if y == label and yhat != label)
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        per_label_f1.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = _safe_div(total_tp, total_tp + total_fp)
    micro_r = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)
    macro_f1 = sum(per_label_f1) / len(per_label_f1)
    return {"rel_micro_f1": micro_f1, "rel_macro_f1": macro_f1}


def _chain_to_node_set(chain: Sequence[Tuple[str, str, str]]) -> set:
    nodes = set()
    for head, _, tail in chain:
        nodes.add(head)
        nodes.add(tail)
    return nodes


def _set_f1(gold_set: set, pred_set: set) -> float:
    if not gold_set and not pred_set:
        return 1.0
    inter = len(gold_set & pred_set)
    p = _safe_div(inter, len(pred_set))
    r = _safe_div(inter, len(gold_set))
    return _safe_div(2 * p * r, p + r)


def compute_chain_metrics(
    gold_chains: Sequence[Sequence[Tuple[str, str, str]]],
    pred_chains: Sequence[Sequence[Tuple[str, str, str]]],
) -> Dict[str, float]:
    assert len(gold_chains) == len(pred_chains), "gold/pred chain list length mismatch"
    n = len(gold_chains)
    if n == 0:
        return {
            "chain_exact_match": 0.0,
            "chain_edge_f1": 0.0,
            "chain_node_f1": 0.0,
        }

    exact = 0.0
    edge_f1 = 0.0
    node_f1 = 0.0
    for gold_chain, pred_chain in zip(gold_chains, pred_chains):
        gold_edge_set = set(tuple(x) for x in gold_chain)
        pred_edge_set = set(tuple(x) for x in pred_chain)
        gold_node_set = _chain_to_node_set(gold_chain)
        pred_node_set = _chain_to_node_set(pred_chain)
        if list(gold_chain) == list(pred_chain):
            exact += 1.0
        edge_f1 += _set_f1(gold_edge_set, pred_edge_set)
        node_f1 += _set_f1(gold_node_set, pred_node_set)

    return {
        "chain_exact_match": exact / n,
        "chain_edge_f1": edge_f1 / n,
        "chain_node_f1": node_f1 / n,
    }


def compute_ranking_metrics(
    ranked_gold_flags: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """Each row is ranked candidate list, value 1 means relevant chain."""
    mrr = 0.0
    hits1 = 0.0
    hits3 = 0.0
    hits5 = 0.0
    ndcg5 = 0.0
    n = len(ranked_gold_flags)
    if n == 0:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@5": 0.0, "ndcg@5": 0.0}

    for flags in ranked_gold_flags:
        first_rank = None
        dcg = 0.0
        ideal = sorted(flags, reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(flags[:5], start=1):
            if rel and first_rank is None:
                first_rank = rank
            if rel:
                dcg += 1.0 / math.log2(rank + 1)
        for rank, rel in enumerate(ideal[:5], start=1):
            if rel:
                idcg += 1.0 / math.log2(rank + 1)

        if first_rank is not None:
            mrr += 1.0 / first_rank
            hits1 += 1.0 if first_rank <= 1 else 0.0
            hits3 += 1.0 if first_rank <= 3 else 0.0
            hits5 += 1.0 if first_rank <= 5 else 0.0
        ndcg5 += _safe_div(dcg, idcg)

    return {
        "mrr": mrr / n,
        "hits@1": hits1 / n,
        "hits@3": hits3 / n,
        "hits@5": hits5 / n,
        "ndcg@5": ndcg5 / n,
    }


def summarize_for_paper(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = set()
    for row in rows:
        keys.update(row.keys())
    summary = {}
    for key in sorted(keys):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[key] = sum(values) / len(values)
    return summary

```


## experiments/causal_jointlk/train_causal_jointlk.py
```python

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from experiments.causal_jointlk.dataset import CausalEdgeDataset
from experiments.causal_jointlk.metrics import compute_binary_edge_metrics, compute_relation_metrics
from modeling.causal_jointlk_io import batchify_examples
from modeling.modeling_causal_jointlk import CausalJointLKModel, compute_training_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_collate_fn(
    tokenizer,
    relation_to_id: Dict[str, int],
    node_type_to_id: Dict[str, int],
    max_text_length: int,
    max_node_text_length: int,
):
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return batchify_examples(
            examples=examples,
            tokenizer=tokenizer,
            relation_to_id=relation_to_id,
            node_type_to_id=node_type_to_id,
            max_text_length=max_text_length,
            max_node_text_length=max_node_text_length,
            device=None,
        )
    return collate_fn


@torch.no_grad()
def evaluate(
    model: CausalJointLKModel,
    data_loader: DataLoader,
    device: torch.device,
    relation_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    gold: List[int] = []
    prob: List[float] = []
    gold_rel: List[int] = []
    pred_rel: List[int] = []
    losses: List[float] = []

    for batch in tqdm(data_loader, desc="eval", leave=False):
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        outputs = model(batch)
        loss_dict = compute_training_loss(
            outputs=outputs,
            labels=batch["labels"],
            relation_labels=batch["relation_labels"],
            relation_loss_weight=relation_loss_weight,
        )
        losses.append(float(loss_dict["loss"].detach().cpu()))
        gold.extend(batch["labels"].long().detach().cpu().tolist())
        prob.extend(outputs["support_prob"].detach().cpu().tolist())
        gold_rel.extend(batch["relation_labels"].detach().cpu().tolist())
        pred_rel.extend(outputs["relation_logits"].argmax(dim=-1).detach().cpu().tolist())

    metrics = {"loss": sum(losses) / max(len(losses), 1)}
    metrics.update(compute_binary_edge_metrics(gold, prob))
    metrics.update(compute_relation_metrics(gold_rel, pred_rel))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", required=True)
    parser.add_argument("--prior_config", default="configs/causal_prior.yaml")
    parser.add_argument("--model_name", default="roberta-large")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_text_length", type=int, default=320)
    parser.add_argument("--max_node_text_length", type=int, default=24)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--relation_loss_weight", type=float, default=0.20)
    parser.add_argument("--freeze_lm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_set = CausalEdgeDataset(args.train_jsonl, args.prior_config)
    dev_set = CausalEdgeDataset(args.dev_jsonl, args.prior_config)

    collate_fn = build_collate_fn(
        tokenizer=tokenizer,
        relation_to_id=train_set.relation_to_id,
        node_type_to_id=train_set.node_type_to_id,
        max_text_length=args.max_text_length,
        max_node_text_length=args.max_node_text_length,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalJointLKModel(
        model_name=args.model_name,
        num_relations=len(train_set.relation_to_id),
        num_node_types=len(train_set.node_type_to_id),
        hidden_size=args.hidden_size,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        freeze_lm=args.freeze_lm,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_metric = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss_dict = compute_training_loss(
                outputs=outputs,
                labels=batch["labels"],
                relation_labels=batch["relation_labels"],
                relation_loss_weight=args.relation_loss_weight,
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))

        dev_metrics = evaluate(
            model=model,
            data_loader=dev_loader,
            device=device,
            relation_loss_weight=args.relation_loss_weight,
        )
        dev_metrics["epoch"] = epoch
        dev_metrics["train_loss"] = sum(epoch_losses) / max(len(epoch_losses), 1)
        history.append(dev_metrics)

        current_metric = dev_metrics.get("edge_f1", 0.0)
        print(json.dumps(dev_metrics, ensure_ascii=False, indent=2))

        if current_metric > best_metric:
            best_metric = current_metric
            ckpt = {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "relation_to_id": train_set.relation_to_id,
                "node_type_to_id": train_set.node_type_to_id,
                "best_dev_metrics": dev_metrics,
            }
            torch.save(ckpt, output_dir / "best_model.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")

        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(dev_metrics, ensure_ascii=False) + "\n")

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_edge_f1": best_metric,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

```


## experiments/causal_jointlk/eval_causal_jointlk.py
```python

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.causal_jointlk.dataset import CausalEdgeDataset
from experiments.causal_jointlk.metrics import compute_binary_edge_metrics, compute_relation_metrics
from modeling.causal_jointlk_io import batchify_examples
from modeling.modeling_causal_jointlk import CausalJointLKModel


def build_collate_fn(tokenizer, relation_to_id, node_type_to_id, max_text_length, max_node_text_length):
    def collate_fn(examples):
        return batchify_examples(
            examples=examples,
            tokenizer=tokenizer,
            relation_to_id=relation_to_id,
            node_type_to_id=node_type_to_id,
            max_text_length=max_text_length,
            max_node_text_length=max_node_text_length,
            device=None,
        )
    return collate_fn


@torch.no_grad()
def run_eval(model, data_loader, device):
    model.eval()
    gold: List[int] = []
    prob: List[float] = []
    gold_rel: List[int] = []
    pred_rel: List[int] = []
    rows: List[Dict[str, Any]] = []

    for batch in data_loader:
        sample_ids = batch.get("sample_ids", [])
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)

        outputs = model(batch)
        pred_prob = outputs["support_prob"].detach().cpu().tolist()
        pred_rel_batch = outputs["relation_logits"].argmax(dim=-1).detach().cpu().tolist()
        gold_batch = batch["labels"].long().detach().cpu().tolist()
        gold_rel_batch = batch["relation_labels"].detach().cpu().tolist()

        gold.extend(gold_batch)
        prob.extend(pred_prob)
        gold_rel.extend(gold_rel_batch)
        pred_rel.extend(pred_rel_batch)

        for sample_id, p, y, gr, pr in zip(sample_ids, pred_prob, gold_batch, gold_rel_batch, pred_rel_batch):
            rows.append(
                {
                    "sample_id": sample_id,
                    "support_prob": p,
                    "gold_label": y,
                    "gold_relation": gr,
                    "pred_relation": pr,
                }
            )

    metrics = {}
    metrics.update(compute_binary_edge_metrics(gold, prob))
    metrics.update(compute_relation_metrics(gold_rel, pred_rel))
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior_config", default="configs/causal_prior.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_text_length", type=int, default=320)
    parser.add_argument("--max_node_text_length", type=int, default=24)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    relation_to_id = checkpoint["relation_to_id"]
    node_type_to_id = checkpoint["node_type_to_id"]

    model_name = saved_args.get("model_name", "roberta-large")
    hidden_size = int(saved_args.get("hidden_size", 256))
    num_gnn_layers = int(saved_args.get("num_gnn_layers", 3))
    dropout = float(saved_args.get("dropout", 0.2))
    freeze_lm = bool(saved_args.get("freeze_lm", False))

    tokenizer_dir = Path(args.checkpoint).parent / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_set = CausalEdgeDataset(args.test_jsonl, args.prior_config)
    collate_fn = build_collate_fn(
        tokenizer,
        relation_to_id=relation_to_id,
        node_type_to_id=node_type_to_id,
        max_text_length=args.max_text_length,
        max_node_text_length=args.max_node_text_length,
    )
    data_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalJointLKModel(
        model_name=model_name,
        num_relations=len(relation_to_id),
        num_node_types=len(node_type_to_id),
        hidden_size=hidden_size,
        num_gnn_layers=num_gnn_layers,
        dropout=dropout,
        freeze_lm=freeze_lm,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    metrics, rows = run_eval(model, data_loader, device)
    output = {
        "metrics": metrics,
        "rows": rows,
        "checkpoint": args.checkpoint,
    }
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

```


## backend/src/causal_jointlk/__init__.py
```python

from .service import CausalJointLKService
from .prior import CausalPrior, load_prior_config
from .schemas import CausalEdge, CausalNode, CandidateChain, ExtractionResult

__all__ = [
    "CausalJointLKService",
    "CausalPrior",
    "load_prior_config",
    "CausalEdge",
    "CausalNode",
    "CandidateChain",
    "ExtractionResult",
]

```


## backend/src/causal_jointlk/schemas.py
```python

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceUnitRef:
    unit_id: str
    content: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalNode:
    node_id: str
    text: str
    layer: str = "UNK"
    doc_id: Optional[str] = None
    kg_scope: Optional[str] = None
    kg_id: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    source_text: str
    target_text: str
    source_layer: str = "UNK"
    target_layer: str = "UNK"
    evidence_unit_id: Optional[str] = None
    evidence_text: Optional[str] = None
    doc_id: Optional[str] = None
    kg_scope: Optional[str] = None
    kg_id: Optional[str] = None
    score: float = 0.0
    supported: bool = False
    support_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateChain:
    chain_id: str
    nodes: List[str]
    edges: List[CausalEdge]
    score: float
    missing_layers: List[str] = field(default_factory=list)
    unsupported_edges: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "nodes": self.nodes,
            "score": self.score,
            "missing_layers": self.missing_layers,
            "unsupported_edges": self.unsupported_edges,
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "source_text": e.source_text,
                    "target_text": e.target_text,
                    "source_layer": e.source_layer,
                    "target_layer": e.target_layer,
                    "evidence_unit_id": e.evidence_unit_id,
                    "evidence_text": e.evidence_text,
                    "score": e.score,
                    "supported": e.supported,
                    "support_score": e.support_score,
                    "meta": e.meta,
                }
                for e in self.edges
            ],
        }


@dataclass
class ExtractionResult:
    mode: str
    query: Optional[str]
    doc_id: Optional[str]
    target_node_id: Optional[str]
    chains: List[CandidateChain]
    subgraph_nodes: List[CausalNode]
    subgraph_edges: List[CausalEdge]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "query": self.query,
            "doc_id": self.doc_id,
            "target_node_id": self.target_node_id,
            "chains": [c.to_dict() for c in self.chains],
            "subgraph_nodes": [vars(n) for n in self.subgraph_nodes],
            "subgraph_edges": [vars(e) for e in self.subgraph_edges],
            "meta": self.meta,
        }

```


## backend/src/causal_jointlk/prior.py
```python

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def load_prior_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"prior config not found: {config_path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read YAML config.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class CausalPrior:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rel_whitelist = set((config.get("relations") or {}).get("whitelist") or [])
        self.ctp = {str(k).upper(): [str(v).upper() for v in vs] for k, vs in (config.get("ctp_allowed_transitions") or {}).items()}
        self.layer_order = [str(x).upper() for x in config.get("layer_order") or []]
        trigger_words = config.get("trigger_words") or {}
        self.trigger_words = {
            lang: [str(x).strip().lower() for x in values]
            for lang, values in trigger_words.items()
        }
        relation_aliases = config.get("relation_aliases") or {}
        self.relation_aliases = {
            str(rel).upper(): [str(x).strip().lower() for x in aliases]
            for rel, aliases in relation_aliases.items()
        }

    def is_relation_allowed(self, rel: Optional[str]) -> bool:
        if not rel:
            return False
        return str(rel).upper() in self.rel_whitelist

    def normalize_relation(self, rel: Optional[str], text: Optional[str] = None) -> str:
        rel_up = str(rel or "").upper().strip()
        if rel_up in self.rel_whitelist:
            return rel_up

        haystack = f"{rel or ''} {text or ''}".strip().lower()
        for canonical_rel, aliases in self.relation_aliases.items():
            if any(alias in haystack for alias in aliases):
                return canonical_rel
        return rel_up if rel_up else "UNK"

    def allowed_transition(self, source_layer: Optional[str], target_layer: Optional[str]) -> bool:
        src = str(source_layer or "UNK").upper()
        tgt = str(target_layer or "UNK").upper()
        if src == "UNK" or tgt == "UNK":
            return True
        if src not in self.ctp:
            return True
        return tgt in self.ctp[src]

    def layer_distance(self, source_layer: Optional[str], target_layer: Optional[str]) -> int:
        src = str(source_layer or "UNK").upper()
        tgt = str(target_layer or "UNK").upper()
        if src not in self.layer_order or tgt not in self.layer_order:
            return 0
        return max(0, self.layer_order.index(tgt) - self.layer_order.index(src))

    def count_missing_layers(self, layers: Sequence[str]) -> List[str]:
        if not self.layer_order:
            return []
        normalized = [str(x).upper() for x in layers]
        seen = set(normalized)
        wanted = [x for x in self.layer_order if x in seen]
        if not wanted:
            return []
        start_idx = self.layer_order.index(wanted[0])
        end_idx = self.layer_order.index(wanted[-1])
        segment = self.layer_order[start_idx : end_idx + 1]
        return [layer for layer in segment if layer not in seen]

    def find_trigger_words(self, text: Optional[str]) -> List[str]:
        text = (text or "").lower()
        hits: List[str] = []
        for words in self.trigger_words.values():
            for word in words:
                if word and word in text:
                    hits.append(word)
        return sorted(set(hits))

    def has_trigger(self, text: Optional[str]) -> bool:
        return len(self.find_trigger_words(text)) > 0

    def as_beam_params(self) -> Dict[str, Any]:
        return dict(self.config.get("beam_search") or {})

    def as_evidence_params(self) -> Dict[str, Any]:
        return dict(self.config.get("evidence_gate") or {})


def normalize_name(text: Optional[str]) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

```


## backend/src/causal_jointlk/instance_kg_builder.py
```python

import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .evidence_gate import choose_best_evidence
from .prior import CausalPrior
from .schemas import CausalEdge, CausalNode


class InstanceCausalKGBuilder:
    """将普通抽取结果后处理成“因果 Instance-KG”。

    输入可以是：
    - 已经抽出来的 node/edge dict
    - EvidenceStore 中的 evidence_units

    输出：
    - 仅保留/规范化因果关系白名单
    - 在边上补充 doc_id / kg_scope / kg_id / evidence_unit_id / span
    """

    def __init__(self, graph: Any, prior: CausalPrior, evidence_store: Any):
        self.graph = graph
        self.prior = prior
        self.evidence_store = evidence_store

    def canonicalize_edges(
        self,
        nodes: Sequence[CausalNode],
        raw_edges: Sequence[Dict[str, Any]],
        evidence_units: Sequence[Dict[str, Any]],
        doc_id: Optional[str],
        kg_scope: str = "instance",
        kg_id: Optional[str] = None,
    ) -> List[CausalEdge]:
        node_map = {n.node_id: n for n in nodes}
        out: List[CausalEdge] = []
        for row in raw_edges:
            source_id = row.get("source_id")
            target_id = row.get("target_id")
            if source_id not in node_map or target_id not in node_map:
                continue

            raw_relation = row.get("relation") or row.get("type") or row.get("rel_type")
            edge_text = row.get("text") or row.get("evidence_text") or ""
            relation = self.prior.normalize_relation(raw_relation, edge_text)
            if not self.prior.is_relation_allowed(relation):
                continue

            source_node = node_map[source_id]
            target_node = node_map[target_id]
            best = choose_best_evidence(
                prior=self.prior,
                source_text=source_node.text,
                target_text=target_node.text,
                relation=relation,
                evidence_units=evidence_units,
                relation_text=edge_text,
            )
            edge = CausalEdge(
                edge_id=row.get("edge_id") or str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                source_text=source_node.text,
                target_text=target_node.text,
                source_layer=source_node.layer,
                target_layer=target_node.layer,
                evidence_unit_id=best.get("unit_id"),
                evidence_text=best.get("content"),
                doc_id=doc_id,
                kg_scope=kg_scope,
                kg_id=kg_id,
                supported=bool(best.get("supported")),
                support_score=float(best.get("support_score") or 0.0),
                meta={
                    "source_span": best.get("source_span"),
                    "target_span": best.get("target_span"),
                    "trigger_hits": best.get("trigger_hits") or [],
                },
            )
            out.append(edge)
        return out

    def persist_edges(self, edges: Sequence[CausalEdge]) -> None:
        for edge in edges:
            cypher = f"""
            MATCH (s) WHERE elementId(s) = $source_id
            MATCH (t) WHERE elementId(t) = $target_id
            MERGE (s)-[r:{edge.relation} {{
                doc_id: $doc_id,
                kg_scope: $kg_scope,
                kg_id: $kg_id,
                evidence_unit_id: $evidence_unit_id
            }}]->(t)
            SET r.evidence_text = $evidence_text,
                r.source_span = $source_span,
                r.target_span = $target_span,
                r.support_score = $support_score,
                r.supported = $supported,
                r.trigger_hits = $trigger_hits
            """
            self.graph.query(
                cypher,
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "doc_id": edge.doc_id,
                    "kg_scope": edge.kg_scope,
                    "kg_id": edge.kg_id,
                    "evidence_unit_id": edge.evidence_unit_id,
                    "evidence_text": edge.evidence_text,
                    "source_span": edge.meta.get("source_span") if edge.meta else None,
                    "target_span": edge.meta.get("target_span") if edge.meta else None,
                    "support_score": edge.support_score,
                    "supported": edge.supported,
                    "trigger_hits": edge.meta.get("trigger_hits") if edge.meta else [],
                },
            )

```


## backend/src/causal_jointlk/neo4j_accessor.py
```python

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .prior import CausalPrior
from .schemas import CausalEdge, CausalNode


class InstanceKGAccessor:
    """Queries only Instance-KG, never BG-KG."""

    def __init__(self, graph: Any):
        self.graph = graph

    def find_seed_nodes(
        self,
        query_text: str,
        doc_id: Optional[str] = None,
        kg_scope: Optional[str] = "instance",
        kg_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[CausalNode]:
        cypher = """
        MATCH (n)
        WHERE NOT n:Chunk AND NOT n:Document
          AND ($doc_id IS NULL OR coalesce(n.doc_id, '') = $doc_id)
          AND ($kg_scope IS NULL OR coalesce(n.kg_scope, '') = $kg_scope)
          AND ($kg_id IS NULL OR coalesce(n.kg_id, '') = $kg_id)
          AND (
                toLower(coalesce(n.name, '')) CONTAINS toLower($query_text)
             OR toLower(coalesce(n.id, '')) CONTAINS toLower($query_text)
             OR toLower(coalesce(n.text, '')) CONTAINS toLower($query_text)
          )
        RETURN elementId(n) AS node_id,
               labels(n) AS labels,
               properties(n) AS props
        LIMIT $limit
        """
        rows = self.graph.query(
            cypher,
            {
                "query_text": query_text,
                "doc_id": doc_id,
                "kg_scope": kg_scope,
                "kg_id": kg_id,
                "limit": int(limit),
            },
        )
        return [self._row_to_node(x) for x in rows]

    def get_k_hop_subgraph(
        self,
        seed_node_ids: Sequence[str],
        prior: CausalPrior,
        k_hop: int = 2,
        doc_id: Optional[str] = None,
        kg_scope: Optional[str] = "instance",
        kg_id: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        if not seed_node_ids:
            return {"nodes": [], "edges": []}

        rel_types = list(prior.rel_whitelist)
        if not rel_types:
            return {"nodes": [], "edges": []}

        cypher = """
        MATCH (s)
        WHERE elementId(s) IN $seed_node_ids
        CALL apoc.path.subgraphAll(
          s,
          {
            maxLevel: $k_hop,
            relationshipFilter: $relationship_filter,
            labelFilter: '-Chunk|-Document'
          }
        ) YIELD nodes, relationships
        WITH apoc.coll.toSet(apoc.coll.flatten(collect(nodes))) AS ns,
             apoc.coll.toSet(apoc.coll.flatten(collect(relationships))) AS rs
        UNWIND ns AS n
        WITH collect(DISTINCT n) AS nodes, rs
        UNWIND rs AS r
        WITH nodes, collect(DISTINCT r) AS relationships
        RETURN
          [n IN nodes
             WHERE ($doc_id IS NULL OR coalesce(n.doc_id, '') = $doc_id)
               AND ($kg_scope IS NULL OR coalesce(n.kg_scope, '') = $kg_scope)
               AND ($kg_id IS NULL OR coalesce(n.kg_id, '') = $kg_id)
           | {
               node_id: elementId(n),
               labels: labels(n),
               props: properties(n)
             }] AS nodes,
          [r IN relationships
             WHERE type(r) IN $rel_types
               AND ($doc_id IS NULL OR coalesce(r.doc_id, '') = $doc_id)
               AND ($kg_scope IS NULL OR coalesce(r.kg_scope, '') = $kg_scope)
               AND ($kg_id IS NULL OR coalesce(r.kg_id, '') = $kg_id)
           | {
               edge_id: elementId(r),
               rel_type: type(r),
               props: properties(r),
               source_id: elementId(startNode(r)),
               target_id: elementId(endNode(r))
             }] AS edges
        """
        rows = self.graph.query(
            cypher,
            {
                "seed_node_ids": list(seed_node_ids),
                "k_hop": int(k_hop),
                "relationship_filter": "|".join([f"{rel}>|<{rel}" for rel in rel_types]),
                "rel_types": rel_types,
                "doc_id": doc_id,
                "kg_scope": kg_scope,
                "kg_id": kg_id,
            },
        )
        if not rows:
            return {"nodes": [], "edges": []}
        row = rows[0]
        nodes = [self._row_to_node(x) for x in row.get("nodes", [])]
        node_map = {n.node_id: n for n in nodes}
        edges = [self._row_to_edge(x, node_map) for x in row.get("edges", [])]
        return {"nodes": nodes, "edges": edges}

    def hydrate_edge_evidence(
        self,
        evidence_store: Any,
        edges: Sequence[CausalEdge],
    ) -> List[CausalEdge]:
        hydrated: List[CausalEdge] = []
        for edge in edges:
            evidence_text = edge.evidence_text
            unit_id = edge.evidence_unit_id or edge.meta.get("evidence_unit_id")
            if not evidence_text and unit_id:
                unit = evidence_store.get_unit(unit_id)
                if unit is not None:
                    edge.evidence_text = unit.content
            hydrated.append(edge)
        return hydrated

    @staticmethod
    def _row_to_node(row: Dict[str, Any]) -> CausalNode:
        props = dict(row.get("props") or {})
        text = props.get("name") or props.get("text") or props.get("id") or row.get("node_id")
        layer = props.get("layer") or props.get("ctp_layer") or "UNK"
        return CausalNode(
            node_id=row.get("node_id"),
            text=text,
            layer=str(layer).upper(),
            doc_id=props.get("doc_id"),
            kg_scope=props.get("kg_scope"),
            kg_id=props.get("kg_id"),
            labels=list(row.get("labels") or []),
            properties=props,
        )

    @staticmethod
    def _row_to_edge(row: Dict[str, Any], node_map: Dict[str, CausalNode]) -> CausalEdge:
        props = dict(row.get("props") or {})
        source = node_map.get(row.get("source_id"))
        target = node_map.get(row.get("target_id"))
        return CausalEdge(
            edge_id=row.get("edge_id"),
            source_id=row.get("source_id"),
            target_id=row.get("target_id"),
            relation=str(row.get("rel_type") or "UNK").upper(),
            source_text=source.text if source else row.get("source_id"),
            target_text=target.text if target else row.get("target_id"),
            source_layer=source.layer if source else "UNK",
            target_layer=target.layer if target else "UNK",
            evidence_unit_id=props.get("evidence_unit_id"),
            evidence_text=props.get("evidence_text"),
            doc_id=props.get("doc_id"),
            kg_scope=props.get("kg_scope"),
            kg_id=props.get("kg_id"),
            meta=props,
        )

```


## backend/src/causal_jointlk/evidence_gate.py
```python

import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .prior import CausalPrior, normalize_name


_WORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_\-]+")


def _tokens(text: Optional[str]) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def lexical_overlap(a: Optional[str], b: Optional[str]) -> float:
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def find_best_span(text: Optional[str], phrase: Optional[str]) -> Optional[Tuple[int, int]]:
    text = text or ""
    phrase = phrase or ""
    if not text or not phrase:
        return None

    idx = text.lower().find(phrase.lower())
    if idx >= 0:
        return (idx, idx + len(phrase))

    # fallback: rough fuzzy span
    best_score = 0.0
    best_span = None
    window = max(len(phrase), 4)
    for start in range(0, max(len(text) - window + 1, 1)):
        segment = text[start : start + window]
        score = SequenceMatcher(None, segment.lower(), phrase.lower()).ratio()
        if score > best_score:
            best_score = score
            best_span = (start, start + window)
    return best_span if best_score >= 0.70 else None


def score_edge_support(
    prior: CausalPrior,
    source_text: str,
    target_text: str,
    relation: str,
    evidence_text: Optional[str],
    relation_text: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = prior.as_evidence_params()
    evidence_text = evidence_text or ""
    source_text = source_text or ""
    target_text = target_text or ""
    relation = relation or "UNK"

    source_span = find_best_span(evidence_text, source_text)
    target_span = find_best_span(evidence_text, target_text)
    source_hit = source_span is not None
    target_hit = target_span is not None

    trigger_hits = prior.find_trigger_words(evidence_text)
    rel_match = relation.upper() in (relation_text or "").upper() or relation.upper() in evidence_text.upper()
    overlap = max(
        lexical_overlap(f"{source_text} {target_text}", evidence_text),
        lexical_overlap(f"{source_text} {relation} {target_text}", evidence_text),
    )

    score = 0.0
    score += min(overlap, 0.35)
    if source_hit:
        score += 0.20
    if target_hit:
        score += 0.20
    if source_hit and target_hit:
        score += float(cfg.get("two_sided_entity_bonus", 0.20))
    if trigger_hits:
        score += float(cfg.get("trigger_bonus", 0.25))
    if rel_match:
        score += 0.10

    ordered_bonus = float(cfg.get("ordered_bonus", 0.10))
    if source_span and target_span and source_span[0] <= target_span[0]:
        score += ordered_bonus

    negative_cues = []
    for xs in (cfg.get("negative_cues") or {}).values():
        negative_cues.extend(xs)
    if any(cue.lower() in evidence_text.lower() for cue in negative_cues):
        score -= 0.25

    require_trigger_or_reltype = bool(cfg.get("require_trigger_or_reltype", True))
    min_overlap = float(cfg.get("min_lexical_overlap", 0.18))
    min_support = float(cfg.get("min_support_score", 0.55))

    supported = (
        overlap >= min_overlap
        and source_hit
        and target_hit
        and (not require_trigger_or_reltype or bool(trigger_hits or rel_match))
        and score >= min_support
    )

    return {
        "supported": supported,
        "support_score": max(0.0, min(score, 1.0)),
        "trigger_hits": trigger_hits,
        "source_span": source_span,
        "target_span": target_span,
    }


def choose_best_evidence(
    prior: CausalPrior,
    source_text: str,
    target_text: str,
    relation: str,
    evidence_units: Sequence[Dict[str, Any]],
    relation_text: Optional[str] = None,
) -> Dict[str, Any]:
    best = {
        "supported": False,
        "support_score": 0.0,
        "unit_id": None,
        "content": None,
        "source_span": None,
        "target_span": None,
        "trigger_hits": [],
    }
    for unit in evidence_units:
        current = score_edge_support(
            prior=prior,
            source_text=source_text,
            target_text=target_text,
            relation=relation,
            evidence_text=unit.get("content"),
            relation_text=relation_text,
        )
        if current["support_score"] > best["support_score"]:
            best = {
                **current,
                "unit_id": unit.get("unit_id"),
                "content": unit.get("content"),
            }
    return best

```


## backend/src/causal_jointlk/baseline_extractor.py
```python

from typing import Any, Dict, Iterable, List, Sequence

from .evidence_gate import choose_best_evidence
from .prior import CausalPrior
from .schemas import CausalEdge


class BaselineCausalExtractor:
    """Rule-based / prior-based local edge scorer.

    这个 baseline 很适合做论文实验中的“普通抽取方式”：
    - 只用关系白名单
    - 只用 CTP 约束
    - 只用 evidence gate
    - 不使用神经网络 reranking
    """

    def __init__(self, prior: CausalPrior):
        self.prior = prior

    def score_edges(
        self,
        edges: Sequence[CausalEdge],
        evidence_by_edge_id: Dict[str, List[Dict[str, Any]]],
    ) -> List[CausalEdge]:
        out: List[CausalEdge] = []
        for edge in edges:
            relation = self.prior.normalize_relation(edge.relation, edge.evidence_text)
            edge.relation = relation

            if not self.prior.is_relation_allowed(relation):
                edge.score = -1.0
                out.append(edge)
                continue

            score = 0.0
            if self.prior.allowed_transition(edge.source_layer, edge.target_layer):
                score += 0.30
            else:
                score -= 0.50

            distance = self.prior.layer_distance(edge.source_layer, edge.target_layer)
            score -= 0.05 * max(distance - 1, 0)

            candidate_units = evidence_by_edge_id.get(edge.edge_id) or []
            if edge.evidence_text:
                candidate_units = [{"unit_id": edge.evidence_unit_id, "content": edge.evidence_text}] + candidate_units

            best = choose_best_evidence(
                prior=self.prior,
                source_text=edge.source_text,
                target_text=edge.target_text,
                relation=relation,
                evidence_units=candidate_units,
                relation_text=edge.relation,
            )
            edge.evidence_unit_id = best.get("unit_id") or edge.evidence_unit_id
            edge.evidence_text = best.get("content") or edge.evidence_text
            edge.supported = bool(best.get("supported"))
            edge.support_score = float(best.get("support_score") or 0.0)
            edge.meta = dict(edge.meta or {})
            edge.meta["trigger_hits"] = best.get("trigger_hits") or []
            edge.meta["source_span"] = best.get("source_span")
            edge.meta["target_span"] = best.get("target_span")

            score += edge.support_score
            if edge.supported:
                score += 0.25

            edge.score = score
            out.append(edge)
        return out

```


## backend/src/causal_jointlk/beam_search.py
```python

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

        beam: List[Tuple[float, List[str], List[CausalEdge]]] = []
        for seed in seed_node_ids:
            heapq.heappush(beam, (-0.0, [seed], []))

        finished: List[CandidateChain] = []
        while beam and len(finished) < max(top_k * 5, top_k):
            neg_score, node_path, edge_path = heapq.heappop(beam)
            current_score = -neg_score
            current_node = node_path[-1]

            if edge_path:
                last_edge = edge_path[-1]
                should_finish = target_node_id is not None and current_node == target_node_id
                should_finish = should_finish or (
                    target_node_id is None and str(last_edge.target_layer).upper() == "OUTCOME"
                )
                if should_finish:
                    finished.append(self._build_chain(node_path, edge_path, current_score))

            if len(edge_path) >= max_hops:
                continue

            for next_edge in adjacency.get(current_node, []):
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

                hop_penalty = float(self.params.get("hop_penalty", 0.25))

                delta = next_edge.score - hop_penalty - transition_penalty - unsupported_penalty - duplicate_penalty
                new_node_path = node_path + [next_edge.target_id]
                new_edge_path = edge_path + [next_edge]
                new_score = current_score + delta
                heapq.heappush(beam, (-new_score, new_node_path, new_edge_path))

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
    ) -> CandidateChain:
        layers = [edge.source_layer for edge in edge_path] + ([edge_path[-1].target_layer] if edge_path else [])
        missing_layers = self.prior.count_missing_layers(layers)
        score -= float(self.params.get("missing_layer_penalty", 0.50)) * len(missing_layers)

        unsupported_edges = [edge.edge_id for edge in edge_path if not edge.supported]
        chain_id = str(uuid.uuid4())
        return CandidateChain(
            chain_id=chain_id,
            nodes=node_path,
            edges=list(edge_path),
            score=score,
            missing_layers=missing_layers,
            unsupported_edges=unsupported_edges,
        )

```


## backend/src/causal_jointlk/jointlk_edge_scorer.py
```python

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

        out: List[CausalEdge] = []
        for edge, p in zip(valid_edges, probs):
            edge.score = float(p)
            edge.support_score = float(p)
            edge.supported = bool(p >= 0.5)
            out.append(edge)
        return out

```


## backend/src/causal_jointlk/service.py
```python

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

```


## PATCH_PLAN.md

# causal_jointlk patch bundle

这个补丁的目标是把你仓库里现有的“硬接 QA JointLK”方式，重构成一个**面向因果链抽取**的、**可训练**、**可插拔**、**可与普通抽取方式直接对比**的模块。

## 设计原则

1. **只在 Instance-KG 上工作**  
   不依赖 BG-KG / ConceptNet 作为推理图；图只来自你自己的报告实例图谱。
2. **JointLK-style，而不是 QA-style**  
   不是把 QA 的 `num_choice` 强行套用到因果链任务，而是改造成**边支持打分器**。
3. **先做边打分，再做全局组链**  
   Local edge score + global beam search，是最适合论文 ablation 的结构。
4. **普通抽取方式保留为 baseline**  
   baseline 和 jointlk 共享相同候选图与 beam search，唯一差别是 local scorer。
5. **证据闸门单独保留**  
   神经网络分数和 evidence gate 解耦，方便做四种对照：
   - baseline
   - baseline + evidence gate
   - jointlk
   - jointlk + evidence gate

## 文件列表

### 新增
- `configs/causal_prior.yaml`
- `modeling/causal_jointlk_io.py`
- `modeling/modeling_causal_jointlk.py`
- `experiments/causal_jointlk/dataset.py`
- `experiments/causal_jointlk/metrics.py`
- `experiments/causal_jointlk/train_causal_jointlk.py`
- `experiments/causal_jointlk/eval_causal_jointlk.py`
- `backend/src/causal_jointlk/__init__.py`
- `backend/src/causal_jointlk/schemas.py`
- `backend/src/causal_jointlk/prior.py`
- `backend/src/causal_jointlk/instance_kg_builder.py`
- `backend/src/causal_jointlk/neo4j_accessor.py`
- `backend/src/causal_jointlk/evidence_gate.py`
- `backend/src/causal_jointlk/baseline_extractor.py`
- `backend/src/causal_jointlk/beam_search.py`
- `backend/src/causal_jointlk/jointlk_edge_scorer.py`
- `backend/src/causal_jointlk/service.py`

### 建议保留但不再继续扩展
- `backend/src/jointlk_integration/*`  
  这套更像“QA 模型接 Neo4j demo”，不适合继续演化成因果链抽取主线。

## 论文实验推荐指标

### 1. 候选子图覆盖
- Gold edge coverage
- Gold node coverage

### 2. 边级别抽取
- Edge Precision / Recall / F1
- Relation micro-F1 / macro-F1
- AUROC / AUPR（support score）

### 3. 链级别抽取
- Chain Exact Match
- Chain Edge-F1
- Chain Node-F1
- MRR / Hits@K / nDCG@K（Top-K chain ranking）

### 4. 证据一致性
- Evidence Support Rate
- Unsupported Edge Rate（越低越好）

### 5. 结构合理性
- Layer Violation Rate
- Missing Layer Count

## 建议的主表

| 方法 | Edge-F1 | Chain Edge-F1 | Chain EM | MRR | Evidence Support Rate | Layer Violation Rate |
|---|---:|---:|---:|---:|---:|---:|
| 普通抽取 |  |  |  |  |  |  |
| 普通抽取 + 闸门 |  |  |  |  |  |  |
| JointLK-style |  |  |  |  |  |  |
| JointLK-style + 闸门 |  |  |  |  |  |  |

## 运行建议

### 训练
```bash
python experiments/causal_jointlk/train_causal_jointlk.py \
  --train_jsonl data/causal/train.jsonl \
  --dev_jsonl data/causal/dev.jsonl \
  --prior_config configs/causal_prior.yaml \
  --output_dir saved_models/causal_jointlk
```

### 评估
```bash
python experiments/causal_jointlk/eval_causal_jointlk.py \
  --test_jsonl data/causal/test.jsonl \
  --checkpoint saved_models/causal_jointlk/best_model.pt \
  --prior_config configs/causal_prior.yaml \
  --output_json saved_models/causal_jointlk/test_eval.json
```

## 在线服务用法（示意）

```python
from src.causal_jointlk.service import CausalJointLKService
from src.evidence_store.sqlite_store import EvidenceStore

service = CausalJointLKService(
    graph=neo4j_graph,
    evidence_store=EvidenceStore(),
    prior_config_path="configs/causal_prior.yaml",
    jointlk_checkpoint_path="saved_models/causal_jointlk/best_model.pt",
)

result = service.extract(
    query="桥梁坍塌的致因链",
    target_text="桥梁坍塌",
    doc_id="doc-001",
    mode="jointlk",
    k_hop=2,
    top_k=5,
    persist=True,
)
print(result.to_dict())
```

