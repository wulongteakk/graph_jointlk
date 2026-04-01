from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    "RISK": 11,
    "CONSEQUENCE": 12,
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
    rels = ["UNK"] + [str(x).upper() for x in whitelist]
    return {rel: idx for idx, rel in enumerate(rels)}


def build_node_type_to_id(prior_config: Dict[str, Any]) -> Dict[str, int]:
    ctp = prior_config.get("ctp_allowed_transitions", {})
    node_type_to_id = dict(DEFAULT_NODE_TYPE_TO_ID)
    for key in ctp:
        key_u = str(key).upper()
        if key_u not in node_type_to_id:
            node_type_to_id[key_u] = len(node_type_to_id)
        for value in ctp[key]:
            value_u = str(value).upper()
            if value_u not in node_type_to_id:
                node_type_to_id[value_u] = len(node_type_to_id)
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

    cleaned_evidence = [normalize_text(x) for x in evidence_texts if normalize_text(x)]
    cleaned_evidence = cleaned_evidence[:max_evidence]

    parts: List[str] = []
    if doc_title:
        parts.append(f"[DOC] {doc_title}")
    if query:
        parts.append(f"[QUERY] {query}")
    parts.append(f"[SOURCE] {source_text}")
    parts.append(f"[RELATION] {relation_text}")
    parts.append(f"[TARGET] {target_text}")
    if cleaned_evidence:
        parts.append("[EVIDENCE] " + " [SEP] ".join(cleaned_evidence))
    return " ".join(parts)


def _ensure_non_empty_edges(
    edge_index: List[List[int]],
    num_nodes: int,
    edge_type_ids: Optional[List[int]] = None,
) -> Tuple[List[List[int]], List[int]]:
    edge_type_ids = list(edge_type_ids or [])
    if edge_index and len(edge_index) == 2 and len(edge_index[0]) > 0:
        if len(edge_type_ids) < len(edge_index[0]):
            edge_type_ids.extend([0] * (len(edge_index[0]) - len(edge_type_ids)))
        return edge_index, edge_type_ids

    srcs = list(range(num_nodes))
    dsts = list(range(num_nodes))
    rels = [0 for _ in range(num_nodes)]
    return [srcs, dsts], rels


def normalize_edge_index(edge_index: Any) -> List[List[int]]:
    """
    支持三种输入：
    1) [[0,1,2],[1,2,3]]
    2) [[0,1],[1,2],[2,3]]
    3) {"src": [...], "dst": [...]} / {"rows": [...], "cols": [...]}
    """
    if edge_index is None:
        return [[], []]

    if isinstance(edge_index, dict):
        src = edge_index.get("src") or edge_index.get("rows") or []
        dst = edge_index.get("dst") or edge_index.get("cols") or []
        return [[int(x) for x in src], [int(x) for x in dst]]

    if not isinstance(edge_index, list):
        return [[], []]

    if len(edge_index) == 2 and all(isinstance(part, list) for part in edge_index):
        # Already COO-like: [[srcs], [dsts]]
        if all(not part or isinstance(part[0], int) for part in edge_index):
            return [
                [int(x) for x in edge_index[0]],
                [int(x) for x in edge_index[1]],
            ]

    # Pair list: [[src, dst], ...]
    srcs: List[int] = []
    dsts: List[int] = []
    for item in edge_index:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            srcs.append(int(item[0]))
            dsts.append(int(item[1]))
    return [srcs, dsts]


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
        [str(ex["text"]) for ex in examples],
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
    causal_labels: List[float] = []
    enable_labels: List[float] = []
    dir_labels: List[float] = []
    temp_labels: List[float] = []
    src_first_labels: List[float] = []
    dst_first_labels: List[float] = []
    causal_mask: List[float] = []
    enable_mask: List[float] = []
    dir_mask: List[float] = []
    temp_mask: List[float] = []
    src_first_mask: List[float] = []
    dst_first_mask: List[float] = []
    sample_ids: List[str] = []
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_types: List[int] = []
    sample_weights: List[float] = []
    twin_group_ids: List[str] = []
    cf_roles: List[int] = []
    meta_rows: List[Dict[str, Any]] = []

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

        # Fallback: infer indices from node_ids and source/target node ids.
        node_ids = list(ex.get("node_ids") or [])
        if node_ids:
            source_node_id = ex.get("source_node_id")
            target_node_id = ex.get("target_node_id")
            if source_node_id in node_ids:
                local_source_idx = int(node_ids.index(source_node_id))
            if target_node_id in node_ids:
                local_target_idx = int(node_ids.index(target_node_id))

        flat_node_texts.extend([normalize_text(str(x)) or "<EMPTY>" for x in node_texts])
        flat_node_type_ids.extend([node_type_to_id.get(str(x).upper(), 0) for x in node_layers])
        flat_node_scores.extend([float(x) for x in node_scores])
        flat_graph_batch.extend([graph_idx] * local_num_nodes)

        global_source_indices.append(node_offset + local_source_idx)
        global_target_indices.append(node_offset + local_target_idx)

        rel_name = str(ex.get("candidate_relation") or ex.get("relation_text") or "UNK").upper()
        gold_rel_name = str(ex.get("gold_relation") or rel_name).upper()
        relation_ids.append(relation_to_id.get(rel_name, 0))
        relation_labels.append(relation_to_id.get(gold_rel_name, 0))
        labels.append(float(ex.get("label", 0.0)))
        c_label = float(ex.get("causal_labels", ex.get("silver_edge_causal", ex.get("label", -1))))
        e_label = float(ex.get("enable_labels", ex.get("silver_edge_enable", -1)))
        d_label = float(ex.get("dir_labels", ex.get("silver_causal_dir", -1)))
        t_label = float(ex.get("temp_labels", ex.get("silver_temporal_before", -1)))
        s_label = float(ex.get("src_first_labels", ex.get("silver_node_first_src", -1)))
        d2_label = float(ex.get("dst_first_labels", ex.get("silver_node_first_dst", -1)))
        causal_labels.append(max(0.0, c_label))
        enable_labels.append(max(0.0, e_label))
        dir_labels.append(max(0.0, d_label))
        temp_labels.append(max(0.0, t_label))
        src_first_labels.append(max(0.0, s_label))
        dst_first_labels.append(max(0.0, d2_label))
        causal_mask.append(float(ex.get("causal_mask", 0 if c_label < 0 else 1)))
        enable_mask.append(float(ex.get("enable_mask", 0 if e_label < 0 else 1)))
        dir_mask.append(float(ex.get("dir_mask", 0 if d_label < 0 else 1)))
        temp_mask.append(float(ex.get("temp_mask", 0 if t_label < 0 else 1)))
        src_first_mask.append(float(ex.get("src_first_mask", 0 if s_label < 0 else 1)))
        dst_first_mask.append(float(ex.get("dst_first_mask", 0 if d2_label < 0 else 1)))

        sample_ids.append(str(ex.get("sample_id") or f"sample-{graph_idx}"))
        sample_weights.append(float(ex.get("sample_weight", 1.0)))
        twin_group_id = str(ex.get("twin_group_id") or "")
        cf_role_raw = ex.get("cf_role", "none")
        if isinstance(cf_role_raw, str):
            cf_role_norm = cf_role_raw.strip().lower()
            if cf_role_norm == "positive":
                cf_role = 1
            elif cf_role_norm == "negative":
                cf_role = 0
            else:
                cf_role = -1
        else:
            try:
                cf_role = int(cf_role_raw)
            except Exception:
                cf_role = -1
        twin_group_ids.append(twin_group_id)
        cf_roles.append(cf_role)

        ex_edge_index = normalize_edge_index(ex.get("edge_index"))
        ex_edge_type = ex.get("edge_types") or ex.get("edge_type_labels") or []
        ex_edge_type_ids = [relation_to_id.get(str(rel).upper(), 0) for rel in ex_edge_type]
        ex_edge_index, ex_edge_type_ids = _ensure_non_empty_edges(ex_edge_index, local_num_nodes, ex_edge_type_ids)
        for src, dst, rel_id in zip(ex_edge_index[0], ex_edge_index[1], ex_edge_type_ids):
            edge_src.append(node_offset + int(src))
            edge_dst.append(node_offset + int(dst))
            edge_types.append(int(rel_id))

        meta_rows.append(
            {
                "sample_id": sample_ids[-1],
                "doc_id": ex.get("doc_id"),
                "file_name": ex.get("file_name") or ex.get("doc_title"),
                "doc_title": ex.get("doc_title") or ex.get("file_name"),
                "label_source": ex.get("label_source"),
                "review_status": ex.get("review_status"),
                "candidate_relation": rel_name,
                "gold_relation": gold_rel_name,
                "source_node_id": ex.get("source_node_id"),
                "source_text": ex.get("source_text"),
                "target_node_id": ex.get("target_node_id"),
                "target_text": ex.get("target_text"),
                "pseudo_label_id": ex.get("pseudo_label_id"),
                "label_confidence": ex.get("label_confidence"),
                "gold_label": int(ex.get("label", 0)),
                "task_masks": ex.get("task_masks"),
                "module_id": ex.get("module_id"),
                "twin_group_id": twin_group_id,
                "cf_role": cf_role,
            }
        )

        node_offset += local_num_nodes

    node_inputs = tokenizer(
        flat_node_texts,
        padding=True,
        truncation=True,
        max_length=max_node_text_length,
        return_tensors="pt",
    )

    batch: Dict[str, Any] = {
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
        "causal_labels": torch.tensor(causal_labels, dtype=torch.float),
        "enable_labels": torch.tensor(enable_labels, dtype=torch.float),
        "dir_labels": torch.tensor(dir_labels, dtype=torch.float),
        "temp_labels": torch.tensor(temp_labels, dtype=torch.float),
        "src_first_labels": torch.tensor(src_first_labels, dtype=torch.float),
        "dst_first_labels": torch.tensor(dst_first_labels, dtype=torch.float),
        "causal_mask": torch.tensor(causal_mask, dtype=torch.float),
        "enable_mask": torch.tensor(enable_mask, dtype=torch.float),
        "dir_mask": torch.tensor(dir_mask, dtype=torch.float),
        "temp_mask": torch.tensor(temp_mask, dtype=torch.float),
        "src_first_mask": torch.tensor(src_first_mask, dtype=torch.float),
        "dst_first_mask": torch.tensor(dst_first_mask, dtype=torch.float),
        "sample_weights": torch.tensor(sample_weights, dtype=torch.float),
        "cf_roles": torch.tensor(cf_roles, dtype=torch.long),
        "twin_group_ids": twin_group_ids,
        "sample_ids": sample_ids,
        "meta_rows": meta_rows,
    }

    if "token_type_ids" in text_inputs:
        batch["token_type_ids"] = text_inputs["token_type_ids"]
    if "token_type_ids" in node_inputs:
        batch["node_token_type_ids"] = node_inputs["token_type_ids"]

    if device is not None:
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)

    return batch


def sigmoid_to_labels(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (prob >= threshold).long()


def detach_to_list(x: torch.Tensor) -> List[Any]:
    return x.detach().cpu().tolist()