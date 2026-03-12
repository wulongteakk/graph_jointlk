

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
