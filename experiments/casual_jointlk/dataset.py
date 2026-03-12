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
