from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import Dataset

from modeling.causal_jointlk_io import (
    build_example_text,
    build_node_type_to_id,
    build_relation_to_id,
    load_prior_config,
)


DEFAULT_LABEL_WEIGHTS = {
    "gold_chain": 1.00,
    "pseudo_review_edited": 1.00,
    "pseudo_review_accepted": 0.95,
    "pseudo_pending": 0.80,
}


class CausalEdgeDataset(Dataset):
    """JSONL dataset for final causal JointLK training.

    兼容最终 schema，核心字段包括：
    {
      "sample_id": "...",
      "doc_id": "...",
      "file_name": "...",
      "source_node_id": "...",
      "source_text": "...",
      "target_node_id": "...",
      "target_text": "...",
      "candidate_relation": "CAUSES",
      "gold_relation": "CAUSES",
      "label": 1,
      "label_source": "gold_chain|pseudo_review_accepted|pseudo_pending",
      "review_status": "pending|accepted|edited|rejected",
      "label_confidence": 0.97,
      "evidence_texts": [...],
      "node_ids": [...],
      "node_texts": [...],
      "node_layer_types": [...],
      "edge_index": [[0,1], [1,2], ...]   # 或 [[srcs], [dsts]]
      "edge_types": [...],
      "source_idx": 0,
      "target_idx": 1
    }
    """

    def __init__(
        self,
        jsonl_path: str,
        prior_config_path: str,
        *,
        label_weight_map: Optional[Dict[str, float]] = None,
        max_evidence: int = 3,
        include_label_sources: Optional[Sequence[str]] = None,
        exclude_label_sources: Optional[Sequence[str]] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.prior_config = load_prior_config(prior_config_path)
        self.relation_to_id = build_relation_to_id(self.prior_config)
        self.node_type_to_id = build_node_type_to_id(self.prior_config)
        self.label_weight_map = {**DEFAULT_LABEL_WEIGHTS, **(label_weight_map or {})}
        self.max_evidence = int(max_evidence)
        self.include_label_sources = set(include_label_sources or [])
        self.exclude_label_sources = set(exclude_label_sources or [])
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
                label_source = str(obj.get("label_source") or "unknown")
                if self.include_label_sources and label_source not in self.include_label_sources:
                    continue
                if self.exclude_label_sources and label_source in self.exclude_label_sources:
                    continue

                obj.setdefault("sample_id", f"{self.jsonl_path.stem}-{line_idx}")
                obj.setdefault("doc_title", obj.get("file_name") or obj.get("doc_id") or "")
                obj.setdefault("evidence_texts", [])
                obj.setdefault("node_texts", [])
                obj.setdefault("node_layer_types", [])
                obj.setdefault("node_scores", [1.0] * len(obj.get("node_texts") or []))

                source_text = obj.get("source_text") or _safe_pick_text(obj, obj.get("source_idx", 0), "source")
                target_text = obj.get("target_text") or _safe_pick_text(obj, obj.get("target_idx", 1), "target")
                relation_text = obj.get("candidate_relation") or obj.get("relation_text") or "UNK"

                obj["text"] = build_example_text(
                    query=obj.get("query"),
                    source_text=source_text,
                    relation_text=relation_text,
                    target_text=target_text,
                    evidence_texts=obj.get("evidence_texts") or [],
                    doc_title=obj.get("doc_title"),
                    max_evidence=self.max_evidence,
                )

                if "sample_weight" not in obj:
                    obj["sample_weight"] = self._derive_sample_weight(obj)

                rows.append(obj)
        return rows

    def _derive_sample_weight(self, record: Dict[str, Any]) -> float:
        label_source = str(record.get("label_source") or "unknown")
        confidence = record.get("label_confidence")
        base_weight = float(self.label_weight_map.get(label_source, 1.0))

        if confidence is not None:
            try:
                confidence_val = float(confidence)
                # 只对 pseudo 样本做轻量 confidence shrink/boost。
                if label_source != "gold_chain":
                    base_weight *= max(0.5, min(1.0, confidence_val))
            except Exception:
                pass

        review_status = str(record.get("review_status") or "").lower()
        if review_status == "edited":
            base_weight = max(base_weight, 1.0)
        elif review_status == "accepted":
            base_weight = max(base_weight, 0.95)
        elif review_status == "pending" and label_source.startswith("pseudo"):
            base_weight = min(base_weight, 0.85)

        return float(base_weight)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]



def _safe_pick_text(record: Dict[str, Any], idx: int, default_value: str) -> str:
    node_texts: Sequence[str] = record.get("node_texts") or []
    if node_texts and 0 <= int(idx) < len(node_texts):
        return str(node_texts[int(idx)])
    return default_value
