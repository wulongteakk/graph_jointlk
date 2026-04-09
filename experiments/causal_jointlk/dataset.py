from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import re
from torch.utils.data import Dataset

from modeling.causal_jointlk_io import (
    build_example_text,
    build_node_type_to_id,
    build_relation_to_id,
    load_prior_config,
)



REPO_ROOT = Path(__file__).resolve().parents[2]

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
      "label": 1,  # fallback 主任务
      "causal_labels": 1,
      "enable_labels": 0/1/-1,
      "dir_labels": 0/1/-1,
      "temp_labels": 0/1/-1,
      "src_first_labels": 0/1/-1,
      "dst_first_labels": 0/1/-1,
      "causal_mask": 0/1, ...
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
        self.jsonl_path = _resolve_input_path(jsonl_path)
        self.prior_config = load_prior_config(prior_config_path)
        self.relation_to_id = build_relation_to_id(self.prior_config)
        self.node_type_to_id = build_node_type_to_id(self.prior_config)
        self.label_weight_map = {**DEFAULT_LABEL_WEIGHTS, **(label_weight_map or {})}
        self.max_evidence = int(max_evidence)
        self.include_label_sources = set(include_label_sources or [])
        self.exclude_label_sources = set(exclude_label_sources or [])
        self.records = self._load_records()
        self.twin_index = self.build_twin_index()

    def _load_records(self) -> List[Dict[str, Any]]:
        if not _path_exists(self.jsonl_path):
            raise FileNotFoundError(
                "dataset not found: "
                f"{self.jsonl_path} (cwd={Path.cwd()}, repo_root={REPO_ROOT})"
            )

        rows: List[Dict[str, Any]] = []
        readable_path = _to_windows_long_path(self.jsonl_path)
        with readable_path.open("r", encoding="utf-8") as fin:
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

                if "sample_weight" not in obj:
                    obj["sample_weight"] = self._derive_sample_weight(obj)
                obj.setdefault("twin_group_id", obj.get("twin_group_id"))
                obj["cf_role"] = _normalize_cf_role(obj.get("cf_role", "none"))
                obj.setdefault("causal_labels", obj.get("silver_edge_causal", obj.get("label", -1)))
                obj.setdefault("enable_labels", obj.get("silver_edge_enable", -1))
                obj.setdefault("dir_labels", obj.get("silver_causal_dir", -1))
                obj.setdefault("temp_labels", obj.get("silver_temporal_before", -1))
                obj.setdefault("src_first_labels", obj.get("silver_node_first_src", -1))
                obj.setdefault("dst_first_labels", obj.get("silver_node_first_dst", -1))
                obj.setdefault("causal_mask", 0 if int(obj.get("causal_labels", -1)) < 0 else 1)
                obj.setdefault("enable_mask", 0 if int(obj.get("enable_labels", -1)) < 0 else 1)
                obj.setdefault("dir_mask", 0 if int(obj.get("dir_labels", -1)) < 0 else 1)
                obj.setdefault("temp_mask", 0 if int(obj.get("temp_labels", -1)) < 0 else 1)
                obj.setdefault("src_first_mask", 0 if int(obj.get("src_first_labels", -1)) < 0 else 1)
                obj.setdefault("dst_first_mask", 0 if int(obj.get("dst_first_labels", -1)) < 0 else 1)
                obj.setdefault("task_masks", {"causal": obj.get("causal_mask"), "enable": obj.get("enable_mask"), "dir": obj.get("dir_mask"), "temp": obj.get("temp_mask"), "src_first": obj.get("src_first_mask"), "dst_first": obj.get("dst_first_mask")})
                obj.setdefault("module_id", obj.get("module_id") or "construction_safety")

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

    def build_twin_index(self) -> Dict[str, Dict[str, List[int]]]:
        twin_index: Dict[str, Dict[str, List[int]]] = {}
        for idx, row in enumerate(self.records):
            twin_group_id = str(row.get("twin_group_id") or "").strip()
            if not twin_group_id:
                continue
            role = _normalize_cf_role(row.get("cf_role", "none"))
            if role == "none":
                continue
            slot = twin_index.setdefault(twin_group_id, {"positive": [], "negative": []})
            slot[role].append(idx)
        return twin_index

def _safe_pick_text(record: Dict[str, Any], idx: int, default_value: str) -> str:
    node_texts: Sequence[str] = record.get("node_texts") or []
    if node_texts and 0 <= int(idx) < len(node_texts):
        return str(node_texts[int(idx)])
    return default_value

def _normalize_cf_role(value: Any) -> str:
    if isinstance(value, str):
        role = value.strip().lower()
        if role in {"positive", "negative"}:
            return role
        return "none"
    try:
        role_int = int(value)
    except Exception:
        return "none"
    if role_int == 1:
        return "positive"
    if role_int == 0:
        return "negative"
    return "none"

def _resolve_input_path(path_str: str) -> Path:
    """
    兼容从任意工作目录启动训练脚本的相对路径。
    优先级：
    1) 原始路径（绝对路径或当前工作目录相对）
    2) 仓库根目录相对
    """
    raw = str(path_str).strip().strip('"').strip("'")
    normalized_variants = {
        raw,
        raw.replace("\\", "/"),
        raw.replace("/", os.sep),
        raw.replace("\\", os.sep),
    }

    candidates: List[Path] = []
    seen: set[str] = set()
    for variant in normalized_variants:
        p = Path(variant).expanduser()
        for candidate in (p, REPO_ROOT / p):
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

    for candidate in candidates:
        if _path_exists(candidate):
            return candidate
    recovered = _recover_manual_review_jsonl(candidates[0] if candidates else Path(raw))
    if recovered is not None:
        return recovered
    return candidates[0] if candidates else Path(raw)


def _recover_manual_review_jsonl(missing_path: Path) -> Optional[Path]:
    """
    针对 manual_review 目录名被截断/轻微清洗差异的场景做兜底恢复。
    """
    if missing_path.suffix.lower() != ".jsonl":
        return None
    target_name = missing_path.name
    expected_dir = missing_path.parent.name
    if target_name != "jointlk_multitask_train.jsonl":
        return None

    search_roots = [Path.cwd() / "artifacts" / "manual_review", REPO_ROOT / "artifacts" / "manual_review"]
    all_hits: List[Path] = []
    for root in search_roots:
        if _path_exists(root):
            all_hits.extend(root.rglob(target_name))
    if not all_hits:
        return None

    expected_norm = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", expected_dir).lower()

    def _score(p: Path) -> tuple[int, float]:
        parent_norm = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", p.parent.name).lower()
        overlap = 0
        for a, b in zip(expected_norm, parent_norm):
            if a != b:
                break
            overlap += 1
        return overlap, p.stat().st_mtime

    return sorted(all_hits, key=_score, reverse=True)[0]


def _to_windows_long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    s = str(p)
    if s.startswith("\\\\?\\"):
        return p
    if s.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + s.lstrip("\\"))
    return Path("\\\\?\\" + s)


def _path_exists(path: Path) -> bool:
    p = Path(path)
    if p.exists():
        return True
    if os.name == "nt":
        try:
            return _to_windows_long_path(p).exists()
        except Exception:
            return False
    return False