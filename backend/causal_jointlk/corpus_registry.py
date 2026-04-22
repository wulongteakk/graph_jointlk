from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_REGISTRY_PATH = Path("artifacts/causal_corpus/registry.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_train_rows(registry_rows: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for item in registry_rows:
        train_jsonl = str(item.get("train_jsonl") or "").strip()
        if not train_jsonl:
            continue
        train_path = Path(train_jsonl)
        if not train_path.exists():
            continue
        for row in _read_jsonl(train_path):
            yield row


def _count_support(rows: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
    pos = neg = unresolved = 0
    for row in rows:
        mask = int(row.get("support_mask", row.get("causal_mask", 0)) or 0)
        label = int(row.get("support_label", row.get("label", 0)) or 0)
        if mask != 1:
            unresolved += 1
        elif label == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg, unresolved


def register_doc_package(
    *,
    pseudo_result: Dict[str, Any],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    accident_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Append/replace one doc package record in the global causal corpus registry."""
    registry_file = Path(registry_path)
    rows = _read_jsonl(registry_file)

    doc_id = str(pseudo_result.get("doc_id") or "").strip()
    file_name = pseudo_result.get("file_name")
    paths = pseudo_result.get("paths") or {}
    train_jsonl = str(paths.get("jointlk_multitask_train_jsonl") or "").strip()
    edge_jsonl = str(paths.get("candidate_edge_table_jsonl") or "").strip()
    node_prior_jsonl = str(paths.get("candidate_node_prior_table_jsonl") or "").strip()
    manifest_json = str(Path(pseudo_result.get("export_dir") or ".") / "manifest.json")

    train_rows = _read_jsonl(Path(train_jsonl)) if train_jsonl and Path(train_jsonl).exists() else []
    pos, neg, unresolved = _count_support(train_rows)

    record = {
        "doc_id": doc_id,
        "file_name": file_name,
        "kg_scope": pseudo_result.get("kg_scope"),
        "kg_id": pseudo_result.get("kg_id"),
        "train_jsonl": train_jsonl,
        "candidate_edge_table_jsonl": edge_jsonl,
        "node_prior_jsonl": node_prior_jsonl,
        "manifest_json": manifest_json,
        "num_edges": len(train_rows),
        "num_positive_support": pos,
        "num_negative_support": neg,
        "num_unresolved": unresolved,
        "accident_type": accident_type or pseudo_result.get("accident_type") or "UNKNOWN",
        "created_at": _now_iso(),
    }

    deduped: List[Dict[str, Any]] = []
    replaced = False
    for item in rows:
        same_doc = doc_id and str(item.get("doc_id") or "") == doc_id
        same_train = train_jsonl and str(item.get("train_jsonl") or "") == train_jsonl
        if same_doc or same_train:
            deduped.append(record)
            replaced = True
        else:
            deduped.append(item)
    if not replaced:
        deduped.append(record)

    _write_jsonl(registry_file, deduped)
    return {
        "registry_path": str(registry_file.resolve()),
        "record": record,
        "num_docs": len(deduped),
    }


def build_corpus_train_dev(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    train_out: str | Path = "artifacts/causal_corpus/corpus_train.jsonl",
    dev_out: str | Path = "artifacts/causal_corpus/corpus_dev.jsonl",
    exclude_doc_id: Optional[str] = None,
    dev_doc_ids: Optional[Sequence[str]] = None,
    dev_ratio: float = 0.2,
    max_edges_per_doc: Optional[int] = None,
) -> Dict[str, Any]:
    registry_rows = _read_jsonl(Path(registry_path))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _iter_train_rows(registry_rows):
        did = str(row.get("doc_id") or "UNKNOWN")
        if exclude_doc_id and did == exclude_doc_id:
            continue
        grouped[did].append(row)

    doc_ids = sorted(grouped.keys())
    dev_doc_set = set(str(x) for x in (dev_doc_ids or []))
    if not dev_doc_set and doc_ids:
        split_idx = max(1, int(round(len(doc_ids) * (1.0 - float(dev_ratio)))))
        split_idx = min(split_idx, len(doc_ids))
        dev_doc_set = set(doc_ids[split_idx:])

    train_rows: List[Dict[str, Any]] = []
    dev_rows: List[Dict[str, Any]] = []
    for did, rows in grouped.items():
        picked = rows[: int(max_edges_per_doc)] if max_edges_per_doc and max_edges_per_doc > 0 else rows
        if did in dev_doc_set:
            dev_rows.extend(picked)
        else:
            train_rows.extend(picked)

    _write_jsonl(Path(train_out), train_rows)
    _write_jsonl(Path(dev_out), dev_rows)

    return {
        "registry_path": str(Path(registry_path).resolve()),
        "train_jsonl": str(Path(train_out).resolve()),
        "dev_jsonl": str(Path(dev_out).resolve()),
        "num_docs": len(doc_ids),
        "train_docs": len([d for d in doc_ids if d not in dev_doc_set]),
        "dev_docs": len(dev_doc_set),
        "train_edges": len(train_rows),
        "dev_edges": len(dev_rows),
        "excluded_doc_id": exclude_doc_id,
    }


def read_corpus_stats(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    rows = _read_jsonl(Path(registry_path))
    accident_type_dist: Dict[str, int] = defaultdict(int)
    support_pos = support_neg = support_unresolved = 0
    per_doc: List[Dict[str, Any]] = []
    total_edges = 0

    for row in rows:
        atype = str(row.get("accident_type") or "UNKNOWN")
        accident_type_dist[atype] += 1
        pos = int(row.get("num_positive_support") or 0)
        neg = int(row.get("num_negative_support") or 0)
        unr = int(row.get("num_unresolved") or 0)
        edges = int(row.get("num_edges") or 0)
        total_edges += edges
        support_pos += pos
        support_neg += neg
        support_unresolved += unr
        per_doc.append(
            {
                "doc_id": row.get("doc_id"),
                "num_edges": edges,
                "positive": pos,
                "negative": neg,
                "unresolved": unr,
            }
        )

    return {
        "registry_path": str(Path(registry_path).resolve()),
        "num_docs": len(rows),
        "total_candidate_edges": total_edges,
        "support_positive": support_pos,
        "support_negative": support_neg,
        "support_unresolved": support_unresolved,
        "accident_type_distribution": dict(sorted(accident_type_dist.items())),
        "per_doc": per_doc,
    }