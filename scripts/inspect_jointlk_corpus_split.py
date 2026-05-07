#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            rows.append(json.loads(t))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect which docs are eligible for offline JointLK batch training and show train/dev split.")
    parser.add_argument("--registry", default="artifacts/causal_corpus/registry.jsonl")
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--max_edges_per_doc", type=int, default=800)
    parser.add_argument("--exclude_doc_id", default=None)
    parser.add_argument("--show", type=int, default=200, help="max docs to print")
    args = parser.parse_args()

    reg = Path(args.registry)
    rows = read_jsonl(reg)
    if not rows:
        print(f"[EMPTY] no registry rows found: {reg}")
        return

    eligible: list[dict[str, Any]] = []
    for item in rows:
        doc_id = str(item.get("doc_id") or "").strip()
        train_jsonl = str(item.get("train_jsonl") or "").strip()
        train_path = Path(train_jsonl)
        if not doc_id or not train_jsonl or not train_path.exists():
            continue
        if args.exclude_doc_id and doc_id == args.exclude_doc_id:
            continue
        eligible.append(
            {
                "doc_id": doc_id,
                "file_name": item.get("file_name"),
                "train_jsonl": train_jsonl,
                "num_edges": int(item.get("num_edges") or 0),
                "positive": int(item.get("num_positive_support") or 0),
                "negative": int(item.get("num_negative_support") or 0),
                "unresolved": int(item.get("num_unresolved") or 0),
            }
        )

    eligible = sorted(eligible, key=lambda x: x["doc_id"])
    n = len(eligible)
    if n == 0:
        print("[EMPTY] no eligible docs (doc_id/train_jsonl missing or file not found).")
        return

    split_idx = max(1, int(round(n * (1.0 - float(args.dev_ratio)))))
    split_idx = min(split_idx, n)
    train_docs = eligible[:split_idx]
    dev_docs = eligible[split_idx:]

    print(f"[REGISTRY] {reg.resolve()}")
    print(f"[ELIGIBLE_DOCS] {n} | dev_ratio={args.dev_ratio} | max_edges_per_doc={args.max_edges_per_doc}")
    print(f"[SPLIT] train_docs={len(train_docs)} dev_docs={len(dev_docs)} (doc_id lexicographic split)")

    def _sum_edges(docs: list[dict[str, Any]]) -> int:
        cap = max(1, int(args.max_edges_per_doc)) if args.max_edges_per_doc > 0 else None
        total = 0
        for d in docs:
            e = int(d["num_edges"])
            total += min(e, cap) if cap else e
        return total

    print(f"[SPLIT_EDGES_EST] train~{_sum_edges(train_docs)} dev~{_sum_edges(dev_docs)}")

    print("\n== Eligible docs ==")
    for idx, d in enumerate(eligible[: args.show], start=1):
        bucket = "train" if idx <= len(train_docs) else "dev"
        print(
            f"[{idx:03d}][{bucket}] doc_id={d['doc_id']} file={d['file_name']} "
            f"edges={d['num_edges']} pos={d['positive']} neg={d['negative']} unr={d['unresolved']}"
        )

    if len(eligible) > args.show:
        print(f"... truncated, showing {args.show}/{len(eligible)} docs")


if __name__ == "__main__":
    main()