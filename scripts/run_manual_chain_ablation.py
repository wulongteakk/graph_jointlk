from __future__ import annotations

import argparse
import json
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data.get("metrics", {})
    return {
        "exp": path.name.replace(".eval.json", ""),
        "f1": m.get("f1"),
        "auc": m.get("auc"),
        "ap": m.get("ap"),
        "acc": m.get("acc"),
        "rel_acc": m.get("relation_acc"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _deterministic_split(doc_key: str, train_ratio: float, dev_ratio: float) -> str:
    h = int(hashlib.sha1(doc_key.encode("utf-8")).hexdigest(), 16) % 10_000
    x = h / 10_000.0
    if x < train_ratio:
        return "train"
    if x < train_ratio + dev_ratio:
        return "dev"
    return "test"


def _auto_build_from_registry(
    *,
    registry_path: Path,
    train_out: Path,
    dev_out: Path,
    test_out: Path,
    train_ratio: float,
    dev_ratio: float,
    max_edges_per_doc: int,
) -> dict[str, Any]:
    reg_rows = _read_jsonl(registry_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in reg_rows:
        train_jsonl = str(item.get("train_jsonl") or "").strip()
        if not train_jsonl:
            continue
        src = Path(train_jsonl)
        if not src.exists():
            continue
        rows = _read_jsonl(src)
        if not rows:
            continue
        # same doc rows in one source jsonl
        doc_id = str(rows[0].get("doc_id") or item.get("doc_id") or src.stem)
        if max_edges_per_doc > 0:
            rows = rows[:max_edges_per_doc]
        grouped[doc_id] = rows

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for doc_id, rows in sorted(grouped.items(), key=lambda x: x[0]):
        split = _deterministic_split(doc_id, train_ratio=train_ratio, dev_ratio=dev_ratio)
        if split == "train":
            train_rows.extend(rows)
        elif split == "dev":
            dev_rows.extend(rows)
        else:
            test_rows.extend(rows)

    _write_jsonl(train_out, train_rows)
    _write_jsonl(dev_out, dev_rows)
    _write_jsonl(test_out, test_rows)
    return {
        "registry_path": str(registry_path),
        "num_docs": len(grouped),
        "train_edges": len(train_rows),
        "dev_edges": len(dev_rows),
        "test_edges": len(test_rows),
        "train_jsonl": str(train_out),
        "dev_jsonl": str(dev_out),
        "test_jsonl": str(test_out),
    }


def train_eval(tag: str, train_jsonl: Path, dev_jsonl: Path, test_jsonl: Path, out_dir: Path, prior: str, extra_args: list[str]) -> None:
    exp_dir = out_dir / tag
    run([
        sys.executable,
        "experiments/causal_jointlk/train_causal_jointlk.py",
        "--train_jsonl", str(train_jsonl),
        "--dev_jsonl", str(dev_jsonl),
        "--prior_config", prior,
        "--output_dir", str(exp_dir),
        "--epochs", "8",
        "--batch_size", "8",
        *extra_args,
    ])

    run([
        sys.executable,
        "experiments/causal_jointlk/eval_causal_jointlk.py",
        "--test_jsonl", str(test_jsonl),
        "--checkpoint", str(exp_dir / "best_model.pt"),
        "--prior_config", prior,
        "--output_json", str(out_dir / f"{tag}.eval.json"),
        "--output_rows_jsonl", str(out_dir / f"{tag}.eval.rows.jsonl"),
    ])


def main() -> None:
    p = argparse.ArgumentParser(description="Run manual-chain ablation experiments (Windows/Linux).")
    p.add_argument("--data_dir", default="data/causal_jointlk_supervised_v1")
    p.add_argument("--train_jsonl", default="")
    p.add_argument("--dev_jsonl", default="")
    p.add_argument("--test_jsonl", default="")
    p.add_argument("--out_dir", default="results/manual_chain_ablation")
    p.add_argument("--prior", default="configs/causal_prior.yaml")
    p.add_argument("--auto_build", action="store_true", help="Auto-generate train/dev/test from registry when missing.")
    p.add_argument("--registry", default="artifacts/causal_corpus/registry.jsonl")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--dev_ratio", type=float, default=0.1)
    p.add_argument("--max_edges_per_doc", type=int, default=0)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = Path(args.train_jsonl) if str(args.train_jsonl).strip() else data_dir / "train.jsonl"
    dev_jsonl = Path(args.dev_jsonl) if str(args.dev_jsonl).strip() else data_dir / "dev.jsonl"
    test_jsonl = Path(args.test_jsonl) if str(args.test_jsonl).strip() else data_dir / "test.jsonl"

    missing = [p for p in [train_jsonl, dev_jsonl, test_jsonl] if not p.exists()]
    if missing and args.auto_build:
        reg = Path(args.registry)
        if not reg.exists():
            raise FileNotFoundError(f"--auto_build 已启用，但 registry 不存在: {reg}")
        split_stats = _auto_build_from_registry(
            registry_path=reg,
            train_out=train_jsonl,
            dev_out=dev_jsonl,
            test_out=test_jsonl,
            train_ratio=float(args.train_ratio),
            dev_ratio=float(args.dev_ratio),
            max_edges_per_doc=int(args.max_edges_per_doc),
        )
        print("[auto-build]", json.dumps(split_stats, ensure_ascii=False))
        missing = [p for p in [train_jsonl, dev_jsonl, test_jsonl] if not p.exists()]

    if missing:
        missing_lines = "\n".join([f"  - {p}" for p in missing])
        raise FileNotFoundError(
            "未找到数据集文件，请先准备 train/dev/test jsonl。\n"
            f"缺失文件:\n{missing_lines}\n"
            "可选方案：\n"
            "1) 将数据放到 --data_dir 下并命名为 train.jsonl/dev.jsonl/test.jsonl；\n"
            "2) 显式传参 --train_jsonl/--dev_jsonl/--test_jsonl 指向真实路径；\n"
            "3) 使用 --auto_build --registry artifacts/causal_corpus/registry.jsonl 自动生成。"
        )

    plans = [
        ("pseudo_only", ["--exclude_label_sources", "gold_chain"]),
        ("pseudo_reviewed_only", ["--exclude_label_sources", "gold_chain", "pseudo_pending"]),
        ("gold_only", ["--include_label_sources", "gold_chain"]),
        (
            "mixed",
            [
                "--include_label_sources",
                "gold_chain",
                "pseudo_review_accepted",
                "pseudo_review_edited",
                "pseudo_pending",
            ],
        ),
    ]

    for tag, extra in plans:
        train_eval(tag, train_jsonl, dev_jsonl, test_jsonl, out_dir, args.prior, extra)

    rows = []
    for pth in sorted(out_dir.glob("*.eval.json")):
        rows.append(read_metrics(pth))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] summary =>", summary_path)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()