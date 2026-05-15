#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


def train_eval(tag: str, data_dir: Path, out_dir: Path, prior: str, extra_args: list[str]) -> None:
    exp_dir = out_dir / tag
    run([
        sys.executable,
        "experiments/causal_jointlk/train_causal_jointlk.py",
        "--train_jsonl", str(data_dir / "train.jsonl"),
        "--dev_jsonl", str(data_dir / "dev.jsonl"),
        "--prior_config", prior,
        "--output_dir", str(exp_dir),
        "--epochs", "8",
        "--batch_size", "8",
        *extra_args,
    ])

    run([
        sys.executable,
        "experiments/causal_jointlk/eval_causal_jointlk.py",
        "--test_jsonl", str(data_dir / "test.jsonl"),
        "--checkpoint", str(exp_dir / "best_model.pt"),
        "--prior_config", prior,
        "--output_json", str(out_dir / f"{tag}.eval.json"),
        "--output_rows_jsonl", str(out_dir / f"{tag}.eval.rows.jsonl"),
    ])


def main() -> None:
    p = argparse.ArgumentParser(description="Run manual-chain ablation experiments (Windows/Linux).")
    p.add_argument("--data_dir", default="data/causal_jointlk_supervised_v1")
    p.add_argument("--out_dir", default="results/manual_chain_ablation")
    p.add_argument("--prior", default="configs/causal_prior.yaml")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        train_eval(tag, data_dir, out_dir, args.prior, extra)

    rows = []
    for pth in sorted(out_dir.glob("*.eval.json")):
        rows.append(read_metrics(pth))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] summary =>", summary_path)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()