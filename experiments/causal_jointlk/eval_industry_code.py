from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .metrics_industry import summarize_industry_metrics


def load_jsonl(path: str) -> List[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="预测 jsonl，每行包含 industry_code 与 industry_candidates")
    parser.add_argument("--gold", required=True, help="标注 jsonl，每行包含 industry_code")
    args = parser.parse_args()

    preds = load_jsonl(args.pred)
    golds = load_jsonl(args.gold)

    pred_codes = [row.get("industry_code") for row in preds]
    gold_codes = [row.get("industry_code") for row in golds]
    pred_topk = [[c.get("gbt4754_full_code") for c in row.get("industry_candidates", [])] for row in preds]

    metrics = summarize_industry_metrics(pred_codes, gold_codes, pred_topk)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
