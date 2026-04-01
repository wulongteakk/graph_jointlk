from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in [str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _load_service_factory(factory_path: str):
    module_name, fn_name = factory_path.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"service factory is not callable: {factory_path}")
    return fn


def _normalize_text(v: Any) -> str:
    return str(v or "").strip()


def _code_match(pred: Any, gold: Any) -> bool:
    return _normalize_text(pred) == _normalize_text(gold) and _normalize_text(gold) != ""


def run_eval_end2end(service: Any, rows: List[Dict[str, Any]], mode: str, top_k: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = 0
    branch_first_hits = 0
    severity_acc_hits = 0
    severity_trigger_total = 0
    severity_trigger_hits = 0
    basic_hits = 0
    injury_hits = 0
    industry_hits = 0
    full_hits = 0
    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        n += 1
        result = service.extract(
            query=row.get("query"),
            target_text=row.get("target_text"),
            target_node_id=row.get("target_node_id"),
            doc_id=row.get("doc_id"),
            mode=mode,
            k_hop=int(row.get("k_hop", 2)),
            top_k=top_k,
            persist=False,
            trace=True,
        )
        decision = result.branch_decision
        decoded = result.decoded_result
        severity_trace = (result.meta or {}).get("severity_trace", {})

        gold_branch = row.get("gold_selected_branch_id")
        if gold_branch and decision is not None and decision.selected_branch_id == gold_branch:
            branch_first_hits += 1

        gold_triggered = row.get("gold_severity_fallback_triggered")
        if isinstance(gold_triggered, bool):
            severity_trigger_total += 1
            pred_triggered = bool(severity_trace.get("triggered", False))
            if pred_triggered == gold_triggered:
                severity_trigger_hits += 1

        gold_severity_branch = row.get("gold_severity_selected_branch_id")
        if gold_severity_branch and severity_trace.get("selected_branch_id") == gold_severity_branch:
            severity_acc_hits += 1

        gold_basic = row.get("gold_basic_type")
        gold_injury = row.get("gold_injury_code")
        gold_industry = row.get("gold_industry_code")
        pred_basic = decoded.basic_type if decoded else None
        pred_injury = decoded.injury_code if decoded else None
        pred_industry = decoded.industry_code if decoded else None

        basic_ok = _code_match(pred_basic, gold_basic)
        injury_ok = _code_match(pred_injury, gold_injury)
        industry_ok = _code_match(pred_industry, gold_industry)

        basic_hits += int(basic_ok)
        injury_hits += int(injury_ok)
        industry_hits += int(industry_ok)
        full_hits += int(basic_ok and injury_ok and industry_ok)

        out_rows.append(
            {
                "doc_id": row.get("doc_id"),
                "query": row.get("query"),
                "pred_selected_branch_id": decision.selected_branch_id if decision else None,
                "pred_needs_severity_fallback": bool(severity_trace.get("triggered", False)),
                "pred_basic_type": pred_basic,
                "pred_injury_code": pred_injury,
                "pred_industry_code": pred_industry,
                "gold_selected_branch_id": gold_branch,
                "gold_basic_type": gold_basic,
                "gold_injury_code": gold_injury,
                "gold_industry_code": gold_industry,
                "trace": (result.meta or {}).get("trace", {}),
            }
        )

    metrics = {
        "size": n,
        "branch_first_acc": _safe_div(branch_first_hits, n),
        "severity_fallback_trigger_rate": _safe_div(sum(1 for r in out_rows if r["pred_needs_severity_fallback"]), n),
        "severity_fallback_acc": _safe_div(severity_acc_hits, n),
        "severity_fallback_trigger_acc": _safe_div(severity_trigger_hits, severity_trigger_total),
        "basic_type_acc": _safe_div(basic_hits, n),
        "injury_code_acc": _safe_div(injury_hits, n),
        "industry_code_acc": _safe_div(industry_hits, n),
        "full_code_acc": _safe_div(full_hits, n),
    }
    return metrics, out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end document evaluation for JointLK service outputs.")
    parser.add_argument("--input_jsonl", required=True, help="Doc-level evaluation set in JSONL format.")
    parser.add_argument(
        "--service_factory",
        required=True,
        help="Import path for service factory, e.g. `my_pkg.eval_support:build_service`.",
    )
    parser.add_argument("--mode", default="jointlk+gate")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_rows_jsonl", default="")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    service_factory = _load_service_factory(args.service_factory)
    service = service_factory()

    metrics, rows = run_eval_end2end(service=service, rows=records, mode=args.mode, top_k=args.top_k)

    output = {
        "metrics": metrics,
        "size": len(records),
        "mode": args.mode,
        "top_k": args.top_k,
        "input_jsonl": args.input_jsonl,
        "service_factory": args.service_factory,
    }
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_rows_jsonl:
        out_path = Path(args.output_rows_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()