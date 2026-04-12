from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_service_factory(factory_path: str) -> Callable[[], Any]:
    module_name, fn_name = factory_path.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"service factory is not callable: {factory_path}")
    return fn


def _result_preview(result: Any) -> Dict[str, Any]:
    decoded = getattr(result, "decoded_result", None)
    decision = getattr(result, "branch_decision", None)
    trace_payload = (getattr(result, "meta", {}) or {}).get("trace", {})
    return {
        "selected_branch_id": getattr(decision, "selected_branch_id", None),
        "decision_reason": getattr(decision, "reason", None),
        "basic_type": getattr(decoded, "basic_type", None),
        "injury_code": getattr(decoded, "injury_code", None),
        "industry_code": getattr(decoded, "industry_code", None),
        "decode_confidence": getattr(decoded, "decode_confidence", None),
        "retrieval_trace": trace_payload.get("retrieval_trace", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JointLK service.extract for a single document/debug case.")
    parser.add_argument("--service_factory", required=True, help="Import path: package.module:function")
    parser.add_argument("--doc_id", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--target_text", default="")
    parser.add_argument("--target_node_id", default="")
    parser.add_argument("--mode", default="jointlk+gate+branch")
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace_top_edges", type=int, default=12)
    parser.add_argument("--trace_top_chains", type=int, default=5)
    parser.add_argument("--trace_top_branches", type=int, default=5)
    args = parser.parse_args()

    service = _load_service_factory(args.service_factory)()
    result = service.extract(
        query=args.query or None,
        target_text=args.target_text or None,
        target_node_id=args.target_node_id or None,
        doc_id=args.doc_id or None,
        mode=args.mode,
        k_hop=int(args.k_hop),
        top_k=int(args.top_k),
        persist=False,
        trace=bool(args.trace),
        trace_top_edges=int(args.trace_top_edges),
        trace_top_chains=int(args.trace_top_chains),
        trace_top_branches=int(args.trace_top_branches),
    )
    print("[JointLK][cli-summary]", json.dumps(_result_preview(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()