from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evidence_store.sqlite_store import EvidenceStore  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import manual review CSV into EvidenceStore pseudo label table.")
    parser.add_argument("--review_csv", required=True)
    parser.add_argument("--evidence_db_path", default="./data/evidence_store.sqlite3")
    return parser.parse_args()


def parse_optional_int(text: str):
    text = (text or "").strip()
    if text == "":
        return None
    return int(text)


def main() -> None:
    args = parse_args()
    store = EvidenceStore(args.evidence_db_path)

    updated = 0
    with open(args.review_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pseudo_label_id = (row.get("pseudo_label_id") or "").strip()
            manual_decision = (row.get("manual_decision") or "").strip().lower()
            if not pseudo_label_id or not manual_decision:
                continue

            if manual_decision not in {"accepted", "rejected", "edited"}:
                continue

            manual_label = parse_optional_int(row.get("manual_label") or "")
            manual_relation_type = (row.get("manual_relation_type") or "").strip() or None
            manual_comment = (row.get("manual_comment") or "").strip() or None
            reviewer = (row.get("reviewer") or "").strip() or None

            store.update_pseudo_edge_label_review(
                pseudo_label_id=pseudo_label_id,
                review_status=manual_decision,
                manual_label=manual_label,
                manual_relation_type=manual_relation_type,
                manual_comment=manual_comment,
                reviewer=reviewer,
            )
            updated += 1

    print({"ok": True, "updated": updated})


if __name__ == "__main__":
    main()
