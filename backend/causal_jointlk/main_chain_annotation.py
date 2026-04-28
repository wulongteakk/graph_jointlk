import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def normalize_chain_text(raw_chain: str) -> Dict[str, object]:
    text = str(raw_chain or "").strip()
    if not text:
        return {"final_causal_chain_text": "", "final_causal_chain_steps": []}

    normalized = text.replace("\n", " -> ").replace("/", " -> ").replace("-", " -> ")
    steps = [step.strip() for step in normalized.split("->") if step and step.strip()]
    final_text = " -> ".join(steps)
    return {"final_causal_chain_text": final_text, "final_causal_chain_steps": steps}


def build_main_chain_annotation_record(
    *,
    doc_id: str,
    file_name: str,
    kg_scope: Optional[str],
    kg_id: Optional[str],
    accident_type: str,
    final_causal_chain: str,
    alignment_status: str = "unresolved",
) -> Dict[str, object]:
    normalized = normalize_chain_text(final_causal_chain)
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "doc_id": doc_id,
        "fileName": file_name,
        "kg_scope": kg_scope,
        "kg_id": kg_id,
        "reviewed_accident_type": accident_type,
        "final_causal_chain": normalized["final_causal_chain_text"],
        "final_causal_chain_text": normalized["final_causal_chain_text"],
        "final_causal_chain_steps": normalized["final_causal_chain_steps"],
        "alignment_status": alignment_status,
        "annotation_status": "annotated",
        "annotation_source": "manual",
        "label_source": "gold_chain",
        "review_status": "accepted",
        "updated_at": now_iso,
    }


def persist_main_chain_artifact(record: Dict[str, object]) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / "outputs" / "manual_main_chain"
    output_root.mkdir(parents=True, exist_ok=True)
    safe_doc = str(record.get("doc_id") or "unknown").replace("/", "_")
    artifact_path = output_root / f"{safe_doc}.json"
    artifact_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(artifact_path)


def build_gold_chain_rows(record: Dict[str, object]) -> List[Dict[str, object]]:
    steps = record.get("final_causal_chain_steps") or []
    rows: List[Dict[str, object]] = []
    for idx, step in enumerate(steps):
        rows.append(
            {
                "doc_id": record.get("doc_id"),
                "file_name": record.get("fileName"),
                "label_source": "gold_chain",
                "review_status": "accepted",
                "accident_type": record.get("reviewed_accident_type"),
                "chain_step_index": idx,
                "chain_step_text": step,
                "alignment_status": record.get("alignment_status", "unresolved"),
            }
        )
    return rows