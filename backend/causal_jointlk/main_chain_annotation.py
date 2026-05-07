import hashlib
import json
import re
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
    raw_doc_id = str(record.get("doc_id") or "unknown").strip()
    # Windows 文件名不允许: \ / : * ? " < > |
    safe_doc = re.sub(r'[\\/:*?"<>|]+', "_", raw_doc_id)
    safe_doc = re.sub(r"\s+", "_", safe_doc).strip("._")
    if not safe_doc:
        safe_doc = "unknown"
    # 避免超长路径导致写入失败，追加 hash 保持可追踪性
    doc_hash = hashlib.sha1(raw_doc_id.encode("utf-8")).hexdigest()[:12]
    max_name_len = 120
    if len(safe_doc) > max_name_len:
        safe_doc = safe_doc[:max_name_len].rstrip("._")
    safe_doc = f"{safe_doc}__{doc_hash}"
    artifact_path = output_root / f"{safe_doc}.json"
    artifact_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(artifact_path)

def align_chain_steps_to_kg(steps: List[str], kg_nodes: Optional[List[Dict[str, object]]] = None) -> List[Dict[str, object]]:
    kg_nodes = kg_nodes or []
    node_index = {str((n.get("text") or "")).strip().lower(): n for n in kg_nodes}
    out = []
    for idx, step in enumerate(steps):
        hit = node_index.get(step.strip().lower())
        out.append({
            "chain_step_index": idx,
            "chain_step_text": step,
            "kg_node_id": (hit or {}).get("node_id"),
            "match_type": "exact" if hit else "unresolved",
        })
    return out


def align_chain_steps_to_prototypes(steps: List[str], prototype_alias: Optional[Dict[str, str]] = None) -> List[Dict[str, object]]:
    prototype_alias = prototype_alias or {}
    out = []
    for idx, step in enumerate(steps):
        proto = prototype_alias.get(step.strip().lower())
        out.append({
            "chain_step_index": idx,
            "chain_step_text": step,
            "prototype_id": proto,
            "match_type": "alias" if proto else "unresolved",
        })
    return out


def align_chain_steps_to_bn_vars(steps: List[str], bn_alias: Optional[Dict[str, str]] = None) -> List[Dict[str, object]]:
    bn_alias = bn_alias or {}
    out = []
    for idx, step in enumerate(steps):
        bn_var = bn_alias.get(step.strip().lower())
        out.append({
            "chain_step_index": idx,
            "chain_step_text": step,
            "bn_variable": bn_var,
            "match_type": "alias" if bn_var else "unresolved",
        })
    return out


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


def build_gold_chain_edge_rows(record: Dict[str, object]) -> List[Dict[str, object]]:
    steps = record.get("final_causal_chain_steps") or []
    rows: List[Dict[str, object]] = []
    for idx in range(max(0, len(steps) - 1)):
        rows.append({
            "doc_id": record.get("doc_id"),
            "file_name": record.get("fileName"),
            "label_source": "gold_chain",
            "review_status": "accepted",
            "accident_type": record.get("reviewed_accident_type"),
            "chain_id": f"gold::{record.get('doc_id')}",
            "chain_step_index": idx,
            "source_text": steps[idx],
            "target_text": steps[idx + 1],
            "candidate_relation": "causes",
            "source_node_id": None,
            "target_node_id": None,
            "causal_label": 1,
            "enable_label": 0,
            "dir_label": 1,
            "temp_label": 1,
            "alignment_status": record.get("alignment_status", "unresolved"),
        })
    return rows


def build_chain_benchmark_record(record: Dict[str, object], alignment_score: float) -> Dict[str, object]:
    return {
        "doc_id": record.get("doc_id"),
        "reviewed_accident_type": record.get("reviewed_accident_type"),
        "chain_text": record.get("final_causal_chain_text"),
        "chain_steps": record.get("final_causal_chain_steps") or [],
        "alignment_status": record.get("alignment_status", "unresolved"),
        "alignment_score": float(max(0.0, min(1.0, alignment_score))),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }