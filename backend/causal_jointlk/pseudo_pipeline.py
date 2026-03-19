from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .neo4j_accessor import CandidateEdge, Neo4jAccessor
from .pseudo_labeler import (
    CausalPseudoLabeler,
    lexical_overlap,
    load_yaml_file,
    merge_prior_and_pseudo_config,
)
from src.evidence_store.sqlite_store import EvidenceStore  # type: ignore


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value]
    return [value]


def relation_unit_ids(rel_props: Dict[str, Any]) -> List[str]:
    unit_ids: List[str] = []
    for key in [
        "evidence_unit_id",
        "evidence_unit_ids",
        "unit_id",
        "unit_ids",
        "support_unit_id",
        "support_unit_ids",
    ]:
        unit_ids.extend([str(x) for x in safe_list(rel_props.get(key)) if str(x).strip()])
    return list(dict.fromkeys(unit_ids))


def relation_chunk_refs(rel_props: Dict[str, Any], edge: CandidateEdge) -> List[str]:
    refs: List[str] = []
    for key in ["chunk_ref", "chunk_refs", "evidence_id", "evidence_ids", "parent_evidence_id", "parent_evidence_ids"]:
        refs.extend([str(x) for x in safe_list(rel_props.get(key)) if str(x).strip()])
    if edge.source_chunk_id:
        refs.append(str(edge.source_chunk_id))
    if edge.target_chunk_id:
        refs.append(str(edge.target_chunk_id))
    return list(dict.fromkeys(refs))


def retrieve_lexical_units(
    source_text: str,
    target_text: str,
    unit_rows: Sequence[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in unit_rows:
        text = row.get("content") or ""
        score = 0.0
        score += 1.2 * lexical_overlap(source_text, text)
        score += 1.2 * lexical_overlap(target_text, text)
        if source_text and str(source_text).lower() in str(text).lower():
            score += 0.6
        if target_text and str(target_text).lower() in str(text).lower():
            score += 0.6
        if score > 0.25:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def read_doc_evidence_units(
    store: EvidenceStore,
    *,
    file_name: str,
    max_units: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    chunks = store.list_chunks_for_file(file_name)
    unit_rows: List[Dict[str, Any]] = []
    unit_by_id: Dict[str, Dict[str, Any]] = {}
    chunk_by_id: Dict[str, Dict[str, Any]] = {}

    for ch in chunks:
        chunk_by_id[str(ch.evidence_id)] = {
            "evidence_id": ch.evidence_id,
            "content": ch.content,
            "position": ch.position,
            "meta": ch.meta,
        }
        for unit in store.list_units_for_parent(ch.evidence_id):
            item = {
                "unit_id": unit.unit_id,
                "parent_evidence_id": unit.parent_evidence_id,
                "content": unit.content,
                "unit_kind": unit.unit_kind,
                "start_char": unit.start_char,
                "end_char": unit.end_char,
                "meta": unit.meta,
            }
            unit_rows.append(item)
            unit_by_id[str(unit.unit_id)] = item
            if len(unit_rows) >= max_units:
                break
        if len(unit_rows) >= max_units:
            break

    return unit_rows, unit_by_id, chunk_by_id


def gather_evidence_units_for_edge(
    store: EvidenceStore,
    edge: CandidateEdge,
    unit_rows: Sequence[Dict[str, Any]],
    unit_by_id: Dict[str, Dict[str, Any]],
    chunk_by_id: Dict[str, Dict[str, Any]],
    lexical_top_k: int,
    include_chunk_fallback: bool,
) -> List[Dict[str, Any]]:
    rel_props = edge.rel_props or {}
    gathered: List[Dict[str, Any]] = []

    for unit_id in relation_unit_ids(rel_props):
        item = unit_by_id.get(unit_id)
        if item is None:
            unit_obj = store.get_unit(unit_id)
            if unit_obj is not None:
                item = {
                    "unit_id": unit_obj.unit_id,
                    "parent_evidence_id": unit_obj.parent_evidence_id,
                    "content": unit_obj.content,
                    "unit_kind": unit_obj.unit_kind,
                    "start_char": unit_obj.start_char,
                    "end_char": unit_obj.end_char,
                    "meta": unit_obj.meta,
                }
        if item is not None:
            gathered.append(item)

    if not gathered:
        for ref in relation_chunk_refs(rel_props, edge):
            if ref in chunk_by_id:
                chunk_item = chunk_by_id[ref]
                child_units = store.list_units_for_parent(ref)
                if child_units:
                    for unit in child_units[: lexical_top_k]:
                        gathered.append(
                            {
                                "unit_id": unit.unit_id,
                                "parent_evidence_id": unit.parent_evidence_id,
                                "content": unit.content,
                                "unit_kind": unit.unit_kind,
                                "start_char": unit.start_char,
                                "end_char": unit.end_char,
                                "meta": unit.meta,
                            }
                        )
                elif include_chunk_fallback:
                    gathered.append(
                        {
                            "unit_id": f"chunk::{ref}",
                            "parent_evidence_id": ref,
                            "content": chunk_item.get("content") or "",
                            "unit_kind": "chunk",
                            "start_char": None,
                            "end_char": None,
                            "meta": chunk_item.get("meta") or {},
                        }
                    )

    if not gathered:
        gathered.extend(
            retrieve_lexical_units(
                edge.source_text,
                edge.target_text,
                unit_rows,
                top_k=lexical_top_k,
            )
        )

    dedup: Dict[str, Dict[str, Any]] = {}
    for item in gathered:
        uid = str(item.get("unit_id") or "")
        if uid and uid not in dedup:
            dedup[uid] = item
    return list(dedup.values())


def build_pseudo_label_record(
    edge: CandidateEdge,
    decision: Any,
) -> Dict[str, Any]:
    base = f"{edge.doc_id}|{edge.file_name}|{edge.source_node_id}|{edge.relation_type}|{edge.target_node_id}"
    pseudo_label_id = f"pl_{sha1_text(base)}"
    return {
        "pseudo_label_id": pseudo_label_id,
        "doc_id": edge.doc_id,
        "kg_scope": edge.kg_scope,
        "kg_id": edge.kg_id,
        "file_name": edge.file_name,
        "source_node_id": edge.source_node_id,
        "source_text": edge.source_text,
        "source_layer": edge.source_layer,
        "target_node_id": edge.target_node_id,
        "target_text": edge.target_text,
        "target_layer": edge.target_layer,
        "relation_type": decision.relation_type or edge.relation_type,
        "label": int(decision.label),
        "confidence": float(decision.confidence),
        "evidence_unit_id": decision.evidence_unit_id,
        "evidence_start": None,
        "evidence_end": None,
        "evidence_text": decision.evidence_text,
        "rule_hits": {
            "primary_rule": decision.primary_rule,
            "hits": decision.rule_hits,
        },
        "features": decision.features,
        "review_status": "pending",
        "manual_label": None,
        "manual_relation_type": None,
        "manual_comment": None,
        "reviewer": None,
    }


def choose_review_candidates(
    labels: Sequence[Dict[str, Any]],
    sample_size: int,
    per_doc_cap: int,
    per_rule_cap: int,
) -> List[Dict[str, Any]]:
    by_group: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in labels:
        primary_rule = (row.get("rule_hits") or {}).get("primary_rule")
        key = (
            row.get("label"),
            primary_rule,
            row.get("relation_type"),
        )
        by_group[key].append(row)

    for rows in by_group.values():
        rows.sort(key=lambda r: float(r.get("confidence") or 0.0))

    doc_used: Dict[str, int] = defaultdict(int)
    rule_used: Dict[str, int] = defaultdict(int)
    selected: List[Dict[str, Any]] = []

    groups = sorted(
        by_group.items(),
        key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2])),
    )

    exhausted = False
    round_idx = 0
    while len(selected) < sample_size and not exhausted:
        exhausted = True
        for _, rows in groups:
            if round_idx >= len(rows):
                continue
            exhausted = False
            row = rows[round_idx]
            doc_key = str(row.get("doc_id") or row.get("file_name") or "")
            rule_key = str((row.get("rule_hits") or {}).get("primary_rule") or "")
            if doc_used[doc_key] >= per_doc_cap:
                continue
            if rule_used[rule_key] >= per_rule_cap:
                continue
            selected.append(row)
            doc_used[doc_key] += 1
            rule_used[rule_key] += 1
            if len(selected) >= sample_size:
                break
        round_idx += 1
    return selected


def effective_pseudo_label(record: Any) -> Optional[Tuple[int, Optional[str], str]]:
    review_status = str(getattr(record, "review_status", "pending") or "pending").lower()
    if review_status == "rejected":
        return None
    if review_status == "edited":
        manual_label = getattr(record, "manual_label", None)
        manual_rel = getattr(record, "manual_relation_type", None)
        if manual_label in (0, 1):
            return int(manual_label), manual_rel, "pseudo_review_edited"
        return None
    if review_status == "accepted":
        return int(getattr(record, "label")), getattr(record, "relation_type", None), "pseudo_review_accepted"
    return int(getattr(record, "label")), getattr(record, "relation_type", None), "pseudo_pending"


def sanitize_fs_part(text: Optional[str]) -> str:
    s = str(text or "unknown").strip()
    s = s.replace("/", "_").replace("\\", "_").replace("|", "_").replace(":", "_")
    s = s.replace(" ", "_")
    return s[:160]


@dataclass
class AutoPseudoPipelineConfig:
    enabled: bool = True
    export_root: str = "./artifacts/manual_review"
    evidence_db_path: str = "./data/evidence_store.sqlite3"
    prior_config_path: str = "configs/causal_prior.yaml"
    pseudo_rule_config_path: str = "configs/causal_pseudo_label_rules.yaml"
    store_ambiguous: bool = False
    max_edges_per_doc: Optional[int] = None
    max_units_per_doc_scan: int = 500
    lexical_top_k: int = 5
    include_chunk_fallback: bool = True
    review_sample_size: int = 200
    review_per_doc_cap: int = 200
    review_per_rule_cap: int = 50

    @classmethod
    def from_env(cls) -> "AutoPseudoPipelineConfig":
        def _get_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}

        def _get_int(name: str, default: Optional[int]) -> Optional[int]:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            return int(raw)

        return cls(
            enabled=_get_bool("AUTO_PSEUDO_LABEL_AFTER_UPLOAD", True),
            export_root=os.getenv("AUTO_PSEUDO_LABEL_EXPORT_DIR", "./artifacts/manual_review"),
            evidence_db_path=os.getenv("EVIDENCE_DB_PATH", "./data/evidence_store.sqlite3"),
            prior_config_path=os.getenv("AUTO_PSEUDO_LABEL_PRIOR_CONFIG", "configs/causal_prior.yaml"),
            pseudo_rule_config_path=os.getenv("AUTO_PSEUDO_LABEL_RULE_CONFIG", "configs/causal_pseudo_label_rules.yaml"),
            store_ambiguous=_get_bool("AUTO_PSEUDO_LABEL_STORE_AMBIGUOUS", False),
            max_edges_per_doc=_get_int("AUTO_PSEUDO_LABEL_MAX_EDGES_PER_DOC", None),
            max_units_per_doc_scan=_get_int("AUTO_PSEUDO_LABEL_MAX_UNITS_PER_DOC_SCAN", 500) or 500,
            lexical_top_k=_get_int("AUTO_PSEUDO_LABEL_LEXICAL_TOP_K", 5) or 5,
            include_chunk_fallback=_get_bool("AUTO_PSEUDO_LABEL_INCLUDE_CHUNK_FALLBACK", True),
            review_sample_size=_get_int("AUTO_PSEUDO_LABEL_REVIEW_SAMPLE_SIZE", 200) or 200,
            review_per_doc_cap=_get_int("AUTO_PSEUDO_LABEL_REVIEW_PER_DOC_CAP", 200) or 200,
            review_per_rule_cap=_get_int("AUTO_PSEUDO_LABEL_REVIEW_PER_RULE_CAP", 50) or 50,
        )


def export_pseudo_label_package(
    out_dir: Path,
    label_rows: Sequence[Dict[str, Any]],
    *,
    config: AutoPseudoPipelineConfig,
    manifest_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csv = out_dir / "all_pseudo_labels.csv"
    all_jsonl = out_dir / "pseudo_labels.jsonl"
    review_csv = out_dir / "manual_review_candidates.csv"

    fieldnames = [
        "pseudo_label_id",
        "doc_id",
        "file_name",
        "source_node_id",
        "source_text",
        "source_layer",
        "target_node_id",
        "target_text",
        "target_layer",
        "relation_type",
        "label",
        "confidence",
        "primary_rule",
        "rule_hits_json",
        "features_json",
        "evidence_unit_id",
        "evidence_text",
        "manual_decision",
        "manual_label",
        "manual_relation_type",
        "manual_comment",
        "reviewer",
    ]

    with all_jsonl.open("w", encoding="utf-8") as f:
        for row in label_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with all_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames[:-5])  # all labels不包含人工列
        writer.writeheader()
        for row in label_rows:
            writer.writerow(
                {
                    "pseudo_label_id": row.get("pseudo_label_id"),
                    "doc_id": row.get("doc_id"),
                    "file_name": row.get("file_name"),
                    "source_node_id": row.get("source_node_id"),
                    "source_text": row.get("source_text"),
                    "source_layer": row.get("source_layer"),
                    "target_node_id": row.get("target_node_id"),
                    "target_text": row.get("target_text"),
                    "target_layer": row.get("target_layer"),
                    "relation_type": row.get("relation_type"),
                    "label": row.get("label"),
                    "confidence": row.get("confidence"),
                    "primary_rule": (row.get("rule_hits") or {}).get("primary_rule"),
                    "rule_hits_json": json.dumps(row.get("rule_hits") or {}, ensure_ascii=False),
                    "features_json": json.dumps(row.get("features") or {}, ensure_ascii=False),
                    "evidence_unit_id": row.get("evidence_unit_id"),
                    "evidence_text": row.get("evidence_text"),
                }
            )

    review_candidates = choose_review_candidates(
        label_rows,
        sample_size=min(len(label_rows), config.review_sample_size),
        per_doc_cap=config.review_per_doc_cap,
        per_rule_cap=config.review_per_rule_cap,
    )
    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_candidates:
            writer.writerow(
                {
                    "pseudo_label_id": row.get("pseudo_label_id"),
                    "doc_id": row.get("doc_id"),
                    "file_name": row.get("file_name"),
                    "source_node_id": row.get("source_node_id"),
                    "source_text": row.get("source_text"),
                    "source_layer": row.get("source_layer"),
                    "target_node_id": row.get("target_node_id"),
                    "target_text": row.get("target_text"),
                    "target_layer": row.get("target_layer"),
                    "relation_type": row.get("relation_type"),
                    "label": row.get("label"),
                    "confidence": row.get("confidence"),
                    "primary_rule": (row.get("rule_hits") or {}).get("primary_rule"),
                    "rule_hits_json": json.dumps(row.get("rule_hits") or {}, ensure_ascii=False),
                    "features_json": json.dumps(row.get("features") or {}, ensure_ascii=False),
                    "evidence_unit_id": row.get("evidence_unit_id"),
                    "evidence_text": row.get("evidence_text"),
                    "manual_decision": "",
                    "manual_label": "",
                    "manual_relation_type": "",
                    "manual_comment": "",
                    "reviewer": "",
                }
            )

    manifest = {
        "num_pseudo_labels": len(label_rows),
        "label_breakdown": {
            "positive": sum(1 for x in label_rows if int(x.get("label", -1)) == 1),
            "negative": sum(1 for x in label_rows if int(x.get("label", -1)) == 0),
        },
        "review_candidate_count": len(review_candidates),
        "paths": {
            "pseudo_labels_jsonl": str(all_jsonl),
            "all_pseudo_labels_csv": str(all_csv),
            "manual_review_candidates_csv": str(review_csv),
        },
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_pseudo_label_pipeline_for_doc(
    *,
    graph: Any,
    file_name: str,
    doc_id: Optional[str],
    kg_scope: Optional[str],
    kg_id: Optional[str],
    config: Optional[AutoPseudoPipelineConfig] = None,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    console_preview_limit: int = 0,
) -> Dict[str, Any]:
    cfg = config or AutoPseudoPipelineConfig.from_env()
    if not cfg.enabled:
        return {"ok": False, "skipped": True, "reason": "disabled"}

    prior_cfg = load_yaml_file(cfg.prior_config_path) if cfg.prior_config_path and os.path.exists(cfg.prior_config_path) else {}
    pseudo_cfg = load_yaml_file(cfg.pseudo_rule_config_path) if cfg.pseudo_rule_config_path and os.path.exists(cfg.pseudo_rule_config_path) else {}
    merged_cfg = merge_prior_and_pseudo_config(prior_cfg, pseudo_cfg)

    store = EvidenceStore(cfg.evidence_db_path)
    accessor = Neo4jAccessor(graph)
    labeler = CausalPseudoLabeler(merged_cfg)

    doc_edges = accessor.get_doc_candidate_edges(
        doc_id=doc_id,
        file_name=file_name,
        kg_scope=kg_scope or "instance",
        kg_id=kg_id,
        relation_types=None,
        limit=cfg.max_edges_per_doc,
    )

    unit_rows, unit_by_id, chunk_by_id = read_doc_evidence_units(
        store,
        file_name=file_name,
        max_units=cfg.max_units_per_doc_scan,
    )

    label_rows: List[Dict[str, Any]] = []
    ambiguous = 0
    for edge_idx, edge in enumerate(doc_edges, start=1):
        evidence_units = gather_evidence_units_for_edge(
            store=store,
            edge=edge,
            unit_rows=unit_rows,
            unit_by_id=unit_by_id,
            chunk_by_id=chunk_by_id,
            lexical_top_k=cfg.lexical_top_k,
            include_chunk_fallback=cfg.include_chunk_fallback,
        )
        explicit_ptr = bool(evidence_units and relation_unit_ids(edge.rel_props))
        chunk_distance = None
        if edge.source_chunk_pos is not None and edge.target_chunk_pos is not None:
            try:
                chunk_distance = abs(int(edge.source_chunk_pos) - int(edge.target_chunk_pos))
            except Exception:
                chunk_distance = None

        decision = labeler.decide(
            source_node_id=edge.source_node_id,
            source_text=edge.source_text,
            source_layer=edge.source_layer,
            target_node_id=edge.target_node_id,
            target_text=edge.target_text,
            target_layer=edge.target_layer,
            relation_type=edge.relation_type,
            rel_props=edge.rel_props,
            evidence_units=evidence_units,
            explicit_evidence_pointer=explicit_ptr,
            chunk_distance=chunk_distance,
        )

        if decision.label is None:
            ambiguous += 1
            if progress_hook is not None:
                progress_hook(
                    {
                        "stage": "edge_decision",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "edge_index": edge_idx,
                        "num_candidate_edges": len(doc_edges),
                        "source_text": edge.source_text,
                        "relation_type": edge.relation_type,
                        "target_text": edge.target_text,
                        "label": None,
                        "confidence": float(decision.confidence),
                        "primary_rule": decision.primary_rule,
                    }
                )
            if not cfg.store_ambiguous:
                continue

        if decision.label is not None:
            row = build_pseudo_label_record(edge, decision)
            label_rows.append(row)
            if progress_hook is not None:
                progress_hook(
                    {
                        "stage": "edge_decision",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "edge_index": edge_idx,
                        "num_candidate_edges": len(doc_edges),
                        "source_text": edge.source_text,
                        "relation_type": row.get("relation_type"),
                        "target_text": edge.target_text,
                        "label": int(row.get("label") or 0),
                        "confidence": float(row.get("confidence") or 0.0),
                        "primary_rule": (row.get("rule_hits") or {}).get("primary_rule"),
                    }
                )

    if label_rows:
        store.upsert_pseudo_edge_labels(label_rows)

    out_dir = Path(cfg.export_root) / sanitize_fs_part(doc_id or file_name)
    manifest = export_pseudo_label_package(
        out_dir,
        label_rows,
        config=cfg,
        manifest_extra={
            "doc_id": doc_id,
            "file_name": file_name,
            "kg_scope": kg_scope,
            "kg_id": kg_id,
            "num_candidate_edges": len(doc_edges),
            "num_evidence_units_scanned": len(unit_rows),
            "ambiguous_edges": ambiguous,
            "configs": {
                "prior_config": cfg.prior_config_path,
                "pseudo_rule_config": cfg.pseudo_rule_config_path,
            },
        },
    )
    preview = label_rows[: max(0, int(console_preview_limit or 0))]
    return {"ok": True, "export_dir": str(out_dir), **manifest,"preview":preview}
    if progress_hook is not None:
        progress_hook(
            {
                "stage": "doc_summary",
                "doc_id": doc_id,
                "file_name": file_name,
                "num_candidate_edges": len(doc_edges),
                "num_pseudo_labels":len(label_rows),
                "ambiguous_edges": ambiguous,
                "label_breakdown": manifest.get("label_breakdown",{}),
            }
        )
