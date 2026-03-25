from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .candidate_generator import CandidateGenerator, CandidateGeneratorConfig
from .counterfactual_sampler import CounterfactualSampler
from .neo4j_accessor import CandidateEdge, Neo4jAccessor
from .pseudo_labeler import (
    CausalPseudoLabeler,
    lexical_overlap,
    load_yaml_file,
    merge_prior_and_pseudo_config,
    relation_confidence_from_props,
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


def edge_dedup_key(edge: CandidateEdge) -> Tuple[str, str, str]:
    return (
        str(edge.source_node_id or "").strip(),
        str(edge.relation_type or "").strip().upper(),
        str(edge.target_node_id or "").strip(),
    )


def edge_quality_score(edge: CandidateEdge) -> float:
    rel_props = edge.rel_props or {}
    score = 0.0
    score += relation_confidence_from_props(rel_props)
    evidence_text = (
        getattr(edge, "evidence_text", None)
        or rel_props.get("evidence_text")
        or rel_props.get("support_text")
        or rel_props.get("text")
    )
    if evidence_text:
        score += 0.1
    if relation_unit_ids(rel_props):
        score += 0.1
    if edge.source_chunk_pos is not None and edge.target_chunk_pos is not None:
        score += 0.05
    return float(score)


def deduplicate_candidate_edges(edges: Sequence[CandidateEdge]) -> Tuple[List[CandidateEdge], int]:
    """Deduplicate repeated candidate edges by (src, rel, dst)."""
    best_by_key: Dict[Tuple[str, str, str], CandidateEdge] = {}
    dropped = 0
    for edge in edges:
        key = edge_dedup_key(edge)
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = edge
            continue
        if edge_quality_score(edge) > edge_quality_score(prev):
            best_by_key[key] = edge
        dropped += 1
    return list(best_by_key.values()), dropped


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
        "relation_type": edge.relation_type,
        "silver_edge_causal": int(decision.silver_edge_causal),
        "causal_conf": float(decision.causal_conf),
        "silver_edge_enable": int(decision.silver_edge_enable),
        "enable_conf": float(decision.enable_conf),
        "silver_causal_dir": int(decision.silver_causal_dir),
        "dir_conf": float(decision.dir_conf),
        "silver_temporal_before": int(decision.silver_temporal_before),
        "temporal_conf": float(decision.temporal_conf),
        "silver_node_first_src": int(decision.silver_node_first_src),
        "src_first_conf": float(decision.src_first_conf),
        "silver_node_first_dst": int(decision.silver_node_first_dst),
        "dst_first_conf": float(decision.dst_first_conf),
        "evidence_unit_id": decision.evidence_unit_id,
        "evidence_start": None,
        "evidence_end": None,
        "evidence_text": decision.evidence_text,
        "rule_hits": decision.rule_hits,
        "features": decision.features,
        "sample_weight": float(decision.sample_weight),
        "twin_group_id": decision.twin_group_id,
        "review_status": decision.review_status,
        "label_source": "pseudo_multitask",
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
        primary_rule = ((row.get("rule_hits") or {}).get("edge_causal") or [None])[0]
        key = (
            row.get("silver_edge_causal"),
            primary_rule,
            row.get("relation_type"),
        )
        by_group[key].append(row)

    for rows in by_group.values():
        rows.sort(key=lambda r: float(r.get("causal_conf") or 0.0))

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


def normalize_entity_surface(text: Optional[str]) -> str:
    raw = str(text or "").strip().lower()
    if "|" in raw:
        raw = raw.split("|")[-1].strip()
    raw = " ".join(raw.split())
    return raw


def is_trivial_or_self_edge(edge: CandidateEdge, *, text_overlap_threshold: float = 0.95) -> bool:
    if str(edge.source_node_id or "") == str(edge.target_node_id or ""):
        return True
    src_norm = normalize_entity_surface(edge.source_text)
    tgt_norm = normalize_entity_surface(edge.target_text)
    if src_norm and tgt_norm and src_norm == tgt_norm:
        return True
    return lexical_overlap(src_norm, tgt_norm) >= float(text_overlap_threshold)


def print_pseudo_label_rows_console(
    rows: Sequence[Dict[str, Any]],
    *,
    limit: int = 20,
) -> None:
    """Print a compact multi-task pseudo-label preview to console."""
    if not rows:
        print("[pseudo-label] no rows generated.")
        return
    show_n = min(max(int(limit), 0), len(rows))
    if show_n <= 0:
        return
    print(f"[pseudo-label] showing {show_n}/{len(rows)} generated edges:")
    for idx, row in enumerate(rows[:show_n], start=1):
        src = str(row.get("source_text") or row.get("source_node_id") or "")
        rel = str(row.get("relation_type") or "")
        tgt = str(row.get("target_text") or row.get("target_node_id") or "")
        print(
            f"  [{idx:03d}] {src} --{rel}--> {tgt} | "
            f"causal={row.get('silver_edge_causal')}({float(row.get('causal_conf') or 0.0):.2f}) "
            f"enable={row.get('silver_edge_enable')}({float(row.get('enable_conf') or 0.0):.2f}) "
            f"dir={row.get('silver_causal_dir')}({float(row.get('dir_conf') or 0.0):.2f}) "
            f"temp={row.get('silver_temporal_before')}({float(row.get('temporal_conf') or 0.0):.2f}) "
            f"src_first={row.get('silver_node_first_src')}({float(row.get('src_first_conf') or 0.0):.2f}) "
            f"dst_first={row.get('silver_node_first_dst')}({float(row.get('dst_first_conf') or 0.0):.2f})"
        )


@dataclass
class AutoPseudoPipelineConfig:
    enabled: bool = True
    export_root: str = "./artifacts/manual_review"
    evidence_db_path: str = "./data/evidence_store.sqlite3"
    prior_config_path: str = "configs/causal_prior.yaml"
    pseudo_rule_config_path: str = "configs/causal_pseudo_labe_rules.yaml"
    store_ambiguous: bool = False
    max_edges_per_doc: Optional[int] = None
    max_units_per_doc_scan: int = 500
    lexical_top_k: int = 5
    include_chunk_fallback: bool = True
    candidate_generator_config_path: str = "configs/casual_candidate_generator.yaml"
    review_sample_size: int = 200
    review_per_doc_cap: int = 200
    review_per_rule_cap: int = 50
    skip_trivial_edges: bool = True
    trivial_edge_overlap_threshold: float = 0.95

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
            pseudo_rule_config_path=os.getenv("AUTO_PSEUDO_LABEL_RULE_CONFIG", "configs/causal_pseudo_labe_rules.yaml"),
            store_ambiguous=_get_bool("AUTO_PSEUDO_LABEL_STORE_AMBIGUOUS", False),
            max_edges_per_doc=_get_int("AUTO_PSEUDO_LABEL_MAX_EDGES_PER_DOC", None),
            max_units_per_doc_scan=_get_int("AUTO_PSEUDO_LABEL_MAX_UNITS_PER_DOC_SCAN", 500) or 500,
            lexical_top_k=_get_int("AUTO_PSEUDO_LABEL_LEXICAL_TOP_K", 5) or 5,
            include_chunk_fallback=_get_bool("AUTO_PSEUDO_LABEL_INCLUDE_CHUNK_FALLBACK", True),
            candidate_generator_config_path=os.getenv("AUTO_PSEUDO_LABEL_CANDIDATE_GENERATOR_CONFIG", "configs/casual_candidate_generator.yaml"),
            review_sample_size=_get_int("AUTO_PSEUDO_LABEL_REVIEW_SAMPLE_SIZE", 200) or 200,
            review_per_doc_cap=_get_int("AUTO_PSEUDO_LABEL_REVIEW_PER_DOC_CAP", 200) or 200,
            review_per_rule_cap=_get_int("AUTO_PSEUDO_LABEL_REVIEW_PER_RULE_CAP", 50) or 50,
            skip_trivial_edges=_get_bool("AUTO_PSEUDO_LABEL_SKIP_TRIVIAL_EDGES", True),
            trivial_edge_overlap_threshold=float(os.getenv("AUTO_PSEUDO_LABEL_TRIVIAL_EDGE_OVERLAP", "0.95")),
        )


def export_pseudo_label_package(
    out_dir: Path,
    label_rows: Sequence[Dict[str, Any]],
    *,
    config: AutoPseudoPipelineConfig,
    manifest_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_table_jsonl = out_dir / "candidate_edge_table.jsonl"
    node_prior_jsonl = out_dir / "candidate_node_prior_table.jsonl"
    cf_jsonl = out_dir / "counterfactual_pairs.jsonl"
    train_jsonl = out_dir / "jointlk_multitask_train.jsonl"
    review_csv = out_dir / "manual_review_candidates.csv"

    with edge_table_jsonl.open("w", encoding="utf-8") as f:
        for row in label_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    node_rows: Dict[str, Dict[str, Any]] = {}
    for row in label_rows:
        src_key = f"{row.get('doc_id')}::{row.get('source_node_id')}"
        dst_key = f"{row.get('doc_id')}::{row.get('target_node_id')}"
        node_rows[src_key] = {
            "doc_id": row.get("doc_id"),
            "node_id": row.get("source_node_id"),
            "node_text": row.get("source_text"),
            "node_layer": row.get("source_layer"),
            "silver_node_first": row.get("silver_node_first_src"),
            "first_conf": row.get("src_first_conf"),
        }
        node_rows[dst_key] = {
            "doc_id": row.get("doc_id"),
            "node_id": row.get("target_node_id"),
            "node_text": row.get("target_text"),
            "node_layer": row.get("target_layer"),
            "silver_node_first": row.get("silver_node_first_dst"),
            "first_conf": row.get("dst_first_conf"),
        }

    with node_prior_jsonl.open("w", encoding="utf-8") as f:
        for row in node_rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cf_sampler = CounterfactualSampler(getattr(config, "counterfactual", None))
    cf_rows = cf_sampler.build_pairs(label_rows)
    with cf_jsonl.open("w", encoding="utf-8") as f:
        for row in cf_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with train_jsonl.open("w", encoding="utf-8") as f:
        for row in label_rows:
            flat = dict(row)
            flat["causal_labels"] = row.get("silver_edge_causal", -1)
            flat["enable_labels"] = row.get("silver_edge_enable", -1)
            flat["dir_labels"] = row.get("silver_causal_dir", -1)
            flat["temp_labels"] = row.get("silver_temporal_before", -1)
            flat["src_first_labels"] = row.get("silver_node_first_src", -1)
            flat["dst_first_labels"] = row.get("silver_node_first_dst", -1)
            flat["causal_mask"] = 0 if int(flat["causal_labels"]) == -1 else 1
            flat["enable_mask"] = 0 if int(flat["enable_labels"]) == -1 else 1
            flat["dir_mask"] = 0 if int(flat["dir_labels"]) == -1 else 1
            flat["temp_mask"] = 0 if int(flat["temp_labels"]) == -1 else 1
            flat["src_first_mask"] = 0 if int(flat["src_first_labels"]) == -1 else 1
            flat["dst_first_mask"] = 0 if int(flat["dst_first_labels"]) == -1 else 1
            flat["label"] = 1 if int(flat["causal_labels"]) == 1 else 0
            flat["label_confidence"] = float(row.get("causal_conf", 0.0))
            f.write(json.dumps(flat, ensure_ascii=False) + "\n")

    review_candidates = choose_review_candidates(
        label_rows,
        sample_size=min(len(label_rows), config.review_sample_size),
        per_doc_cap=config.review_per_doc_cap,
        per_rule_cap=config.review_per_rule_cap,
    )
    review_fields = [
        "pseudo_label_id",
        "doc_id",
        "file_name",
        "source_node_id",
        "source_text",
        "target_node_id",
        "target_text",
        "relation_type",
        "silver_edge_causal",
        "causal_conf",
        "silver_edge_enable",
        "enable_conf",
        "silver_causal_dir",
        "dir_conf",
        "silver_temporal_before",
        "temporal_conf",
        "silver_node_first_src",
        "src_first_conf",
        "silver_node_first_dst",
        "dst_first_conf",
        "rule_hits_json",
        "sample_weight",
        "twin_group_id",
        "manual_decision",
        "manual_comment",
        "reviewer",
    ]
    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_fields)
        writer.writeheader()
        for row in review_candidates:
            writer.writerow(
                {
                    "pseudo_label_id": row.get("pseudo_label_id"),
                    "doc_id": row.get("doc_id"),
                    "file_name": row.get("file_name"),
                    "source_node_id": row.get("source_node_id"),
                    "source_text": row.get("source_text"),
                    "target_node_id": row.get("target_node_id"),
                    "target_text": row.get("target_text"),
                    "relation_type": row.get("relation_type"),
                    "silver_edge_causal": row.get("silver_edge_causal"),
                    "causal_conf": row.get("causal_conf"),
                    "silver_edge_enable": row.get("silver_edge_enable"),
                    "enable_conf": row.get("enable_conf"),
                    "silver_causal_dir": row.get("silver_causal_dir"),
                    "dir_conf": row.get("dir_conf"),
                    "silver_temporal_before": row.get("silver_temporal_before"),
                    "temporal_conf": row.get("temporal_conf"),
                    "silver_node_first_src": row.get("silver_node_first_src"),
                    "src_first_conf": row.get("src_first_conf"),
                    "silver_node_first_dst": row.get("silver_node_first_dst"),
                    "dst_first_conf": row.get("dst_first_conf"),
                    "rule_hits_json": json.dumps(row.get("rule_hits") or {}, ensure_ascii=False),
                    "sample_weight": row.get("sample_weight"),
                    "twin_group_id": row.get("twin_group_id"),
                    "manual_decision": "",
                    "manual_comment": "",
                    "reviewer": "",
                }
            )

    manifest = {
        "num_pseudo_labels": len(label_rows),
        "task_coverage": {
            "edge_causal": sum(1 for x in label_rows if int(x.get("silver_edge_causal", -1)) != -1) / max(1, len(label_rows)),
            "edge_enable": sum(1 for x in label_rows if int(x.get("silver_edge_enable", -1)) != -1) / max(1, len(label_rows)),
            "causal_dir": sum(1 for x in label_rows if int(x.get("silver_causal_dir", -1)) != -1) / max(1, len(label_rows)),
            "temporal_before": sum(1 for x in label_rows if int(x.get("silver_temporal_before", -1)) != -1) / max(1, len(label_rows)),
            "node_first_src": sum(1 for x in label_rows if int(x.get("silver_node_first_src", -1)) != -1) / max(1, len(label_rows)),
            "node_first_dst": sum(1 for x in label_rows if int(x.get("silver_node_first_dst", -1)) != -1) / max(1, len(label_rows)),
        },
        "abstain_rate": {
            "edge_causal": sum(1 for x in label_rows if int(x.get("silver_edge_causal", -1)) == -1) / max(1, len(label_rows)),
            "edge_enable": sum(1 for x in label_rows if int(x.get("silver_edge_enable", -1)) == -1) / max(1, len(label_rows)),
            "causal_dir": sum(1 for x in label_rows if int(x.get("silver_causal_dir", -1)) == -1) / max(1, len(label_rows)),
            "temporal_before": sum(1 for x in label_rows if int(x.get("silver_temporal_before", -1)) == -1) / max(1, len(label_rows)),
            "node_first_src": sum(1 for x in label_rows if int(x.get("silver_node_first_src", -1)) == -1) / max(1, len(label_rows)),
            "node_first_dst": sum(1 for x in label_rows if int(x.get("silver_node_first_dst", -1)) == -1) / max(1, len(label_rows)),
        },
        "review_candidate_count": len(review_candidates),
        "paths": {
            "candidate_edge_table_jsonl": str(edge_table_jsonl),
            "candidate_node_prior_table_jsonl": str(node_prior_jsonl),
            "counterfactual_pairs_jsonl": str(cf_jsonl),
            "jointlk_multitask_train_jsonl": str(train_jsonl),
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

    unit_rows, unit_by_id, chunk_by_id = read_doc_evidence_units(
        store,
        file_name=file_name,
        max_units=cfg.max_units_per_doc_scan,
    )

    candidate_gen_cfg_raw = load_yaml_file(cfg.candidate_generator_config_path) if cfg.candidate_generator_config_path and os.path.exists(cfg.candidate_generator_config_path) else {}
    candidate_generator = CandidateGenerator(accessor, CandidateGeneratorConfig(**(candidate_gen_cfg_raw or {})))
    doc_edges, candidate_stats = candidate_generator.generate_for_doc(
        doc_id=doc_id,
        file_name=file_name,
        kg_scope=kg_scope or "instance",
        kg_id=kg_id,
        relation_types=None,
        limit=cfg.max_edges_per_doc,
        unit_rows=unit_rows,
    )
    raw_candidate_edges = len(doc_edges)
    doc_edges, dedup_dropped_edges = deduplicate_candidate_edges(doc_edges)

    label_rows: List[Dict[str, Any]] = []
    task_abstain_counts = defaultdict(int)
    skipped_trivial_edges = 0
    for edge_idx, edge in enumerate(doc_edges, start=1):
        if cfg.skip_trivial_edges and is_trivial_or_self_edge(
            edge,
            text_overlap_threshold=cfg.trivial_edge_overlap_threshold,
        ):
            skipped_trivial_edges += 1
            if progress_hook is not None:
                progress_hook(
                    {
                        "stage": "edge_skip",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "edge_index": edge_idx,
                        "source_text": edge.source_text,
                        "relation_type": edge.relation_type,
                        "target_text": edge.target_text,
                        "reason": "trivial_or_self_edge",
                    }
                )
            continue

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

        decision = labeler.decide_multitask(
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

        row = build_pseudo_label_record(edge, decision)
        label_rows.append(row)
        for task_field in [
            "silver_edge_causal",
            "silver_edge_enable",
            "silver_causal_dir",
            "silver_temporal_before",
            "silver_node_first_src",
            "silver_node_first_dst",
        ]:
            if int(row.get(task_field, -1)) == -1:
                task_abstain_counts[task_field] += 1
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
                    "silver_edge_causal": row.get("silver_edge_causal"),
                    "causal_conf": float(row.get("causal_conf") or 0.0),
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
            "num_candidate_edges_before_dedup": raw_candidate_edges,
            "dedup_dropped_edges": dedup_dropped_edges,
            "num_evidence_units_scanned": len(unit_rows),
            "skipped_trivial_edges": skipped_trivial_edges,
            "abstain_counts": dict(task_abstain_counts),
            "candidate_generation": candidate_stats,
            "configs": {
                "prior_config": cfg.prior_config_path,
                "pseudo_rule_config": cfg.pseudo_rule_config_path,
                "candidate_generator_config": cfg.candidate_generator_config_path,
            },
        },
    )
    preview_limit = max(0, int(console_preview_limit or 0))
    preview = label_rows[:preview_limit]
    if preview_limit > 0:
        print_pseudo_label_rows_console(label_rows, limit=preview_limit)
    if progress_hook is not None:
        progress_hook(
            {
                "stage": "doc_summary",
                "doc_id": doc_id,
                "file_name": file_name,
                "num_candidate_edges": len(doc_edges),
                "num_candidate_edges_before_dedup": raw_candidate_edges,
                "dedup_dropped_edges": dedup_dropped_edges,
                "num_pseudo_labels": len(label_rows),
                "skipped_trivial_edges": skipped_trivial_edges,
                "abstain_counts": dict(task_abstain_counts),
                "task_coverage": manifest.get("task_coverage", {}),
            }
        )
    return {"ok": True, "export_dir": str(out_dir), **manifest, "preview": preview}