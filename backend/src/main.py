from langchain_community.graphs import Neo4jGraph
from src.shared.constants import BUCKET_UPLOAD, PROJECT_ID
from src.shared.schema_extraction import schema_extraction_from_text
from dotenv import load_dotenv
from datetime import datetime
import logging
from src.create_chunks import CreateChunksofDocument
from src.graphDB_dataAccess import graphDBdataAccess
from src.document_sources.local_file import get_documents_from_file_by_path
from src.entities.source_node import sourceNode
from src.generate_graphDocuments_from_llm import generate_graphDocuments
from src.document_sources.gcs_bucket import *
from src.document_sources.s3_bucket import *
from src.document_sources.wikipedia import *
from src.document_sources.youtube import *
from src.shared.common_fn import *
from src.make_relationships import *
from src.evidence_unit_builder import EvidenceUnitBuilder
from src.document_sources.web_pages import *
from src.kg_content import build_kg_context, scope_graph_documents
from causal_jointlk.pseudo_pipeline import run_pseudo_label_pipeline_for_doc, AutoPseudoPipelineConfig
from src.graph_export import generate_gpickle_export,export_jointlk_json_artifacts

import re
from langchain_community.document_loaders import WikipediaLoader, WebBaseLoader
import warnings
from pytube import YouTube
import sys
import shutil
import urllib.parse
import json
import subprocess
import threading
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(message)s', level='INFO')



def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _truncate_console_text(text, max_len: int = 28) -> str:
    value = str(text or "")
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _resolve_runtime_path(path_value: str, repo_root: Path | None = None) -> Path:
    """
    Resolve mixed slash paths (Windows/Unix) into a valid runtime path.
    It also tries repo-root relative fallback for relative artifact paths.
    """
    raw = str(path_value or "").strip()
    if not raw:
        return Path("")

    candidates = [
        Path(raw),
        Path(raw.replace("\\", "/")),
        Path(raw.replace("/", "\\")),
    ]
    if repo_root is not None:
        repo_root = Path(repo_root).resolve()
        candidates.extend([repo_root / c for c in candidates if not c.is_absolute()])

    for c in candidates:
        if c.exists():
            return c.resolve()

    fallback = Path(raw.replace("\\", os.sep).replace("/", os.sep))
    if not fallback.is_absolute() and repo_root is not None:
        return (repo_root / fallback).resolve()
    return fallback.resolve() if fallback.is_absolute() else fallback

def _sanitize_fs_part(value: str) -> str:
    """
    Local safe filename sanitizer.
    Avoid depending on external helper availability across branches/environments.
    """
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    return text[:180] if len(text) > 180 else text

def _log_jointlk_stage(stage: str, payload: dict) -> None:
    """Unified end-to-end console logger for JointLK pipeline stages."""
    try:
        message = json.dumps(payload, ensure_ascii=False)
    except Exception:
        message = str(payload)
    logging.info("[JointLK][%s] %s", stage, message)

def _build_pseudo_console_hook(show_edge_process: bool):
    def _hook(event):
        stage = event.get("stage")
        if stage == "edge_decision" and show_edge_process:
            _log_jointlk_stage(
                "pseudo-edge-decision",
                {
                    "edge_index": event.get("edge_index"),
                    "num_candidate_edges": event.get("num_candidate_edges"),
                    "source_text": _truncate_console_text(event.get("source_text")),
                    "relation_type": event.get("relation_type") or "?",
                    "target_text": _truncate_console_text(event.get("target_text")),
                    "label": event.get("label"),
                    "confidence": float(event.get("confidence") or 0.0),
                    "primary_rule": event.get("primary_rule") or "-",
                },
            )
            return
        if stage == "doc_summary":
            breakdown = event.get("label_breakdown") or {}
            _log_jointlk_stage(
                "pseudo-doc-summary",
                {
                    "doc_id": event.get("doc_id"),
                    "file_name": event.get("file_name"),
                    "num_candidate_edges": event.get("num_candidate_edges"),
                    "num_pseudo_labels": event.get("num_pseudo_labels"),
                    "positive": breakdown.get("positive", 0),
                    "negative": breakdown.get("negative", 0),
                    "ambiguous_edges": event.get("ambiguous_edges", 0),
                },
            )

    return _hook

def maybe_run_auto_pseudo_label_pipeline(graph, *, file_name: str, doc_id: str, kg_scope: str, kg_id: str):
    cfg = AutoPseudoPipelineConfig.from_env()
    if not cfg.enabled:
        logging.info("[pseudo-label] skipped because AUTO_PSEUDO_LABEL_AFTER_UPLOAD is disabled")
        return None

    console_preview = _get_int_env("AUTO_PSEUDO_LABEL_CONSOLE_PREVIEW", 10)
    show_edge_process = _get_bool_env("AUTO_PSEUDO_LABEL_SHOW_EDGE_PROCESS", True)
    progress_hook = _build_pseudo_console_hook(show_edge_process)

    logging.info(
        "[pseudo-label] start for file=%s doc_id=%s | show_edge_process=%s console_preview=%s",
        file_name,
        doc_id,
        show_edge_process,
        console_preview,
    )
    result = run_pseudo_label_pipeline_for_doc(
        graph=graph,
        file_name=file_name,
        doc_id=doc_id,
        kg_scope=kg_scope,
        kg_id=kg_id,
        config=cfg,
        progress_hook=progress_hook,
        console_preview_limit=console_preview,
    )

    preview_rows = result.get("preview") or []
    if preview_rows:
        logging.info("[pseudo-label][preview] top %s rows for file=%s", len(preview_rows), file_name)
        for idx, row in enumerate(preview_rows, start=1):
            # 兼容旧版单标签字段与新版多任务字段。
            legacy_label = row.get("label")
            legacy_conf = float(row.get("confidence") or 0.0)
            legacy_rule = (row.get("rule_hits") or {}).get("primary_rule") or "-"

            causal_label = row.get("silver_edge_causal", legacy_label)
            causal_conf = float(row.get("causal_conf") or legacy_conf)
            enable_label = row.get("silver_edge_enable")
            enable_conf = float(row.get("enable_conf") or 0.0)
            dir_label = row.get("silver_causal_dir")
            dir_conf = float(row.get("dir_conf") or 0.0)
            temp_label = row.get("silver_temporal_before")
            temp_conf = float(row.get("temporal_conf") or 0.0)
            src_first_label = row.get("silver_node_first_src")
            src_first_conf = float(row.get("src_first_conf") or 0.0)
            dst_first_label = row.get("silver_node_first_dst")
            dst_first_conf = float(row.get("dst_first_conf") or 0.0)

            logging.info(
                "[pseudo-label][preview #%s] %s --%s--> %s | "
                "silver_edge_causal=%s(causal_conf=%.3f) "
                "silver_edge_enable=%s(enable_conf=%.3f) "
                "silver_causal_dir=%s(dir_conf=%.3f) "
                "silver_temporal_before=%s(temporal_conf=%.3f) "
                "silver_node_first_src=%s(src_first_conf=%.3f) "
                "silver_node_first_dst=%s(dst_first_conf=%.3f) "
                "sample_weight=%.3f twin_group_id=%s review_status=%s | rule=%s",
                idx,
                _truncate_console_text(row.get("source_text")),
                row.get("relation_type") or "?",
                _truncate_console_text(row.get("target_text")),
                causal_label,
                causal_conf,
                enable_label,
                enable_conf,
                dir_label,
                dir_conf,
                temp_label,
                temp_conf,
                src_first_label,
                src_first_conf,
                dst_first_label,
                dst_first_conf,
                float(row.get("sample_weight") or 0.0),
                row.get("twin_group_id"),
                row.get("review_status") or "pending",
                legacy_rule,
            )
    else:
        logging.info("[pseudo-label][preview] no pseudo-label generated for file=%s", file_name)

    return result
def _auto_jointlk_train_enabled() -> bool:
    return _get_bool_env("AUTO_JOINTLK_TRAIN_AFTER_UPLOAD", True)

def _build_jointlk_train_attempts(model_name: str, batch_size: int):
    """
    Build a conservative fallback plan to improve training success rate.
    Typical failure case in production is GPU OOM on large backbone + large batch.
    """
    normalized = str(model_name or "").strip() or "roberta-large"
    base_batch = max(1, int(batch_size))

    attempts = [
        {
            "model_name": normalized,
            "batch_size": base_batch,
            "freeze_lm": False,
            "force_cpu": False,
            "reason": "primary",
        }
    ]

    # GPU-memory-friendly fallback: smaller batch + freeze LM encoder
    attempts.append(
        {
            "model_name": normalized,
            "batch_size": max(1, base_batch // 2),
            "freeze_lm": True,
            "force_cpu": False,
            "reason": "retry_freeze_lm_and_smaller_batch",
        }
    )

    # Lighter model fallback for most memory-related failures
    if normalized != "roberta-base":
        attempts.append(
            {
                "model_name": "roberta-base",
                "batch_size": 1,
                "freeze_lm": True,
                "force_cpu": False,
                "reason": "retry_smaller_backbone",
            }
        )

    # Final safety fallback: CPU path (slow but much less likely to OOM)
    attempts.append(
        {
            "model_name": "distilroberta-base",
            "batch_size": 1,
            "freeze_lm": True,
            "force_cpu": True,
            "reason": "retry_cpu_safe_mode",
        }
    )

    # de-duplicate by effective config
    deduped = []
    seen = set()
    for it in attempts:
        key = (it["model_name"], int(it["batch_size"]), bool(it["freeze_lm"]), bool(it["force_cpu"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped

def _tail_text_file(path: Path, max_lines: int = 120) -> str:
    try:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if max_lines <= 0:
            return "\n".join(lines)
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""


def _jointlk_failure_hint(log_tail: str) -> str:
    text = (log_tail or "").lower()
    if not text:
        return "unknown_error"
    if "out of memory" in text or "cuda oom" in text:
        return "cuda_oom"
    if "no module named" in text:
        return "missing_python_dependency"
    if "can't load tokenizer" in text or "cannot load tokenizer" in text:
        return "tokenizer_load_failed"
    if "401 client error" in text or "403 client error" in text:
        return "model_download_auth_error"
    if "not found" in text and "jsonl" in text:
        return "dataset_file_not_found"
    if "assertionerror" in text:
        return "assertion_failed"
    if "runtimeerror" in text and "cuda" in text:
        return "cuda_runtime_error"
    if "valueerror" in text:
        return "value_error"
    return "unclassified_runtime_error"

def maybe_trigger_auto_jointlk_training(*, pseudo_result: dict, file_name: str, doc_id: str):
    """
    Trigger causal JointLK training automatically after pseudo labels are generated.
    The training process runs asynchronously in a background thread.
    """
    if not _auto_jointlk_train_enabled():
        logging.info("[jointlk-train] skipped because AUTO_JOINTLK_TRAIN_AFTER_UPLOAD is disabled")
        return {"enabled": False, "started": False, "reason": "disabled"}

    if not pseudo_result or not pseudo_result.get("ok"):
        logging.info("[jointlk-train] skipped because pseudo pipeline result is not ok")
        return {"enabled": True, "started": False, "reason": "pseudo_not_ok"}

    num_pseudo_labels = int(pseudo_result.get("num_pseudo_labels") or 0)
    if num_pseudo_labels <= 0:
        logging.info("[jointlk-train] skipped because no pseudo labels were generated")
        return {"enabled": True, "started": False, "reason": "no_pseudo_labels"}

    manifest_paths = pseudo_result.get("paths") or {}
    train_jsonl = manifest_paths.get("jointlk_multitask_train_jsonl")
    if not train_jsonl:
        logging.info("[jointlk-train] skipped because train_jsonl is missing")
        return {"enabled": True, "started": False, "reason": "missing_train_jsonl"}

    repo_root = Path(__file__).resolve().parents[2]
    train_jsonl_path = _resolve_runtime_path(train_jsonl, repo_root=repo_root)
    if not train_jsonl_path.exists():
        logging.warning(
            "[jointlk-train] skipped because train_jsonl does not exist at runtime: %s (raw=%s)",
            str(train_jsonl_path),
            str(train_jsonl),
        )
        return {
            "enabled": True,
            "started": False,
            "reason": "train_jsonl_not_found",
            "train_jsonl": str(train_jsonl),
            "resolved_train_jsonl": str(train_jsonl_path),
        }

    train_script = repo_root / "experiments" / "causal_jointlk" / "train_causal_jointlk.py"
    if not train_script.exists():
        logging.warning("[jointlk-train] script not found: %s", train_script)
        return {"enabled": True, "started": False, "reason": "missing_train_script"}

    output_root = os.getenv("AUTO_JOINTLK_TRAIN_OUTPUT_ROOT", "./outputs/auto_jointlk_training")
    output_dir = Path(output_root) / _sanitize_fs_part(doc_id or file_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"

    dev_jsonl = os.getenv("AUTO_JOINTLK_TRAIN_DEV_JSONL", str(train_jsonl_path))
    dev_jsonl_path = _resolve_runtime_path(dev_jsonl, repo_root=repo_root)
    if not dev_jsonl_path.exists():
        logging.info(
            "[jointlk-train] dev_jsonl not found (%s), fallback to train_jsonl=%s",
            str(dev_jsonl_path),
            str(train_jsonl_path),
        )
        dev_jsonl_path = train_jsonl_path

    model_name = os.getenv("AUTO_JOINTLK_TRAIN_MODEL_NAME", "roberta-large")
    prior_config = os.getenv("AUTO_JOINTLK_TRAIN_PRIOR_CONFIG", "configs/causal_prior.yaml")
    epochs = str(_get_int_env("AUTO_JOINTLK_TRAIN_EPOCHS", 3))
    batch_size_int = _get_int_env("AUTO_JOINTLK_TRAIN_BATCH_SIZE", 4)
    batch_size = str(batch_size_int)
    lr = os.getenv("AUTO_JOINTLK_TRAIN_LR", "2e-5")
    attempt_plan = _build_jointlk_train_attempts(model_name=model_name, batch_size=batch_size_int)

    def _build_train_command(*, model_name: str, batch_size: int, freeze_lm: bool):
        command = [
            sys.executable,
            str(train_script),
            "--train_jsonl",
            str(train_jsonl_path),
            "--dev_jsonl",
            str(dev_jsonl_path),
            "--prior_config",
            str(prior_config),
            "--model_name",
            str(model_name),
            "--output_dir",
            str(output_dir),
            "--epochs",
            epochs,
            "--batch_size",
            str(max(1, int(batch_size))),
            "--lr",
            lr,
        ]
        if freeze_lm:
            command.append("--freeze_lm")
        return command

    def _run():
        with log_file.open("a", encoding="utf-8") as fout:
            fout.write(
                f"\n\n=== auto jointlk training started at {datetime.now().isoformat()} "
                f"for doc_id={doc_id} file_name={file_name} ===\n"
            )
            final_code = 1
            for idx, attempt in enumerate(attempt_plan, start=1):
                attempt_cmd = _build_train_command(
                    model_name=str(attempt["model_name"]),
                    batch_size=int(attempt["batch_size"]),
                    freeze_lm=bool(attempt["freeze_lm"]),
                )
                env = os.environ.copy()
                if bool(attempt.get("force_cpu")):
                    env["CUDA_VISIBLE_DEVICES"] = ""
                env.setdefault("PYTHONIOENCODING", "utf-8")

                fout.write(
                    f"\n--- attempt {idx}/{len(attempt_plan)} "
                    f"reason={attempt.get('reason')} model={attempt.get('model_name')} "
                    f"batch_size={attempt.get('batch_size')} freeze_lm={attempt.get('freeze_lm')} "
                    f"force_cpu={attempt.get('force_cpu')} ---\n"
                )
                fout.write("CMD: " + " ".join(attempt_cmd) + "\n")
                fout.flush()
                proc = subprocess.Popen(
                    attempt_cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )
                if proc.stdout is not None:
                    for raw_line in proc.stdout:
                        fout.write(raw_line)
                        if _should_echo_train_progress(raw_line):
                            logging.info(
                                "[jointlk-train][progress] doc_id=%s file_name=%s %s",
                                doc_id,
                                file_name,
                                raw_line.rstrip(),
                            )
                    proc.stdout.close()
                final_code = proc.wait()
                fout.write(f"--- attempt {idx} finished with exit_code={final_code} ---\n")
                fout.flush()
                if final_code == 0:
                    break

            fout.write(f"\n=== auto jointlk training finished with exit_code={final_code} ===\n")
            fout.flush()
            log_tail = _tail_text_file(log_file, max_lines=160)
            failure_hint = _jointlk_failure_hint(log_tail)
            status_payload = {
                "doc_id": doc_id,
                "file_name": file_name,
                "exit_code": int(final_code),
                "failure_hint": failure_hint,
                "finished_at": datetime.now().isoformat(),
                "log_file": str(log_file),
            }
            try:
                (output_dir / "train_status.json").write_text(
                    json.dumps(status_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                logging.exception("[jointlk-train] failed to write train_status.json for doc_id=%s", doc_id)
            logging.info(
                "[jointlk-train] finished for doc_id=%s file_name=%s exit_code=%s log=%s",
                doc_id,
                file_name,
                final_code,
                str(log_file),
            )
            if final_code != 0:
                logging.error(
                    "[jointlk-train] failure_hint=%s doc_id=%s file_name=%s log_tail=\n%s",
                    failure_hint,
                    doc_id,
                    file_name,
                    log_tail[-4000:],
                )

    thread = threading.Thread(target=_run, daemon=True, name=f"jointlk-train-{_sanitize_fs_part(doc_id or file_name)}")
    thread.start()


    logging.info(
        "[jointlk-train] started in background for doc_id=%s file_name=%s output_dir=%s log=%s",
        doc_id,
        file_name,
        str(output_dir),
        str(log_file),
    )
    return {
        "enabled": True,
        "started": True,
        "train_jsonl": str(train_jsonl_path),
        "dev_jsonl": str(dev_jsonl_path),
        "output_dir": str(output_dir),
        "log_file": str(log_file),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "model_name": model_name,
        "attempt_plan": attempt_plan,
    }


def _read_jsonl_rows(path: str, max_rows: int = 20000):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def _should_echo_train_progress(line: str) -> bool:
    text = (line or "").strip()
    if not text:
        return False
    keys = (
        "[JointLK][epoch-summary]",
        "[JointLK][epoch-dashboard]",
        "[JointLK][training-metrics]",
        "[JointLK][train-epoch]",
    )
    return any(k in text for k in keys)


def build_causal_chain_preview_from_pseudo(*, pseudo_result: dict, file_name: str, doc_id: str):
    """
    Build causal-edge/chain preview directly from pseudo labels so users can see
    "current extracted causal chains" immediately after upload.
    """
    if not pseudo_result or not pseudo_result.get("ok"):
        return {"ok": False, "reason": "pseudo_not_ok", "num_edges": 0, "num_chains": 0}

    manifest_paths = pseudo_result.get("paths") or {}
    train_jsonl = manifest_paths.get("jointlk_multitask_train_jsonl")
    if not train_jsonl or not os.path.exists(train_jsonl):
        return {"ok": False, "reason": "missing_train_jsonl", "num_edges": 0, "num_chains": 0}

    min_causal_conf = float(os.getenv("AUTO_JOINTLK_CHAIN_MIN_CAUSAL_CONF", "0.55"))
    top_k_edges = int(os.getenv("AUTO_JOINTLK_CHAIN_TOPK_EDGES", "20"))
    top_k_chains = int(os.getenv("AUTO_JOINTLK_CHAIN_TOPK_CHAINS", "8"))
    max_chain_depth = int(os.getenv("AUTO_JOINTLK_CHAIN_MAX_DEPTH", "4"))

    rows = _read_jsonl_rows(train_jsonl)
    causal_edges = []
    for row in rows:
        is_pos = int(row.get("causal_labels", row.get("silver_edge_causal", -1)) or -1) == 1
        causal_conf = float(row.get("causal_conf", 0.0) or 0.0)
        relaxed_pos = (not is_pos) and causal_conf >= min_causal_conf
        if not (is_pos or relaxed_pos):
            continue
        edge = {
            "doc_id": row.get("doc_id"),
            "source_node_id": row.get("source_node_id"),
            "source_text": row.get("source_text"),
            "target_node_id": row.get("target_node_id"),
            "target_text": row.get("target_text"),
            "relation_type": row.get("relation_type"),
            "causal_label": 1 if is_pos else 0,
            "causal_conf": causal_conf,
            "enable_label": int(row.get("enable_labels", row.get("silver_edge_enable", -1)) or -1),
            "dir_label": int(row.get("dir_labels", row.get("silver_causal_dir", -1)) or -1),
            "temporal_label": int(row.get("temp_labels", row.get("silver_temporal_before", -1)) or -1),
            "review_status": row.get("review_status"),
        }
        edge["score"] = float(edge["causal_conf"]) + 0.05 * max(edge["enable_label"], 0)
        causal_edges.append(edge)

    if not causal_edges:
        # fallback: provide likely causal candidates for inspection even if no hard positive edges
        soft_threshold = float(os.getenv("AUTO_JOINTLK_CHAIN_SOFT_MIN_CAUSAL_CONF", "0.30"))
        for row in rows:
            causal_conf = float(row.get("causal_conf", 0.0) or 0.0)
            if causal_conf < soft_threshold:
                continue
            edge = {
                "doc_id": row.get("doc_id"),
                "source_node_id": row.get("source_node_id"),
                "source_text": row.get("source_text"),
                "target_node_id": row.get("target_node_id"),
                "target_text": row.get("target_text"),
                "relation_type": row.get("relation_type"),
                "causal_label": int(row.get("causal_labels", row.get("silver_edge_causal", -1)) or -1),
                "causal_conf": causal_conf,
                "enable_label": int(row.get("enable_labels", row.get("silver_edge_enable", -1)) or -1),
                "dir_label": int(row.get("dir_labels", row.get("silver_causal_dir", -1)) or -1),
                "temporal_label": int(row.get("temp_labels", row.get("silver_temporal_before", -1)) or -1),
                "review_status": row.get("review_status"),
                "candidate_type": "soft",
            }
            edge["score"] = float(edge["causal_conf"]) + 0.03 * max(edge["enable_label"], 0)
            causal_edges.append(edge)

    causal_edges.sort(key=lambda x: x["score"], reverse=True)
    top_edges = causal_edges[:max(top_k_edges, 0)]

    out_adj = defaultdict(list)
    in_deg = defaultdict(int)
    for e in top_edges:
        s = str(e.get("source_node_id") or "")
        t = str(e.get("target_node_id") or "")
        if not s or not t or s == t:
            continue
        out_adj[s].append(e)
        in_deg[t] += 1
        in_deg.setdefault(s, 0)

    roots = [nid for nid, deg in in_deg.items() if deg == 0]
    if not roots:
        roots = list(out_adj.keys())

    chains = []

    def _dfs(node_id, path_edges, visited):
        if len(path_edges) >= max_chain_depth:
            chains.append(list(path_edges))
            return
        next_edges = out_adj.get(node_id, [])
        if not next_edges:
            if path_edges:
                chains.append(list(path_edges))
            return
        for e in next_edges:
            nxt = str(e.get("target_node_id") or "")
            if not nxt or nxt in visited:
                continue
            path_edges.append(e)
            visited.add(nxt)
            _dfs(nxt, path_edges, visited)
            visited.remove(nxt)
            path_edges.pop()

    for r in roots:
        _dfs(r, [], {r})

    chain_rows = []
    for path in chains:
        if not path:
            continue
        chain_score = sum(float(x.get("score", 0.0)) for x in path) / max(1, len(path))
        chain_text = " -> ".join([str(path[0].get("source_text") or path[0].get("source_node_id") or "")] + [
            str(x.get("target_text") or x.get("target_node_id") or "") for x in path
        ])
        chain_rows.append(
            {
                "length": len(path),
                "score": round(chain_score, 4),
                "chain_text": chain_text,
                "edges": path,
            }
        )
    chain_rows.sort(key=lambda x: (x["length"], x["score"]), reverse=True)
    chain_rows = chain_rows[:max(top_k_chains, 0)]
    final_chain = None
    if chain_rows:
        final_chain = sorted(
            chain_rows,
            key=lambda x: (float(x.get("score", 0.0)), int(x.get("length", 0))),
            reverse=True,
        )[0]

    _log_jointlk_stage(
        "causal-chain-preview",
        {
            "file_name": file_name,
            "doc_id": doc_id,
            "num_causal_edges": len(causal_edges),
            "num_chain_candidates": len(chain_rows),
            "min_causal_conf": min_causal_conf,
            "top_chain": (final_chain["chain_text"] if final_chain else None),
        },
    )
    return {
        "ok": True,
        "num_edges": len(causal_edges),
        "num_chains": len(chain_rows),
        "final_chain": final_chain,
        "min_causal_conf": min_causal_conf,
        "top_edges": top_edges,
        "top_chains": chain_rows,
        "train_jsonl": train_jsonl,
    }

def create_source_node_graph_url_s3(graph, model, source_url, aws_access_key_id, aws_secret_access_key, source_type):
    lst_file_name = []
    files_info = get_s3_files_info(source_url, aws_access_key_id=aws_access_key_id,
                                   aws_secret_access_key=aws_secret_access_key)
    if len(files_info) == 0:
        raise Exception('No pdf files found.')
    logging.info(f'files info : {files_info}')
    success_count = 0
    failed_count = 0

    for file_info in files_info:
        file_name = file_info['file_key']
        obj_source_node = sourceNode()
        obj_source_node.file_name = file_name.split('/')[-1]
        obj_source_node.file_type = 'pdf'
        obj_source_node.file_size = file_info['file_size_bytes']
        obj_source_node.file_source = source_type
        obj_source_node.total_pages = 'N/A'
        obj_source_node.model = model
        obj_source_node.url = str(source_url + file_name)
        obj_source_node.awsAccessKeyId = aws_access_key_id
        obj_source_node.created_at = datetime.now()
        try:
            graphDb_data_Access = graphDBdataAccess(graph)
            graphDb_data_Access.create_source_node(obj_source_node)
            success_count += 1
            lst_file_name.append({'fileName': obj_source_node.file_name, 'fileSize': obj_source_node.file_size,
                                  'url': obj_source_node.url, 'status': 'Success'})

        except Exception as e:
            failed_count += 1
            # error_message = str(e)
            lst_file_name.append({'fileName': obj_source_node.file_name, 'fileSize': obj_source_node.file_size,
                                  'url': obj_source_node.url, 'status': 'Failed'})
    return lst_file_name, success_count, failed_count


def create_source_node_graph_url_gcs(graph, model, gcs_project_id, gcs_bucket_name, gcs_bucket_folder, source_type,
                                     credentials):
    success_count = 0
    failed_count = 0
    lst_file_name = []

    lst_file_metadata = get_gcs_bucket_files_info(gcs_project_id, gcs_bucket_name, gcs_bucket_folder, credentials)
    for file_metadata in lst_file_metadata:
        obj_source_node = sourceNode()
        obj_source_node.file_name = file_metadata['fileName']
        obj_source_node.file_size = file_metadata['fileSize']
        obj_source_node.url = file_metadata['url']
        obj_source_node.file_source = source_type
        obj_source_node.total_pages = 'N/A'
        obj_source_node.model = model
        obj_source_node.file_type = 'pdf'
        obj_source_node.gcsBucket = gcs_bucket_name
        obj_source_node.gcsBucketFolder = file_metadata['gcsBucketFolder']
        obj_source_node.gcsProjectId = file_metadata['gcsProjectId']
        obj_source_node.created_at = datetime.now()
        obj_source_node.access_token = credentials.token

        try:
            graphDb_data_Access = graphDBdataAccess(graph)
            graphDb_data_Access.create_source_node(obj_source_node)
            success_count += 1
            lst_file_name.append({'fileName': obj_source_node.file_name, 'fileSize': obj_source_node.file_size,
                                  'url': obj_source_node.url, 'status': 'Success',
                                  'gcsBucketName': gcs_bucket_name, 'gcsBucketFolder': obj_source_node.gcsBucketFolder,
                                  'gcsProjectId': obj_source_node.gcsProjectId})
        except Exception as e:
            failed_count += 1
            lst_file_name.append({'fileName': obj_source_node.file_name, 'fileSize': obj_source_node.file_size,
                                  'url': obj_source_node.url, 'status': 'Failed',
                                  'gcsBucketName': gcs_bucket_name, 'gcsBucketFolder': obj_source_node.gcsBucketFolder,
                                  'gcsProjectId': obj_source_node.gcsProjectId})
    return lst_file_name, success_count, failed_count


def create_source_node_graph_web_url(graph, model, source_url, source_type):
    success_count = 0
    failed_count = 0
    lst_file_name = []
    pages = WebBaseLoader(source_url, verify_ssl=False).load()
    if pages == None or len(pages) == 0:
        failed_count += 1
        message = f"Unable to read data for given url : {source_url}"
        raise Exception(message)
    obj_source_node = sourceNode()
    obj_source_node.file_type = 'text'
    obj_source_node.file_source = source_type
    obj_source_node.model = model
    obj_source_node.total_pages = 1
    obj_source_node.url = urllib.parse.unquote(source_url)
    obj_source_node.created_at = datetime.now()
    obj_source_node.file_name = pages[0].metadata['title']
    obj_source_node.language = pages[0].metadata['language']
    obj_source_node.file_size = sys.getsizeof(pages[0].page_content)

    graphDb_data_Access = graphDBdataAccess(graph)
    graphDb_data_Access.create_source_node(obj_source_node)
    lst_file_name.append(
        {'fileName': obj_source_node.file_name, 'fileSize': obj_source_node.file_size, 'url': obj_source_node.url,
         'status': 'Success'})
    success_count += 1
    return lst_file_name, success_count, failed_count


def create_source_node_graph_url_youtube(graph, model, source_url, source_type):
    youtube_url, language = check_url_source(source_type=source_type, yt_url=source_url)
    success_count = 0
    failed_count = 0
    lst_file_name = []
    obj_source_node = sourceNode()
    obj_source_node.file_type = 'text'
    obj_source_node.file_source = source_type
    obj_source_node.model = model
    obj_source_node.total_pages = 1
    obj_source_node.url = youtube_url
    obj_source_node.created_at = datetime.now()
    match = re.search(r'(?:v=)([0-9A-Za-z_-]{11})\s*', obj_source_node.url)
    logging.info(f"match value: {match}")
    obj_source_node.file_name = YouTube(obj_source_node.url).title
    transcript = get_youtube_combined_transcript(match.group(1))
    if transcript == None or len(transcript) == 0:
        message = f"Youtube transcript is not available for : {obj_source_node.file_name}"
        raise Exception(message)
    else:
        obj_source_node.file_size = sys.getsizeof(transcript)

    graphDb_data_Access = graphDBdataAccess(graph)
    graphDb_data_Access.create_source_node(obj_source_node)
    lst_file_name.append(
        {'fileName': obj_source_node.file_name, 'fileSize': obj_source_node.file_size, 'url': obj_source_node.url,
         'status': 'Success'})
    success_count += 1
    return lst_file_name, success_count, failed_count


def create_source_node_graph_url_wikipedia(graph, model, wiki_query, source_type):
    success_count = 0
    failed_count = 0
    lst_file_name = []
    # queries_list =  wiki_query.split(',')
    wiki_query_id, language = check_url_source(source_type=source_type, wiki_query=wiki_query)
    logging.info(f"Creating source node for {wiki_query_id.strip()}, {language}")
    pages = WikipediaLoader(query=wiki_query_id.strip(), lang=language, load_max_docs=1,
                            load_all_available_meta=True).load()
    if pages == None or len(pages) == 0:
        failed_count += 1
        message = f"Unable to read data for given Wikipedia url : {wiki_query}"
        raise Exception(message)
    else:
        obj_source_node = sourceNode()
        obj_source_node.file_name = wiki_query_id.strip()
        obj_source_node.file_type = 'text'
        obj_source_node.file_source = source_type
        obj_source_node.file_size = sys.getsizeof(pages[0].page_content)
        obj_source_node.total_pages = len(pages)
        obj_source_node.model = model
        obj_source_node.url = urllib.parse.unquote(pages[0].metadata['source'])
        obj_source_node.created_at = datetime.now()
        obj_source_node.language = language
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.create_source_node(obj_source_node)
        success_count += 1
        lst_file_name.append(
            {'fileName': obj_source_node.file_name, 'fileSize': obj_source_node.file_size, 'url': obj_source_node.url,
             'language': obj_source_node.language, 'status': 'Success'})
    return lst_file_name, success_count, failed_count


def extract_graph_from_file_local_file(graph, model, merged_file_path, fileName, allowedNodes, allowedRelationship, uri,
                                       kg_scope=None, kg_id=None):
    logging.info(f'Process file name :{fileName}')
    gcs_file_cache = os.environ.get('GCS_FILE_CACHE')
    if gcs_file_cache == 'True':
        folder_name = create_gcs_bucket_folder_name_hashed(uri, fileName)
        file_name, pages = get_documents_from_gcs(PROJECT_ID, BUCKET_UPLOAD, folder_name, fileName)
    else:
        file_name, pages, file_extension = get_documents_from_file_by_path(merged_file_path, fileName)
    if pages == None or len(pages) == 0:
        raise Exception(f'File content is not available for file : {file_name}')

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, True, merged_file_path,
                             uri, kg_scope=kg_scope, kg_id=kg_id, domain_pack_id="construction")


def extract_graph_from_file_s3(graph, model, source_url, aws_access_key_id, aws_secret_access_key, allowedNodes,
                               allowedRelationship, kg_scope=None, kg_id=None):
    if (aws_access_key_id == None or aws_secret_access_key == None):
        raise Exception('Please provide AWS access and secret keys')
    else:
        logging.info("Insert in S3 Block")
        file_name, pages = get_documents_from_s3(source_url, aws_access_key_id, aws_secret_access_key)

    if pages == None or len(pages) == 0:
        raise Exception(f'File content is not available for file : {file_name}')

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, kg_scope=kg_scope,
                             kg_id=kg_id, domain_pack_id="construction")


def extract_graph_from_web_page(graph, model, source_url, allowedNodes, allowedRelationship, kg_scope=None, kg_id=None):
    file_name, pages = get_documents_from_web_page(source_url)

    if pages == None or len(pages) == 0:
        raise Exception(f'Content is not available for given URL : {file_name}')

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, kg_scope=kg_scope,
                             kg_id=kg_id, domain_pack_id="construction")


def extract_graph_from_file_youtube(graph, model, source_url, allowedNodes, allowedRelationship, kg_scope=None,
                                    kg_id=None):
    file_name, pages = get_documents_from_youtube(source_url)

    if pages == None or len(pages) == 0:
        raise Exception(f'Youtube transcript is not available for file : {file_name}')

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, kg_scope=kg_scope,
                             kg_id=kg_id, domain_pack_id="construction")


def extract_graph_from_file_Wikipedia(graph, model, wiki_query, max_sources, language, allowedNodes,
                                      allowedRelationship, kg_scope=None, kg_id=None):
    file_name, pages = get_documents_from_Wikipedia(wiki_query, language)
    if pages == None or len(pages) == 0:
        raise Exception(f'Wikipedia page is not available for file : {file_name}')

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, kg_scope=kg_scope,
                             kg_id=kg_id, domain_pack_id="construction")


def extract_graph_from_file_gcs(graph, model, gcs_project_id, gcs_bucket_name, gcs_bucket_folder, gcs_blob_filename,
                                access_token, allowedNodes, allowedRelationship, kg_scope=None, kg_id=None):
    file_name, pages = get_documents_from_gcs(gcs_project_id, gcs_bucket_name, gcs_bucket_folder, gcs_blob_filename,
                                              access_token)
    if pages == None or len(pages) == 0:
        raise Exception(f'File content is not available for file : {file_name}')

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, kg_scope=kg_scope,
                             kg_id=kg_id, domain_pack_id="construction")


def processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, is_uploaded_from_local=None,
                      merged_file_path=None, uri=None, kg_scope=None, kg_id=None, domain_pack_id="construction"):
    """
     Extracts a Neo4jGraph from a PDF file based on the model.

     Args:
          uri: URI of the graph to extract
       db_name : db_name is database name to connect graph db
          userName: Username to use for graph creation ( if None will use username from config file )
          password: Password to use for graph creation ( if None will use password from config file )
          file: File object containing the PDF file to be used
          model: Type of model to use ('Diffbot'or'OpenAI GPT')

     Returns:
          Json response to API with fileName, nodeCount, relationshipCount, processingTime,
       status and model as attributes.
    """
    start_time = datetime.now()
    graphDb_data_Access = graphDBdataAccess(graph)

    # KG scoping (BG-KG / Instance-KG)
    ctx = build_kg_context(kg_scope, kg_id, file_name)
    _log_jointlk_stage(
        "upload-document",
        {
            "file_name": file_name,
            "doc_id": ctx.doc_id,
            "kg_scope": ctx.kg_scope,
            "kg_id": ctx.kg_id,
            "num_pages": len(pages),
            "model": model,
            "domain_pack_id": domain_pack_id,
        },
    )
    try:
        result = graphDb_data_Access.get_current_status_document_node_scoped(ctx.doc_id)
    except Exception:
        result = []
    if result is None or len(result) == 0:
        result = graphDb_data_Access.get_current_status_document_node(file_name)
    logging.info("Break down file into chunks")
    bad_chars = ['"', "\n", "'"]
    for i in range(0, len(pages)):
        text = pages[i].page_content
        for j in bad_chars:
            if j == '\n':
                text = text.replace(j, ' ')
            else:
                text = text.replace(j, '')
        page_metadata = dict(pages[i].metadata or {})
        page_metadata.setdefault("doc_id", ctx.doc_id)
        page_metadata.setdefault("kg_scope", ctx.kg_scope)
        page_metadata.setdefault("kg_id", ctx.kg_id)
        page_metadata.setdefault("fileName", file_name)
        pages[i] = Document(page_content=str(text), metadata=page_metadata)
    create_chunks_obj = CreateChunksofDocument(pages, graph)
    chunks = create_chunks_obj.split_file_into_chunks()
    _log_jointlk_stage(
        "chunking-finished",
        {
            "file_name": file_name,
            "doc_id": ctx.doc_id,
            "num_chunks": len(chunks),
        },
    )
    chunkId_chunkDoc_list = create_relation_between_chunks(graph, file_name, chunks, doc_id=ctx.doc_id,
                                                           kg_scope=ctx.kg_scope, kg_id=ctx.kg_id)
    evidence_units = EvidenceUnitBuilder().build_from_chunks(chunkId_chunkDoc_list)
    evidence_units_by_chunk = index_units(evidence_units)
    persist_evidence_units(
        graph=graph,
        evidence_store=None,
        evidence_units=evidence_units,
        file_name=file_name,
        doc_id=ctx.doc_id,
        kg_scope=ctx.kg_scope,
        kg_id=ctx.kg_id,
    )
    if result[0]['Status'] != 'Processing':
        obj_source_node = sourceNode()
        status = "Processing"

        obj_source_node.file_name = file_name
        obj_source_node.status = status
        obj_source_node.total_chunks = len(chunks)
        obj_source_node.total_pages = len(pages)
        obj_source_node.model = model
        obj_source_node.doc_id = ctx.doc_id
        obj_source_node.kg_scope = ctx.kg_scope
        obj_source_node.kg_id = ctx.kg_id
        logging.info(file_name)
        logging.info(obj_source_node)
        graphDb_data_Access.update_source_node_scoped(obj_source_node)

        logging.info('Update the status as Processing')
        update_graph_chunk_processed = int(os.environ.get('UPDATE_GRAPH_CHUNKS_PROCESSED'))
        # selected_chunks = []
        is_cancelled_status = False
        job_status = "Completed"
        node_count = 0
        rel_count = 0
        comparison_summary = None
        for i in range(0, len(chunkId_chunkDoc_list), update_graph_chunk_processed):
            select_chunks_upto = i + update_graph_chunk_processed
            logging.info(f'Selected Chunks upto: {select_chunks_upto}')
            if len(chunkId_chunkDoc_list) <= select_chunks_upto:
                select_chunks_upto = len(chunkId_chunkDoc_list)
            selected_chunks = chunkId_chunkDoc_list[i:select_chunks_upto]
            try:
                result = graphDb_data_Access.get_current_status_document_node_scoped(ctx.doc_id)
            except Exception:
                result = []
            if result is None or len(result) == 0:
                result = graphDb_data_Access.get_current_status_document_node(file_name)
            is_cancelled_status = result[0]['is_cancelled']
            logging.info(f"Value of is_cancelled : {result[0]['is_cancelled']}")
            if bool(is_cancelled_status) == True:
                job_status = "Cancelled"
                logging.info('Exit from running loop of processing file')
                exit
            else:
                node_count, rel_count ,batch_comparison_summary = processing_chunks(selected_chunks, graph, file_name, model, allowedNodes,
                                                          allowedRelationship, node_count, rel_count, ctx,evidence_units_by_chunk,
                                                          domain_pack_id=domain_pack_id,)
                _log_jointlk_stage(
                    "kg-batch-progress",
                    {
                        "file_name": file_name,
                        "doc_id": ctx.doc_id,
                        "processed_chunk": select_chunks_upto,
                        "total_chunks": len(chunkId_chunkDoc_list),
                        "node_count": node_count,
                        "relationship_count": rel_count,
                    },
                )
                comparison_summary = _merge_kg_quality_comparison(
                    comparison_summary,
                    batch_comparison_summary,
                )

                end_time = datetime.now()
                processed_time = end_time - start_time

                obj_source_node = sourceNode()
                obj_source_node.file_name = file_name
                obj_source_node.doc_id = ctx.doc_id
                obj_source_node.kg_scope = ctx.kg_scope
                obj_source_node.kg_id = ctx.kg_id
                obj_source_node.updated_at = end_time
                obj_source_node.processing_time = processed_time
                obj_source_node.node_count = node_count
                obj_source_node.processed_chunk = select_chunks_upto
                obj_source_node.relationship_count = rel_count
                graphDb_data_Access.update_source_node_scoped(obj_source_node)

        try:
            result = graphDb_data_Access.get_current_status_document_node_scoped(ctx.doc_id)
            if result is None or len(result) == 0:
                result = graphDb_data_Access.get_current_status_document_node(file_name)
        except Exception:
            result = graphDb_data_Access.get_current_status_document_node(file_name)
        is_cancelled_status = result[0]['is_cancelled']
        if bool(is_cancelled_status) == True:
            logging.info(f'Is_cancelled True at the end extraction')
            job_status = 'Cancelled'
        logging.info(f'Job Status at the end : {job_status}')
        end_time = datetime.now()
        processed_time = end_time - start_time
        pseudo_result = None
        auto_train_result = None
        causal_chain_preview = None
        auto_train_result = None
        obj_source_node = sourceNode()
        obj_source_node.file_name = file_name
        obj_source_node.doc_id = ctx.doc_id
        obj_source_node.kg_scope = ctx.kg_scope
        obj_source_node.kg_id = ctx.kg_id
        obj_source_node.status = job_status
        obj_source_node.processing_time = processed_time

        graphDb_data_Access.update_source_node_scoped(obj_source_node)
        logging.info('Updated the nodeCount and relCount properties in Document node')
        logging.info(f'file:{file_name} extraction has been completed')

        # 生成词汇表和gpickle
        if job_status == "Completed":
            try:
                logging.info("Starting graph export to gpickle format...")
                query_nodes, query_relations = graphDb_data_Access.export_concept()
                generate_gpickle_export(query_nodes, query_relations)
                export_jointlk_json_artifacts(query_nodes, query_relations)

                logging.info(
                    "Successfully exported graph artifacts (concept.txt, graph.gpickle, vocab JSONs)."
                )
                _log_jointlk_stage(
                    "graph-artifacts-exported",
                    {
                        "file_name": file_name,
                        "doc_id": ctx.doc_id,
                        "node_count": node_count,
                        "relationship_count": rel_count,
                    },
                )

            except Exception as e:
                logging.error(f"Failed to export gpickle graph: {e}")

            try:
                pseudo_result = maybe_run_auto_pseudo_label_pipeline(
                    graph,
                    file_name=file_name,
                    doc_id=ctx.doc_id,
                    kg_scope=ctx.kg_scope,
                    kg_id=ctx.kg_id,
                )
                if pseudo_result:
                    manifest_paths = (pseudo_result.get("paths") or {})
                    _log_jointlk_stage(
                        "pseudo-label-exported",
                        {
                            "file_name": file_name,
                            "doc_id": ctx.doc_id,
                            "num_pseudo_labels": pseudo_result.get("num_pseudo_labels"),
                            "train_jsonl": manifest_paths.get("jointlk_multitask_train_jsonl"),
                            "counterfactual_pairs": manifest_paths.get("counterfactual_pairs_jsonl"),
                            "manual_review_csv": manifest_paths.get("manual_review_candidates_csv"),
                        },
                    )
            except Exception as e:
                pseudo_result = {
                    "ok": False,
                    "error": str(e),
                }
                logging.exception(f"[pseudo-label] failed for file={file_name}: {e}")

                # merged_file_path have value only when file uploaded from local
            try:
                causal_chain_preview = build_causal_chain_preview_from_pseudo(
                    pseudo_result=pseudo_result or {},
                    file_name=file_name,
                    doc_id=ctx.doc_id,
                )
                if causal_chain_preview and causal_chain_preview.get("ok"):
                    _log_jointlk_stage(
                        "causal-chain-exported",
                        {
                            "file_name": file_name,
                            "doc_id": ctx.doc_id,
                            "num_edges": causal_chain_preview.get("num_edges"),
                            "num_chains": causal_chain_preview.get("num_chains"),
                            "top_chain": (
                                (causal_chain_preview.get("top_chains") or [{}])[0].get("chain_text")
                                if causal_chain_preview.get("top_chains")
                                else None
                            ),
                        },
                    )
                    top_chain_rows = causal_chain_preview.get("top_chains") or []
                    final_chain_row = causal_chain_preview.get("final_chain") or (top_chain_rows[0] if top_chain_rows else None)
                    _log_jointlk_stage(
                        "causal-chain-preview",
                        {
                            "file_name": file_name,
                            "doc_id": ctx.doc_id,
                            "top_chain_count": len(top_chain_rows),
                            "final_chain": {
                                "score": final_chain_row.get("score") if final_chain_row else None,
                                "chain_text": final_chain_row.get("chain_text") if final_chain_row else None,
                                "length": final_chain_row.get("length") if final_chain_row else None,
                            },
                        },
                    )
                    logging.info(
                        "[JointLK][final-causal-chain] %s",
                        json.dumps(
                            {
                                "file_name": file_name,
                                "doc_id": ctx.doc_id,
                                "score": final_chain_row.get("score") if final_chain_row else None,
                                "chain_text": final_chain_row.get("chain_text") if final_chain_row else None,
                                "length": final_chain_row.get("length") if final_chain_row else None,
                            },
                            ensure_ascii=False,
                        ),
                    )
            except Exception as e:
                causal_chain_preview = {
                    "ok": False,
                    "error": str(e),
                }
                logging.exception(f"[jointlk-chain] failed to build preview for file={file_name}: {e}")

            try:
                auto_train_result = maybe_trigger_auto_jointlk_training(
                    pseudo_result=pseudo_result or {},
                    file_name=file_name,
                    doc_id=ctx.doc_id,
                )
                if auto_train_result and auto_train_result.get("started"):
                    _log_jointlk_stage(
                        "jointlk-training-started",
                        {
                            "file_name": file_name,
                            "doc_id": ctx.doc_id,
                            "train_jsonl": auto_train_result.get("train_jsonl"),
                            "output_dir": auto_train_result.get("output_dir"),
                            "log_file": auto_train_result.get("log_file"),
                            "epochs": auto_train_result.get("epochs"),
                            "batch_size": auto_train_result.get("batch_size"),
                            "model_name": auto_train_result.get("model_name"),
                        },
                    )
            except Exception as e:
                auto_train_result = {
                    "enabled": True,
                    "started": False,
                    "error": str(e),
                }
                logging.exception(f"[jointlk-train] failed to start for file={file_name}: {e}")


        # merged_file_path have value only when file uploaded from local

        if is_uploaded_from_local:
            gcs_file_cache = os.environ.get('GCS_FILE_CACHE')
            if gcs_file_cache == 'True':
                folder_name = create_gcs_bucket_folder_name_hashed(uri, file_name)
                delete_file_from_gcs(BUCKET_UPLOAD, folder_name, file_name)
            else:
                delete_uploaded_local_file(merged_file_path, file_name)

        return {
            "fileName": file_name,
            "nodeCount": node_count,
            "relationshipCount": rel_count,
            "processingTime": round(processed_time.total_seconds(), 2),
            "status": job_status,
            "model": model,
            "success_count": 1,
            "pseudo_label_summary": pseudo_result if job_status == "Completed" else None,
            "jointlk_causal_chain_preview": causal_chain_preview if job_status == "Completed" else None,
            "jointlk_training_summary": auto_train_result if job_status == "Completed" else None,
            "kg_quality_comparison": comparison_summary if job_status == "Completed" else None,
        }
    else:
        logging.info('File does not process because it\'s already in Processing status')


def processing_chunks(chunkId_chunkDoc_list, graph, file_name, model, allowedNodes, allowedRelationship, node_count,
                      rel_count, ctx=None, evidence_units_by_chunk=None, domain_pack_id="construction",):
    # create vector index and update chunk node with embedding
    update_embedding_create_vector_index(graph, chunkId_chunkDoc_list, file_name, doc_id=(ctx.doc_id if ctx else None))
    logging.info("Get graph document list from models")
    graph_documents = generate_graphDocuments(
        model, graph, chunkId_chunkDoc_list, allowedNodes, allowedRelationship,
        domain_pack_id=domain_pack_id,evidence_units_by_chunk=evidence_units_by_chunk,
    )

    # Scope node ids to avoid BG/Instance collisions
    if ctx is not None:
        graph_documents = scope_graph_documents(graph_documents, ctx)

    # 检查 graph_documents 是否为空
    if not graph_documents:
        logging.warning(f"No graph documents were generated for file {file_name}. Skipping chunk processing.")
        # 即使没有生成图，也需要保存已有的图文档（如果之前批次有的话）
        save_graphDocuments_in_neo4j(graph, [])  # 传入空列表以避免错误
        return node_count, rel_count ,build_kg_quality_comparison([], []) # 返回未修改的计数值

    save_graphDocuments_in_neo4j(graph, graph_documents)
    chunks_and_graphDocuments_list = get_chunk_and_graphDocument(graph_documents, chunkId_chunkDoc_list)
    merge_relationship_between_chunk_and_entites(graph, chunks_and_graphDocuments_list,evidence_units_by_chunk)

    distinct_nodes = set()
    relations = []

    # 遍历每一个图文档
    for graph_document in graph_documents:

        # 1. 从当前文档中获取所有独立节点
        if graph_document.nodes:
            for node in graph_document.nodes:
                node_id = node.id
                node_type = node.type
                if (node_id, node_type) not in distinct_nodes:
                    distinct_nodes.add((node_id, node_type))

        # 2. 从当前文档中获取所有关系
        if graph_document.relationships:
            for relation in graph_document.relationships:
                relations.append(relation.type)

    node_count += len(distinct_nodes)
    rel_count += len(relations)
    print(f'node :{distinct_nodes}')
    print(f'relations :{relations}')
    print(f'node count internal func:{node_count}')
    print(f'relation count internal func:{rel_count}')

    current_chunk_ids={
        item.get("chunk_id")
        for item in (chunks_and_graphDocuments_list or [])
        if isinstance(item, dict) and item.get("chunk_id")
    }
    comparison_summary = build_kg_quality_comparison(
        graph_documents,
        [
            unit
            for chunk_id, units in (evidence_units_by_chunk or {}).items()
            if chunk_id in current_chunk_ids
            for unit in units
        ],
    )
    return node_count, rel_count, comparison_summary


def _merge_kg_quality_comparison(existing_summary, new_summary):
    if not existing_summary:
        return new_summary
    if not new_summary:
        return existing_summary

    merged_summary = {
        "baseline_llm_only": dict(existing_summary.get("baseline_llm_only") or {}),
        "enhanced_with_evidence_units": dict(existing_summary.get("enhanced_with_evidence_units") or {}),
    }
    for section in ("baseline_llm_only", "enhanced_with_evidence_units"):
        existing_section = existing_summary.get(section) or {}
        new_section = new_summary.get(section) or {}
        target_section = merged_summary[section]
        for key in set(existing_section) | set(new_section):
            existing_value = existing_section.get(key)
            new_value = new_section.get(key)
            if isinstance(existing_value, (int, float)) or isinstance(new_value, (int, float)):
                target_section[key] = (existing_value or 0) + (new_value or 0)
            elif new_value is not None:
                target_section[key] = new_value
            else:
                target_section[key] = existing_value

    grounded_nodes = merged_summary["enhanced_with_evidence_units"].get("evidence_grounded_node_count", 0)
    total_candidate_evidence = merged_summary["enhanced_with_evidence_units"].get("total_candidate_evidence_links", 0)
    if grounded_nodes:
        merged_summary["enhanced_with_evidence_units"]["avg_candidate_evidence_per_node"] = round(
            total_candidate_evidence / grounded_nodes, 2
        )

    return merged_summary

def get_source_list_from_graph(uri, userName, password, db_name=None):
    """
    Args:
      uri: URI of the graph to extract
      db_name: db_name is database name to connect to graph db
      userName: Username to use for graph creation ( if None will use username from config file )
      password: Password to use for graph creation ( if None will use password from config file )
      file: File object containing the PDF file to be used
      model: Type of model to use ('Diffbot'or'OpenAI GPT')
    Returns:
     Returns a list of sources that are in the database by querying the graph and
     sorting the list by the last updated date.
   """
    logging.info("Get existing files list from graph")
    graph = Neo4jGraph(url=uri, database=db_name, username=userName, password=password)
    graph_DB_dataAccess = graphDBdataAccess(graph)
    if not graph._driver._closed:
        logging.info(f"closing connection for sources_list api")
        graph._driver.close()
    return graph_DB_dataAccess.get_source_list()


def update_graph(graph):
    """
    Update the graph node with SIMILAR relationship where embedding scrore match
    """
    graph_DB_dataAccess = graphDBdataAccess(graph)
    graph_DB_dataAccess.update_KNN_graph()


def connection_check(graph):
    """
    Args:
      uri: URI of the graph to extract
      userName: Username to use for graph creation ( if None will use username from config file )
      password: Password to use for graph creation ( if None will use password from config file )
      db_name: db_name is database name to connect to graph db
    Returns:
     Returns a status of connection from NEO4j is success or failure
   """
    graph_DB_dataAccess = graphDBdataAccess(graph)
    return graph_DB_dataAccess.connection_check()


def merge_chunks_local(file_name, total_chunks, chunk_dir, merged_dir):
    if not os.path.exists(merged_dir):
        os.mkdir(merged_dir)
    logging.info(f'Merged File Path: {merged_dir}')
    merged_file_path = os.path.join(merged_dir, file_name)
    with open(merged_file_path, "wb") as write_stream:
        for i in range(1, total_chunks + 1):
            chunk_file_path = os.path.join(chunk_dir, f"{file_name}_part_{i}")
            logging.info(f'Chunk File Path While Merging Parts:{chunk_file_path}')
            with open(chunk_file_path, "rb") as chunk_file:
                shutil.copyfileobj(chunk_file, write_stream)
            os.unlink(chunk_file_path)  # Delete the individual chunk file after merging
    logging.info("Chunks merged successfully and return file size")
    file_name, pages, file_extension = get_documents_from_file_by_path(merged_file_path, file_name)
    pdf_total_pages = pages[0].metadata['total_pages']
    file_size = os.path.getsize(merged_file_path)
    return pdf_total_pages, file_size


def upload_file(graph, model, chunk, chunk_number: int, total_chunks: int, originalname, uri, chunk_dir, merged_dir,
                kg_scope=None, kg_id=None):
    gcs_file_cache = os.environ.get('GCS_FILE_CACHE')
    logging.info(f'gcs file cache: {gcs_file_cache}')

    if gcs_file_cache == 'True':
        folder_name = create_gcs_bucket_folder_name_hashed(uri, originalname)
        upload_file_to_gcs(chunk, chunk_number, originalname, BUCKET_UPLOAD, folder_name)
    else:
        if not os.path.exists(chunk_dir):
            os.mkdir(chunk_dir)

        chunk_file_path = os.path.join(chunk_dir, f"{originalname}_part_{chunk_number}")
        logging.info(f'Chunk File Path: {chunk_file_path}')

        with open(chunk_file_path, "wb") as chunk_file:
            chunk_file.write(chunk.file.read())

    if int(chunk_number) == int(total_chunks):
        # If this is the last chunk, merge all chunks into a single file
        if gcs_file_cache == 'True':
            file_size = merge_file_gcs(BUCKET_UPLOAD, originalname, folder_name, int(total_chunks))
            total_pages = 1
        else:
            total_pages, file_size = merge_chunks_local(originalname, int(total_chunks), chunk_dir, merged_dir)

        logging.info("File merged successfully")
        file_extension = originalname.split('.')[-1]
        obj_source_node = sourceNode()
        obj_source_node.file_name = originalname
        obj_source_node.file_type = file_extension
        obj_source_node.file_size = file_size
        obj_source_node.file_source = 'local file'
        obj_source_node.model = model
        obj_source_node.total_pages = total_pages
        obj_source_node.created_at = datetime.now()
        # KG scoping (BG-KG / Instance-KG)
        ctx = build_kg_context(kg_scope, kg_id, originalname)
        obj_source_node.doc_id = ctx.doc_id
        obj_source_node.kg_scope = ctx.kg_scope
        obj_source_node.kg_id = ctx.kg_id
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.create_source_node_scoped(obj_source_node)
        return {'file_size': file_size, 'total_pages': total_pages, 'file_name': originalname,
                'file_extension': file_extension, 'doc_id': obj_source_node.doc_id,
                'kg_scope': obj_source_node.kg_scope, 'kg_id': obj_source_node.kg_id,
                'message': f"Chunk {chunk_number}/{total_chunks} saved"}
    return f"Chunk {chunk_number}/{total_chunks} saved"


def get_labels_and_relationtypes(graph):
    query = """
          RETURN collect { 
          CALL db.labels() yield label 
          WHERE NOT label  IN ['Chunk','_Bloom_Perspective_'] 
          return label order by label limit 100 } as labels, 
          collect { 
          CALL db.relationshipTypes() yield relationshipType  as type 
          WHERE NOT type  IN ['PART_OF', 'NEXT_CHUNK', 'HAS_ENTITY', '_Bloom_Perspective_'] 
          return type order by type LIMIT 100 } as relationshipTypes
          """
    graphDb_data_Access = graphDBdataAccess(graph)
    result = graphDb_data_Access.execute_query(query)
    if result is None:
        result = []
    return result


def manually_cancelled_job(graph, filenames, source_types, merged_dir, uri):
    filename_list = list(map(str.strip, json.loads(filenames)))
    source_types_list = list(map(str.strip, json.loads(source_types)))
    gcs_file_cache = os.environ.get('GCS_FILE_CACHE')

    for (file_name, source_type) in zip(filename_list, source_types_list):
        obj_source_node = sourceNode()
        obj_source_node.file_name = file_name
        obj_source_node.is_cancelled = True
        obj_source_node.status = 'Cancelled'
        obj_source_node.updated_at = datetime.now()
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.update_source_node_scoped(obj_source_node)
        obj_source_node = None
        merged_file_path = os.path.join(merged_dir, file_name)
        if source_type == 'local file' and gcs_file_cache == 'True':
            folder_name = create_gcs_bucket_folder_name_hashed(uri, file_name)
            delete_file_from_gcs(BUCKET_UPLOAD, folder_name, file_name)
        else:
            logging.info(f'Deleted File Path: {merged_file_path} and Deleted File Name : {file_name}')
            delete_uploaded_local_file(merged_file_path, file_name)
    return "Cancelled the processing job successfully"


def populate_graph_schema_from_text(text, model, is_schema_description_cheked):
    """_summary_

    Args:
        graph (Neo4Graph): Neo4jGraph connection object
        input_text (str): rendom text from PDF or user input.
        model (str): AI model to use extraction from text

    Returns:
        data (list): list of lebels and relationTypes
    """
    result = schema_extraction_from_text(text, model, is_schema_description_cheked)
    return {"labels": result.labels, "relationshipTypes": result.relationshipTypes}

def index_units(evidence_units):
    grouped = {}
    for unit in evidence_units or []:
        grouped.setdefault(unit.parent_chunk_id, []).append(unit)
    return grouped


def build_kg_quality_comparison(graph_documents, evidence_units):
    baseline_nodes = sum(len(doc.nodes) for doc in graph_documents or [])
    baseline_relationships = sum(len(doc.relationships) for doc in graph_documents or [])
    enhanced_supported_nodes = 0
    enhanced_triggered_units = 0
    total_unit_links = 0
    for doc in graph_documents or []:
        for node in doc.nodes:
            props = node.properties or {}
            if props.get("evidence_unit_ids"):
                enhanced_supported_nodes += 1
                total_unit_links += len(props.get("evidence_unit_ids") or [])
    for unit in evidence_units or []:
        if unit.trigger_words:
            enhanced_triggered_units += 1
    return {
        "baseline_llm_only": {
            "node_count": baseline_nodes,
            "relationship_count": baseline_relationships,
            "evidence_grounded_node_count": 0,
            "causal_evidence_unit_count": 0,
        },
        "enhanced_with_evidence_units": {
            "node_count": baseline_nodes,
            "relationship_count": baseline_relationships,
            "evidence_grounded_node_count": enhanced_supported_nodes,
            "causal_evidence_unit_count": len(evidence_units or []),
            "triggered_unit_count": enhanced_triggered_units,
            "total_candidate_evidence_links": total_unit_links,
            "avg_candidate_evidence_per_node": round(total_unit_links / max(enhanced_supported_nodes, 1), 2),
        },
    }