from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional ,Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from langchain_community.graphs import Neo4jGraph  # type: ignore

from src.causal_jointlk.pseudo_pipeline import (
    AutoPseudoPipelineConfig,
    run_pseudo_label_pipeline_for_doc,
)
from src.causal_jointlk.neo4j_accessor import Neo4jAccessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo labels from Instance-KG and EvidenceStore.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evidence_db_path", default="./data/evidence_store.sqlite3")
    parser.add_argument("--prior_config", default="configs/causal_prior.yaml")
    parser.add_argument("--pseudo_rule_config", default="configs/causal_pseudo_label_rules.yaml")

    parser.add_argument("--neo4j_url", required=True)
    parser.add_argument("--neo4j_username", required=True)
    parser.add_argument("--neo4j_password", required=True)
    parser.add_argument("--neo4j_database", default="neo4j")

    parser.add_argument("--kg_scope", default="instance")
    parser.add_argument("--kg_id", default=None)
    parser.add_argument("--doc_ids_json", default=None)
    parser.add_argument("--file_names_json", default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--max_edges_per_doc", type=int, default=None)
    parser.add_argument("--max_units_per_doc_scan", type=int, default=500)
    parser.add_argument("--lexical_top_k", type=int, default=5)
    parser.add_argument("--store_ambiguous", action="store_true")
    parser.add_argument("--review_sample_size", type=int, default=300)
    parser.add_argument("--review_per_doc_cap", type=int, default=20)
    parser.add_argument("--review_per_rule_cap", type=int, default=40)
    parser.add_argument("--console_preview", type=int, default=5, help="每个文档在控制台展示前N条pseudo-label结果")
    parser.add_argument("--show_edge_process", action="store_true", help="在控制台展示每条候选边的打标过程")
    return parser.parse_args()


def _truncate(text: Optional[str], max_len: int = 28) -> str:
    s = str(text or "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def make_console_progress_hook(show_edge_process: bool):
    def _hook(event: dict) -> None:
        stage = event.get("stage")
        if stage == "edge_decision" and show_edge_process:
            print(
                "[edge {idx}/{total}] {src} --{rel}--> {tgt} | label={label} | conf={conf:.3f} | rule={rule}".format(
                    idx=event.get("edge_index"),
                    total=event.get("num_candidate_edges"),
                    src=_truncate(event.get("source_text")),
                    rel=event.get("relation_type") or "?",
                    tgt=_truncate(event.get("target_text")),
                    label=event.get("label"),
                    conf=float(event.get("confidence") or 0.0),
                    rule=event.get("primary_rule") or "-",
                )
            )
            return
        if stage == "doc_summary":
            bd = event.get("label_breakdown") or {}
            print(
                "[doc-summary] doc_id={doc} file={file} | candidates={cand} pseudo={pseudo} (pos={pos}, neg={neg}) ambiguous={amb}".format(
                    doc=event.get("doc_id"),
                    file=event.get("file_name"),
                    cand=event.get("num_candidate_edges"),
                    pseudo=event.get("num_pseudo_labels"),
                    pos=bd.get("positive", 0),
                    neg=bd.get("negative", 0),
                    amb=event.get("ambiguous_edges", 0),
                )
            )

    return _hook


def main() -> None:
    args = parse_args()
    graph = Neo4jGraph(
        url=args.neo4j_url,
        username=args.neo4j_username,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )
    accessor = Neo4jAccessor(graph)

    doc_ids = json.loads(args.doc_ids_json) if args.doc_ids_json else None
    file_names = json.loads(args.file_names_json) if args.file_names_json else None
    docs = accessor.list_documents(
        kg_scope=args.kg_scope,
        kg_id=args.kg_id,
        doc_ids=doc_ids,
        file_names=file_names,
        limit=args.max_docs,
    )

    cfg = AutoPseudoPipelineConfig(
        enabled=True,
        export_root=args.output_dir,
        evidence_db_path=args.evidence_db_path,
        prior_config_path=args.prior_config,
        pseudo_rule_config_path=args.pseudo_rule_config,
        store_ambiguous=args.store_ambiguous,
        max_edges_per_doc=args.max_edges_per_doc,
        max_units_per_doc_scan=args.max_units_per_doc_scan,
        lexical_top_k=args.lexical_top_k,
        review_sample_size=args.review_sample_size,
        review_per_doc_cap=args.review_per_doc_cap,
        review_per_rule_cap=args.review_per_rule_cap,
    )

    summaries = []
    progress_hook = make_console_progress_hook(args.show_edge_process)
    for doc in docs:
        file_name = doc.get("file_name")
        if not file_name:
            continue
        print(f"\n>>> Processing doc: doc_id={doc.get('doc_id')} file_name={file_name}")
        result = run_pseudo_label_pipeline_for_doc(
            graph=graph,
            file_name=file_name,
            doc_id=doc.get("doc_id"),
            kg_scope=doc.get("kg_scope") or args.kg_scope,
            kg_id=doc.get("kg_id") or args.kg_id,
            config=cfg,
            progress_hook=progress_hook,
            console_preview_limit=args.console_preview,
        )
        if args.console_preview > 0:
            preview_rows = result.get("preview") or []
            if preview_rows:
                print(f"[preview] top {len(preview_rows)} pseudo-labels:")
                for i, row in enumerate(preview_rows, start=1):
                    print(
                        "  #{idx}: {src} --{rel}--> {tgt} | label={label} | conf={conf:.3f} | rule={rule}".format(
                            idx=i,
                            src=_truncate(row.get("source_text")),
                            rel=row.get("relation_type") or "?",
                            tgt=_truncate(row.get("target_text")),
                            label=row.get("label"),
                            conf=float(row.get("confidence") or 0.0),
                            rule=(row.get("rule_hits") or {}).get("primary_rule") or "-",
                        )
                    )
            else:
                print("[preview] no pseudo-label generated for this doc.")

        summaries.append(result)

    manifest = {
        "ok": True,
        "num_docs": len(summaries),
        "summaries": summaries,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
