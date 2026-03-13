from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

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
    return parser.parse_args()


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
    for doc in docs:
        file_name = doc.get("file_name")
        if not file_name:
            continue
        result = run_pseudo_label_pipeline_for_doc(
            graph=graph,
            file_name=file_name,
            doc_id=doc.get("doc_id"),
            kg_scope=doc.get("kg_scope") or args.kg_scope,
            kg_id=doc.get("kg_id") or args.kg_id,
            config=cfg,
        )
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
