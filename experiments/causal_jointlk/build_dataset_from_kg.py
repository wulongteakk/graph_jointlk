from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from langchain_community.graphs import Neo4jGraph  # type: ignore

from src.causal_jointlk.neo4j_accessor import CandidateEdge, Neo4jAccessor
from src.causal_jointlk.pseudo_labeler import load_yaml_file
from src.causal_jointlk.pseudo_pipeline import (
    effective_pseudo_label,
    read_doc_evidence_units,
    retrieve_lexical_units,
)
from src.evidence_store.sqlite_store import EvidenceStore  # type: ignore


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def deterministic_split(doc_key: str, train_ratio: float, dev_ratio: float) -> str:
    h = int(hashlib.sha1(doc_key.encode("utf-8")).hexdigest(), 16) % 10_000
    x = h / 10_000.0
    if x < train_ratio:
        return "train"
    if x < train_ratio + dev_ratio:
        return "dev"
    return "test"


def build_node_lookup(doc_edges: Sequence[CandidateEdge]) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    for e in doc_edges:
        nodes.setdefault(
            e.source_node_id,
            {
                "node_id": e.source_node_id,
                "text": e.source_text,
                "layer": e.source_layer,
                "labels": e.source_labels,
                "props": e.source_props,
            },
        )
        nodes.setdefault(
            e.target_node_id,
            {
                "node_id": e.target_node_id,
                "text": e.target_text,
                "layer": e.target_layer,
                "labels": e.target_labels,
                "props": e.target_props,
            },
        )
    return nodes


def build_pair_centered_subgraph(
    doc_edges: Sequence[CandidateEdge],
    source_node_id: str,
    target_node_id: str,
    hops: int = 1,
    max_nodes: int = 32,
) -> Dict[str, Any]:
    node_lookup = build_node_lookup(doc_edges)
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in doc_edges:
        adj[e.source_node_id].add(e.target_node_id)
        adj[e.target_node_id].add(e.source_node_id)

    seeds = [source_node_id, target_node_id]
    visited: Set[str] = set(seeds)
    q = deque([(source_node_id, 0), (target_node_id, 0)])
    ordered_nodes: List[str] = []

    while q and len(visited) < max_nodes:
        node_id, depth = q.popleft()
        if node_id not in ordered_nodes:
            ordered_nodes.append(node_id)
        if depth >= hops:
            continue
        for nb in sorted(adj.get(node_id, [])):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, depth + 1))
                if len(visited) >= max_nodes:
                    break

    for nid in [source_node_id, target_node_id]:
        if nid not in ordered_nodes:
            ordered_nodes.insert(0, nid)
    ordered_nodes = ordered_nodes[:max_nodes]

    node_idx = {nid: idx for idx, nid in enumerate(ordered_nodes)}

    edge_index: List[List[int]] = []
    edge_types: List[str] = []
    for e in doc_edges:
        if e.source_node_id in node_idx and e.target_node_id in node_idx:
            edge_index.append([node_idx[e.source_node_id], node_idx[e.target_node_id]])
            edge_types.append(e.relation_type)

    node_texts = [node_lookup[nid]["text"] for nid in ordered_nodes]
    node_layers = [node_lookup[nid]["layer"] for nid in ordered_nodes]
    node_scores = [1.0 for _ in ordered_nodes]

    return {
        "node_ids": ordered_nodes,
        "node_texts": node_texts,
        "node_layer_types": node_layers,
        "node_scores": node_scores,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "source_idx": node_idx[source_node_id],
        "target_idx": node_idx[target_node_id],
    }


def candidate_key(edge: CandidateEdge) -> Tuple[str, str, str]:
    return (edge.source_node_id, edge.target_node_id, edge.relation_type.upper())


def pair_key(src: str, tgt: str) -> Tuple[str, str]:
    return (src, tgt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/dev/test JSONL from gold chains + pseudo labels.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evidence_db_path", default="./data/evidence_store.sqlite3")
    parser.add_argument("--prior_config", default="configs/causal_prior.yaml")
    parser.add_argument("--min_pseudo_confidence", type=float, default=0.90)

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

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--local_hops", type=int, default=1)
    parser.add_argument("--max_local_nodes", type=int, default=32)
    parser.add_argument("--max_units_per_doc_scan", type=int, default=500)
    parser.add_argument("--lexical_top_k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_cfg = load_yaml_file(args.prior_config) if args.prior_config and os.path.exists(args.prior_config) else {}
    relation_whitelist = {
        str(x).strip().upper()
        for x in (
            prior_cfg.get("relation_type_whitelist")
            or prior_cfg.get("relation_whitelist")
            or prior_cfg.get("relations", {}).get("whitelist")
            or prior_cfg.get("relation_types")
            or []
        )
        if str(x).strip()
    }

    store = EvidenceStore(args.evidence_db_path)
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

    split_manifest: Dict[str, str] = {}
    samples_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    per_doc_stats: List[Dict[str, Any]] = []

    for doc in docs:
        doc_id = doc.get("doc_id")
        file_name = doc.get("file_name")
        if not file_name:
            continue

        split_key = str(doc_id or file_name)
        split = deterministic_split(split_key, args.train_ratio, args.dev_ratio)
        split_manifest[split_key] = split

        doc_edges = accessor.get_doc_candidate_edges(
            doc_id=doc_id,
            file_name=file_name,
            kg_scope=args.kg_scope,
            kg_id=args.kg_id,
            relation_types=None,
            limit=args.max_edges_per_doc,
        )
        if not doc_edges:
            per_doc_stats.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "split": split,
                    "num_samples": 0,
                    "reason": "no_doc_edges",
                }
            )
            continue

        unit_rows, unit_by_id, _ = read_doc_evidence_units(
            store,
            file_name=file_name,
            max_units=args.max_units_per_doc_scan,
        )

        gold_pair_to_edge_records: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for ge in store.list_causal_chain_edges(file_name=file_name):
            gold_pair_to_edge_records[(ge.source_node_id, ge.target_node_id)].append(ge)

        pseudo_map_exact: Dict[Tuple[str, str, str], Any] = {}
        pseudo_map_pair: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for pl in store.list_pseudo_edge_labels(file_name=file_name, doc_id=doc_id, min_confidence=args.min_pseudo_confidence):
            pseudo_map_exact[(pl.source_node_id, pl.target_node_id, (pl.relation_type or "").upper())] = pl
            pseudo_map_pair[(pl.source_node_id, pl.target_node_id)].append(pl)

        doc_samples: List[Dict[str, Any]] = []

        for edge in doc_edges:
            if relation_whitelist and edge.relation_type.upper() not in relation_whitelist:
                exact_pl = pseudo_map_exact.get(candidate_key(edge))
                if exact_pl is None:
                    continue

            label = None
            gold_relation = edge.relation_type
            label_source = None
            review_status = None
            pseudo_label_id = None
            label_confidence = None
            matched_pseudo = pseudo_map_exact.get(candidate_key(edge))

            if pair_key(edge.source_node_id, edge.target_node_id) in gold_pair_to_edge_records:
                label = 1
                label_source = "gold_chain"
                label_confidence = 1.0
            elif matched_pseudo is not None:
                eff = effective_pseudo_label(matched_pseudo)
                if eff is None:
                    continue
                label, maybe_rel, label_source = eff
                if maybe_rel:
                    gold_relation = maybe_rel
                review_status = matched_pseudo.review_status
                pseudo_label_id = matched_pseudo.pseudo_label_id
                label_confidence = float(getattr(matched_pseudo, "confidence", 1.0) or 1.0)
            else:
                pair_labels = pseudo_map_pair.get(pair_key(edge.source_node_id, edge.target_node_id), [])
                if pair_labels:
                    valid = [x for x in pair_labels if effective_pseudo_label(x) is not None]
                    if len(valid) == 1:
                        eff = effective_pseudo_label(valid[0])
                        if eff is not None:
                            label, maybe_rel, label_source = eff
                            if maybe_rel:
                                gold_relation = maybe_rel
                            review_status = valid[0].review_status
                            pseudo_label_id = valid[0].pseudo_label_id
                            label_confidence = float(getattr(valid[0], "confidence", 1.0) or 1.0)
                if label is None:
                    continue

            evidence_texts: List[str] = []
            evidence_unit_ids: List[str] = []

            if matched_pseudo is not None:
                if getattr(matched_pseudo, "evidence_text", None):
                    evidence_texts.append(matched_pseudo.evidence_text)
                if getattr(matched_pseudo, "evidence_unit_id", None):
                    evidence_unit_ids.append(matched_pseudo.evidence_unit_id)
                    if matched_pseudo.evidence_unit_id in unit_by_id:
                        evidence_texts.append(unit_by_id[matched_pseudo.evidence_unit_id]["content"])
                    else:
                        unit_obj = store.get_unit(matched_pseudo.evidence_unit_id)
                        if unit_obj is not None:
                            evidence_texts.append(unit_obj.content)

            if not evidence_texts and pair_key(edge.source_node_id, edge.target_node_id) in gold_pair_to_edge_records:
                for ge in gold_pair_to_edge_records[pair_key(edge.source_node_id, edge.target_node_id)]:
                    if getattr(ge, "evidence_unit_id", None):
                        evidence_unit_ids.append(ge.evidence_unit_id)
                        if ge.evidence_unit_id in unit_by_id:
                            evidence_texts.append(unit_by_id[ge.evidence_unit_id]["content"])
                        else:
                            unit_obj = store.get_unit(ge.evidence_unit_id)
                            if unit_obj is not None:
                                evidence_texts.append(unit_obj.content)

            if not evidence_texts:
                for unit in retrieve_lexical_units(edge.source_text, edge.target_text, unit_rows, top_k=args.lexical_top_k):
                    evidence_texts.append(unit["content"])
                    evidence_unit_ids.append(unit["unit_id"])

            evidence_texts = list(dict.fromkeys([x for x in evidence_texts if x]))
            evidence_unit_ids = list(dict.fromkeys([x for x in evidence_unit_ids if x]))

            local_subgraph = build_pair_centered_subgraph(
                doc_edges,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                hops=args.local_hops,
                max_nodes=args.max_local_nodes,
            )

            sample_id = f"smp_{sha1_text('|'.join([str(doc_id), file_name, edge.source_node_id, edge.relation_type, edge.target_node_id]))}"
            sample = {
                "sample_id": sample_id,
                "doc_id": doc_id,
                "file_name": file_name,
                "kg_scope": edge.kg_scope,
                "kg_id": edge.kg_id,
                "query": f"{file_name} 因果链抽取",
                "doc_title": file_name,
                "source_node_id": edge.source_node_id,
                "source_text": edge.source_text,
                "source_layer": edge.source_layer,
                "target_node_id": edge.target_node_id,
                "target_text": edge.target_text,
                "target_layer": edge.target_layer,
                "candidate_relation": edge.relation_type,
                "gold_relation": gold_relation,
                "label": int(label),
                "label_source": label_source,
                "review_status": review_status,
                "pseudo_label_id": pseudo_label_id,
                "label_confidence": label_confidence,
                "evidence_texts": evidence_texts,
                "evidence_unit_ids": evidence_unit_ids,
                **local_subgraph,
            }
            doc_samples.append(sample)

        samples_by_split[split].extend(doc_samples)
        per_doc_stats.append(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "split": split,
                "num_doc_edges": len(doc_edges),
                "num_gold_pairs": len(gold_pair_to_edge_records),
                "num_pseudo_exact": len(pseudo_map_exact),
                "num_samples": len(doc_samples),
            }
        )

    for split, rows in samples_by_split.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "num_docs": len(docs),
        "splits": {k: len(v) for k, v in samples_by_split.items()},
        "label_breakdown": {
            split: {
                "positive": sum(1 for x in rows if int(x["label"]) == 1),
                "negative": sum(1 for x in rows if int(x["label"]) == 0),
            }
            for split, rows in samples_by_split.items()
        },
        "split_manifest": split_manifest,
        "per_doc_stats": per_doc_stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, **manifest}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
