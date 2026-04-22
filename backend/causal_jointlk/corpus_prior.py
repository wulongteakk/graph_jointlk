from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .corpus_registry import _read_jsonl


def build_prototype_causal_prior(
    *,
    registry_path: str | Path = "artifacts/causal_corpus/registry.jsonl",
    output_path: str | Path = "artifacts/causal_corpus/prototype_causal_prior.json",
    alpha: float = 1.0,
    beta: float = 3.0,
) -> Dict[str, Any]:
    rows = _read_jsonl(Path(registry_path))
    edge_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "source_proto": None,
        "target_proto": None,
        "relation_family": "CAUSES",
        "doc_ids": set(),
        "positive_docs": set(),
        "negative_docs": set(),
        "unresolved_docs": set(),
        "positive_count": 0,
        "negative_count": 0,
        "unresolved_count": 0,
        "examples": [],
    })

    for item in rows:
        train_jsonl = str(item.get("train_jsonl") or "")
        if not train_jsonl or not Path(train_jsonl).exists():
            continue
        for rec in _read_jsonl(Path(train_jsonl)):
            srcp = rec.get("source_proto")
            tgtp = rec.get("target_proto")
            if not srcp or not tgtp:
                continue
            key = f"{srcp}->{tgtp}"
            stat = edge_stats[key]
            stat["source_proto"] = srcp
            stat["target_proto"] = tgtp
            did = str(rec.get("doc_id") or item.get("doc_id") or "UNKNOWN")
            stat["doc_ids"].add(did)
            mask = int(rec.get("support_mask", rec.get("causal_mask", 0)) or 0)
            label = int(rec.get("support_label", rec.get("label", 0)) or 0)
            if mask != 1:
                stat["unresolved_count"] += 1
                stat["unresolved_docs"].add(did)
            elif label == 1:
                stat["positive_count"] += 1
                stat["positive_docs"].add(did)
            else:
                stat["negative_count"] += 1
                stat["negative_docs"].add(did)
            if len(stat["examples"]) < 5:
                stat["examples"].append(
                    {
                        "doc_id": did,
                        "source_text": rec.get("source_text"),
                        "target_text": rec.get("target_text"),
                        "support_label": label,
                    }
                )

    output_edges: List[Dict[str, Any]] = []
    for key, stat in sorted(edge_stats.items()):
        pos_docs = len(stat["positive_docs"])
        neg_docs = len(stat["negative_docs"])
        doc_support = len(stat["doc_ids"])
        prior_prob = (pos_docs + float(alpha)) / max(pos_docs + neg_docs + float(alpha + beta), 1e-8)
        confidence = doc_support / max(doc_support + 2.0, 1.0)
        output_edges.append(
            {
                "edge_proto_key": key,
                "source_proto": stat["source_proto"],
                "target_proto": stat["target_proto"],
                "relation_family": stat["relation_family"],
                "doc_support": doc_support,
                "positive_count": stat["positive_count"],
                "negative_count": stat["negative_count"],
                "unlabeled_count": stat["unresolved_count"],
                "prior_prob": prior_prob,
                "confidence": confidence,
                "example_doc_ids": sorted(list(stat["doc_ids"]))[:10],
                "examples": stat["examples"],
            }
        )

    payload = {
        "registry_path": str(Path(registry_path).resolve()),
        "num_patterns": len(output_edges),
        "patterns": output_edges,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload