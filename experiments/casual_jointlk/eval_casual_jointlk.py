from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in [str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.causal_jointlk.dataset import CausalEdgeDataset
from experiments.causal_jointlk.metrics import (
    compute_binary_edge_metrics,
    compute_breakdown_by_field,
    compute_grouped_ranking_metrics,
    compute_relation_metrics,
)
from modeling.causal_jointlk_io import batchify_examples
from modeling.modeling_causal_jointlk import CausalJointLKModel



def build_collate_fn(tokenizer, relation_to_id, node_type_to_id, max_text_length, max_node_text_length):
    def collate_fn(examples):
        return batchify_examples(
            examples=examples,
            tokenizer=tokenizer,
            relation_to_id=relation_to_id,
            node_type_to_id=node_type_to_id,
            max_text_length=max_text_length,
            max_node_text_length=max_node_text_length,
            device=None,
        )

    return collate_fn


@torch.no_grad()
def run_eval(model, data_loader, device, threshold, id_to_relation):
    model.eval()
    gold: List[int] = []
    prob: List[float] = []
    gold_rel: List[int] = []
    pred_rel: List[int] = []
    rows: List[Dict[str, Any]] = []

    for batch in data_loader:
        meta_rows = batch.get("meta_rows", [])
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)

        outputs = model(batch)
        pred_prob = outputs["support_prob"].detach().cpu().tolist()
        pred_rel_batch = outputs["relation_logits"].argmax(dim=-1).detach().cpu().tolist()
        gold_batch = batch["labels"].long().detach().cpu().tolist()
        gold_rel_batch = batch["relation_labels"].detach().cpu().tolist()

        gold.extend(gold_batch)
        prob.extend(pred_prob)
        gold_rel.extend(gold_rel_batch)
        pred_rel.extend(pred_rel_batch)

        for meta, p, y, gr, pr in zip(meta_rows, pred_prob, gold_batch, gold_rel_batch, pred_rel_batch):
            row = dict(meta)
            row["support_prob"] = float(p)
            row["pred_label"] = int(float(p) >= threshold)
            row["gold_label"] = int(y)
            row["gold_relation_id"] = int(gr)
            row["pred_relation_id"] = int(pr)
            row["gold_relation_name"] = id_to_relation.get(int(gr), "UNK")
            row["pred_relation_name"] = id_to_relation.get(int(pr), "UNK")
            rows.append(row)

    metrics = {}
    metrics.update(compute_binary_edge_metrics(gold, prob, threshold=threshold))
    metrics.update(compute_relation_metrics(gold_rel, pred_rel))
    metrics["ranking_by_doc"] = compute_grouped_ranking_metrics(rows, group_keys=["doc_id"])
    metrics["ranking_by_doc_source"] = compute_grouped_ranking_metrics(rows, group_keys=["doc_id", "source_node_id"])
    metrics["breakdown_by_label_source"] = compute_breakdown_by_field(rows, field="label_source", threshold=threshold)
    metrics["breakdown_by_review_status"] = compute_breakdown_by_field(rows, field="review_status", threshold=threshold)
    return metrics, rows



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final causal JointLK model.")
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior_config", default="configs/causal_prior.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_text_length", type=int, default=320)
    parser.add_argument("--max_node_text_length", type=int, default=24)
    parser.add_argument("--max_evidence", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=-1.0, help="<0 means load best_threshold from checkpoint or fallback 0.5")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_rows_jsonl", default="")
    parser.add_argument("--include_label_sources", nargs="*", default=None)
    parser.add_argument("--exclude_label_sources", nargs="*", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    relation_to_id = checkpoint["relation_to_id"]
    node_type_to_id = checkpoint["node_type_to_id"]
    label_weight_map = checkpoint.get("label_weight_map")

    model_name = saved_args.get("model_name", "roberta-large")
    hidden_size = int(saved_args.get("hidden_size", 256))
    num_gnn_layers = int(saved_args.get("num_gnn_layers", 3))
    dropout = float(saved_args.get("dropout", 0.2))
    freeze_lm = bool(saved_args.get("freeze_lm", False))

    tokenizer_dir = Path(args.checkpoint).parent / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_set = CausalEdgeDataset(
        args.test_jsonl,
        args.prior_config,
        label_weight_map=label_weight_map,
        max_evidence=args.max_evidence,
        include_label_sources=args.include_label_sources,
        exclude_label_sources=args.exclude_label_sources,
    )
    collate_fn = build_collate_fn(
        tokenizer,
        relation_to_id=relation_to_id,
        node_type_to_id=node_type_to_id,
        max_text_length=args.max_text_length,
        max_node_text_length=args.max_node_text_length,
    )
    data_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalJointLKModel(
        model_name=model_name,
        num_relations=len(relation_to_id),
        num_node_types=len(node_type_to_id),
        hidden_size=hidden_size,
        num_gnn_layers=num_gnn_layers,
        dropout=dropout,
        freeze_lm=freeze_lm,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    threshold = float(args.threshold)
    if threshold < 0:
        threshold = float(checkpoint.get("best_threshold", 0.5))

    id_to_relation = {v: k for k, v in relation_to_id.items()}
    metrics, rows = run_eval(model, data_loader, device, threshold, id_to_relation)

    output = {
        "metrics": metrics,
        "rows": rows,
        "checkpoint": args.checkpoint,
        "threshold": threshold,
        "test_size": len(test_set),
    }
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_rows_jsonl:
        out_path = Path(args.output_rows_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
