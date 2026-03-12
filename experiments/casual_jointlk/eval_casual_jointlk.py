
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.causal_jointlk.dataset import CausalEdgeDataset
from experiments.causal_jointlk.metrics import compute_binary_edge_metrics, compute_relation_metrics
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
def run_eval(model, data_loader, device):
    model.eval()
    gold: List[int] = []
    prob: List[float] = []
    gold_rel: List[int] = []
    pred_rel: List[int] = []
    rows: List[Dict[str, Any]] = []

    for batch in data_loader:
        sample_ids = batch.get("sample_ids", [])
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

        for sample_id, p, y, gr, pr in zip(sample_ids, pred_prob, gold_batch, gold_rel_batch, pred_rel_batch):
            rows.append(
                {
                    "sample_id": sample_id,
                    "support_prob": p,
                    "gold_label": y,
                    "gold_relation": gr,
                    "pred_relation": pr,
                }
            )

    metrics = {}
    metrics.update(compute_binary_edge_metrics(gold, prob))
    metrics.update(compute_relation_metrics(gold_rel, pred_rel))
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior_config", default="configs/causal_prior.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_text_length", type=int, default=320)
    parser.add_argument("--max_node_text_length", type=int, default=24)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    relation_to_id = checkpoint["relation_to_id"]
    node_type_to_id = checkpoint["node_type_to_id"]

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

    test_set = CausalEdgeDataset(args.test_jsonl, args.prior_config)
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

    metrics, rows = run_eval(model, data_loader, device)
    output = {
        "metrics": metrics,
        "rows": rows,
        "checkpoint": args.checkpoint,
    }
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
