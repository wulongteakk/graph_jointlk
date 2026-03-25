from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

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
    tune_binary_threshold,
)
from modeling.causal_jointlk_io import batchify_examples
from modeling.modeling_causal_jointlk import CausalJointLKModel, compute_training_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_collate_fn(
    tokenizer,
    relation_to_id: Dict[str, int],
    node_type_to_id: Dict[str, int],
    max_text_length: int,
    max_node_text_length: int,
):
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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



def build_label_weight_map(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "gold_chain": float(args.weight_gold_chain),
        "pseudo_review_edited": float(args.weight_pseudo_edited),
        "pseudo_review_accepted": float(args.weight_pseudo_accepted),
        "pseudo_pending": float(args.weight_pseudo_pending),
    }



def infer_pos_weight(dataset: CausalEdgeDataset) -> float:
    labels = [int(row.get("label", 0)) for row in dataset.records]
    pos = sum(labels)
    neg = max(len(labels) - pos, 0)
    if pos <= 0:
        return 1.0
    return max(1.0, neg / max(pos, 1))


@torch.no_grad()
def evaluate(
    model: CausalJointLKModel,
    data_loader: DataLoader,
    device: torch.device,
    relation_loss_weight: float,
    pos_weight_value: Optional[float],
    threshold: Optional[float] = None,
    tune_threshold: bool = False,
    id_to_relation: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    gold: List[int] = []
    prob: List[float] = []
    gold_rel: List[int] = []
    pred_rel: List[int] = []
    losses: List[float] = []
    rows: List[Dict[str, Any]] = []

    pos_weight_tensor = None
    if pos_weight_value is not None:
        pos_weight_tensor = torch.tensor(float(pos_weight_value), device=device)

    for batch in tqdm(data_loader, desc="eval", leave=False):
        meta_rows = batch.get("meta_rows", [])
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)

        outputs = model(batch)
        loss_dict = compute_training_loss(
            outputs=outputs,
            labels=batch["labels"],
            relation_labels=batch["relation_labels"],
            multitask_labels={
                "enable_labels": batch.get("enable_labels"),
                "dir_labels": batch.get("dir_labels"),
                "temp_labels": batch.get("temp_labels"),
                "src_first_labels": batch.get("src_first_labels"),
                "dst_first_labels": batch.get("dst_first_labels"),
                "enable_mask": batch.get("enable_mask"),
                "dir_mask": batch.get("dir_mask"),
                "temp_mask": batch.get("temp_mask"),
                "src_first_mask": batch.get("src_first_mask"),
                "dst_first_mask": batch.get("dst_first_mask"),
            },
            relation_loss_weight=relation_loss_weight,
            sample_weights=batch.get("sample_weights"),
            pos_weight=pos_weight_tensor,
        )
        losses.append(float(loss_dict["loss"].detach().cpu()))

        batch_gold = batch["labels"].long().detach().cpu().tolist()
        batch_prob = outputs["support_prob"].detach().cpu().tolist()
        batch_gold_rel = batch["relation_labels"].detach().cpu().tolist()
        batch_pred_rel = outputs["relation_logits"].argmax(dim=-1).detach().cpu().tolist()

        gold.extend(batch_gold)
        prob.extend(batch_prob)
        gold_rel.extend(batch_gold_rel)
        pred_rel.extend(batch_pred_rel)

        for meta, p, y, gr, pr in zip(meta_rows, batch_prob, batch_gold, batch_gold_rel, batch_pred_rel):
            row = dict(meta)
            row["support_prob"] = float(p)
            row["gold_label"] = int(y)
            row["gold_relation_id"] = int(gr)
            row["pred_relation_id"] = int(pr)
            if id_to_relation is not None:
                row["gold_relation_name"] = id_to_relation.get(int(gr), "UNK")
                row["pred_relation_name"] = id_to_relation.get(int(pr), "UNK")
            rows.append(row)

    if threshold is None and tune_threshold:
        threshold_info = tune_binary_threshold(gold, prob)
        threshold = float(threshold_info["best_threshold"])
    else:
        threshold = float(threshold if threshold is not None else 0.5)

    metrics: Dict[str, Any] = {"loss": sum(losses) / max(len(losses), 1)}
    metrics.update(compute_binary_edge_metrics(gold, prob, threshold=threshold))
    metrics.update(compute_relation_metrics(gold_rel, pred_rel))
    metrics["best_threshold"] = threshold
    metrics["ranking_by_doc"] = compute_grouped_ranking_metrics(rows, group_keys=["doc_id"])
    metrics["ranking_by_doc_source"] = compute_grouped_ranking_metrics(rows, group_keys=["doc_id", "source_node_id"])
    metrics["breakdown_by_label_source"] = compute_breakdown_by_field(rows, field="label_source", threshold=threshold)
    metrics["breakdown_by_review_status"] = compute_breakdown_by_field(rows, field="review_status", threshold=threshold)
    return metrics, rows



def main() -> None:
    parser = argparse.ArgumentParser(description="Train final causal JointLK model on pseudo/gold edge dataset.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", required=True)
    parser.add_argument("--prior_config", default="configs/causal_prior.yaml")
    parser.add_argument("--model_name", default="roberta-large")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_text_length", type=int, default=320)
    parser.add_argument("--max_node_text_length", type=int, default=24)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--relation_loss_weight", type=float, default=0.20)
    parser.add_argument("--freeze_lm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_evidence", type=int, default=3)
    parser.add_argument("--pos_weight", type=float, default=-1.0, help="<=0 means auto infer from train set")
    parser.add_argument("--eval_threshold", type=float, default=-1.0, help="<0 means tune on dev each epoch")
    parser.add_argument("--weight_gold_chain", type=float, default=1.00)
    parser.add_argument("--weight_pseudo_edited", type=float, default=1.00)
    parser.add_argument("--weight_pseudo_accepted", type=float, default=0.95)
    parser.add_argument("--weight_pseudo_pending", type=float, default=0.80)
    parser.add_argument("--include_label_sources", nargs="*", default=None)
    parser.add_argument("--exclude_label_sources", nargs="*", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_weight_map = build_label_weight_map(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_set = CausalEdgeDataset(
        args.train_jsonl,
        args.prior_config,
        label_weight_map=label_weight_map,
        max_evidence=args.max_evidence,
        include_label_sources=args.include_label_sources,
        exclude_label_sources=args.exclude_label_sources,
    )
    dev_set = CausalEdgeDataset(
        args.dev_jsonl,
        args.prior_config,
        label_weight_map=label_weight_map,
        max_evidence=args.max_evidence,
        include_label_sources=args.include_label_sources,
        exclude_label_sources=args.exclude_label_sources,
    )

    collate_fn = build_collate_fn(
        tokenizer=tokenizer,
        relation_to_id=train_set.relation_to_id,
        node_type_to_id=train_set.node_type_to_id,
        max_text_length=args.max_text_length,
        max_node_text_length=args.max_node_text_length,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalJointLKModel(
        model_name=args.model_name,
        num_relations=len(train_set.relation_to_id),
        num_node_types=len(train_set.node_type_to_id),
        hidden_size=args.hidden_size,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        freeze_lm=args.freeze_lm,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    pos_weight_value = float(args.pos_weight) if args.pos_weight > 0 else infer_pos_weight(train_set)
    pos_weight_tensor = torch.tensor(pos_weight_value, device=device)

    best_metric = -1.0
    best_threshold = 0.5
    history: List[Dict[str, Any]] = []
    id_to_relation = {v: k for k, v in train_set.relation_to_id.items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss_dict = compute_training_loss(
                outputs=outputs,
                labels=batch["labels"],
                relation_labels=batch["relation_labels"],
                multitask_labels={
                    "enable_labels": batch.get("enable_labels"),
                    "dir_labels": batch.get("dir_labels"),
                    "temp_labels": batch.get("temp_labels"),
                    "src_first_labels": batch.get("src_first_labels"),
                    "dst_first_labels": batch.get("dst_first_labels"),
                    "enable_mask": batch.get("enable_mask"),
                    "dir_mask": batch.get("dir_mask"),
                    "temp_mask": batch.get("temp_mask"),
                    "src_first_mask": batch.get("src_first_mask"),
                    "dst_first_mask": batch.get("dst_first_mask"),
                },
                relation_loss_weight=args.relation_loss_weight,
                sample_weights=batch.get("sample_weights"),
                pos_weight=pos_weight_tensor,
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))

        dev_metrics, _ = evaluate(
            model=model,
            data_loader=dev_loader,
            device=device,
            relation_loss_weight=args.relation_loss_weight,
            pos_weight_value=pos_weight_value,
            threshold=(args.eval_threshold if args.eval_threshold >= 0 else None),
            tune_threshold=(args.eval_threshold < 0),
            id_to_relation=id_to_relation,
        )
        dev_metrics["epoch"] = epoch
        dev_metrics["train_loss"] = sum(epoch_losses) / max(len(epoch_losses), 1)
        dev_metrics["pos_weight"] = pos_weight_value
        history.append(dev_metrics)

        current_metric = float(dev_metrics.get("edge_f1", 0.0))
        current_threshold = float(dev_metrics.get("best_threshold", 0.5))
        print(json.dumps(dev_metrics, ensure_ascii=False, indent=2))

        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = current_threshold
            ckpt = {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "relation_to_id": train_set.relation_to_id,
                "node_type_to_id": train_set.node_type_to_id,
                "best_dev_metrics": dev_metrics,
                "best_threshold": best_threshold,
                "label_weight_map": label_weight_map,
                "pos_weight": pos_weight_value,
            }
            torch.save(ckpt, output_dir / "best_model.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")

        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(dev_metrics, ensure_ascii=False) + "\n")

    summary = {
        "best_edge_f1": best_metric,
        "best_threshold": best_threshold,
        "history": history,
        "train_size": len(train_set),
        "dev_size": len(dev_set),
        "label_weight_map": label_weight_map,
        "pos_weight": pos_weight_value,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
