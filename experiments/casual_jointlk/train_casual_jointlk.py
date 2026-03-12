
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from experiments.causal_jointlk.dataset import CausalEdgeDataset
from experiments.causal_jointlk.metrics import compute_binary_edge_metrics, compute_relation_metrics
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


@torch.no_grad()
def evaluate(
    model: CausalJointLKModel,
    data_loader: DataLoader,
    device: torch.device,
    relation_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    gold: List[int] = []
    prob: List[float] = []
    gold_rel: List[int] = []
    pred_rel: List[int] = []
    losses: List[float] = []

    for batch in tqdm(data_loader, desc="eval", leave=False):
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        outputs = model(batch)
        loss_dict = compute_training_loss(
            outputs=outputs,
            labels=batch["labels"],
            relation_labels=batch["relation_labels"],
            relation_loss_weight=relation_loss_weight,
        )
        losses.append(float(loss_dict["loss"].detach().cpu()))
        gold.extend(batch["labels"].long().detach().cpu().tolist())
        prob.extend(outputs["support_prob"].detach().cpu().tolist())
        gold_rel.extend(batch["relation_labels"].detach().cpu().tolist())
        pred_rel.extend(outputs["relation_logits"].argmax(dim=-1).detach().cpu().tolist())

    metrics = {"loss": sum(losses) / max(len(losses), 1)}
    metrics.update(compute_binary_edge_metrics(gold, prob))
    metrics.update(compute_relation_metrics(gold_rel, pred_rel))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_set = CausalEdgeDataset(args.train_jsonl, args.prior_config)
    dev_set = CausalEdgeDataset(args.dev_jsonl, args.prior_config)

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

    best_metric = -1.0
    history: List[Dict[str, Any]] = []

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
                relation_loss_weight=args.relation_loss_weight,
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))

        dev_metrics = evaluate(
            model=model,
            data_loader=dev_loader,
            device=device,
            relation_loss_weight=args.relation_loss_weight,
        )
        dev_metrics["epoch"] = epoch
        dev_metrics["train_loss"] = sum(epoch_losses) / max(len(epoch_losses), 1)
        history.append(dev_metrics)

        current_metric = dev_metrics.get("edge_f1", 0.0)
        print(json.dumps(dev_metrics, ensure_ascii=False, indent=2))

        if current_metric > best_metric:
            best_metric = current_metric
            ckpt = {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "relation_to_id": train_set.relation_to_id,
                "node_type_to_id": train_set.node_type_to_id,
                "best_dev_metrics": dev_metrics,
            }
            torch.save(ckpt, output_dir / "best_model.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")

        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(dev_metrics, ensure_ascii=False) + "\n")

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_edge_f1": best_metric,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
