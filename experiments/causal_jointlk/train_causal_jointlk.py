from __future__ import annotations

import argparse
import json
import os
import random
import re
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
    compute_joint_score,
    compute_masked_binary_metrics,
    compute_relation_metrics,
    tune_binary_threshold,
)
from modeling.causal_jointlk_io import batchify_examples
from modeling.modeling_causal_jointlk import CausalJointLKModel, compute_training_loss
from backend.causal_jointlk.runtime_trace import RuntimeTracer

def resolve_input_path(path_str: str) -> Path:
    raw = str(path_str).strip().strip('"').strip("'")
    normalized_variants = {
        raw,
        raw.replace("\\", "/"),
        raw.replace("/", os.sep),
        raw.replace("\\", os.sep),
    }
    candidates: List[Path] = []
    seen: set[str] = set()
    for variant in normalized_variants:
        p = Path(variant).expanduser()
        for candidate in (p, REPO_ROOT / p):
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    for candidate in candidates:
        if path_exists(candidate):
            return candidate
    recovered = recover_manual_review_jsonl(candidates[0] if candidates else Path(raw))
    if recovered is not None:
        return recovered
    return candidates[0] if candidates else Path(raw)


def recover_manual_review_jsonl(missing_path: Path) -> Optional[Path]:
    if missing_path.suffix.lower() != ".jsonl":
        return None
    if missing_path.name != "jointlk_multitask_train.jsonl":
        return None
    expected_dir = missing_path.parent.name
    search_roots = [Path.cwd() / "artifacts" / "manual_review", REPO_ROOT / "artifacts" / "manual_review"]
    all_hits: List[Path] = []
    for root in search_roots:
        if root.exists():
            all_hits.extend(root.rglob("jointlk_multitask_train.jsonl"))
    if not all_hits:
        return None

    expected_norm = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", expected_dir).lower()

    def _score(p: Path) -> tuple[int, float]:
        parent_norm = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", p.parent.name).lower()
        overlap = 0
        for a, b in zip(expected_norm, parent_norm):
            if a != b:
                break
            overlap += 1
        return overlap, p.stat().st_mtime

    return sorted(all_hits, key=_score, reverse=True)[0]


def to_windows_long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    s = str(p)
    if s.startswith("\\\\?\\"):
        return p
    if s.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + s.lstrip("\\"))
    return Path("\\\\?\\" + s)


def path_exists(path: Path) -> bool:
    p = Path(path)
    if p.exists():
        return True
    if os.name == "nt":
        try:
            return to_windows_long_path(p).exists()
        except Exception:
            return False
    return False

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
    causal_losses: List[float] = []
    relation_losses: List[float] = []
    aux_losses: List[float] = []
    cf_losses: List[float] = []
    rows: List[Dict[str, Any]] = []
    multitask: Dict[str, Dict[str, List[float]]] = {
        "enable": {"gold": [], "prob": [], "mask": []},
        "dir": {"gold": [], "prob": [], "mask": []},
        "temp": {"gold": [], "prob": [], "mask": []},
        "src_first": {"gold": [], "prob": [], "mask": []},
        "dst_first": {"gold": [], "prob": [], "mask": []},
    }

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
            cf_group_ids=batch.get("twin_group_ids"),
            cf_roles=batch.get("cf_roles"),
        )
        losses.append(float(loss_dict["loss"].detach().cpu()))
        causal_losses.append(float(loss_dict["support_loss"].detach().cpu()))
        relation_losses.append(float(loss_dict["relation_loss"].detach().cpu()))
        aux_losses.append(float(loss_dict["aux_loss"].detach().cpu()))
        cf_losses.append(float(loss_dict["cf_loss"].detach().cpu()))

        batch_gold = batch["labels"].long().detach().cpu().tolist()
        batch_prob = outputs["support_prob"].detach().cpu().tolist()
        enable_prob = outputs["enable_prob"].detach().cpu().tolist()
        dir_prob = outputs["dir_prob"].detach().cpu().tolist()
        temp_prob = outputs["temporal_prob"].detach().cpu().tolist()
        src_first_prob = outputs["src_first_prob"].detach().cpu().tolist()
        dst_first_prob = outputs["dst_first_prob"].detach().cpu().tolist()
        batch_gold_rel = batch["relation_labels"].detach().cpu().tolist()
        batch_pred_rel = outputs["relation_logits"].argmax(dim=-1).detach().cpu().tolist()

        gold.extend(batch_gold)
        prob.extend(batch_prob)
        gold_rel.extend(batch_gold_rel)
        pred_rel.extend(batch_pred_rel)
        multitask["enable"]["gold"].extend(batch["enable_labels"].long().detach().cpu().tolist())
        multitask["enable"]["mask"].extend(batch["enable_mask"].long().detach().cpu().tolist())
        multitask["enable"]["prob"].extend(enable_prob)
        multitask["dir"]["gold"].extend(batch["dir_labels"].long().detach().cpu().tolist())
        multitask["dir"]["mask"].extend(batch["dir_mask"].long().detach().cpu().tolist())
        multitask["dir"]["prob"].extend(dir_prob)
        multitask["temp"]["gold"].extend(batch["temp_labels"].long().detach().cpu().tolist())
        multitask["temp"]["mask"].extend(batch["temp_mask"].long().detach().cpu().tolist())
        multitask["temp"]["prob"].extend(temp_prob)
        multitask["src_first"]["gold"].extend(batch["src_first_labels"].long().detach().cpu().tolist())
        multitask["src_first"]["mask"].extend(batch["src_first_mask"].long().detach().cpu().tolist())
        multitask["src_first"]["prob"].extend(src_first_prob)
        multitask["dst_first"]["gold"].extend(batch["dst_first_labels"].long().detach().cpu().tolist())
        multitask["dst_first"]["mask"].extend(batch["dst_first_mask"].long().detach().cpu().tolist())
        multitask["dst_first"]["prob"].extend(dst_first_prob)

        for meta, p, e_p, d_p, t_p, sf_p, df_p, y, gr, pr in zip(
                meta_rows,
                batch_prob,
                enable_prob,
                dir_prob,
                temp_prob,
                src_first_prob,
                dst_first_prob,
                batch_gold,
                batch_gold_rel,
                batch_pred_rel,
        ):
            row = dict(meta)
            row["support_prob"] = float(p)
            row["causal_prob"] = float(p)
            row["enable_prob"] = float(e_p)
            row["dir_prob"] = float(d_p)
            row["temporal_prob"] = float(t_p)
            row["src_first_prob"] = float(sf_p)
            row["dst_first_prob"] = float(df_p)
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

    metrics: Dict[str, Any] = {
        "loss": sum(losses) / max(len(losses), 1),
        "causal_loss": sum(causal_losses) / max(len(causal_losses), 1),
        "relation_loss": sum(relation_losses) / max(len(relation_losses), 1),
        "aux_loss": sum(aux_losses) / max(len(aux_losses), 1),
        "cf_loss": sum(cf_losses) / max(len(cf_losses), 1),
    }
    metrics.update(compute_binary_edge_metrics(gold, prob, threshold=threshold))
    metrics.update(compute_relation_metrics(gold_rel, pred_rel))
    metrics.update(compute_masked_binary_metrics(**multitask["enable"], prefix="enable", threshold=threshold))
    metrics.update(compute_masked_binary_metrics(**multitask["dir"], prefix="dir", threshold=threshold))
    metrics.update(compute_masked_binary_metrics(**multitask["temp"], prefix="temp", threshold=threshold))
    metrics.update(compute_masked_binary_metrics(**multitask["src_first"], prefix="src_first", threshold=threshold))
    metrics.update(compute_masked_binary_metrics(**multitask["dst_first"], prefix="dst_first", threshold=threshold))
    metrics["best_threshold"] = threshold
    metrics["ranking_by_doc"] = compute_grouped_ranking_metrics(rows, group_keys=["doc_id"])
    metrics["ranking_by_doc_source"] = compute_grouped_ranking_metrics(rows, group_keys=["doc_id", "source_node_id"])
    metrics["breakdown_by_label_source"] = compute_breakdown_by_field(rows, field="label_source", threshold=threshold)
    metrics["breakdown_by_review_status"] = compute_breakdown_by_field(rows, field="review_status", threshold=threshold)
    metrics["joint_score"] = compute_joint_score(metrics)
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
    raw_train_jsonl = args.train_jsonl
    raw_dev_jsonl = args.dev_jsonl
    raw_prior_config = args.prior_config
    args.train_jsonl = str(resolve_input_path(args.train_jsonl))
    args.dev_jsonl = str(resolve_input_path(args.dev_jsonl))
    args.prior_config = str(resolve_input_path(args.prior_config))
    if raw_train_jsonl != args.train_jsonl:
        print(f"[jointlk-train] resolved train_jsonl: {raw_train_jsonl} -> {args.train_jsonl}")
    if raw_dev_jsonl != args.dev_jsonl:
        print(f"[jointlk-train] resolved dev_jsonl: {raw_dev_jsonl} -> {args.dev_jsonl}")
    if raw_prior_config != args.prior_config:
        print(f"[jointlk-train] resolved prior_config: {raw_prior_config} -> {args.prior_config}")

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
    tracer = RuntimeTracer(enabled=True)
    tracer.log_stage_console(
        "train-setup",
        {
            "train_size": len(train_set),
            "dev_size": len(dev_set),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model_name": args.model_name,
            "relation_loss_weight": args.relation_loss_weight,
            "pos_weight": pos_weight_value,
        },
    )
    tracer.log_jointlk_input_preview_console(train_set.records, top_k=3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_causal_losses: List[float] = []
        epoch_relation_losses: List[float] = []
        epoch_aux_losses: List[float] = []
        epoch_cf_losses: List[float] = []

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
                cf_group_ids=batch.get("twin_group_ids"),
                cf_roles=batch.get("cf_roles"),
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_causal_losses.append(float(loss_dict["support_loss"].detach().cpu()))
            epoch_relation_losses.append(float(loss_dict["relation_loss"].detach().cpu()))
            epoch_aux_losses.append(float(loss_dict["aux_loss"].detach().cpu()))
            epoch_cf_losses.append(float(loss_dict["cf_loss"].detach().cpu()))

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
        dev_metrics["train_causal_loss"] = sum(epoch_causal_losses) / max(len(epoch_causal_losses), 1)
        dev_metrics["train_relation_loss"] = sum(epoch_relation_losses) / max(len(epoch_relation_losses), 1)
        dev_metrics["train_aux_loss"] = sum(epoch_aux_losses) / max(len(epoch_aux_losses), 1)
        dev_metrics["train_cf_loss"] = sum(epoch_cf_losses) / max(len(epoch_cf_losses), 1)
        dev_metrics["pos_weight"] = pos_weight_value
        history.append(dev_metrics)

        current_metric_raw = dev_metrics.get("joint_score")
        if current_metric_raw is None:
            current_metric_raw = compute_joint_score(dev_metrics)
        try:
            current_metric = float(current_metric_raw)
        except Exception:
            current_metric = 0.0
        dev_metrics["joint_score"] = current_metric
        current_threshold = float(dev_metrics.get("best_threshold", 0.5))
        dashboard = {
            "epoch": epoch,
            "train_loss": float(dev_metrics.get("train_loss", 0.0)),
            "train_causal_loss": float(dev_metrics.get("train_causal_loss", 0.0)),
            "train_relation_loss": float(dev_metrics.get("train_relation_loss", 0.0)),
            "train_aux_loss": float(dev_metrics.get("train_aux_loss", 0.0)),
            "train_cf_loss": float(dev_metrics.get("train_cf_loss", 0.0)),
            "edge_f1": float(dev_metrics.get("edge_f1", 0.0)),
            "enable_f1": float(dev_metrics.get("enable_f1", 0.0)),
            "dir_f1": float(dev_metrics.get("dir_f1", 0.0)),
            "temp_f1": float(dev_metrics.get("temp_f1", 0.0)),
            "src_first_f1": float(dev_metrics.get("src_first_f1", 0.0)),
            "dst_first_f1": float(dev_metrics.get("dst_first_f1", 0.0)),
            "rel_micro_f1": float(dev_metrics.get("rel_micro_f1", 0.0)),
            "rel_macro_f1": float(dev_metrics.get("rel_macro_f1", 0.0)),
            "mrr": float((dev_metrics.get("ranking_by_doc", {}) or {}).get("mrr", dev_metrics.get("mrr", 0.0))),
            "hits@1": float((dev_metrics.get("ranking_by_doc", {}) or {}).get("hits@1", 0.0)),
            "hits@3": float((dev_metrics.get("ranking_by_doc", {}) or {}).get("hits@3", 0.0)),
            "hits@5": float((dev_metrics.get("ranking_by_doc", {}) or {}).get("hits@5", 0.0)),
            "joint_score": float(dev_metrics.get("joint_score", 0.0)),
            "best_threshold": float(dev_metrics.get("best_threshold", 0.5)),
        }
        print("[JointLK][epoch-summary]", json.dumps(dashboard, ensure_ascii=False))
        print("[JointLK][train-epoch]", json.dumps(tracer.log_train_epoch(dev_metrics, epoch=epoch), ensure_ascii=False))
        tracer.log_train_epoch_console(dashboard, epoch=epoch)
        tracer.log_stage_console("training-metrics", dashboard)

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
        "best_joint_score": best_metric,
        "best_edge_f1": max((float(x.get("edge_f1", 0.0)) for x in history), default=0.0),
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
