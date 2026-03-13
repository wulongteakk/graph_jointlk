# 自动 pseudo-label 流水线 + 可直接跑训练/评估实验包




## 一、上传后自动流水线



流程：

上传报告
→ 构图
→ 规则打 pseudo-label
→ 写回 SQLite `pseudo_edge_labels`
→ 导出人工抽检包

## 二、人工抽检与训练数据导出

### 1. 生成 pseudo-label 和抽检包

```bash
python experiments/causal_jointlk/build_pseudo_labels_from_kg.py \
  --output_dir data/pseudo_label_round1 \
  --evidence_db_path ./data/evidence_store.sqlite3 \
  --prior_config configs/causal_prior.yaml \
  --pseudo_rule_config configs/causal_pseudo_label_rules.yaml \
  --neo4j_url bolt://localhost:7687 \
  --neo4j_username neo4j \
  --neo4j_password your_password \
  --kg_scope instance
```

### 2. 人工修改 `manual_review_candidates.csv` 后回写

```bash
python experiments/causal_jointlk/import_manual_review.py \
  --review_csv data/pseudo_label_round1/manual_review_candidates.csv \
  --evidence_db_path ./data/evidence_store.sqlite3
```

### 3. 导出最终训练集

```bash
python experiments/causal_jointlk/build_dataset_from_kg.py \
  --output_dir data/causal_jointlk_supervised_v1 \
  --evidence_db_path ./data/evidence_store.sqlite3 \
  --prior_config configs/causal_prior.yaml \
  --min_pseudo_confidence 0.90 \
  --neo4j_url bolt://localhost:7687 \
  --neo4j_username neo4j \
  --neo4j_password your_password \
  --kg_scope instance
```

生成结果：

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `manifest.json`

## 三、训练

这版训练已经对齐最终 schema：

- 兼容 `edge_index = [[src, dst], ...]` 和 `[[srcs], [dsts]]`
- 使用 `label_source / review_status / label_confidence` 自动生成 `sample_weight`
- 支持 gold / pseudo 混合训练
- dev 集自动调阈值，保存 `best_threshold`
- 评估输出总体指标、label_source 分组指标、按文档排序指标

```bash
python experiments/causal_jointlk/train_causal_jointlk.py \
  --train_jsonl data/causal_jointlk_supervised_v1/train.jsonl \
  --dev_jsonl data/causal_jointlk_supervised_v1/dev.jsonl \
  --prior_config configs/causal_prior.yaml \
  --model_name roberta-large \
  --output_dir saved_models/causal_jointlk_final \
  --batch_size 4 \
  --epochs 8 \
  --lr 2e-5 \
  --weight_gold_chain 1.0 \
  --weight_pseudo_edited 1.0 \
  --weight_pseudo_accepted 0.95 \
  --weight_pseudo_pending 0.80
```

训练输出：

- `best_model.pt`
- `tokenizer/`
- `train_log.jsonl`
- `summary.json`

## 四、评估

```bash
python experiments/causal_jointlk/eval_causal_jointlk.py \
  --test_jsonl data/causal_jointlk_supervised_v1/test.jsonl \
  --checkpoint saved_models/causal_jointlk_final/best_model.pt \
  --prior_config configs/causal_prior.yaml \
  --output_json results/causal_jointlk_eval.json \
  --output_rows_jsonl results/causal_jointlk_eval_rows.jsonl
```

默认会使用训练时保存在 checkpoint 里的 `best_threshold`。

## 五、这版相对旧训练脚本的关键修正

1. `build_dataset_from_kg.py` 导出的最终 schema 已经被训练包直接兼容。  
2. 修正了 `edge_index` 旧版读法和新版导出格式不一致的问题。  
3. 支持 pseudo/gold 样本权重，不再把弱监督和 gold 一视同仁。  
4. 增加 dev 阈值调优，避免固定 0.5 导致不稳。  
5. eval 增加 `label_source` / `review_status` 分组和文档级排序指标，适合论文表格。  
6. `modeling/modeling_causal_jointlk.py` 增加了无 `torch_geometric` 时的 fallback relational conv，环境更稳。  

## 六、对比

- Rule / baseline only
- Rule + evidence gate
- JointLK-style + pseudo weak supervision
- JointLK-style + pseudo + manual review
- JointLK-style + gold only
- JointLK-style + gold + pseudo mixed

核心报告指标：

- `edge_precision / edge_recall / edge_f1`
- `rel_micro_f1 / rel_macro_f1`
- `ranking_by_doc.mrr / hits@k / ndcg@5`
- `breakdown_by_label_source`
- `breakdown_by_review_status`
