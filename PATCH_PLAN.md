
# causal_jointlk patch bundle

这个补丁的目标是把你仓库里现有的“硬接 QA JointLK”方式，重构成一个**面向因果链抽取**的、**可训练**、**可插拔**、**可与普通抽取方式直接对比**的模块。

## 设计原则

1. **只在 Instance-KG 上工作**  
   不依赖 BG-KG / ConceptNet 作为推理图；图只来自你自己的报告实例图谱。
2. **JointLK-style，而不是 QA-style**  
   不是把 QA 的 `num_choice` 强行套用到因果链任务，而是改造成**边支持打分器**。
3. **先做边打分，再做全局组链**  
   Local edge score + global beam search，是最适合论文 ablation 的结构。
4. **普通抽取方式保留为 baseline**  
   baseline 和 jointlk 共享相同候选图与 beam search，唯一差别是 local scorer。
5. **证据闸门单独保留**  
   神经网络分数和 evidence gate 解耦，方便做四种对照：
   - baseline
   - baseline + evidence gate
   - jointlk
   - jointlk + evidence gate

## 文件列表

### 新增
- `configs/causal_prior.yaml`
- `modeling/causal_jointlk_io.py`
- `modeling/modeling_causal_jointlk.py`
- `experiments/causal_jointlk/dataset.py`
- `experiments/causal_jointlk/metrics.py`
- `experiments/causal_jointlk/train_causal_jointlk.py`
- `experiments/causal_jointlk/eval_causal_jointlk.py`
- `backend/src/causal_jointlk/__init__.py`
- `backend/src/causal_jointlk/schemas.py`
- `backend/src/causal_jointlk/prior.py`
- `backend/src/causal_jointlk/instance_kg_builder.py`
- `backend/src/causal_jointlk/neo4j_accessor.py`
- `backend/src/causal_jointlk/evidence_gate.py`
- `backend/src/causal_jointlk/baseline_extractor.py`
- `backend/src/causal_jointlk/beam_search.py`
- `backend/src/causal_jointlk/jointlk_edge_scorer.py`
- `backend/src/causal_jointlk/service.py`

### 建议保留但不再继续扩展
- `backend/src/jointlk_integration/*`  
  这套更像“QA 模型接 Neo4j demo”，不适合继续演化成因果链抽取主线。

## 论文实验推荐指标

### 1. 候选子图覆盖
- Gold edge coverage
- Gold node coverage

### 2. 边级别抽取
- Edge Precision / Recall / F1
- Relation micro-F1 / macro-F1
- AUROC / AUPR（support score）

### 3. 链级别抽取
- Chain Exact Match
- Chain Edge-F1
- Chain Node-F1
- MRR / Hits@K / nDCG@K（Top-K chain ranking）

### 4. 证据一致性
- Evidence Support Rate
- Unsupported Edge Rate（越低越好）

### 5. 结构合理性
- Layer Violation Rate
- Missing Layer Count

## 建议的主表

| 方法 | Edge-F1 | Chain Edge-F1 | Chain EM | MRR | Evidence Support Rate | Layer Violation Rate |
|---|---:|---:|---:|---:|---:|---:|
| 普通抽取 |  |  |  |  |  |  |
| 普通抽取 + 闸门 |  |  |  |  |  |  |
| JointLK-style |  |  |  |  |  |  |
| JointLK-style + 闸门 |  |  |  |  |  |  |

## 运行建议

### 训练
```bash
python experiments/causal_jointlk/train_causal_jointlk.py \
  --train_jsonl data/causal/train.jsonl \
  --dev_jsonl data/causal/dev.jsonl \
  --prior_config configs/causal_prior.yaml \
  --output_dir saved_models/causal_jointlk
```

### 评估
```bash
python experiments/causal_jointlk/eval_causal_jointlk.py \
  --test_jsonl data/causal/test.jsonl \
  --checkpoint saved_models/causal_jointlk/best_model.pt \
  --prior_config configs/causal_prior.yaml \
  --output_json saved_models/causal_jointlk/test_eval.json
```

## 在线服务用法（示意）

```python
from src.causal_jointlk.service import CausalJointLKService
from src.evidence_store.sqlite_store import EvidenceStore

service = CausalJointLKService(
    graph=neo4j_graph,
    evidence_store=EvidenceStore(),
    prior_config_path="configs/causal_prior.yaml",
    jointlk_checkpoint_path="saved_models/causal_jointlk/best_model.pt",
)

result = service.extract(
    query="桥梁坍塌的致因链",
    target_text="桥梁坍塌",
    doc_id="doc-001",
    mode="jointlk",
    k_hop=2,
    top_k=5,
    persist=True,
)
print(result.to_dict())
```
