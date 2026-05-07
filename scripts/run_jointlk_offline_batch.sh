#!/usr/bin/env bash
set -euo pipefail

# 批量离线训练 causal JointLK（基于已上传文档生成的 pseudo labels）
# 用法示例：
#   bash scripts/run_jointlk_offline_batch.sh \
#     --registry artifacts/causal_corpus/registry.jsonl \
#     --output outputs/offline_jointlk_batch_$(date +%Y%m%d_%H%M%S)

REGISTRY="artifacts/causal_corpus/registry.jsonl"
TRAIN_JSONL="artifacts/causal_corpus/corpus_train.jsonl"
DEV_JSONL="artifacts/causal_corpus/corpus_dev.jsonl"
OUTPUT_DIR="outputs/offline_jointlk_batch"
DEV_RATIO="0.2"
MAX_EDGES_PER_DOC="800"
MODEL_NAME="roberta-large"
EPOCHS="5"
BATCH_SIZE="4"
LR="2e-5"
PRIOR_CONFIG="configs/causal_prior.yaml"
SEED="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift 2 ;;
    --train_jsonl) TRAIN_JSONL="$2"; shift 2 ;;
    --dev_jsonl) DEV_JSONL="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --dev_ratio) DEV_RATIO="$2"; shift 2 ;;
    --max_edges_per_doc) MAX_EDGES_PER_DOC="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --prior_config) PRIOR_CONFIG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ ! -f "$REGISTRY" ]]; then
  echo "[ERROR] registry not found: $REGISTRY"
  echo "请先确保每个文档已完成 pseudo label 导出并登记到 registry。"
  exit 1
fi

mkdir -p "$(dirname "$TRAIN_JSONL")" "$(dirname "$DEV_JSONL")" "$OUTPUT_DIR"

python - <<PY
from backend.causal_jointlk.corpus_registry import build_corpus_train_dev, read_corpus_stats
stats = read_corpus_stats("$REGISTRY")
print("[INFO] corpus stats:", stats)
res = build_corpus_train_dev(
    registry_path="$REGISTRY",
    train_out="$TRAIN_JSONL",
    dev_out="$DEV_JSONL",
    exclude_doc_id=None,
    dev_ratio=float("$DEV_RATIO"),
    max_edges_per_doc=int("$MAX_EDGES_PER_DOC"),
)
print("[INFO] split result:", res)
PY

python experiments/causal_jointlk/train_causal_jointlk.py \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --prior_config "$PRIOR_CONFIG" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --seed "$SEED"

echo "[DONE] Offline batch training finished. Output: $OUTPUT_DIR"