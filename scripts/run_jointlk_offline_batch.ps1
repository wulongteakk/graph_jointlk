Param(
    [string]$Registry = "artifacts/causal_corpus/registry.jsonl",
    [string]$TrainJsonl = "artifacts/causal_corpus/corpus_train.jsonl",
    [string]$DevJsonl = "artifacts/causal_corpus/corpus_dev.jsonl",
    [string]$OutputDir = "outputs/offline_jointlk_batch",
    [double]$DevRatio = 0.2,
    [int]$MaxEdgesPerDoc = 800,
    [string]$ModelName = "roberta-large",
    [int]$Epochs = 5,
    [int]$BatchSize = 4,
    [string]$Lr = "2e-5",
    [string]$PriorConfig = "configs/causal_prior.yaml",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Registry: $Registry"
if (!(Test-Path $Registry)) {
    Write-Error "registry not found: $Registry`n请先上传多份报告并确认 pseudo label 已注册到 registry。"
}

$trainDir = Split-Path -Parent $TrainJsonl
$devDir = Split-Path -Parent $DevJsonl
if ($trainDir) { New-Item -ItemType Directory -Force -Path $trainDir | Out-Null }
if ($devDir) { New-Item -ItemType Directory -Force -Path $devDir | Out-Null }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$splitCode = @"
from backend.causal_jointlk.corpus_registry import build_corpus_train_dev, read_corpus_stats
reg = r"""$Registry"""
train_out = r"""$TrainJsonl"""
dev_out = r"""$DevJsonl"""
print("[INFO] corpus stats:", read_corpus_stats(reg))
res = build_corpus_train_dev(
    registry_path=reg,
    train_out=train_out,
    dev_out=dev_out,
    exclude_doc_id=None,
    dev_ratio=float($DevRatio),
    max_edges_per_doc=int($MaxEdgesPerDoc),
)
print("[INFO] split result:", res)
"@

python -c $splitCode

python experiments/causal_jointlk/train_causal_jointlk.py `
  --train_jsonl "$TrainJsonl" `
  --dev_jsonl "$DevJsonl" `
  --prior_config "$PriorConfig" `
  --model_name "$ModelName" `
  --output_dir "$OutputDir" `
  --epochs "$Epochs" `
  --batch_size "$BatchSize" `
  --lr "$Lr" `
  --seed "$Seed"

Write-Host "[DONE] Offline batch training finished. Output: $OutputDir"