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

function Resolve-PathWithBackendFallback {
    Param(
        [string]$PathValue,
        [string]$Label
    )

    if (Test-Path $PathValue) {
        return $PathValue
    }

    $backendCandidate = Join-Path "backend" $PathValue
    if (Test-Path $backendCandidate) {
        Write-Host "[WARN] $Label not found at '$PathValue', fallback to '$backendCandidate'"
        return $backendCandidate
    }

    return $PathValue
}

$Registry = Resolve-PathWithBackendFallback -PathValue $Registry -Label "registry"
$TrainJsonl = Resolve-PathWithBackendFallback -PathValue $TrainJsonl -Label "train_jsonl"
$DevJsonl = Resolve-PathWithBackendFallback -PathValue $DevJsonl -Label "dev_jsonl"
$PriorConfig = Resolve-PathWithBackendFallback -PathValue $PriorConfig -Label "prior_config"

Write-Host "[INFO] Registry: $Registry"
if (!(Test-Path $Registry)) {
    Write-Error "registry not found: $Registry`n请先上传多份报告并确认 pseudo label 已注册到 registry。"
}

$trainDir = Split-Path -Parent $TrainJsonl
$devDir = Split-Path -Parent $DevJsonl
if ($trainDir) { New-Item -ItemType Directory -Force -Path $trainDir | Out-Null }
if ($devDir) { New-Item -ItemType Directory -Force -Path $devDir | Out-Null }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# 通过临时 .py 文件调用，避免不同 shell/编码下对 here-string 的解析差异
$splitPyPath = Join-Path $env:TEMP ("jointlk_split_" + [guid]::NewGuid().ToString("N") + ".py")

@"
from backend.causal_jointlk.corpus_registry import build_corpus_train_dev, read_corpus_stats

reg = r'''$Registry'''
train_out = r'''$TrainJsonl'''
dev_out = r'''$DevJsonl'''

print('[INFO] corpus stats:', read_corpus_stats(reg))
res = build_corpus_train_dev(
    registry_path=reg,
    train_out=train_out,
    dev_out=dev_out,
    exclude_doc_id=None,
    dev_ratio=float($DevRatio),
    max_edges_per_doc=int($MaxEdgesPerDoc),
)
print('[INFO] split result:', res)
"@ | Set-Content -Path $splitPyPath -Encoding UTF8

try {
    python $splitPyPath
}
finally {
    if (Test-Path $splitPyPath) {
        Remove-Item -Force $splitPyPath
    }
}

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