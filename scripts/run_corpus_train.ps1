# scripts/run_corpus_train.ps1
param(
    [string]$Registry = "backend/artifacts/causal_corpus/registry.jsonl",
    [int]$Epochs = 5,
    [int]$BatchSize = 4
)

# 激活虚拟环境（可选，如果已手动激活，可注释）
# & conda activate llm-graph-builder

# 运行 Python 脚本，实时打印日志，忽略警告
python -W ignore -u scripts/run_corpus_train.py --registry $Registry --epochs $Epochs --batch_size $BatchSize