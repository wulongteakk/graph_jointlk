# scripts/run_corpus_train.py
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)



import logging
import argparse
from backend.causal_jointlk.corpus_registry import build_corpus_train, read_corpus_stats

# 配置日志，实时输出到 PowerShell
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=str, required=True, help='Path to registry.jsonl')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_edges_per_doc', type=int, default=100, help='Max edges per doc')
    args = parser.parse_args()

    logging.info("Starting corpus build...")
    build_corpus_train(
        registry_path=args.registry,
        max_edges_per_doc=args.max_edges_per_doc,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    logging.info("Corpus build finished!")

    stats = read_corpus_stats(args.registry)
    logging.info(f"Corpus stats: {stats}")

if __name__ == "__main__":
    main()