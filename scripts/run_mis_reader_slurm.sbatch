#!/usr/bin/env bash
#SBATCH --job-name=rag-mis-reader
#SBATCH --output=/data/home/yixh/RAG/runs/phase2_unique/logs/slurm_reader_%A.out
#SBATCH --error=/data/home/yixh/RAG/runs/phase2_unique/logs/slurm_reader_%A.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

set -euo pipefail

ROOT="${ROOT:-/data/home/yixh/RAG}"
ENV_NAME="${ENV_NAME:-vsdn-casvr}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/yixh/.conda/envs/$ENV_NAME/bin/python}"
READER_MODEL="${READER_MODEL:-/data/home/yixh/models/Qwen2.5-14B-Instruct}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/phase2_unique}"

cd "$ROOT"
mkdir -p "$RUN_DIR/logs"

"$PYTHON_BIN" scripts/run_mis_reader_support.py \
  --workset "$RUN_DIR/workset.jsonl" \
  --out "$RUN_DIR/mis_reader_support.jsonl" \
  --model "$READER_MODEL"

"$PYTHON_BIN" scripts/summarize_reader_support.py \
  --reader-output "$RUN_DIR/mis_reader_support.jsonl" \
  --out "$RUN_DIR/mis_reader_support_summary.json"
