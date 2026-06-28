#!/usr/bin/env bash
#SBATCH --job-name=rag-llm-judge
#SBATCH --output=/data/home/yixh/RAG/runs/phase2_five_condition/logs/slurm_judge_%j.out
#SBATCH --error=/data/home/yixh/RAG/runs/phase2_five_condition/logs/slurm_judge_%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G

set -euo pipefail

ROOT="${ROOT:-/data/home/yixh/RAG}"
ENV_NAME="${ENV_NAME:-vsdn-casvr}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/yixh/.conda/envs/$ENV_NAME/bin/python}"
CANDIDATES="${CANDIDATES:?Set CANDIDATES JSONL path}"
OUT="${OUT:?Set OUT JSONL path}"
MODEL="${MODEL:?Set MODEL path or HuggingFace id}"
LIMIT="${LIMIT:-}"
DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

cd "$ROOT"
mkdir -p "$(dirname "$OUT")" "$ROOT/runs/phase2_five_condition/logs"

args=(
  scripts/run_llm_judge.py
  --candidates "$CANDIDATES"
  --out "$OUT"
  --model "$MODEL"
  --dtype "$DTYPE"
  --device-map "$DEVICE_MAP"
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" "${args[@]}"
