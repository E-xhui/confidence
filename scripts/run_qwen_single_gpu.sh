#!/usr/bin/env bash
set -euo pipefail

# gpu01 is a shared test node. This script is intentionally conservative:
# by default it uses one GPU, five questions, and k=1.

ROOT="${ROOT:-/data/home/yixh/RAG}"
ENV_NAME="${ENV_NAME:-vsdn-casvr}"
CONDA_SH="${CONDA_SH:-/local/ssd1/software/anaconda3/etc/profile.d/conda.sh}"
MODEL="${MODEL:-/data/home/yixh/models/Qwen2.5-7B-Instruct}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/phase2_main}"
GPU_ID="${GPU_ID:-0}"
K="${K:-1}"
LIMIT_QUESTIONS="${LIMIT_QUESTIONS:-5}"
OUT="${OUT:-$RUN_DIR/predictions_qwen_smoke_gpu${GPU_ID}.jsonl}"

source "$CONDA_SH"
cd "$ROOT"
mkdir -p "$RUN_DIR/logs"

limit_args=()
if [[ "$LIMIT_QUESTIONS" != "all" ]]; then
  limit_args=(--limit-questions "$LIMIT_QUESTIONS")
fi

echo "Running single-GPU smoke/test on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n "$ENV_NAME" python scripts/run_model_phase2.py \
  --contexts "$RUN_DIR/contexts_phase2.jsonl" \
  --out "$OUT" \
  --model "$MODEL" \
  --k "$K" \
  "${limit_args[@]}"

echo "Wrote: $OUT"

