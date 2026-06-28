#!/usr/bin/env bash
#SBATCH --job-name=t3-phi3
#SBATCH --output=/data/home/yixh/RAG/runs/t3_cross_family/logs/slurm_phi3_%A_%a.out
#SBATCH --error=/data/home/yixh/RAG/runs/t3_cross_family/logs/slurm_phi3_%A_%a.err
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

ROOT="${ROOT:-/data/home/yixh/RAG}"
ENV_NAME="${ENV_NAME:-vsdn-casvr}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/yixh/.conda/envs/$ENV_NAME/bin/python}"
MODEL="${MODEL:-/data/home/yixh/models/Phi-3-mini-4k-instruct}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/t3_cross_family}"
CONTEXTS="${CONTEXTS:-$RUN_DIR/contexts_phi3_300.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-phi3}"
NUM_SHARDS="${NUM_SHARDS:-8}"
K="${K:-5}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "$ROOT"
mkdir -p "$RUN_DIR/shards" "$RUN_DIR/logs"

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

contexts = Path(os.environ["CONTEXTS"])
run_dir = Path(os.environ["RUN_DIR"])
num_shards = int(os.environ.get("NUM_SHARDS", "8"))
outs = [(run_dir / "shards" / f"contexts_shard_{i}.jsonl").open("w", encoding="utf-8") for i in range(num_shards)]
try:
    with contexts.open("r", encoding="utf-8") as src:
        for idx, line in enumerate(src):
            outs[idx % num_shards].write(line)
finally:
    for out in outs:
        out.close()
PY

"$PYTHON_BIN" scripts/run_model_phase2.py \
  --contexts "$RUN_DIR/shards/contexts_shard_${SHARD_ID}.jsonl" \
  --out "$RUN_DIR/predictions_${OUTPUT_PREFIX}_shard_${SHARD_ID}.jsonl" \
  --model "$MODEL" \
  --k "$K" \
  --dtype bfloat16 \
  --device-map auto \
  --no-trust-remote-code
