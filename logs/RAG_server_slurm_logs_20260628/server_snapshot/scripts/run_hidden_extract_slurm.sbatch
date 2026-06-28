#!/usr/bin/env bash
#SBATCH --job-name=rag-hidden-probe
#SBATCH --output=/data/home/yixh/RAG/runs/phase4_probe/logs/slurm_hidden_%j.out
#SBATCH --error=/data/home/yixh/RAG/runs/phase4_probe/logs/slurm_hidden_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

set -euo pipefail

ROOT="${ROOT:-/data/home/yixh/RAG}"
ENV_NAME="${ENV_NAME:-vsdn-casvr}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/yixh/.conda/envs/$ENV_NAME/bin/python}"
MODEL="${MODEL:?Set MODEL to a local model path or HuggingFace id}"
CONTEXTS="${CONTEXTS:-$ROOT/runs/phase2_five_condition/contexts_phase2.jsonl}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for hidden-state artifacts}"
CONDITIONS="${CONDITIONS:-irr,same_entity_irr,sup,mis}"
LAYERS="${LAYERS:-all}"
POOLINGS="${POOLINGS:-last,mean}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

cd "$ROOT"
mkdir -p "$ROOT/runs/phase4_probe/logs" "$OUT_DIR"

"$PYTHON_BIN" scripts/extract_hidden_states_phase4.py \
  --contexts "$CONTEXTS" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL" \
  --conditions "$CONDITIONS" \
  --layers "$LAYERS" \
  --poolings "$POOLINGS" \
  --batch-size "$BATCH_SIZE" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP"
