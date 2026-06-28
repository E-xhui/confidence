#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
Refusing to launch an 8-GPU job directly.

gpu01 is only for smoke tests / interactive checks. Use:
  scripts/run_qwen_single_gpu.sh

For full/batch runs, submit through Slurm:
  sbatch scripts/run_qwen_slurm.sbatch
MSG

exit 1

