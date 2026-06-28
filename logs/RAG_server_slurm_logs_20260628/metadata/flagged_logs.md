# Flagged Logs

These were flagged by a lightweight text scan of stderr files.

| Source | Job ID | Path | Reason From Log Tail |
|---|---:|---|---|
| server | 3765 | `server_snapshot/runs/phase2_mistral_smoke/logs/slurm_3765.err` | Hugging Face config request timed out repeatedly; job cancelled on `gpu-07` at `2026-06-21T11:08:51`. |
| server | 3766 | `server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3766.err` | `ModuleNotFoundError: No module named 'statsmodels'`. |
| server | 3767 | `server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3767.err` | Missing input file `runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl`. |
| server | 3768 | `server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3768.err` | Job cancelled on `gpu-03` at `2026-06-21T11:08:03`. |
| server | 3769 | `server_snapshot/runs/phase4_probe/qwen7_probe_quick/logs/slurm_3769.err` | Job cancelled on `gpu-03` at `2026-06-21T11:08:03`. |

Other `empty` hints usually mean one side of a Slurm `.out` / `.err` pair had no content, which is normal for several array jobs.
