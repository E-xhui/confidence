# Slurm Job Index

| Job ID | 实验 | 模型 | 数据 | 日志文件 |
|--------|------|------|------|----------|
| 3765 | Phase 2 Mistral smoke test | Mistral-7B-Instruct-v0.3 | phase2_main (smoke) | `runs/phase2_mistral_smoke_slurm_3765.err` |
| 3691 | Phase 2 mis reader support | Qwen2.5-14B-Instruct | phase2_unique mis条件 | `runs/phase2_unique/logs/slurm_reader_3691.*` |
| 3692 | Phase 2 sup reader support | Qwen2.5-14B-Instruct | phase2_unique sup条件 | `runs/phase2_unique/logs/slurm_reader_3692.*` |
| 3732 | Phase 2 irr 条件采样 | Qwen2.5-7B-Instruct | phase2_unique_irr (8 shards) | `runs/phase2_unique_irr/logs/slurm_qwen_3732_0~7.*` |
| 3895 | T3 跨家族复现 | Phi-3-mini-4k-instruct | 300题×5条件×5采样 (8 shards) | `runs/t3_cross_family/logs/slurm_phi3_3895_0~7.*` |

## 说明

- `.out`：标准输出（模型加载进度、推理日志）
- `.err`：标准错误（transformers warning 等，正常有内容）
- phase2_main 的 `qwen_shard_0~7.log` 是本地运行日志，没有 Slurm job ID
