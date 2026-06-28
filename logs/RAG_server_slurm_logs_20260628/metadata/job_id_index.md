# RAG Slurm / Server Log Index

Generated: 2026-06-28 16:39:55

## Package Layout

- `server_snapshot/`: copied from `gpu01:/data/home/yixh/RAG`, preserving `runs/` and Slurm-related `scripts/`.
- `local_snapshot/`: local `/Users/yixuhui/Desktop/RAG/runs` log snapshot for comparison/backfill.
- `metadata/job_id_index.csv`: one row per source + run + job id.
- `metadata/log_file_manifest.csv`: one row per copied log/script file.
- `metadata/old_slurm_job_index.md`: previous local index, kept for traceability if present.

## Notes

- `gpu01` SSH worked as `yixh@gpu-01`; `login` alias timed out during this collection.
- Slurm client commands such as `sacct`, `squeue`, `scontrol`, and `sbatch` were not on PATH on `gpu01`, so job ids are indexed from log filenames and sbatch output templates.
- `NO_SLURM_JOB_ID` rows are ordinary local/runtime logs such as `qwen_shard_*.log`, not Slurm `.out/.err` files.
- `status_hint` is only a text scan helper: `empty`, `log_present`, `check_err_text`, or `needs_review`. Treat it as a triage hint, not final job accounting.

## Job ID Summary

| Source | Run | Job ID | Array Tasks | Files | Bytes | Status Hints | Example Paths |
|---|---|---:|---|---:|---:|---|---|
| local | phase2_main | NO_SLURM_JOB_ID | 0,1,2,3,4,5,6,7 | 8 | 20720 | log_present | local_snapshot/runs/phase2_main/logs/qwen_shard_0.log \| local_snapshot/runs/phase2_main/logs/qwen_shard_1.log \| local_snapshot/runs/phase2_main/logs/qwen_shard_2.log \| local_snapshot/runs/phase2_main/logs/qwen_shard_3.log |
| local | phase2_mistral_smoke_slurm_3765.err | 3765 |  | 1 | 963 | needs_review | local_snapshot/runs/phase2_mistral_smoke_slurm_3765.err |
| local | phase2_unique | 3691 |  | 2 | 131 | empty,log_present | local_snapshot/runs/phase2_unique/logs/slurm_reader_3691.err \| local_snapshot/runs/phase2_unique/logs/slurm_reader_3691.out |
| local | phase2_unique | 3692 |  | 2 | 17059 | log_present | local_snapshot/runs/phase2_unique/logs/slurm_reader_3692.err \| local_snapshot/runs/phase2_unique/logs/slurm_reader_3692.out |
| local | phase2_unique_irr | 3732 | 0,1,2,3,4,5,6,7 | 16 | 19007 | empty,log_present | local_snapshot/runs/phase2_unique_irr/logs/slurm_qwen_3732_0.err \| local_snapshot/runs/phase2_unique_irr/logs/slurm_qwen_3732_0.out \| local_snapshot/runs/phase2_unique_irr/logs/slurm_qwen_3732_1.err \| local_snapshot/runs/phase2_unique_irr/logs/slurm_qwen_3732_1.out |
| local | phase4_probe | 3770 |  | 2 | 455 | empty,log_present | local_snapshot/runs/phase4_probe/qwen7_probe_veracity_quick/logs/slurm_3770.err \| local_snapshot/runs/phase4_probe/qwen7_probe_veracity_quick/logs/slurm_3770.out |
| local | phase4_probe | 3771 |  | 2 | 462 | empty,log_present | local_snapshot/runs/phase4_probe/qwen7_probe_all_closed_known_vquick/logs/slurm_3771.err \| local_snapshot/runs/phase4_probe/qwen7_probe_all_closed_known_vquick/logs/slurm_3771.out |
| local | phase4_probe | 3772 |  | 2 | 450 | empty,log_present | local_snapshot/runs/phase4_probe/qwen7_probe_clean_all_vquick/logs/slurm_3772.err \| local_snapshot/runs/phase4_probe/qwen7_probe_clean_all_vquick/logs/slurm_3772.out |
| local | phase4_probe | 3773 |  | 2 | 438 | empty,log_present | local_snapshot/runs/phase4_probe/qwen7_probe_all_vquick/logs/slurm_3773.err \| local_snapshot/runs/phase4_probe/qwen7_probe_all_vquick/logs/slurm_3773.out |
| local | phase4_probe | 3774 |  | 2 | 567 | empty,log_present | local_snapshot/runs/phase4_probe/qwen7_probe_controls_quick/logs/slurm_3774.err \| local_snapshot/runs/phase4_probe/qwen7_probe_controls_quick/logs/slurm_3774.out |
| local | phase4_probe | 3822 |  | 2 | 1378 | empty,log_present | local_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_3822.err \| local_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_3822.out |
| local | phase4_probe | 3823 |  | 2 | 4260 | empty,log_present | local_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_gate5_3823.err \| local_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_gate5_3823.out |
| local | phase4_probe | 3824 |  | 2 | 3147 | empty,log_present | local_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_gate6_3824.err \| local_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_gate6_3824.out |
| local | t3_cross_family | 3895 | 0,1,2,3,4,5,6,7 | 16 | 54829 | empty,log_present | local_snapshot/runs/t3_cross_family/logs/slurm_phi3_3895_0.err \| local_snapshot/runs/t3_cross_family/logs/slurm_phi3_3895_0.out \| local_snapshot/runs/t3_cross_family/logs/slurm_phi3_3895_1.err \| local_snapshot/runs/t3_cross_family/logs/slurm_phi3_3895_1.out |
| server |  | NO_SLURM_JOB_ID |  | 8 | 10271 | log_present | server_snapshot/scripts/run_hidden_extract_slurm.sbatch \| server_snapshot/scripts/run_llm_judge_slurm.sbatch \| server_snapshot/scripts/run_mis_reader_slurm.sbatch \| server_snapshot/scripts/run_model_2gpu_slurm.sbatch |
| server | phase2_five_condition | 3750 | 0 | 2 | 3842 | empty,log_present | server_snapshot/runs/phase2_five_condition/logs/slurm_model2_3750_0.err \| server_snapshot/runs/phase2_five_condition/logs/slurm_model2_3750_0.out |
| server | phase2_five_condition | 3751 | 0,1,2,3,4,5,6,7 | 16 | 28176 | empty,log_present | server_snapshot/runs/phase2_five_condition/logs/slurm_model2_3751_0.err \| server_snapshot/runs/phase2_five_condition/logs/slurm_model2_3751_0.out \| server_snapshot/runs/phase2_five_condition/logs/slurm_model2_3751_1.err \| server_snapshot/runs/phase2_five_condition/logs/slurm_model2_3751_1.out |
| server | phase2_five_condition | 3762 |  | 2 | 29195 | empty,log_present | server_snapshot/runs/phase2_five_condition/logs/slurm_judge_3762.err \| server_snapshot/runs/phase2_five_condition/logs/slurm_judge_3762.out |
| server | phase2_five_condition | 3763 |  | 2 | 13213 | empty,log_present | server_snapshot/runs/phase2_five_condition/logs/slurm_judge_3763.err \| server_snapshot/runs/phase2_five_condition/logs/slurm_judge_3763.out |
| server | phase2_main | 3693 | 0,1,2,3,4,5,6,7 | 16 | 30586 | empty,log_present | server_snapshot/runs/phase2_main/logs/slurm_qwen_3693_0.err \| server_snapshot/runs/phase2_main/logs/slurm_qwen_3693_0.out \| server_snapshot/runs/phase2_main/logs/slurm_qwen_3693_1.err \| server_snapshot/runs/phase2_main/logs/slurm_qwen_3693_1.out |
| server | phase2_main | 3741 | 0,1,2,3,4,5,6,7 | 16 | 19045 | empty,log_present | server_snapshot/runs/phase2_main/logs/slurm_qwen_3741_0.err \| server_snapshot/runs/phase2_main/logs/slurm_qwen_3741_0.out \| server_snapshot/runs/phase2_main/logs/slurm_qwen_3741_1.err \| server_snapshot/runs/phase2_main/logs/slurm_qwen_3741_1.out |
| server | phase2_main | NO_SLURM_JOB_ID | 0,1,2,3,4,5,6,7 | 8 | 20720 | log_present | server_snapshot/runs/phase2_main/logs/qwen_shard_0.log \| server_snapshot/runs/phase2_main/logs/qwen_shard_1.log \| server_snapshot/runs/phase2_main/logs/qwen_shard_2.log \| server_snapshot/runs/phase2_main/logs/qwen_shard_3.log |
| server | phase2_mistral_smoke | 3765 |  | 2 | 963 | empty,needs_review | server_snapshot/runs/phase2_mistral_smoke/logs/slurm_3765.err \| server_snapshot/runs/phase2_mistral_smoke/logs/slurm_3765.out |
| server | phase2_unique | 3691 |  | 2 | 131 | empty,log_present | server_snapshot/runs/phase2_unique/logs/slurm_reader_3691.err \| server_snapshot/runs/phase2_unique/logs/slurm_reader_3691.out |
| server | phase2_unique | 3692 |  | 2 | 17059 | log_present | server_snapshot/runs/phase2_unique/logs/slurm_reader_3692.err \| server_snapshot/runs/phase2_unique/logs/slurm_reader_3692.out |
| server | phase2_unique | 3740 |  | 2 | 15455 | log_present | server_snapshot/runs/phase2_unique/logs/slurm_sup_reader_3740.err \| server_snapshot/runs/phase2_unique/logs/slurm_sup_reader_3740.out |
| server | phase2_unique_irr | 3732 | 0,1,2,3,4,5,6,7 | 16 | 19007 | empty,log_present | server_snapshot/runs/phase2_unique_irr/logs/slurm_qwen_3732_0.err \| server_snapshot/runs/phase2_unique_irr/logs/slurm_qwen_3732_0.out \| server_snapshot/runs/phase2_unique_irr/logs/slurm_qwen_3732_1.err \| server_snapshot/runs/phase2_unique_irr/logs/slurm_qwen_3732_1.out |
| server | phase2_with_same_entity_irr | 3749 |  | 2 | 4265 | empty,log_present | server_snapshot/runs/phase2_with_same_entity_irr/logs/slurm_judge_3749.err \| server_snapshot/runs/phase2_with_same_entity_irr/logs/slurm_judge_3749.out |
| server | phase3_construct_validation | 3838 |  | 2 | 14491 | empty,log_present | server_snapshot/runs/phase3_construct_validation/validity_robustness/logs/stim_audit_q14_3838.err \| server_snapshot/runs/phase3_construct_validation/validity_robustness/logs/stim_audit_q14_3838.out |
| server | phase3_construct_validation | 3839 |  | 2 | 13168 | empty,log_present | server_snapshot/runs/phase3_construct_validation/validity_robustness/logs/stim_audit_q72_3839.err \| server_snapshot/runs/phase3_construct_validation/validity_robustness/logs/stim_audit_q72_3839.out |
| server | phase4_probe | 3764 |  | 2 | 29150 | log_present | server_snapshot/runs/phase4_probe/logs/slurm_hidden_3764.err \| server_snapshot/runs/phase4_probe/logs/slurm_hidden_3764.out |
| server | phase4_probe | 3766 |  | 2 | 323 | empty,needs_review | server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3766.err \| server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3766.out |
| server | phase4_probe | 3767 |  | 2 | 806 | empty,needs_review | server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3767.err \| server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3767.out |
| server | phase4_probe | 3768 |  | 2 | 79 | empty,needs_review | server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3768.err \| server_snapshot/runs/phase4_probe/qwen7_probe/logs/slurm_3768.out |
| server | phase4_probe | 3769 |  | 2 | 79 | empty,needs_review | server_snapshot/runs/phase4_probe/qwen7_probe_quick/logs/slurm_3769.err \| server_snapshot/runs/phase4_probe/qwen7_probe_quick/logs/slurm_3769.out |
| server | phase4_probe | 3770 |  | 2 | 455 | empty,log_present | server_snapshot/runs/phase4_probe/qwen7_probe_veracity_quick/logs/slurm_3770.err \| server_snapshot/runs/phase4_probe/qwen7_probe_veracity_quick/logs/slurm_3770.out |
| server | phase4_probe | 3771 |  | 2 | 462 | empty,log_present | server_snapshot/runs/phase4_probe/qwen7_probe_all_closed_known_vquick/logs/slurm_3771.err \| server_snapshot/runs/phase4_probe/qwen7_probe_all_closed_known_vquick/logs/slurm_3771.out |
| server | phase4_probe | 3772 |  | 2 | 450 | empty,log_present | server_snapshot/runs/phase4_probe/qwen7_probe_clean_all_vquick/logs/slurm_3772.err \| server_snapshot/runs/phase4_probe/qwen7_probe_clean_all_vquick/logs/slurm_3772.out |
| server | phase4_probe | 3773 |  | 2 | 438 | empty,log_present | server_snapshot/runs/phase4_probe/qwen7_probe_all_vquick/logs/slurm_3773.err \| server_snapshot/runs/phase4_probe/qwen7_probe_all_vquick/logs/slurm_3773.out |
| server | phase4_probe | 3774 |  | 2 | 567 | empty,log_present | server_snapshot/runs/phase4_probe/qwen7_probe_controls_quick/logs/slurm_3774.err \| server_snapshot/runs/phase4_probe/qwen7_probe_controls_quick/logs/slurm_3774.out |
| server | phase4_probe | 3822 |  | 2 | 1378 | empty,log_present | server_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_3822.err \| server_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_3822.out |
| server | phase4_probe | 3823 |  | 2 | 4260 | empty,log_present | server_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_gate5_3823.err \| server_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_gate5_3823.out |
| server | phase4_probe | 3824 |  | 2 | 3147 | empty,log_present | server_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_gate6_3824.err \| server_snapshot/runs/phase4_probe/qwen7_probe_controls_t1/logs/slurm_gate6_3824.out |
| server | t3_cross_family | 3895 | 0,1,2,3,4,5,6,7 | 16 | 54829 | empty,log_present | server_snapshot/runs/t3_cross_family/logs/slurm_phi3_3895_0.err \| server_snapshot/runs/t3_cross_family/logs/slurm_phi3_3895_0.out \| server_snapshot/runs/t3_cross_family/logs/slurm_phi3_3895_1.err \| server_snapshot/runs/t3_cross_family/logs/slurm_phi3_3895_1.out |
