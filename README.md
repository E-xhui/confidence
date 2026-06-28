# What does confidence in RAG actually measure?
Existing RAG calibration methods evaluate confidence against answer correctness, yet confidence is primarily driven by retrieval topicality and answer support rather than evidence veracity.

# RAG Calibration Experiments
## 目录结构

```
ragcalib/       # 核心模块（指标计算、数据加载、模型推理）
scripts/        # 实验脚本（数据构造、模型采样、分析）
runs/           # 实验输出（预测文件、分析报告）
figures/        # 生成的图表
```

## 实验进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 实验定义、数据集选择（PopQA + ConflictQA） | 完成 |
| Phase 1 | 上下文构造（sup/mis/irr/same_entity_irr） | 完成 |
| Phase 2 | 五条件采样（Qwen-2.5-7B/14B, k=5, T=0.7） | 完成 |
| Phase 2.5 | 辅助分析（closed_known、rho 分解、reader 质量） | 完成 |
| Phase 3 | H3 校准验证、条件化校准（T2） | 完成 |
| T3 | 跨家族复现（Phi-3-mini） | 完成 |
