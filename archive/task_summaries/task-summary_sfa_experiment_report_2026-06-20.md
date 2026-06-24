# SFA 实验结果汇总报告 — 任务总结

**时间：** 2026-06-20 15:29 GMT+8  
**角色：** 数据整合专家（subagent）

## 目标
整合所有 SFA benchmark 数据，生成对比图表描述和技术报告，作为开源文章的核心数据支撑。

## 数据来源（共 11 个 JSON/MD 文件）
1. `benchmark_results.json` — 正确性(Cosine)、速度、内存压缩比、PPL
2. `diverse_dataset_test_results.json` — 8 类文本 200 条测试
3. `qwen25_7b_real_test_results.json` — 7B 真实测试结果
4. `qwen25_7b_ultra_fast_results.json` — 7B 快速测试结果
5. `qwen25_7b_baseline_result.json` — 7B 基线 PPL
6. `distillation_results_fixed.json` — 蒸馏实验结果
7. `metal_gpu_optimization_report.json` — Metal GPU 性能
8. `extended_experiment_results.json` — FLOPs/延迟/失败案例/超参鲁棒性
9. `HIGH_ENERGY_PROGRESS_2026-06-19.md` — 高能模式推进报告
10. `测试数据汇总_v3.md` — 数据标注与修正记录
11. `final_expert_mode_report.md` — 全专家模式最终测试报告

## 关键发现
- SFA 在 0.5B 长文本上 PPL 改善 -10.02% (α=2.0)
- WikiText-2 风格合成数据全局改善 -2.53%
- 8 类多样化文本全局改善 -4.91%
- 7B 模型最佳改善 -2.56% (α=10.0)
- 理论压缩比 3,972× (7B/64K)，远超论文声称 248×
- Metal GPU Prefill 加速 2.66×，Generate 加速 1.41×
- Cosine Similarity > 0.9999，证明 SFA 为正交增强

## 产出文件
- `SFA_Experiment_Results_Summary.md` — 完整技术报告 (299 行, 11.5 KB)
- `task-summary_sfa_experiment_report_2026-06-20.md` — 本文件
