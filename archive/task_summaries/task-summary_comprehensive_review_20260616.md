# Task Summary: Comprehensive Peer Review of Three Soma Papers

## Objective
Review Soma Heritage, Soma Native, and Soma LingYa papers from the perspective of top-tier conference reviewers and industry experts. Identify all critical flaws, major issues, and minor suggestions.

## Key Reasoning & Methodology

1. **Cross-referenced papers with actual code**: Traced every experimental claim back to `扩展实验.py`, `benchmark_suite.py`, and `soma_engine.py` to verify whether data is real or simulated.
2. **Verified mathematical proofs**: Checked Theorem 1 (Heritage), Lemma 1 (Heritage), Proposition 2 (Heritage), Theorem 1 (Native), Lemma 1 (LingYa) for correctness.
3. **Validated numerical calculations**: Checked memory estimates (462KB vs 1.8MB), FLOPs ratios, latency figures.
4. **Compared with state-of-the-art**: Evaluated whether baselines (LoRA, QLoRA, Mamba-2, FlashAttention, RetNet) are adequately covered.

## Critical Findings

### 🔴 CRITICAL (Will cause immediate rejection)

1. **大量数据是模拟/推算的，非真实实验**: 10项补充实验中，至少6项使用硬编码值或虚构损失函数（如GradNorm实验用 `losses = [0.5 + step*0.001, ...]` 代替真实蒸馏损失）
2. **预填充实测慢10×，论文暗示更快**: `benchmark_results.json` 显示 Soma Prefill 在 seq=1024 时为 164.54ms vs Standard 6.70ms（慢24.5×），但论文未区分 MLX 原型与 C++/Metal 部署
3. **内存计算错误**: 论文用 0.5B 配置（dims=896）的 115.5KB 推算 7B 为 462KB，实际应为 ~1.8MB
4. **Layer 23 超越教师 -10.57% 无合理解释**: SFA 是近似，不可能比精确注意力更好，可能是隐式正则化效应但未论证
5. **下游任务准确率是推算的**: LAMBADA +1.76% 基于经验回归系数，无任何实际评测
6. **未与 QLoRA/DoRA/AdaLoRA 对比**: PEFT 领域 2026 年不对比这些方法是不可接受的

### 🟡 MAJOR (Should fix before submission)

1. Heritage 定理 1 不等号方向错误（k ≤ 应为 k ≥）
2. Heritage Lemma 1 只是代数恒等式，不是真正的引理
3. Native Homeostasis 和 GrowthTemporal 无任何实验验证
4. Native 推理延迟在所有序列长度下恒定 11.8μs 不合理
5. LingYa 与 LoRA 对比不公平（冻结一个矩阵 vs 训练两个矩阵）
6. 层重要性评分使用 mock 数据，无实际计算

### 🟢 MINOR

1. 论文中使用 emoji（✅🎉）不符合学术规范
2. "Soma Labs" 机构可信度不足，建议标注独立研究者
3. 参考文献不完整（缺少 Linformer, Performer, BigBird, Mamba-2）
4. 英文摘要为翻译体，需润色

## Recommendations

- **投稿策略**: 三篇论文当前状态更适合先发 arXiv 积累反馈，完成 P0+P1 修改后再投顶会
- **最紧迫**: 区分模拟数据与真实数据，补充至少一个完整训练实验
- **架构定位**: 明确标注 "Python/MLX prototype" vs "C++/Metal deployment target"

## Files Created

- `顶会审查报告.md` (8.4KB) - 完整审查报告
