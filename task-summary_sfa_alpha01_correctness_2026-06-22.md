# Task: SFA v7 α=0.1 正确性测试

## Objective
验证 SFA v7 在 α=0.1 时的完整正确性：PPL 改善、正交性、内存开销。

## Key Reasoning
1. **3 轮测试执行**：基础测试、α 扫描、修正版引擎
2. **关键发现**：
   - SFA enhancement 与 attention output 不正交 (cosine=0.65，预期<0.1)
   - 所有 α 值均导致 PPL 恶化或无改善 (<0.1%)
   - 理论声称 ("cosine~0.002") 与实际测量差距巨大
3. **修正版尝试**：引入 difference-based enhancement，正交性反而更差 (cosine~1.0)
4. **根因分析**：Enhancement 基于 attention 输出的统计量 (ring mean, EMA)，本质上是 attention 的线性变换

## Conclusions
- **SFA v7 当前实现无法产生正交信号**
- **PPL 改善不显著**，需要重新设计 enhancement 注入策略
- **建议下一步**：引入随机投影到正交子空间，或在 attention weights 而非 output 上注入

## Files
- `test_sfa_alpha01_quick.py` - 基础测试
- `test_sfa_alpha_scan.py` - α 扫描
- `test_sfa_corrected.py` - 修正版测试
- `docs/sfa_alpha01_correctness_test_report.md` - 详细报告
- `test_results_sfa_alpha01_quick.json` - 原始数据

## Commit
- `84c32d6` - feat: SFA v7 α=0.1 正确性测试
