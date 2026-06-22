# Task: SFA v7 随机投影正交化修复

## Objective
修复 SFA v7 enhancement 与 attention output 不正交的问题 (cosine=0.65 → 目标 <0.1)

## Key Reasoning
1. **v1-v3 迭代**: 尝试了多种正交化方案，均失败
   - v1: 原始 SFA — cosine=0.65
   - v2: Difference-based — cosine~1.0
   - v3: Manual forward — position embedding 问题

2. **v4 成功**: 随机投影 + Gram-Schmidt 正交化
   - Cosine = -0.042 ~ 0.007 (完美正交)
   - 方法: 从 enhancement 中减去沿 attention 方向的投影，再混合随机投影

3. **PPL 仍无改善**: 所有 α 值均导致 PPL 恶化
   - 根因: Enhancement 幅度被 alpha 压缩到接近 0
   - 或: 正交通道本身不能提升 PPL

## Conclusions
- ✅ **正交性修复成功** — 随机投影方案有效
- ❌ **PPL 改善未实现** — 需要调整 enhancement 幅度或注入策略
- 📝 **论文需要更新** — 实测数据与理论值存在差距

## Files
- `test_sfa_random_projection_v4.py` - 最终修复版测试
- `docs/sfa_alpha01_correctness_test_report.md` - 测试报告 (已更新)

## Commits
- `84c32d6` - SFA v7 α=0.1 正确性测试 (发现正交性问题)
- `f191c76` - SFA v7 随机投影正交化 — v4 测试完成
