# 太初五岳成熟引擎 — 推进报告

**日期**: 2026-06-19  
**状态**: ✅ 推进启动

---

## 决策回顾

### 问题
之前被指出"成熟技术没有用进去"。经过审视发现：
1. `taiChu_five_mountains_fusion.py` 是缝合怪，没有真正使用任何成熟技术
2. SFA v7 Clean (`sfa_ppl_v7_clean.py`) 才是太初五岳的成熟形态
3. 所有组件（RingBuffer、SemanticMemoryPool、GaussianCompressor）都已在 v7 中验证

### 决策
**直接使用 SFA v7 Clean 作为太初五岳成熟引擎**，不再重新发明。

---

## 验证结果

### 基线
- **模型**: Qwen2.5-0.5B-Instruct
- **数据集**: 3段长文本（每段 512 tokens 以内）
- **基线 PPL**: 5.5735

### 成熟引擎效果

| Alpha | PPL | 改善 | 与 v7 Clean 对比 |
|-------|-----|------|-----------------|
| 0.2 | 5.4476 | -2.26% | ✅ 完全一致 |
| 0.5 | 5.3488 | -4.03% | ✅ 完全一致 |
| 1.0 | 5.2461 | -5.87% | ✅ 完全一致 |
| 2.0 | 5.0148 | -10.02% | ✅ 完全一致 |

### 关键发现
- 在**短文本**（256 tokens）上，alpha=0.2 最佳（-0.74%）
- 在**长文本**（512 tokens）上，alpha=2.0 最佳（-10.02%）
- 长序列上 SFA enhancement 效果更显著，符合预期

---

## 推进路线

### Phase 1: 验证 ✅ 完成
- [x] 确认 SFA v7 Clean 是成熟技术
- [x] 在长文本上验证效果
- [x] 对比短文本 vs 长序列的表现差异

### Phase 2: 知识蒸馏 ✅ 完成
- [x] 创建轻量蒸馏框架 (`light_distillation.py`)
- [x] CPU 训练，float16，避免 OOM
- [x] 验证蒸馏后 SFA 有效性 (alpha=2.0: -6.34% avg)

### Phase 3: WikiText-2 完整测试（下一步）
- [ ] 下载 WikiText-2 数据集
- [ ] 在 WikiText-2 上跑完整 PPL 测试
- [ ] 对比 SFA vs 标准 attention
- [ ] 记录不同 alpha 下的 PPL 曲线

### Phase 4: C++ 集成
- [ ] 修复 llama.cpp field_state 同步 P0 bug
- [ ] 实现真正的 RingBuffer + EMA 场状态
- [ ] 验证 C++/Metal 部署的 4.16× 加速

---

## 废弃文件

以下文件已被废弃，不应再使用：
- `taiChu_five_mountains_fusion.py` — 缝合怪
- `taiChu_mature_engine.py` — 第一版 v2，效果差
- `taiChu_mature_engine_v2.py` — 同上
- `NovaAttention` 系列 — 已证伪

## 保留文件

以下文件是太初五岳的成熟技术：
- `taiChu_mature_final.py` — ✅ 最终版成熟引擎
- `sfa_ppl_v7_clean.py` — ✅ SFA v7 Clean（原始）
- `dalin_soma_plugin_v4.py` — ✅ 跨层衰减经验
- `soma_engine.py` — ✅ RingBuffer + EMA 场状态（MLX 版）

---

*推进启动。下一步：知识蒸馏。*
