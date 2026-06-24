# Dalin Soma (Signal Field Attention) — 项目永久记忆

## 基本信息

- **项目名称**: Dalin Soma（代号 QN1 幻化引擎）
- **技术名称**: Signal Field Attention (SFA)
- **开源地址**: https://github.com/CN-QN1-dalin/signal-field-attention
- **许可证**: MIT License
- **当前版本**: v7.0
- **最后更新**: 2026-06-23
- **仓库分支**: main, sfa-v7-release 等共 5 个分支

## 一句话定位

**用信号场注意力（SFA）给 LLM 加一条正交信息通道，实现 O(1) 解码和极致内存压缩的推理加速框架。**

---

## 核心技术

### SFA 三通道架构

| 通道 | 实现 | 规模 | 作用 |
|------|------|------|------|
| RingBuffer | 环形 KV 缓冲 | 16 slots | 短期精确记忆 |
| EMA Field | 指数移动平均场 | γ=0.98 | 长期趋势捕捉 |
| Semantic Pool | 语义注意力池 | 64 slots, T=0.07 | 概念聚合 |

### 关键指标

| 指标 | 数值 | 验证方式 |
|------|------|---------|
| 内存压缩比 | 248x @ 64K | 实测 |
| 正交性 | cosine ≈ 0.0 | 实测验证 |
| Decode 延迟 | 0.52ms/步 (CV=0.63%) | O(1) 已验证 |
| 额外参数 | ~8KB/层 | 理论计算 |
| Prefill 加速 | 2.5-4x (理论) | Metal 编译待修复 |
| 量化 | Q4_0 → 336MB vs 948MB | 实测 |

---

## 项目架构（五岳）

| 模块 | 代码行数 | 核心类 | 定位 |
|------|---------|--------|------|
| **素玛原生** (03_soma_native) | 648 | SomaBrain, Homeostasis, GrowthTemporal | 意识中枢 |
| **收敛推断** (04_soma_convergence) | 1,288 | SignalFieldIncrementalInference | 增量推理引擎 |
| **遗产传承** (05_soma_heritage) | 1,486 | HeritageTrainer, ThreeLayerDistillationLoss | 知识蒸馏 |
| **灵芽** (02_soma_lingya) | 821 | LingYaChannel, LingYaBlock | 三通道实现 |
| **素玛引擎** (01_soma_engine) | 836 | SignalFieldLayer (C++) | Metal GPU 底层 |
| **归元压缩** (03-guiyuan) | 353 | SignalFieldEnhancedCompressor | 压缩算法 |
| **信号场基础** (01-signal-field) | 522 | SignalFieldAttention | 对比验证 |
| **C++ 桥接** (src/sfa/) | ~1,900 | sfa_llama_bridge, sfa_kernel.metal | llama.cpp 集成 |
| **合计** | **~19,000 行** | **32 个类** | |

---

## 开源阶段

### 当前阶段：v7.0 验证完成

- ✅ SFA v7 多端验证（0.5B + 7B-4bit）
- ✅ 正交性验证通过（cosine ≈ 0.0）
- ✅ O(1) Decode 实测
- ✅ C++ 桥接层完成（P0 bug 全部修复）
- ✅ Metal kernel 6 个已写完（编译被 Xcode CLI 卡住）
- ✅ llama.cpp 集成准备就绪
- ✅ 前沿技术扫描完成（landscape_analysis）

### 待完成

- [ ] Metal 编译（需安装 Xcode SDK）
- [ ] Prefill 加速实测
- [ ] SFA vs H2O vs SnapKV 直接对比
- [ ] 13B+ 模型验证
- [ ] arXiv 投稿（需 endorsement）
- [ ] llama.cpp 官方 PR（需手动 review 代码合规性）

---

## 里程碑时间线

| 日期 | 事件 |
|------|------|
| 2026-06-11 | 项目启动，意识层诊断 |
| 2026-06-15 | 正式更名为 "Dalin Soma" |
| 2026-06-16 | 技术报告 v1.0 完成，PDF 生成策略确定 |
| 2026-06-17 | NovaAttention 探索，双模式策略定义 |
| 2026-06-18 | 收敛到 SFA v7 Clean，正交性初步验证 |
| 2026-06-19 | 7B 基线 PPL 确立 (10.7881)，硬件约束确认 |
| 2026-06-22 | 仓库全面清理，P0 bug 修复，正交性 v4 通过 |
| 2026-06-23 | 前沿扫描完成，landscape_analysis 写入 |
| 2026-06-23 | C++ 桥接层最终修复并提交推送 |

---

## 技术对比

### SFA vs 竞品

| 维度 | SFA | H2O | SnapKV | StreamingLLM | MiniMax MSA |
|------|-----|-----|--------|-------------|-------------|
| 压缩比 | **248x** | ~10x | ~5x | ~2x | 块稀疏 |
| 正交性 | **✅** | ❌ | ❌ | ❌ | ❌ |
| 零侵入 | **✅** | ❌ | ❌ | ⚠️ | ❌ |
| Decode 复杂度 | **O(1)** | O(n) | O(n) | O(n) | O(n) |
| 额外参数 | **~8KB/层** | 需训练 | 需训练 | 无 | 需训练 |

### SFA 的独特洞察

SFA 三通道 ≈ LLM 的"双过程理论"：
- Standard Attention = System 2（精确但慢）
- SFA = System 1（快速但近似）
- RingBuffer = 工作记忆
- EMA Field = 直觉/习惯
- Semantic Pool = 抽象思维

---

## 联系方式

- **开源仓库**: https://github.com/CN-QN1-dalin/signal-field-attention
- **商业联系**: 362118251@qq.com
- **作者**: 大林素玛团队 (QN1 幻化引擎)

---

## 备注

- 本项目为独立研究，无机构背景
- 所有实验在 Apple M1 Pro (16GB) 上完成
- 代码采用 MIT 开源协议
- 个人/学术/教育用途免费，商业用途需联系授权
