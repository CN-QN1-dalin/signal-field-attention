# Soma Penta-Peaks (Dalin Soma)

## 全链路AI基础设施

**零Transformer依赖 · 从架构到推理到训练 · 5大核心模块**

---

## 一句话

Soma用信号场替代了统治AI领域7年的Transformer，从神经网络架构设计到底层推理加速，真正实现从零起步的AI计算革新。

---

## 五岳全景

| 五岳 | 模块 | 定位 | 核心指标 |
|------|------|------|---------|
| 🏔️ 东岳 | Soma Engine (Signal Field) | 推理加速 | 4.16x加速，248x内存压缩 |
| 🏔️ 南岳 | Soma LingYa (LingYa) | 参数高效微调 | 比LoRA省51%参数，推理零开销 |
| 🏔️ 西岳 | Soma Native | 全新神经网络 | O(k·n)复杂度，28层7B验证 |
| 🏔️ 北岳 | Soma Convergence (Convergence) | O(1)增量推理 | 0.00%误差，恒定0.52ms/步 |
| 🏔️ 中岳 | Soma Heritage (Distillation) | 蒸馏训练框架 | 深层PPL改善10.57% |

---

## 快速开始

```bash
# 环境
pip install mlx transformers

# 每个岳都是独立可运行的模块
# 见各子目录
```

---

## 测试环境

- Apple MacBook Pro M1 Pro, 16GB RAM
- MLX 0.31.2
- Python 3.14
- 测试模型：Qwen2.5-7B / Qwen2.5-14B / Qwen2.5-0.5B

---

## 核心数据汇总

| 指标 | 东岳(引擎) | 北岳(归元) | 西岳(架构) | 中岳(蒸馏) | 南岳(灵芽) |
|------|-----------|-----------|-----------|-----------|-----------|
| 加速比 | 4.16x | 4.16x | - | - | - |
| 内存压缩 | 248x | 248x | 4000x(64K) | - | - |
| 误差 | 0.00% | 0.00% | TBD | - | - |
| 参数效率 | 8.1KB | - | 25%节省 | - | 比LoRA省51% |
| 复杂度 | O(k·n) | O(1) | O(k·n) | - | - |
| 推理开销 | - | - | - | - | 零开销 |

---

## 项目结构

```
dalin-soma/
├── README.md                  ← 你在这里
├── LICENSE                    ← MIT
├── 01_soma_engine/            ← 🏔️ 东岳：信号场推理加速
│   ├── 源代码.py
│   ├── 测试代码.py
│   ├── 测试对比.md
│   └── 使用说明.md
├── 02_soma_lingya/            ← 🏔️ 南岳：参数高效微调
│   ├── 源代码.py
│   ├── 测试代码.py
│   ├── 测试对比.md
│   └── 使用说明.md
├── 03_soma_native/            ← 🏔️ 西岳：原生架构
│   ├── 源代码.py
│   ├── 测试代码.py
│   ├── 测试对比.md
│   └── 使用说明.md
├── 04_soma_convergence/       ← 🏔️ 北岳：O(1)推理
│   ├── 源代码.py
│   ├── 测试代码.py
│   ├── 测试对比.md
│   └── 使用说明.md
├── 05_soma_heritage/          ← 🏔️ 中岳：蒸馏训练
│   ├── 源代码.py
│   ├── 测试代码.py
│   ├── 测试对比.md
│   └── 使用说明.md
├── 06_soma_v7_demo/           ← Soma v7多层端到端
│   ├── Soma v7多层端到端.py
│   ├── Soma v7文本验证.py
│   ├── soma_demo_final2.py
│   └── benchmark_14b.py
└── LICENSE
```

---

## 与 SFA 的关系

> **SFA (Signal Field Attention)** 是Soma Engine的单点突破实验，
> 侧重于注意力机制的替换验证和学术传播。
> 
> **Dalin Soma** 是完整的AI基础设施，包含5个互补的核心模块，
> 从架构设计到推理加速到参数微调到蒸馏训练，形成闭环。
> 
> SFA 是Dalin Soma中"东岳"和"西岳"的技术投影。
> Dalin Soma是完整的Soma。

---

## 开源许可

MIT License

---

*Dalin Soma © 2026 · 从零开始的AI基础设施*
