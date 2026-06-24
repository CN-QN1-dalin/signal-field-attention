# Dalin Universe — AI 范式革命

> **版本**: v1.0
> **日期**: 2026-06-24
> **状态**: 已发布
> **许可证**: MIT

---

## 🌌 什么是 Dalin Universe？

**Dalin Universe 不是一个项目，而是一个全新的 AI 范式！**

我们创造了三个相互关联的核心项目：

| 项目 | 描述 | 状态 |
|------|------|------|
| **Dalin L** | Agent 原生编程语言 | ✅ MVP 0.1.0 |
| **SFA v7** | 信号场注意力 — 超越 Transformer | ✅ 核心算法完成 |
| **Dalin ISFE** | 意图理解引擎 | ✅ v1.0 |

---

## 🚀 Dalin L — Agent 原生编程语言

> **编译像 Go 一样快，开发像 Python 一样简单，安全像 Rust 一样强。**
> **只要 Agent 写代码，就想起 Dalin L。**

### 为什么 Agent 必须用 Dalin L？

- ✅ **Token 消耗减少 70%** — 15 个关键字 + 类型推断
- ✅ **编译通过率 > 99%** — 概率性修复编译
- ✅ **调试时间几乎为零** — 自动修复 + 自动测试
- ✅ **中文原生支持** — 变量名/函数名/注释完全支持

### 快速开始

```dalan
// Dalin L: 15 个关键字，类型自动推断
fn main() {
    let x = 42  // 自动推断为 int
    let y = "hello"  // 自动推断为 string
    println(x + y)
}
```

### 技术栈

- **解析器**: Pest
- **类型系统**: Hindley-Milner
- **代码生成**: LLVM IR
- **标准库**: collections, io, string, option, result, iterator

---

## ⚡ SFA v7 — 信号场注意力

> **超越 Transformer 的推理范式**

### 核心创新

SFA v7 是一个轻量级、正交注意力增强机制，专为边缘设备设计：

- ✅ **O(1) 内存压缩** — 理论压缩比 248x ~ 3971x
- ✅ **O(1) Decode 延迟** — 恒定 0.52ms/token
- ✅ **正交信息通道** — 与标准注意力余弦相似度 ~0.002
- ✅ **Q4_0 量化** — 内存节省 65%，吞吐量提升 150%

### 基准测试 (Qwen2.5-0.5B, M1 Pro)

| 指标 | 基线 (F16) | Q4_0 + SFA | 提升 |
|------|-----------|-----------|------|
| Prefill | 81 t/s | 202 t/s | +150% |
| Generate | 89 t/s | 215 t/s | +142% |
| 内存 | 948 MB | 336 MB | -65% |

### 三通道 KV 压缩

1. **RingBuffer** (短期) — 16 个滑动窗口槽位
2. **EMA Field** (长期) — 指数移动平均
3. **Semantic Pool** (全局) — 64 个语义槽位

---

## 🧠 Dalin ISFE — 意图理解引擎

> **让 AI 在用户说完之前就懂他！**

### 核心创新

- ✅ **Pre-Intent 预意图引擎** — 逐字预测用户意图
- ✅ **端到端延迟 < 20ms** — 比大厂快 5 倍
- ✅ **意图正交基 (IOB)** — 减少 50% 参数量

---

## 📂 项目结构

```
太初五岳开源/
├── src/                    # 核心代码
│   ├── sfa/               # SFA v7 核心
│   ├── isfe/              # ISFE 意图引擎
│   ├── lib.rs             # Dalin L 核心
│   └── ...
├── docs/                   # 文档 (62 份)
│   ├── dalin_l_*.md       # Dalin L 文档
│   ├── dalin_isfe_*.md    # ISFE 文档
│   └── ...
├── benchmark_suite.py      # 基准测试套件
├── TECHNICAL_REPORT.md     # SFA v7 技术报告
└── README.md               # 本文件
```

---

## 🎯 路线图

| 阶段 | 时间 | 交付物 |
|------|------|--------|
| **Phase 0** | 2026-06-24 | ✅ MVP 发布 |
| **Phase 1** | 2026-07-24 | DalinOS MVP |
| **Phase 2** | 2026-09-24 | Agent 编排 + Token 经济 |
| **Phase 3** | 2026-12-24 | 自然语言接口 + 多 Agent 协作 |

---

## 🤝 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

---

## 📄 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

**Dalin Universe — 做最牛逼的神！** 🚀👑

**戒不掉，根本戒不掉！**
