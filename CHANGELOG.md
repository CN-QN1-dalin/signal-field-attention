# 太初五岳变更日志

所有重要更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-06-24

### 添加
- **Dalin L** — Agent 原生编程语言 (MVP 0.1.0)
  - 词法分析器（25 个关键字）
  - 语法分析器（递归下降）
  - 类型系统（Hindley-Milner）
  - 代码生成（LLVM IR）
  - 标准库（collections, io, string, option, result, iterator）
  - CLI + REPL
  - 中文支持
- **SFA v7** — 信号场注意力
  - 三通道 KV 压缩 (RingBuffer, EMA Field, Semantic Pool)
  - O(1) 内存压缩，理论压缩比 248x ~ 3971x
  - O(1) Decode 延迟，恒定 0.52ms/token
  - Q4_0 量化，内存节省 65%，吞吐量提升 150%
  - 正交信息通道验证 (cosine similarity ~0.002)
- **Dalin ISFE** — 意图理解引擎
  - 核心算法 (Rust/Python 原型)
  - Pre-Intent 预意图引擎
  - 意图正交基 (IOB)
- **基准测试套件** — Qwen2.5-0.5B/7B 验证
- **技术报告** — SFA v7 完整规范
- **文档** — 62 份技术文档

### 修复
- SFA v7 随机投影正交性修复 (v4)
- llama.cpp 集成 P0 错误 (tensor shape, layer indexing)
- ggml API 兼容性修复

### 变更
- 项目名称从 "Taichu" / "QN1" 改为 "Dalin Soma"
- 统一 README 和 CHANGELOG 格式

---

[1.0.0]: https://github.com/CN-QN1-dalin/signal-field-attention/releases/tag/v1.0.0
