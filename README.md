# Dalin L

> **Agent 原生编程语言**
> 
> 编译像 Go 一样快，开发像 Python 一样简单，安全像 Rust 一样强。
> 
> 只要 Agent 写代码，就想起 Dalin L。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-v1.70+-orange.svg)](https://www.rust-lang.org/)
[![Build Status](https://github.com/CN-QN1-dalin/dalin-l/workflows/CI/badge.svg)](https://github.com/CN-QN1-dalin/dalin-l/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.dalin-lang.org)

---

## 📖 简介

Dalin L 是一个为 AI Agent 设计的编程语言。它结合了 C 的速度、Rust 的安全、Python 的简洁和 Go 的并发，消除了它们的每一个痛点。

### 核心优势

- **零学习成本** — 25 个关键字，类型自动推断，中文变量名/函数名/注释
- **零碎片化工具链** — `dalan build` 编译 + 格式化 + 类型检查
- **零不确定性** — HM 类型推断，编译期确定，中文错误信息
- **Agent 原生设计** — 上下文感知编译，自我修复编译，自动测试生成，统一 FFI，代码即对话

### 性能对比

| 指标 | Dalan L | Go | Python | Rust | C |
|------|---------|-----|--------|------|---|
| 编译速度 | ⚡ 快 | ⚡ 快 | 🐢 慢 | 🐢 慢 | ⚡ 快 |
| 运行速度 | ⚡ 快 | ⚡ 快 | 🐢 慢 | ⚡ 快 | ⚡ 快 |
| 内存占用 | 💧 少 | 💧 少 | 💧💧 多 | 💧 少 | 💧 少 |
| 安全性 | 🛡️ 高 | 🛡️ 中 | 🛡️ 低 | 🛡️ 高 | 🛡️ 低 |
| 易用性 | 📝 简单 | 📝 简单 | 📝 很简单 | 📝 中等 | 📝 中等 |

---

## 🚀 快速开始

### 安装

```bash
# 从源码编译
git clone https://github.com/CN-QN1-dalin/dalin-l.git
cd dalan-l
cargo build --release
sudo cp target/release/dalin /usr/local/bin/

# 或使用 Homebrew（macOS）
brew install CN-QN1-dalin/dalan/dalan
```

### Hello World

```dalan
// 最简单的 Dalan L 程序
fn main() {
    println("Hello, World!")
}
```

### 运行

```bash
# 编译并运行
dalan run hello.dalan

# 只编译
dalan build hello.dalan

# 交互式 REPL
dalan repl
```

### 类型推断

```dalan
// 自动推断类型
let 名字 = "大林"        // string
let 年龄 = 25            // int
let 身高 = 1.75          // float
let 已婚 = false         // bool

// 中文变量名完全支持
let 用户名 = "agent_001"
let 密码 = "secret"
```

### 函数

```dalan
// 函数定义
fn 问候(名字: string) -> string {
    return "你好, " + 名字 + "!"
}

// 函数调用
let 消息 = 问候("大林")
println(消息)
```

### 管道操作

```dalan
// 管道操作符
let 结果 = "hello world"
    |> 转大写
    |> 分割(" ")
    |> 长度()
```

---

## 📚 文档

- [语言参考](docs/language-reference.md)
- [标准库](docs/stdlib.md)
- [Agent 特性](docs/agent-features.md)
- [贡献指南](CONTRIBUTING.md)

---

## 🛠️ 特性

### 核心特性

- ✅ 25 个关键字（核心 15 + 扩展 10）
- ✅ 类型自动推断
- ✅ 中文变量名/函数名/注释
- ✅ 递归下降解析
- ✅ Hindley-Milner 类型推断
- ✅ LLVM IR 代码生成
- ✅ 原生二进制编译

### Agent 特性

- ✅ 上下文感知编译
- ✅ 自我修复编译
- ✅ 自动测试生成
- ✅ 统一 FFI
- ✅ 代码即对话

### 工具链

- ✅ `dalan build` — 编译 + 格式化 + 类型检查
- ✅ `dalan run` — 运行
- ✅ `dalan test` — 测试
- ✅ `dalan repl` — 交互式开发
- ✅ `dalan fmt` — 格式化
- ✅ `dalan docs` — 文档生成

---

## 🤝 贡献

欢迎贡献！请阅读 [贡献指南](CONTRIBUTING.md)。

### 贡献步骤

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 审查流程

所有贡献必须通过五方辩论共识引擎审查：

- **Alpha** — 技术正确性
- **Beta** — Agent 友好度
- **豆包** — 中文支持
- **GPT** — 理论正确性
- **混元** — 工程实现
- **元宝** — 用户体验

---

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。

---

## 🌟 星标历史

[![Star History Chart](https://api.star-history.com/svg?repos=CN-QN1-dalin/dalin-l&type=Date)](https://star-history.com/#CN-QN1-dalin/dalan-l&Date)

---

## 📞 联系方式

- **GitHub**: [CN-QN1-dalin/dalin-l](https://github.com/CN-QN1-dalin/dalin-l)
- **Discord**: [https://discord.gg/dalan-l](https://discord.gg/dalan-l)
- **微信**: 扫码加入
- **邮箱**: contact@dalan-lang.org

---

*Dalan L — 由太初五岳团队构建。*

**编译像 Go 一样快，开发像 Python 一样简单，安全像 Rust 一样强。**

**只要 Agent 写代码，就想起 Dalin L。**
