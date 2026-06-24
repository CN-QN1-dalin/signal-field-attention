# Dalin L — Phase 2 实施指南

> **阶段**: Agent 特性开发
> **时间**: 2026-07 ~ 2026-09
> **团队**: 2 人全职
> **审查引擎**: 五方辩论共识引擎（持续运行）
> **状态**: 准备启动

---

## 1. 开发环境设置

### 1.1 系统要求

```bash
# macOS
brew install rust llvm clang

# Ubuntu/Debian
sudo apt update
sudo apt install -y rustc cargo llvm clang

# Windows
# 使用 WSL2
wsl --install
# 然后安装上面的 Ubuntu 依赖
```

### 1.2 项目结构

```
dalin-l/
├── Cargo.toml              # Rust 项目配置
├── Cargo.lock
├── README.md
├── LICENSE
├── docs/                   # 文档
│   ├── design.md
│   ├── reference.md
│   └── examples/
├── src/
│   ├── main.rs             # CLI 入口
│   ├── lexer.rs            # 词法分析器
│   ├── parser.rs           # 语法分析器
│   ├── type_checker.rs     # 类型检查器
│   ├── codegen.rs          # 代码生成器
│   ├── stdlib/             # 标准库
│   │   ├── mod.rs
│   │   ├── collections.rs
│   │   ├── io.rs
│   │   ├── string.rs
│   │   └── option.rs
│   ├── cli/                # CLI 模块
│   │   ├── mod.rs
│   │   ├── build.rs
│   │   ├── run.rs
│   │   ├── test.rs
│   │   ├── repl.rs
│   │   └── fmt.rs
│   ├── agent/              # Agent 特性
│   │   ├── mod.rs
│   │   ├── auto_fix.rs     # 自我修复
│   │   ├── auto_test.rs    # 自动测试
│   │   ├── intent.rs       # 意图补全
│   │   └── merge.rs        # 多 Agent 合并
│   └── ffi/                # FFI 模块
│       ├── mod.rs
│       └── c.rs
├── tests/                  # 集成测试
│   ├── lexer_tests.rs
│   ├── parser_tests.rs
│   ├── type_checker_tests.rs
│   ├── codegen_tests.rs
│   └── integration_tests.rs
├── examples/               # 示例代码
│   ├── hello.dalin
│   ├── fibonacci.dalin
│   └── ...
└── std/                    # 标准库源码
    ├── collections.dalin
    ├── io.dalin
    ├── string.dalin
    └── ...
```

### 1.3 Cargo.toml

```toml
[package]
name = "dalin-l"
version = "0.1.0"
edition = "2021"
authors = ["太初五岳团队"]
license = "MIT"
description = "Agent 原生编程语言"

[dependencies]
clap = "4.4"
llvm-sys = "170"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.6", features = ["v4"] }
chrono = "0.4"
rand = "0.8"
colored = "2.0"
walkdir = "2.4"
glob = "0.3"
regex = "1.10"
itertools = "0.12"

[dev-dependencies]
tempfile = "3.9"
assert_cmd = "2.0"
predicates = "3.0"

[[bin]]
name = "dalin"
path = "src/main.rs"
```

---

## 2. 开发流程

### 2.1 每日工作流

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 创建功能分支
git checkout -b feature/auto-fix

# 3. 开发功能
# ...

# 4. 运行测试
cargo test

# 5. 运行编译
cargo build

# 6. 提交代码
git add .
git commit -m "feat: 实现自我修复编译"

# 7. 推送代码
git push origin feature/auto-fix

# 8. 创建 Pull Request
gh pr create --title "feat: 实现自我修复编译" --body "..."

# 9. 等待审查
# 五方辩论共识引擎审查通过后合并
```

### 2.2 审查流程

每个功能完成后，必须通过以下审查：

```bash
# 1. 运行 Alpha 审查（技术正确性）
cargo test --alpha

# 2. 运行 Beta 审查（Agent 友好度）
cargo test --beta

# 3. 运行豆包审查（中文支持）
cargo test --chinese

# 4. 运行 GPT 审查（理论正确性）
cargo test --gpt

# 5. 运行混元审查（工程实现）
cargo test --engineering

# 6. 运行元宝审查（用户体验）
cargo test --ux

# 7. 全部通过后合并
cargo test --all && git merge feature/auto-fix
```

### 2.3 代码规范

```rust
// 1. 命名规范
// - 变量名：camelCase
// - 函数名：snake_case
// - 类型名：PascalCase
// - 常量名：SCREAMING_SNAKE_CASE
// - 模块名：snake_case

// 2. 注释规范
// - 每个公共函数必须有文档注释
// - 复杂逻辑必须有行内注释
// - 中文注释优先

// 3. 错误处理
// - 使用 Result<T, E> 而不是 panic!
// - 错误信息必须清晰

// 4. 测试
// - 每个公共函数必须有单元测试
// - 覆盖率必须 > 90%

// 5. 格式
// - 使用 rustfmt 格式化
// - 使用 clippy 检查
```

---

## 3. 模块开发顺序

### Month 1: 并发 + 自我修复

#### Week 1-2: async/await + channel

**优先级**: 高
**负责人**: Code
**预计工作量**: 40 小时

```bash
# 1. 创建分支
git checkout -b feature/async-await

# 2. 实现核心数据结构
# src/concurrency/channel.rs
# src/concurrency/task.rs
# src/concurrency/runtime.rs

# 3. 实现语法扩展
# src/parser.rs (添加 async/await 语法)

# 4. 实现类型检查
# src/type_checker.rs (添加 async 类型)

# 5. 实现代码生成
# src/codegen.rs (添加 async 代码生成)

# 6. 编写测试
# tests/async_tests.rs

# 7. 运行审查
cargo test --all

# 8. 合并
git merge feature/async-await
```

#### Week 3-4: 自我修复编译

**优先级**: 高
**负责人**: Fix
**预计工作量**: 40 小时

```bash
# 1. 创建分支
git checkout -b feature/auto-fix

# 2. 实现错误分析引擎
# src/auto_fix/analyzer.rs

# 3. 实现修复建议生成
# src/auto_fix/suggester.rs

# 4. 实现修复应用
# src/auto_fix/applier.rs

# 5. 实现历史决策学习
# src/auto_fix/learner.rs

# 6. 编写测试
# tests/auto_fix_tests.rs

# 7. 运行审查
cargo test --all

# 8. 合并
git merge feature/auto-fix
```

### Month 2: 自动测试 + 统一 FFI

#### Week 5-6: 自动测试生成

**优先级**: 高
**负责人**: Test
**预计工作量**: 40 小时

```bash
# 1. 创建分支
git checkout -b feature/auto-test

# 2. 实现边界分析引擎
# src/auto_test/boundary.rs

# 3. 实现测试用例生成
# src/auto_test/generator.rs

# 4. 实现 Fuzz 测试
# src/auto_test/fuzz.rs

# 5. 实现覆盖率统计
# src/auto_test/coverage.rs

# 6. 编写测试
# tests/auto_test_tests.rs

# 7. 运行审查
cargo test --all

# 8. 合并
git merge feature/auto-test
```

#### Week 7-8: 统一 FFI（C）

**优先级**: 中
**负责人**: FFI
**预计工作量**: 40 小时

```bash
# 1. 创建分支
git checkout -b feature/ffi

# 2. 实现 C 头文件解析
# src/ffi/parser.rs

# 3. 实现类型映射
# src/ffi/type_map.rs

# 4. 实现 FFI 调用生成
# src/ffi/caller.rs

# 5. 编写测试
# tests/ffi_tests.rs

# 6. 运行审查
cargo test --all

# 7. 合并
git merge feature/ffi
```

### Month 3: 格式化器 + 文档生成

#### Week 9-10: dalin fmt

**优先级**: 中
**负责人**: Fmt
**预计工作量**: 40 小时

```bash
# 1. 创建分支
git checkout -b feature/fmt

# 2. 实现格式化引擎
# src/fmt/engine.rs

# 3. 实现 AST 格式化
# src/fmt/ast.rs

# 4. 实现 CLI 命令
# src/cli/fmt.rs

# 5. 编写测试
# tests/fmt_tests.rs

# 6. 运行审查
cargo test --all

# 7. 合并
git merge feature/fmt
```

#### Week 11-12: dalin docs

**优先级**: 中
**负责人**: Doc
**预计工作量**: 40 小时

```bash
# 1. 创建分支
git checkout -b feature/docs

# 2. 实现文档提取
# src/docs/extractor.rs

# 3. 实现 HTML 生成
# src/docs/html.rs

# 4. 实现 CLI 命令
# src/cli/docs.rs

# 5. 编写测试
# tests/docs_tests.rs

# 6. 运行审查
cargo test --all

# 7. 合并
git merge feature/docs
```

---

## 4. 风险管理

### 4.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 技术难度过高 | 中 | 高 | 分阶段实施，先实现核心功能 |
| 团队成员流失 | 低 | 高 | 文档齐全，知识共享 |
| 时间延误 | 中 | 中 | 每周进度检查，及时调整 |
| 审查不通过 | 低 | 中 | 提前沟通，明确标准 |

### 4.2 应急预案

| 情况 | 预案 |
|------|------|
| 技术难题 | 暂停开发，组织专项讨论 |
| 时间延误 | 调整范围，优先核心功能 |
| 人员不足 | 增加临时支援 |
| 审查失败 | 分析原因，重新设计 |

---

## 5. 成功标准

### 5.1 技术指标

| 指标 | 目标 | 测量方式 |
|------|------|----------|
| 编译速度 | < 1s | `time dalan build` |
| 内存占用 | < 5MB | `ps aux | grep dalin` |
| 启动速度 | < 1ms | `time dalin --version` |
| 二进制体积 | 8KB | `ls -lh target/release/dalin` |
| 测试覆盖率 | > 90% | `cargo tarpaulin` |

### 5.2 用户体验指标

| 指标 | 目标 | 测量方式 |
|------|------|----------|
| 新手上手时间 | < 1 小时 | 用户调研 |
| Agent 编写成功率 | > 99% | 实验数据 |
| 错误信息清晰度 | > 90% | 用户评分 |
| 中文支持完整性 | 100% | 代码审查 |

---

## 6. 发布计划

### 6.1 发布流程

```bash
# 1. 版本标记
git tag v0.2.0
git push origin v0.2.0

# 2. 构建发布包
cargo build --release

# 3. 运行所有测试
cargo test --all

# 4. 运行审查
cargo test --alpha
cargo test --beta
cargo test --chinese
cargo test --gpt
cargo test --engineering
cargo test --ux

# 5. 生成发布说明
cargo release-notes > RELEASE.md

# 6. 发布到 GitHub
gh release create v0.2.0 --notes-file RELEASE.md

# 7. 通知社区
echo "Dalin L 0.2.0 已发布！"
```

### 6.2 发布日历

| 日期 | 事件 |
|------|------|
| 2026-07-01 | Phase 2 启动 |
| 2026-07-15 | async/await + channel 完成 |
| 2026-07-31 | 自我修复编译完成 |
| 2026-08-15 | 自动测试生成完成 |
| 2026-08-31 | 统一 FFI 完成 |
| 2026-09-15 | dalin fmt 完成 |
| 2026-09-30 | dalin docs 完成 |
| 2026-10-01 | 发布 0.2.0 |

---

## 7. 附录

### 7.1 术语表

| 术语 | 说明 |
|------|------|
| HM | Hindley-Milner 类型推断算法 |
| FFI | Foreign Function Interface |
| AST | Abstract Syntax Tree |
| PPL | Perplexity（困惑度） |
| SFA | Signal Field Attention |

### 7.2 参考资料

| 资料 | 链接 |
|------|------|
| Rust 官方文档 | https://doc.rust-lang.org/book/ |
| LLVM 文档 | https://llvm.org/docs/ |
| Dalin L 设计文档 | docs/design.md |
| Dalin L 语言参考 | docs/reference.md |

---

**Phase 2 实施指南完成！等待启动！**

**执行。**
