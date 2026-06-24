# Dalin L — 下一步行动计划

> **日期**: 2026-06-24
> **目标**: MVP 0.1.0（4 个月，2 人团队）
> **状态**: 设计完成，进入执行阶段

---

## 立即可做的事（本周内）

### 1. 创建仓库结构
```bash
# 新建 dalin-l 仓库
mkdir -p dalin-l/{src,tests,docs,examples,std}
cd dalin-l
git init
```

### 2. 编写词法分析器原型
```rust
// src/lexer.rs — 词法分析器
// 目标：识别 25 个关键字 + 标识符 + 字面量
```

### 3. 编写语法分析器原型
```rust
// src/parser.rs — 语法分析器
// 目标：生成 AST
```

### 4. 搭建 CI/CD
```yaml
# .github/workflows/build.yml
# 目标：每次 commit 自动编译 + 测试
```

---

## 第一阶段：MVP（4 个月）

### Month 1：词法/语法分析 + HM 类型推断

| 周 | 任务 | 交付物 |
|----|------|--------|
| W1 | 词法分析器 | 可以 tokenize Dalin L 源码 |
| W2 | 语法分析器 | 可以生成 AST |
| W3 | HM 类型推断 | 可以推断简单表达式的类型 |
| W4 | 类型检查 | 可以检测类型错误 |

**Month 1 交付物**：
- ✅ `dalin lex` — 词法分析
- ✅ `dalin parse` — 语法分析
- ✅ `dalin type-check` — 类型检查

### Month 2：LLVM backend + 基本 I/O

| 周 | 任务 | 交付物 |
|----|------|--------|
| W1 | LLVM IR 生成 | 可以生成 LLVM IR |
| W2 | LLVM 编译 | 可以生成原生二进制 |
| W3 | 基本 I/O | println! / print! |
| W4 | 字符串处理 | 字符串插值 + 拼接 |

**Month 2 交付物**：
- ✅ `dalin build` — 编译为原生二进制
- ✅ `dalin run` — 运行程序
- ✅ Hello World 可以运行

### Month 3：模式匹配 + dalin CLI

| 周 | 任务 | 交付物 |
|----|------|--------|
| W1 | 模式匹配 | match 表达式 |
| W2 | 结构体/枚举 | struct / enum |
| W3 | dalin CLI | dalin build/run/test |
| W4 | 标准库基础 | collections/string |

**Month 3 交付物**：
- ✅ `dalin build` — 编译 + 类型检查
- ✅ `dalin run` — 运行
- ✅ `dalin test` — 测试
- ✅ 模式匹配可以工作

### Month 4：REPL + 中文支持

| 周 | 任务 | 交付物 |
|----|------|--------|
| W1 | REPL | dalin repl 交互式开发 |
| W2 | 中文变量名 | 支持中文标识符 |
| W3 | 中文错误信息 | 编译错误中文显示 |
| W4 | 打包发布 | 0.1.0 发布 |

**Month 4 交付物**：
- ✅ `dalin repl` — 交互式开发（响应 < 100ms）
- ✅ 中文变量名/函数名
- ✅ 中文错误信息
- ✅ **MVP 0.1.0 发布**

---

## 第二阶段：Agent 特性（5 个月）

### Month 5-6：并发 + 自我修复

| 任务 | 交付物 |
|------|--------|
| async/await | 可以写异步代码 |
| channel | spawn + await + channel |
| 自我修复编译 | 概率性修复 + 确认 |

### Month 7-8：自动测试 + 统一 FFI

| 任务 | 交付物 |
|------|--------|
| 自动测试生成 | #[auto-test] 覆盖率 > 90% |
| 统一 FFI（C） | use c "libm" |

### Month 9：格式化器 + 文档生成

| 任务 | 交付物 |
|------|--------|
| dalin fmt | 统一代码风格 |
| dalin docs | 自动生成文档 |

---

## 第三阶段：生态建设（6 个月）

### Month 10-12：包管理 + 中文全面支持 + VSCode 插件

| 任务 | 交付物 |
|------|--------|
| dalin add/remove/update | 包管理器 |
| 中文全面支持 | 变量名/注释/错误信息 |
| VSCode 插件 | 补全/跳转/重构 |

### Month 13-15：标准库 + 文档站点 + 社区

| 任务 | 交付物 |
|------|--------|
| collections/io/net/crypto | 标准库扩展 |
| dalin-lang.org | 文档站点 |
| GitHub/Discord/微信群 | 社区建设 |

---

## 第四阶段：高级特性（5.5 个月）

### Month 16-18：自然语言补全 + 多 Agent 协作 + 扩展关键字

| 任务 | 交付物 |
|------|--------|
| LLM 集成 | 意图补全 |
| AST 合并 | 多 Agent 协作 |
| pub/impl/struct 等 | 扩展关键字 |

### Month 19-20：WASM + 1.0 发布

| 任务 | 交付物 |
|------|--------|
| WASM 支持 | WebAssembly 编译 |
| 1.0 发布 | 生产就绪 |

---

## 关键里程碑

| 日期 | 里程碑 | 状态 |
|------|--------|------|
| 2026-07 | Month 1 完成：词法/语法分析 | ⏳ 待开始 |
| 2026-08 | Month 2 完成：LLVM backend | ⏳ 待开始 |
| 2026-09 | Month 3 完成：模式匹配 + CLI | ⏳ 待开始 |
| **2026-10** | **MVP 0.1.0 发布** | ⏳ 待开始 |
| 2026-12 | Phase 2 完成：Agent 特性 | ⏳ 待开始 |
| 2027-06 | Phase 3 完成：生态建设 | ⏳ 待开始 |
| **2027-11** | **1.0 发布** | ⏳ 待开始 |

---

## 资源需求

| 资源 | 数量 | 说明 |
|------|------|------|
| 团队 | 2 人 | 编译器工程师 + 前端工程师 |
| 时间 | 20.5 个月 | 全职开发 |
| 预算 | TBD | 取决于团队薪资 |
| 基础设施 | 1 | GitHub + CI/CD |
| 文档 | 1 | 中文 + 英文 |

---

## 下一步行动（今天）

1. **创建 dalin-l 仓库** — GitHub 上新建 `CN-QN1-dalin/dalin-l`
2. **编写 lexer 原型** — Rust + pest parser
3. **搭建 CI/CD** — GitHub Actions 自动编译
4. **写第一个 Hello World** — 证明编译器可以工作

**这就是下一步。没有犹豫，没有讨论。执行。**
