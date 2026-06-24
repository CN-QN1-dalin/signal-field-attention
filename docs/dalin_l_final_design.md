# Dalin L 最终设计文档

> **版本**: v2.0
> **日期**: 2026-06-24
> **状态**: 最终定稿

---

## 1. 设计原则

### 1.1 核心原则

1. **Agent 优先**: 所有设计围绕 Agent 编程需求展开
2. **零学习成本**: 15 个核心关键字 + 类型自动推断
3. **中文原生**: 变量名、函数名、注释完全支持中文
4. **概率性修复**: 编译器提供智能建议修复，准确率 > 99%
5. **增量编译**: O(1) per change 增量编译，编译速度 < 1 秒

### 1.2 设计边界

- **不是通用编程语言**: Dalin L 是 Agent 专用 DSL
- **不是编译器竞赛**: 聚焦 Agent 编程体验，而非编译器性能
- **不是语言创新实验室**: 参考 Python/JS 语法，降低 Agent 学习成本

---

## 2. 关键字设计

### 2.1 核心关键字 (15 个)

```
fn, let, return, if, else, for, while, match, struct, impl, use, mod, pub, trait, async
```

### 2.2 扩展关键字 (10 个)

```
await, spawn, channel, import, export, type, enum, interface, const, static
```

### 2.3 关键字设计理由

- **参考 Python**: 关键字设计参考 Python 的简洁性
- **参考 Rust**: 类型系统和所有权模型参考 Rust
- **参考 Go**: 并发模型参考 Go 的 goroutine/channel
- **参考 TypeScript**: 类型推断参考 TypeScript

---

## 3. 类型系统

### 3.1 基础类型

```dalan
// 基础类型
let x: int = 42
let y: float = 3.14
let z: string = "hello"
let b: bool = true
```

### 3.2 复合类型

```dalan
// 复合类型
let arr: array<int> = [1, 2, 3]
let map: map<string, int> = {"a": 1, "b": 2}
let opt: option<int> = Some(42)
```

### 3.3 类型推断

```dalan
// 类型自动推断
let x = 42  // 推断为 int
let y = "hello"  // 推断为 string
```

### 3.4 类型约束

```dalan
// 类型约束
fn add<T: numeric>(a: T, b: T) -> T {
    return a + b
}
```

---

## 4. 并发模型

### 4.1 Goroutine 风格

```dalan
// 并发任务
spawn fn task1() {
    println("Task 1")
}

spawn fn task2() {
    println("Task 2")
}
```

### 4.2 Channel 通信

```dalan
// Channel 通信
let ch = channel<int>()

spawn fn producer() {
    for i in 0..10 {
        ch.send(i)
    }
}

spawn fn consumer() {
    while let msg = ch.recv() {
        println(msg)
    }
}
```

### 4.3 异步编程

```dalan
// 异步编程
async fn fetch_data(url: string) -> string {
    let response = http.get(url)
    return response.body
}

fn main() {
    let data = await fetch_data("https://api.example.com")
    println(data)
}
```

---

## 5. 概率性修复

### 5.1 修复策略

```dalan
// 类型错误修复
let x: int = "hello"  // 错误
// 建议修复: let x: string = "hello"

// 语法错误修复
fn add(a: int, b: int) -> int {
    return a + b  // 缺少分号
}
// 建议修复: 添加分号
```

### 5.2 修复准确率目标

- **类型修复**: > 99%
- **语法修复**: > 95%
- **逻辑修复**: > 90%

### 5.3 修复回滚机制

```dalan
// 修复回滚
let x: int = "hello"  // 错误
// 应用修复: let x: string = "hello"
// 回滚: let x: int = "hello"  // 恢复原始代码
```

---

## 6. 增量编译

### 6.1 编译缓存

```dalan
// 编译缓存
// 维护完整的 AST cache
// 仅重新编译变更部分
// 编译速度 < 1 秒
```

### 6.2 编译速度目标

- **增量编译**: < 1 秒
- **全量编译**: < 10 秒
- **缓存命中**: < 0.1 秒

---

## 7. 中文支持

### 7.1 中文变量名

```dalan
// 中文变量名
let 名字 = "大林"
let 年龄 = 25
let 身高 = 1.75
```

### 7.2 中文函数名

```dalan
// 中文函数名
fn 问候(名字: string) -> string {
    return "你好, " + 名字 + "!"
}

fn main() {
    println(问候("大林"))
}
```

### 7.3 中文注释

```dalan
// 这是一个中文注释
/* 这也是中文注释 */
```

---

## 8. Agent 特性

### 8.1 自然语言代码生成

```dalan
// 自然语言代码生成
#[intent] "创建一个排序函数"
fn 排序(arr: array<int>) -> array<int> {
    // 自动生成的排序逻辑
}
```

### 8.2 自动测试生成

```dalan
// 自动测试生成
#[auto-test]
fn 添加(a: int, b: int) -> int {
    return a + b
}
// 自动生成测试用例
// - 添加(1, 2) = 3
// - 添加(0, 0) = 0
// - 添加(-1, 1) = 0
```

### 8.3 统一 FFI

```dalan
// 统一 FFI
use c "stdio.h" as stdio

fn main() {
    stdio.printf("Hello, %s!\n", "Dalin L")
}
```

---

## 9. 路线图

### 9.1 Phase 1: MVP (3 个月)

- [ ] 词法/语法分析
- [ ] 类型推断（HM 类型系统）
- [ ] 基本 I/O
- [ ] LLVM backend
- [ ] `dalin build` / `dalin run`

### 9.2 Phase 2: Agent 特性 (3 个月)

- [ ] 概率性修复编译
- [ ] 自动测试生成
- [ ] 统一 FFI（C FFI）

### 9.3 Phase 3: 完整工具链 (6 个月)

- [ ] 包管理器
- [ ] REPL
- [ ] 格式化器
- [ ] 文档生成
- [ ] async/await + channel

### 9.4 Phase 4: 高级特性 (持续)

- [ ] 自然语言代码生成
- [ ] 多 Agent 协作
- [ ] 模式匹配 + trait

---

## 10. 总结

Dalin L 是一个专为 AI Agent 设计的编程语言，具有以下特点：

1. **零学习成本**: 15 个核心关键字 + 类型自动推断
2. **中文原生**: 变量名、函数名、注释完全支持中文
3. **概率性修复**: 编译器提供智能建议修复，准确率 > 99%
4. **增量编译**: O(1) per change 增量编译，编译速度 < 1 秒
5. **Agent 友好**: 自然语言代码生成、自动测试生成、统一 FFI

**Dalin L — 编译像 Go 一样快，开发像 Python 一样简单，安全像 Rust 一样强。**

**只要 Agent 写代码，就想起 Dalin L。**
