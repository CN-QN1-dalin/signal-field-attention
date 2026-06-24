# Dalin L 语言参考

## 1. 概述

Dalin L 是一种静态类型的编程语言，具有类型自动推断、中文标识符支持和 Agent 原生设计。

## 2. 基本语法

### 2.1 变量声明

```dalan
// 基本类型
let 名字 = "大林"        // string
let 年龄 = 25            // int
let 身高 = 1.75          // float
let 已婚 = false         // bool

// 显式类型
let x: int = 42
let y: float = 3.14
let z: string = "hello"
```

### 2.2 常量

```dalan
const MAX_USERS: int = 1000
const APP_NAME: string = "Dalin L"
```

### 2.3 函数

```dalan
fn 问候(名字: string) -> string {
    return "你好, " + 名字 + "!"
}

// 无返回值
fn 打印(消息: string) {
    println(消息)
}
```

## 3. 控制流

### 3.1 条件语句

```dalan
if 年龄 >= 18 {
    println("成年人")
} else {
    println("未成年人")
}
```

### 3.2 循环

```dalan
// for 循环
for i in 0..10 {
    println(i)
}

// while 循环
while 条件 {
    // 循环体
}
```

### 3.3 模式匹配

```dalan
match 值 {
    1 => println("一")
    2 => println("二")
    _ => println("其他")
}
```

## 4. 类型系统

### 4.1 基本类型

| 类型 | 说明 | 示例 |
|------|------|------|
| int | 整数 | 42 |
| float | 浮点数 | 3.14 |
| string | 字符串 | "hello" |
| bool | 布尔 | true/false |
| char | 字符 | 'a' |
| unit | 空值 | () |

### 4.2 复合类型

| 类型 | 说明 | 示例 |
|------|------|------|
| Array | 数组 | [1, 2, 3] |
| Tuple | 元组 | (1, "hello", true) |
| Option | 可选 | Some(42) / None |
| Result | 结果 | Ok(42) / Error("fail") |

## 5. Agent 特性

### 5.1 自我修复编译

```dalan
// 自动修复类型错误
let x: int = "hello"  // 自动修复为 let x: string = "hello"
```

### 5.2 自动测试生成

```dalan
// 自动生成测试
fn 添加(a: int, b: int) -> int {
    return a + b
}

// 自动生成测试用例
// - 添加(1, 2) = 3
// - 添加(0, 0) = 0
// - 添加(-1, 1) = 0
```

### 5.3 统一 FFI

```dalan
// 自动 C FFI 绑定
use c "stdio.h" as stdio

fn main() {
    stdio.printf("Hello, %s!\n", "Dalin L")
}
```

## 6. 标准库

### 6.1 collections

```dalan
use std.collections

let mut vec = Vec::new()
vec.push(1)
vec.push(2)
vec.push(3)

let map = HashMap::new()
map.insert("key", "value")
```

### 6.2 io

```dalan
use std.io

println("Hello, World!")
print("Hello, ")
let line = read_line()
```

### 6.3 string

```dalan
use std.string

let s = "hello world"
s.len()
s.contains("world")
s.split(" ")
```

## 7. 关键字

### 7.1 核心关键字（15 个）

| 关键字 | 说明 |
|--------|------|
| let | 变量声明 |
| fn | 函数定义 |
| return | 返回值 |
| if | 条件语句 |
| else | 条件分支 |
| match | 模式匹配 |
| for | for 循环 |
| while | while 循环 |
| spawn | 并发任务 |
| async | 异步函数 |
| await | 异步等待 |
| try | 异常处理 |
| catch | 异常捕获 |
| use | 模块导入 |
| trait | 特质定义 |

### 7.2 扩展关键字（10 个）

| 关键字 | 说明 |
|--------|------|
| ok | 成功结果 |
| error | 错误结果 |
| export | 导出模块 |
| pub | 公开可见性 |
| impl | trait 实现 |
| struct | 结构体定义 |
| enum | 枚举定义 |
| type | 类型别名 |
| const | 常量定义 |
| assert | 断言 |

## 8. 运算符

### 8.1 算术运算符

| 运算符 | 说明 | 示例 |
|--------|------|------|
| + | 加法 | 1 + 2 = 3 |
| - | 减法 | 2 - 1 = 1 |
| * | 乘法 | 2 * 3 = 6 |
| / | 除法 | 6 / 2 = 3 |
| % | 取模 | 5 % 2 = 1 |

### 8.2 比较运算符

| 运算符 | 说明 | 示例 |
|--------|------|------|
| == | 等于 | 1 == 1 = true |
| != | 不等于 | 1 != 2 = true |
| < | 小于 | 1 < 2 = true |
| > | 大于 | 2 > 1 = true |
| <= | 小于等于 | 1 <= 1 = true |
| >= | 大于等于 | 2 >= 1 = true |

### 8.3 逻辑运算符

| 运算符 | 说明 | 示例 |
|--------|------|------|
| && | 与 | true && false = false |
| \|\| | 或 | true \|\| false = true |
| ! | 非 | !true = false |

### 8.4 赋值运算符

| 运算符 | 说明 | 示例 |
|--------|------|------|
| = | 赋值 | x = 1 |
| += | 加赋值 | x += 1 |
| -= | 减赋值 | x -= 1 |
| *= | 乘赋值 | x *= 2 |
| /= | 除赋值 | x /= 2 |

### 8.5 其他运算符

| 运算符 | 说明 | 示例 |
|--------|------|------|
| -> | 箭头 | fn foo() -> int |
| => | 模式匹配 | 1 => "one" |
| \|> | 管道 | data \|> filter \|> map |
| ? | 可选 | x? |
| @ | 属性 | #[derive(Debug)] |
| $ | 插值 | "Hello, $name" |

## 9. 错误处理

### 9.1 Result 类型

```dalan
fn 除法(a: int, b: int) -> Result<int, Error> {
    if b == 0 {
        error("除零错误")
    } else {
        ok(a / b)
    }
}

// 使用
match 除法(10, 2) {
    ok(结果) => println("结果: {}", 结果)
    error(错误) => println("错误: {}", 错误)
}
```

### 9.2 可选类型

```dalan
fn 查找(列表: &[int], 值: int) -> Option<int> {
    for i in 0..列表.len() {
        if 列表[i] == 值 {
            return Some(i)
        }
    }
    None
}
```

## 10. 并发

### 10.1 Channel

```dalan
let (发送, 接收) = channel::<int>(16)

spawn async {
    for i in 0..100 {
        发送.send(i).await
    }
}

for await 值 in 接收 {
    处理(值)
}
```

### 10.2 Async/Await

```dalan
fn 异步任务() async {
    let 结果 = 等待某个操作()
    处理(结果)
}
```

## 11. 模块系统

### 11.1 导入模块

```dalan
use std.io
use std.string
use std.collections
```

### 11.2 创建模块

```dalan
// my_module.dalan
pub fn 公共函数() {
    // ...
}

fn 私有函数() {
    // ...
}
```

### 11.3 使用模块

```dalan
use my_module.公共函数

fn main() {
    公共函数()
}
```

## 12. 类型系统

### 12.1 Hindley-Milner

Dalin L 使用 Hindley-Milner 类型推断算法，可以在编译期确定所有类型。

### 12.2 泛型

```dalan
fn 最大值<T: Ord>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}
```

### 12.3 Trait

```dalan
trait 可序列化 {
    fn 序列化(&self) -> Vec<u8>
    fn 反序列化(data: &[u8]) -> Self
}

impl 可序列化 for User {
    fn 序列化(&self) -> Vec<u8> {
        // ...
    }
    fn 反序列化(data: &[u8]) -> Self {
        // ...
    }
}
```

## 13. 中文支持

### 13.1 标识符

```dalan
let 用户名 = "agent_001"
let 密码 = "secret"
fn 问候(名字: string) -> string {
    return "你好, " + 名字 + "!"
}
```

### 13.2 注释

```dalan
// 这是单行注释
/* 这是多行注释
   可以跨越多行 */
```

### 13.3 错误信息

```
错误 [E0001]: 类型不匹配：期望 int，得到 string
  --> main.dalan:5:10
   |
5  | let x: int = "hello"
   |          ^^^ 期望 int，得到 string
```

## 14. 附录

### 14.1 关键字列表

| 核心关键字 | 扩展关键字 |
|------------|------------|
| let | ok |
| fn | error |
| return | export |
| if | pub |
| else | impl |
| match | struct |
| for | enum |
| while | type |
| spawn | const |
| async | assert |
| await | |
| try | |
| catch | |
| use | |
| trait | |

### 14.2 运算符优先级

| 优先级 | 运算符 |
|--------|--------|
| 最高 | . :: |
| 高 | * / % |
| 中 | + - |
| 低 | < > <= >= |
| 最低 | == != && \|\| |
