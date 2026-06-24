# Dalin L 示例代码

## 1. Hello World

```dalan
// 最简单的 Dalan L 程序
fn main() {
    println("Hello, World!")
}
```

## 2. 变量与类型

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

// 常量
const MAX_USERS: int = 1000
const APP_NAME: string = "Dalin L"
```

## 3. 函数

```dalan
// 基本函数
fn 问候(名字: string) -> string {
    return "你好, " + 名字 + "!"
}

// 多参数
fn 添加(a: int, b: int) -> int {
    return a + b
}

// 无返回值
fn 打印(消息: string) {
    println(消息)
}

// 使用
fn main() {
    let 消息 = 问候("大林")
    println(消息)
    
    let 结果 = 添加(1, 2)
    println("1 + 2 = {}", 结果)
}
```

## 4. 控制流

```dalan
// 条件语句
fn 判断年龄(年龄: int) {
    if 年龄 >= 18 {
        println("成年人")
    } else {
        println("未成年人")
    }
}

// for 循环
fn 打印数字() {
    for i in 0..10 {
        println("{}", i)
    }
}

// while 循环
fn 计数() {
    let mut count = 0
    while count < 10 {
        println("{}", count)
        count += 1
    }
}

// 模式匹配
fn 描述值(值: int) {
    match 值 {
        0 => println("零")
        1 => println("一")
        2 => println("二")
        _ => println("其他")
    }
}
```

## 5. 数据结构

```dalan
// 数组
fn 数组示例() {
    let mut 数字 = [1, 2, 3, 4, 5]
    数字.push(6)
    
    for 数字 in 数字 {
        println("{}", 数字)
    }
}

// 哈希表
fn 哈希表示例() {
    let mut 地图 = HashMap::new()
    地图.insert("名字", "大林")
    地图.insert("年龄", "25")
    
    println("{}", 地图.get("名字"))
}

// 结构体
fn 结构体示例() {
    struct 用户 {
        名字: string,
        邮箱: string,
        年龄: int,
    }
    
    let 用户 = 用户 {
        名字: "大林",
        邮箱: "dalin@example.com",
        年龄: 25,
    }
    
    println("{} ({}岁)", 用户.名字, 用户.年龄)
}
```

## 6. 错误处理

```dalan
// Result 类型
fn 除法(a: int, b: int) -> Result<int, Error> {
    if b == 0 {
        error("除零错误")
    } else {
        ok(a / b)
    }
}

// 使用
fn main() {
    match 除法(10, 2) {
        ok(结果) => println("结果: {}", 结果)
        error(错误) => println("错误: {}", 错误)
    }
}

// Option 类型
fn 查找(列表: &[int], 值: int) -> Option<int> {
    for i in 0..列表.len() {
        if 列表[i] == 值 {
            return Some(i)
        }
    }
    None
}
```

## 7. 并发

```dalan
// Channel
fn main() {
    let (发送, 接收) = channel::<int>(16)
    
    spawn async {
        for i in 0..100 {
            发送.send(i).await
        }
    }
    
    for await 值 in 接收 {
        println("收到: {}", 值)
    }
}
```

## 8. 管道操作

```dalan
// 管道操作符
fn main() {
    let 结果 = "hello world"
        |> 转大写
        |> 分割(" ")
        |> 长度()
    
    println("单词数: {}", 结果)
}
```

## 9. 标准库使用

```dalan
// 使用标准库
use std.io
use std.string
use std.collections

fn main() {
    // IO
    println("Hello, World!")
    
    // String
    let s = "hello world"
    println("长度: {}", s.len())
    println("包含 world: {}", s.contains("world"))
    
    // Collections
    let mut vec = Vec::new()
    vec.push(1)
    vec.push(2)
    vec.push(3)
    
    println("向量长度: {}", vec.len())
}
```

## 10. Agent 特性

```dalan
// 自我修复编译
let x: int = "hello"  // 自动修复为 let x: string = "hello"

// 自动测试生成
fn 添加(a: int, b: int) -> int {
    return a + b
}
// 自动生成测试用例

// 统一 FFI
use c "stdio.h" as stdio

fn main() {
    stdio.printf("Hello, %s!\n", "Dalin L")
}
```

## 11. 完整示例：用户管理系统

```dalan
use std.io
use std.collections

// 用户结构体
struct 用户 {
    id: int,
    名字: string,
    邮箱: string,
    年龄: int,
}

// 用户管理器
struct 用户管理器 {
    用户列表: Vec<用户>,
}

impl 用户管理器 {
    fn new() -> 用户管理器 {
        用户管理器 {
            用户列表: Vec::new(),
        }
    }
    
    fn 添加用户(&mut self, 名字: string, 邮箱: string, 年龄: int) -> Result<用户, Error> {
        // 验证邮箱
        if !邮箱.contains("@") {
            return error("无效的邮箱")
        }
        
        // 验证年龄
        if 年龄 < 0 || 年龄 > 150 {
            return error("无效的年龄")
        }
        
        let 用户 = 用户 {
            id: self.用户列表.len() + 1,
            名字,
            邮箱,
            年龄,
        }
        
        self.用户列表.push(用户)
        ok(用户)
    }
    
    fn 查找用户(&self, id: int) -> Option<&用户> {
        self.用户列表.iter().find(|u| u.id == id)
    }
    
    fn 删除用户(&mut self, id: int) -> bool {
        let 初始长度 = self.用户列表.len()
        self.用户列表.retain(|u| u.id != id)
        self.用户列表.len() < 初始长度
    }
    
    fn 列出所有用户(&self) {
        println!("用户列表:")
        for 用户 in self.用户列表 {
            println!("  ID: {}, 名字: {}, 邮箱: {}, 年龄: {}", 
                     用户.id, 用户.名字, 用户.邮箱, 用户.年龄)
        }
    }
}

// 主函数
fn main() {
    let mut 管理器 = 用户管理器::new()
    
    // 添加用户
    match 管理器.添加用户("大林", "dalin@example.com", 25) {
        ok(用户) => println("添加成功: {}", 用户.名字)
        error(错误) => println("添加失败: {}", 错误)
    }
    
    // 列出所有用户
    管理器.列出所有用户()
}
```

## 12. 完整示例：Web 服务器

```dalan
use std.net
use std.io

// HTTP 处理器
fn 处理请求(请求: 请求) -> 响应 {
    match 请求.路径 {
        "/" => 响应 {
            状态码: 200,
            主体: "Hello, World!",
        }
        "/api/users" => {
            // 查询用户
            let 用户列表 = 获取所有用户()
            响应 {
                状态码: 200,
                主体: json_encode(用户列表),
            }
        }
        "/api/users/:id" => {
            // 查询单个用户
            let 用户 = 获取用户(请求.路径参数["id"])
            match 用户 {
                Some(用户) => 响应 {
                    状态码: 200,
                    主体: json_encode(用户),
                }
                None => 响应 {
                    状态码: 404,
                    主体: "用户不存在",
                }
            }
        }
        _ => 响应 {
            状态码: 404,
            主体: "页面不存在",
        }
    }
}

// 主函数
fn main() {
    let 服务器 = net.http_server(8080)
    服务器.处理(处理请求)
    服务器.启动()
    
    println("服务器运行在 http://localhost:8080")
}
```

## 13. 完整示例：数据处理管道

```dalan
use std.io
use std.string
use std.collections
use std.iterator

// 数据处理管道
fn main() {
    // 读取数据
    let 数据 = std.io.read_file("data.csv")
    
    // 解析 CSV
    let 行列表 = 数据.split("\n")
        |> 过滤(|行| 行.len() > 0)
        |> 映射(|行| 行.split(","))
        |> 收集()
    
    // 处理数据
    let 结果 = 行列表
        |> 过滤(|行| 行[1].parse::<int>().is_ok())
        |> 映射(|行| (行[0], 行[1].parse::<int>().unwrap()))
        |> 分组(|(名字, 年龄)| 名字)
        |> 映射(|(名字, 年龄列表)| (名字, 年龄列表.iter().sum::<int>() / 年龄列表.len()))
    
    // 输出结果
    for (名字, 平均年龄) in 结果 {
        println("{}: {}岁", 名字, 平均年龄)
    }
}
```

## 14. 完整示例：并发爬虫

```dalan
use std.net
use std.io
use std.collections
use std.iterator

// 网页爬虫
fn main() {
    let 种子 URLs = ["https://example.com", "https://example.org"]
    let 已爬取 = HashSet::new()
    let (URL 队列, URL 处理器) = channel::<string>(100)
    
    // 启动爬虫
    for 种子 URL in 种子 URLs {
        URL 队列.send(种子 URL).await
    }
    
    // 并发处理
    for _ in 0..10 {
        spawn async {
            for await URL in URL 处理器 {
                if !已爬取.contains(&URL) {
                    let 内容 = net.http_get(URL)
                    let 链接 = 解析链接(内容)
                    
                    for 链接 in 链接 {
                        if !已爬取.contains(&链接) {
                            URL 队列.send(链接).await
                        }
                    }
                    
                    已爬取.insert(URL)
                    println("已爬取: {}", URL)
                }
            }
        }
    }
    
    // 等待完成
    // ...
}
```
