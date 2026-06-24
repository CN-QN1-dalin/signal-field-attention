# Dalin L 标准库参考

## 1. std.collections

### 1.1 Vec

```dalan
use std.collections

// 创建
let mut vec = Vec::new()
let vec2 = Vec::from([1, 2, 3])

// 操作
vec.push(1)
vec.pop()
vec.get(0)
vec.len()
vec.is_empty()
vec.clear()
```

### 1.2 HashMap

```dalan
use std.collections

// 创建
let mut map = HashMap::new()

// 操作
map.insert("key", "value")
map.get("key")
map.remove("key")
map.len()
map.is_empty()
map.clear()
```

### 1.3 HashSet

```dalan
use std.collections

// 创建
let mut set = HashSet::new()

// 操作
set.insert(1)
set.remove(1)
set.contains(1)
set.len()
set.is_empty()
set.clear()
```

## 2. std.io

### 2.1 打印

```dalan
use std.io

println("Hello, World!")
print("Hello, ")
println("Number: {}", 42)
println("String: {}", "hello")
```

### 2.2 文件操作

```dalan
use std.io

// 读取文件
let content = std.io.read_file("test.txt")

// 写入文件
std.io.write_file("test.txt", "Hello, World!")

// 追加文件
std.io.append_file("test.txt", "More content")
```

### 2.3 标准输入

```dalan
use std.io

let input = std.io.read_line()
println("You entered: {}", input)
```

## 3. std.string

### 3.1 基本操作

```dalan
use std.string

let s = "hello world"

s.len()           // 5
s.contains("world")  // true
s.starts_with("hello")  // true
s.ends_with("world")    // true
s.trim()          // "hello world"
s.to_upper()      // "HELLO WORLD"
s.to_lower()      // "hello world"
```

### 3.2 分割

```dalan
use std.string

let s = "hello,world,test"
let parts = s.split(",")
// parts: ["hello", "world", "test"]
```

### 3.3 拼接

```dalan
use std.string

let s1 = "hello"
let s2 = "world"
let s3 = s1 + " " + s2  // "hello world"
```

## 4. std.option

### 4.1 Option

```dalan
use std.option

let some_value: Option<int> = Some(42)
let none_value: Option<int> = None

some_value.is_some()    // true
some_value.is_none()    // false
some_value.unwrap()     // 42
some_value.unwrap_or(0) // 42
```

## 5. std.result

### 5.1 Result

```dalan
use std.result

let ok_result: Result<int, Error> = ok(42)
let err_result: Result<int, Error> = error("failed")

ok_result.is_ok()       // true
ok_result.is_err()      // false
err_result.is_ok()      // false
err_result.is_err()     // true
ok_result.unwrap()      // 42
```

## 6. std.iterator

### 6.1 Iterator

```dalan
use std.iterator

let vec = Vec::from([1, 2, 3, 4, 5])

// map
let doubled = vec.iter().map(|x| x * 2)
// [2, 4, 6, 8, 10]

// filter
let evens = vec.iter().filter(|x| x % 2 == 0)
// [2, 4]

// reduce
let sum = vec.iter().reduce(|acc, x| acc + x)
// 15

// count
let count = vec.iter().count()
// 5

// collect
let collected: Vec<int> = vec.iter().collect()
// [1, 2, 3, 4, 5]
```

## 7. std.math

### 7.1 基本函数

```dalan
use std.math

math.abs(-5)      // 5
math.sqrt(16)     // 4.0
math.pow(2, 3)    // 8.0
math.max(1, 2)    // 2
math.min(1, 2)    // 1
math.round(3.14)  // 3
math.floor(3.14)  // 3
math.ceil(3.14)   // 4
```

### 7.2 三角函数

```dalan
use std.math

math.sin(0.0)     // 0.0
math.cos(0.0)     // 1.0
math.tan(0.0)     // 0.0
math.atan2(1.0, 1.0)  // 0.785...
```

## 8. std.time

### 8.1 时间

```dalan
use std.time

let now = time.now()
let timestamp = time.timestamp()
let duration = time.duration_since(start, end)
```

## 9. std.net

### 9.1 HTTP

```dalan
use std.net

// GET 请求
let response = net.http_get("https://example.com")

// POST 请求
let response = net.http_post("https://example.com", body)
```

### 9.2 TCP

```dalan
use std.net

// 连接
let socket = net.tcp_connect("localhost", 8080)

// 发送
socket.send("Hello")

// 接收
let response = socket.receive()

// 关闭
socket.close()
```

## 10. std.crypto

### 10.1 哈希

```dalan
use std.crypto

let hash = crypto.sha256("hello")
let md5 = crypto.md5("hello")
```

### 10.2 加密

```dalan
use std.crypto

// AES 加密
let encrypted = crypto.aes_encrypt(key, data)

// AES 解密
let decrypted = crypto.aes_decrypt(key, encrypted)
```

## 11. std.sqlite

### 11.1 数据库操作

```dalan
use std.sqlite

// 打开数据库
let db = sqlite.open("test.db")

// 执行查询
let results = db.execute("SELECT * FROM users")

// 执行插入
db.execute("INSERT INTO users (name) VALUES (?)", ["John"])

// 关闭数据库
db.close()
```

## 12. 错误处理

### 12.1 标准错误

```dalan
use std.error

// 创建错误
let err = error.new("Something went wrong")

// 错误信息
err.message()    // "Something went wrong"
err.code()       // 1
```

### 12.2 自定义错误

```dalan
use std.error

// 定义错误类型
trait CustomError: error.Error {
    fn custom_message(&self) -> string
}

// 实现错误类型
struct MyError {
    message: string,
}

impl CustomError for MyError {
    fn custom_message(&self) -> string {
        self.message
    }
}
```
