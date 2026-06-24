# DalinOS API 文档

## 基础信息

- **Base URL**: `http://localhost:3000/api/v1`
- **认证**: Bearer Token (JWT)

---

## Agent 商店 API

### 列出所有 Agent

```
GET /agents
```

**Response**:
```json
[
  {
    "id": "1",
    "name": "Test Agent",
    "description": "A test agent",
    "version": "0.1.0",
    "author": "Dalin",
    "tags": ["test"],
    "download_count": 0,
    "rating": 0.0
  }
]
```

### 创建 Agent

```
POST /agents
Content-Type: application/json

{
  "name": "My Agent",
  "description": "My awesome agent",
  "version": "0.1.0",
  "author": "Dalin",
  "tags": ["awesome", "test"]
}
```

**Response**: `201 Created`
```json
{
  "id": "uuid-here",
  "name": "My Agent",
  "description": "My awesome agent",
  "version": "0.1.0",
  "author": "Dalin",
  "tags": ["awesome", "test"],
  "download_count": 0,
  "rating": 0.0
}
```

### 获取 Agent 详情

```
GET /agents/:id
```

**Response**: `200 OK`
```json
{
  "id": "1",
  "name": "Test Agent",
  "description": "A test agent",
  "version": "0.1.0",
  "author": "Dalin",
  "tags": ["test"],
  "download_count": 0,
  "rating": 0.0
}
```

---

## Dalin L 编译器 API

### 编译代码

```
POST /compile
Content-Type: application/json

{
  "code": "fn main() { let x = 42; println(x); }",
  "version": "0.1.0"
}
```

**Response**: `200 OK`
```json
{
  "success": true,
  "output": "42",
  "errors": [],
  "suggestions": []
}
```

**Error Response**: `400 Bad Request`
```json
{
  "success": false,
  "output": "",
  "errors": ["Type mismatch: expected int, got string"],
  "suggestions": ["Try: let x: int = \"hello\""]
}
```

---

## 错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求错误 |
| 401 | 未认证 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

---

*最后更新: 2026-06-24*
