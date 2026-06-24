-- API Endpoints Documentation
# DalinOS API v1 文档

## 认证相关

### 用户注册
```http
POST /api/v1/auth/register
Content-Type: application/json

{
    "username": "dalinos",
    "email": "dalinos@dalinos.dev",
    "password": "Admin123456",
    "password_confirm": "Admin123456"
}

Response 201:
{
    "id": "uuid",
    "username": "dalinos",
    "email": "dalinos@dalinos.dev",
    "avatar_url": null,
    "role": "user",
    "created_at": "2026-06-24T10:00:00Z"
}
```

### 用户登录
```http
POST /api/v1/auth/login
Content-Type: application/json

{
    "email": "dalinos@dalinos.dev",
    "password": "Admin123456"
}

Response 200:
{
    "access_token": "eyJhbGci...",
    "refresh_token": "xyz123",
    "user": { ... }
}
```

### 刷新 Token
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
    "refresh_token": "xyz123"
}

Response 200:
{
    "access_token": "new_eyJhbGci...",
    "refresh_token": "new_xyz123",
    "user": { ... }
}
```

### 登出
```http
POST /api/v1/auth/logout
Content-Type: application/json

{
    "refresh_token": "xyz123"
}

Response 204: No Content
```

---

## Agent 相关

### 列出 Agent
```http
GET /api/v1/agents?category=coding&tag=assistant&sort=rating&page=1&limit=20

Response 200:
[
    {
        "id": "uuid",
        "name": "Code Assistant",
        "slug": "code-assistant",
        "description": "AI 编程助手",
        "version": "1.0.0",
        "author_name": "dalinos",
        "tags": ["coding", "assistant"],
        "category": "coding",
        "download_count": 150,
        "rating": 4.8,
        "rating_count": 12,
        "is_featured": true,
        "is_verified": true,
        "status": "active",
        "created_at": "2026-06-24T10:00:00Z"
    }
]
```

### 创建 Agent (需要认证)
```http
POST /api/v1/agents
Authorization: Bearer eyJhbGci...
Content-Type: application/json

{
    "name": "My Agent",
    "description": "My awesome agent",
    "long_description": "Detailed description...",
    "version": "0.1.0",
    "tags": ["ai", "assistant"],
    "category": "productivity",
    "repository_url": "https://github.com/...",
    "documentation_url": "https://docs..."
}

Response 201:
{
    "id": "uuid",
    "name": "My Agent",
    "slug": "my-agent",
    ...
}
```

### 获取 Agent 详情
```http
GET /api/v1/agents/code-assistant

Response 200:
{
    "id": "uuid",
    "name": "Code Assistant",
    "slug": "code-assistant",
    ...
}
```

### 更新 Agent (需要认证)
```http
PUT /api/v1/agents/code-assistant
Authorization: Bearer eyJhbGci...
Content-Type: application/json

{
    "name": "Updated Name",
    "description": "Updated description"
}

Response 200:
{
    "id": "uuid",
    "name": "Updated Name",
    ...
}
```

### 删除 Agent (需要认证)
```http
DELETE /api/v1/agents/code-assistant
Authorization: Bearer eyJhbGci...

Response 204: No Content
```

### 获取 Agent 版本
```http
GET /api/v1/agents/code-assistant/versions

Response 200:
[
    {
        "id": "uuid",
        "agent_id": "uuid",
        "version": "1.0.0",
        "changelog": "First release",
        "download_url": "https://...",
        "file_size_bytes": 1048576,
        "checksum_sha256": "abc123...",
        "is_latest": true,
        "created_at": "2026-06-24T10:00:00Z"
    }
]
```

### 获取 Agent 评价
```http
GET /api/v1/agents/code-assistant/reviews?limit=10

Response 200:
[
    {
        "id": "uuid",
        "agent_id": "uuid",
        "user_id": "uuid",
        "username": "dalinos",
        "rating": 5,
        "comment": "Excellent!",
        "created_at": "2026-06-24T10:00:00Z"
    }
]
```

### 创建评价 (需要认证)
```http
POST /api/v1/agents/code-assistant/reviews
Authorization: Bearer eyJhbGci...
Content-Type: application/json

{
    "rating": 5,
    "comment": "Great agent!"
}

Response 201:
{
    "id": "uuid",
    "agent_id": "uuid",
    "user_id": "uuid",
    "rating": 5,
    "comment": "Great agent!",
    ...
}
```

---

## 编译相关

### 编译代码 (需要认证)
```http
POST /api/v1/compile
Authorization: Bearer eyJhbGci...
Content-Type: application/json

{
    "code": "fn main() { println!(\"Hello\"); }",
    "version": "0.1.0"
}

Response 200:
{
    "success": true,
    "output": "Compilation successful!\nOutput: Hello",
    "errors": [],
    "warnings": [],
    "suggestions": []
}
```

---

## 通知相关

### 获取未读通知数
```http
GET /api/v1/notifications/unread-count
Authorization: Bearer eyJhbGci...

Response 200:
{
    "count": 5
}
```

### 获取通知列表
```http
GET /api/v1/notifications?page=1&limit=20&unread_only=false
Authorization: Bearer eyJhbGci...

Response 200:
[
    {
        "id": "uuid",
        "type": "agent_published",
        "title": "New Agent Published",
        "message": "Code Assistant v1.0.0 is now available",
        "data": {
            "agent_id": "uuid",
            "agent_name": "Code Assistant"
        },
        "is_read": false,
        "created_at": "2026-06-24T10:00:00Z"
    }
]
```

### 标记通知为已读
```http
PUT /api/v1/notifications/{id}/read
Authorization: Bearer eyJhbGci...

Response 204: No Content
```

### 标记所有通知为已读
```http
PUT /api/v1/notifications/read-all
Authorization: Bearer eyJhbGci...

Response 204: No Content
```

---

## 文件上传相关

### 上传文件
```http
POST /api/v1/upload
Authorization: Bearer eyJhbGci...
Content-Type: multipart/form-data

file: <binary>

Response 201:
{
    "id": "uuid",
    "file_name": "avatar.jpg",
    "file_path": "/uploads/avatars/uuid.jpg",
    "file_size_bytes": 102400,
    "mime_type": "image/jpeg",
    "checksum_sha256": "abc123..."
}
```

### 获取文件
```http
GET /api/v1/files/{file_id}

Response 200:
<binary file content>
```

---

## 错误响应

### 400 Bad Request
```json
{
    "error": "validation_error",
    "message": "Name must be between 2 and 100 characters",
    "details": [
        {
            "field": "name",
            "message": "Too short"
        }
    ]
}
```

### 401 Unauthorized
```json
{
    "error": "authentication_error",
    "message": "Invalid or expired token"
}
```

### 403 Forbidden
```json
{
    "error": "authorization_error",
    "message": "You don't have permission to perform this action"
}
```

### 404 Not Found
```json
{
    "error": "not_found",
    "message": "Agent not found"
}
```

### 409 Conflict
```json
{
    "error": "conflict",
    "message": "Username already exists"
}
```

### 429 Too Many Requests
```json
{
    "error": "rate_limit_exceeded",
    "message": "Too many requests. Please try again later.",
    "retry_after": 60
}
```

### 500 Internal Server Error
```json
{
    "error": "internal_error",
    "message": "An unexpected error occurred"
}
```

---

## 速率限制

| 端点 | 限制 |
|------|------|
| 所有 API | 100 请求/分钟 |
| 登录/注册 | 10 请求/分钟 |
| 编译接口 | 30 请求/分钟 |
| 文件上传 | 5 请求/分钟 |

---

*最后更新: 2026-06-24*
