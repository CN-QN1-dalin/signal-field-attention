# 1. 用户注册
curl -X POST http://localhost:3000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "dalinos",
    "email": "admin@dalinos.ai",
    "password": "DalinOS2026!",
    "password_confirm": "DalinOS2026!"
  }'

# 2. 用户登录
curl -X POST http://localhost:3000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@dalinos.ai",
    "password": "DalinOS2026!"
  }'

# 3. 创建 Agent
curl -X POST http://localhost:3000/api/v1/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -d '{
    "name": "Code Assistant",
    "description": "AI 编程助手",
    "version": "1.0.0",
    "tags": ["coding", "assistant"],
    "category": "development"
  }'

# 4. 列出所有 Agent
curl http://localhost:3000/api/v1/agents

# 5. 编译 Dalin L 代码
curl -X POST http://localhost:3000/api/v1/compile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -d '{
    "code": "让 x = 1\n让 y = 2\n返回 x + y"
  }'

# 6. 健康检查
curl http://localhost:3000/api/v1/health
