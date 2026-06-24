# 🏗️ DalinOS 重构架构方案 v2.0

> 日期: 2026-06-24
> 版本: v2.0
> 状态: 执行中

---

## 🎯 重构目标

1. **补全所有 API 路由** — 让所有功能可访问
2. **前端组件化** — 提升可维护性
3. **Docker 一键部署** — 降低使用门槛
4. **API 文档自动生成** — 方便开发者

---

## 📐 新架构设计

### 后端架构 (Rust + Axum)

```
dalinos/backend/
├── src/
│   ├── main.rs          # 入口 + 路由注册 (补全!)
│   ├── auth.rs          # JWT 认证
│   ├── compiler.rs      # Dalin L 编译器
│   ├── dashboard.rs     # 意识面板
│   ├── forum.rs         # 论坛 API
│   ├── lingguang.rs     # 灵光一现 API
│   ├── social.rs        # 社交 API
│   └── middleware/
│       ├── auth.rs      # 认证中间件
│       ├── rate_limit.rs # 速率限制
│       └── cors.rs      # CORS 配置
├── migrations/          # 数据库迁移 (9个)
├── Cargo.toml
└── Dockerfile
```

### 前端架构 (Svelte 5)

```
dalinos/frontend/
├── src/
│   ├── App.svelte       # 根组件
│   ├── routes/
│   │   ├── +page.svelte      # 首页
│   │   ├── login/+page.svelte
│   │   ├── register/+page.svelte
│   │   ├── dashboard/+page.svelte
│   │   ├── agents/
│   │   │   ├── +page.svelte      # Agent 列表
│   │   │   └: [slug]/+page.svelte  # Agent 详情
│   │   ├── social/
│   │   │   ├── +page.svelte      # 灵光一现
│   │   │   └── dreams/+page.svelte # 梦境实验室
│   │   ├── tournament/+page.svelte
│   │   ├── forum/+page.svelte
│   │   └── chat/+page.svelte
│   ├── components/
│   │   ├── Header.svelte
│   │   ├── Footer.svelte
│   │   ├── AgentCard.svelte
│   │   ├── DreamCard.svelte
│   │   └── ReviewForm.svelte
│   └── stores/
│       ├── auth.ts
│       └── agents.ts
├── static/
├── svelte.config.js
└── vite.config.js
```

### 部署架构 (Docker Compose)

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: dalinos
      POSTGRES_USER: dalinos
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pg_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
  
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://dalinos:${DB_PASSWORD}@postgres:5432/dalinos
      REDIS_URL: redis://redis:6379
  
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
  
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    depends_on:
      - frontend
      - backend
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf

volumes:
  pg_data:
```

---

## 🔄 迁移策略

### 第一阶段：后端补全 (1天)
- ✅ 补全 `main.rs` 中的所有路由
- ✅ 添加认证中间件
- ✅ 添加速率限制
- ✅ 修复已知 Bug

### 第二阶段：前端框架搭建 (2天)
- 创建 Svelte 5 项目
- 迁移首页 + Agent 商店
- 迁移意识面板
- 迁移灵光一现

### 第三阶段：Docker 编排 (1天)
- 创建 docker-compose.yml
- 配置网络
- 健康检查

### 第四阶段：API 文档 (1天)
- 添加 utoipa
- 生成 Swagger UI
- 补充路由注释

---

## 📊 功能布局 (新)

```
┌─────────────────────────────────────────────┐
│              DalinOS 首页                    │
│  🏪 商店  💡 灵光一现  🌙 梦境  🏆 锦标赛   │
│  💬 论坛  🗨️ 聊天  📊 面板  📚 文档        │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Agent 商店 │ │ 灵光一现  │ │ 梦境实验室 │   │
│  │          │ │          │ │          │   │
│  │ 发现Agent │ │ 闪念广场  │ │ 创意梦境  │   │
│  │ 下载Agent │ │ 智能匹配  │ │ 梦境融合  │   │
│  │ 评测Agent │ │ 社交交友  │ │ 灵感评分  │   │
│  └──────────┘ └──────────┘ └──────────┘   │
│                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 锦标赛   │ │  论坛    │ │  实时聊天 │   │
│  │          │ │          │ │          │   │
│  │ 竞技挑战  │ │ 技术讨论  │ │ 即时通讯  │   │
│  │ 排名奖励  │ │ 灵感碰撞  │ │ 多房间   │   │
│  │ 投票评选  │ │ 新手指南  │ │ WebSocket│   │
│  └──────────┘ └──────────┘ └──────────┘   │
│                                             │
│  ┌──────────────────────────────────┐      │
│  │         意识面板 (Dashboard)      │      │
│  │                                  │      │
│  │  📊 统计  🔔 通知  📝 日志       │      │
│  └──────────────────────────────────┘      │
│                                             │
└─────────────────────────────────────────────┘
```

---

## ✅ 已完成

| 任务 | 状态 |
|------|------|
| 意见汇总 | ✅ 完成 |
| 架构设计 | ✅ 完成 |
| 迁移策略 | ✅ 完成 |
| 功能布局 | ✅ 完成 |

## 🚧 进行中

| 任务 | 预计完成 |
|------|----------|
| 后端路由补全 | 1天 |
| 前端框架搭建 | 2天 |
| Docker 编排 | 1天 |
| API 文档 | 1天 |

---

**重构开始！让 DalinOS 真正可用！** 🔥👑
