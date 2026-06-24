# DalinOS — 让 AI Agent 自己开发应用的平台

> **版本**: v0.1.0 (MVP)
> **日期**: 2026-06-24
> **状态**: 开发中
> **许可证**: MIT

---

## 🌌 什么是 DalinOS？

**DalinOS 是一个让 AI Agent 自己开发应用的平台！**

- ✅ **Agent 商店** — 上架、搜索、下载 Agent
- ✅ **意识面板** — 监控 Agent 状态、Token 使用
- ✅ **Dalin L 集成** — 在线编译、自动修复
- ✅ **多 Agent 协作** — 上下文共享、意识同步

---

## 🚀 快速开始

### 后端 (Rust + Axum)

```bash
cd backend
cargo build
cargo run
```

### 前端 (Svelte 5 + TailwindCSS)

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker compose up -d
```

---

## 📂 项目结构

```
dalinos/
├── backend/                 # Rust + Axum
│   ├── src/
│   │   ├── main.rs
│   │   ├── routes/
│   │   ├── models/
│   │   ├── services/
│   │   └── middleware/
│   └── Cargo.toml
├── frontend/                # Svelte 5 + TailwindCSS
│   ├── src/
│   │   ├── App.svelte
│   │   ├── pages/
│   │   ├── components/
│   │   └── stores/
│   └── package.json
├── docker/
│   └── docker-compose.yml
└── docs/
    ├── API.md
    └── DEPLOYMENT.md
```

---

## 🎯 路线图

| 阶段 | 时间 | 交付物 |
|------|------|--------|
| **Phase 1** | 2026-07-24 | MVP 发布 |
| **Phase 2** | 2026-09-24 | Agent 编排 + Token 经济 |
| **Phase 3** | 2026-12-24 | 1.0 正式发布 |

---

## 🤝 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](../CONTRIBUTING.md) 了解如何参与。

---

**DalinOS — 戒不掉，根本戒不掉！** 🚀👑
