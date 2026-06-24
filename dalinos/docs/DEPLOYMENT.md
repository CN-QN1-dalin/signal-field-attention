# DalinOS Deployment Guide

## 环境要求

- Docker 20.10+
- Docker Compose v2
- Rust 1.70+ (本地开发)
- Node.js 18+ (本地开发)

---

## 快速部署 (Docker)

```bash
# 1. 克隆仓库
git clone https://github.com/CN-QN1-dalin/signal-field-attention.git
cd signal-field-attention/dalinos

# 2. 启动所有服务
docker compose up -d

# 3. 检查服务状态
docker compose ps

# 4. 查看日志
docker compose logs -f backend
```

**访问**: http://localhost:8080

---

## 本地开发部署

### 1. 启动基础设施

```bash
# 启动 PostgreSQL + Redis
docker compose -f docker/docker-compose.yml up -d postgres redis
```

### 2. 运行数据库迁移

```bash
cd backend
cargo install sqlx-cli
export DATABASE_URL="postgresql://dalinos:dalinos_password@localhost:5432/dalinos"
sqlx migrate run
```

### 3. 启动后端

```bash
cargo run --release
# 访问: http://localhost:3000/api/v1/health
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
# 访问: http://localhost:5173
```

---

## 生产环境部署

### 1. 环境变量配置

```bash
# .env
DATABASE_URL=postgresql://user:pass@db-host:5432/dalinos
REDIS_URL=redis://redis-host:6379
JWT_SECRET=your-secret-key-here
NODE_ENV=production
```

### 2. 构建 Docker 镜像

```bash
docker compose build
docker compose push  # 推送到镜像仓库
```

### 3. 部署到云服务器

```bash
# SSH 到服务器
ssh user@your-server

# 拉取最新代码
git clone https://github.com/CN-QN1-dalin/signal-field-attention.git
cd signal-field-attention/dalinos

# 启动服务
docker compose up -d

# 配置 Nginx 反向代理
sudo nano /etc/nginx/sites-available/dalinos
```

### 4. Nginx 配置示例

```nginx
server {
    listen 80;
    server_name dalinos.dev;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api/ {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
    }
}
```

---

## 故障排查

### 数据库连接失败

```bash
# 检查 PostgreSQL 是否运行
docker compose ps postgres

# 查看日志
docker compose logs postgres

# 手动测试连接
docker exec -it dalinos-postgres-1 psql -U dalinos -d dalinos
```

### 后端启动失败

```bash
# 检查端口占用
lsof -i :3000

# 查看后端日志
docker compose logs backend

# 手动运行
cd backend && cargo run
```

### 前端构建失败

```bash
# 清除缓存
cd frontend && rm -rf node_modules package-lock.json
npm install

# 手动构建
npm run build
```

---

## 备份与恢复

### 数据库备份

```bash
docker exec dalinos-postgres-1 pg_dump -U dalinos dalinos > backup.sql
```

### 数据库恢复

```bash
docker exec -i dalinos-postgres-1 psql -U dalinos dalinos < backup.sql
```

---

*最后更新: 2026-06-24*
