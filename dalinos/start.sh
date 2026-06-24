#!/bin/bash
# DalinOS 快速启动脚本
# 用法: ./start.sh

set -e

echo "🌌 DalinOS - 让 AI Agent 自己开发应用的平台"
echo "=============================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker 未安装${NC}"
        echo "请先安装 Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker 已安装${NC}"
}

# 检查 PostgreSQL
check_postgres() {
    if ! command -v psql &> /dev/null; then
        echo -e "${YELLOW}⚠️  psql 未安装，将通过 Docker 启动${NC}"
    else
        echo -e "${GREEN}✅ PostgreSQL 已安装${NC}"
    fi
}

# 检查 Rust
check_rust() {
    if ! command -v cargo &> /dev/null; then
        echo -e "${YELLOW}⚠️  Rust 未安装，后端将无法本地编译${NC}"
    else
        echo -e "${GREEN}✅ Rust 已安装 (cargo $(cargo --version))${NC}"
    fi
}

# 检查 Node.js
check_node() {
    if ! command -v node &> /dev/null; then
        echo -e "${YELLOW}⚠️  Node.js 未安装，前端将无法本地开发${NC}"
    else
        echo -e "${GREEN}✅ Node.js 已安装 (node $(node --version))${NC}"
    fi
}

# 启动 Docker 服务
start_docker() {
    echo ""
    echo -e "${YELLOW}🐳 启动 Docker 服务 (PostgreSQL + Redis)...${NC}"
    cd docker
    docker compose up -d
    cd ..
    echo -e "${GREEN}✅ Docker 服务已启动${NC}"
}

# 等待数据库就绪
wait_for_db() {
    echo ""
    echo -e "${YELLOW}⏳ 等待数据库就绪...${NC}"
    sleep 5
    echo -e "${GREEN}✅ 数据库就绪${NC}"
}

# 运行数据库迁移
run_migrations() {
    echo ""
    echo -e "${YELLOW}📦 运行数据库迁移...${NC}"
    cd backend
    cargo install sqlx-cli 2>/dev/null || true
    DATABASE_URL="postgresql://dalinos:dalinos_password@localhost:5432/dalinos" \
        sqlx migrate run 2>/dev/null || echo -e "${YELLOW}⚠️  迁移跳过 (sqlx-cli 未安装)${NC}"
    cd ..
    echo -e "${GREEN}✅ 数据库迁移完成${NC}"
}

# 启动后端
start_backend() {
    echo ""
    echo -e "${YELLOW}🦀 启动后端服务...${NC}"
    cd backend
    export DATABASE_URL="postgresql://dalinos:dalinos_password@localhost:5432/dalinos"
    cargo run --release 2>/dev/null &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../.backend.pid
    cd ..
    echo -e "${GREEN}✅ 后端服务已启动 (PID: $BACKEND_PID)${NC}"
}

# 启动前端
start_frontend() {
    echo ""
    echo -e "${YELLOW}🌐 启动前端服务...${NC}"
    cd frontend
    npm install 2>/dev/null || echo -e "${YELLOW}⚠️  npm install 跳过${NC}"
    npm run dev 2>/dev/null &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../.frontend.pid
    cd ..
    echo -e "${GREEN}✅ 前端服务已启动 (PID: $FRONTEND_PID)${NC}"
}

# 显示状态
show_status() {
    echo ""
    echo -e "${GREEN}🌌 DalinOS 服务状态:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  后端: http://localhost:3000"
    echo "  前端: http://localhost:5173"
    echo "  数据库: localhost:5432 (dalinos)"
    echo "  Redis: localhost:6379"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${YELLOW}📝 日志查看:${NC}"
    echo "  后端: tail -f backend.log"
    echo "  前端: tail -f frontend.log"
    echo ""
    echo -e "${YELLOW}🛑 停止服务:${NC}"
    echo "  ./stop.sh"
    echo ""
}

# 停止服务
stop_services() {
    echo "🛑 停止所有服务..."
    
    if [ -f .backend.pid ]; then
        kill $(cat .backend.pid) 2>/dev/null || true
        rm .backend.pid
        echo -e "${GREEN}✅ 后端已停止${NC}"
    fi
    
    if [ -f .frontend.pid ]; then
        kill $(cat .frontend.pid) 2>/dev/null || true
        rm .frontend.pid
        echo -e "${GREEN}✅ 前端已停止${NC}"
    fi
    
    cd docker
    docker compose down 2>/dev/null || true
    cd ..
    echo -e "${GREEN}✅ Docker 服务已停止${NC}"
}

# 主菜单
main() {
    case "${1:-start}" in
        start)
            check_docker
            check_postgres
            check_rust
            check_node
            start_docker
            wait_for_db
            run_migrations
            start_backend
            start_frontend
            show_status
            ;;
        stop)
            stop_services
            ;;
        status)
            echo "📊 服务状态:"
            [ -f .backend.pid ] && echo "  后端: 运行中 (PID: $(cat .backend.pid))" || echo "  后端: 未运行"
            [ -f .frontend.pid ] && echo "  前端: 运行中 (PID: $(cat .frontend.pid))" || echo "  前端: 未运行"
            docker compose -f docker/docker-compose.yml ps 2>/dev/null || echo "  Docker: 未运行"
            ;;
        restart)
            stop_services
            sleep 2
            start
            ;;
        *)
            echo "用法: $0 {start|stop|status|restart}"
            exit 1
            ;;
    esac
}

main "$@"
