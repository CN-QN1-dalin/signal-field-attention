# Contributing to DalinOS

Thank you for your interest in contributing to DalinOS! 🚀

## Getting Started

### Prerequisites

- Rust 1.70+
- Node.js 18+
- PostgreSQL 15+
- Docker & Docker Compose

### Local Setup

```bash
# 1. Fork and clone
git clone https://github.com/CN-QN1-dalin/signal-field-attention.git
cd signal-field-attention/dalinos

# 2. Start infrastructure
docker compose -f docker/docker-compose.yml up -d postgres redis

# 3. Run migrations
cd backend && cargo install sqlx-cli
export DATABASE_URL="postgresql://dalinos:dalinos_password@localhost:5432/dalinos"
sqlx migrate run

# 4. Start development servers
cargo run &  # Backend on :3000
cd ../frontend && npm install && npm run dev  # Frontend on :5173
```

## Development Guidelines

### Backend (Rust)

- Use `tracing` for logging
- Follow Rust naming conventions
- Write tests for new features
- Run `cargo clippy` before committing

```bash
cargo clippy --all-targets --all-features
cargo test
```

### Frontend

- Use Svelte 5 runes syntax
- Follow TailwindCSS utility-first approach
- Keep components small and reusable
- Add comments for complex logic

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add agent search functionality
fix: resolve login redirect loop
docs: update API documentation
```

## Pull Request Process

1. Create a feature branch (`feat/your-feature`)
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Submit PR with clear description

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the problem, not the person

---

*DalinOS — 让 AI Agent 自己开发应用的平台*
