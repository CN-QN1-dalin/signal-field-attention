-- DalinOS Database Schema
-- Version: 0.1.0
-- Date: 2026-06-24

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==================== Users ====================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'developer')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);

-- ==================== Agents ====================
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    long_description TEXT,
    version VARCHAR(20) DEFAULT '0.1.0',
    author_id UUID REFERENCES users(id) ON DELETE SET NULL,
    tags TEXT[] DEFAULT '{}',
    category VARCHAR(50),
    repository_url TEXT,
    documentation_url TEXT,
    download_count INT DEFAULT 0,
    rating DECIMAL(3,2) DEFAULT 0.00,
    rating_count INT DEFAULT 0,
    is_featured BOOLEAN DEFAULT FALSE,
    is_verified BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deprecated')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agents_slug ON agents(slug);
CREATE INDEX idx_agents_category ON agents(category);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_rating ON agents(rating DESC);
CREATE INDEX idx_agents_downloads ON agents(download_count DESC);
CREATE INDEX idx_agents_tags ON agents USING GIN(tags);

-- ==================== Agent Versions ====================
CREATE TABLE agent_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,
    changelog TEXT,
    download_url TEXT,
    file_size_bytes BIGINT,
    checksum_sha256 VARCHAR(64),
    requires_dalin_l_version VARCHAR(20),
    is_latest BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agent_versions_agent ON agent_versions(agent_id);
CREATE INDEX idx_agent_versions_latest ON agent_versions(agent_id, is_latest);

-- ==================== Reviews ====================
CREATE TABLE reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating INT NOT NULL CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_reviews_agent ON reviews(agent_id);
CREATE INDEX idx_reviews_user ON reviews(user_id);
CREATE UNIQUE INDEX idx_reviews_user_agent ON reviews(user_id, agent_id);

-- ==================== Downloads ====================
CREATE TABLE downloads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    version VARCHAR(20),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_downloads_agent ON downloads(agent_id);
CREATE INDEX idx_downloads_created ON downloads(created_at DESC);

-- ==================== Token Stats ====================
CREATE TABLE token_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    tokens_used INT NOT NULL DEFAULT 0,
    cost DECIMAL(10,4) NOT NULL DEFAULT 0.0000,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_token_stats_user ON token_stats(user_id);
CREATE INDEX idx_token_stats_agent ON token_stats(agent_id);
CREATE INDEX idx_token_stats_recorded ON token_stats(recorded_at DESC);

-- ==================== Consciousness Sessions ====================
CREATE TABLE consciousness_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_ids UUID[] NOT NULL,
    context JSONB,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed')),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_consciousness_sessions_user ON consciousness_sessions(user_id);
CREATE INDEX idx_consciousness_sessions_status ON consciousness_sessions(status);
CREATE INDEX idx_consciousness_sessions_started ON consciousness_sessions(started_at DESC);

-- ==================== Activity Logs ====================
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_activity_logs_user ON activity_logs(user_id);
CREATE INDEX idx_activity_logs_agent ON activity_logs(agent_id);
CREATE INDEX idx_activity_logs_created ON activity_logs(created_at DESC);

-- ==================== Functions ====================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_reviews_updated_at BEFORE UPDATE ON reviews
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Recalculate agent rating when review changes
CREATE OR REPLACE FUNCTION recalculate_agent_rating()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE agents
    SET rating = (
        SELECT COALESCE(AVG(rating), 0)::DECIMAL(3,2)
        FROM reviews
        WHERE agent_id = NEW.agent_id
    ),
    rating_count = (
        SELECT COUNT(*)
        FROM reviews
        WHERE agent_id = NEW.agent_id
    )
    WHERE id = NEW.agent_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER after_review_insert
    AFTER INSERT ON reviews
    FOR EACH ROW EXECUTE FUNCTION recalculate_agent_rating();

CREATE TRIGGER after_review_update
    AFTER UPDATE ON reviews
    FOR EACH ROW EXECUTE FUNCTION recalculate_agent_rating();

CREATE TRIGGER after_review_delete
    AFTER DELETE ON reviews
    FOR EACH ROW EXECUTE FUNCTION recalculate_agent_rating();

-- ==================== Seed Data ====================

-- Admin user (password: admin123 - hash this properly in production!)
INSERT INTO users (username, email, password_hash, role) VALUES
('admin', 'admin@dalinos.dev', '$2b$12$example_hash', 'admin'),
('dalinos', 'dalinos@dalinos.dev', '$2b$12$example_hash', 'developer');

-- Sample agents
INSERT INTO agents (name, slug, description, category, tags, author_id) VALUES
('Code Assistant', 'code-assistant', 'AI 编程助手，支持多种语言', 'development', ARRAY['coding', 'assistant', 'multi-lang'], (SELECT id FROM users WHERE username = 'dalinos')),
('Data Analyzer', 'data-analyzer', '数据分析与可视化 Agent', 'data', ARRAY['analysis', 'visualization', 'data'], (SELECT id FROM users WHERE username = 'dalinos')),
('Content Creator', 'content-creator', '内容创作与文案生成 Agent', 'creative', ARRAY['writing', 'content', 'creative'], (SELECT id FROM users WHERE username = 'dalinos'));
