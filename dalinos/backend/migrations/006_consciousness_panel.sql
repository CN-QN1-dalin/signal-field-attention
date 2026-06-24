-- Consciousness Panel Schema
-- 意识面板 — 实时监控 Agent 状态

-- ==================== Agent Runtime State ====================
CREATE TABLE agent_runtime (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'idle' CHECK (status IN ('idle', 'running', 'paused', 'error', 'completed')),
    current_task TEXT,
    tokens_used INT DEFAULT 0,
    memory_mb REAL DEFAULT 0,
    cpu_percent REAL DEFAULT 0,
    uptime_seconds INT DEFAULT 0,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agent_runtime_agent ON agent_runtime(agent_id);
CREATE INDEX idx_agent_runtime_user ON agent_runtime(user_id);
CREATE INDEX idx_agent_runtime_status ON agent_runtime(status);

-- ==================== Agent Collaboration Graph ====================
CREATE TABLE collaboration_graph (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    initiator_agent_id UUID NOT NULL REFERENCES agents(id),
    participant_agent_ids UUID[] NOT NULL,
    graph_data JSONB NOT NULL,  -- 协作关系图
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_collab_session ON collaboration_graph(session_id);
CREATE INDEX idx_collab_status ON collaboration_graph(status);

-- ==================== Token Economy ====================
CREATE TABLE token_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_user_id UUID REFERENCES users(id),
    to_user_id UUID REFERENCES users(id),
    amount DECIMAL(18, 4) NOT NULL,
    transaction_type VARCHAR(30) NOT NULL CHECK (transaction_type IN (
        'agent_purchase',
        'agent_download',
        'review_reward',
        'platform_reward',
        'transfer',
        'refund'
    )),
    reference_id UUID,  -- 关联的交易 ID (如 agent_id, review_id)
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_token_from_user ON token_transactions(from_user_id);
CREATE INDEX idx_token_to_user ON token_transactions(to_user_id);
CREATE INDEX idx_token_type ON token_transactions(transaction_type);
CREATE INDEX idx_token_created ON token_transactions(created_at DESC);

-- ==================== Agent Wallet ====================
CREATE TABLE agent_wallets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    balance DECIMAL(18, 4) DEFAULT 0.0000,
    total_earned DECIMAL(18, 4) DEFAULT 0.0000,
    total_spent DECIMAL(18, 4) DEFAULT 0.0000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_wallet_user ON agent_wallets(user_id);

-- ==================== Functions ====================

-- Update wallet balance after transaction
CREATE OR REPLACE FUNCTION update_wallet_balance()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.transaction_type IN ('agent_purchase', 'transfer') AND NEW.from_user_id IS NOT NULL THEN
        UPDATE agent_wallets 
        SET balance = balance - NEW.amount, total_spent = total_spent + NEW.amount
        WHERE user_id = NEW.from_user_id;
    ELSIF NEW.transaction_type IN ('review_reward', 'platform_reward') AND NEW.to_user_id IS NOT NULL THEN
        UPDATE agent_wallets 
        SET balance = balance + NEW.amount, total_earned = total_earned + NEW.amount
        WHERE user_id = NEW.to_user_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger wallet update on transaction
CREATE TRIGGER trigger_update_wallet_balance
AFTER INSERT ON token_transactions
FOR EACH ROW
EXECUTE FUNCTION update_wallet_balance();

-- Get user wallet summary
CREATE OR REPLACE FUNCTION get_wallet_summary(p_user_id UUID)
RETURNS TABLE (
    balance DECIMAL(18, 4),
    total_earned DECIMAL(18, 4),
    total_spent DECIMAL(18, 4),
    recent_transactions INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        aw.balance,
        aw.total_earned,
        aw.total_spent,
        (SELECT COUNT(*) FROM token_transactions 
         WHERE (from_user_id = p_user_id OR to_user_id = p_user_id)
         AND created_at > NOW() - INTERVAL '30 days')
    FROM agent_wallets aw
    WHERE aw.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- ==================== Seed Data ====================

-- Initialize wallets for existing users
INSERT INTO agent_wallets (user_id, balance, total_earned, total_spent)
SELECT id, 100.00, 0.00, 0.00 FROM users;
