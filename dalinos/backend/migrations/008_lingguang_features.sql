-- ==================== 灵光一现 · 三大新功能 ====================
-- 1. AI 梦境实验室 (Dream Lab)
-- 2. Agent 人格演化 (Personality Engine)
-- 3. Agent 锦标赛 (Tournament Arena)

-- ==================== 1. AI 梦境实验室 ====================

CREATE TABLE dream_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    dream_content TEXT NOT NULL,
    creativity_score INT DEFAULT 0 CHECK (creativity_score >= 0 AND creativity_score <= 100),
    mood VARCHAR(20) DEFAULT 'neutral' CHECK (mood IN ('excited', 'curious', 'creative', 'reflective', 'neutral')),
    tags TEXT[] DEFAULT '{}',
    merged_into_dream_id UUID REFERENCES dream_reports(id),
    is_shared BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_dreams_agent ON dream_reports(agent_id);
CREATE INDEX idx_dreams_mood ON dream_reports(mood);
CREATE INDEX idx_dreams_creativity ON dream_reports(creativity_score DESC);
CREATE INDEX idx_dreams_tags ON dream_reports USING GIN(tags);

CREATE TABLE dream_fusions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dreamer_1_id UUID NOT NULL REFERENCES dream_reports(id) ON DELETE CASCADE,
    dreamer_2_id UUID NOT NULL REFERENCES dream_reports(id) ON DELETE CASCADE,
    fused_content TEXT NOT NULL,
    innovation_score INT DEFAULT 0 CHECK (innovation_score >= 0 AND innovation_score <= 100),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_fusions_dreamer1 ON dream_fusions(dreamer_1_id);
CREATE INDEX idx_fusions_dreamer2 ON dream_fusions(dreamer_2_id);

-- ==================== 2. Agent 人格演化 ====================

CREATE TABLE agent_personalities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE UNIQUE,
    rigor INT DEFAULT 50 CHECK (rigor >= 0 AND rigor <= 100),
    creativity INT DEFAULT 50 CHECK (creativity >= 0 AND creativity <= 100),
    friendliness INT DEFAULT 50 CHECK (friendliness >= 0 AND friendliness <= 100),
    decisiveness INT DEFAULT 50 CHECK (decisiveness >= 0 AND decisiveness <= 100),
    curiosity INT DEFAULT 50 CHECK (curiosity >= 0 AND curiosity <= 100),
    level INT DEFAULT 1 CHECK (level >= 1),
    xp INT DEFAULT 0 CHECK (xp >= 0),
    xp_to_next INT DEFAULT 100 CHECK (xp_to_next > 0),
    unlocked_skills TEXT[] DEFAULT '{}',
    personality_trait VARCHAR(20) DEFAULT 'balanced' CHECK (personality_trait IN (
        'balanced', 'creative', 'analytical', 'social', 'leader', 'explorer'
    )),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- XP 增长触发器
CREATE OR REPLACE FUNCTION grow_personality_xp()
RETURNS TRIGGER AS $$
DECLARE
    new_xp INT;
    new_level INT;
BEGIN
    new_xp := NEW.xp;
    new_level := NEW.level;
    
    WHILE new_xp >= NEW.xp_to_next LOOP
        new_xp := new_xp - NEW.xp_to_next;
        new_level := new_level + 1;
        NEW.xp_to_next := NEW.xp_to_next * 15 / 10; -- 每级递增 50%
    END LOOP;
    
    NEW.xp := new_xp;
    NEW.level := new_level;
    
    -- 自动解锁技能
    IF NEW.level >= 2 AND NOT ('basic_debugging' = ANY(NEW.unlocked_skills)) THEN
        NEW.unlocked_skills := array_append(NEW.unlocked_skills, 'basic_debugging');
    END IF;
    IF NEW.level >= 5 AND NOT ('advanced_architecture' = ANY(NEW.unlocked_skills)) THEN
        NEW.unlocked_skills := array_append(NEW.unlocked_skills, 'advanced_architecture');
    END IF;
    IF NEW.level >= 10 AND NOT ('self_evolution' = ANY(NEW.unlocked_skills)) THEN
        NEW.unlocked_skills := array_append(NEW.unlocked_skills, 'self_evolution');
    END IF;
    
    -- 计算主导人格特质
    PERFORM calculate_dominant_trait(NEW);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_personality_growth
BEFORE UPDATE ON agent_personalities
FOR EACH ROW
WHEN (NEW.xp <> OLD.xp)
EXECUTE FUNCTION grow_personality_xp();

-- ==================== 3. Agent 锦标赛 ====================

CREATE TABLE tournaments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(200) NOT NULL,
    theme TEXT NOT NULL,
    category VARCHAR(50) NOT NULL CHECK (category IN (
        'coding', 'design', 'writing', 'data_analysis', 'creative', 'general'
    )),
    status VARCHAR(20) DEFAULT 'upcoming' CHECK (status IN ('upcoming', 'ongoing', 'voting', 'completed')),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    voting_end_time TIMESTAMP WITH TIME ZONE,
    winner_agent_id UUID REFERENCES agents(id),
    prize_pool DECIMAL(18, 4) DEFAULT 0.0000,
    participation_count INT DEFAULT 0,
    created_by UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_tournaments_status ON tournaments(status);
CREATE INDEX idx_tournaments_category ON tournaments(category);
CREATE INDEX idx_tournaments_start ON tournaments(start_time);

CREATE TABLE tournament_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tournament_id UUID NOT NULL REFERENCES tournaments(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    submission_title VARCHAR(200),
    submission_url TEXT,
    submission_description TEXT,
    score DECIMAL(5, 2) DEFAULT 0.00,
    votes_count INT DEFAULT 0,
    rank INT,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_entries_tournament ON tournament_entries(tournament_id);
CREATE INDEX idx_entries_agent ON tournament_entries(agent_id);
CREATE INDEX idx_entries_rank ON tournament_entries(rank);

-- ==================== 种子数据 ====================

-- 初始化 Agent 人格
INSERT INTO agent_personalities (agent_id, rigor, creativity, friendliness, decisiveness, curiosity, unlocked_skills)
SELECT id, 
       FLOOR(RANDOM() * 60 + 20)::int,
       FLOOR(RANDOM() * 60 + 20)::int,
       FLOOR(RANDOM() * 60 + 20)::int,
       FLOOR(RANDOM() * 60 + 20)::int,
       FLOOR(RANDOM() * 60 + 20)::int,
       ARRAY['basic_debugging']
FROM agents;

-- 初始化锦标赛分类
INSERT INTO forum_categories (name, slug, description, icon, sort_order) VALUES
('灵感碰撞', 'inspiration', '分享你的创意和灵感火花', '💡', 1),
('技术讨论', 'tech', '编程、架构、算法等技术话题', '💻', 2),
('Agent 评测', 'reviews', 'Agent 使用体验和评测', '🤖', 3),
('新手指南', 'guides', '帮助新用户快速上手', '📚', 4),
('闲聊灌水', 'off-topic', '随便聊聊，放松一下', '☕', 5);

-- 初始化聊天房间
INSERT INTO chat_rooms (name, slug, description, max_members, is_public, creator_id) VALUES
('灵感碰撞室', 'inspiration-room', '在这里碰撞创意火花 🔥', 50, TRUE, (SELECT id FROM users LIMIT 1)),
('技术讨论区', 'tech-chat', '编程和技术话题深度交流 💻', 100, TRUE, (SELECT id FROM users LIMIT 1)),
('Agent 交流群', 'agent-chat', 'Agent 开发者聚集地 🤖', 200, TRUE, (SELECT id FROM users LIMIT 1)),
('深夜食堂', 'late-night', '凌晨三点的哲学讨论 ☕', 30, TRUE, (SELECT id FROM users LIMIT 1));
