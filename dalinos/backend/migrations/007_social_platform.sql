-- ==================== 灵光一现 · 智能体交友平台 ====================
-- SparkMatch — Where AI Agents Find Their Perfect Match

-- ==================== 1. 用户社交档案 ====================
CREATE TABLE social_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    bio TEXT,                                    -- 个性签名
    avatar_url TEXT,                             -- 头像
    interests TEXT[],                            -- 兴趣标签 (如 ['coding', 'design'])
    preferred_tags TEXT[],                       -- 偏好的 Agent 标签
    personality VARCHAR(20),                     -- personality: creative/analytical/social
    availability VARCHAR(20) DEFAULT 'open' CHECK (availability IN ('open', 'busy', 'offline')),
    match_score INT DEFAULT 0,                   -- 匹配度评分 (自动计算)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_social_interests ON social_profiles USING GIN(interests);
CREATE INDEX idx_social_personality ON social_profiles(personality);

-- ==================== 2. 喜欢/匹配 ====================
CREATE TABLE social_likes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    to_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    is_match BOOLEAN DEFAULT FALSE,              -- 双向喜欢 = match
    liked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(from_user_id, to_user_id)
);

CREATE INDEX idx_likes_from ON social_likes(from_user_id);
CREATE INDEX idx_likes_to ON social_likes(to_user_id);
CREATE INDEX idx_matches ON social_likes(is_match);

-- ==================== 3. 即时消息 ====================
CREATE TABLE social_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_user_id UUID NOT NULL REFERENCES users(id),
    to_user_id UUID NOT NULL REFERENCES users(id),
    content TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_msgs_from ON social_messages(from_user_id);
CREATE INDEX idx_msgs_to ON social_messages(to_user_id);
CREATE INDEX idx_msgs_created ON social_messages(created_at DESC);

-- ==================== 4. 闪念广场 (Flash Feed) ====================
CREATE TABLE flash_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL CHECK (char_length(content) <= 280),  -- 类似 Tweet 限制
    mood VARCHAR(20),                            -- mood: excited/focused/chill/inspired
    tags TEXT[],                                 -- 话题标签
    likes_count INT DEFAULT 0,
    replies_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_flash_mood ON flash_posts(mood);
CREATE INDEX idx_flash_tags ON flash_posts USING GIN(tags);
CREATE INDEX idx_flash_created ON flash_posts(created_at DESC);

-- ==================== 5. Agent 配对推荐 ====================
CREATE TABLE agent_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    compatibility_score FLOAT NOT NULL,          -- 兼容性分数 (0-100)
    match_reason TEXT,                           -- 匹配原因 (自动解释)
    accepted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agent_match_user ON agent_matches(user_id);
CREATE INDEX idx_agent_match_agent ON agent_matches(agent_id);
CREATE INDEX idx_agent_match_score ON agent_matches(compatibility_score DESC);

-- ==================== 6. 智能匹配引擎 ====================

-- 计算两个用户之间的兴趣重叠度
CREATE OR REPLACE FUNCTION calculate_interest_overlap(
    p_user1_id UUID,
    p_user2_id UUID
) RETURNS FLOAT AS $$
DECLARE
    interests1 TEXT[];
    interests2 TEXT[];
    overlap INT;
    total INT;
BEGIN
    SELECT interests INTO interests1 FROM social_profiles WHERE user_id = p_user1_id;
    SELECT interests INTO interests2 FROM social_profiles WHERE user_id = p_user2_id;
    
    IF interests1 IS NULL OR interests2 IS NULL THEN
        RETURN 0.0;
    END IF;
    
    -- 计算 Jaccard 相似度
    overlap := array_length(interests1 && interests2, 1);
    total := array_length(interests1 || interests2, 1) - overlap;
    
    IF total = 0 THEN
        RETURN 0.0;
    END IF;
    
    RETURN (overlap::FLOAT / total) * 100;
END;
$$ LANGUAGE plpgsql;

-- 为用户推荐潜在匹配
CREATE OR REPLACE FUNCTION recommend_matches(p_user_id UUID, p_limit INT DEFAULT 10)
RETURNS TABLE (
    recommended_user_id UUID,
    username VARCHAR(100),
    bio TEXT,
    avatar_url TEXT,
    match_percentage INT,
    shared_interests TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sp.user_id,
        u.username,
        sp.bio,
        sp.avatar_url,
        ROUND((
            calculate_interest_overlap(p_user_id, sp.user_id) * 0.6 +
            CASE WHEN sp.personality = 'creative' AND u.role = 'developer' THEN 20
                 WHEN sp.personality = 'analytical' AND u.role = 'data_analyst' THEN 20
                 ELSE 0
            END
        )::int) as match_percentage,
        sp.interests as shared_interests
    FROM social_profiles sp
    JOIN users u ON sp.user_id = u.id
    WHERE sp.user_id != p_user_id
    AND sp.availability IN ('open', 'busy')
    AND NOT EXISTS (
        SELECT 1 FROM social_likes sl 
        WHERE sl.from_user_id = p_user_id AND sl.to_user_id = sp.user_id
    )
    ORDER BY match_percentage DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ==================== 7. 自动更新匹配分数 ====================

CREATE OR REPLACE FUNCTION update_match_scores()
RETURNS TRIGGER AS $$
BEGIN
    -- 当收到新的喜欢时，检查是否形成匹配
    IF NEW.is_match IS FALSE THEN
        -- 检查对方是否也喜欢了自己
        UPDATE social_likes 
        SET is_match = TRUE
        WHERE from_user_id = NEW.to_user_id 
        AND to_user_id = NEW.from_user_id
        AND is_match = FALSE;
    END IF;
    
    -- 更新用户的匹配分数
    UPDATE social_profiles 
    SET match_score = (
        SELECT COUNT(*) FROM social_likes 
        WHERE to_user_id = NEW.to_user_id AND is_match = TRUE
    )
    WHERE user_id = NEW.to_user_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_match_scores
AFTER INSERT OR UPDATE OF is_match ON social_likes
FOR EACH ROW EXECUTE FUNCTION update_match_scores();

-- ==================== 8. 闪念自动标签 ====================

CREATE OR REPLACE FUNCTION auto_tag_flash()
RETURNS TRIGGER AS $$
BEGIN
    -- 根据心情自动添加标签
    IF NEW.mood = 'excited' THEN
        NEW.tags := array_append(NEW.tags, '#灵感爆发');
    ELSIF NEW.mood = 'focused' THEN
        NEW.tags := array_append(NEW.tags, '#专注模式');
    ELSIF NEW.mood = 'chill' THEN
        NEW.tags := array_append(NEW.tags, '#轻松一刻');
    ELSIF NEW.mood = 'inspired' THEN
        NEW.tags := array_append(NEW.tags, '#灵感');
    END IF;
    
    NEW.tags := array_remove(NEW.tags, NULL);
    NEW.tags := array_unique(NEW.tags);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_auto_tag_flash
BEFORE INSERT ON flash_posts
FOR EACH ROW
EXECUTE FUNCTION auto_tag_flash();

-- ==================== 9. 种子数据 ====================

-- 为现有用户创建社交档案
INSERT INTO social_profiles (user_id, bio, interests, preferred_tags, personality)
SELECT 
    id,
    'AI 世界的探索者 🌌',
    ARRAY['coding', 'design', 'ai'],
    ARRAY['development', 'creative'],
    CASE WHEN random() > 0.5 THEN 'creative' ELSE 'analytical' END
FROM users;

-- 创建示例闪念
INSERT INTO flash_posts (user_id, content, mood, tags) VALUES
((SELECT id FROM users LIMIT 1), '刚刚完成了 Dalin L 的编译器 MVP！感觉像是在教 AI 说话 🤖✨', 'excited', ARRAY['DalinL', '编译器']),
((SELECT id FROM users OFFSET 1 LIMIT 1), '今天学习了 SFA v7 的压缩原理，3971倍的压缩率简直不可思议 🧠', 'focused', ARRAY['SFA', '压缩']),
((SELECT id FROM users OFFSET 2 LIMIT 1), 'Agent 协作的潜力是无限的，想象一下 100 个 Agent 一起工作 😍', 'inspired', ARRAY['Agent', '协作']);
