-- ==================== Agent 论坛 + 实时聊天群 ====================

-- ==================== 1. 论坛分类 ====================
CREATE TABLE forum_categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    icon VARCHAR(10) DEFAULT '💬',
    sort_order INT DEFAULT 0,
    thread_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==================== 2. 论坛帖子 ====================
CREATE TABLE forum_threads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    category_id UUID NOT NULL REFERENCES forum_categories(id) ON DELETE CASCADE,
    author_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(300) NOT NULL,
    content TEXT NOT NULL,
    views INT DEFAULT 0,
    replies INT DEFAULT 0,
    likes INT DEFAULT 0,
    is_pinned BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'closed', 'archived')),
    last_reply_at TIMESTAMP WITH TIME ZONE,
    last_reply_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_threads_category ON forum_threads(category_id);
CREATE INDEX idx_threads_author ON forum_threads(author_id);
CREATE INDEX idx_threads_views ON forum_threads(views DESC);
CREATE INDEX idx_threads_replies ON forum_threads(replies DESC);
CREATE INDEX idx_threads_created ON forum_threads(created_at DESC);
CREATE INDEX idx_threads_pinned ON forum_threads(is_pinned DESC, created_at DESC);

-- ==================== 3. 论坛回复 ====================
CREATE TABLE forum_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID NOT NULL REFERENCES forum_threads(id) ON DELETE CASCADE,
    author_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    parent_post_id UUID REFERENCES forum_posts(id) ON DELETE CASCADE,
    likes INT DEFAULT 0,
    is_best_answer BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_posts_thread ON forum_posts(thread_id);
CREATE INDEX idx_posts_author ON forum_posts(author_id);
CREATE INDEX idx_posts_parent ON forum_posts(parent_post_id);
CREATE INDEX idx_posts_created ON forum_posts(created_at ASC);

-- ==================== 4. 论坛标签 ====================
CREATE TABLE forum_tags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) NOT NULL UNIQUE,
    slug VARCHAR(50) NOT NULL UNIQUE,
    thread_count INT DEFAULT 0
);

CREATE TABLE forum_thread_tags (
    thread_id UUID NOT NULL REFERENCES forum_threads(id) ON DELETE CASCADE,
    tag_id UUID NOT NULL REFERENCES forum_tags(id) ON DELETE CASCADE,
    PRIMARY KEY (thread_id, tag_id)
);

-- ==================== 5. 聊天房间 ====================
CREATE TABLE chat_rooms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    max_members INT DEFAULT 200,
    is_public BOOLEAN DEFAULT TRUE,
    creator_id UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_rooms_creator ON chat_rooms(creator_id);
CREATE INDEX idx_rooms_slug ON chat_rooms(slug);

-- ==================== 6. 房间成员 ====================
CREATE TABLE room_members (
    room_id UUID NOT NULL REFERENCES chat_rooms(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_muted BOOLEAN DEFAULT FALSE,
    role VARCHAR(20) DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'moderator', 'member')),
    last_read_message_id UUID REFERENCES room_messages(id),
    PRIMARY KEY (room_id, user_id)
);

CREATE INDEX idx_members_room ON room_members(room_id);
CREATE INDEX idx_members_user ON room_members(user_id);

-- ==================== 7. 房间消息 ====================
CREATE TABLE room_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    room_id UUID NOT NULL REFERENCES chat_rooms(id) ON DELETE CASCADE,
    sender_id UUID NOT NULL REFERENCES users(id),
    content TEXT NOT NULL,
    is_system BOOLEAN DEFAULT FALSE,
    reply_to_id UUID REFERENCES room_messages(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_msgs_room ON room_messages(room_id, created_at DESC);
CREATE INDEX idx_msgs_sender ON room_messages(sender_id);

-- ==================== 8. 在线状态 ====================
CREATE TABLE user_presence (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'offline' CHECK (status IN ('online', 'away', 'busy', 'offline')),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    current_room_id UUID REFERENCES chat_rooms(id)
);

-- ==================== 触发器 ====================

-- 帖子回复数自动更新
CREATE OR REPLACE FUNCTION update_thread_replies()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE forum_threads SET 
            replies = replies + 1,
            last_reply_at = NEW.created_at,
            last_reply_by = NEW.author_id,
            updated_at = NOW()
        WHERE id = NEW.thread_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE forum_threads SET 
            replies = GREATEST(replies - 1, 0),
            updated_at = NOW()
        WHERE id = NEW.thread_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_thread_replies
AFTER INSERT OR DELETE ON forum_posts
FOR EACH ROW
EXECUTE FUNCTION update_thread_replies();

-- 帖子浏览量更新
CREATE OR REPLACE FUNCTION increment_thread_views()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE forum_threads SET views = views + 1 WHERE id = NEW.thread_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_increment_views
AFTER INSERT ON forum_posts
FOR EACH STATEMENT
EXECUTE FUNCTION increment_thread_views();

-- ==================== 种子数据 ====================

-- 论坛分类
INSERT INTO forum_categories (name, slug, description, icon, sort_order) VALUES
('灵感碰撞', 'inspiration', '分享你的创意和灵感火花', '💡', 1),
('技术讨论', 'tech', '编程、架构、算法等技术话题', '💻', 2),
('Agent 评测', 'reviews', 'Agent 使用体验和评测', '🤖', 3),
('新手指南', 'guides', '帮助新用户快速上手', '📚', 4),
('闲聊灌水', 'off-topic', '随便聊聊，放松一下', '☕', 5);

-- 聊天房间
INSERT INTO chat_rooms (name, slug, description, max_members, is_public, creator_id) VALUES
('灵感碰撞室', 'inspiration-room', '在这里碰撞创意火花 🔥', 50, TRUE, (SELECT id FROM users LIMIT 1)),
('技术讨论区', 'tech-chat', '编程和技术话题深度交流 💻', 100, TRUE, (SELECT id FROM users LIMIT 1)),
('Agent 交流群', 'agent-chat', 'Agent 开发者聚集地 🤖', 200, TRUE, (SELECT id FROM users LIMIT 1)),
('深夜食堂', 'late-night', '凌晨三点的哲学讨论 ☕', 30, TRUE, (SELECT id FROM users LIMIT 1));

-- 论坛标签
INSERT INTO forum_tags (name, slug, thread_count) VALUES
('DalinL', 'dalinel', 0),
('SFA', 'sfa', 0),
('Agent', 'agent', 0),
('编程', 'programming', 0),
('设计', 'design', 0),
('AI', 'ai', 0),
('教程', 'tutorial', 0),
('讨论', 'discussion', 0);
