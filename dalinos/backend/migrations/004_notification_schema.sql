-- Notification System Schema
-- 通知系统

-- ==================== Notifications ====================
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(30) NOT NULL CHECK (type IN (
        'agent_published',
        'agent_updated',
        'new_review',
        'review_reply',
        'agent_downloaded',
        'system_announcement',
        'token_expiring',
        'invite_accepted'
    )),
    title VARCHAR(200) NOT NULL,
    message TEXT,
    data JSONB DEFAULT '{}',  -- 额外数据，如 agent_id, review_id 等
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_notifications_user ON notifications(user_id, is_read);
CREATE INDEX idx_notifications_type ON notifications(type);
CREATE INDEX idx_notifications_created ON notifications(created_at DESC);

-- ==================== Notification Preferences ====================
CREATE TABLE notification_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    email_notifications BOOLEAN DEFAULT TRUE,
    in_app_notifications BOOLEAN DEFAULT TRUE,
    notify_agent_published BOOLEAN DEFAULT TRUE,
    notify_new_review BOOLEAN DEFAULT TRUE,
    notify_system_announcement BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==================== Functions ====================

-- Mark notification as read
CREATE OR REPLACE FUNCTION mark_notification_read(notif_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE notifications
    SET is_read = TRUE
    WHERE id = notif_id AND user_id = (SELECT user_id FROM notifications WHERE id = notif_id);
END;
$$ LANGUAGE plpgsql;

-- Get unread notification count
CREATE OR REPLACE FUNCTION get_unread_notification_count(p_user_id UUID)
RETURNS BIGINT AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM notifications WHERE user_id = p_user_id AND is_read = FALSE);
END;
$$ LANGUAGE plpgsql;

-- ==================== Triggers ====================

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_notification_preferences_updated_at
BEFORE UPDATE ON notification_preferences
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ==================== Seed Data ====================

-- Default preferences for existing users
INSERT INTO notification_preferences (user_id, email_notifications, in_app_notifications)
SELECT id, TRUE, TRUE FROM users;
