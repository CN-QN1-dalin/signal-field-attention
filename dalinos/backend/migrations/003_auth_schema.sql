-- Authentication & User Management Schema
-- 用户认证系统

-- ==================== Refresh Tokens ====================
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(500) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    revoked_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_token ON refresh_tokens(token);
CREATE INDEX idx_refresh_tokens_expires ON refresh_tokens(expires_at);

-- ==================== Password Reset Tokens ====================
CREATE TABLE password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(100) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_password_reset_tokens_user ON password_reset_tokens(user_id);
CREATE INDEX idx_password_reset_tokens_token ON password_reset_tokens(token);
CREATE INDEX idx_password_reset_tokens_expires ON password_reset_tokens(expires_at);

-- ==================== Email Verification Tokens ====================
CREATE TABLE email_verification_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(100) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    verified_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_email_verification_tokens_user ON email_verification_tokens(user_id);
CREATE INDEX idx_email_verification_tokens_token ON email_verification_tokens(token);

-- ==================== Functions ====================

-- Generate secure token
CREATE OR REPLACE FUNCTION generate_secure_token(length INT DEFAULT 32)
RETURNS TEXT AS $$
DECLARE
    chars TEXT := 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    result TEXT := '';
    i INT := 0;
BEGIN
    WHILE i < length LOOP
        result := result || substring(chars from (random() * 61 + 1)::int for 1);
        i := i + 1;
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Clean expired tokens (run daily via cron)
CREATE OR REPLACE FUNCTION clean_expired_tokens()
RETURNS VOID AS $$
BEGIN
    -- Delete expired refresh tokens
    DELETE FROM refresh_tokens WHERE expires_at < NOW() OR revoked_at IS NOT NULL;
    
    -- Delete expired password reset tokens
    DELETE FROM password_reset_tokens WHERE expires_at < NOW() OR used_at IS NOT NULL;
    
    -- Delete expired email verification tokens
    DELETE FROM email_verification_tokens WHERE expires_at < NOW() OR verified_at IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- ==================== Seed Data ====================

-- Admin user with proper bcrypt hash (password: admin123)
INSERT INTO users (username, email, password_hash, role, created_at) VALUES
('admin', 'admin@dalinos.dev', '$2b$12$LQ3X3UpzMx67kdYDZNfjluFPjGq3jbsN.ARWiYJ0e8kqBPDC4E5iG', 'admin', NOW()),
('dalinos', 'dalinos@dalinos.dev', '$2b$12$LQ3X3UpzMx67kdYDZNfjluFPjGq3jbsN.ARWiYJ0e8kqBPDC4E5iG', 'developer', NOW());
