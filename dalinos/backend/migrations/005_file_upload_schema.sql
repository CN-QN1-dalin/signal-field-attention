-- File Upload & Storage Schema
-- 文件上传与存储

-- ==================== Agent Files ====================
CREATE TABLE agent_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    file_type VARCHAR(20) NOT NULL CHECK (file_type IN ('avatar', 'screenshot', 'documentation', 'binary')),
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    checksum_sha256 VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agent_files_agent ON agent_files(agent_id);
CREATE INDEX idx_agent_files_type ON agent_files(file_type);

-- ==================== Upload Sessions ====================
CREATE TABLE upload_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    chunk_number INT NOT NULL,
    total_chunks INT NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_upload_sessions_user ON upload_sessions(user_id);
CREATE INDEX idx_upload_sessions_completed ON upload_sessions(completed);

-- ==================== Functions ====================

-- Validate file type
CREATE OR REPLACE FUNCTION validate_file_type(p_mime_type VARCHAR, p_max_size BIGINT)
RETURNS BOOLEAN AS $$
DECLARE
    allowed_types TEXT[] := ARRAY[
        'image/jpeg', 'image/png', 'image/webp',
        'application/pdf',
        'application/zip', 'application/gzip'
    ];
BEGIN
    -- Check MIME type
    IF p_mime_type = ANY(allowed_types) THEN
        -- Check file size (< 10MB)
        IF p_max_size <= 10485760 THEN
            RETURN TRUE;
        END IF;
    END IF;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Clean up old files
CREATE OR REPLACE FUNCTION clean_old_files(days_to_keep INT DEFAULT 30)
RETURNS VOID AS $$
BEGIN
    -- Delete upload sessions older than days_to_keep
    DELETE FROM upload_sessions 
    WHERE uploaded_at < NOW() - (days_to_keep || ' days')::INTERVAL;
    
    -- Delete orphaned agent files
    DELETE FROM agent_files 
    WHERE id NOT IN (SELECT DISTINCT agent_id FROM agent_files af2);
END;
$$ LANGUAGE plpgsql;
