// Social Platform API Routes
// 灵光一现 · SparkMatch — 智能体交友平台
// 作者: 元宝 (前端专家) + 混元 (后端专家)
// 日期: 2026-06-24

use axum::{
    routing::{get, post, put},
    extract::{State, Path, Query, Json as JsonReq, ws::{WebSocket, WS}},
    response::IntoResponse,
    http::StatusCode,
    ws::Ws,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ==================== Models ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub id: Uuid,
    pub user_id: Uuid,
    pub username: String,
    pub bio: Option<String>,
    pub avatar_url: Option<String>,
    pub interests: Vec<String>,
    pub personality: String,
    pub availability: String,
    pub match_score: i32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub recommended_user_id: Uuid,
    pub username: String,
    pub bio: Option<String>,
    pub avatar_url: Option<String>,
    pub match_percentage: i32,
    pub shared_interests: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashPost {
    pub id: Uuid,
    pub user_id: Uuid,
    pub username: String,
    pub content: String,
    pub mood: Option<String>,
    pub tags: Vec<String>,
    pub likes_count: i32,
    pub replies_count: i32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub from_user_id: Uuid,
    pub from_username: String,
    pub content: String,
    pub is_read: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMatch {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub agent_name: String,
    pub agent_slug: String,
    pub compatibility_score: f32,
    pub match_reason: Option<String>,
    pub accepted: bool,
}

#[derive(Debug, Deserialize)]
pub struct CreateProfileRequest {
    pub bio: Option<String>,
    pub interests: Option<Vec<String>>,
    pub personality: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LikeRequest {
    pub to_user_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct FlashRequest {
    pub content: String,
    pub mood: Option<String>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct MessageRequest {
    pub to_user_id: Uuid,
    pub content: String,
}

// ==================== Handlers ====================

/// GET /api/v1/social/profiles - 浏览用户档案
pub async fn get_profiles(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<UserProfile>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let limit: i64 = params.get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20) as i64;
    let page: i64 = params.get("page")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1) as i64;
    let offset = (page - 1) * limit;
    
    let profiles = sqlx::query_as::<_, UserProfile>(r#"
        SELECT sp.id, sp.user_id, u.username, sp.bio, sp.avatar_url,
               sp.interests, sp.personality, sp.availability,
               sp.match_score, sp.created_at
        FROM social_profiles sp
        JOIN users u ON sp.user_id = u.id
        ORDER BY sp.match_score DESC, sp.updated_at DESC
        LIMIT $1 OFFSET $2
    "#)
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(profiles))
}

/// GET /api/v1/social/matches - 获取推荐匹配
pub async fn get_matches(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<MatchResult>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let uid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;
    
    let limit: i64 = params.get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10) as i64;
    
    // Use the SQL function we defined
    let matches = sqlx::query_as::<_, MatchResult>(r#"
        SELECT * FROM recommend_matches($1, $2)
    "#)
    .bind(uid)
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(matches))
}

/// POST /api/v1/social/likes - 表达喜欢
pub async fn like_profile(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    JsonReq(req): JsonReq<LikeRequest>,
) -> Result<Json<HashMap<String, bool>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let uid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;
    
    // Insert like
    sqlx::query(r#"
        INSERT INTO social_likes (from_user_id, to_user_id)
        VALUES ($1, $2)
        ON CONFLICT (from_user_id, to_user_id) DO NOTHING
    "#)
    .bind(uid)
    .bind(req.to_user_id)
    .execute(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // Check if it's a match
    let is_match: (bool,) = sqlx::query_scalar(r#"
        SELECT EXISTS (
            SELECT 1 FROM social_likes 
            WHERE from_user_id = $2 AND to_user_id = $1 AND is_match = TRUE
        )
    "#)
    .bind(req.to_user_id)
    .bind(uid)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    let mut result = HashMap::new();
    result.insert("liked".to_string(), true);
    result.insert("match".to_string(), is_match.0);
    
    Ok(Json(result))
}

/// POST /api/v1/social/flash - 发布闪念
pub async fn create_flash(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    JsonReq(req): JsonReq<FlashRequest>,
) -> Result<Json<FlashPost>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let uid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;
    
    if req.content.len() > 280 {
        return Err((StatusCode::BAD_REQUEST, "闪念不能超过 280 字".to_string()));
    }
    
    let post = sqlx::query_as::<_, FlashPost>(r#"
        INSERT INTO flash_posts (user_id, content, mood, tags)
        VALUES ($1, $2, $3, $4)
        RETURNING id, user_id, 
                  (SELECT username FROM users WHERE id = $1) as username,
                  content, mood, tags, likes_count, replies_count, created_at
    "#)
    .bind(uid)
    .bind(&req.content)
    .bind(&req.mood)
    .bind(&req.tags)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(post))
}

/// GET /api/v1/social/flash/feed - 闪念广场
pub async fn get_flash_feed(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<FlashPost>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let limit: i64 = params.get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20) as i64;
    
    let posts = sqlx::query_as::<_, FlashPost>(r#"
        SELECT fp.id, fp.user_id, u.username, fp.content, fp.mood, 
               fp.tags, fp.likes_count, fp.replies_count, fp.created_at
        FROM flash_posts fp
        JOIN users u ON fp.user_id = u.id
        ORDER BY fp.created_at DESC
        LIMIT $1
    "#)
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(posts))
}

/// GET /api/v1/social/messages - 获取消息
pub async fn get_messages(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<Vec<Message>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let uid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;
    
    let messages = sqlx::query_as::<_, Message>(r#"
        SELECT m.id, m.from_user_id, u.username as from_username,
               m.content, m.is_read, m.created_at
        FROM social_messages m
        JOIN users u ON m.from_user_id = u.id
        WHERE m.to_user_id = $1 OR m.from_user_id = $1
        ORDER BY m.created_at DESC
        LIMIT 50
    "#)
    .bind(uid)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // Mark as read
    sqlx::query("UPDATE social_messages SET is_read = TRUE WHERE to_user_id = $1 AND is_read = FALSE")
        .bind(uid)
        .execute(pool)
        .await
        .ok();
    
    Ok(Json(messages))
}

/// POST /api/v1/social/messages - 发送消息
pub async fn send_message(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    JsonReq(req): JsonReq<MessageRequest>,
) -> Result<Json<Message>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let uid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;
    
    let msg = sqlx::query_as::<_, Message>(r#"
        INSERT INTO social_messages (from_user_id, to_user_id, content)
        VALUES ($1, $2, $3)
        RETURNING id, from_user_id,
                  (SELECT username FROM users WHERE id = $1) as from_username,
                  content, is_read, created_at
    "#)
    .bind(uid)
    .bind(req.to_user_id)
    .bind(&req.content)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(msg))
}

/// GET /api/v1/social/agent-matches - Agent 推荐
pub async fn get_agent_matches(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<Vec<AgentMatch>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let uid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;
    
    let matches = sqlx::query_as::<_, AgentMatch>(r#"
        SELECT am.id, am.agent_id, a.name as agent_name, a.slug as agent_slug,
               am.compatibility_score, am.match_reason, am.accepted
        FROM agent_matches am
        JOIN agents a ON am.agent_id = a.id
        WHERE am.user_id = $1 AND am.accepted = FALSE
        ORDER BY am.compatibility_score DESC
    "#)
    .bind(uid)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(matches))
}

// ==================== WebSocket Chat ====================

/// WS /api/v1/social/chat/{user_id} - 实时聊天
pub async fn websocket_chat(
    ws: Ws,
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws(socket, state, user_id))
}

async fn handle_ws(socket: WebSocket, _state: AppState, _user_id: String) {
    use axum::ws::{Message, CloseFrame};
    
    let (mut sender, mut receiver) = socket.split();
    
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            // Echo back (in production, broadcast to recipients)
            let response = format!("Echo: {}", text);
            sender.send(Message::Text(response.into())).await.ok();
        }
    }
}

// ==================== Routes Setup ====================

pub fn social_routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/v1/social/profiles", get(get_profiles))
        .route("/api/v1/social/matches/:user_id", get(get_matches))
        .route("/api/v1/social/likes", post(like_profile))
        .route("/api/v1/social/flash", post(create_flash))
        .route("/api/v1/social/flash/feed", get(get_flash_feed))
        .route("/api/v1/social/messages/:user_id", get(get_messages).post(send_message))
        .route("/api/v1/social/agent-matches/:user_id", get(get_agent_matches))
        .route("/api/v1/social/chat/{user_id}", get(websocket_chat))
}
