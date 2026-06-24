// ==================== Agent 论坛 + 实时聊天群 API ====================
// 作者: Agnes-Flash (架构师)
// 日期: 2026-06-24

use axum::{
    routing::{get, post},
    extract::{State, Path, Json as JsonReq, Query},
    response::Json,
    http::StatusCode,
    ws::{WebSocket, Ws},
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ==================== Models ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForumCategory {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub icon: String,
    pub sort_order: i32,
    pub thread_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForumThread {
    pub id: Uuid,
    pub category_id: Uuid,
    pub category_name: String,
    pub author_id: Uuid,
    pub author_name: String,
    pub title: String,
    pub content_preview: String,
    pub views: i32,
    pub replies: i32,
    pub likes: i32,
    pub is_pinned: bool,
    pub tags: Vec<String>,
    pub last_reply_at: Option<chrono::DateTime<chrono::Utc>>,
    pub last_reply_by: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForumPost {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub author_id: Uuid,
    pub author_name: String,
    pub content: String,
    pub parent_post_id: Option<Uuid>,
    pub likes: i32,
    pub is_best_answer: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForumTag {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub thread_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRoom {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub max_members: i32,
    pub is_public: bool,
    pub member_count: i32,
    pub online_count: i32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomMessage {
    pub id: Uuid,
    pub room_id: Uuid,
    pub sender_id: Uuid,
    pub sender_name: String,
    pub content: String,
    pub is_system: bool,
    pub reply_to_id: Option<Uuid>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomMember {
    pub user_id: Uuid,
    pub username: String,
    pub role: String,
    pub joined_at: chrono::DateTime<chrono::Utc>,
    pub is_online: bool,
}

// ==================== Request Models ====================

#[derive(Debug, Deserialize)]
pub struct CreateThreadRequest {
    pub category_id: Uuid,
    pub title: String,
    pub content: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct CreatePostRequest {
    pub content: String,
    pub parent_post_id: Option<Uuid>,
}

#[derive(Debug, Deserialize)]
pub struct CreateRoomRequest {
    pub name: String,
    pub description: Option<String>,
    pub max_members: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub content: String,
    pub reply_to_id: Option<Uuid>,
}

// ==================== Handlers ====================

// ==================== Forum ====================

/// GET /api/v1/forum/categories - 获取分类列表
pub async fn get_categories(
    State(state): State<AppState>,
) -> Result<Json<Vec<ForumCategory>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let categories = sqlx::query_as::<_, ForumCategory>(r#"
        SELECT id, name, slug, description, icon, sort_order, thread_count
        FROM forum_categories
        ORDER BY sort_order ASC
    "#)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(categories))
}

/// GET /api/v1/forum/threads - 获取帖子列表
pub async fn get_threads(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<ForumThread>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let category = params.get("category");
    let tag = params.get("tag");
    let sort = params.get("sort").map(|s| s.as_str()).unwrap_or("latest");
    let page: i64 = params.get("page").and_then(|v| v.parse().ok()).unwrap_or(1);
    let limit: i64 = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(20);
    let offset = (page - 1) * limit;
    
    let mut query = r#"
        SELECT ft.id, ft.category_id, fc.name as category_name,
               ft.author_id, u.username as author_name,
               ft.title, LEFT(ft.content, 200) as content_preview,
               ft.views, ft.replies, ft.likes, ft.is_pinned,
               ft.last_reply_at, ft.last_reply_by,
               u2.username as last_reply_by_name,
               ft.created_at
        FROM forum_threads ft
        JOIN forum_categories fc ON ft.category_id = fc.id
        JOIN users u ON ft.author_id = u.id
        LEFT JOIN users u2 ON ft.last_reply_by = u2.id
        WHERE ft.status = 'active'
    "#.to_string();
    
    let mut binding = vec![];
    
    if let Some(cat) = category {
        query.push_str(" AND fc.slug = $").push((binding.len() + 1) as u8 + '0');
        binding.push(cat);
    }
    
    if let Some(t) = tag {
        query.push_str(" AND EXISTS (SELECT 1 FROM forum_thread_tags ftt JOIN forum_tags ftg ON ftt.tag_id = ftg.id WHERE ftt.thread_id = ft.id AND ftg.slug = $").push((binding.len() + 1) as u8 + '0');
        query.push(')');
        binding.push(t);
    }
    
    query.push_str(" ORDER BY ");
    match sort {
        "popular" => query.push_str("ft.views DESC"),
        "replies" => query.push_str("ft.replies DESC"),
        _ => query.push_str("ft.is_pinned DESC, ft.created_at DESC"),
    };
    
    query.push_str(" LIMIT $").push((binding.len() + 1) as u8 + '0');
    query.push_str(" OFFSET $").push((binding.len() + 2) as u8 + '0');
    
    let threads = sqlx::query(&query)
        .bind_values(binding.into_iter())
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // 转换为 ForumThread (简化处理)
    let threads: Vec<ForumThread> = threads.iter().map(|row| {
        ForumThread {
            id: row.get("id"),
            category_id: row.get("category_id"),
            category_name: row.get("category_name"),
            author_id: row.get("author_id"),
            author_name: row.get("author_name"),
            title: row.get("title"),
            content_preview: row.get("content_preview"),
            views: row.get("views"),
            replies: row.get("replies"),
            likes: row.get("likes"),
            is_pinned: row.get("is_pinned"),
            tags: vec![],
            last_reply_at: row.get("last_reply_at"),
            last_reply_by: row.get("last_reply_by_name"),
            created_at: row.get("created_at"),
        }
    }).collect();
    
    Ok(Json(threads))
}

/// POST /api/v1/forum/threads - 创建帖子
pub async fn create_thread(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<CreateThreadRequest>,
) -> Result<Json<ForumThread>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let thread = sqlx::query_as::<_, ForumThread>(r#"
        INSERT INTO forum_threads (category_id, author_id, title, content)
        VALUES ($1, (SELECT id FROM users LIMIT 1), $2, $3)
        RETURNING id, category_id, 
                  (SELECT name FROM forum_categories WHERE id = $1) as category_name,
                  author_id, 
                  (SELECT username FROM users WHERE id = author_id) as author_name,
                  title, LEFT(content, 200) as content_preview,
                  0 as views, 0 as replies, 0 as likes,
                  FALSE as is_pinned, NULL as last_reply_at, NULL as last_reply_by,
                  created_at
    "#)
    .bind(req.category_id)
    .bind(&req.title)
    .bind(&req.content)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(thread))
}

/// GET /api/v1/forum/threads/:id - 帖子详情
pub async fn get_thread_detail(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<(Json<ForumThread>, Json<Vec<ForumPost>>), (StatusCode, String)> {
    let pool = &state.pool;
    
    let tid = Uuid::parse_str(&thread_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid thread ID".to_string()))?;
    
    // 增加浏览量
    sqlx::query("UPDATE forum_threads SET views = views + 1 WHERE id = $1")
        .bind(tid)
        .execute(pool)
        .await
        .ok();
    
    // 获取帖子
    let thread = sqlx::query_as::<_, ForumThread>(r#"
        SELECT ft.id, ft.category_id, fc.name as category_name,
               ft.author_id, u.username as author_name,
               ft.title, ft.content as content_preview,
               ft.views, ft.replies, ft.likes, ft.is_pinned,
               ft.last_reply_at, u2.username as last_reply_by,
               ft.created_at
        FROM forum_threads ft
        JOIN forum_categories fc ON ft.category_id = fc.id
        JOIN users u ON ft.author_id = u.id
        LEFT JOIN users u2 ON ft.last_reply_by = u2.id
        WHERE ft.id = $1
    "#)
    .bind(tid)
    .fetch_optional(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .ok_or((StatusCode::NOT_FOUND, "Thread not found".to_string()))?;
    
    // 获取回复
    let posts = sqlx::query_as::<_, ForumPost>(r#"
        SELECT fp.id, fp.thread_id, fp.author_id, u.username as author_name,
               fp.content, fp.parent_post_id, fp.likes, fp.is_best_answer, fp.created_at
        FROM forum_posts fp
        JOIN users u ON fp.author_id = u.id
        WHERE fp.thread_id = $1 AND fp.parent_post_id IS NULL
        ORDER BY fp.created_at ASC
    "#)
    .bind(tid)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok((Json(thread), Json(posts)))
}

/// POST /api/v1/forum/threads/:id/posts - 回复帖子
pub async fn create_post(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    JsonReq(req): JsonReq<CreatePostRequest>,
) -> Result<Json<ForumPost>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let tid = Uuid::parse_str(&thread_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid thread ID".to_string()))?;
    
    let post = sqlx::query_as::<_, ForumPost>(r#"
        INSERT INTO forum_posts (thread_id, author_id, content, parent_post_id)
        VALUES ($1, (SELECT id FROM users LIMIT 1), $2, $3)
        RETURNING id, thread_id, author_id,
                  (SELECT username FROM users WHERE id = author_id) as author_name,
                  content, parent_post_id, 0 as likes, FALSE as is_best_answer, created_at
    "#)
    .bind(tid)
    .bind(&req.content)
    .bind(&req.parent_post_id)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(post))
}

/// GET /api/v1/forum/popular - 热门帖子
pub async fn get_popular_threads(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<ForumThread>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let limit: i64 = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(10) as i64;
    
    let threads = sqlx::query_as::<_, ForumThread>(r#"
        SELECT ft.id, ft.category_id, fc.name as category_name,
               ft.author_id, u.username as author_name,
               ft.title, LEFT(ft.content, 200) as content_preview,
               ft.views, ft.replies, ft.likes, ft.is_pinned,
               ft.last_reply_at, u2.username as last_reply_by,
               ft.created_at
        FROM forum_threads ft
        JOIN forum_categories fc ON ft.category_id = fc.id
        JOIN users u ON ft.author_id = u.id
        LEFT JOIN users u2 ON ft.last_reply_by = u2.id
        WHERE ft.status = 'active'
        ORDER BY (ft.views * 0.3 + ft.replies * 0.5 + ft.likes * 0.2) DESC
        LIMIT $1
    "#)
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(threads))
}

// ==================== Chat Rooms ====================

/// GET /api/v1/rooms - 获取房间列表
pub async fn get_rooms(
    State(state): State<AppState>,
) -> Result<Json<Vec<ChatRoom>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let rooms = sqlx::query_as::<_, ChatRoom>(r#"
        SELECT cr.id, cr.name, cr.slug, cr.description,
               cr.max_members, cr.is_public,
               (SELECT COUNT(*) FROM room_members rm WHERE rm.room_id = cr.id) as member_count,
               (SELECT COUNT(*) FROM user_presence up WHERE up.current_room_id = cr.id AND up.status = 'online') as online_count,
               cr.created_at
        FROM chat_rooms cr
        ORDER BY member_count DESC
    "#)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(rooms))
}

/// POST /api/v1/rooms - 创建房间
pub async fn create_room(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<CreateRoomRequest>,
) -> Result<Json<ChatRoom>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let slug = req.name.chars()
        .map(|c| if c.is_alphanumeric() { c.to_lowercase().next().unwrap() } else { '-' })
        .collect::<String>();
    
    let room = sqlx::query_as::<_, ChatRoom>(r#"
        INSERT INTO chat_rooms (name, slug, description, max_members, creator_id)
        VALUES ($1, $2, $3, $4, (SELECT id FROM users LIMIT 1))
        RETURNING id, name, slug, description, max_members, TRUE as is_public,
                  0 as member_count, 0 as online_count, created_at
    "#)
    .bind(&req.name)
    .bind(&slug)
    .bind(&req.description)
    .bind(req.max_members.unwrap_or(100))
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(room))
}

/// POST /api/v1/rooms/:slug/join - 加入房间
pub async fn join_room(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<Json<ChatRoom>, (StatusCode, String)> {
    let pool = &state.pool;
    
    sqlx::query(r#"
        INSERT INTO room_members (room_id, user_id)
        SELECT id, (SELECT id FROM users LIMIT 1)
        FROM chat_rooms WHERE slug = $1
        ON CONFLICT (room_id, user_id) DO NOTHING
    "#)
    .bind(&slug)
    .execute(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    let room = sqlx::query_as::<_, ChatRoom>(r#"
        SELECT cr.id, cr.name, cr.slug, cr.description,
               cr.max_members, cr.is_public,
               (SELECT COUNT(*) FROM room_members rm WHERE rm.room_id = cr.id) as member_count,
               (SELECT COUNT(*) FROM user_presence up WHERE up.current_room_id = cr.id AND up.status = 'online') as online_count,
               cr.created_at
        FROM chat_rooms cr WHERE cr.slug = $1
    "#)
    .bind(&slug)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(room))
}

/// GET /api/v1/rooms/:slug/messages - 获取消息历史
pub async fn get_room_messages(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<RoomMessage>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let limit: i64 = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50) as i64;
    
    let messages = sqlx::query_as::<_, RoomMessage>(r#"
        SELECT rm.id, rm.room_id, rm.sender_id, u.username as sender_name,
               rm.content, rm.is_system, rm.reply_to_id, rm.created_at
        FROM room_messages rm
        JOIN users u ON rm.sender_id = u.id
        JOIN chat_rooms cr ON rm.room_id = cr.id
        WHERE cr.slug = $1
        ORDER BY rm.created_at DESC
        LIMIT $2
    "#)
    .bind(&slug)
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // 反转顺序 (从旧到新)
    let mut messages = messages;
    messages.reverse();
    
    Ok(Json(messages))
}

/// POST /api/v1/rooms/:slug/messages - 发送消息
pub async fn send_room_message(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    JsonReq(req): JsonReq<SendMessageRequest>,
) -> Result<Json<RoomMessage>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let msg = sqlx::query_as::<_, RoomMessage>(r#"
        INSERT INTO room_messages (room_id, sender_id, content, reply_to_id)
        SELECT cr.id, (SELECT id FROM users LIMIT 1), $2, $3
        FROM chat_rooms cr WHERE cr.slug = $1
        RETURNING id, room_id, sender_id,
                  (SELECT username FROM users WHERE id = sender_id) as sender_name,
                  content, FALSE as is_system, reply_to_id, created_at
    "#)
    .bind(&slug)
    .bind(&req.content)
    .bind(&req.reply_to_id)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(msg))
}

/// GET /api/v1/rooms/:slug/members - 获取成员列表
pub async fn get_room_members(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<RoomMember>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let members = sqlx::query_as::<_, RoomMember>(r#"
        SELECT rm.user_id, u.username, rm.role, rm.joined_at,
               EXISTS(SELECT 1 FROM user_presence up WHERE up.user_id = rm.user_id AND up.status = 'online') as is_online
        FROM room_members rm
        JOIN users u ON rm.user_id = u.id
        JOIN chat_rooms cr ON rm.room_id = cr.id
        WHERE cr.slug = $1
        ORDER BY rm.role ASC, rm.joined_at ASC
    "#)
    .bind(&slug)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(members))
}

// ==================== WebSocket ====================

/// WS /ws/chat/{room_slug} - 实时聊天
pub async fn websocket_chat(
    ws: Ws,
    State(_state): State<AppState>,
    Path(room_slug): Path<String>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(|socket| handle_chat_ws(socket, room_slug))
}

async fn handle_chat_ws(socket: WebSocket, _room_slug: String) {
    use axum::ws::{Message, CloseFrame};
    let (mut sender, mut receiver) = socket.split();
    
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            // Echo back (production: broadcast to room)
            let response = format!("Echo: {}", text);
            sender.send(Message::Text(response.into())).await.ok();
        }
    }
}

// ==================== Routes Setup ====================

pub fn forum_routes() -> axum::Router<AppState> {
    axum::Router::new()
        // Forum
        .route("/api/v1/forum/categories", get(get_categories))
        .route("/api/v1/forum/threads", get(get_threads).post(create_thread))
        .route("/api/v1/forum/threads/:id", get(get_thread_detail))
        .route("/api/v1/forum/threads/:id/posts", post(create_post))
        .route("/api/v1/forum/popular", get(get_popular_threads))
        
        // Chat Rooms
        .route("/api/v1/rooms", get(get_rooms).post(create_room))
        .route("/api/v1/rooms/:slug/join", post(join_room))
        .route("/api/v1/rooms/:slug/messages", get(get_room_messages).post(send_room_message))
        .route("/api/v1/rooms/:slug/members", get(get_room_members))
        
        // WebSocket
        .route("/ws/chat/:slug", get(websocket_chat))
}
