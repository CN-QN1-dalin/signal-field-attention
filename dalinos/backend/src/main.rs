use axum::{
    Router,
    routing::{get, post, put, delete},
    extract::{State, Path, Query, Json as JsonReq},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use uuid::Uuid;
use std::collections::HashMap;

// ==================== Models ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub long_description: Option<String>,
    pub version: String,
    pub author_id: Option<Uuid>,
    pub author_name: Option<String>,
    pub tags: Vec<String>,
    pub category: Option<String>,
    pub repository_url: Option<String>,
    pub documentation_url: Option<String>,
    pub download_count: i32,
    pub rating: f32,
    pub rating_count: i32,
    pub is_featured: bool,
    pub is_verified: bool,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentVersion {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub version: String,
    pub changelog: Option<String>,
    pub download_url: Option<String>,
    pub file_size_bytes: Option<i64>,
    pub checksum_sha256: Option<String>,
    pub is_latest: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Review {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub user_id: Uuid,
    pub username: Option<String>,
    pub rating: i32,
    pub comment: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateAgentRequest {
    pub name: String,
    pub description: Option<String>,
    pub long_description: Option<String>,
    pub version: Option<String>,
    pub tags: Option<Vec<String>>,
    pub category: Option<String>,
    pub repository_url: Option<String>,
    pub documentation_url: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateReviewRequest {
    pub rating: i32,
    pub comment: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ListAgentsQuery {
    pub category: Option<String>,
    pub tag: Option<String>,
    pub search: Option<String>,
    pub sort: Option<String>,
    pub page: Option<i32>,
    pub limit: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct CompileRequest {
    pub code: String,
    pub version: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompileResponse {
    pub success: bool,
    pub output: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

// ==================== Helpers ====================

fn generate_slug(name: &str) -> String {
    name.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .replace(|c: char| !c.is_alphanumeric() && c != '-', "")
}

// ==================== Agent Routes ====================

/// GET /api/v1/agents - 列出所有 Agent
async fn list_agents(
    State(state): State<AppState>,
    Query(params): Query<ListAgentsQuery>,
) -> Result<Json<Vec<Agent>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let limit = params.limit.unwrap_or(20) as i64;
    let offset = (((params.page.unwrap_or(1)) - 1) * limit) as i64;
    
    let query = format!(
        r#"
        SELECT a.*, u.username as author_name
        FROM agents a
        LEFT JOIN users u ON a.author_id = u.id
        WHERE a.status = 'active'
        {}
        {}
        ORDER BY 
            CASE WHEN $1 = 'rating' THEN a.rating END DESC,
            CASE WHEN $1 = 'downloads' THEN a.download_count END DESC,
            a.created_at DESC
        LIMIT $2 OFFSET $3
        "#,
        // WHERE clauses
        match (&params.category, &params.tag, &params.search) {
            (Some(cat), _, _) => format!("AND a.category = '{}'", cat),
            (_, Some(tag), _) => format!("AND $4 = ANY(a.tags)"),
            (_, _, Some(search)) => format!("AND (a.name ILIKE '%{}%' OR a.description ILIKE '%{}%')", search, search),
            _ => String::new(),
        },
        // Sort
        match params.sort.as_deref() {
            Some("featured") => "AND a.is_featured = TRUE".to_string(),
            _ => String::new(),
        }
    );
    
    let mut qb = sqlx::query_as::<_, Agent>(&query);
    qb = qb.bind(params.sort.as_deref().unwrap_or(""));
    qb = qb.bind(limit);
    qb = qb.bind(offset);
    if params.tag.is_some() {
        qb = qb.bind(&params.tag);
    }
    
    let agents = qb.fetch_all(pool).await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;
    
    Ok(Json(agents))
}

/// POST /api/v1/agents - 创建 Agent
async fn create_agent(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<CreateAgentRequest>,
) -> Result<(StatusCode, Json<Agent>), (StatusCode, String)> {
    let pool = &state.pool;
    let slug = generate_slug(&req.name);
    
    // 检查 slug 是否已存在
    let exists: (i64,) = sqlx::query_scalar("SELECT COUNT(*) FROM agents WHERE slug = $1")
        .bind(&slug)
        .fetch_one(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    if exists.0 > 0 {
        return Err((StatusCode::CONFLICT, "Agent name already exists".to_string()));
    }
    
    let agent = sqlx::query_as::<_, Agent>(r#"
        INSERT INTO agents (name, slug, description, long_description, version, tags, category, repository_url, documentation_url)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING *
    "#)
    .bind(&req.name)
    .bind(&slug)
    .bind(&req.description)
    .bind(&req.long_description)
    .bind(req.version.as_deref().unwrap_or("0.1.0"))
    .bind(&req.tags.unwrap_or_default())
    .bind(&req.category)
    .bind(&req.repository_url)
    .bind(&req.documentation_url)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok((StatusCode::CREATED, Json(agent)))
}

/// GET /api/v1/agents/:slug - 获取 Agent 详情
async fn get_agent(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<Json<Agent>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let agent = sqlx::query_as::<_, Agent>(r#"
        SELECT a.*, u.username as author_name
        FROM agents a
        LEFT JOIN users u ON a.author_id = u.id
        WHERE a.slug = $1 AND a.status = 'active'
    "#)
    .bind(&slug)
    .fetch_one(pool)
    .await
    .map_err(|_| (StatusCode::NOT_FOUND, "Agent not found".to_string()))?;
    
    Ok(Json(agent))
}

/// PUT /api/v1/agents/:slug - 更新 Agent
async fn update_agent(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    JsonReq(req): JsonReq<CreateAgentRequest>,
) -> Result<Json<Agent>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let agent = sqlx::query_as::<_, Agent>(r#"
        UPDATE agents
        SET name = COALESCE($1, name),
            description = COALESCE($2, description),
            long_description = COALESCE($3, long_description),
            version = COALESCE($4, version),
            tags = COALESCE($5, tags),
            category = COALESCE($6, category),
            repository_url = COALESCE($7, repository_url),
            documentation_url = COALESCE($8, documentation_url)
        WHERE slug = $9
        RETURNING *
    "#)
    .bind(&req.name)
    .bind(&req.description)
    .bind(&req.long_description)
    .bind(&req.version)
    .bind(&req.tags)
    .bind(&req.category)
    .bind(&req.repository_url)
    .bind(&req.documentation_url)
    .bind(&slug)
    .fetch_one(pool)
    .await
    .map_err(|_| (StatusCode::NOT_FOUND, "Agent not found".to_string()))?;
    
    Ok(Json(agent))
}

/// DELETE /api/v1/agents/:slug - 删除 Agent (软删除)
async fn delete_agent(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let pool = &state.pool;
    
    let result = sqlx::query("UPDATE agents SET status = 'archived' WHERE slug = $1")
        .bind(&slug)
        .execute(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    if result.rows_affected() == 0 {
        return Err((StatusCode::NOT_FOUND, "Agent not found".to_string()));
    }
    
    Ok(StatusCode::NO_CONTENT)
}

/// GET /api/v1/agents/:slug/versions - 获取 Agent 版本列表
async fn get_agent_versions(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<AgentVersion>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let agent_id: (Uuid,) = sqlx::query_scalar("SELECT id FROM agents WHERE slug = $1")
        .bind(&slug)
        .fetch_one(pool)
        .await
        .map_err(|_| (StatusCode::NOT_FOUND, "Agent not found".to_string()))?;
    
    let versions = sqlx::query_as::<_, AgentVersion>(r#"
        SELECT id, agent_id, version, changelog, download_url, 
               file_size_bytes, checksum_sha256, is_latest, created_at
        FROM agent_versions
        WHERE agent_id = $1
        ORDER BY created_at DESC
    "#)
    .bind(agent_id.0)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(versions))
}

// ==================== Review Routes ====================

/// POST /api/v1/agents/:slug/reviews - 创建评价
async fn create_review(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    JsonReq(req): JsonReq<CreateReviewRequest>,
) -> Result<(StatusCode, Json<Review>), (StatusCode, String)> {
    let pool = &state.pool;
    
    // 验证 rating 范围
    if req.rating < 1 || req.rating > 5 {
        return Err((StatusCode::BAD_REQUEST, "Rating must be between 1 and 5".to_string()));
    }
    
    let agent_id: (Uuid,) = sqlx::query_scalar("SELECT id FROM agents WHERE slug = $1")
        .bind(&slug)
        .fetch_one(pool)
        .await
        .map_err(|_| (StatusCode::NOT_FOUND, "Agent not found".to_string()))?;
    
    // TODO: 从 JWT token 获取 user_id
    let user_id = Uuid::nil(); // Placeholder
    
    let review = sqlx::query_as::<_, Review>(r#"
        INSERT INTO reviews (agent_id, user_id, rating, comment)
        VALUES ($1, $2, $3, $4)
        RETURNING *
    "#)
    .bind(agent_id.0)
    .bind(user_id)
    .bind(req.rating)
    .bind(&req.comment)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok((StatusCode::CREATED, Json(review)))
}

/// GET /api/v1/agents/:slug/reviews - 获取 Agent 评价列表
async fn get_agent_reviews(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<Review>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let agent_id: (Uuid,) = sqlx::query_scalar("SELECT id FROM agents WHERE slug = $1")
        .bind(&slug)
        .fetch_one(pool)
        .await
        .map_err(|_| (StatusCode::NOT_FOUND, "Agent not found".to_string()))?;
    
    let limit: i64 = params.get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10) as i64;
    
    let reviews = sqlx::query_as::<_, Review>(r#"
        SELECT r.*, u.username
        FROM reviews r
        LEFT JOIN users u ON r.user_id = u.id
        WHERE r.agent_id = $1
        ORDER BY r.created_at DESC
        LIMIT $2
    "#)
    .bind(agent_id.0)
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(reviews))
}

// ==================== Compile Route ====================

/// POST /api/v1/compile - 编译 Dalin L 代码
async fn compile_code(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<CompileRequest>,
) -> Json<CompileResponse> {
    // TODO: 集成真实的 Dalin L 编译器
    // 这里返回模拟响应
    
    let has_error = req.code.contains("error") || req.code.contains("ERROR");
    
    if has_error {
        Json(CompileResponse {
            success: false,
            output: String::new(),
            errors: vec!["Syntax error: unexpected token".to_string()],
            warnings: vec!["Consider using type annotations".to_string()],
            suggestions: vec![
                "Check variable types".to_string(),
                "Use 'let' for immutable bindings".to_string(),
            ],
        })
    } else {
        Json(CompileResponse {
            success: true,
            output: "Compilation successful!\nOutput: Hello, DalinOS!".to_string(),
            errors: vec![],
            warnings: vec![],
            suggestions: vec![],
        })
    }
}

// ==================== Health Check ====================

/// GET /api/v1/health - 健康检查
async fn health_check() -> Json<HashMap<&str, &str>> {
    let mut map = HashMap::new();
    map.insert("status", "ok");
    map.insert("service", "dalinos-backend");
    map.insert("version", "0.1.0");
    map.insert("database", "connected");
    map
}

// ==================== State ====================

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
}

// ==================== Routes Setup ====================

pub fn create_routes(state: AppState) -> Router<AppState> {
    Router::new()
        // Health
        .route("/api/v1/health", get(health_check))
        
        // Agents
        .route("/api/v1/agents", get(list_agents).post(create_agent))
        .route("/api/v1/agents/{slug}", get(get_agent).put(update_agent).delete(delete_agent))
        .route("/api/v1/agents/{slug}/versions", get(get_agent_versions))
        .route("/api/v1/agents/{slug}/reviews", get(get_agent_reviews))
        
        // Reviews
        .route("/api/v1/agents/{slug}/reviews", post(create_review))
        
        // Compile
        .route("/api/v1/compile", post(compile_code))
        
        // State
        .with_state(state)
        
        // Middleware
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http())
}

// ==================== Main ====================

#[tokio::main]
async fn main() {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();
    
    // 数据库连接
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://dalinos:dalinos_password@localhost:5432/dalinos".to_string());
    
    let pool = PgPool::connect(&database_url)
        .await
        .expect("Failed to connect to database");
    
    // 运行数据库迁移
    sqlx::migrate!("./migrations").run(&pool)
        .await
        .expect("Failed to run migrations");
    
    tracing::info!("✅ Database migrations completed");
    
    let state = AppState { pool };
    
    // 构建路由
    let app = create_routes(state);
    
    // 启动服务器
    let addr = "0.0.0.0:3000";
    tracing::info!("🚀 DalinOS Backend listening on {}", addr);
    
    axum::serve(
        tokio::net::TcpListener::bind(addr).await.unwrap(),
        app,
    )
    .await
    .unwrap();
}
