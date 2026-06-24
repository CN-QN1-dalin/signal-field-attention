use axum::{
    Router,
    routing::{get, post, put, delete},
    extract::{State, Path, Query},
    http::StatusCode,
    response::Json,
    Json as JsonResp,
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
    pub version: String,
    pub author_id: Option<Uuid>,
    pub tags: Vec<String>,
    pub category: Option<String>,
    pub download_count: i32,
    pub rating: f32,
    pub rating_count: i32,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateAgentRequest {
    pub name: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub author_id: Option<Uuid>,
    pub tags: Option<Vec<String>>,
    pub category: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateAgentRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub version: Option<String>,
    pub tags: Option<Vec<String>>,
    pub category: Option<String>,
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ListAgentsQuery {
    pub category: Option<String>,
    pub tag: Option<String>,
    pub sort: Option<String>,  // rating, downloads, created_at
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
    pub suggestions: Vec<String>,
}

// ==================== Database Helpers ====================

async fn get_pool(state: &AppState) -> Result<&PgPool, (StatusCode, String)> {
    Ok(&state.pool)
}

// ==================== Routes ====================

/// GET /api/v1/agents - 列出所有 Agent (支持分页和筛选)
async fn list_agents(
    State(state): State<AppState>,
    Query(params): Query<ListAgentsQuery>,
) -> Result<JsonResp<Vec<Agent>>, (StatusCode, String)> {
    let pool = get_pool(&state).await?;
    
    let limit = params.limit.unwrap_or(20) as i64;
    let offset = ((params.page.unwrap_or(1) - 1) * limit) as i64;
    
    let query = r#"
        SELECT id, name, slug, description, version, author_id, 
               tags, category, download_count, rating, rating_count, status,
               created_at, updated_at
        FROM agents
        WHERE status = 'active'
        ORDER BY 
            CASE WHEN $1 = 'rating' THEN rating END DESC,
            CASE WHEN $1 = 'downloads' THEN download_count END DESC,
            created_at DESC
        LIMIT $2 OFFSET $3
    "#;
    
    let agents = sqlx::query_as::<_, Agent>(query)
        .bind(params.sort.as_deref().unwrap_or(""))
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(JsonResp(agents))
}

/// POST /api/v1/agents - 创建 Agent
async fn create_agent(
    State(state): State<AppState>,
    JsonReq(req): JsonResp<CreateAgentRequest>,
) -> Result<(StatusCode, JsonResp<Agent>), (StatusCode, String)> {
    let pool = get_pool(&state).await?;
    
    // 生成 slug
    let slug = req.name.to_lowercase()
        .replace(' ', "-")
        .replace(|c: char| !c.is_alphanumeric() && c != '-', "");
    
    let agent = sqlx::query_as::<_, Agent>(r#"
        INSERT INTO agents (name, slug, description, version, author_id, tags, category)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING *
    "#)
    .bind(&req.name)
    .bind(&slug)
    .bind(&req.description)
    .bind(req.version.as_deref().unwrap_or("0.1.0"))
    .bind(req.author_id)
    .bind(&req.tags.unwrap_or_default())
    .bind(&req.category)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok((StatusCode::CREATED, JsonResp(agent)))
}

/// GET /api/v1/agents/:slug - 获取 Agent 详情
async fn get_agent(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<JsonResp<Agent>, (StatusCode, String)> {
    let pool = get_pool(&state).await?;
    
    let agent = sqlx::query_as::<_, Agent>(r#"
        SELECT id, name, slug, description, version, author_id,
               tags, category, download_count, rating, rating_count, status,
               created_at, updated_at
        FROM agents
        WHERE slug = $1 AND status = 'active'
    "#)
    .bind(&slug)
    .fetch_one(pool)
    .await
    .map_err(|_| (StatusCode::NOT_FOUND, "Agent not found".to_string()))?;
    
    Ok(JsonResp(agent))
}

/// PUT /api/v1/agents/:slug - 更新 Agent
async fn update_agent(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    JsonReq(req): JsonResp<UpdateAgentRequest>,
) -> Result<JsonResp<Agent>, (StatusCode, String)> {
    let pool = get_pool(&state).await?;
    
    let agent = sqlx::query_as::<_, Agent>(r#"
        UPDATE agents
        SET name = COALESCE($1, name),
            description = COALESCE($2, description),
            version = COALESCE($3, version),
            tags = COALESCE($4, tags),
            category = COALESCE($5, category),
            status = COALESCE($6, status)
        WHERE slug = $7
        RETURNING *
    "#)
    .bind(&req.name)
    .bind(&req.description)
    .bind(&req.version)
    .bind(&req.tags)
    .bind(&req.category)
    .bind(&req.status)
    .bind(&slug)
    .fetch_one(pool)
    .await
    .map_err(|_| (StatusCode::NOT_FOUND, "Agent not found".to_string()))?;
    
    Ok(JsonResp(agent))
}

/// DELETE /api/v1/agents/:slug - 删除 Agent
async fn delete_agent(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let pool = get_pool(&state).await?;
    
    let result = sqlx::query(r#"
        UPDATE agents SET status = 'archived' WHERE slug = $1
    "#)
    .bind(&slug)
    .execute(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    if result.rows_affected() == 0 {
        return Err((StatusCode::NOT_FOUND, "Agent not found".to_string()));
    }
    
    Ok(StatusCode::NO_CONTENT)
}

/// POST /api/v1/compile - 编译 Dalin L 代码
async fn compile_code(
    State(state): State<AppState>,
    JsonReq(req): JsonResp<CompileRequest>,
) -> JsonResp<CompileResponse> {
    // TODO: 集成真实的 Dalin L 编译器
    // 这里返回模拟响应
    
    let has_syntax_error = req.code.contains("error");
    
    if has_syntax_error {
        JsonResp(CompileResponse {
            success: false,
            output: String::new(),
            errors: vec!["Syntax error detected".to_string()],
            suggestions: vec!["Check your syntax".to_string()],
        })
    } else {
        JsonResp(CompileResponse {
            success: true,
            output: "Compilation successful!".to_string(),
            errors: vec![],
            suggestions: vec![],
        })
    }
}

/// GET /api/v1/health - 健康检查
async fn health_check() -> JsonResp<HashMap<&str, &str>> {
    let mut map = HashMap::new();
    map.insert("status", "ok");
    map.insert("service", "dalinos-backend");
    map.insert("version", "0.1.0");
    JsonResp(map)
}

// ==================== State ====================

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
}

// ==================== Main ====================

#[tokio::main]
async fn main() {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
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
    
    tracing::info!("Database migrations completed");
    
    let state = AppState { pool };
    
    // 构建路由
    let app = Router::new()
        // Health check
        .route("/api/v1/health", get(health_check))
        
        // Agent routes
        .route("/api/v1/agents", get(list_agents).post(create_agent))
        .route("/api/v1/agents/{slug}", get(get_agent).put(update_agent).delete(delete_agent))
        
        // Compile route
        .route("/api/v1/compile", post(compile_code))
        
        // State
        .with_state(state)
        
        // Middleware
        .layer(tower_http::cors::CorsLayer::permissive());
    
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
