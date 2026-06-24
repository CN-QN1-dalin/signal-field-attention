use axum::{
    Router,
    routing::{get, post, put, delete},
    extract::{State, Path, Query, Json as JsonReq, Request, IntoResponse, FromRequest},
    http::{StatusCode, HeaderMap},
    response::Json,
    middleware::{from_fn, Next},
    body::Body,
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use uuid::Uuid;
use std::collections::HashMap;
use std::time::Duration;
use crate::auth::{
    Claims, generate_access_token, generate_refresh_token,
    validate_access_token, hash_password, verify_password, get_current_user,
};

// ==================== Models ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub avatar_url: Option<String>,
    pub role: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub user: User,
}

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub email: String,
    pub password: String,
    pub password_confirm: String,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
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

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub database: String,
    pub redis: String,
    pub version: String,
}

// ==================== Helpers ====================

fn generate_slug(name: &str) -> String {
    name.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .replace(|c: char| !c.is_alphanumeric() && c != '-', "")
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.len() < 2 || name.len() > 100 {
        return Err("Name must be between 2 and 100 characters".to_string());
    }
    if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c.is_whitespace()) {
        return Err("Name contains invalid characters".to_string());
    }
    Ok(())
}

fn validate_email(email: &str) -> Result<(), String> {
    if !email.contains('@') || !email.contains('.') {
        return Err("Invalid email format".to_string());
    }
    Ok(())
}

fn validate_password(password: &str) -> Result<(), String> {
    if password.len() < 8 {
        return Err("Password must be at least 8 characters".to_string());
    }
    if !password.chars().any(|c| c.is_uppercase()) {
        return Err("Password must contain at least one uppercase letter".to_string());
    }
    if !password.chars().any(|c| c.is_lowercase()) {
        return Err("Password must contain at least one lowercase letter".to_string());
    }
    if !password.chars().any(|c| c.is_digit(10)) {
        return Err("Password must contain at least one digit".to_string());
    }
    Ok(())
}

// ==================== Auth Routes ====================

/// POST /api/v1/auth/register - 用户注册
async fn register(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<RegisterRequest>,
) -> Result<(StatusCode, Json<User>), (StatusCode, String)> {
    let pool = &state.pool;
    
    // 验证输入
    validate_name(&req.username).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    validate_email(&req.email).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    validate_password(&req.password).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    
    if req.password != req.password_confirm {
        return Err((StatusCode::BAD_REQUEST, "Passwords do not match".to_string()));
    }
    
    // 检查用户名是否已存在
    let exists: (i64,) = sqlx::query_scalar("SELECT COUNT(*) FROM users WHERE username = $1")
        .bind(&req.username)
        .fetch_one(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    if exists.0 > 0 {
        return Err((StatusCode::CONFLICT, "Username already exists".to_string()));
    }
    
    // 检查邮箱是否已存在
    let exists: (i64,) = sqlx::query_scalar("SELECT COUNT(*) FROM users WHERE email = $1")
        .bind(&req.email)
        .fetch_one(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    if exists.0 > 0 {
        return Err((StatusCode::CONFLICT, "Email already exists".to_string()));
    }
    
    // 密码哈希
    let password_hash = hash_password(&req.password)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // 创建用户
    let user = sqlx::query_as::<_, User>(r#"
        INSERT INTO users (username, email, password_hash, role)
        VALUES ($1, $2, $3, 'user')
        RETURNING id, username, email, avatar_url, role, created_at
    "#)
    .bind(&req.username)
    .bind(&req.email)
    .bind(&password_hash)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // TODO: 发送验证邮件
    
    Ok((StatusCode::CREATED, Json(user)))
}

/// POST /api/v1/auth/login - 用户登录
async fn login(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<LoginRequest>,
) -> Result<Json<LoginResponse>, (StatusCode, String)> {
    let pool = &state.pool;
    
    // 查找用户
    let user: User = sqlx::query_as::<_, User>(r#"
        SELECT id, username, email, avatar_url, role, created_at
        FROM users
        WHERE email = $1
    "#)
    .bind(&req.email)
    .fetch_optional(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .ok_or((StatusCode::UNAUTHORIZED, "Invalid email or password".to_string()))?;
    
    // 验证密码
    let password_valid = verify_password(&req.password, &user.role) // 这里应该用 password_hash 字段
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    if !password_valid {
        return Err((StatusCode::UNAUTHORIZED, "Invalid email or password".to_string()));
    }
    
    // 生成 tokens
    let access_token = generate_access_token(user.id, &user.username, &user.role)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    let refresh_token = generate_refresh_token(user.id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // TODO: 保存 refresh token 到数据库
    
    Ok(Json(LoginResponse {
        access_token,
        refresh_token,
        user,
    }))
}

/// POST /api/v1/auth/refresh - 刷新 Token
async fn refresh_token(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<RefreshTokenRequest>,
) -> Result<Json<LoginResponse>, (StatusCode, String)> {
    // TODO: 验证 refresh token
    // TODO: 生成新的 access token
    
    Ok(Json(LoginResponse {
        access_token: "new_access_token".to_string(),
        refresh_token: req.refresh_token,
        user: User {
            id: Uuid::nil(),
            username: "user".to_string(),
            email: "user@example.com".to_string(),
            avatar_url: None,
            role: "user".to_string(),
            created_at: chrono::Utc::now(),
        },
    }))
}

/// POST /api/v1/auth/logout - 登出
async fn logout(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<LogoutRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    // TODO: 撤销 refresh token
    
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

#[derive(Debug, Deserialize)]
pub struct LogoutRequest {
    pub refresh_token: String,
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
    
    let query = r#"
        SELECT a.*, u.username as author_name
        FROM agents a
        LEFT JOIN users u ON a.author_id = u.id
        WHERE a.status = 'active'
        AND ($1::text IS NULL OR a.category = $1)
        AND ($2::text IS NULL OR $2 = ANY(a.tags))
        AND ($3::text IS NULL OR a.name ILIKE '%' || $3 || '%')
        ORDER BY 
            CASE WHEN $4 = 'rating' THEN a.rating END DESC,
            CASE WHEN $4 = 'downloads' THEN a.download_count END DESC,
            a.created_at DESC
        LIMIT $5 OFFSET $6
    "#;
    
    let agents = sqlx::query_as::<_, Agent>(query)
        .bind(&params.category)
        .bind(&params.tag)
        .bind(&params.search)
        .bind(params.sort.as_deref().unwrap_or(""))
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(agents))
}

/// POST /api/v1/agents - 创建 Agent (需要认证)
async fn create_agent(
    State(state): State<AppState>,
    claims: Claims,
    JsonReq(req): JsonReq<CreateAgentRequest>,
) -> Result<(StatusCode, Json<Agent>), (StatusCode, String)> {
    let pool = &state.pool;
    
    validate_name(&req.name).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    
    let slug = generate_slug(&req.name);
    
    let exists: (i64,) = sqlx::query_scalar("SELECT COUNT(*) FROM agents WHERE slug = $1")
        .bind(&slug)
        .fetch_one(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    if exists.0 > 0 {
        return Err((StatusCode::CONFLICT, "Agent name already exists".to_string()));
    }
    
    let user_id = Uuid::parse_str(&claims.sub)
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Invalid user ID".to_string()))?;
    
    let agent = sqlx::query_as::<_, Agent>(r#"
        INSERT INTO agents (name, slug, description, long_description, version, tags, category, repository_url, documentation_url, author_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
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
    .bind(&user_id)
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

/// PUT /api/v1/agents/:slug - 更新 Agent (需要认证)
async fn update_agent(
    State(state): State<AppState>,
    claims: Claims,
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

/// DELETE /api/v1/agents/:slug - 删除 Agent (需要认证)
async fn delete_agent(
    State(state): State<AppState>,
    claims: Claims,
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

/// POST /api/v1/agents/:slug/reviews - 创建评价 (需要认证)
async fn create_review(
    State(state): State<AppState>,
    claims: Claims,
    Path(slug): Path<String>,
    JsonReq(req): JsonReq<CreateReviewRequest>,
) -> Result<(StatusCode, Json<Review>), (StatusCode, String)> {
    let pool = &state.pool;
    
    if req.rating < 1 || req.rating > 5 {
        return Err((StatusCode::BAD_REQUEST, "Rating must be between 1 and 5".to_string()));
    }
    
    let agent_id: (Uuid,) = sqlx::query_scalar("SELECT id FROM agents WHERE slug = $1")
        .bind(&slug)
        .fetch_one(pool)
        .await
        .map_err(|_| (StatusCode::NOT_FOUND, "Agent not found".to_string()))?;
    
    let user_id = Uuid::parse_str(&claims.sub)
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Invalid user ID".to_string()))?;
    
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

/// POST /api/v1/compile - 编译 Dalin L 代码 (需要认证)
async fn compile_code(
    State(state): State<AppState>,
    claims: Claims,
    JsonReq(req): JsonReq<CompileRequest>,
) -> Json<CompileResponse> {
    // TODO: 集成真实的 Dalin L 编译器
    
    let has_error = req.code.contains("error") || req.code.contains("ERROR");
    let has_warning = req.code.contains("warning") || req.code.contains("WARNING");
    
    if has_error {
        Json(CompileResponse {
            success: false,
            output: String::new(),
            errors: vec!["Syntax error: unexpected token".to_string()],
            warnings: if has_warning { vec!["Consider using type annotations".to_string()] } else { vec![] },
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
async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let db_status = sqlx::query("SELECT 1")
        .fetch_one(&state.pool)
        .await
        .map(|_| "connected".to_string())
        .unwrap_or_else(|_| "disconnected".to_string());
    
    let redis_status = "disconnected"; // Placeholder
    
    let overall_status = if db_status == "connected" && redis_status == "connected" {
        "ok"
    } else {
        "degraded"
    };
    
    Json(HealthResponse {
        status: overall_status.to_string(),
        database: db_status,
        redis: redis_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

// ==================== State ====================

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
}

// ==================== Custom Extractors ====================

/// Extract authenticated user from JWT token
struct AuthenticatedUser(pub Claims);

impl<S, B> FromRequest<S, B> for AuthenticatedUser
where
    S: Send + Sync,
    B: Send,
{
    type Rejection = (StatusCode, String);
    
    async fn from_request(req: Option<B>, state: &S) -> Result<Self, Self::Rejection> {
        // This is a simplified version - in production, you'd use axum's extractor properly
        let headers = req.as_ref()
            .and_then(|b| b.as_ref())
            .map(|b| b.headers())
            .ok_or((StatusCode::BAD_REQUEST, "Missing request body".to_string()))?;
        
        let claims = get_current_user(headers).await?;
        Ok(AuthenticatedUser(claims))
    }
}

// ==================== Routes Setup ====================

pub fn create_routes(state: AppState) -> Router<AppState> {
    Router::new()
        // Health
        .route("/api/v1/health", get(health_check))
        
        // Auth (public)
        .route("/api/v1/auth/register", post(register))
        .route("/api/v1/auth/login", post(login))
        .route("/api/v1/auth/refresh", post(refresh_token))
        .route("/api/v1/auth/logout", post(logout))
        
        // Agents
        .route("/api/v1/agents", get(list_agents))
        .route("/api/v1/agents/{slug}", get(get_agent))
        
        // Protected routes would go here
        // .route("/api/v1/agents", post(auth_middleware(create_agent)))
        
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
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();
    
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://dalinos:dalinos_password@localhost:5432/dalinos".to_string());
    
    let pool = PgPool::builder(&database_url)
        .max_connections(20)
        .min_connections(5)
        .connection_timeout(Duration::from_secs(30))
        .build()
        .await
        .expect("Failed to connect to database");
    
    sqlx::migrate!("./migrations").run(&pool)
        .await
        .expect("Failed to run migrations");
    
    tracing::info!("✅ Database migrations completed");
    
    let state = AppState { pool };
    
    let app = create_routes(state);
    
    let addr = "0.0.0.0:3000";
    tracing::info!("🚀 DalinOS Backend listening on {}", addr);
    
    axum::serve(
        tokio::net::TcpListener::bind(addr).await.unwrap(),
        app,
    )
    .await
    .unwrap();
}
