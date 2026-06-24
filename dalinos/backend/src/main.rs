use axum::{
    Router,
    routing::{get, post},
    extract::{State, Path},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

// ==================== Models ====================

#[derive(Debug, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub version: String,
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub download_count: i32,
    pub rating: f32,
}

#[derive(Debug, Deserialize)]
pub struct CreateAgentRequest {
    pub name: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub author: Option<String>,
    pub tags: Option<Vec<String>>,
}

// ==================== Routes ====================

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
}

// GET /api/v1/agents — 列出所有 Agent
async fn list_agents(State(state): State<AppState>) -> Json<Vec<Agent>> {
    // TODO: 从数据库查询
    let agents = vec![
        Agent {
            id: "1".to_string(),
            name: "Test Agent".to_string(),
            description: Some("A test agent".to_string()),
            version: "0.1.0".to_string(),
            author: Some("Dalin".to_string()),
            tags: vec!["test".to_string()],
            download_count: 0,
            rating: 0.0,
        },
    ];
    Json(agents)
}

// POST /api/v1/agents — 创建 Agent
async fn create_agent(
    State(state): State<AppState>,
    Json(req): Json<CreateAgentRequest>,
) -> Result<Json<Agent>, (StatusCode, String)> {
    // TODO: 验证输入
    // TODO: 插入数据库
    let agent = Agent {
        id: uuid::Uuid::new_v4().to_string(),
        name: req.name,
        description: req.description,
        version: req.version.unwrap_or_else(|| "0.1.0".to_string()),
        author: req.author,
        tags: req.tags.unwrap_or_default(),
        download_count: 0,
        rating: 0.0,
    };
    Ok(Json(agent))
}

// GET /api/v1/agents/:id — 获取 Agent 详情
async fn get_agent(
    State(_state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Agent>, (StatusCode, String)> {
    // TODO: 从数据库查询
    let agent = Agent {
        id: id.clone(),
        name: "Test Agent".to_string(),
        description: Some("A test agent".to_string()),
        version: "0.1.0".to_string(),
        author: Some("Dalin".to_string()),
        tags: vec!["test".to_string()],
        download_count: 0,
        rating: 0.0,
    };
    Ok(Json(agent))
}

// POST /api/v1/compile — 编译 Dalin L 代码
async fn compile_code(
    State(_state): State<AppState>,
    Json(req): Json<CompileRequest>,
) -> Json<CompileResponse> {
    // TODO: 集成 Dalin L 编译器
    Json(CompileResponse {
        success: true,
        output: "Hello, DalinOS!".to_string(),
        errors: vec![],
        suggestions: vec![],
    })
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

// ==================== Main ====================

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    
    // 数据库连接池 (TODO: 从环境变量读取)
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://localhost/dalinos".to_string());
    let pool = PgPool::connect(&database_url)
        .await
        .expect("Failed to connect to database");
    
    let state = AppState { pool };
    
    let app = Router::new()
        .route("/api/v1/agents", get(list_agents).post(create_agent))
        .route("/api/v1/agents/{id}", get(get_agent))
        .route("/api/v1/compile", post(compile_code))
        .with_state(state)
        .layer(CorsLayer::permissive());
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    
    println!("🚀 DalinOS Backend running on http://localhost:3000");
    
    axum::serve(listener, app)
        .await
        .unwrap();
}
