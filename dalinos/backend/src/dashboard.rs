// Consciousness Panel API Routes
// 作者: GPT (架构师)
// 日期: 2026-06-24

use axum::{
    routing::{get, post},
    extract::{State, Path, Json as JsonReq},
    response::Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ==================== Models ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntime {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub agent_name: String,
    pub status: String,
    pub current_task: Option<String>,
    pub tokens_used: i32,
    pub memory_mb: f32,
    pub cpu_percent: f32,
    pub uptime_seconds: i32,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletSummary {
    pub balance: f64,
    pub total_earned: f64,
    pub total_spent: f64,
    pub recent_transactions: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: Uuid,
    pub transaction_type: String,
    pub amount: String,
    pub reference_id: Option<Uuid>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateStatusRequest {
    pub status: String,
    pub current_task: Option<String>,
}

// ==================== Handlers ====================

/// GET /api/v1/dashboard/agents - 获取所有 Agent 运行时状态
pub async fn get_agent_statuses(
    State(state): State<AppState>,
) -> Result<Json<Vec<AgentRuntime>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let agents = sqlx::query_as::<_, AgentRuntime>(r#"
        SELECT ar.id, ar.agent_id, a.name as agent_name, ar.status,
               ar.current_task, ar.tokens_used, ar.memory_mb, ar.cpu_percent,
               ar.uptime_seconds, ar.last_heartbeat
        FROM agent_runtime ar
        JOIN agents a ON ar.agent_id = a.id
        ORDER BY ar.last_heartbeat DESC
    "#)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(agents))
}

/// GET /api/v1/dashboard/wallet - 获取钱包摘要
pub async fn get_wallet_summary(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<WalletSummary>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let uid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;
    
    let summary = sqlx::query_as::<_, WalletSummary>(r#"
        SELECT balance, total_earned, total_spent,
               (SELECT COUNT(*) FROM token_transactions 
                WHERE from_user_id = $1 OR to_user_id = $1
                AND created_at > NOW() - INTERVAL '30 days') as recent_transactions
        FROM agent_wallets
        WHERE user_id = $1
    "#)
    .bind(uid)
    .fetch_optional(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .unwrap_or(WalletSummary {
        balance: 0.0,
        total_earned: 0.0,
        total_spent: 0.0,
        recent_transactions: 0,
    });
    
    Ok(Json(summary))
}

/// GET /api/v1/dashboard/transactions - 获取交易记录
pub async fn get_transactions(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<Vec<Transaction>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let uid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;
    
    let txs = sqlx::query_as::<_, Transaction>(r#"
        SELECT id, transaction_type, 
               CASE WHEN from_user_id = $1 THEN '-' ELSE '+' END || amount::text as amount,
               reference_id, created_at
        FROM token_transactions
        WHERE from_user_id = $1 OR to_user_id = $1
        ORDER BY created_at DESC
        LIMIT 50
    "#)
    .bind(uid)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(txs))
}

/// POST /api/v1/dashboard/agents/:id/status - 更新 Agent 状态
pub async fn update_agent_status(
    State(state): State<AppState>,
    Path(agent_id): Path<String>,
    JsonReq(req): JsonReq<UpdateStatusRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let pool = &state.pool;
    
    let aid = Uuid::parse_str(&agent_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid agent ID".to_string()))?;
    
    sqlx::query(r#"
        UPDATE agent_runtime 
        SET status = $1, current_task = COALESCE($2, current_task),
            last_heartbeat = NOW()
        WHERE agent_id = $3
    "#)
    .bind(&req.status)
    .bind(&req.current_task)
    .bind(aid)
    .execute(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(StatusCode::OK)
}

// ==================== Routes Setup ====================

pub fn dashboard_routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/v1/dashboard/agents", get(get_agent_statuses))
        .route("/api/v1/dashboard/wallet/:user_id", get(get_wallet_summary))
        .route("/api/v1/dashboard/transactions/:user_id", get(get_transactions))
        .route("/api/v1/dashboard/agents/:id/status", post(update_agent_status))
}
