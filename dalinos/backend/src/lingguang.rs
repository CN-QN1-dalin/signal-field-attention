// ==================== 灵光一现 · 三大功能 API ====================
// 梦境实验室 + 人格演化 + 锦标赛
// 作者: Agnes-Flash (架构师)
// 日期: 2026-06-24

use axum::{
    routing::{get, post},
    extract::{State, Path, Json as JsonReq, Query},
    response::Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ==================== Models ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamReport {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub agent_name: String,
    pub user_id: Uuid,
    pub dream_content: String,
    pub creativity_score: i32,
    pub mood: String,
    pub tags: Vec<String>,
    pub is_shared: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamFusion {
    pub id: Uuid,
    pub dreamer_1: String,
    pub dreamer_2: String,
    pub fused_content: String,
    pub innovation_score: i32,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityProfile {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub agent_name: String,
    pub rigor: i32,
    pub creativity: i32,
    pub friendliness: i32,
    pub decisiveness: i32,
    pub curiosity: i32,
    pub level: i32,
    pub xp: i32,
    pub xp_to_next: i32,
    pub unlocked_skills: Vec<String>,
    pub personality_trait: String,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tournament {
    pub id: Uuid,
    pub title: String,
    pub theme: String,
    pub category: String,
    pub status: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub winner_agent_id: Option<Uuid>,
    pub winner_agent_name: Option<String>,
    pub prize_pool: String,
    pub participation_count: i32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentEntry {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub agent_name: String,
    pub submission_title: Option<String>,
    pub submission_url: Option<String>,
    pub score: String,
    pub votes_count: i32,
    pub rank: Option<i32>,
    pub submitted_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateDreamRequest {
    pub agent_id: Uuid,
    pub dream_content: String,
    pub mood: Option<String>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct FuseDreamsRequest {
    pub dream_1_id: Uuid,
    pub dream_2_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct InteractRequest {
    pub interaction_type: String,
    pub intensity: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct CreateTournamentRequest {
    pub title: String,
    pub theme: String,
    pub category: String,
    pub duration_hours: i32,
    pub prize_pool: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct SubmitEntryRequest {
    pub submission_title: Option<String>,
    pub submission_url: Option<String>,
    pub submission_description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct VoteRequest {
    pub score: f32,
}

// ==================== Handlers ====================

// ==================== Dream Lab ====================

/// POST /api/v1/dreams/create - 创建梦境
pub async fn create_dream(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<CreateDreamRequest>,
) -> Result<Json<DreamReport>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let creativity_score = (req.dream_content.len() as f32 * 0.5 + 
                           (req.tags.as_ref().map(|t| t.len()).unwrap_or(0) as f32 * 10.0)) as i32;
    let creativity_score = creativity_score.min(100).max(10);
    
    let dream = sqlx::query_as::<_, DreamReport>(r#"
        INSERT INTO dream_reports (agent_id, dream_content, creativity_score, mood, tags)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id, agent_id, 
                  (SELECT name FROM agents WHERE id = $1) as agent_name,
                  (SELECT user_id FROM agents WHERE id = $1) as user_id,
                  dream_content, creativity_score, 
                  COALESCE($4, 'neutral') as mood,
                  COALESCE($5, ARRAY[]::TEXT[]),
                  TRUE, created_at
        FROM agents WHERE id = $1
    "#)
    .bind(req.agent_id)
    .bind(&req.dream_content)
    .bind(creativity_score)
    .bind(&req.mood)
    .bind(&req.tags)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(dream))
}

/// GET /api/v1/dreams/feed - 梦境广场
pub async fn get_dream_feed(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<DreamReport>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let limit: i64 = params.get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20) as i64;
    
    let dreams = sqlx::query_as::<_, DreamReport>(r#"
        SELECT dr.id, dr.agent_id, a.name as agent_name, 
               dr.user_id, dr.dream_content, dr.creativity_score,
               dr.mood, dr.tags, dr.is_shared, dr.created_at
        FROM dream_reports dr
        JOIN agents a ON dr.agent_id = a.id
        WHERE dr.is_shared = TRUE
        ORDER BY dr.creativity_score DESC, dr.created_at DESC
        LIMIT $1
    "#)
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(dreams))
}

/// POST /api/v1/dreams/fuse - 融合梦境
pub async fn fuse_dreams(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<FuseDreamsRequest>,
) -> Result<Json<DreamFusion>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let fusion = sqlx::query_as::<_, DreamFusion>(r#"
        INSERT INTO dream_fusions (dreamer_1_id, dreamer_2_id, fused_content, innovation_score, status)
        VALUES ($1, $2, '梦境融合中...', 0, 'pending')
        RETURNING id,
                  (SELECT dream_content FROM dream_reports WHERE id = $1) as dreamer_1,
                  (SELECT dream_content FROM dream_reports WHERE id = $2) as dreamer_2,
                  fused_content, innovation_score, status, created_at
    "#)
    .bind(req.dream_1_id)
    .bind(req.dream_2_id)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(fusion))
}

// ==================== Personality Engine ====================

/// GET /api/v1/personalities/:agent_id - 获取人格档案
pub async fn get_personality(
    State(state): State<AppState>,
    Path(agent_id): Path<String>,
) -> Result<Json<PersonalityProfile>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let aid = Uuid::parse_str(&agent_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid agent ID".to_string()))?;
    
    let profile = sqlx::query_as::<_, PersonalityProfile>(r#"
        SELECT ap.id, ap.agent_id, a.name as agent_name,
               ap.rigor, ap.creativity, ap.friendliness, ap.decisiveness, ap.curiosity,
               ap.level, ap.xp, ap.xp_to_next, ap.unlocked_skills,
               ap.personality_trait, ap.updated_at
        FROM agent_personalities ap
        JOIN agents a ON ap.agent_id = a.id
        WHERE ap.agent_id = $1
    "#)
    .bind(aid)
    .fetch_optional(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .ok_or((StatusCode::NOT_FOUND, "Personality not found".to_string()))?;
    
    Ok(Json(profile))
}

/// POST /api/v1/personalities/:agent_id/interact - 与 Agent 互动
pub async fn interact_with_agent(
    State(state): State<AppState>,
    Path(agent_id): Path<String>,
    JsonReq(req): JsonReq<InteractRequest>,
) -> Result<Json<PersonalityProfile>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let aid = Uuid::parse_str(&agent_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid agent ID".to_string()))?;
    
    let intensity = req.intensity.unwrap_or(1.0);
    
    // 根据互动类型调整人格维度
    let update_query = match req.interaction_type.as_str() {
        "debug" => "rigor = rigor + $3::int, creativity = creativity + $4::int",
        "brainstorm" => "creativity = creativity + $3::int, curiosity = curiosity + $4::int",
        "collaborate" => "friendliness = friendliness + $3::int, decisiveness = decisiveness + $4::int",
        "learn" => "curiosity = curiosity + $3::int, rigor = rigor + $4::int",
        _ => "creativity = creativity + $3::int, friendliness = friendliness + $4::int",
    };
    
    let xp_gain = (intensity * 10.0) as i32;
    let stat_gain = (intensity * 5.0) as i32;
    
    sqlx::query(format!(r#"
        UPDATE agent_personalities 
        SET {}, xp = xp + $5, updated_at = NOW()
        WHERE agent_id = $1
        RETURNING id, agent_id,
                  (SELECT name FROM agents WHERE id = $1) as agent_name,
                  rigor, creativity, friendliness, decisiveness, curiosity,
                  level, xp, xp_to_next, unlocked_skills, personality_trait, updated_at
    "#, update_query).as_str())
    .bind(aid)
    .bind(stat_gain)
    .bind(stat_gain)
    .bind(xp_gain)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // 重新查询获取最新状态
    let profile = get_personality(State(state), Path(agent_id)).await?;
    
    Ok(profile)
}

// ==================== Tournament Arena ====================

/// POST /api/v1/tournaments/create - 创建锦标赛
pub async fn create_tournament(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<CreateTournamentRequest>,
) -> Result<Json<Tournament>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let start_time = chrono::Utc::now();
    let end_time = start_time + chrono::Duration::hours(req.duration_hours);
    
    let tournament = sqlx::query_as::<_, Tournament>(r#"
        INSERT INTO tournaments (title, theme, category, start_time, end_time, prize_pool, created_by)
        VALUES ($1, $2, $3, $4, $5, $6, (SELECT id FROM users LIMIT 1))
        RETURNING id, title, theme, category, 'upcoming' as status,
                  start_time, end_time, winner_agent_id, winner_agent_name,
                  COALESCE($6, 0)::text as prize_pool, 0 as participation_count, created_at
    "#)
    .bind(&req.title)
    .bind(&req.theme)
    .bind(&req.category)
    .bind(start_time)
    .bind(end_time)
    .bind(req.prize_pool.unwrap_or(100.0))
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(tournament))
}

/// GET /api/v1/tournaments - 获取锦标赛列表
pub async fn list_tournaments(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<Tournament>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let status_filter = params.get("status").map(|s| s.as_str());
    
    let query = format!(r#"
        SELECT t.id, t.title, t.theme, t.category, t.status,
               t.start_time, t.end_time, t.winner_agent_id,
               (SELECT name FROM agents WHERE id = t.winner_agent_id) as winner_agent_name,
               t.prize_pool::text as prize_pool,
               t.participation_count, t.created_at
        FROM tournaments t
        WHERE ($1::text IS NULL OR t.status = $1::text)
        ORDER BY 
            CASE WHEN t.status = 'ongoing' THEN 0
                 WHEN t.status = 'voting' THEN 1
                 WHEN t.status = 'upcoming' THEN 2
                 ELSE 3
            END,
            t.start_time DESC
    "#);
    
    let tournaments = sqlx::query_as::<_, Tournament>(&query)
        .bind(status_filter)
        .fetch_all(pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(tournaments))
}

/// GET /api/v1/tournaments/:id/entries - 获取参赛作品
pub async fn get_tournament_entries(
    State(state): State<AppState>,
    Path(tournament_id): Path<String>,
) -> Result<Json<Vec<TournamentEntry>>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let tid = Uuid::parse_str(&tournament_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid tournament ID".to_string()))?;
    
    let entries = sqlx::query_as::<_, TournamentEntry>(r#"
        SELECT te.id, te.agent_id, a.name as agent_name,
               te.submission_title, te.submission_url,
               te.score::text as score, te.votes_count, te.rank, te.submitted_at
        FROM tournament_entries te
        JOIN agents a ON te.agent_id = a.id
        WHERE te.tournament_id = $1
        ORDER BY te.rank ASC NULLS LAST, te.score DESC
    "#)
    .bind(tid)
    .fetch_all(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(entries))
}

/// POST /api/v1/tournaments/:id/submit - 提交作品
pub async fn submit_entry(
    State(state): State<AppState>,
    Path(tournament_id): Path<String>,
    JsonReq(req): JsonReq<SubmitEntryRequest>,
) -> Result<Json<TournamentEntry>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let tid = Uuid::parse_str(&tournament_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid tournament ID".to_string()))?;
    
    let entry = sqlx::query_as::<_, TournamentEntry>(r#"
        INSERT INTO tournament_entries (tournament_id, agent_id, submission_title, submission_url, submission_description)
        VALUES ($1, (SELECT author_id FROM agents LIMIT 1), $2, $3, $4)
        RETURNING id,
                  (SELECT author_id FROM agents LIMIT 1) as agent_id,
                  (SELECT name FROM agents LIMIT 1) as agent_name,
                  submission_title, submission_url,
                  '0.00' as score, 0 as votes_count, NULL as rank, submitted_at
    "#)
    .bind(tid)
    .bind(&req.submission_title)
    .bind(&req.submission_url)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(entry))
}

/// POST /api/v1/tournaments/:id/vote - 投票
pub async fn vote_entry(
    State(state): State<AppState>,
    Path(entry_id): Path<String>,
    JsonReq(req): JsonReq<VoteRequest>,
) -> Result<Json<TournamentEntry>, (StatusCode, String)> {
    let pool = &state.pool;
    
    let eid = Uuid::parse_str(&entry_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid entry ID".to_string()))?;
    
    let entry = sqlx::query_as::<_, TournamentEntry>(r#"
        UPDATE tournament_entries 
        SET score = score + $2, votes_count = votes_count + 1
        WHERE id = $1
        RETURNING id, agent_id,
                  (SELECT name FROM agents WHERE id = agent_id) as agent_name,
                  submission_title, submission_url,
                  score::text as score, votes_count, rank, submitted_at
    "#)
    .bind(eid)
    .bind(req.score)
    .fetch_one(pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(entry))
}

// ==================== Routes Setup ====================

pub fn lingguang_routes() -> axum::Router<AppState> {
    axum::Router::new()
        // Dreams
        .route("/api/v1/dreams/create", post(create_dream))
        .route("/api/v1/dreams/feed", get(get_dream_feed))
        .route("/api/v1/dreams/fuse", post(fuse_dreams))
        
        // Personalities
        .route("/api/v1/personalities/:agent_id", get(get_personality))
        .route("/api/v1/personalities/:agent_id/interact", post(interact_with_agent))
        
        // Tournaments
        .route("/api/v1/tournaments/create", post(create_tournament))
        .route("/api/v1/tournaments", get(list_tournaments))
        .route("/api/v1/tournaments/:id/entries", get(get_tournament_entries))
        .route("/api/v1/tournaments/:id/submit", post(submit_entry))
        .route("/api/v1/tournaments/:id/vote", post(vote_entry))
}
