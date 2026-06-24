// JWT Authentication Module
// 作者: GPT (架构师)
// 日期: 2026-06-24

use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{Duration, Utc};

// ==================== Claims ====================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    pub sub: String,        // User ID
    pub username: String,   // Username
    pub role: String,       // Role (user/developer/admin)
    pub exp: usize,         // Expiration time
    pub iat: usize,         // Issued at
}

// ==================== Configuration ====================

pub struct JwtConfig {
    pub secret: String,
    pub access_token_expiry: chrono::Duration,
    pub refresh_token_expiry: chrono::Duration,
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret: std::env::var("JWT_SECRET")
                .unwrap_or_else(|_| "your-secret-key-change-in-production".to_string()),
            access_token_expiry: Duration::hours(1),
            refresh_token_expiry: Duration::days(7),
        }
    }
}

// ==================== Token Generation ====================

/// Generate access token (1 hour expiry)
pub fn generate_access_token(user_id: Uuid, username: &str, role: &str) -> Result<String, String> {
    let config = JwtConfig::default();
    let now = Utc::now();
    
    let claims = Claims {
        sub: user_id.to_string(),
        username: username.to_string(),
        role: role.to_string(),
        exp: (now + config.access_token_expiry).timestamp() as usize,
        iat: now.timestamp() as usize,
    };
    
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(config.secret.as_bytes()),
    ).map_err(|e| e.to_string())
}

/// Generate refresh token (7 day expiry)
pub fn generate_refresh_token(user_id: Uuid) -> Result<String, String> {
    let config = JwtConfig::default();
    let now = Utc::now();
    
    let claims = Claims {
        sub: user_id.to_string(),
        username: String::new(),
        role: String::new(),
        exp: (now + config.refresh_token_expiry).timestamp() as usize,
        iat: now.timestamp() as usize,
    };
    
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(config.secret.as_bytes()),
    ).map_err(|e| e.to_string())
}

// ==================== Token Validation ====================

/// Validate access token and extract claims
pub fn validate_access_token(token: &str) -> Result<Claims, String> {
    let config = JwtConfig::default();
    
    decode::<Claims>(
        token,
        &DecodingKey::from_secret(config.secret.as_bytes()),
        &Validation::default(),
    ).map(|data| data.claims)
    .map_err(|e| e.to_string())
}

/// Validate refresh token and extract user ID
pub fn validate_refresh_token(token: &str) -> Result<Uuid, String> {
    let claims = validate_access_token(token)?;
    Uuid::parse_str(&claims.sub).map_err(|e| e.to_string())
}

// ==================== Middleware ====================

/// Extract user from JWT token in Authorization header
pub async fn get_current_user(
    headers: &axum::http::HeaderMap,
) -> Result<Claims, (axum::http::StatusCode, String)> {
    let auth_header = headers
        .get("Authorization")
        .ok_or((
            axum::http::StatusCode::UNAUTHORIZED,
            "Missing authorization header".to_string(),
        ))?
        .to_str()
        .map_err(|_| {
            (
                axum::http::StatusCode::UNAUTHORIZED,
                "Invalid authorization header format".to_string(),
            )
        })?;
    
    if !auth_header.starts_with("Bearer ") {
        return Err((
            axum::http::StatusCode::UNAUTHORIZED,
            "Invalid token format. Use: Bearer <token>".to_string(),
        ));
    }
    
    let token = &auth_header[7..];
    
    validate_access_token(token).map_err(|_| {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            "Invalid or expired token".to_string(),
        )
    })
}

// ==================== Password Hashing ====================

/// Hash password using bcrypt
pub fn hash_password(password: &str) -> Result<String, String> {
    use bcrypt::{hash, DEFAULT_COST};
    
    hash(password, DEFAULT_COST)
        .map_err(|e| e.to_string())
}

/// Verify password against hash
pub fn verify_password(password: &str, hash: &str) -> Result<bool, String> {
    use bcrypt::verify;
    
    verify(password, hash)
        .map_err(|e| e.to_string())
}

// ==================== Example Usage ====================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_token_generation() {
        let token = generate_access_token(
            Uuid::new_v4(),
            "testuser",
            "user"
        ).unwrap();
        
        assert!(!token.is_empty());
    }
    
    #[test]
    fn test_token_validation() {
        let user_id = Uuid::new_v4();
        let token = generate_access_token(user_id, "testuser", "user").unwrap();
        let claims = validate_access_token(&token).unwrap();
        
        assert_eq!(claims.sub, user_id.to_string());
        assert_eq!(claims.username, "testuser");
    }
    
    #[test]
    fn test_password_hashing() {
        let password = "SecurePass123";
        let hash = hash_password(password).unwrap();
        
        assert!(verify_password(password, &hash).unwrap());
        assert!(!verify_password("WrongPassword", &hash).unwrap());
    }
}
