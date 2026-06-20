//! Core math utilities
//! 论文 Definition 1-3, Section 1.2

/// Gaussian spatial decay: φ(r) = exp(-r²/(2σ²))
/// 论文 Definition 2
#[inline]
pub fn gaussian_decay(r: f32, sigma: f32) -> f32 {
    (-r * r / (2.0 * sigma * sigma)).exp()
}

/// Exponential temporal decay: ψ(Δt) = exp(-λ·Δt)
/// 论文 Definition 3
#[inline]
pub fn exponential_decay(delta_t: f32, lambda: f32) -> f32 {
    (-lambda * delta_t).exp()
}

/// Stable softmax in-place: softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
/// 论文 Section 1.2: Attention_near = softmax(q·K^T/√d_h)·V_ring
pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() { return; }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in logits.iter_mut() { *x /= sum; }
    }
}

/// Compute attention weights: softmax(q · K^T · scale)
/// q: [d], keys: [k][d] → weights: [k]
pub fn attention_weights(q: &[f32], keys: &[Vec<f32>], scale: f32) -> Vec<f32> {
    let mut logits: Vec<f32> = keys.iter().map(|k| {
        q.iter().zip(k.iter()).map(|(&qi, &ki)| qi * ki).sum::<f32>() * scale
    }).collect();
    softmax(&mut logits);
    logits
}

/// Weighted sum: weights[k] · values[k][d] → output[d]
pub fn weighted_sum(weights: &[f32], values: &[Vec<f32>]) -> Vec<f32> {
    if values.is_empty() { return vec![]; }
    let d = values[0].len();
    let mut out = vec![0.0f32; d];
    for (i, &w) in weights.iter().enumerate() {
        for (j, &v) in values[i].iter().enumerate() {
            out[j] += w * v;
        }
    }
    out
}

/// Cosine similarity: cos(a, b) = a·b / (|a|·|b|)
/// 论文 Table 2: 验证指标 (cos > 0.9999999)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { 0.0 } else { dot / (na * nb) }
}

/// EMA update: S^(t) = γ · S^(t-1) + (1-γ) · x_t
/// 论文 Section 1.2 & 2.2.2
#[inline]
pub fn ema_update(state: &mut [f32], new_val: &[f32], gamma: f32) {
    let one_m_gamma = 1.0 - gamma;
    for (s, &x) in state.iter_mut().zip(new_val.iter()) {
        *s = gamma * *s + one_m_gamma * x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_decay() {
        assert!((gaussian_decay(0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((gaussian_decay(1.0, 1.0) - 0.6065).abs() < 0.001);
    }

    #[test]
    fn test_exponential_decay() {
        assert!((exponential_decay(0.0, 0.01) - 1.0).abs() < 1e-6);
        assert!((exponential_decay(100.0, 0.01) - 0.368).abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        let c = [0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_ema() {
        let mut s = [0.0f32; 3];
        ema_update(&mut s, &[1.0, 1.0, 1.0], 0.98);
        assert!((s[0] - 0.02).abs() < 1e-6);
    }
}
