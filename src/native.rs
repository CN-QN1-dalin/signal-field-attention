//! 西岳: Soma Native — 零设计神经网络架构
//!
//! 论文 Section 2.4:
//!   SomaBlock(x) = x + Homeostasis₂(LingYaBlock(Homeostasis₁(x + SignalFieldLayer(x))))
//!
//! Homeostasis 替代 LayerNorm
//! LingYaBlock 替代 FFN
//! SignalFieldLayer 替代 MHA

use crate::config::SomaConfig;
use crate::engine::SomaEngine;
use crate::lingya::SomaLingYa;

/// Homeostasis 归一化层 — 替代 LayerNorm
pub struct Homeostasis {
    gamma: Vec<f32>,  // 可学习缩放
    beta: Vec<f32>,   // 可学习偏移
    eps: f32,
}

impl Homeostasis {
    pub fn new(dims: usize) -> Self {
        Self {
            gamma: vec![1.0f32; dims],
            beta: vec![0.0f32; dims],
            eps: 1e-5,
        }
    }

    /// Homeostasis(x) = γ ⊙ normalize(x) + β
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len() as f32;
        let mean: f32 = input.iter().sum::<f32>() / n;
        let var: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = (var + self.eps).sqrt();
        input.iter().zip(self.gamma.iter().zip(self.beta.iter()))
            .map(|(&x, (&g, &b))| g * (x - mean) / std + b)
            .collect()
    }
}

/// LingYaBlock — 替代 FFN
pub struct LingYaBlock {
    up_proj: SomaLingYa,
    down_proj: SomaLingYa,
}

impl LingYaBlock {
    pub fn new(dims: usize, rank: usize, alpha: f32) -> Self {
        let hidden_dims = dims * 4;
        Self {
            up_proj: SomaLingYa::new(dims, hidden_dims, rank, alpha),
            down_proj: SomaLingYa::new(hidden_dims, dims, rank, alpha),
        }
    }

    /// LingYaBlock(x) = DownProj(GELU(UpProj(x)))
    pub fn forward(&self, input: &[f32], base_up: &[Vec<f32>], base_down: &[Vec<f32>]) -> Vec<f32> {
        let hidden = self.up_proj.forward(input, base_up);
        let activated: Vec<f32> = hidden.iter().map(|&x| gelu(x)).collect();
        self.down_proj.forward(&activated, base_down)
    }
}

/// SomaBlock — 统一场块
/// 论文 Section 2.4
pub struct SomaBlock {
    homeostasis_1: Homeostasis,
    homeostasis_2: Homeostasis,
    signal_field: SomaEngine,
    lingya_block: LingYaBlock,
}

impl SomaBlock {
    pub fn new(config: &SomaConfig, lingya_rank: usize, lingya_alpha: f32) -> Self {
        Self {
            homeostasis_1: Homeostasis::new(config.dims),
            homeostasis_2: Homeostasis::new(config.dims),
            signal_field: SomaEngine::new(config),
            lingya_block: LingYaBlock::new(config.dims, lingya_rank, lingya_alpha),
        }
    }

    /// SomaBlock(x) = x + H₂(LingYaBlock(H₁(x + SFLayer(x))))
    pub fn forward(
        &mut self,
        input: &[f32],
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        base_up: &[Vec<f32>],
        base_down: &[Vec<f32>],
    ) -> Vec<f32> {
        let sf_out = self.signal_field.decode_step(query, key, value);
        // Flatten multi-head → [dims]
        let sf_flat: Vec<f32> = sf_out.into_iter().flatten().take(input.len()).collect();
        let mut residual = input.to_vec();
        for (i, &v) in sf_flat.iter().enumerate() {
            if i < residual.len() { residual[i] += v; }
        }
        let norm1 = self.homeostasis_1.forward(&residual);
        let ffn_out = self.lingya_block.forward(&norm1, base_up, base_down);
        let norm2 = self.homeostasis_2.forward(&ffn_out);
        input.iter().zip(norm2.iter()).map(|(&x, &n)| x + n).collect()
    }
}

/// GELU 激活 (近似)
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homeostasis() {
        let h = Homeostasis::new(64);
        let input = vec![3.0f32; 64];
        let out = h.forward(&input);
        let mean: f32 = out.iter().sum::<f32>() / 64.0;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_gelu() {
        assert!(gelu(0.0).abs() < 1e-5);
        assert!(gelu(1.0) > 0.0);
        assert!(gelu(-1.0) < 0.0);
    }
}
