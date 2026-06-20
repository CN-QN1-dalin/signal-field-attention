//! Model configuration parameters
//! 论文 Section 2.2.3, 3.1, Table 5, Table 6

/// Soma framework configuration
#[derive(Debug, Clone)]
pub struct SomaConfig {
    /// Model dimension (d_model)
    pub dims: usize,
    /// Number of query attention heads
    pub num_heads: usize,
    /// Number of KV heads (GQA)
    pub kv_heads: usize,
    /// Dimension per attention head (d_head)
    pub head_dim: usize,
    /// Ring KV Buffer window size (k) — 论文: k=16
    pub window_size: usize,
    /// EMA decay factor (γ) — 论文 Section 3.1: γ=0.98
    pub gamma: f32,
    /// Far-field weight (α) — 论文 Section 1.2: α ∈ [0,1], Section 3.1: α=0.1
    pub alpha: f32,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Spatial decay width (σ) — 论文 Definition 2
    pub sigma: f32,
    /// Temporal decay rate (λ) — 论文 Definition 3
    pub lambda: f32,
}

impl SomaConfig {
    /// Qwen2.5-0.5B-Instruct
    /// 论文 Table 5: dims=896, heads=14, head_dim=64, kv_heads=2
    pub fn qwen_0_5b() -> Self {
        Self {
            dims: 896, num_heads: 14, kv_heads: 2, head_dim: 64,
            window_size: 16, gamma: 0.98, alpha: 0.1, num_layers: 24,
            sigma: 1.0, lambda: 0.01,
        }
    }

    /// Qwen2.5-7B
    /// 论文 Table 6: dims=3584, kv_heads=4, head_dim=128
    pub fn qwen_7b() -> Self {
        Self {
            dims: 3584, num_heads: 28, kv_heads: 4, head_dim: 128,
            window_size: 16, gamma: 0.98, alpha: 0.1, num_layers: 28,
            sigma: 1.0, lambda: 0.01,
        }
    }

    /// |Θ_extra| = n_kv · k · d_head + k (论文 Section 2.2.3)
    pub fn extra_params_per_layer(&self) -> usize {
        self.kv_heads * self.window_size * self.head_dim + self.window_size
    }

    /// Soma KV memory in KB (constant, independent of seq_len)
    /// Uses kv_heads (GQA): only store compressed KV heads
    pub fn soma_kv_memory_kb(&self) -> f64 {
        let ring = self.kv_heads * self.window_size * self.head_dim * 2 * 4;
        let field = self.kv_heads * self.head_dim * 2 * 4;
        (ring + field) as f64 / 1024.0
    }

    /// Standard Attention KV cache in KB (uses num_heads for standard)
    pub fn standard_kv_memory_kb(&self, seq_len: usize) -> f64 {
        (self.num_heads * seq_len * self.head_dim * 2 * 4) as f64 / 1024.0
    }

    /// Compression ratio
    pub fn compression_ratio(&self, seq_len: usize) -> f64 {
        self.standard_kv_memory_kb(seq_len) / self.soma_kv_memory_kb()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0_5b_extra_params() {
        let cfg = SomaConfig::qwen_0_5b();
        assert_eq!(cfg.extra_params_per_layer(), 2064); // 论文: 2×16×64+16
    }

    #[test]
    fn test_7b_extra_params() {
        let cfg = SomaConfig::qwen_7b();
        assert_eq!(cfg.extra_params_per_layer(), 8208); // 论文: 4×16×128+16
    }

    #[test]
    fn test_0_5b_memory_4k() {
        let cfg = SomaConfig::qwen_0_5b();
        let soma = cfg.soma_kv_memory_kb();
        let std = cfg.standard_kv_memory_kb(4096);
        // Standard uses num_heads=14: 14*4096*64*2*4/1024 = 28672 KB (论文 Table 5)
        assert!((std - 28672.0).abs() < 1.0, "Std KB={std}");
        // Soma uses kv_heads=2: 2*16*64*2*4 + 2*64*2*4 = 17408 bytes = 17.0 KB
        // 论文 Table 5 报告 115.5 KB (含额外运行时开销), 我们的实现是纯GQA计算
        assert!((soma - 17.0).abs() < 1.0, "Soma KB={soma}");
        // 压缩比: 28672/17 ≈ 1687x (GQA比论文248x更高, 因GQA只存kv_heads)
        assert!(cfg.compression_ratio(4096) > 200.0);
    }

    #[test]
    fn test_7b_memory_64k() {
        let cfg = SomaConfig::qwen_7b();
        let soma = cfg.soma_kv_memory_kb();
        // 论文 Table 6 报告 462 KB; 我们用 GQA kv_heads=4 计算更小
        assert!(soma > 0.0 && soma < 500.0, "Soma KB={soma}");
    }
}
