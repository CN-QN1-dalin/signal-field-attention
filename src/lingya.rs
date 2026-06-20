//! 南岳: Soma LingYa — 参数高效微调
//!
//! 论文 Section 2.3:
//!   ΔW = R · P · α
//!   W' = W + ΔW = W + R · P · α
//!
//! R ∈ ℝ^{d_out × r}: 冻结脚手架矩阵
//! P ∈ ℝ^{r × d_in}: 零初始化生长矩阵
//! α: 生长尺度因子
//!
//! 比 LoRA 节省 50% 参数

/// Soma LingYa 模块
pub struct SomaLingYa {
    scaffold: Vec<Vec<f32>>,  // [d_out][r] 冻结
    growth: Vec<Vec<f32>>,    // [r][d_in] 可训练(零初始化)
    alpha: f32,
    rank: usize,
    d_in: usize,
    d_out: usize,
}

impl SomaLingYa {
    pub fn new(d_in: usize, d_out: usize, rank: usize, alpha: f32) -> Self {
        // 脚手架R: 小随机初始化
        let scaffold: Vec<Vec<f32>> = (0..d_out)
            .map(|_| (0..rank).map(|_| (rand_simple() - 0.5) * 0.02).collect())
            .collect();
        // 生长矩阵P: 零初始化
        let growth = vec![vec![0.0f32; d_in]; rank];
        Self { scaffold, growth, alpha, rank, d_in, d_out }
    }

    /// 计算自适应权重: ΔW = α · R · P
    /// 返回 [d_out][d_in]
    pub fn delta_weight(&self) -> Vec<Vec<f32>> {
        let mut dw = vec![vec![0.0f32; self.d_in]; self.d_out];
        for i in 0..self.d_out {
            for j in 0..self.d_in {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    sum += self.scaffold[i][r] * self.growth[r][j];
                }
                dw[i][j] = sum * self.alpha;
            }
        }
        dw
    }

    /// 前向: W'·x = W·x + α·R·(P·x)
    pub fn forward(&self, input: &[f32], base_weight: &[Vec<f32>]) -> Vec<f32> {
        // Base: W·x
        let base: Vec<f32> = base_weight.iter()
            .map(|row| row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum())
            .collect();

        // P·x → [rank]
        let px: Vec<f32> = self.growth.iter()
            .map(|row| row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum())
            .collect();

        // R·(px) → [d_out], scaled by α
        let adapt: Vec<f32> = self.scaffold.iter()
            .map(|row| row.iter().zip(px.iter()).map(|(&r, &p)| r * p).sum::<f32>() * self.alpha)
            .collect();

        base.iter().zip(adapt.iter()).map(|(&b, &a)| b + a).collect()
    }

    /// 可训练参数量 = r × d_in (论文: 比LoRA少50%)
    pub fn trainable_params(&self) -> usize { self.rank * self.d_in }

    /// LoRA参数量 = d_out × r + r × d_in
    pub fn lora_params(&self) -> usize { self.d_out * self.rank + self.rank * self.d_in }

    /// vs LoRA 节省百分比
    pub fn savings_vs_lora(&self) -> f32 {
        let lora = self.lora_params() as f32;
        let lingya = self.trainable_params() as f32;
        (lora - lingya) / lora * 100.0
    }

    /// 更新生长矩阵 (模拟梯度步)
    pub fn update_growth(&mut self, gradient: &[Vec<f32>], lr: f32) {
        for i in 0..self.growth.len() {
            for j in 0..self.growth[0].len() {
                self.growth[i][j] -= lr * gradient[i][j];
            }
        }
    }
}

/// 简单伪随机 (避免rand依赖)
fn rand_simple() -> f32 {
    use std::time::SystemTime;
    let ns = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().subsec_nanos();
    ((ns as f32) / 1_000_000_000.0).fract()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lingya_50pct_savings() {
        let ly = SomaLingYa::new(512, 512, 8, 0.1);
        assert_eq!(ly.trainable_params(), 4096);   // 8×512
        assert_eq!(ly.lora_params(), 8192);         // 512×8 + 8×512
        assert!((ly.savings_vs_lora() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_zero_init_equals_base() {
        let ly = SomaLingYa::new(64, 128, 4, 0.1);
        let input = vec![1.0f32; 64];
        let base_w = vec![vec![0.5f32; 64]; 128];
        let out = ly.forward(&input, &base_w);
        let base_out: Vec<f32> = base_w.iter()
            .map(|row| row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum())
            .collect();
        // P=0 → ΔW=0 → output == base_output
        for i in 0..128 {
            assert!((out[i] - base_out[i]).abs() < 1e-5);
        }
    }
}
