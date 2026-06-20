//! 中岳: Soma Heritage — 蒸馏训练框架
//!
//! 论文 Section 2.6:
//!   L_total = w_attn · L_attn + w_hidden · L_hidden + w_output · L_output
//!
//! 华岳层分配策略 (v7实验: 71%替换 + 29%保留):
//!   - Sigmoid分布决定每层替换概率
//!   - 浅层低替换(保留基础语义), 深层高替换(推理密集层)
//!   - 渐进替换: 0% → 100% 层替换

/// 蒸馏损失权重
#[derive(Debug, Clone)]
pub struct DistillWeights {
    pub w_attn: f32,
    pub w_hidden: f32,
    pub w_output: f32,
}

impl Default for DistillWeights {
    fn default() -> Self { Self { w_attn: 1.0, w_hidden: 0.5, w_output: 1.0 } }
}

/// 渐进替换调度
pub enum ScheduleType {
    Linear,
    Cosine,
    Sigmoid { steepness: f32 },
}

pub struct ProgressiveSchedule {
    total_steps: usize,
    schedule_type: ScheduleType,
}

impl ProgressiveSchedule {
    pub fn new(total_steps: usize, schedule_type: ScheduleType) -> Self {
        Self { total_steps, schedule_type }
    }

    /// 当前替换比例
    pub fn ratio_at(&self, step: usize) -> f32 {
        let progress = (step as f32 / self.total_steps as f32).min(1.0);
        match &self.schedule_type {
            ScheduleType::Linear => progress,
            ScheduleType::Cosine => 0.5 * (1.0 - (std::f32::consts::PI * progress).cos()),
            ScheduleType::Sigmoid { steepness } => {
                let x = steepness * (progress - 0.5);
                1.0 / (1.0 + (-x).exp())
            }
        }
    }

    /// 需替换的层数
    pub fn layers_to_replace(&self, step: usize, total_layers: usize) -> usize {
        (self.ratio_at(step) * total_layers as f32).round() as usize
    }
}

// ─── 华岳: S型分布层分配 ─────────────────────────────────────────

/// 华岳层分配策略 — v7实验: 29%保留 + 71%替换
///
/// 核心思想:
///   浅层(0~20%) → 低替换概率, 保留基础语义提取
///   中层(20~70%) → Sigmoid过渡, 逐步增加替换
///   深层(70~100%) → 高替换概率, 推理密集层用SFA替换
///
/// 论文实验结果: 71%层替换 + 29%层保留 → +19%加速, PPL仅+0.9%
#[derive(Debug, Clone)]
pub struct SigmoidDistribution {
    /// Sigmoid陡度 (越大过渡越尖锐)
    steepness: f32,
    /// Sigmoid中心位置 (0~1, 在哪个深度开始过渡)
    center: f32,
    /// 目标替换比例 (v7: 0.71)
    target_ratio: f32,
}

impl SigmoidDistribution {
    /// v7实验配置: 71%替换 + 29%保留
    pub fn v7_default() -> Self {
        Self {
            steepness: 8.0,
            center: 0.35,  // 浅层偏移中心, 让浅层更多保留
            target_ratio: 0.71,
        }
    }

    pub fn new(steepness: f32, center: f32, target_ratio: f32) -> Self {
        Self { steepness, center, target_ratio }
    }

    /// 层l (0-indexed) 的替换概率
    /// 深度归一化到 [0, 1], 应用Sigmoid得到替换概率
    pub fn replace_probability(&self, layer: usize, total_layers: usize) -> f32 {
        let depth = layer as f32 / (total_layers - 1).max(1) as f32;
        let raw = 1.0 / (1.0 + (-self.steepness * (depth - self.center)).exp());
        // 缩放到目标比例
        raw * self.target_ratio / (1.0 / (1.0 + (-self.steepness * (1.0 - self.center)).exp())).max(0.01)
    }

    /// 生成层分配掩码: true=SFA替换, false=保留原生Attention
    /// 从深层开始替换, 浅层保留 (论文v7策略)
    pub fn layer_mask(&self, total_layers: usize, replace_ratio: f32) -> Vec<bool> {
        // 计算每层的替换优先级分数 (深层优先)
        let priorities: Vec<(usize, f32)> = (0..total_layers)
            .map(|l| (l, self.replace_probability(l, total_layers)))
            .collect();

        // 按替换概率降序排序
        let mut sorted = priorities.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 选择前 N 层替换
        let n_replace = (replace_ratio * total_layers as f32).round() as usize;
        let mut mask = vec![false; total_layers];
        for (i, _) in sorted.iter().take(n_replace) {
            mask[*i] = true;
        }
        mask
    }

    /// v7实验: 28层中20层SFA + 8层原生 (71%+29%)
    pub fn v7_28layer_mask() -> Vec<bool> {
        let dist = Self::v7_default();
        dist.layer_mask(28, 0.71)
    }
}

// ─── Soma Heritage 蒸馏训练器 ─────────────────────────────────────

/// Soma Heritage 蒸馏训练器
pub struct SomaHeritage {
    weights: DistillWeights,
    pub schedule: ProgressiveSchedule,
    sigmoid_dist: SigmoidDistribution,
    total_layers: usize,
    step: usize,
}

impl SomaHeritage {
    pub fn new(total_layers: usize, total_steps: usize, weights: DistillWeights, schedule_type: ScheduleType) -> Self {
        Self {
            weights,
            schedule: ProgressiveSchedule::new(total_steps, schedule_type),
            sigmoid_dist: SigmoidDistribution::v7_default(),
            total_layers,
            step: 0,
        }
    }

    /// L_total = w_attn · L_attn + w_hidden · L_hidden + w_output · L_output
    pub fn compute_loss(&self, l_attn: f32, l_hidden: f32, l_output: f32) -> f32 {
        self.weights.w_attn * l_attn + self.weights.w_hidden * l_hidden + self.weights.w_output * l_output
    }

    /// 层替换掩码 (华岳S型分布 + 渐进替换)
    ///
    /// 结合当前训练进度(渐进)和Sigmoid分布(层优先级)
    /// 训练初期少替换, 后期达到目标替换比例
    pub fn replacement_mask(&self) -> Vec<bool> {
        let current_ratio = self.schedule.ratio_at(self.step);
        self.sigmoid_dist.layer_mask(self.total_layers, current_ratio)
    }

    /// 当前替换比例
    pub fn current_ratio(&self) -> f32 { self.schedule.ratio_at(self.step) }

    /// 当前保留/替换统计
    pub fn layer_stats(&self) -> (usize, usize) {
        let mask = self.replacement_mask();
        let replaced = mask.iter().filter(|&&x| x).count();
        (self.total_layers - replaced, replaced)
    }

    pub fn advance(&mut self) -> usize { self.step += 1; self.step }

    /// MSE loss
    pub fn mse_loss(teacher: &[f32], student: &[f32]) -> f32 {
        teacher.iter().zip(student.iter())
            .map(|(&t, &s)| (t - s).powi(2)).sum::<f32>() / teacher.len() as f32
    }

    /// KL散度 (注意力分布蒸馏)
    pub fn kl_divergence(teacher: &[f32], student: &[f32]) -> f32 {
        teacher.iter().zip(student.iter())
            .map(|(&p, &q)| if p > 1e-10 && q > 1e-10 { p * (p / q).ln() } else { 0.0 })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_schedule() {
        let s = ProgressiveSchedule::new(1000, ScheduleType::Linear);
        assert!((s.ratio_at(0) - 0.0).abs() < 1e-5);
        assert!((s.ratio_at(500) - 0.5).abs() < 1e-5);
        assert!((s.ratio_at(1000) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_schedule() {
        let s = ProgressiveSchedule::new(1000, ScheduleType::Sigmoid { steepness: 10.0 });
        assert!(s.ratio_at(250) < s.ratio_at(500));
        assert!(s.ratio_at(500) < s.ratio_at(750));
    }

    #[test]
    fn test_distillation_loss() {
        let h = SomaHeritage::new(28, 1000, DistillWeights::default(), ScheduleType::Linear);
        let total = h.compute_loss(0.1, 0.2, 0.3);
        assert!((total - 0.5).abs() < 1e-5); // 1.0*0.1 + 0.5*0.2 + 1.0*0.3
    }

    #[test]
    fn test_mse_loss() {
        let t = vec![1.0f32, 2.0, 3.0];
        let s = vec![1.1f32, 2.1, 3.1];
        let loss = SomaHeritage::mse_loss(&t, &s);
        assert!((loss - 0.01).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_distribution_71pct() {
        let dist = SigmoidDistribution::v7_default();
        let mask = dist.layer_mask(28, 0.71);
        let replaced = mask.iter().filter(|&&x| x).count();
        // 71% of 28 ≈ 20 layers
        assert_eq!(replaced, 20, "Expected 20 replaced, got {}", replaced);
        // 29% = 8 layers retained
        assert_eq!(28 - replaced, 8);
    }

    #[test]
    fn test_sigmoid_shallow_preserved() {
        let dist = SigmoidDistribution::v7_default();
        // 浅层(0-3)应该被保留 (替换概率低)
        let probs: Vec<f32> = (0..6).map(|l| dist.replace_probability(l, 28)).collect();
        // 浅层概率应该 < 深层概率
        assert!(probs[0] < probs[5], "Shallow layers should have lower replace probability");
    }

    #[test]
    fn test_v7_28layer_mask() {
        let mask = SigmoidDistribution::v7_28layer_mask();
        let replaced = mask.iter().filter(|&&x| x).count();
        assert_eq!(replaced, 20); // 71% of 28
        // 深层应该更多被替换
        let deep_replace = mask[24..28].iter().filter(|&&x| x).count();
        let shallow_replace = mask[0..4].iter().filter(|&&x| x).count();
        assert!(deep_replace >= shallow_replace, "Deep layers should have more replacements");
    }

    #[test]
    fn test_heritage_progressive_replacement() {
        let mut h = SomaHeritage::new(28, 10000, DistillWeights::default(),
            ScheduleType::Sigmoid { steepness: 10.0 });
        // Step 0: 替换0层
        let mask0 = h.replacement_mask();
        let replaced0 = mask0.iter().filter(|&&x| x).count();
        assert_eq!(replaced0, 0);

        // Step 4000: sigmoid进度≈0.4, 部分替换
        for _ in 0..4000 { h.advance(); }
        let mask4k = h.replacement_mask();
        let replaced4k = mask4k.iter().filter(|&&x| x).count();
        assert!(replaced4k > 0 && replaced4k < 28, "Partial replacement: got {}", replaced4k);

        // Step 10000: 全量替换
        for _ in 0..6000 { h.advance(); }
        let mask10k = h.replacement_mask();
        let replaced10k = mask10k.iter().filter(|&&x| x).count();
        assert!(replaced10k >= 20, "Expected >= 20 at full progress, got {}", replaced10k);
    }
}
