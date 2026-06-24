/// Intent Semantic Pool — 全局意图语义

#[derive(Debug, Clone)]
pub struct IntentSemanticPool {
    slots: Vec<Vec<f32>>,
    num_slots: usize,
    dim: usize,
    temperature: f32,
}

impl IntentSemanticPool {
    /// 创建新的 Semantic Pool
    pub fn new(num_slots: usize, dim: usize, temperature: f32) -> Self {
        Self {
            slots: vec![vec![0.0; dim]; num_slots],
            num_slots,
            dim,
            temperature,
        }
    }

    /// 添加意图到 Pool
    pub fn add_intent(&mut self, intent: Vec<f32>) {
        assert_eq!(intent.len(), self.dim, "Intent dimension mismatch");
        for i in 0..self.num_slots {
            let weight = self.calculate_slot_weight(i, &intent);
            for j in 0..self.dim {
                self.slots[i][j] = self.slots[i][j] * 0.9 + intent[j] * weight * 0.1;
            }
        }
    }

    /// 计算槽位权重 (余弦相似度 / temperature)
    fn calculate_slot_weight(&self, slot_idx: usize, intent: &[f32]) -> f32 {
        let slot = &self.slots[slot_idx];
        let dot: f32 = slot.iter().zip(intent.iter()).map(|(a, b)| a * b).sum();
        let norm_slot: f32 = slot.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_intent: f32 = intent.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_slot > 1e-10 && norm_intent > 1e-10 {
            let cosine_sim = dot / (norm_slot * norm_intent);
            cosine_sim.exp() / self.temperature
        } else {
            0.0
        }
    }

    /// 查询语义 Pool
    pub fn query(&self, current_intent: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.dim];
        let mut total_weight = 0.0;
        for i in 0..self.num_slots {
            let weight = self.calculate_slot_weight(i, current_intent);
            total_weight += weight;
            for j in 0..self.dim {
                result[j] += self.slots[i][j] * weight;
            }
        }
        if total_weight > 1e-10 {
            for j in 0..self.dim {
                result[j] /= total_weight;
            }
        }
        result
    }

    pub fn num_slots(&self) -> usize {
        self.num_slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_pool_query() {
        let mut pool = IntentSemanticPool::new(4, 3, 0.07);
        pool.add_intent(vec![1.0, 2.0, 3.0]);
        pool.add_intent(vec![4.0, 5.0, 6.0]);
        let result = pool.query(&vec![1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3);
        assert!(result[0] > 0.0);
    }
}
