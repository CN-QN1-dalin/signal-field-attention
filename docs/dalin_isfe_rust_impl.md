# Dalin ISFE — Rust 核心实现

> **版本**: v1.0
> **日期**: 2026-06-24
> **状态**: 执行中

---

## 1. 项目结构

```
dalin-isfe/
├── Cargo.toml
├── src/
│   ├── lib.rs          # 库入口
│   ├── ring_buffer.rs  # RingBuffer 实现
│   ├── ema_field.rs    # EMA Field 实现
│   ├── semantic_pool.rs # Semantic Pool 实现
│   ├── fusion.rs       # Fusion 实现
│   └── engine.rs       # 引擎主入口
└── tests/
    └── integration.rs  # 集成测试
```

---

## 2. Cargo.toml

```toml
[package]
name = "dalin-isfe"
version = "0.1.0"
edition = "2021"
description = "Intent Signal Field Engine — 让 AI 真正理解人类意图"
license = "MIT"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

---

## 3. src/lib.rs

```rust
//! Dalin ISFE — Intent Signal Field Engine
//!
//! 全球首个将 SFA 信号场思想应用到意图理解的引擎。

pub mod ring_buffer;
pub mod ema_field;
pub mod semantic_pool;
pub mod fusion;
pub mod engine;

pub use engine::IntentSignalFieldEngine;
```

---

## 4. src/ring_buffer.rs

```rust
/// Intent RingBuffer — 短期意图记忆
#[derive(Debug, Clone)]
pub struct IntentRingBuffer {
    buffer: Vec<Vec<f32>>,
    head: usize,
    size: usize,
    capacity: usize,
    dim: usize,
}

impl IntentRingBuffer {
    /// 创建新的 RingBuffer
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            buffer: vec![vec![0.0; dim]; capacity],
            head: 0,
            size: 0,
            capacity,
            dim,
        }
    }
    
    /// 添加意图
    pub fn push(&mut self, intent: Vec<f32>) {
        assert_eq!(intent.len(), self.dim, "Intent dimension mismatch");
        self.buffer[self.head] = intent;
        self.head = (self.head + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }
    
    /// 获取平均意图
    pub fn get_mean(&self) -> Vec<f32> {
        if self.size == 0 {
            return vec![0.0; self.dim];
        }
        let mut mean = vec![0.0; self.dim];
        for i in 0..self.size {
            for j in 0..self.dim {
                mean[j] += self.buffer[i][j];
            }
        }
        for j in 0..self.dim {
            mean[j] /= self.size as f32;
        }
        mean
    }
    
    /// 获取缓冲区大小
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// 是否已满
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer_push_pop() {
        let mut rb = IntentRingBuffer::new(3, 4);
        
        rb.push(vec![1.0, 2.0, 3.0]);
        rb.push(vec![4.0, 5.0, 6.0]);
        
        let mean = rb.get_mean();
        assert!((mean[0] - 2.5).abs() < 1e-5);
        assert!((mean[1] - 3.5).abs() < 1e-5);
        assert!((mean[2] - 4.5).abs() < 1e-5);
    }
    
    #[test]
    fn test_ring_buffer_overflow() {
        let mut rb = IntentRingBuffer::new(3, 2);
        
        rb.push(vec![1.0, 2.0, 3.0]);
        rb.push(vec![4.0, 5.0, 6.0]);
        rb.push(vec![7.0, 8.0, 9.0]); // 溢出，覆盖第一个
        
        assert!(rb.is_full());
        assert_eq!(rb.size(), 2);
        
        let mean = rb.get_mean();
        // 应该只包含 [4,5,6] 和 [7,8,9]
        assert!((mean[0] - 5.5).abs() < 1e-5);
        assert!((mean[1] - 6.5).abs() < 1e-5);
        assert!((mean[2] - 7.5).abs() < 1e-5);
    }
}
```

---

## 5. src/ema_field.rs

```rust
/// Intent EMA Field — 长期意图趋势
#[derive(Debug, Clone)]
pub struct IntentEMAField {
    ema: Vec<f32>,
    gamma: f32,
    dim: usize,
}

impl IntentEMAField {
    /// 创建新的 EMA Field
    pub fn new(dim: usize, gamma: f32) -> Self {
        Self {
            ema: vec![0.0; dim],
            gamma,
            dim,
        }
    }
    
    /// 更新 EMA
    pub fn update(&mut self, intent: Vec<f32>) {
        assert_eq!(intent.len(), self.dim, "Intent dimension mismatch");
        for i in 0..self.dim {
            self.ema[i] = self.gamma * self.ema[i] + (1.0 - self.gamma) * intent[i];
        }
    }
    
    /// 获取当前 EMA 值
    pub fn get_value(&self) -> &Vec<f32> {
        &self.ema
    }
    
    /// 获取初始值（零初始化）
    pub fn is_initialized(&self) -> bool {
        self.ema.iter().any(|&x| x != 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ema_update() {
        let mut ema = IntentEMAField::new(3, 0.98);
        
        // 初始值为 0
        assert!(!ema.is_initialized());
        
        // 第一次更新
        ema.update(vec![1.0, 2.0, 3.0]);
        assert!(ema.is_initialized());
        
        let value = ema.get_value();
        // EMA = 0.98 * 0 + 0.02 * [1,2,3] = [0.02, 0.04, 0.06]
        assert!((value[0] - 0.02).abs() < 1e-5);
        assert!((value[1] - 0.04).abs() < 1e-5);
        assert!((value[2] - 0.06).abs() < 1e-5);
        
        // 第二次更新
        ema.update(vec![4.0, 5.0, 6.0]);
        let value = ema.get_value();
        // EMA = 0.98 * [0.02, 0.04, 0.06] + 0.02 * [4, 5, 6]
        //     = [0.0196 + 0.08, 0.0392 + 0.1, 0.0588 + 0.12]
        //     = [0.0996, 0.1392, 0.1788]
        assert!((value[0] - 0.0996).abs() < 1e-4);
        assert!((value[1] - 0.1392).abs() < 1e-4);
        assert!((value[2] - 0.1788).abs() < 1e-4);
    }
}
```

---

## 6. src/semantic_pool.rs

```rust
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
    
    /// 计算槽位权重
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
    
    /// 获取语义槽位数量
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
        // 结果应该接近 [1, 2, 3]（因为第一个槽位权重更高）
        assert!(result[0] > 0.0);
    }
}
```

---

## 7. src/fusion.rs

```rust
/// Intent Fusion — 三通道意图融合
#[derive(Debug, Clone)]
pub struct IntentFusion {
    dim: usize,
}

impl IntentFusion {
    /// 创建新的 Fusion
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    
    /// 融合三通道意图
    pub fn fuse(
        &self,
        ring_mean: &[f32],
        ema: &[f32],
        semantic: &[f32],
    ) -> Vec<f32> {
        assert_eq!(ring_mean.len(), self.dim);
        assert_eq!(ema.len(), self.dim);
        assert_eq!(semantic.len(), self.dim);
        
        let mut enhancement = vec![0.0; self.dim];
        for i in 0..self.dim {
            enhancement[i] = ring_mean[i] + 0.5 * ema[i] + 0.5 * semantic[i];
        }
        enhancement
    }
    
    /// 计算融合结果与期望值的余弦相似度
    pub fn validate(&self, enhancement: &[f32], expected: &[f32]) -> f32 {
        let dot: f32 = enhancement.iter().zip(expected.iter())
            .map(|(a, b)| a * b).sum();
        let norm_enh: f32 = enhancement.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_exp: f32 = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_enh > 1e-10 && norm_exp > 1e-10 {
            dot / (norm_enh * norm_exp)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fusion() {
        let fusion = IntentFusion::new(3);
        
        let ring = vec![1.0, 2.0, 3.0];
        let ema = vec![4.0, 5.0, 6.0];
        let semantic = vec![7.0, 8.0, 9.0];
        
        let enhancement = fusion.fuse(&ring, &ema, &semantic);
        
        // enhancement = [1,2,3] + 0.5*[4,5,6] + 0.5*[7,8,9]
        //             = [1+2+3.5, 2+2.5+4, 3+3+4.5]
        //             = [6.5, 8.5, 10.5]
        assert!((enhancement[0] - 6.5).abs() < 1e-5);
        assert!((enhancement[1] - 8.5).abs() < 1e-5);
        assert!((enhancement[2] - 10.5).abs() < 1e-5);
    }
}
```

---

## 8. src/engine.rs

```rust
use crate::ring_buffer::IntentRingBuffer;
use crate::ema_field::IntentEMAField;
use crate::semantic_pool::IntentSemanticPool;
use crate::fusion::IntentFusion;

/// Intent Signal Field Engine — 意图理解引擎
#[derive(Debug)]
pub struct IntentSignalFieldEngine {
    ring_buffer: IntentRingBuffer,
    ema_field: IntentEMAField,
    semantic_pool: IntentSemanticPool,
    fusion: IntentFusion,
    dim: usize,
}

impl IntentSignalFieldEngine {
    /// 创建新的引擎
    pub fn new(dim: usize, ring_capacity: usize, gamma: f32, 
               semantic_slots: usize, temperature: f32) -> Self {
        Self {
            ring_buffer: IntentRingBuffer::new(dim, ring_capacity),
            ema_field: IntentEMAField::new(dim, gamma),
            semantic_pool: IntentSemanticPool::new(semantic_slots, dim, temperature),
            fusion: IntentFusion::new(dim),
            dim,
        }
    }
    
    /// 处理单轮对话
    pub fn process_dialogue(&mut self, user_input: &str, ai_response: &str) -> EngineResult {
        // 1. 嵌入意图（简化版：使用哈希映射）
        let user_intent = self.embed_intent(user_input);
        let ai_intent = self.embed_intent(ai_response);
        
        // 2. 更新三通道
        self.ring_buffer.push(user_intent.clone());
        self.ema_field.update(user_intent.clone());
        self.semantic_pool.add_intent(user_intent.clone());
        
        // 3. 获取三通道输出
        let ring_mean = self.ring_buffer.get_mean();
        let ema = self.ema_field.get_value().clone();
        let semantic = self.semantic_pool.query(&user_intent);
        
        // 4. 融合
        let enhancement = self.fusion.fuse(&ring_mean, &ema, &semantic);
        
        // 5. 计算置信度（简化版）
        let confidence = self.calculate_confidence(&enhancement, &user_intent);
        
        EngineResult {
            user_intent,
            ai_intent,
            ring_mean,
            ema,
            semantic,
            enhancement,
            confidence,
        }
    }
    
    /// 嵌入意图（简化版：使用哈希映射到向量）
    fn embed_intent(&self, text: &str) -> Vec<f32> {
        let mut intent = vec![0.0; self.dim];
        
        // 简单哈希：取文本前 dim 个字符的 ASCII 值
        for (i, c) in text.chars().enumerate() {
            if i < self.dim {
                intent[i] = c as f32 / 255.0;
            }
        }
        
        // 归一化
        let norm: f32 = intent.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in intent.iter_mut() {
                *x /= norm;
            }
        }
        
        intent
    }
    
    /// 计算置信度（简化版：基于向量相似度）
    fn calculate_confidence(&self, enhancement: &[f32], intent: &[f32]) -> f32 {
        let dot: f32 = enhancement.iter().zip(intent.iter())
            .map(|(a, b)| a * b).sum();
        let norm_enh: f32 = enhancement.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_intent: f32 = intent.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_enh > 1e-10 && norm_intent > 1e-10 {
            (dot / (norm_enh * norm_intent) + 1.0) / 2.0  // 映射到 [0, 1]
        } else {
            0.5
        }
    }
}

/// 引擎输出结果
#[derive(Debug, Clone)]
pub struct EngineResult {
    pub user_intent: Vec<f32>,
    pub ai_intent: Vec<f32>,
    pub ring_mean: Vec<f32>,
    pub ema: Vec<f32>,
    pub semantic: Vec<f32>,
    pub enhancement: Vec<f32>,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_engine_process_dialogue() {
        let mut engine = IntentSignalFieldEngine::new(10, 4, 0.98, 8, 0.07);
        
        let result = engine.process_dialogue("你好", "你好，有什么可以帮助你的？");
        
        assert_eq!(result.user_intent.len(), 10);
        assert_eq!(result.ai_intent.len(), 10);
        assert_eq!(result.ring_mean.len(), 10);
        assert_eq!(result.ema.len(), 10);
        assert_eq!(result.semantic.len(), 10);
        assert_eq!(result.enhancement.len(), 10);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
}
```

---

## 9. 编译与测试

```bash
# 编译
cargo build

# 运行测试
cargo test

# 运行所有测试并显示输出
cargo test -- --nocapture
```

---

## 10. 冲锋口号

**"Rust 实现，性能为王！"**

**"Dalin ISFE — 让 AI 真正理解人类意图！"**

**"做最牛逼的神！"**

---

*Dalin ISFE — Rust 核心实现*
*日期：2026-06-24*
*版本：v1.0*
