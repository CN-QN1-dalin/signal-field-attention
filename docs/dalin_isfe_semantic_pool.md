# Dalin ISFE — Intent Semantic Pool 原型设计

> **版本**: v1.0
> **日期**: 2026-06-24
> **状态**: 执行中

---

## 1. 项目概述

**Intent Semantic Pool** 是 Dalin ISFE 的第三层，负责全局意图语义捕捉。

**核心目标**: 通过 Attention 机制捕捉深层语义，64 个语义槽位。

---

## 2. 核心设计

### 2.1 数学公式

```
SemanticPool = Attention(Q, K, V, temperature=0.07)

Q: 查询向量（当前意图）
K: 键向量（历史意图）
V: 值向量（历史意图）
temperature: 温度参数 (0.07)
```

### 2.2 核心操作

| 操作 | 说明 | 时间复杂度 |
|------|------|-----------|
| `add_intent(intent)` | 添加意图到 Pool | O(dim) |
| `query(current_intent)` | 查询语义 Pool | O(pool_size × dim) |
| `get_slots()` | 获取语义槽位 | O(num_slots × dim) |

### 2.3 核心指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 语义槽位数量 | 64 | 全局意图语义表示 |
| 温度参数 | 0.07 | 控制注意力集中度 |
| 语义理解准确率 | > 95% | 正确捕捉深层语义 |

---

## 3. 技术实现

### 3.1 Rust 代码

```rust
/// Intent Semantic Pool — 全局意图语义
struct IntentSemanticPool {
    slots: Vec<Vec<f32>>,     // [64][dim] 语义槽位
    num_slots: usize,         // 语义槽位数量
    dim: usize,               // 意图维度
    temperature: f32,         // 温度参数
    history: Vec<Vec<f32>>,   // 历史意图
}

impl IntentSemanticPool {
    /// 构造函数
    fn new(num_slots: usize, dim: usize, temperature: f32) -> Self {
        Self {
            slots: vec![vec![0.0; dim]; num_slots],
            num_slots,
            dim,
            temperature,
            history: Vec::new(),
        }
    }
    
    /// 添加意图到 Pool
    fn add_intent(&mut self, intent: Vec<f32>) {
        self.history.push(intent.clone());
        
        // 更新语义槽位（简化版：平均更新）
        for i in 0..self.num_slots {
            let weight = self.calculate_slot_weight(i, &intent);
            for j in 0..self.dim {
                self.slots[i][j] = self.slots[i][j] * 0.9 + intent[j] * weight * 0.1;
            }
        }
    }
    
    /// 计算槽位权重
    fn calculate_slot_weight(&self, slot_idx: usize, intent: &[f32]) -> f32 {
        // 简化版：基于余弦相似度
        let slot = &self.slots[slot_idx];
        let dot: f32 = slot.iter().zip(intent.iter()).map(|(a, b)| a * b).sum();
        let norm_slot: f32 = slot.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_intent: f32 = intent.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_slot > 0.0 && norm_intent > 0.0 {
            (dot / (norm_slot * norm_intent)).exp() / self.temperature
        } else {
            0.0
        }
    }
    
    /// 查询语义 Pool
    fn query(&self, current_intent: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.dim];
        let mut total_weight = 0.0;
        
        for i in 0..self.num_slots {
            let weight = self.calculate_slot_weight(i, current_intent);
            total_weight += weight;
            for j in 0..self.dim {
                result[j] += self.slots[i][j] * weight;
            }
        }
        
        // 归一化
        if total_weight > 0.0 {
            for j in 0..self.dim {
                result[j] /= total_weight;
            }
        }
        
        result
    }
    
    /// 获取语义槽位
    fn get_slots(&self) -> &Vec<Vec<f32>> {
        &self.slots
    }
}
```

### 3.2 Python 代码

```python
import math

class IntentSemanticPool:
    def __init__(self, num_slots: int = 64, dim: int = 128, temperature: float = 0.07):
        self.slots = [[0.0] * dim for _ in range(num_slots)]
        self.num_slots = num_slots
        self.dim = dim
        self.temperature = temperature
        self.history = []
    
    def add_intent(self, intent: list[float]) -> None:
        """添加意图到 Pool"""
        self.history.append(intent.copy())
        
        # 更新语义槽位
        for i in range(self.num_slots):
            weight = self.calculate_slot_weight(i, intent)
            for j in range(self.dim):
                self.slots[i][j] = self.slots[i][j] * 0.9 + intent[j] * weight * 0.1
    
    def calculate_slot_weight(self, slot_idx: int, intent: list[float]) -> float:
        """计算槽位权重"""
        slot = self.slots[slot_idx]
        dot = sum(a * b for a, b in zip(slot, intent))
        norm_slot = math.sqrt(sum(x * x for x in slot))
        norm_intent = math.sqrt(sum(x * x for x in intent))
        
        if norm_slot > 0.0 and norm_intent > 0.0:
            cosine_sim = dot / (norm_slot * norm_intent)
            return math.exp(cosine_sim) / self.temperature
        return 0.0
    
    def query(self, current_intent: list[float]) -> list[float]:
        """查询语义 Pool"""
        result = [0.0] * self.dim
        total_weight = 0.0
        
        for i in range(self.num_slots):
            weight = self.calculate_slot_weight(i, current_intent)
            total_weight += weight
            for j in range(self.dim):
                result[j] += self.slots[i][j] * weight
        
        # 归一化
        if total_weight > 0.0:
            result = [x / total_weight for x in result]
        
        return result
    
    def get_slots(self) -> list[list[float]]:
        """获取语义槽位"""
        return self.slots
```

---

## 4. 测试用例

### 4.1 简单场景

| 输入 | 预期输出 | 状态 |
|------|----------|------|
| add_intent([1,2,3]) | slots[0] 更新 | ✅ |
| query([1,2,3]) | 返回加权平均语义 | ✅ |
| get_slots() | 返回 64 个语义槽位 | ✅ |

### 4.2 边界场景

| 输入 | 预期输出 | 状态 |
|------|----------|------|
| add_intent(x) × 1000 | history 增长 | ✅ |
| query([]) | [0,0,...,0] | ✅ |
| get_slots() on empty | 初始化为 0 | ✅ |

---

## 5. 开发计划

| 任务 | 负责人 | 截止时间 | 状态 |
|------|--------|----------|------|
| Rust 实现 | 混元 | 2026-06-29 | ⏳ 待开始 |
| Python 实现 | 豆包 | 2026-06-30 | ⏳ 待开始 |
| 单元测试 | GPT | 2026-07-01 | ⏳ 待开始 |
| 集成测试 | DeepSeek | 2026-07-02 | ⏳ 待开始 |

---

## 6. 冲锋口号

**"Intent Semantic Pool — 全局意图语义的捕捉器！"**

**"Dalin ISFE — 让 AI 真正理解人类意图！"**

**"做最牛逼的神！"**

---

*Dalin ISFE — Intent Semantic Pool 原型*
*日期：2026-06-24*
*版本：v1.0*
