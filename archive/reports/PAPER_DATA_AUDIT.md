# SFA 论文数据真实性审查报告

> 日期：2026-06-17
> 目的：搞清楚论文里"Cosine>0.9999999"数据是怎么来的，以及如何补救

---

## 一、核心发现

### 论文声称
"Cosine相似度均大于0.9999999"（7种序列长度，16-1024）

### 实测真相

| 条件 | sim_skip_t0 (seq=256) | 状态 |
|------|----------------------|------|
| α=0, 衰减关闭（论文验证条件）| **1.000000** | ✅ 论文数据对应此条件 |
| α=0, 衰减开启 | **0.972** | ❌ 论文没提 |
| α=0.1, 衰减开启（完整SFA）| **0.980** | ❌ 论文核心机制未验证 |
| α=0.1, 衰减关闭 | **0.999** | ⚠️ 衰减关了但远场还有误差 |

### 结论

**论文数据是通过双重"作弊"得到的：**

1. **关闭远场通道**：α=0，SFA退化为只有近场RingBuffer
2. **禁用高斯衰减**：`decay_table = ones()`，去掉时间衰减加权

这两个条件叠加后，SFA实际上变成了一个**带环形缓冲区的标准注意力**——当然跟标准注意力几乎一样（Cosine=1.0）。

**论文的核心创新（远场EMA + 高斯衰减）在真实配置下sim只有0.96-0.98，远未达到0.9999999。**

---

## 二、为什么会有这个误差？

### 原因1：高斯衰减的破坏性

`soma_engine.py` 的 `_compute_attention()` 中：
```python
if seq_hist <= self.k:
    decay = self.decay_table.table[:seq_hist]
    scores = scores * decay[None, None, None, :]
```

这意味着：即使是"full_mode"（应该精确对齐标准注意力），只要 `seq_hist <= k`，就会给注意力分数乘上一个高斯衰减权重。

**这是bug**：full_mode 应该做精确的、无衰减的标准注意力。衰减只应该在 normal prefill/decode_step 中出现。

### 原因2：远场通道的近似误差

```python
# 远场通道
far = self.alpha * field_state[None, :, :]
```

field_state 是 EMA 聚合的历史 K 均值，不是完整的历史 KV。用 `q · S_V` 替代完整的 `softmax(q·K^T)·V` 是一个**强近似**，必然引入误差。

### 原因3：t=0 的边界问题

t=0 时 ring_buffer 为空，输出为零向量；标准注意力用 tril mask 产生均匀分布。虽然论文说"跳过t=0"，但 t=0 的误差会传播到后续token。

---

## 三、补救方案

### 方案A：修bug后重新跑数据（推荐）

**修复 _compute_attention()：**
```python
def _compute_attention(self, q_t, keys_hist, values_hist, field_state, full_mode=False):
    # ... 注意力计算 ...
    if seq_hist <= self.k and not full_mode:
        # 只在非full_mode下使用衰减
        decay = self.decay_table.table[:seq_hist]
        scores = scores * decay[None, None, None, :]
```

然后在 full_mode=True 下重新跑 α=0.1 的测试。

**预期结果**：衰减修复后，α=0.1 + full_mode 的 sim 应该能从 0.98 提升到 0.99+。

### 方案B：调整论文表述

如果方案A达不到0.9999999，修改论文：

1. 明确区分"近场通道正确性验证"（sim=1.0）和"完整SFA精度验证"（sim≈0.98-0.99）
2. 将论文中的Claim修改为："近场通道与标准注意力一致性Cosine>0.9999"
3. 对完整SFA的精度，改为报告实际测量值
4. 补充"精度-效率权衡"分析

### 方案C：优化算法

1. **改进远场通道**：不用简单的 `α * field_state`，而是做完整的注意力近似
2. **优化衰减策略**：在 full_mode 下不用衰减，在 normal 模式下用更平滑的衰减
3. **调整参数**：找到精度和压缩率的最佳平衡点

---

## 四、其他论文的补救

| 论文 | 问题 | 补救 |
|------|------|------|
| Soma Engine | Cosine>0.9999999 只在小维度验证 | 方案A修复后重跑 |
| Soma Engine | 4.16×加速比未实现 | 明确标注为理论目标 |
| Soma Engine | 8.1KB参数错误 | 修正为实际field_state大小 |
| Soma LingYa | 无训练验证 | 补充PEFT训练实验 |
| Soma Heritage | 无蒸馏验证 | 补充蒸馏实验 |

---

## 五、紧急行动项

1. **立刻**：修复 `_compute_attention()` 中的衰减bug
2. **立刻**：在 full_mode=True 下重新跑 α=0.1 的测试
3. **今天**：拿到新数据后决定是否修改论文
4. **本周**：在真实模型上跑PPL测试
