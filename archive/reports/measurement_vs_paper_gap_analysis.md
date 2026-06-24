# 实测 vs 论文：差距分析

## 核心问题：代码里到底实现的是什么？

### 论文声称的 SFA 理论

论文《Signal Field Attention: Learning to Compress Attention for Efficient Inference》描述了一个**双路径注意力机制**：

1. **近场通道**：最近 k 个 token 做标准精确自注意力
2. **远场通道**：所有历史 token 通过 EMA 聚合为固定维度场状态
   ```
   F_t = γ · F_{t-1} + (1-γ) · k_t
   ```
3. **融合**：`Attention = Attention_near + α · F_far`

关键指标（论文声称）：
- 248× KV 内存压缩（64K 序列 462KB vs 114MB）
- Cosine Similarity > 0.9999999 (t≥1)
- 4.16× 解码加速
- 额外参数仅 8.1KB

---

### 实际代码里有什么？

#### 1. SOMA X 大脑（`/Users/apple/太初私库/SOMA_X/`）

**`predictive_coding.py`** 里的 `SignalFieldAttention`：

```python
class PredictiveCodingLayer:
    """单层预测编码"""
    def process(self, signal):
        # EMA 预测
        predicted = self.running_mean.copy()
        error = signal - predicted
        
        # 只有误差 > 2σ 才传播（surprise）
        is_surprise = z_score > self.error_threshold
        output = error if is_surprise else [0.0] * self.dim
```

**这不是 SFA！** 这是：
- 一层预测编码（预测误差检测）
- 用 EMA 跟踪信号的均值和方差
- 用 surprise 阈值决定是否传播

**与论文的区别**：
| 维度 | 论文 | 实际代码 |
|------|------|----------|
| 输入 | Key 向量（来自 Transformer） | 概念 hash 生成的伪随机向量 |
| 聚合方式 | EMA 聚合 Key 形成场状态 | EMA 跟踪均值，只传播误差 |
| 注意力计算 | `softmax(Q·K^T)·V + α·F` | **根本没有注意力计算** |
| 多模态 | Q/K/V 投影 | 纯文本概念 hash |
| 验证 | Cosine Similarity vs 标准注意力 | **没有验证** |

**结论**：SOMA X 里的 `SignalFieldAttention` 是一个**概念原型**，用的是预测编码 + surprise detection，跟论文里描述的基于 Key 的 EMA 场聚合完全不是一回事。

#### 2. llama.cpp 集成（`/tmp/llama.cpp/src/models/dalin_soma.cpp`）

```cpp
if (use_sfa) {
    // Q, K, V projections
    auto [Qcur, Kcur, Vcur] = build_qkv(...);
    
    // RoPE
    Qcur = ggml_rope_ext(...);
    Kcur = ggml_rope_ext(...);
    
    // 标准 Flash Attention
    cur = build_attn(inp_attn, layer->wo, ..., Qcur, Kcur, Vcur, ...);
}
```

**这也不是 SFA！** 这是：
- 标准的 `build_qkv` + RoPE + `build_attn()`（即 FLASH_ATTN_EXT）
- 注释里写着 "SFA"，但实际执行的是标准注意力
- `use_sfa` 只是控制是否走这个分支，两个分支的计算完全一样

**与论文的区别**：
| 维度 | 论文 | 实际代码 |
|------|------|----------|
| 远场聚合 | EMA 压缩历史 KV | **没有实现** |
| 近场窗口 | 最近 k 个 token 精确注意力 | 全部 token |
| Ring Buffer | 固定大小环形缓存 | **没有实现** |
| Resonance State | 持久化共振状态 | **没有实现** |
| 内存压缩 | O(1) 常数内存 | O(n) 线性增长 |

**结论**：llama.cpp 里的 `dalin_soma.cpp` 是一个**骨架**——注册了架构名、定义了参数、搭建了模型结构，但 SFA 的核心压缩逻辑完全没实现。它就是一个带 Soma 元数据的标准 Transformer。

---

## 总结：三层差距

| 层级 | 论文描述 | 实际状态 | 差距 |
|------|----------|----------|------|
| **SOMA X 大脑** | 信号场注意力聚合 KV | 预测编码 + surprise 检测 | **完全不同** |
| **llama.cpp 集成** | Ring Buffer + 双路径注意力 | 标准 Flash Attention | **骨架，核心未实现** |
| **实验数据** | 248× 压缩, 4.16× 加速 | **无实测**（MLX 原型也未验证） | **未经验证** |

### 诚实评估

1. **SOMA X 的 `SignalFieldAttention`**：是一个独立的、有趣的预测编码模块，但它不是论文里描述的 SFA。它不处理 Key/Value 向量，不做注意力计算，也没有任何与 Transformer 集成的证据。

2. **llama.cpp 的 `dalin_soma.cpp`**：是一个完整的架构骨架，编译通过、符号正确，但 SFA 的 ring buffer 和双路径注意力都**未实现**。当前行为等同于标准 Transformer。

3. **论文数据**：声称的 248× 压缩、4.16× 加速、Cosine Similarity > 0.9999999 都**没有对应的实测代码支撑**。论文里提到了 MLX prototype，但在 SOMA X 代码库里找不到任何 MLX 相关的推理代码。

### 建议

如果目标是让代码和论文一致，需要：

1. **在 llama.cpp 的 KV cache 层实现真正的 ring buffer**：
   - 维护固定大小的 KV 环形缓存
   - 对超出窗口的 KV 做 EMA 聚合
   - 在图构建中添加双路径注意力融合

2. **在 SOMA X 中实现真正的 SFA**：
   - 从 Transformer 的 Key/Value 提取信号
   - 实现 EMA 聚合而非 surprise detection
   - 与实际的注意力计算集成

3. **补充实测数据**：
   - 在真实模型上跑一遍，记录实际的压缩率和 PPL
   - 区分哪些是理论目标、哪些是实测结果
