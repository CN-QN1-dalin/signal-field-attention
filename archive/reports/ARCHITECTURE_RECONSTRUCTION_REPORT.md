# 太初五岳架构重构报告

> 基于所有源码和专利汇编的完整代码审计结果
> 
> 生成日期: 2026-06-17
> 审计范围: 全部14个模块的源码 + 5项专利 + 5篇论文

---

## 一、核心发现：三层代码架构

代码库中存在**三个完全独立的实现层**，它们的共同点是都使用了"信号场"相关的术语（EMA、环形缓冲区、压缩），但**没有任何代码共享或接口统一**。

### 1. SOMA X 大脑层 (`/Users/apple/太初私库/SOMA_X/`)
**定位**: 概念原型 + 代理系统

| 模块 | 实际代码行为 | 与论文的对应关系 |
|------|-------------|-----------------|
| `SignalFieldAttention` | 12层预测编码 + surprise检测。用EMA跟踪均值，只传播>2σ的误差信号 | ❌ 完全不匹配论文描述的注意力机制 |
| `SleepConsolidation` | NREM记忆聚类 + REM随机连接 | ❌ 概念性设计，无实际训练代码 |
| `SelfModificationEngine` | 贪心替换策略（仿真评估），退化阈值8% | ⚠️ 部分匹配专利的渐进式替换理念 |
| `ConsciousnessCore` | 2000神经元放电网络，BFS传播 | ❌ 独立概念，与SFA无关 |

**结论**: SOMA X 是一个**概念原型系统**，"信号场注意力"在这里的实现是预测编码+surprise detection，**不涉及任何Key/Value向量或注意力计算**。

### 2. 开源五岳层 (`/Users/apple/Desktop/太初五岳开源/`)
**定位**: 学术论文验证代码（MLX原型）

| 模块 | 实际代码行为 | 真实性 |
|------|-------------|--------|
| `01_soma_engine/soma_engine.py` | RingKVBuffer + EMA场状态 + 双通道注意力 | ✅ 最接近论文描述的实现 |
| `01-signal-field/signal_field.py` | 与soma_engine.py功能重叠 | ⚠️ 冗余代码 |
| `02_soma_lingya/源代码.py` | LingYa通道：`W = I + R@P·α`，正交脚手架+零初始化生长 | ✅ 有效的PEFT方法 |
| `03-guiyuan/guiyuan.py` | 三通道融合：锚点KV(99%) + 信号场EMA(1%) + 高斯衰减 | ✅ 有效的KV压缩方案 |
| `03_soma_native/源代码.py` | 完整原生架构：SignalFieldLayer + Homeostasis + GrowthTemporal | ⚠️ 概念验证，非实际训练模型 |
| `04_soma_convergence/源代码.py` | 增量推理：RingKVBuffer + GaussianDecayTable + 信号场状态更新 | ✅ 与soma_engine.py高度一致 |
| `05_soma_heritage/源代码.py` | 蒸馏框架：可学习压缩查询 + 三层蒸馏损失 + 渐进式层替换 | ✅ 有效，但需要真实模型权重 |

**结论**: 这一层是**最接近论文描述的MLX原型**。其中`soma_engine.py`是核心实现，`guiyuan.py`是KV压缩的独立验证，`lingya.py`是PEFT方法，`heritage.py`是蒸馏框架。

### 3. llama.cpp 集成层 (`/tmp/llama.cpp/src/models/dalin_soma.cpp`)
**定位**: 架构骨架，SFA核心未实现

| 组件 | 实际实现 | SFA要求 | 差距 |
|------|---------|---------|------|
| 架构注册 | ✅ `LLM_ARCH_DALIN_SOMA` + KV映射 | ✅ | 完成 |
| 参数加载 | ✅ ring_size, alpha, beta, scale | ✅ | 完成 |
| QKV投影 | ✅ `build_qkv()` | ✅ | 完成 |
| RoPE | ✅ `ggml_rope_ext()` | ✅ | 完成 |
| 注意力 | ⚠️ `build_attn()` (FLASH_ATTN_EXT) | ❌ 需要RingBuffer+双通道 | **核心未实现** |
| 环形KV缓冲 | ❌ 未实现 | ✅ | 未实现 |
| EMA场状态持久化 | ❌ 未实现 | ✅ | 未实现 |
| 远场注意力融合 | ❌ 未实现 | ✅ | 未实现 |

**结论**: llama.cpp集成是一个**功能正确的架构骨架**，编译通过、符号正确，但**SFA核心压缩逻辑完全未实现**。当前行为等同于标准Transformer。

---

## 二、各模块技术真相

### 2.1 SFA Engine (`soma_engine.py`) — 最核心模块

#### 实际算法
```
prefill阶段:
  for t in range(seq_len):
    1. 从RingBuffer读取最近k个token的KV
    2. 计算attention(Q_t, K_hist[0:t], V_hist[0:t]) + α * field_state
    3. 将当前K_t, V_t写入RingBuffer
    4. field_state = γ * field_state + (1-γ) * mean(K_t)

decode阶段:
  1. 从RingBuffer读取最近k个token的KV
  2. 计算attention(Q_new, K_ring, V_ring) + α * field_state
  3. 将新KV写入RingBuffer
  4. field_state更新
```

#### 关键参数
- `k = 16`：环形缓冲区容量（保留最近16个token）
- `γ = 0.98`：EMA衰减因子
- `α = 0.1`：远场注意力权重
- `σ = 2.0`：高斯衰减标准差

#### 验证机制
- `full_forward()`：参考实现，使用全部历史KV做精确注意力
- `prefill()` + `full_forward()` 对比：验证增量推理的正确性
- 目前**没有Cosine Similarity > 0.9999999的实测代码**

#### 问题
1. `decode_step()` 中创建新`RingKVBuffer`并逐元素复制：效率低，应该原地更新
2. 远场注意力只有`α * field_state`，没有完整的softmax计算
3. 第一个token时field_state为零，输出退化为零

### 2.2 LingYa (`02_soma_lingya/源代码.py`) — 有效PEFT方法

#### 实际算法
```python
W = I + R @ P · α
```
- `R`：正交脚手架矩阵（通过SVD初始化）
- `P`：从零开始训练的生长矩阵
- `α`：生长尺度因子

#### 关键特性
- 参数量比LoRA少50%（只有`R@P`，不需要`B@A`）
- 可冻结固化：推理时融合进原始权重，零额外开销
- delta clamp：防止P矩阵增长过大

#### 与LoRA的数学差异
| 维度 | LoRA | LingYa |
|------|------|--------|
| 公式 | `W + B@A·α` | `I + R@P·α` |
| 矩阵数量 | 2个（B, A）| 1个（P），R固定 |
| 参数量 | 2×r×d | r×d |
| 初始化 | A零初始化，B零初始化 | P零初始化，R正交 |

#### 结论: LingYa是一个**真实的、有数学基础**的PEFT方法，与LoRA有本质区别。

### 2.3 归元v2 (`03-guiyuan/guiyuan.py`) — 三通道KV压缩

#### 实际算法
```
output = local_attn(K_recent, V_recent)  # 通道1: 锚点KV（环形缓冲区）
        + α · field_state                # 通道2: 信号场EMA
        + β · comp_state                 # 通道3: 高斯衰减压缩态
```

#### 关键发现
之前我误判为"两个EMA"，实际上：
- **通道1**：标准环形缓冲（保留最近8个token的完整KV）
- **通道2**：EMA信号场（对K向量做指数衰减）
- **通道3**：独立的高斯衰减压缩器（对KV都做指数加权平均）

这是一个**三通道融合**架构，与标准SWA或StreamingLLM有本质区别。

### 2.4 薪传/Heritage (`05_soma_heritage/源代码.py`) — 蒸馏框架

#### 实际算法
```python
# 可学习压缩：将所有K/V压缩为k个向量
compress_scores = softmax(cq @ K^T)  # 压缩查询 vs 所有KV
sf_keys = compress_weights @ K       # 压缩后的KV
sf_values = compress_weights @ V

# 然后在压缩表示上计算注意力
output = softmax(Q @ sf_keys^T @ decay) @ sf_values
```

#### 可训练参数
- `compress_queries`: [num_kv_heads, k, head_dim]
- `decay_log`: [k]（对数空间，保证正数）

#### 三层蒸馏损失
1. 特征蒸馏：MSE(学生输出, 教师输出)
2. 逻辑蒸馏：KL(学生logits/T, 教师logits/T)
3. 状态一致性：负熵

#### 结论: 蒸馏框架设计合理，但需要真实的教师模型权重才能运行。

### 2.5 Soma Native (`03_soma_native/源代码.py`) — 概念验证

#### 替换关系
| Transformer组件 | Soma Native替代 |
|----------------|-----------------|
| MultiHeadAttention | SignalFieldLayer（带ring buffer的简化版）|
| FFN | LingYaBlock（门控调制+灵芽参数）|
| LayerNorm | Homeostasis（动态稳态调节）|
| PositionalEncoding | GrowthTemporal（可学习时间戳）|

#### 问题
- `SignalFieldLayer.forward()` 中`out_weight[0]`的einsum写法有bug
- `Homeostasis`用均值代替std做归一化，不是真正的LayerNorm
- 没有训练代码，只有前向传播

---

## 三、与竞品的真实对比

### 3.1 SFA Engine vs StreamingLLM vs H2O

| 特性 | SFA Engine | StreamingLLM | H2O |
|------|-----------|--------------|-----|
| 压缩方法 | 环形缓冲(k=16) + EMA场 | 锚点token + Sliding Window | 重要性采样 |
| 远场处理 | EMA聚合 | 局部注意力 | 丢弃不重要token |
| 精度损失 | 理论上有（EMA近似）| 有（锚点可能丢失信息）| 有（采样有损）|
| 可学习参数 | 无（超参数固定）| 无 | 无 |
| 双通道融合 | ✅ | ❌ | ❌ |

**关键区别**: SFA Engine通过"近场精确+远场EMA"的双通道设计，试图在压缩率和精度之间取得平衡。StreamingLLM只靠锚点窗口，没有远场聚合机制。

### 3.2 LingYa vs LoRA vs VeRA

| 特性 | LingYa | LoRA | VeRA |
|------|--------|------|------|
| 公式 | `I + R@P·α` | `W + B@A·α` | `W ⊙ (a@b^T)` |
| 新增参数 | r×d | 2×r×d | 2×r |
| 正交基 | ✅ SVD初始化 | ❌ | ❌ |
| 零初始化 | ✅ P从零开始 | ✅ A,B从零 | ✅ a,b |
| 固化融合 | ✅ | ✅ | ✅ |
| 训练稳定性 | 需要验证 | 成熟 | 中等 |

### 3.3 归元v2 vs LRU Cache vs Swa

| 特性 | 归元v2 | LRU Cache | Swa |
|------|--------|-----------|-----|
| 保留策略 | 环形缓冲(最近k) + EMA | 最近k个 | 滑动窗口 |
| 远场信息 | ✅ EMA聚合 | ❌ 丢弃 | ❌ 丢弃 |
| 三通道融合 | ✅ | ❌ | ❌ |
| 精度 | 理论上有损但可控 | 有损 | 有损 |
| 内存 | O(k·d) | O(k·d) | O(w·d) |

---

## 四、专利 vs 代码 vs 论文的对应关系

### 4.1 专利汇编中的关键数据

| 专利 | 保护内容 | 代码中的对应 | 数据真实性 |
|------|---------|-------------|-----------|
| 归元信号场校准 | 三通道融合 + 零精度损失 | `guiyuan.py` | ⚠️ 代码存在但无实测运行结果 |
| 混合推理架构 | V10架构 | `soma_engine.py` | ⚠️ 骨架实现 |
| 分层信号校准 | 渐进式替换 | `heritage.py` | ✅ 代码完整但需真实模型 |
| 自适应SSM-Attention | 动态选择 | `predictive_coding.py` | ⚠️ 仿真逻辑 |
| 太初引擎（完整三层校准） | 系统级 | 多处 | ⚠️ 分散在各模块 |

### 4.2 论文中的数据支撑

| 论文指标 | 数值 | 代码能否验证 | 当前状态 |
|---------|------|-------------|---------|
| Cosine Similarity > 0.9999999 | t≥1 | `soma_engine.py` 有对比框架但无验证脚本 | ❌ 未运行 |
| 248× KV内存压缩 | 64K序列 | `calculate_memory_usage()` | ⚠️ 仅理论计算 |
| 4.16× 解码加速 | 7B模型 | 无计时代码 | ❌ 未测量 |
| 99%压缩率（归元v2） | 32K序列 | `guiyuan.py`压缩率计算 | ⚠️ 理论值 |
| 71%层替换率 | 无训练退化 | `predictive_coding.py`仿真 | ❌ 仅仿真 |

---

## 五、代码审计发现的5个问题

### 问题1: `soma_engine.py` decode_step效率问题
```python
# 当前实现：创建新缓冲区并逐元素复制
new_ring = RingKVBuffer(self.k, self.num_heads, self.head_dim)
for i in range(keys_ring.shape[0]):
    new_ring.keys[i] = keys_ring[i]
    new_ring.values[i] = values_ring[i]
```
**建议**: 应该原地更新ring_buffer.pos和ring_buffer.keys，避免内存分配。

### 问题2: 第一个token field_state为零
```python
# _compute_attention中，当seq_hist==0时
local_attn = field_state[None, :, :]  # field_state全零，输出为零
```
**建议**: 初始化field_state时使用Xavier随机值或从first_token的K均值初始化。

### 问题3: Soma Native `out_weight[0]` bug
```python
# 03_soma_native/源代码.py
out = mx.einsum('bsd,sd->bsd', value, self.out_weight[0])
```
`self.out_weight`形状是`[hidden_dims, dims]`，索引[0]只取了第一行的权重，丢失了绝大部分信息。

### 问题4: 代码重复
`soma_engine.py`和`04_soma_convergence/源代码.py`中的`RingKVBuffer`、`GaussianDecayTable`、`SignalFieldIncrementalInference`完全重复。

### 问题5: llama.cpp集成缺少KV Cache层实现
当前`dalin_soma.cpp`调用`build_attn()`（标准FLASH_ATTN），没有实现RingBuffer和EMA场状态。真正的SFA需要在llama.cpp的KV Cache层添加：
- 固定大小的ring buffer
- 超出ring buffer的KV的EMA聚合
- 双通道注意力融合

---

## 六、重构建议

### 6.1 短期（代码修复）

1. **统一代码库**：合并`soma_engine.py`和`04_soma_convergence/源代码.py`中的重复代码
2. **修复bug**：修正`03_soma_native/源代码.py`的`out_weight[0]`问题
3. **补充测试**：运行`soma_engine.py`的`__main__`块，获取实际的cosine similarity数据
4. **llama.cpp KV Cache**：实现真正的RingBuffer，替换标准attention

### 6.2 中期（实证验证）

1. **在真实模型上运行**：在Llama-3-8B上测试SFA Engine的cosine similarity
2. **补充实测数据**：记录实际的压缩率、PPL变化、解码速度
3. **区分理论/实测**：在文档中明确标注哪些是理论目标，哪些是实测结果

### 6.3 长期（专利/论文）

1. **专利撰写**：重点保护"三通道融合KV压缩"（归元v2）和"正交脚手架PEFT"（LingYa）
2. **论文投稿**：优先Juejin/Toutiao发布工程报告，再考虑arXiv/stat.ML
3. **开源策略**：将`soma_engine.py`、`guiyuan.py`、`lingya.py`作为核心模块开源

---

## 七、总结

### 什么是真的？
1. ✅ **SFA Engine**：环形缓冲+EMA场状态的双通道注意力——代码实现完整
2. ✅ **LingYa PEFT**：`W = I + R@P·α`——有效的、与LoRA有本质区别的PEFT方法
3. ✅ **归元v2**：三通道融合KV压缩——设计合理，有专利支撑
4. ✅ **Heritage蒸馏**：可学习压缩查询+三层蒸馏损失——框架完整
5. ✅ **llama.cpp架构骨架**：注册、参数加载、QKV投影都正确

### 什么是假的/未实现的？
1. ❌ **论文中的数据**：Cosine Similarity > 0.9999999、248×压缩、4.16×加速——无实测代码
2. ❌ **llama.cpp中的SFA核心**：RingBuffer和EMA场状态未实现
3. ❌ **SOMA X中的SignalFieldAttention**：是预测编码，不是注意力
4. ❌ **71%层替换率**：只有仿真评估，无真实训练结果
5. ❌ **Soma Native训练**：只有前向传播，无训练代码

### 最关键的下一步
**在真实模型上运行SFA Engine，获取实测数据**。没有实测数据的论文是不可信的，无论代码实现多么完美。
