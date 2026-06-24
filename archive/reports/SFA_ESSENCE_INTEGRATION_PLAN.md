# SFA 精华集成计划：将五大模块核心优势整合到 llama.cpp

## 概述

本文档梳理太初五岳项目中五个核心模块的实际代码能力，提取可移植到 llama.cpp 的精华部分，制定具体的集成方案。

---

## 一、各模块核心能力盘点

### 1. SFA Engine (`01_soma_engine/SFA_Metal.cpp/.metal`)

**实际实现了什么：**
- ✅ **Ring Buffer KV 管理**：固定大小环形缓冲区，write/read 按环形索引管理
- ✅ **双通道注意力融合**：`near_field + alpha * field_state`
- ✅ **EMA 远场聚合**：`F_t = γ · F_{t-1} + (1-γ) · k_t`
- ✅ **Metal GPU 内核**：6个核函数（near_field_attn, ema_update, dual_path_fusion, qkv_project, output_project, ring_write）
- ✅ **CPU 参考实现**：完整的 prefill + decode_step 流程
- ✅ **Correctness 验证**：cosine similarity 对比标准注意力

**核心算法（decode_step）：**
```
1. QKV 投影新 token
2. 从 ring buffer 读最近 k 个 KV
3. 计算近场注意力: softmax(Q·K_ring^T/√d) · V_ring
4. 计算远场贡献: α · field_state (EMA聚合的历史K)
5. 融合: output = near_field + α · far_field
6. 输出投影
7. 更新 ring buffer (环形写入)
8. 更新 field_state: F = γ·F + (1-γ)·k_new
```

### 2. Soma LingYa (`02_soma_lingya/源代码.py`)

**实际实现了什么：**
- ✅ **正交脚手架矩阵 R**：SVD 分解保证正交性
- ✅ **零初始化生长矩阵 P**：P 从零开始训练
- ✅ **三种通道类型**：ROOT(单位阵), BRANCH(正交), LEAF(低秩随机)
- ✅ **Delta Clamp 正则化**：限制 ||P||_fro ≤ max_growth
- ✅ **固化(freeze)机制**：训练后将 W = I + R@P·α 融合进原始权重，推理零开销
- ✅ **参数效率**：比 LoRA 少 50% 参数（单矩阵 vs 双矩阵）

**核心公式：**
```
W_effective = I + R @ P · α  (其中 R 正交, P 从零初始化)
冻结后: W_frozen = I + R@P_final·α  → 融合进原始权重，无额外计算
```

### 3. Soma Convergence (`03_soma_convergence/源代码.py`)

**实际实现了什么：**
- ✅ **RingKVBuffer 类**：完整的环形缓冲区抽象，支持未满/已满两种读取模式
- ✅ **GaussianDecayTable**：预计算高斯衰减权重 `exp(-i²/(2σ²))`
- ✅ **Prefill/Decode 一致性**：prefill() 和 full_forward() 数学等价
- ✅ **状态序列化**：get_state()/load_state() 支持持久化
- ✅ **O(1) decode_step**：单步解码不依赖序列长度

**关键设计决策：**
- 近场注意力使用高斯衰减加权，比均匀窗口更平滑
- decode_step 中 ring_buffer 的 read 返回的是按时间排序的有效 KV
- 场状态更新在注意力计算之后，确保使用上一时刻的状态

### 4. Soma Native (`04_soma_native/源代码.py`)

**实际实现了什么：**
- ⚠️ **SignalFieldLayer**：概念性实现，但使用了简化版注意力（非标准 softmax）
- ✅ **Homeostasis 稳态调节**：动态活跃度监控 + 自适应调节 `reg = target / activity`
- ✅ **GrowthTemporal 生长时序**：可学习的时间戳 + 传统正弦位置编码混合
- ❌ **LingYaBlock**：门控 FFN 概念，但实现有 bug（`out_weight[0]` 索引错误）

**有价值的思想：**
```python
# Homeostasis: 替代 LayerNorm 的动态平衡
activity = mean(|x|)
regulation = target_activity / activity  # 反向调节
x_normalized = x * regulation

# GrowthTemporal: 位置编码可随训练生长
# 预计算正弦频率 + 可学习时间戳表
```

### 5. Soma Heritage (`05_soma_heritage/源代码.py`)

**实际实现了什么：**
- ✅ **SignalFieldAttentionGQA**：可学习压缩注意力，用 k 个压缩查询替代完整 KV
- ✅ **三层蒸馏损失**：特征蒸馏(MSE) + 逻辑蒸馏(KL) + 状态一致性(负熵)
- ✅ **渐进式替换策略**：从浅层到深层逐层替换注意力
- ✅ **Xavier 初始化 + 冻结权重继承**：从原始注意力层复制 QKV/out 权重

**核心创新：**
```python
# 可学习压缩：用 k 个可学习的 compress_queries 压缩所有 KV
compress_scores = softmax(Q_compress @ K^T / √d)
sf_keys = compress_weights @ K   # [batch, heads, k, d]
sf_values = compress_weights @ V # [batch, heads, k, d]
# 然后在压缩表示上计算注意力
```

### 6. 归元v2 (`03-guiyuan/guiyuan.py`)

**实际实现了什么：**
- ✅ **GaussianCompressor**：纯 EMA 累积压缩 `K_acc = γ·K_acc + (1-γ)·K_new`
- ✅ **SignalFieldEnhancedCompressor**：三通道设计（锚点KV + 累积状态 + 信号场）
- ✅ **Prefill/Decode 一致性验证**：证明逐步压缩与一次性压缩数学等价
- ✅ **压缩率计算器**：理论压缩比分析

**与 SFA Engine 的区别：**
- 归元v2 是**纯压缩**：把 KV 压成单一向量
- SFA Engine 是**双通道**：保留近场精确 KV + 远场压缩状态
- 归元v2 精度损失更大，但实现更简单

---

## 二、llama.cpp 当前状态与差距

### 当前 `dalin_soma.cpp` 做了什么：
1. ✅ 架构注册（LLM_ARCH_DALIN_SOMA）
2. ✅ KV 键映射（soma.ring_size, soma.alpha, soma.beta 等）
3. ✅ 标准 QKV 投影 + RoPE
4. ✅ 标准 FLASH_ATTN_EXT 注意力
5. ❌ **没有 Ring Buffer KV 管理**
6. ❌ **没有 EMA 场状态累积**
7. ❌ **没有双通道注意力融合**
8. ❌ **没有 LingYa 生长矩阵**
9. ❌ **没有 Homeostasis 稳态调节**
10. ❌ **没有 Heritage 可学习压缩**

### 本质：当前 llama.cpp 集成只是一个"带 Soma 元数据标签的标准 Transformer"

---

## 三、集成方案：分阶段实施

### Phase 1: 核心 SFA（来自 SFA Engine + Convergence）

**目标**：在 llama.cpp 中实现真正的双通道注意力

#### 3.1 Ring Buffer KV Cache 层

**移植来源**：`SFA_Metal.cpp` 的 `CPUSFAModule::ring_k_/ring_v_` + `Convergence` 的 `RingKVBuffer`

**实现方式**：扩展 `llama_kv_cache_iswa` 或新建 `llama_kv_cache_sfa`

```cpp
// 新增结构：SFA KV Cache
struct soma_ring_buffer {
    ggml_tensor* keys;    // [ring_size, num_heads, head_dim]
    ggml_tensor* values;  // [ring_size, num_heads, head_dim]
    int32_t ring_pos;     // 当前写入位置
    int32_t ring_size;    // 有效数据数量
    int32_t capacity;     // 缓冲区容量 (k)
};
```

**关键修改点**：
- 在 `llama_kv_cache` 基类中添加 `soma_ring_buffer` 成员
- 重写 `seq_rm`/`seq_cp` 处理环形缓冲区的索引映射
- `init_batch` 时为 SFA 层分配固定大小的 ring buffer

#### 3.2 双通道注意力图构建

**移植来源**：`SFA_Metal.cpp` 的 `prefill()` 和 `decode_step()` 中的注意力计算

**在 `dalin_soma.cpp` 的 graph builder 中**：

```cpp
// 近场通道：ring buffer 中的精确 KV
ggml_tensor* near_field = ggml_flash_attn_ext(
    ctx0, Qcur, ring_keys, ring_values, ...
);

// 远场通道：EMA 聚合的场状态
ggml_tensor* far_field = ggml_mul_mat(ctx0, Wo_field, field_ema);

// 融合
ggml_tensor* sfa_output = ggml_add(ctx0, 
    ggml_scale(ctx0, near_field, 1.0f),
    ggml_scale(ctx0, far_field, alpha)
);
```

**场状态更新（在 decode 循环后）**：
```cpp
// field_ema = gamma * field_ema + (1-gamma) * k_current
ggml_tensor* new_field_ema = ggml_add(ctx0,
    ggml_scale(ctx0, field_ema, gamma),
    ggml_scale(ctx0, k_mean, 1.0f - gamma)
);
```

#### 3.3 Metal GPU 内核集成

**移植来源**：`SFA_Metal.metal` 的 6 个核函数

**方案**：将 Metal 内核注册为 ggml 自定义算子

```cpp
// 在 ggml/src/ggml-metal/ggml-metal-ops.cpp 中添加
void ggml_metalsfa_attn(struct ggml_compute_state & state, 
                        ggml_tensor * tensor) {
    // 调用 near_field_attn + ema_update + dual_path_fusion
}
```

### Phase 2: LingYa PEFT（来自 Soma LingYa）

**目标**：在 SFA 层的 QKV/Out 投影中注入 LingYa 生长通道

#### 2.1 LingYa 通道集成

```cpp
// 在 llama_layer 中添加 LingYa 张量
struct llama_layer {
    // ... 现有字段 ...
    
    // LingYa PEFT
    ggml_tensor* lingya_R;   // 脚手架矩阵 [d_out, rank] (冻结)
    ggml_tensor* lingya_P;   // 生长矩阵 [rank, d_in] (可训练)
    float lingya_alpha;      // 生长尺度
};
```

**权重加载**：
```cpp
// 在 load_arch_tensors 中
layer->lingya_R = create_tensor(tn(LLM_TENSOR_LINGYA, "scaffold", il), {d_out, rank}, 0);
layer->lingya_P = create_tensor(tn(LLM_TENSOR_LINGYA, "growth", il), {rank, d_in}, 0);
// P 初始化为零（从零生长）
```

**前向传播修改**：
```cpp
// 原始权重 W
// LingYa 有效权重 W_eff = W + R @ P * alpha
ggml_tensor* lingya_delta = ggml_mul_mat(ctx0, layer->lingya_R, layer->lingya_P);
lingya_delta = ggml_scale(ctx0, lingya_delta, alpha);
ggml_tensor* W_eff = ggml_add(ctx0, W, lingya_delta);
```

#### 2.2 固化机制（推理优化）

```cpp
// 训练完成后调用
void lingya_freeze(llama_layer& layer) {
    // W_frozen = W_original + R @ P_final * alpha
    // 直接替换原始权重，推理零开销
}
```

### Phase 3: Heritage 可学习压缩（来自 Soma Heritage）

**目标**：为超长序列提供可选的可学习 KV 压缩模式

#### 3.1 压缩查询机制

```cpp
// 每个 SFA 层添加可学习压缩查询
ggml_tensor* soma_compress_queries;  // [num_kv_heads, k_compress, head_dim]
ggml_tensor* soma_decay_log;          // [k_compress]
```

**压缩注意力计算**：
```cpp
// compress_scores = softmax(Q_compress @ K^T / √d)
// sf_keys = compress_weights @ K
// sf_values = compress_weights @ V
```

### Phase 4: Homeostasis + GrowthTemporal（来自 Soma Native）

**目标**：可选的稳态归一化和生长位置编码

#### 4.1 Homeostasis 归一化

```cpp
// 替代 RMSNorm 的可选稳态调节
class homeostasis_norm {
    float target_activity;    // 目标活跃度
    ggml_tensor* regulation;  // 动态调节系数 [dims]
};
```

#### 4.2 GrowthTemporal 位置编码

```cpp
// 可学习的时间戳表 + 正弦频率
ggml_tensor* timestamps;  // [max_seq_len, dims/2]
```

---

## 四、优先级与依赖关系

| 优先级 | 模块 | 依赖 | 预计工作量 |
|--------|------|------|-----------|
| **P0** | Phase 1: 核心 SFA | 无 | 高（需改 KV cache 层） |
| **P1** | Phase 2: LingYa PEFT | Phase 1 | 中 |
| **P2** | Phase 3: Heritage 压缩 | Phase 1 | 中 |
| **P3** | Phase 4: Homeostasis | 无 | 低 |

---

## 五、技术风险与缓解

### 风险 1：KV Cache 层改动影响 llama.cpp 稳定性
- **缓解**：仅在 `llama_kv_cache_iswa` 基础上扩展，保持 ISWA 兼容

### 风险 2：Metal 内核调试困难
- **缓解**：先实现 CPU 参考版本验证正确性，再移植 Metal

### 风险 3：LingYa 固化后精度下降
- **缓解**：提供 `lingya_alpha` 可调参数，支持运行时切换

### 风险 4：双通道注意力与 FlashAttention 图兼容性
- **缓解**：使用 ggml 现有算子组合实现，不强制新增 ggml 算子

---

## 六、与现有方案的对比

| 特性 | SFA Engine | ISWA | MLC | DSA |
|------|-----------|------|-----|-----|
| 近场精确注意力 | ✅ Ring Buffer | ✅ Sliding Window | ✅ | ✅ |
| 远场压缩状态 | ✅ EMA 场 | ❌ | ❌ | ✅ State Space |
| O(1) decode | ✅ | ❌ O(w) | ❌ O(n) | ✅ |
| 可学习压缩 | ❌ | ❌ | ❌ | ✅ |
| PEFT 支持 | ❌ | ❌ | ❌ | ✅ (MLA) |
| Metal 加速 | ✅ 6 kernels | ✅ | ✅ | ✅ |

---

## 七、实施路线图

### Week 1-2: Phase 1 核心 SFA
1. 设计 `llama_kv_cache_sfa` 类（继承自 `llama_memory_i`）
2. 实现 ring buffer 的 allocate/seq_cp/seq_rm
3. 修改 `dalin_soma.cpp` graph builder 添加双通道注意力
4. CPU 验证：Python 参考实现 ↔ C++ 输出一致性

### Week 3-4: Metal 内核集成
1. 将 `SFA_Metal.metal` 的 6 个核函数适配为 ggml-metal 算子
2. 实现 kernel dispatch 逻辑
3. 性能基准测试：CPU vs Metal

### Week 5-6: Phase 2 LingYa PEFT
1. 在 `llama_layer` 中添加 LingYa 张量
2. 实现权重加载和固化机制
3. PEFT 训练验证

### Week 7-8: Phase 3-4 高级特性
1. Heritage 可学习压缩
2. Homeostasis + GrowthTemporal 可选模块
3. 完整集成测试

---

## 八、关键代码移植对照表

| 源文件 | 函数/类 | 目标位置 | 移植方式 |
|--------|---------|---------|---------|
| `SFA_Metal.cpp::CPUSFAModule::decode_step` | 双通道注意力 | `dalin_soma.cpp::graph::build_arch_graph` | 直接移植逻辑 |
| `SFA_Metal.cpp::CPUSFAModule::ring_k_/v_` | Ring Buffer | `llama_kv_cache_sfa` | 改为 ggml_tensor |
| `SFA_Metal.metal::near_field_attn` | Metal 近场注意力 | `ggml-metal-ops.cpp` | 注册为 ggml 算子 |
| `SFA_Metal.metal::ema_update` | EMA 场状态更新 | `llama-context.cpp` decode 后同步 | 改为 ggml 图节点 |
| `Soma_LingYa::LingYaChannel` | 灵芽通道 | `llama_layer` 扩展 | 添加 R/P 张量 |
| `Soma_Heritage::SignalFieldAttentionGQA::compress_queries` | 可学习压缩 | `llama_hparams` 扩展 | 添加压缩查询张量 |
| `Soma_Native::Homeostasis` | 稳态调节 | 可选 norm 层 | 新建 ggml 算子 |

---

## 九、验收标准

1. **正确性**：C++ SFA 输出与 Python 参考实现 cosine > 0.9999
2. **内存**：64K 序列下 KV 内存 ≤ 1MB（对比标准 ~500MB）
3. **性能**：Metal decode 速度 ≥ CPU 的 3x
4. **LingYa**：固化后推理延迟增加 = 0（完全融合）
5. **兼容性**：不影响标准 llama 模型的正常运行
