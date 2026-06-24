# Dalin Soma 集成深度审查报告

## 审查视角

1. **维护者视角**（ggml/llama.cpp 设计原则、代码质量、架构合理性）
2. **用户视角**（能否实际运行、是否达到论文声称的效果、部署可行性）
3. **学术诚信视角**（论文声明 vs 实际代码行为）

---

## 一、维护者视角：llama.cpp 架构合规性

### 1.1 ✅ 做得好的部分

**a) 选择了正确的集成路径**
- 维护者明确反对自定义 ggml 算子（SFA_ATTN），我们采纳了建议，改为 model-level 集成
- 使用标准 ggml 原语（FlashAttention、RMSNorm、SwiGLU）构建计算图，符合 ggml 设计哲学

**b) 架构注册完整**
- `llama-arch.h` 添加 `LLM_ARCH_DALIN_SOMA` enum
- `llama-arch.cpp` 注册字符串映射 `"dalin-soma"`
- `llama-model.cpp` 工厂方法正确实例化
- RoPE 类型映射正确
- 编译通过，符号完整

**c) KV cache 继承模式合理**
- `llama_kv_cache_soma` 继承 `llama_memory_i`，包装 `llama_kv_cache_iswa`
- 实现了完整的 `llama_memory_i` 接口（clear、seq_rm/cp/keep/add/div、state_write/read）
- state_write/read 使用 magic number "SFA1" 做格式校验

### 1.2 ⚠️ 严重问题

**a) field_state 同步机制极其脆弱**

`llama-context.cpp` 中的实现：
```cpp
for (int ni = 0; ni < n_nodes; ++ni) {
    ggml_tensor * t = ggml_graph_node(res->get_gf(), ni);
    if (t && strcmp(ggml_get_name(t), "field_ema") == 0 && t->data) {
        if (t->ne[0] == n_kv_heads && t->ne[1] == head_dim) {
            for (int il = 0; il < (int) soma_model->layers_data.size(); ++il) {
                // memcpy(ld.field_state->data, t->data, ...)
            }
        }
    }
}
```

**问题清单：**

1. **名字匹配不可靠**：`ggml_get_name` 返回的是运行时生成的名字，不是编译时确定的。每次图构建可能产生不同的名字。如果 ggml 内部重命名了节点，这段代码静默失效。

2. **shape 匹配不够**：`[n_kv_heads, head_dim]` 可能与其他 tensor 形状冲突。多个 tensor 可能有相同形状，导致 `break` 提前退出，部分层的 field_state 未被更新。

3. **多序列未处理**：循环里 `break` 只处理第一个匹配的 tensor。如果有多个序列（n_seqs > 1），只有第 0 个序列被处理。

4. **跨设备拷贝缺失**：`field_ema` 可能在 GPU 上（如果注意力计算在 GPU），而 `field_state` 可能在 CPU 上。`memcpy` 直接跨设备拷贝是 UB。

5. **竞态条件**：如果图在异步后端执行，`memcpy` 在图完成前执行，读到的是旧数据。

**b) `build_qkv` 与 ISWA 缓存的交互未理解**

调用 `build_qkv(lm.layers[il], cur, ...)` 后，`Kcur` 和 `Vcur` 是当前 token 的 Q/K/V。然后 `build_attn` 内部会：
1. 将 Kcur/Vcur **写入** KV cache（`mctx_cur->cpy_k` / `cpy_v`）
2. 从 KV cache **读取**历史 K/V
3. 做 FlashAttention

这意味着 `Kcur` 在传给 `build_attn` 之前已经被写入了 ISWA 缓存。然后在 SFA 代码中我们又计算 `K_mean = mean(Kcur)` —— 但这里的 Kcur 只是**当前 token 的 K**，不是整个序列的 K。

**这就是致命缺陷**：论文声称的 `F_t = γ·F_{t-1} + (1-γ)·mean(K_t)` 中，`K_t` 应该是当前步新增的所有 KV。但我们的代码计算的是单个 token 的 K，而不是整个 batch 的 K。对于 decode 场景（n_seq_tokens=1），这恰好是对的。但对于 batch 推理（n_seq_tokens > 1），`K_mean` 的计算是错误的。

**c) `load_arch_hparams` 覆盖了标准参数**

```cpp
hparams.f_attn_value_scale = alpha;  // 覆盖了 llama_hparams 标准字段
hparams.f_attention_scale = beta;
```

这些是 `llama_hparams` 的标准字段，被我们 hijack 用来存 SFA 参数。这会导致：
- 如果未来 llama.cpp 在这些字段上增加新功能，SOMA 会冲突
- 其他模型加载时如果这些字段被设置，会影响行为

**d) `llama_kv_cache_soma` 构造函数签名过于复杂**

```cpp
llama_kv_cache_soma(
    const llama_model & model,
    ggml_type type_k, ggml_type type_v,
    bool v_trans, bool offload, bool swa_full, bool unified,
    uint32_t kv_size, uint32_t n_seq_max, uint32_t n_ubatch,
    uint32_t n_pad,
    llama_memory_t mem_other,
    const layer_filter_cb & filter,
    const layer_reuse_cb & reuse,
    const layer_share_cb & share)
```

15 个参数，全部转发给内部的 `llama_kv_cache_iswa`。但 `llama_kv_cache_soma` 自己的职责（field_state 管理）完全没有体现在构造函数参数中。这违反了单一职责原则。

**e) `m_field_states` 的初始化逻辑有 bug**

```cpp
const int n_sfa_layers = (int) hparams.n_swa;
```

`hparams.n_swa` 是 SWA 窗口大小（比如 64），不是 SFA 层数。应该用 `hparams.n_layer()`。这意味着 `n_sfa_layers` 可能是 64，但实际只有 32 层，导致 `m_field_states` 分配了过多的内存。

### 1.3 🔴 代码质量问题

**a) `dalin_soma.cpp` 中 `build_qkv` 被调用了两次**

SFA 分支和非 SFA 分支都有完整的 `build_qkv` + `ggml_rope_ext` + `build_attn` 调用链。两个分支的代码几乎完全相同，唯一的区别是 SFA 分支额外计算了 `field_ema` 和 `far_field`。这违反了 DRY 原则。

**b) `K_sum` 计算有多余的 reshape**

```cpp
ggml_tensor * K_sum = ggml_sum_rows(ctx0, Kcur);  // [n_kv_heads, 1, n_seqs]
K_sum = ggml_scale(ctx0, K_sum, 1.0f / (float)n_seq_tokens);
K_sum = ggml_reshape_3d(ctx0, K_sum, K_sum->ne[0], K_sum->ne[1], n_seqs);
K_sum = ggml_sum_rows(ctx0, K_sum);  // [n_kv_heads, 1]
K_sum = ggml_reshape_2d(ctx0, K_sum, K_sum->ne[0], n_embd_head);
```

4 次 reshape/sum 操作来计算 K 的均值。可以简化为一次 `ggml_sum` 或更少的操作。

**c) `layers_data` 和 `layers` 的数据不一致**

`llama_model_dalin_soma` 有两个并行结构：
- `layers[N]`：标准 `llama_layer`（包含 wq, wk, wv, wo, ffn_*）
- `layers_data[N]`：`soma_layer_data`（包含 field_state, soma_lingya_P, soma_homeostasis_reg）

这两者通过索引关联，但没有运行时一致性检查。如果 `n_layer` 发生变化（比如量化后的模型结构变化），两者可能不同步。

**d) `dynamic_cast` 在性能关键路径**

```cpp
const auto * soma_model = dynamic_cast<const llama_model_dalin_soma *>(&model);
```

在 decode 循环中做 `dynamic_cast` 是有开销的。虽然只在 `model.arch == LLM_ARCH_DALIN_SOMA` 时执行，但仍然可以在编译时确定类型。

### 1.4 ❌ 遗漏的实现

**a) KV cache 的 field_state 未在 `build_attn` 流程中更新**

`llama_kv_cache_soma` 提供了 `update_field_state()` 方法，但没有任何地方调用它。field_state 的更新完全依赖 `llama-context.cpp` 中的 post-graph memcpy。这意味着：
- KV cache 层面的 field_state（`m_field_states`）从未被更新
- 只有 model tensor 层面的 `layers_data[i].field_state` 被更新
- 两者是冗余的，且不同步

**b) `llama_kv_cache_soma::seq_cp` 中 field_state 复制逻辑错误**

```cpp
if ((size_t) seq_id_src < m_field_states.size() && ...
    (size_t) seq_id_dst < m_field_states[0].size()) {
    for (auto & layer_states : m_field_states) {
        layer_states[seq_id_dst] = layer_states[seq_id_src];
    }
}
```

条件检查把 `seq_id_src` 和 `m_field_states.size()`（层数）比较，把 `seq_id_dst` 和 `m_field_states[0].size()`（序列数）比较。这两个比较的对象类型错了。

**c) `llama_kv_cache_soma::seq_rm` 同样有 bug**

```cpp
if (seq_id >= 0 && (size_t) seq_id < m_field_states.size() && !m_field_states[0].empty()) {
    for (auto & layer_states : m_field_states) {
        if ((size_t) seq_id < layer_states.size()) {
            memset(layer_states[seq_id].data(), 0, ...);
        }
    }
}
```

同样，`seq_id` 是序列 ID，应该和 `layer_states.size()`（序列数）比较，而不是 `m_field_states.size()`（层数）。

---

## 二、用户视角：能否实际运行并达到预期？

### 2.1 当前能做什么

✅ 加载 dalin-soma 架构的 GGUF 文件
✅ 推理（本质上是标准 Transformer，因为 SFA 双通道计算在图里但 field_state 同步有问题）
✅ 保存/恢复 KV cache 状态（KV 部分）
✅ 在 Apple Silicon 上通过 Metal 加速

### 2.2 用户期望 vs 实际情况

| 期望 | 实际情况 | 差距 |
|------|----------|------|
| O(1) 内存压缩 | field_state 是 [n_kv_heads, head_dim] ≈ 32×4096×4B ≈ 512KB/层，总共 ~16MB（7B）。加上 ISWA KV cache ~64×32×4096×2B ≈ 16MB。总计 ~32MB。对比完整 KV cache 114MB（64K 序列），确实有压缩，但远达不到论文声称的 248× | 部分实现 |
| 4.16× 解码加速 | 当前实现比标准注意力**更慢**：多了 K_mean 计算、field_ema 更新、far_field 投影、homeostasis 调节、LingYa gate。每个 token 多了约 5 次额外的 ggml 操作 | 反向 |
| Cosine Similarity > 0.9999999 | 没有验证代码，没有实测数据 | 未验证 |
| 真正的 SFA 双通道 | 近场通道是标准 FlashAttention（没问题），远场通道的 field_state 同步有 bug（多序列、跨设备） | 有 bug |

### 2.3 部署可行性

**短期（1-2 周）**：
- 可以跑起来，但输出质量可能不如标准 Transformer（因为 field_state 同步 bug 导致远场通道贡献不稳定）
- 没有 benchmark 数据支撑论文声称的加速比

**中期（1-2 月）**：
- 修复 field_state 同步机制
- 实现真正的 ring buffer 压缩（替换 ISWA 滑动窗口）
- 补充实测数据

**长期**：
- 需要 C++/Metal 内核优化才能达到论文声称的 4.16× 加速
- 目前 MLX/Python 原型 + llama.cpp 标准原语的混合方案不太可能达到这个加速比

---

## 三、学术诚信视角：论文 vs 代码

### 3.1 论文声称的关键指标

| 指标 | 论文声称 | 代码支撑 | 状态 |
|------|----------|----------|------|
| 248× KV 内存压缩 | 64K 序列 462KB vs 114MB | field_state 存在但同步有 bug；ISWA 不是真正的 ring buffer | ⚠️ 部分 |
| Cosine Similarity > 0.9999999 (t≥1) | 与标准注意力的输出相似度 | **完全没有验证代码** | ❌ 未验证 |
| 4.16× 解码加速 | C++/Metal 部署目标 | 当前实现比标准注意力更慢 | ❌ 未实现 |
| 8.1KB 额外参数 | field_state + LingYa P + homeostasis | 7B 模型实际约 16MB（不是 8.1KB） | ❌ 严重不符 |
| O(1) 解码复杂度 | 理论目标 | 图里多了 K_mean + field_ema + far_field 计算 | ⚠️ 理论成立，实际更慢 |

### 3.2 需要修正的论文声明

1. **"8.1KB 额外参数"**：这是错误的。field_state 是 per-layer 的浮点数组，不是可训练参数。7B 模型的 field_state 总计约 16MB（32 层 × 32 heads × 4096 dim × 4 bytes）。LingYa P 矩阵是 32 × (8 × 4096 × 4 bytes) ≈ 4MB。总计约 20MB，不是 8.1KB。

2. **"4.16× 解码加速"**：这是 C++/Metal 内核的理论目标，不是实测结果。当前 llama.cpp 实现使用了标准 ggml 原语，没有定制内核，实际比标准注意力更慢。

3. **"Cosine Similarity > 0.9999999"**：这是 MLX prototype 的结果，但我们找不到任何 MLX 推理代码。需要重新运行验证。

4. **"248× KV 内存压缩"**：这是理论计算（假设 field_state 正确工作），但 ISWA 仍然存储了最近 64 个 token 的完整 KV，所以实际压缩率远低于 248×。

---

## 四、修复优先级

### P0（必须修，否则无法正确工作）

1. **field_state 同步机制重构**
   - 不要用名字匹配，改用 tensor 指针/ID
   - 处理多序列情况
   - 处理跨设备拷贝
   - 考虑在 ggml 图内完成 field_state 更新（通过自定义 op 或图内循环）

2. **`llama_kv_cache_soma` 构造函数 bug**
   - `n_sfa_layers` 应该用 `hparams.n_layer()` 而不是 `hparams.n_swa`
   - `seq_cp` 和 `seq_rm` 中的索引比较对象错误

3. **统一 field_state 管理**
   - 要么完全在 model tensor 层面管理（当前方式），移除 KV cache 层面的冗余
   - 要么完全在 KV cache 层面管理，移除 model tensor 层面的冗余
   - 不能两者并存且不同步

### P1（重要，影响正确性）

4. **`K_mean` 计算简化**
   - 减少不必要的 reshape 操作
   - 确保对 batch 推理也正确

5. **参数命名冲突**
   - 不要 hijack `hparams.f_attn_value_scale` 和 `hparams.f_attention_scale`
   - 添加专门的 SFA 参数字段

6. **DRY 重构**
   - SFA 和非 SFA 分支的 `build_qkv` + `build_attn` 提取为公共函数

### P2（改进，不影响正确性）

7. **性能优化**
   - 移除 `dynamic_cast`，用虚函数或模板
   - 简化 `K_sum` 计算
   - 减少 `ggml_reshape` 调用

8. **补充实测数据**
   - 在真实模型上运行 SFA Engine
   - 记录实际的压缩率、PPL、延迟

---

## 五、总结

### 积极面
- 架构注册完整，编译通过，符号正确
- 选择了正确的集成路径（model-level 而非 ggml operator）
- 双通道注意力的图构建逻辑基本正确
- llama-kv-cache-soma 的 state_write/read 框架合理

### 风险面
- **field_state 同步机制是定时炸弹**：名字匹配 + 单序列假设 + 无跨设备处理 = 多序列推理时静默产生错误结果
- **论文数据缺乏代码支撑**：8.1KB、4.16×、0.9999999 都需要重新验证或修正
- **ISWA 不是真正的 ring buffer**：它是滑动窗口，不是指数衰减压缩
- **性能不升反降**：当前实现比标准注意力更慢

### 建议
1. **立即修复 P0 问题**，特别是 field_state 同步和多序列处理
2. **诚实标注论文数据状态**：区分"理论目标"和"实测结果"
3. **先跑通基准测试**，拿到真实数据后再发表论文
4. **考虑将 field_state 更新移到 ggml 图内**，通过自定义 op 避免 post-graph memcpy 的 fragility
