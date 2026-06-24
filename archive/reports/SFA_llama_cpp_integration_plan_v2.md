# Dalin Soma × llama.cpp 全量集成方案 v2

> 本文基于 llama.cpp 源码深度分析（2026-06-16），从 llama.cpp **维护者/贡献者视角**审视集成方案。
> 每个判断都有源码行号佐证。

---

## 一、总体评估

**疯狂程度**：⭐⭐⭐⭐  
**可行性**：⭐⭐⭐⭐⭐（源码验证可行）

llama.cpp 的架构天然适合集成 SFA，因为：

1. **140+ 架构已存在**，SFA 只需新增一个 `LLM_ARCH_DALIN_SOMA`
2. **非标准注意力有先例**：`GGML_OP_SSM_SCAN` (Mamba)、`GGML_OP_RWKV_WKV6` (RWKV)、`GGML_OP_GATED_DELTA_NET` (Gated Linear Attention)
3. **MAP_CUSTOM 机制**已就位：`GGML_OP_MAP_CUSTOM1/2/3` + `GGML_OP_CUSTOM` 专门留给自定义操作
4. **Metal pipeline 机制**成熟：每个 op 有对应的 `ggml_metal_library_get_pipeline_*`
5. **后端调度**解耦：`ggml_backend_supports_op()` 判断 + 自动 offload

---

## 二、源码架构梳理（精确到行号）

### 数据流

```
GGUF 文件
  → llama-model-loader.cpp:554 (arch_name = gguf_get_val_str())
  → llm_arch_from_string("dalin-soma")  → LLM_ARCH_DALIN_SOMA
  → llama_model_loader::get_arch()      → enum llm_arch
  → 根据架构创建 model (如 llama_model_dalin_soma)
  → model.build_arch_graph(params)       → ggml_cgraph*
  → ggml_backend_sched_split_graph()     → 按 op 类型分发到后端
  → ggml_backend_graph_compute()         → CPU/Metal/CUDA 执行
```

### 关键文件与行号

| 文件 | 关键行号 | 作用 |
|------|---------|------|
| `src/llama-arch.h:23` | `enum llm_arch` | 140+ 架构枚举 |
| `src/llama-arch.cpp:25` | `LLM_ARCH_NAMES` | 架构名→枚举映射 |
| `src/llama-arch.cpp:390` | `LLM_TENSOR_NAMES` | 张量名格式定义 |
| `src/llama-arch.cpp:650` | `LLM_TENSOR_INFOS` | 张量→ggml_op 映射 |
| `src/llama-arch.cpp:881` | `llm_arch_is_recurrent()` | 判断是否为循环架构 |
| `src/llama-arch.cpp:894` | `llm_arch_is_hybrid()` | 判断是否为混合架构 |
| `src/llama-model-loader.cpp:554` | `arch_name = gguf_get_val_str()` | 从 GGUF 读取架构 |
| `src/llama-model-loader.cpp:827` | `get_arch()` | 返回 llm_arch 枚举 |
| `src/llama-graph.cpp:2053` | `build_attn_mha()` | 标准 MHA 构建（~200行） |
| `src/llama-graph.cpp:2176` | `use_flash_attn` 判断 | 决定用 flash 还是标准路径 |
| `src/llama-graph.cpp:2213` | `build_attn()` (no-cache) | 无缓存注意力入口 |
| `src/llama-graph.cpp:2298` | `build_attn()` (with KV) | 有缓存注意力入口 |
| `src/models/mamba.cpp:3` | `llama_model_mamba::load_arch_hparams()` | Mamba 超参数加载 |
| `src/models/mamba.cpp:39` | `llama_model_mamba::load_arch_tensors()` | Mamba 权重加载 |
| `src/models/mamba.cpp:83` | `build_arch_graph()` | Mamba 图构建 |
| `src/models/rwkv6.cpp` | 类似结构 | RWKV 实现参考 |
| `ggml/include/ggml.h:479` | `enum ggml_op` | 150+ 操作符定义 |
| `ggml/include/ggml.h:558` | `GGML_OP_FLASH_ATTN_EXT` | Flash Attention 扩展 |
| `ggml/include/ggml.h:560` | `GGML_OP_SSM_SCAN` | Mamba SSM Scan |
| `ggml/include/ggml.h:566` | `GGML_OP_RWKV_WKV6` | RWKV |
| `ggml/include/ggml.h:570` | `GGML_OP_GATED_DELTA_NET` | Gated Linear Attention |
| `ggml/include/ggml.h:574` | `GGML_OP_MAP_CUSTOM1` | 通用自定义算子 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp:423` | `ggml_backend_cpu_device_supports_op()` | CPU 支持判断 (default: return true) |
| `ggml/src/ggml-cpu/ops.cpp:8320` | `ggml_compute_forward_flash_attn_ext_f16_one_chunk()` | Flash Attn CPU 实现 |
| `ggml/src/ggml-cpu/ops.cpp:9477` | `ggml_compute_forward_ssm_scan_f32()` | SSM Scan CPU 实现 |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp:437` | `case GGML_OP_FLASH_ATTN_EXT` | Metal flash attn |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp:1454` | `ggml_metal_op_ssm_scan()` | Metal SSM scan |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp:1552` | `ggml_metal_op_rwkv()` | Metal RWKV |
| `ggml/src/ggml-metal/ggml-metal-device.h:128` | `ggml_metal_library_get_pipeline_ssm_scan()` | Metal pipeline 获取 |
| `ggml/src/ggml-backend.cpp:921` | `ggml_backend_supports_op()` | 后端支持判断 |

---

## 三、集成方案（从 llama.cpp 视角逐条审视）

### 方案 A：标准方式——新增 GGML OP + 新架构（推荐 ⭐⭐⭐⭐⭐）

这是 **Mamba、RWKV、Gated Linear Attention** 都用过的路径。llama.cpp 社区完全接受。

#### 精确修改清单

```
1. ggml/include/ggml.h:571 (enum ggml_op 中, GGML_OP_GATED_DELTA_NET 之后)
   新增: GGML_OP_SFA_ATTN

2. ggml/include/ggml.h:~2585 (ggml_map_custom1 之后)
   新增构造函数:
   struct ggml_tensor * ggml_sfa_attn(
       struct ggml_context * ctx,
       struct ggml_tensor * q,        // Q projections
       struct ggml_tensor * k,        // K projections (or field state)
       struct ggml_tensor * v,        // V projections (or field state)
       struct ggml_tensor * field_state, // SFA resonance state
       int32_t k_size,                // ring buffer size
       float decay_factor,            // gamma
       float far_field_weight,        // alpha
       int32_t pad_k,                 // padding
       int32_t n_tasks);

3. ggml/src/ggml.c
   - ggml_op_name/ggml_op_symbol 中添加 "sfa_attn"
   - ggml_sfa_attn() 构造函数实现
   - op_params 序列化/反序列化

4. ggml/src/ggml-cpu/ops.cpp (~9477行之后)
   - ggml_compute_forward_sfa_attn_f32()  // CPU 参考 ssm_scan
   - ggml_compute_forward_sfa_attn_f16()  // CPU F16

5. ggml/src/ggml-cpu/ggml-cpu.cpp:423 (supports_op switch)
   无需修改！CPU default 返回 true，未列出的 op 默认支持

6. ggml/src/ggml-metal/ggml-metal-ops.cpp:437 (switch case)
   新增: case GGML_OP_SFA_ATTN:
             n_fuse = ggml_metal_op_sfa_attn(ctx, idx);
             break;

7. ggml/src/ggml-metal/ggml-metal-device.h:~130
   新增声明:
   struct ggml_metal_pipeline_with_params
       ggml_metal_library_get_pipeline_sfa_attn(...);

8. ggml/src/ggml-metal/ggml-metal-ops.cpp (~2700行)
   - ggml_metal_op_sfa_attn() 实现
   - 复用现有 SFA_Metal.metal 着色器

9. src/llama-arch.h:~73 (enum llm_arch, LLM_ARCH_UNKNOWN 之前)
   新增: LLM_ARCH_DALIN_SOMA

10. src/llama-arch.cpp:~75 (LLM_ARCH_NAMES, LLM_ARCH_UNKNOWN 之前)
    新增: { LLM_ARCH_DALIN_SOMA, "dalin-soma" }

11. src/llama-arch.cpp:~390 (LLM_TENSOR_NAMES)
    新增 SFA 张量名:
    { LLM_TENSOR_SFA_ATTN_Q,      "blk.%d.sfa_q" },
    { LLM_TENSOR_SFA_ATTN_K,      "blk.%d.sfa_k" },
    { LLM_TENSOR_SFA_ATTN_V,      "blk.%d.sfa_v" },
    { LLM_TENSOR_SFA_FIELD_STATE, "blk.%d.sfa_field_state" },
    { LLM_TENSOR_SFA_RING_BUFFER, "blk.%d.sfa_ring_buffer" },

12. src/llama-arch.cpp:~650 (LLM_TENSOR_INFOS)
    映射:
    { LLM_TENSOR_SFA_ATTN_Q,  {LAYER_REPEATING, GGML_OP_SFA_ATTN}},
    { LLM_TENSOR_SFA_FIELD_STATE, {LAYER_REPEATING, GGML_OP_NONE}},

13. src/models/dalin-soma.cpp (新建)
    - llama_model_dalin_soma::load_arch_hparams()   [仿 mamba.cpp:3]
    - llama_model_dalin_soma::load_arch_tensors()   [仿 mamba.cpp:39]
    - llama_model_dalin_soma::build_arch_graph()    [仿 mamba.cpp:83]

14. src/llama-model-loader.cpp:~554
    GGUF 中 `general.architecture` = "dalin-soma" 时，
    llm_arch_from_string → LLM_ARCH_DALIN_SOMA

15. src/llama.h (可选)
    新增 SFA 相关推理参数:
    - ring_buffer_size
    - decay_factor
    - far_field_weight
```

#### llama.cpp 视角的评审

| llama.cpp 维护者关注点 | 我们的应对 | 状态 |
|----------------------|-----------|------|
| **API 稳定性** | 新增 enum value 向后兼容 | ✅ |
| **代码风格** | 仿照 mamba.cpp/rwkv6.cpp 风格 | ✅ |
| **反向传播** | Phase 2 实现 `GGML_OP_SFA_ATTN_BACK` | ⚠️ 需规划 |
| **量化兼容** | 权重走 GGUF 量化通道，op 类型不改变 | ✅ |
| **Metal/CUDA 支持** | 先实现 CPU + Metal，CUDA 后续 | ✅ |
| **性能影响** | CPU default true，不影响的 op 不受影响 | ✅ |
| **测试** | 仿照 mamba_test.cpp 添加 dalin_soma_test | ⚠️ 需补充 |
| **文档** | 在 README.md 添加架构说明 | ⚠️ 需补充 |

---

### 方案 B：MAP_CUSTOM1 快速原型（⭐⭐⭐⭐ 短期方案）

在 **不改任何 enum 和架构** 的前提下，利用 `ggml_map_custom1` 快速验证。

**关键路径**（`src/llama.cpp:113-172`）：
```cpp
// 在 llama_model::build_arch_graph() 中
auto * inp_attn = build_attn_inp_kv();

for (int il = 0; il < n_layer; ++il) {
    // ... norm ...
    cur = build_attn(inp_attn, ...);  // <-- 替换这里
    // 当前调用 build_attn() → build_attn_mha()
    // 改为: 如果模型有 SFA 标志，则调用 SFA 路径
}
```

**局限性**（从 llama.cpp 视角看）：
- `ggml_map_custom1` 只接受 1 个输入张量（`const struct ggml_tensor * a`）
- SFA 需要 Q/K/V/field_state 至少 4 个输入
- 需要改用 `ggml_map_custom3` (3 输入) 或 `ggml_custom_4d` (4 输入)
- 即便如此，MAP_CUSTOM 系列算子 **没有反向传播**
- Metal 端需要手动实现 `ggml_metal_map_custom` 分发

**结论**：可以作为 Phase 1 原型验证，但不能作为最终方案。

---

### 方案 C：替换 KV Cache 为 SFA RingBuffer（⭐⭐⭐⭐⭐ 增量验证）

**最小侵入性**方案——不改架构、不改 op、不改模型结构，只改 KV 缓存管理。

**关键修改点**：

```
1. llama-kv-cache.cpp:77 (llama_kv_cache::llama_kv_cache)
   当前: 为每层分配 [kv_size, n_head, head_dim] 张量
   改为: 如果模型是 SFA 模式，分配 [ring_buffer_size, n_head, head_dim]

2. llama-kv-cache.cpp:1239 (get_k / get_v)
   当前: 通过 llama_pos 映射到具体行
   改为: SFA 模式下通过 ring buffer index 读写

3. llama-kv-cache.cpp:760 (prepare / find_slot)
   当前: 为每个 seq 分配连续 KV 位置
   改为: SFA 模式下只分配 ring buffer 槽位

4. llama-kv-cache.cpp:2378 (update)
   当前: 增量更新 KV 缓存
   改为: SFA 模式下更新 resonance field state
```

**优点**：
- 改动范围集中在 `llama-kv-cache.cpp` (~100 行修改)
- 所有现有架构自动受益（Qwen2/Llama 都能用 O(k) 缓存）
- 可以单独提交 upstream PR（影响面小）

**缺点**：
- 只优化了缓存，注意力计算仍是 O(n²)
- 需要精心设计 ring buffer 读写语义和 field state 更新

---

## 四、最终推荐：混合策略

```
Phase 0（1周）: 原型验证
  → 方案 C：替换 KV cache 为 SFA RingBuffer
  → 在现有 Qwen2 模型上验证 O(k) 压缩效果
  → 评估 field state 更新对精度的影响

Phase 1（4周）: 完整集成
  → 方案 A：新增 GGML OP + 新架构
  → 新建 src/models/dalin-soma.cpp
  → CPU + Metal 完整实现
  → GGUF 权重加载 + 推理测试

Phase 2（2周）: 上游 PR
  → 整理 PR，仿照 mamba/rwkv6 的 PR 格式
  → 补充测试用例
  → 提交 upstream

Phase 3（持续）: 进阶
  → CUDA 后端
  → 反向传播（训练支持）
  → SFA MoE 混合架构
```

---

## 五、llama.cpp 视角的风险清单

| 风险 | 严重程度 | 应对 |
|------|---------|------|
| **反向传播缺失** | 高 | Phase 1 只做推理，Phase 2 实现 backward |
| **GGUF 格式冲突** | 中 | 使用独立 architecture name "dalin-soma" |
| **Metal 着色器兼容性** | 中 | 复用现有 pipeline 机制，先跑通 CPU |
| **RingBuffer 并发安全** | 中 | 参照 llama_kv_cells_vec 的线程模型 |
| **上游 PR 被拒** | 低 | llama.cpp 已有大量非标准架构 PR 通过 |
| **精度问题** | 高 | Phase 0 原型验证时重点监控 |
| **长上下文 vs 短上下文性能折中** | 中 | 可配置 ring_buffer_size |

---

## 六、下一步

1. **立即**：开始 Phase 0 原型——方案 C 替换 KV cache
2. **1周后**：验证结果 → 决定是否推进 Phase 1
3. **如果验证成功**：启动 Phase 1 完整集成

要现在开始写 Phase 0 的代码吗？
