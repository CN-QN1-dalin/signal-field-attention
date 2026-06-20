# Dalin Soma llama.cpp Integration Plan

## Overview
This document describes the integration of **Signal Field Attention (SFA)** into llama.cpp's ggml framework, enabling Dalin Soma's O(k·n) attention with resonance-based KV compression to run on CPU, Metal (Apple Silicon), and CUDA backends.

## Architecture Changes Made

### 1. ggml Core Layer (`ggml/include/ggml.h`, `ggml/src/ggml.c`)

**New Operation:** `GGML_OP_SFA_ATTN` (enum value 98)

**API Signature:**
```c
struct ggml_tensor * ggml_sfa_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,        // [d_head, n_tokens, n_head, n_seqs]
    struct ggml_tensor  * ring_kv_buf, // [d_head, ring_size, n_head_kv, n_seqs]
    struct ggml_tensor  * resonance_states, // [d_head, n_head, n_seqs]
    struct ggml_tensor  * pos_bucket,     // [n_tokens] (optional)
    struct ggml_tensor  * rel_pos_bias,   // [n_head, n_ctx, n_ctx] (optional)
    int32_t                ring_size,
    int32_t                n_head,
    int32_t                n_head_kv,
    int32_t                d_head,
    int32_t                n_ctx,
    float                  alpha,
    float                  beta,
    float                  scale);
```

**Op Params Storage:**
- `op_params[0..3]` (int32_t): ring_size, n_head, n_head_kv, d_head
- `op_params[4..7]` (float): alpha, beta, scale, 0.0f

**Backward Pass:** Forward-only op. Gradient flows through Q only (src[0]).

### 2. CPU Backend (`ggml/src/ggml-cpu/ggml-cpu.c`)

- **Dispatch:** Added `case GGML_OP_SFA_ATTN` → `ggml_compute_forward_sfa_attn()`
- **Task Scheduling:** Parallelizes over n_threads for token-level parallelism
- **Declaration:** `ops-sfa-attn.h` provides `ggml_compute_forward_sfa_attn()`

### 3. Metal Backend (`ggml/src/ggml-metal/`)

**Device Pipeline (`ggml-metal-device.cpp`):**
```cpp
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sfa_attn(...)
```
- Uses 8 function constants (4 int16 + 4 float) for hyperparameters
- Shared memory: `d_head * 3 * sizeof(float)` for q/k/v temp buffers

**Kernel Dispatch (`ggml-metal-ops.cpp`):**
```cpp
int ggml_metal_op_sfa_attn(ggml_metal_op_t ctx, int idx)
```
- Launches `[n_tokens, n_head, n_seqs]` grid with 32-thread groups
- Reads Q, RingKVBuffer, ResonanceStates, optional PosBucket, optional RelPosBias

**Metal Kernel (`ggml-metal.metal`):**
```metal
kernel void kernel_sfa_attn_f32(
    constant ggml_metal_kargs_sfa_attn & args,
    device const char * src0, // Q
    device const char * src1, // RingKVBuffer
    device const char * src2, // ResonanceStates
    device const char * src3, // PosBucket (optional)
    device const char * src4, // RelPosBias (optional)
    device char * dst,
    ...)
```

### 4. CUDA Backend (`ggml/src/ggml-cuda/ggml-cuda.cu`)

- **Dispatch:** Added `case GGML_OP_SFA_ATTN` → `ggml_cuda_op_sfa_attn(ctx, dst)`
- **Note:** CUDA kernel implementation is a stub. Full CUDA implementation would follow the same pattern as Metal.

## Model Adapter Layer (To Be Implemented)

### 5. llama.cpp Model Architecture (`src/llama-model.cpp`)

A new architecture `LLM_ARCH_DALIN_SOMA` needs to be added to:
- `llama-arch.cpp`: `LLM_ARCH_NAMES` map entry `"dalin-soma"`
- `llama-arch.cpp`: `llm_arch_is_recurrent()` / `llm_arch_is_hybrid()` checks
- `llama-model.cpp`: `llama_model_rope_type()` → `LLAMA_ROPE_TYPE_NONE` (SFA uses positional buckets instead)
- `llama-model.cpp`: `build_arch_graph()` → instantiate a new `graph` class

### 6. Graph Builder Pattern

Following the Arctic/BailingMoe pattern, create a new model file:
`src/models/dalin-soma.cpp`

```cpp
class graph : public llm_graph_context {
public:
    graph(const llama_model & model, const llm_graph_params & params)
        : llm_graph_context(params) {
        // Build the SFA transformer graph:
        // 1. Embedding lookup
        // 2. For each layer:
        //    a. RMSNorm → QKV projection
        //    b. Position bucket encoding
        //    c. SFA attention via ggml_sfa_attn()
        //    d. FFN projection
        // 3. Output projection
    }
};
```

### 7. GGUF Format Extension

New GGUF metadata keys for Dalin Soma models:
```python
# general.architecture = "dalin-soma"
# dalin-soma.sfa-ring-size = k (e.g., 128)
# dalin-soma.sfa-alpha = α (e.g., 0.1)
# dalin-soma.sfa-beta = β (e.g., 2.0)
# dalin-soma.sfa-gamma = γ (e.g., 0.95)
# dalin-soma.sfa-k = ring buffer size
# dalin-soma.sfa-d-head = head dimension
# dalin-soma.sfa-num-heads = number of attention heads
# dalin-soma.sfa-num-kv-heads = number of KV heads
```

## Tensor Shape Convention

| Tensor | Shape | Description |
|--------|-------|-------------|
| Q | `[d_head, n_tokens, n_head, n_seqs]` | Query projections |
| RingKVBuffer | `[d_head, ring_size, n_head_kv, n_seqs]` | Compressed KV states |
| ResonanceStates | `[d_head, n_head, n_seqs]` | Per-head resonance memory |
| PosBucket | `[n_tokens]` | Positional bucket indices |
| RelPosBias | `[n_head, n_ctx, n_ctx]` | Relative position bias (optional) |
| Output | `[d_head, n_tokens, n_head, n_seqs]` | Attention output |

## Integration Steps

### Phase 1: Core ggml Integration ✅ (DONE)
- [x] Add `GGML_OP_SFA_ATTN` to ggml enum
- [x] Add `ggml_sfa_attn()` API in ggml.h
- [x] Implement `ggml_sfa_attn()` in ggml.c
- [x] Add backward pass in ggml.c
- [x] CPU dispatch in ggml-cpu.c
- [x] Metal pipeline in ggml-metal-device.cpp
- [x] Metal kernel in ggml-metal.metal
- [x] Metal dispatch in ggml-metal-ops.cpp
- [x] CUDA dispatch stub in ggml-cuda.cu

### Phase 2: CPU Kernel Implementation (TODO)
- [ ] Implement `ggml_compute_forward_sfa_attn()` in `ops-sfa-attn.cpp`
- [ ] Add NEON/AVX vectorized optimizations
- [ ] Benchmark against reference implementation

### Phase 3: Model Architecture (TODO)
- [ ] Add `LLM_ARCH_DALIN_SOMA` to llama-arch.cpp
- [ ] Create `src/models/dalin-soma.cpp`
- [ ] Implement graph builder with SFA attention
- [ ] Add GGUF reader/writer for Dalin Soma format

### Phase 4: CUDA Kernel (TODO)
- [ ] Implement CUDA kernel for SFA attention
- [ ] Follow pattern from `ggml-cuda/ggml-cuda.cu`
- [ ] Optimize for different GPU architectures

### Phase 5: Testing & Validation (TODO)
- [ ] Create test binary `bin/test-sfa`
- [ ] Compare SFA output vs. standard attention
- [ ] Verify numerical accuracy (cosine similarity > 0.9999999)
- [ ] Benchmark speedup and memory reduction

## File Change Summary

```
ggml/include/ggml.h                       | +28 lines (enum + API)
ggml/src/ggml.c                           | +58 lines (op impl + backward)
ggml/src/ggml-cpu/ggml-cpu.c              | +11 lines (dispatch + scheduling)
ggml/src/ggml-cuda/ggml-cuda.cu           | +3 lines (dispatch stub)
ggml/src/ggml-metal/ggml-metal-device.cpp | +46 lines (pipeline)
ggml/src/ggml-metal/ggml-metal-ops.cpp    | +97 lines (dispatch)
ggml/src/ggml-metal/ggml-metal-ops.h      | +1 line (declaration)
ggml/src/ggml-metal/ggml-metal.metal      | +180 lines (kernel)
---------------------------------------------------
Total: +424 lines across 8 files
```

## Next Steps

1. **Implement CPU kernel** in `ops-sfa-attn.cpp` (reference implementation from MLX prototype)
2. **Add model architecture** in `src/models/dalin-soma.cpp`
3. **Create test harness** to validate correctness against reference
4. **Benchmark** on Apple Silicon (Metal) and x86 (CPU)
5. **Submit upstream** as a PR to llama.cpp
