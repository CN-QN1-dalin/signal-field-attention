# Dalin Soma × llama.cpp Integration Report

**Date:** 2026-06-17
**Status:** ✅ BUILD SUCCESSFUL
**Commit:** HEAD of main branch

## What Was Integrated

### 1. Architecture Registration
- **Enum:** `LLM_ARCH_DALIN_SOMA` added to `llama-arch.h`
- **String:** `"dalin-soma"` registered in `llama-arch.cpp`
- **Factory:** Model instantiation in `llama-model.cpp`
- **RoPE Type:** `LLAMA_ROPE_TYPE_NEOX` mapped

### 2. SFA Tensor Schema (GGUF Keys)
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `soma.ring_size` | u32 | 64 | Near-field window size |
| `soma.alpha` | f32 | 0.1 | Far-field fusion weight |
| `soma.beta` | f32 | 0.01 | Resonance update rate (EMA decay = 1-β) |
| `soma.scale` | f32 | auto | Attention temperature scaling |
| `soma.pos_buckets_count` | u32 | 0 | Position buckets |
| `soma.rel_pos_bias_dim` | u32 | 0 | Relative position bias dim |
| `soma.n_resonance_states` | u32 | 0 | Resonance state count |

### 3. Per-Layer SFA Tensors
Each SFA layer (controlled by `soma.ring_size` = number of SFA layers) has:

| Tensor | Shape | Type | Source Module |
|--------|-------|------|---------------|
| `soma.field_state` | `[n_kv_heads, head_dim]` | F32 | Heritage + SFA Engine |
| `soma.decay_log` | `[16]` | F32 | Heritage |
| `soma.resonance_phase` | `[n_kv_heads, 16]` | F32 | Heritage |
| `soma.resonance_freq` | `[n_kv_heads, 16]` | F32 | Heritage |
| `soma.lingya_P` | `[8, n_embd]` | F32 | LingYa PEFT |
| `soma.homeostasis_reg` | `[n_embd]` | F32 | Homeostasis |

## Architecture Diagram

```
Input → TokenEmbedding → [Layer 1 ... Layer N] → OutputNorm → Output

Each SFA Layer:
  ┌─────────────────────────────────────────────┐
  │  HomeostasisNorm(input)                      │
  │  └→ RMSNorm × (1 + homeostasis_reg × 0.01)  │
  │                                              │
  │  IF use_sfa:                                 │
  │    ├─ Near-field: FlashAttention(ring_buffer)│
  │    │   build_qkv() + RoPE + FLASH_ATTN_EXT  │
  │    ├─ Far-field: EMA field aggregation       │
  │    │   F_t = γ·F_{t-1} + (1-γ)·mean(K_t)    │
  │    │   projected through layer.wo            │
  │    └─ Fusion: near + α × far                 │
  │  ELSE:                                       │
  │    └─ Standard FlashAttention                │
  │                                              │
  │  LingYa FFN:                                 │
  │    W = I + R@P×α                             │
  │    output = SwiGLU(FFN) + P@x×α              │
  │                                              │
  │  Residual: output + input                    │
  └─────────────────────────────────────────────┘
```

## Build Verification

```
$ cd /tmp/llama.cpp/build2 && make -j
[100%] Built target llama-app

Symbols in binary:
  22llama_model_dalin_soma       ✓
  llama_model_dalin_soma::graph  ✓
  "dalin-soma"                   ✓
  soma.ring_size                 ✓
  soma.alpha                     ✓
  soma.beta                      ✓
  soma.scale                     ✓
```

## Implementation Notes

### True SFA Features Implemented
1. ✅ **Dual-channel attention**: Near-field (ring buffer) + Far-field (EMA aggregation)
2. ✅ **Homeostatic regulation**: Adaptive per-dimension normalization
3. ✅ **LingYa PEFT**: Zero-initialized growth matrix with orthogonal scaffold
4. ✅ **Resonance state storage**: Field state persisted per-layer via GGUF
5. ✅ **Configurable layer count**: `soma.ring_size` controls how many layers use SFA

### Current Limitations
1. ⚠️ **EMA field state is transient**: Currently computed in the graph but not persisted across inference steps. True O(1) memory requires KV cache extension (see below).
2. ⚠️ **Ring buffer uses ISWA**: Leverages existing sliding window attention but doesn't implement true ring buffer eviction with decay weighting
3. ⚠️ **No Metal kernel**: SFA attention falls back to FLASH_ATTN_EXT, not a custom Metal kernel

### Next Steps for True SFA
1. **KV Cache Extension**: Create `llama_kv_cache_soma` that:
   - Stores `field_state` as a persistent tensor per layer
   - Implements ring buffer with exponential decay (not just sliding window)
   - Uses `state_write`/`state_read` for cross-step persistence
   
2. **Decay-weighted aggregation**: Replace simple `mean(K_t)` with Gaussian-decayed weighted sum:
   ```
   F_t = γ · F_{t-1} + (1-γ) · Σ(decay[i] · K_{t-i})
   ```

3. **Metal kernel**: Implement custom `ggml_sfa_attn` operation for the fusion step

## File Changes Summary

| File | Lines Changed | Description |
|------|---------------|-------------|
| `src/llama-arch.h` | +10 | Enum + KV keys |
| `src/llama-arch.cpp` | +12/-1 | Arch registration |
| `src/llama-model.cpp` | +3 | Factory + RoPE |
| `src/models/models.h` | +16 | Class declaration |
| `src/models/dalin_soma.cpp` | +338 | Full implementation |

**Total:** ~379 lines added, 1 line removed

## Comparison: Paper vs Implementation

| Feature | Paper Claim | Current Implementation |
|---------|-------------|----------------------|
| Near-field attention | Ring buffer, O(k) | ✅ ISWA sliding window |
| Far-field EMA | γ·F_{t-1} + (1-γ)·K_t | ✅ Computed in graph |
| Fusion | near + α·far | ✅ alpha-weighted sum |
| Memory compression | 248× (64K→462KB) | ⚠️ Not yet (requires KV cache ext) |
| O(1) decode | Yes | ⚠️ Partial (graph computes, state not persisted) |
| LingYa PEFT | I + R@P·α | ✅ Zero-init growth matrix |
| Homeostasis | Adaptive norm | ✅ Regulation vector |
| Resonance | Amplitude/Phase/Freq | ✅ Stored as tensors (unused in graph) |
