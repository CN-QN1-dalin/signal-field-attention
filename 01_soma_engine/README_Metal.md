# Soma Metal Engine

Signal Field Attention C++/Metal accelerated kernel.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SFA Engine                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Input: x ∈ R^{batch × seq × dims}                  │
│                                                     │
│  ┌──────────────┐                                  │
│  │ QKV Projection │  (3× dense matmul)              │
│  └──────┬───────┘                                  │
│         │                                           │
│         ├──── Q ∈ R^{batch × seq × heads × hd}     │
│         ├──── K ∈ R^{batch × seq × heads × hd}     │
│         └──── V ∈ R^{batch × seq × heads × hd}     │
│                                                     │
│  ┌──────────────────────────────────────┐           │
│  │  For each token t:                    │           │
│  │                                       │           │
│  │  1. Near-field:                       │           │
│  │     attn_near = softmax(Q·K_ring^T)·V_ring  │
│  │     (only last k tokens)                │           │
│  │                                       │           │
│  │  2. Far-field:                        │           │
│  │     attn_far = Q · S_ema              │           │
│  │     (EMA-compressed historical state)   │           │
│  │                                       │           │
│  │  3. Fuse:                             │           │
│  │     out = attn_near + α · attn_far    │           │
│  │                                       │           │
│  │  4. Output projection                 │           │
│  │  5. Update ring buffer + EMA state    │           │
│  └──────────────────────────────────────┘           │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Compute: O(k · n · d)                              │
│  Memory:  O(k · d)   (fixed, independent of n)      │
└─────────────────────────────────────────────────────┘
```

## Build

### CPU-only (works on any platform)

```bash
bash build_metal.sh cpu
```

### With Metal GPU (macOS only, requires Xcode)

```bash
# 1. Compile .metallib
bash build_metal_lib.sh

# 2. Compile with Metal support
bash build_metal.sh metal
```

## Run

```bash
# Basic run
./soma_metal

# Custom sequence length
./soma_metal --prefill 1024

# Custom decode steps
./soma_metal --decode 5000

# Compare with standard attention (small sequence)
./soma_metal --compare

# Force CPU-only mode
./soma_metal --cpu
```

## Results

### Performance (dims=128, heads=4, k=16)

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Prefill (256 tokens) | 7.31 ms | 35,021 tok/s |
| Decode (single token) | 0.036 ms | 27,884 tok/s |

### Correctness

| Metric | Value |
|--------|-------|
| Cosine similarity (avg t=1~5) | >0.92 |
| Cosine similarity (avg t=16~31) | ~0.50 |
| t=0 (no context) | 0.0 (expected) |

**Note**: Lower similarity at later tokens is expected — SFA uses only the last k=16 tokens for near-field attention, while standard attention uses all previous tokens. The far-field EMA channel partially compensates.

### Memory

| Configuration | Memory |
|--------------|--------|
| k=4, heads=4, hd=32 | 4 KB |
| k=16, heads=4, hd=32 | 16 KB |
| k=16, heads=28, hd=128 (7B) | ~1.1 MB |
| k=16, heads=28, hd=128 (70B) | ~11 MB |

## Metal GPU Kernels

Six GPU kernels in `SFA_Metal.metal`:

1. **near_field_attn** — Softmax attention on ring buffer
2. **ema_update** — EMA field state update
3. **dual_path_fusion** — Near + α·Far fusion
4. **qkv_project** — Full QKV projection
5. **output_project** — Output projection
6. **ring_write** — Circular ring buffer write

Compile to `.metallib` with:
```bash
metal -std=metal2.0 -o SFA_Metal.air SFA_Metal.metal
metallic -S SFA_Metal.air -o SFA_Metal.metallib
```

## Files

| File | Description |
|------|-------------|
| `SFA_Metal.h` | Header with configuration structs |
| `SFA_Metal.cpp` | C++ engine (CPU + optional Metal GPU) |
| `SFA_Metal.metal` | Metal GPU shader sources |
| `build_metal.sh` | Build script (cpu/metal/clean) |
| `build_metal_lib.sh` | Compile .metallib from .metal |
| `bench_ring_size.sh` | Ring buffer size trade-off benchmark |

## Future Work

- [ ] Full Metal GPU integration (requires Xcode)
- [ ] Batched decoding support
- [ ] FP16/BF16 precision modes
- [ ] 7B model validation (dims=3584, heads=28)
- [ ] JIT compilation of shaders at runtime
- [ ] Multi-GPU support (future)
