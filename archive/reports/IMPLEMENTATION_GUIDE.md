# Dalin Soma Implementation Guide

## Overview

This document provides a comprehensive guide to implementing the Dalin Soma architecture in llama.cpp. Dalin Soma uses Signal Field Attention (SFA) to achieve efficient inference with O(k·n) complexity instead of O(n²).

## Architecture

### Core Components

1. **Signal Field Attention (SFA)**
   - Near-field channel: Standard attention on recent k tokens
   - Far-field channel: Exponentially weighted moving average (EWMA) state
   - Fusion: near + α · far

2. **Ring Buffer KV Compression**
   - Fixed-size ring buffer for compressed KV states
   - Exponential decay with configurable alpha parameter
   - Memory compression ratio: 248x–3971x depending on sequence length

3. **Resonance States**
   - Per-head resonance memory for long-term dynamics
   - O(1) incremental inference capability
   - Beta-controlled mixing rate

### Implementation Layers

```
┌─────────────────────────────────────┐
│         Application Layer           │
│  (llama.cpp CLI/Server/Tools)       │
├─────────────────────────────────────┤
│         Model Layer                 │
│  (src/models/dalin_soma.cpp)        │
│  - Graph construction               │
│  - Attention computation            │
│  - FFN implementation               │
├─────────────────────────────────────┤
│         KV Cache Layer              │
│  (llama-kv-cache-iswa.cpp)          │
│  - Ring buffer management           │
│  - Resonance state persistence      │
│  - Exponential decay compression    │
├─────────────────────────────────────┤
│         GGML Backend                │
│  (ggml/src/)                        │
│  - Standard tensor operations       │
│  - No custom operators              │
└─────────────────────────────────────┘
```

## Current Implementation Status

### ✅ Completed

1. **Architecture Registration**
   - `LLM_ARCH_DALIN_SOMA` enum added
   - `"dalin-soma"` string registered in GGUF
   - 7 Soma-specific KV keys defined

2. **Model Class**
   - `llama_model_dalin_soma` defined
   - `load_arch_hparams()` reads GGUF parameters
   - `load_arch_tensors()` loads weights
   - `build_arch_graph()` constructs inference graph

3. **Graph Construction**
   - Standard ggml primitives used (no custom ops)
   - Mixed SFA/standard layer support
   - RoPE positional encoding
   - RMS normalization
   - SwiGLU FFN

4. **Compilation**
   - Zero errors, zero warnings
   - 16 Dalin Soma symbols in binary
   - Full llama.cpp build passes

### ⏳ Pending

1. **KV Cache Extension**
   - Ring buffer compression logic
   - Resonance state management
   - State persistence (save/load)

2. **SFA-Specific Attention**
   - Actual ring buffer implementation
   - EWMA state updates
   - Near/far channel fusion

3. **Testing & Validation**
   - Real model export to GGUF
   - Inference testing
   - Performance benchmarking

## Configuration

### GGUF Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `soma.ring_size` | u32 | 64 | Ring buffer KV compression size |
| `soma.alpha` | f32 | 0.1 | Decay rate for KV compression |
| `soma.beta` | f32 | 0.01 | Mixing rate for resonance states |
| `soma.scale` | f32 | 1/√d_head | Attention temperature scaling |
| `soma.pos_buckets_count` | u32 | 128 | Positional bucket table size |
| `llama.attention.swa_impl` | tensor | - | Per-layer SFA flag (0=standard, 1=SFA) |

### Model Types

| Layers | Embedding | Type |
|--------|-----------|------|
| 24 | 2048 | 1.6B |
| 24 | 4096 | 7B |
| 32 | 4096 | 7B |
| 32 | 6656 | 13B |
| 32 | 8192 | 26B |
| 40 | 8192 | 65B |

## Usage Examples

### Compiling with Soma Support

```bash
cd /tmp/llama.cpp
export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:$PATH"
cmake -B build
cmake --build build -j8
```

### Verifying Soma Integration

```bash
# Check for Soma symbols
nm ./build/bin/libllama.0.0.1.dylib | grep dalin_soma

# Check for architecture string
strings ./build/bin/libllama.0.0.1.dylib | grep dalin-soma

# Run version check
./build/bin/llama version
```

### Running Tests

```bash
python3 test_soma_integration.py
```

## Future Work

### Phase 1: KV Cache Integration
- Implement ring buffer compression in `llama_kv_cache_iswa`
- Add resonance state persistence
- Support state save/load

### Phase 2: SFA Attention
- Replace standard attention with SFA-specific computation
- Implement near/far channel fusion
- Add EWMA state updates

### Phase 3: Optimization
- Metal/CUDA kernel optimization
- Quantization support
- Memory layout optimization

### Phase 4: Validation
- Real model training and export
- Inference testing on standard benchmarks
- Performance comparison with standard attention

## Troubleshooting

### Compilation Errors
- Ensure all Soma files are properly included
- Check for missing enum values
- Verify symbol declarations

### Runtime Issues
- Check GGUF format compliance
- Verify parameter ranges
- Ensure correct layer configuration

### Performance Issues
- Monitor memory usage
- Check ring buffer size appropriateness
- Validate alpha/beta parameter tuning

## References

- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp)
- [SFA Technical Report](./Dalin_Soma_Chinese_Academic_Paper.md)
- [Integration Plan](./SFA_llama_cpp_integration_plan.md)
- [Roadmap](./dalin_soma_roadmap.md)
