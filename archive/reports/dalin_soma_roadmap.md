# Dalin Soma Implementation Roadmap

## Phase 1: Foundation (Complete)
- [x] Register LLM_ARCH_DALIN_SOMA in llama.cpp
- [x] Implement model class in src/models/dalin_soma.cpp
- [x] Add GGUF KV keys for Soma parameters
- [x] Verify compilation and symbol resolution
- [x] Create integration report

## Phase 2: KV Cache Integration (Next)
- [ ] Extend llama_kv_cache_iswa for SFA ring buffer
- [ ] Implement exponential decay compression
- [ ] Add resonance state management
- [ ] Support state persistence (save/load)

## Phase 3: Graph Construction Enhancement
- [ ] Implement SFA-specific attention computation
- [ ] Add ring buffer management nodes
- [ ] Integrate EWMA state updates
- [ ] Support mixed SFA/standard layers

## Phase 4: Testing & Validation
- [ ] Export trained Soma model to GGUF
- [ ] Run inference tests
- [ ] Validate memory compression ratios
- [ ] Benchmark performance improvements

## Technical Details

### KV Cache Extension
```cpp
// llama_kv_cache_soma.h
class llama_kv_cache_soma : public llama_kv_cache_iswa {
public:
    // Ring buffer KV compression
    void compress_kv(uint32_t il, float alpha);
    
    // Resonance state management
    void update_resonance(uint32_t il, float beta);
    
    // State persistence
    void save_resonance_states(llama_io_write_i & io);
    void load_resonance_states(llama_io_read_i & io);
};
```

### SFA Attention Implementation
```cpp
// In dalin_soma.cpp graph builder
ggml_tensor * build_sfa_attn(...) {
    // 1. Ring buffer compression
    ggml_tensor * compressed_k = compress_kv(k, alpha);
    ggml_tensor * compressed_v = compress_kv(v, alpha);
    
    // 2. Resonance state update
    ggml_tensor * resonance = update_resonance(compressed_k, beta);
    
    // 3. Standard attention on compressed KV
    ggml_tensor * attention = ggml_flash_attn(Q, compressed_k, compressed_v);
    
    // 4. Fusion with resonance states
    ggml_tensor * output = attention + alpha * resonance;
    
    return output;
}
```

### GGUF Format Specification
```python
# Soma model GGUF metadata
general.architecture = "dalin-soma"
soma.ring_size = 64
soma.alpha = 0.1
soma.beta = 0.01
soma.scale = 0.125
soma.pos_buckets_count = 128
llama.attention.swa_impl = [1, 1, 1, ..., 0, 0, 0]  # Per-layer SFA flag
```

## Success Criteria
1. ✅ Zero compilation errors/warnings
2. ✅ Full llama.cpp integration
3. ⏳ SFA ring buffer compression working
4. ⏳ Memory compression ratio ≥ 100x
5. ⏳ Decoding speedup ≥ 4x (theoretical)
6. ⏳ PPL degradation < 5% vs standard attention

## Dependencies
- llama.cpp ISWA infrastructure
- GGUF format specification
- SFA mathematical formulation
- Resonance state management

## Next Steps
1. Implement KV cache extension for SFA
2. Add resonance state persistence
3. Create test GGUF model
4. Run validation experiments
