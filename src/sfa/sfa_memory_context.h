#pragma once

// ==========================================
// 🧠 SFA完整持久化内存上下文
// ==========================================
// 设计理念：在llama-memory-context基础上扩展SFA持久化状态
// 支持：跨token的RingBuffer、EMA场状态、语义池更新
// ==========================================

#include "ggml.h"
#include <vector>
#include <cmath>
#include <cstring>

struct sfa_config {
    bool enabled;
    float alpha_base;
    float cross_decay;
    float ema_gamma;
    float gaussian_gamma;
    float enhancement_clip;
    int ring_size;
    int semantic_slots;
    
    sfa_config()
        : enabled(true),
          alpha_base(2.0f),
          cross_decay(0.7f),
          ema_gamma(0.98f),
          gaussian_gamma(0.951229f),
          enhancement_clip(0.01f),
          ring_size(16),
          semantic_slots(64) {}
};

class SFA_Memory_Context {
public:
    sfa_config config;
    int n_layers;
    int hidden_size;
    int head_dim;
    int n_heads;
    
    // Per-layer state
    std::vector<float> ring_buffers;    // [n_layers][ring_size][hidden_size]
    std::vector<float> field_states;    // [n_layers][hidden_size]
    std::vector<float> semantic_pool;   // [semantic_slots][hidden_size]
    std::vector<float> gaussian_comps;  // [hidden_size]
    std::vector<int> ring_offsets;      // [n_layers]
    
    SFA_Memory_Context()
        : n_layers(0), hidden_size(0), head_dim(0), n_heads(0) {}
    
    ~SFA_Memory_Context() = default;
    
    void init(int n_layers_, int hidden_size_, int head_dim_, int n_heads_) {
        n_layers = n_layers_;
        hidden_size = hidden_size_;
        head_dim = head_dim_;
        n_heads = n_heads_;
        
        ring_buffers.resize(n_layers * config.ring_size * hidden_size, 0);
        field_states.resize(n_layers * hidden_size, 0);
        semantic_pool.resize(config.semantic_slots * hidden_size, 0);
        gaussian_comps.resize(hidden_size, 0);
        ring_offsets.resize(n_layers, 0);
    }
    
    void reset() {
        std::memset(ring_buffers.data(), 0, ring_buffers.size() * sizeof(float));
        std::memset(field_states.data(), 0, field_states.size() * sizeof(float));
        std::memset(semantic_pool.data(), 0, semantic_pool.size() * sizeof(float));
        std::memset(gaussian_comps.data(), 0, gaussian_comps.size() * sizeof(float));
        std::fill(ring_offsets.begin(), ring_offsets.end(), 0);
    }
    
    // 获取第layer层的ring buffer指针
    inline float* get_ring_buffer(int layer) {
        return ring_buffers.data() + layer * config.ring_size * hidden_size;
    }
    
    // 获取第layer层的field state指针
    inline float* get_field_state(int layer) {
        return field_states.data() + layer * hidden_size;
    }
    
    // 更新ring buffer（滑动窗口）
    void update_ring_buffer(int layer, const float* new_value) {
        float* ring = get_ring_buffer(layer);
        int offset = ring_offsets[layer];
        
        // 写入新值
        std::memcpy(ring + offset * hidden_size, new_value, hidden_size * sizeof(float));
        
        // 更新offset
        ring_offsets[layer] = (offset + 1) % config.ring_size;
    }
    
    // 计算ring buffer均值
    void compute_ring_mean(int layer, float* out) {
        const float* ring = get_ring_buffer(layer);
        std::memset(out, 0, hidden_size * sizeof(float));
        
        for (int i = 0; i < config.ring_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                out[j] += ring[i * hidden_size + j];
            }
        }
        
        // 归一化
        float inv_size = 1.0f / config.ring_size;
        for (int j = 0; j < hidden_size; j++) {
            out[j] *= inv_size;
        }
    }
    
    // EMA场状态更新
    void update_field_state(int layer, const float* attn_output) {
        float* field = get_field_state(layer);
        float gamma = config.ema_gamma;
        
        for (int j = 0; j < hidden_size; j++) {
            field[j] = gamma * field[j] + (1.0f - gamma) * attn_output[j];
        }
    }
    
    // 计算SFA增强
    void compute_enhancement(
        int layer,
        const float* attn_output,
        float* enhancement,
        float alpha) {
        
        // 1. Ring mean
        compute_ring_mean(layer, enhancement);
        
        // 2. Get field state
        float* field = get_field_state(layer);
        
        // 3. Add field contribution (0.5 * field)
        for (int j = 0; j < hidden_size; j++) {
            enhancement[j] += 0.5f * field[j];
        }
        
        // 4. Clip
        float clip = config.enhancement_clip;
        for (int j = 0; j < hidden_size; j++) {
            if (enhancement[j] > clip) enhancement[j] = clip;
            if (enhancement[j] < -clip) enhancement[j] = -clip;
        }
        
        // 5. Scale by alpha
        for (int j = 0; j < hidden_size; j++) {
            enhancement[j] *= alpha;
        }
    }
    
    // 计算effective alpha（随层数和序列位置变化）
    float compute_alpha(int layer, int seq_pos) {
        float ratio = (float)layer / std::max(n_layers - 1, 1);
        float seq_factor = 1.0f / (1.0f + seq_pos * 0.01f);
        return config.alpha_base * (0.3f + ratio * 0.7f) * seq_factor;
    }
};
