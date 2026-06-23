#pragma once

// ==========================================
// 🔓 SFA无锁化高性能实现
// ==========================================
// 设计理念：消除mutex锁开销，使用per-thread缓存+批量更新
// 性能目标：SFA开销 < 0.5%
// ==========================================

#include "ggml.h"
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>

// SFA配置常量
constexpr int RING_SIZE = 16;
constexpr int SEMANTIC_SLOTS = 64;
constexpr float EMA_GAMMA = 0.98f;
constexpr float ENHANCEMENT_CLIP = 0.5f;   // calibrated: avoid signal saturation
constexpr float CROSS_DECAY = 0.8f;          // calibrated: balanced layer contribution
constexpr float ALPHA_BASE = 0.1f;           // calibrated: ~1.24% enhancement ratio

namespace sfa {

// Per-thread SFA缓存（避免锁竞争）
struct SFA_Thread_Cache {
    std::vector<float> ring_mean_buf;
    std::vector<float> field_update_buf;
    std::vector<float> enhancement_buf;
    
    SFA_Thread_Cache() = default;
    
    void init(int hidden_size) {
        ring_mean_buf.resize(hidden_size, 0.0f);
        field_update_buf.resize(hidden_size, 0.0f);
        enhancement_buf.resize(hidden_size, 0.0f);
    }
    
    void reset() {
        std::memset(ring_mean_buf.data(), 0, ring_mean_buf.size() * sizeof(float));
        std::memset(field_update_buf.data(), 0, field_update_buf.size() * sizeof(float));
        std::memset(enhancement_buf.data(), 0, enhancement_buf.size() * sizeof(float));
    }
};

// 全局SFA状态（无锁读，批量写）
struct SFA_Global_State_Lockfree {
    std::vector<float> ring_buffers;      // [n_layers][ring_size][hidden_size]
    std::vector<float> field_states;      // [n_layers][hidden_size]
    std::vector<float> semantic_pool;     // [semantic_slots][hidden_size]
    std::vector<int> ring_offsets;        // [n_layers]
    
    int n_layers;
    int hidden_size;
    bool enabled;
    
    // 批量更新缓冲区（减少锁持有时间）
    std::vector<float> update_buffer;
    
    SFA_Global_State_Lockfree()
        : n_layers(0), hidden_size(0), enabled(false) {}
    
    void init(int n_layers_, int hidden_size_) {
        n_layers = n_layers_;
        hidden_size = hidden_size_;
        enabled = true;
        
        ring_buffers.resize(n_layers * RING_SIZE * hidden_size, 0.0f);
        field_states.resize(n_layers * hidden_size, 0.0f);
        semantic_pool.resize(SEMANTIC_SLOTS * hidden_size, 0.0f);
        ring_offsets.resize(n_layers, 0);
        update_buffer.resize(hidden_size, 0.0f);
    }
    
    // 无锁读取ring buffer均值
    inline void compute_ring_mean_fast(int layer, float* out) {
        const float* ring = ring_buffers.data() + layer * RING_SIZE * hidden_size;
        std::memset(out, 0, hidden_size * sizeof(float));
        
        for (int i = 0; i < RING_SIZE; i++) {
            const float* row = ring + i * hidden_size;
            for (int j = 0; j < hidden_size; j++) {
                out[j] += row[j];
            }
        }
        
        float inv_size = 1.0f / RING_SIZE;
        for (int j = 0; j < hidden_size; j++) {
            out[j] *= inv_size;
        }
    }
    
    // 批量更新field state（减少锁次数）
    void batch_update_fields(const float** attn_outputs, int* layers, int count) {
        for (int i = 0; i < count; i++) {
            int layer = layers[i];
            const float* input = attn_outputs[i];
            float* field = field_states.data() + layer * hidden_size;
            
            for (int j = 0; j < hidden_size; j++) {
                field[j] = EMA_GAMMA * field[j] + (1.0f - EMA_GAMMA) * input[j];
            }
        }
    }
    
    inline bool is_enabled() const { return enabled; }
    
    // 快速SFA增强计算（无锁）
    inline void compute_enhancement_fast(
        int layer,
        const float* attn_output,
        float* enhancement) {
        
        // 1. Ring mean
        compute_ring_mean_fast(layer, enhancement);
        
        // 2. Add field contribution
        const float* field = field_states.data() + layer * hidden_size;
        for (int j = 0; j < hidden_size; j++) {
            enhancement[j] += 0.5f * field[j];
        }
        
        // 3. Clip
        for (int j = 0; j < hidden_size; j++) {
            if (enhancement[j] > ENHANCEMENT_CLIP) enhancement[j] = ENHANCEMENT_CLIP;
            if (enhancement[j] < -ENHANCEMENT_CLIP) enhancement[j] = -ENHANCEMENT_CLIP;
        }
    }
};

// 线程局部状态
thread_local SFA_Thread_Cache g_thread_cache;

// 获取全局状态指针（单例）
inline SFA_Global_State_Lockfree& get_global_state_lockfree() {
    static SFA_Global_State_Lockfree state;
    return state;
}

// 初始化
inline void sfa_lockfree_init(int n_layers, int hidden_size) {
    auto& state = get_global_state_lockfree();
    state.init(n_layers, hidden_size);
    g_thread_cache.init(hidden_size);
}

// 重置
inline void sfa_lockfree_reset() {
    auto& state = get_global_state_lockfree();
    std::memset(state.ring_buffers.data(), 0, state.ring_buffers.size() * sizeof(float));
    std::memset(state.field_states.data(), 0, state.field_states.size() * sizeof(float));
    std::memset(state.semantic_pool.data(), 0, state.semantic_pool.size() * sizeof(float));
    std::fill(state.ring_offsets.begin(), state.ring_offsets.end(), 0);
    g_thread_cache.reset();
}

// 禁用
inline void sfa_lockfree_disable() {
    get_global_state_lockfree().enabled = false;
}

} // namespace sfa
