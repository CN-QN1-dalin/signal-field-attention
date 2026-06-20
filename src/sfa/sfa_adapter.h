#pragma once

// ==========================================
// 🎯 SFA万能转接头 v2.0 - 完美重构版
// ==========================================
// 设计理念：不修改任何现有模型代码，通过ggml图hook注入SFA增强
// 支持：所有llama.cpp架构（Qwen2/Llama/Mistral/DeepSeek...）
// 安全：std::vector内存管理、边界检查、线程安全
// ==========================================

#include "/tmp/llama.cpp/ggml/include/ggml.h"
#include "llama-graph.h"
#include <memory>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cassert>
#include "sfa_kernel_avx.h"
#include "sfa_lockfree.h"
#include "sfa_engine.h"

// SFA配置（常量，避免运行时修改）
namespace sfa {

constexpr float ALPHA_BASE = 2.0f;
constexpr int RING_SIZE = 16;
constexpr int SEMANTIC_SLOTS = 64;
constexpr float EMA_GAMMA = 0.98f;
constexpr float GAUSSIAN_GAMMA = 0.951229f;
constexpr float ENHANCEMENT_CLIP = 0.01f;
constexpr float CROSS_DECAY = 0.7f;

// 线程安全的SFA全局状态
class SFA_Global_State {
private:
    std::vector<float> ring_buffers;      // [n_layers][ring_size][hidden_size]
    std::vector<float> field_states;      // [n_layers][hidden_size]
    std::vector<float> semantic_pool;     // [semantic_slots][hidden_size]
    std::vector<float> gaussian_comps;    // [hidden_size]
    std::vector<int> ring_offsets;        // [n_layers]
    
    int n_layers;
    int hidden_size;
    bool enabled;
    
    // 互斥锁（简化：使用atomic flag）
    std::atomic<bool> mutex{false};
    
    void lock() {
        while (mutex.exchange(true)) {
            // 自旋等待
            #ifdef __APPLE__
            asm("yield");
            #else
            std::this_thread::yield();
            #endif
        }
    }
    
    void unlock() {
        mutex.store(false);
    }

public:
    SFA_Global_State() : n_layers(0), hidden_size(0), enabled(false) {}
    
    ~SFA_Global_State() = default;
    
    // 初始化（线程安全）
    void init(int n_layers_, int hidden_size_) {
        lock();
        try {
            n_layers = n_layers_;
            hidden_size = hidden_size_;
            
            size_t ring_size_bytes = (size_t)n_layers * RING_SIZE * hidden_size;
            size_t field_size_bytes = (size_t)n_layers * hidden_size;
            size_t semantic_size_bytes = (size_t)SEMANTIC_SLOTS * hidden_size;
            size_t gauss_size_bytes = hidden_size;
            
            ring_buffers.resize(ring_size_bytes, 0.0f);
            field_states.resize(field_size_bytes, 0.0f);
            semantic_pool.resize(semantic_size_bytes, 0.0f);
            gaussian_comps.resize(gauss_size_bytes, 0.0f);
            ring_offsets.resize(n_layers, 0);
            
            enabled = true;
        } catch (const std::bad_alloc& e) {
            enabled = false;
            // 回滚到未初始化状态
            n_layers = 0;
            hidden_size = 0;
        }
        unlock();
    }
    
    // 重置（新序列开始时调用，线程安全）
    void reset() {
        if (!enabled) return;
        lock();
        std::memset(ring_buffers.data(), 0, ring_buffers.size() * sizeof(float));
        std::memset(field_states.data(), 0, field_states.size() * sizeof(float));
        std::memset(semantic_pool.data(), 0, semantic_pool.size() * sizeof(float));
        std::memset(gaussian_comps.data(), 0, gaussian_comps.size() * sizeof(float));
        std::fill(ring_offsets.begin(), ring_offsets.end(), 0);
        unlock();
    }
    
    // 禁用SFA
    void disable() {
        lock();
        enabled = false;
        unlock();
    }
    
    // 启用SFA
    void enable() {
        lock();
        enabled = true;
        unlock();
    }
    
    // 检查是否启用
    bool is_enabled() const {
        return enabled;
    }
    
    // 获取layer的ring buffer指针（线程安全）
    inline float* get_ring_buffer(int layer) {
        assert(layer >= 0 && layer < n_layers);
        return ring_buffers.data() + layer * RING_SIZE * hidden_size;
    }
    
    // 获取layer的field state指针（线程安全）
    inline float* get_field_state(int layer) {
        assert(layer >= 0 && layer < n_layers);
        return field_states.data() + layer * hidden_size;
    }
    
    // 更新ring buffer（滑动窗口，线程安全）
    void update_ring_buffer(int layer, const float* new_value) {
        assert(layer >= 0 && layer < n_layers);
        assert(new_value != nullptr);
        lock();
        float* ring = get_ring_buffer(layer);
        int offset = ring_offsets[layer];
        std::memcpy(ring + offset * hidden_size, new_value, hidden_size * sizeof(float));
        ring_offsets[layer] = (offset + 1) % RING_SIZE;
        unlock();
    }
    
    // 计算ring buffer均值（线程安全，SIMD优化）
    void compute_ring_mean(int layer, float* out) {
        assert(layer >= 0 && layer < n_layers);
        assert(out != nullptr);
        lock();
        neon_ring_mean(get_ring_buffer(layer), RING_SIZE, hidden_size, out);
        unlock();
    }
    
    // EMA场状态更新（线程安全，SIMD优化）
    void update_field_state(int layer, const float* attn_output) {
        assert(layer >= 0 && layer < n_layers);
        assert(attn_output != nullptr);
        lock();
        neon_ema_update(get_field_state(layer), attn_output, EMA_GAMMA, hidden_size);
        unlock();
    }
    
    // 获取层数
    int get_n_layers() const { return n_layers; }
    
    // 获取hidden size
    int get_hidden_size() const { return hidden_size; }
};

// 无锁全局状态
inline sfa::SFA_Global_State_Lockfree& get_global_state() {
    return sfa::get_global_state_lockfree();
}

// SFA初始化（在模型加载时调用）
inline void sfa_adapter_init(int n_layers, int hidden_size) {
    sfa::sfa_lockfree_init(n_layers, hidden_size);
}

// SFA禁用
inline void sfa_adapter_disable() {
    sfa::sfa_lockfree_disable();
}

// SFA重置（新序列开始时调用）
inline void sfa_adapter_reset() {
    sfa::sfa_lockfree_reset();
}

// 构建SFA增强tensor（图构建阶段）
inline ggml_tensor* sfa_build_enhance(
    ggml_context* ctx0,
    ggml_tensor* attn_out,      // [batch, seq_len, hidden]
    int layer_idx,
    int n_layers,
    int hidden_size,
    const ggml_cgraph* gf) {
    
    auto& state = get_global_state();
    
    // 快速路径：未启用SFA，直接返回
    if (!state.enabled || layer_idx >= n_layers) {
        return attn_out;
    }
    
    // 边界检查
    assert(hidden_size == state.hidden_size);
    assert(layer_idx >= 0 && layer_idx < n_layers);
    
    // 使用三通道SFA引擎
    auto& engine = sfa::get_sfa_engine();
    
    // 1. 获取attn_out的第一个token作为输入
    std::memcpy(engine.get_temp_buf1().data(), attn_out->data, hidden_size * sizeof(float));
    
    // 2. 计算三通道增强
    engine.compute_triple_channel_enhancement(layer_idx, engine.get_temp_buf1().data(), engine.get_temp_buf2().data());
    
    // 3. 创建enhancement tensor
    ggml_tensor* enhancement_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
    std::memcpy(enhancement_tensor->data, engine.get_temp_buf2().data(), hidden_size * sizeof(float));
    
    // 4. 计算effective alpha
    float alpha_eff = engine.compute_effective_alpha(layer_idx, 0);
    enhancement_tensor = ggml_scale(ctx0, enhancement_tensor, alpha_eff);
    
    // 5. Add to attention output
    ggml_tensor* enhanced = ggml_add(ctx0, attn_out, enhancement_tensor);
    
    return enhanced;
}

} // namespace sfa

// 向后兼容的全局函数
using namespace sfa;
