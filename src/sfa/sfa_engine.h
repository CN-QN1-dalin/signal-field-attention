#pragma once

// ==========================================
// 🧠 SFA完整三通道引擎 v4.0
// ==========================================
// 通道1: RingBuffer - 短期记忆（滑动窗口）
// 通道2: EMA Field - 长期场状态（指数衰减）
// 通道3: Semantic Pool - 语义注意力（跨token关联）
// ==========================================

#include "/tmp/llama.cpp/ggml/include/ggml.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace sfa {

// 三通道SFA配置
struct SFA_Config {
    bool enabled;
    float alpha_base;           // 基础增强强度
    float cross_decay;          // 跨层衰减值
    float ema_gamma;            // EMA衰减系数
    float gaussian_sigma;       // 高斯分布标准差
    float enhancement_clip;     // 增强裁剪阈值
    int ring_size;              // RingBuffer大小
    int semantic_slots;         // 语义槽数量
    float semantic_temperature; // 语义注意力温度
    
    SFA_Config()
        : enabled(true),
          alpha_base(0.1f),       // calibrated: ~1.24% enhancement ratio
          cross_decay(0.8f),      // calibrated: balanced layer contribution
          ema_gamma(0.98f),
          gaussian_sigma(1.0f),
          enhancement_clip(0.5f), // increased from 0.01 to avoid signal saturation
          ring_size(16),
          semantic_slots(64),
          semantic_temperature(0.07f) {}
};

// 单通道状态
struct Channel_State {
    std::vector<float> values;
    int size;
    
    void init(int n) {
        size = n;
        values.resize(n, 0.0f);
    }
    
    void reset() {
        std::memset(values.data(), 0, size * sizeof(float));
    }
};

// 完整三通道SFA上下文
class SFA_Engine {
private:
    SFA_Config config;
    int n_layers;
    int hidden_size;
    
    // 通道1: RingBuffer
    std::vector<std::vector<float>> ring_buffers;  // [layer][ring_size * hidden]
    std::vector<int> ring_offsets;
    
    // 通道2: EMA Field
    std::vector<std::vector<float>> field_states;  // [layer][hidden]
    
    // 通道3: Semantic Pool
    std::vector<float> semantic_pool;              // [slots * hidden]
    std::vector<float> semantic_weights;           // [slots]
    
    // 高性能缓存（thread_local）
    std::vector<float> temp_buf1;
    std::vector<float> temp_buf2;
    std::vector<float> temp_buf3;
    
public:
    SFA_Engine() = default;
    ~SFA_Engine() = default;
    
    // Public accessors for temporary buffers
    std::vector<float>& get_temp_buf1() { return temp_buf1; }
    std::vector<float>& get_temp_buf2() { return temp_buf2; }
    std::vector<float>& get_temp_buf3() { return temp_buf3; }
    
    // 初始化
    void init(int n_layers_, int hidden_size_) {
        n_layers = n_layers_;
        hidden_size = hidden_size_;
        
        // 初始化RingBuffer
        ring_buffers.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            ring_buffers[i].resize(config.ring_size * hidden_size, 0.0f);
        }
        ring_offsets.resize(n_layers, 0);
        
        // 初始化Field State
        field_states.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            field_states[i].resize(hidden_size, 0.0f);
        }
        
        // 初始化Semantic Pool
        semantic_pool.resize(config.semantic_slots * hidden_size, 0.0f);
        semantic_weights.resize(config.semantic_slots, 0.0f);
        
        // 初始化临时缓冲区
        temp_buf1.resize(hidden_size);
        temp_buf2.resize(hidden_size);
        temp_buf3.resize(hidden_size);
    }
    
    // 重置所有状态
    void reset() {
        for (auto& rb : ring_buffers) {
            std::memset(rb.data(), 0, rb.size() * sizeof(float));
        }
        std::fill(ring_offsets.begin(), ring_offsets.end(), 0);
        
        for (auto& fs : field_states) {
            std::memset(fs.data(), 0, fs.size() * sizeof(float));
        }
        
        std::memset(semantic_pool.data(), 0, semantic_pool.size() * sizeof(float));
        std::memset(semantic_weights.data(), 0, semantic_weights.size() * sizeof(float));
        
        temp_buf1.assign(hidden_size, 0.0f);
        temp_buf2.assign(hidden_size, 0.0f);
        temp_buf3.assign(hidden_size, 0.0f);
    }
    
    // 更新RingBuffer
    void update_ring_buffer(int layer, const float* new_value) {
        auto& ring = ring_buffers[layer];
        int offset = ring_offsets[layer];
        int start = offset * hidden_size;
        
        std::memcpy(ring.data() + start, new_value, hidden_size * sizeof(float));
        ring_offsets[layer] = (offset + 1) % config.ring_size;
    }
    
    // 计算RingBuffer均值
    void compute_ring_mean(int layer, float* out) {
        const auto& ring = ring_buffers[layer];
        std::memset(out, 0, hidden_size * sizeof(float));
        
        for (int i = 0; i < config.ring_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                out[j] += ring[i * hidden_size + j];
            }
        }
        
        float inv = 1.0f / config.ring_size;
        for (int j = 0; j < hidden_size; j++) {
            out[j] *= inv;
        }
    }
    
    // 更新EMA Field
    void update_field_state(int layer, const float* input) {
        auto& field = field_states[layer];
        float gamma = config.ema_gamma;
        
        for (int j = 0; j < hidden_size; j++) {
            field[j] = gamma * field[j] + (1.0f - gamma) * input[j];
        }
    }
    
    // 计算Semantic Pool注意力
    void compute_semantic_attention(const float* query, float* output) {
        int slots = config.semantic_slots;
        
        // 计算dot products
        for (int i = 0; i < slots; i++) {
            float dot = 0.0f;
            const float* slot = semantic_pool.data() + i * hidden_size;
            for (int j = 0; j < hidden_size; j++) {
                dot += slot[j] * query[j];
            }
            semantic_weights[i] = dot / std::sqrt(hidden_size);
        }
        
        // Softmax with temperature
        float max_val = semantic_weights[0];
        for (int i = 1; i < slots; i++) {
            max_val = std::max(max_val, semantic_weights[i]);
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < slots; i++) {
            semantic_weights[i] = std::exp((semantic_weights[i] - max_val) / config.semantic_temperature);
            sum_exp += semantic_weights[i];
        }
        
        // 加权求和
        std::memset(output, 0, hidden_size * sizeof(float));
        for (int i = 0; i < slots; i++) {
            float weight = semantic_weights[i] / sum_exp;
            const float* slot = semantic_pool.data() + i * hidden_size;
            for (int j = 0; j < hidden_size; j++) {
                output[j] += weight * slot[j];
            }
        }
    }
    
    // 完整三通道SFA增强
    void compute_triple_channel_enhancement(
        int layer,
        const float* attn_output,
        float* enhancement) {
        
        // 通道1: RingBuffer Mean
        compute_ring_mean(layer, temp_buf1.data());
        
        // 通道2: EMA Field
        update_field_state(layer, attn_output);
        const auto& field = field_states[layer];
        for (int j = 0; j < hidden_size; j++) {
            temp_buf2[j] = 0.5f * field[j];
        }
        
        // 通道3: Semantic Pool
        compute_semantic_attention(temp_buf1.data(), temp_buf3.data());
        
        // 融合三通道: enhancement = ring_mean + 0.5 * field + 0.5 * semantic
        for (int j = 0; j < hidden_size; j++) {
            enhancement[j] = temp_buf1[j] + temp_buf2[j] + 0.5f * temp_buf3[j];
        }
        
        // Clip
        for (int j = 0; j < hidden_size; j++) {
            if (enhancement[j] > config.enhancement_clip) {
                enhancement[j] = config.enhancement_clip;
            } else if (enhancement[j] < -config.enhancement_clip) {
                enhancement[j] = -config.enhancement_clip;
            }
        }
    }
    
    // 计算effective alpha
    float compute_effective_alpha(int layer, int seq_pos) {
        float layer_ratio = (float)layer / std::max(n_layers - 1, 1);
        float seq_factor = 1.0f / (1.0f + seq_pos * 0.01f);
        return config.alpha_base * (0.3f + layer_ratio * 0.7f) * seq_factor * std::pow(config.cross_decay, layer);
    }
    
    // 获取field state
    const std::vector<float>& get_field_state(int layer) const {
        return field_states[layer];
    }
    
    // 获取配置
    const SFA_Config& get_config() const { return config; }
    SFA_Config& get_config_mut() { return config; }
    
    // 获取层数
    int get_n_layers() const { return n_layers; }
    
    // 获取hidden size
    int get_hidden_size() const { return hidden_size; }
};

// 全局引擎实例
inline SFA_Engine& get_sfa_engine() {
    static SFA_Engine engine;
    return engine;
}

// 初始化
inline void sfa_engine_init(int n_layers, int hidden_size) {
    get_sfa_engine().init(n_layers, hidden_size);
}

// 重置
inline void sfa_engine_reset() {
    get_sfa_engine().reset();
}

} // namespace sfa
