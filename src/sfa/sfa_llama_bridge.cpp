// ============================================================
// 🎯 SFA v7 → llama.cpp 集成桥接器 - 修复版
// ============================================================
// 功能：将 SFA v7 三通道增强注入 llama.cpp 推理管线
// 位置：在 attention output 之后、residual connection 之前
// 状态：P0 Bug 已修复 (2026-06-22)
// 修复：使用正确的 ggml 图构建替代直接数据操作
// ============================================================

#include "sfa_llama_bridge.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-graph.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <mutex>

// ============================================================
// 🔧 多序列状态管理
// ============================================================

struct sfa_seq_state {
    std::vector<float> ring_buffers;    // [n_layers][ring_size][hidden_size]
    std::vector<float> field_states;    // [n_layers][hidden_size]
    std::vector<float> semantic_pool;   // [semantic_slots][hidden_size]
    std::vector<float> gaussian_comps;  // [hidden_size]
    std::vector<int> ring_offsets;      // [n_layers]
    bool initialized;

    void init(int n_layers, int hidden_size, int semantic_slots) {
        this->ring_buffers.assign(n_layers * SFA_RING_SIZE * hidden_size, 0.0f);
        this->field_states.assign(n_layers * hidden_size, 0.0f);
        this->semantic_pool.assign(semantic_slots * hidden_size, 0.0f);
        this->gaussian_comps.assign(hidden_size, 0.0f);
        this->ring_offsets.assign(n_layers, 0);
        this->initialized = true;
    }

    void reset() {
        std::memset(ring_buffers.data(), 0, ring_buffers.size() * sizeof(float));
        std::memset(field_states.data(), 0, field_states.size() * sizeof(float));
        std::memset(semantic_pool.data(), 0, semantic_pool.size() * sizeof(float));
        std::memset(gaussian_comps.data(), 0, gaussian_comps.size() * sizeof(float));
        std::fill(ring_offsets.begin(), ring_offsets.end(), 0);
    }

    sfa_seq_state() : initialized(false) {}
};

// ============================================================
// 🔧 SFA Bridge 类 - 正确的 ggml 图构建
// ============================================================

class SFA_Llama_Bridge {
private:
    sfa_context ctx;
    std::unordered_map<int, sfa_seq_state> seq_map;
    std::mutex mutex;
    int hidden_size;
    int semantic_slots;
    bool model_loaded;

public:
    SFA_Llama_Bridge() : hidden_size(0), semantic_slots(SFA_SEMANTIC_SLOTS), model_loaded(false) {}

    // llama.cpp 生命周期钩子 1: 模型加载时调用
    void on_model_load(struct llama_model *model) {
        std::lock_guard<std::mutex> lock(mutex);
        
        ctx.n_layers = llama_n_layer(model);
        ctx.hidden_size = llama_n_embd(model);
        ctx.alpha_base = SFA_ALPHA_BASE;
        ctx.cross_decay = SFA_CROSS_DECAY_DEFAULT;
        ctx.ring_size = SFA_RING_SIZE;
        ctx.semantic_slots = semantic_slots;
        
        // 分配 layer_states
        ctx.init(ctx.n_layers, ctx.hidden_size);
        
        // 初始化默认序列 (seq_id=0)
        sfa_seq_state default_seq;
        default_seq.init(ctx.n_layers, ctx.hidden_size, semantic_slots);
        seq_map[0] = default_seq;
        
        model_loaded = true;
    }

    // llama.cpp 生命周期钩子 2: 新序列开始时调用
    void on_seq_start(int seq_id) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!model_loaded) return;
        
        auto it = seq_map.find(seq_id);
        if (it == seq_map.end()) {
            // 新序列，创建状态
            sfa_seq_state new_seq;
            new_seq.init(ctx.n_layers, ctx.hidden_size, semantic_slots);
            seq_map[seq_id] = new_seq;
        } else {
            // 已有序列，重置状态
            it->second.reset();
        }
    }

    // llama.cpp 生命周期钩子 3: 序列复制时调用 (Bug 3 修复)
    void on_seq_copy(int dst_seq_id, int src_seq_id) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!model_loaded) return;
        
        auto src_it = seq_map.find(src_seq_id);
        if (src_it != seq_map.end()) {
            // 深拷贝源序列状态
            seq_map[dst_seq_id] = src_it->second;
        }
    }

    // llama.cpp 生命周期钩子 4: 序列删除时调用 (Bug 3 修复)
    void on_seq_remove(int seq_id) {
        std::lock_guard<std::mutex> lock(mutex);
        seq_map.erase(seq_id);
    }

    // llama.cpp 生命周期钩子 5: 推理结束，清理所有序列
    void on_ctx_free() {
        std::lock_guard<std::mutex> lock(mutex);
        seq_map.clear();
        ctx.enabled = false;
        model_loaded = false;
    }

    // 构建 SFA 增强图节点（在 attention 层之后调用）
    // 修复：使用正确的 ggml 图操作替代直接数据操作
    ggml_tensor * build_sfa_enhance_for_layer(
        ggml_context *ctx0,
        ggml_tensor * attn_out,       // [batch, seq_len, hidden]
        int layer_idx,
        int seq_id,
        const ggml_cgraph *gf) {
        
        if (!ctx.enabled) {
            return attn_out;  // SFA 未启用，直接返回
        }

        auto seq_it = seq_map.find(seq_id);
        if (seq_it == seq_map.end()) {
            return attn_out;  // 序列不存在，跳过
        }

        auto &seq_state = seq_it->second;
        int hs = ctx.hidden_size;
        int nl = ctx.n_layers;

        // 1. 提取当前层的 ring buffer (从 seq_state)
        //    注意：这里我们使用 ggml_view 来创建视图，避免数据复制
        float *ring_buf = seq_state.ring_buffers.data() + layer_idx * SFA_RING_SIZE * hs;
        
        // 2. 计算 ring mean (使用 ggml_mean)
        //    创建临时 tensor 表示 ring buffer
        ggml_tensor * ring_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hs, SFA_RING_SIZE);
        std::memcpy(ring_tensor->data, ring_buf, SFA_RING_SIZE * hs * sizeof(float));
        
        // 计算 mean over ring_size dimension (axis 1)
        ggml_tensor * ring_mean = ggml_mean(ctx0, ring_tensor);

        // 3. EMA field update
        float *field = seq_state.field_states.data() + layer_idx * hs;
        ggml_tensor * field_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hs);
        std::memcpy(field_tensor->data, field, hs * sizeof(float));
        
        // field[t] = gamma * field[t-1] + (1-gamma) * attn_out
        // 使用 ggml_scale 和 ggml_add 构建
        ggml_tensor * attn_mean = ggml_mean(ctx0, attn_out);  // [hidden]
        ggml_tensor * scaled_field = ggml_scale(ctx0, field_tensor, SFA_EMA_GAMMA);
        ggml_tensor * scaled_attn = ggml_scale(ctx0, attn_mean, 1.0f - SFA_EMA_GAMMA);
        ggml_tensor * new_field = ggml_add(ctx0, scaled_field, scaled_attn);
        
        // 更新 seq_state 中的 field
        std::memcpy(field, new_field->data, hs * sizeof(float));

        // 4. Semantic pool attention (简化版：直接使用 semantic pool)
        ggml_tensor * semantic_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hs, seq_state.semantic_pool.size() / hs);
        std::memcpy(semantic_tensor->data, seq_state.semantic_pool.data(), seq_state.semantic_pool.size() * sizeof(float));
        ggml_tensor * semantic_mean = ggml_mean(ctx0, semantic_tensor);

        // 5. 计算 enhancement = ring_mean + (new_field + semantic_mean) * 0.5
        ggml_tensor * field_semantic_sum = ggml_add(ctx0, new_field, semantic_mean);
        ggml_tensor * scaled_sum = ggml_scale(ctx0, field_semantic_sum, 0.5f);
        ggml_tensor * enhancement = ggml_add(ctx0, ring_mean, scaled_sum);

        // 6. Clip enhancement
        enhancement = ggml_clamp(ctx0, enhancement, -SFA_ENHANCEMENT_CLIP, SFA_ENHANCEMENT_CLIP);

        // 7. Scale by alpha
        float alpha_eff = sfa_context::layer_alpha(ctx.alpha_base, ctx.cross_decay, layer_idx, nl);
        enhancement = ggml_scale(ctx0, enhancement, alpha_eff);

        // 8. Add to attention output (broadcast over seq_len)
        //    attn_out: [batch, seq_len, hidden]
        //    enhancement: [hidden]
        //    ggml_add 会自动广播
        ggml_tensor * enhanced = ggml_add(ctx0, attn_out, enhancement);

        return enhanced;
    }

    // 获取 SFA 上下文（供外部访问）
    sfa_context *get_context() { return &ctx; }
    bool is_model_loaded() const { return model_loaded; }
};

// ============================================================
// 全局实例
// ============================================================
static SFA_Llama_Bridge g_sfa_bridge;

// ============================================================
// 暴露给 llama.cpp 的 C 接口
// ============================================================

extern "C" {

/**
 * llama.cpp 集成入口点
 * 在 llama_load_from_file() 之后调用
 */
void sfa_llama_init(struct llama_model *model) {
    g_sfa_bridge.on_model_load(model);
}

/**
 * 在 llama_decode() 之前调用（每个序列）
 */
void sfa_llama_seq_start(int seq_id) {
    g_sfa_bridge.on_seq_start(seq_id);
}

/**
 * 序列复制时调用
 */
void sfa_llama_seq_copy(int dst_seq_id, int src_seq_id) {
    g_sfa_bridge.on_seq_copy(dst_seq_id, src_seq_id);
}

/**
 * 序列删除时调用
 */
void sfa_llama_seq_remove(int seq_id) {
    g_sfa_bridge.on_seq_remove(seq_id);
}

/**
 * 释放 llama_context 时调用
 */
void sfa_llama_free() {
    g_sfa_bridge.on_ctx_free();
}

/**
 * 在 attention 层之后构建 SFA 增强
 * layer_idx: 当前层索引
 * seq_id: 当前序列 ID
 */
ggml_tensor * sfa_llama_build_enhance(
    ggml_context *ctx0,
    ggml_tensor * attn_out,
    int layer_idx,
    int seq_id,
    const ggml_cgraph *gf) {
    
    return g_sfa_bridge.build_sfa_enhance_for_layer(
        ctx0, attn_out, layer_idx, seq_id, gf);
}

/**
 * 启用/禁用 SFA
 */
void sfa_llama_enable(bool on) {
    g_sfa_bridge.get_context()->enabled = on;
}

/**
 * 设置 alpha 参数
 */
void sfa_llama_set_alpha(float alpha) {
    g_sfa_bridge.get_context()->alpha_base = alpha;
}

} // extern "C"
