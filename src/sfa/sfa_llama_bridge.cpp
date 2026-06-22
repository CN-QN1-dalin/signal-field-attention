// ============================================================
// 🎯 SFA v7 → llama.cpp 集成桥接器
// ============================================================
// 功能：将 SFA v7 三通道增强注入 llama.cpp 推理管线
// 位置：在 attention output 之后、residual connection 之前
// 状态：P0 Bug 已修复 (2026-06-22)
// ============================================================

#include "sfa_llama_cpp.h"
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
// 🔧 P0 Bug 修复 1: field_state 多序列隔离
// ============================================================
// 问题：原 sfa_context::reset_all() 忽略 seq_id，导致多序列状态污染
// 修复：使用 seq_map 追踪每个序列的状态，reset 时按 seq_id 隔离

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
// 🔧 P0 Bug 修复 2: n_sfa_layers → n_layers 统一
// ============================================================
// 问题：build_sfa_enhance 内部可能混淆 n_layers 和 n_sfa_layers
// 修复：统一使用 n_layers，移除所有 n_sfa_layers 引用

// ============================================================
// 🔧 P0 Bug 修复 3: seq_cp / seq_rm 正确实现
// ============================================================
// 问题：原 sfa_reset 完全忽略 seq_id，没有序列复制/删除支持
// 修复：添加 copy_seq / remove_seq 方法

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

        // 1. 提取当前层的 ring buffer
        float *ring_buf = seq_state.ring_buffers.data() + layer_idx * SFA_RING_SIZE * hs;
        int ring_offset = seq_state.ring_offsets[layer_idx];

        // 2. 更新 ring buffer（写入最新 attention output 的 mean）
        //    attn_out: [batch, seq_len, hidden]
        //    取最后一个 token 的 mean 作为当前状态的近似
        for (int h = 0; h < hs; ++h) {
            float sum = 0.0f;
            // 简化：取 batch 维度第一个元素的 mean
            // 实际实现需要遍历 batch 和 seq_len
            sum += attn_out->data[h];  // 简化版本
            ring_buf[(ring_offset * hs) + h] = sum / hs;
        }
        seq_state.ring_offsets[layer_idx] = (ring_offset + 1) % SFA_RING_SIZE;

        // 3. 计算 ring mean
        float ring_mean[hs];
        std::memset(ring_mean, 0, hs * sizeof(float));
        for (int r = 0; r < SFA_RING_SIZE; ++r) {
            for (int h = 0; h < hs; ++h) {
                ring_mean[h] += ring_buf[(r * hs) + h];
            }
        }
        for (int h = 0; h < hs; ++h) {
            ring_mean[h] /= SFA_RING_SIZE;
        }

        // 4. EMA field update
        float *field = seq_state.field_states.data() + layer_idx * hs;
        float gamma = SFA_EMA_GAMMA;
        for (int h = 0; h < hs; ++h) {
            field[h] = gamma * field[h] + (1.0f - gamma) * attn_out->data[h];
        }

        // 5. 计算 enhancement = ring_mean + (field + semantic) * 0.5
        float enhancement[hs];
        for (int h = 0; h < hs; ++h) {
            enhancement[h] = ring_mean[h] + 0.5f * (field[h] + seq_state.semantic_pool[h]);
        }

        // 6. Clip enhancement
        float clip_val = SFA_ENHANCEMENT_CLIP;
        for (int h = 0; h < hs; ++h) {
            if (enhancement[h] > clip_val) enhancement[h] = clip_val;
            if (enhancement[h] < -clip_val) enhancement[h] = -clip_val;
        }

        // 7. Scale by alpha
        float alpha_eff = sfa_context::layer_alpha(ctx.alpha_base, ctx.cross_decay, layer_idx, nl);
        for (int h = 0; h < hs; ++h) {
            enhancement[h] *= alpha_eff;
        }

        // 8. 创建 enhancement tensor 并 add to attn_out
        //    注意：这里需要在 ggml 图中构建，而不是直接操作数据
        //    简化版本：直接修改 attn_out 数据（适用于非图构建场景）
        for (int i = 0; i < ggml_nelements(attn_out); ++i) {
            int h = i % hs;
            attn_out->data[i] += enhancement[h];
        }

        return attn_out;
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
