// ============================================================
// 🎯 SFA v7 → llama.cpp 集成桥接器 - 重写版 (2026-06-23)
// ============================================================
// 架构设计：
//   - 主机端 (CPU/RAM): 维护 SFA 状态 (ring buffers, EMA field, semantic pool)
//   - 计算图端 (ggml): 每层 attention output 后构建 SFA 增强节点
//   - 数据流: attn_out → 提取 mean → ggml graph 计算 enhancement → 加回 attn_out
//
// 关键设计决策：
//   1. SFA 状态存储在 host memory (std::vector)，不通过 ggml graph 管理
//   2. ggml graph 只负责计算 enhancement = f(ring_mean, field, semantic)
//   3. 通过 ggml_view_2d/3d 创建 host data 的视图，避免数据复制
//   4. 使用 ggml_set_1d 等 API 设置标量常量（替代 ggml_new_i32）
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
    std::vector<int> ring_offsets;      // [n_layers]
    bool initialized;

    void init(int n_layers, int hidden_size, int semantic_slots) {
        this->ring_buffers.assign(n_layers * SFA_RING_SIZE * hidden_size, 0.0f);
        this->field_states.assign(n_layers * hidden_size, 0.0f);
        this->semantic_pool.assign(semantic_slots * hidden_size, 0.0f);
        this->ring_offsets.assign(n_layers, 0);
        this->initialized = true;
    }

    void reset() {
        std::memset(ring_buffers.data(), 0, ring_buffers.size() * sizeof(float));
        std::memset(field_states.data(), 0, field_states.size() * sizeof(float));
        std::memset(semantic_pool.data(), 0, semantic_pool.size() * sizeof(float));
        std::fill(ring_offsets.begin(), ring_offsets.end(), 0);
    }

    sfa_seq_state() : initialized(false) {}
};

// ============================================================
// 🔧 SFA Bridge 类
// ============================================================

class SFA_Llama_Bridge {
private:
    sfa_context ctx;
    std::unordered_map<int, sfa_seq_state> seq_map;
    std::mutex mutex;
    int hidden_size;
    int semantic_slots;
    bool model_loaded;

    // 创建一个 view tensor，指向 host memory 中的数据
    // 这是关键：ggml_view_2d 创建的是一个"视图"，data 指向已有的 buffer
    ggml_tensor* make_host_view_2d(
        ggml_context* ctx0,
        void* host_data,
        int d0,   // stride in elements (hidden_size)
        int d1)   // size in elements (ring_size or semantic_slots)
    {
        // ggml_view_2d: [d0, d1], stride0 = d0 * tsize
        size_t tsize = sizeof(float);
        ggml_tensor* t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d0, d1);
        // 将 tensor 的 data 指针指向 host buffer
        t->data = host_data;
        t->buffer = nullptr;  // 清除 buffer 引用，标记为 host-backed
        return t;
    }

public:
    SFA_Llama_Bridge() : hidden_size(0), semantic_slots(SFA_SEMANTIC_SLOTS), model_loaded(false) {}

    // llama.cpp 生命周期钩子 1: 模型加载时调用
    void on_model_load(struct llama_model *model) {
        std::lock_guard<std::mutex> lock(mutex);

        ctx.n_layers = llama_model_n_layer(model);
        ctx.hidden_size = llama_model_n_embd(model);
        ctx.alpha_base = SFA_ALPHA_BASE;
        ctx.cross_decay = SFA_CROSS_DECAY_DEFAULT;
        ctx.ring_size = SFA_RING_SIZE;
        ctx.semantic_slots = semantic_slots;

        ctx.init(ctx.n_layers, ctx.hidden_size);

        // 初始化默认序列 (seq_id=0)
        sfa_seq_state default_seq;
        default_seq.init(ctx.n_layers, ctx.hidden_size, semantic_slots);
        seq_map[0] = default_seq;

        hidden_size = ctx.hidden_size;
        model_loaded = true;
    }

    // llama.cpp 生命周期钩子 2: 新序列开始时调用
    void on_seq_start(int seq_id) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!model_loaded) return;

        auto it = seq_map.find(seq_id);
        if (it == seq_map.end()) {
            sfa_seq_state new_seq;
            new_seq.init(ctx.n_layers, ctx.hidden_size, semantic_slots);
            seq_map[seq_id] = new_seq;
        } else {
            it->second.reset();
        }
    }

    // llama.cpp 生命周期钩子 3: 序列复制时调用
    void on_seq_copy(int dst_seq_id, int src_seq_id) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!model_loaded) return;

        auto src_it = seq_map.find(src_seq_id);
        if (src_it != seq_map.end()) {
            seq_map[dst_seq_id] = src_it->second;
        }
    }

    // llama.cpp 生命周期钩子 4: 序列删除时调用
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

    // 构建 SFA 增强图节点
    ggml_tensor * build_sfa_enhance_for_layer(
        ggml_context *ctx0,
        ggml_tensor * attn_out,       // [batch, seq_len, hidden]
        int layer_idx,
        int seq_id,
        const ggml_cgraph *gf) {

        if (!ctx.enabled) {
            return attn_out;
        }

        auto seq_it = seq_map.find(seq_id);
        if (seq_it == seq_map.end()) {
            return attn_out;
        }

        auto &seq_state = seq_it->second;
        int hs = ctx.hidden_size;
        int nl = ctx.n_layers;

        // ===== 通道1: RingBuffer Mean =====
        float* ring_buf = seq_state.ring_buffers.data() + layer_idx * SFA_RING_SIZE * hs;
        // 创建 host-backed view tensor: [hs, ring_size]
        ggml_tensor* ring_tensor = make_host_view_2d(ctx0, ring_buf, hs, SFA_RING_SIZE);
        ggml_tensor* ring_mean = ggml_mean(ctx0, ring_tensor);  // [hs]

        // ===== 通道2: EMA Field Update =====
        float* field = seq_state.field_states.data() + layer_idx * hs;
        ggml_tensor* field_tensor = make_host_view_1d(ctx0, field, hs);

        // attn_mean: 从 attn_out 计算 mean over [batch, seq] → [hidden]
        ggml_tensor* attn_mean = ggml_mean(ctx0, attn_out);  // [hidden]

        // new_field = gamma * field + (1-gamma) * attn_mean
        ggml_tensor* scaled_field = ggml_scale(ctx0, field_tensor, SFA_EMA_GAMMA);
        ggml_tensor* scaled_attn = ggml_scale(ctx0, attn_mean, 1.0f - SFA_EMA_GAMMA);
        ggml_tensor* new_field = ggml_add(ctx0, scaled_field, scaled_attn);

        // 更新 host 中的 field state（用于下一轮）
        std::memcpy(field, new_field->data, hs * sizeof(float));

        // ===== 通道3: Semantic Pool =====
        int n_semantic_slots = static_cast<int>(seq_state.semantic_pool.size()) / hs;
        float* sem_pool = seq_state.semantic_pool.data();
        ggml_tensor* semantic_tensor = make_host_view_2d(ctx0, sem_pool, hs, n_semantic_slots);
        ggml_tensor* semantic_mean = ggml_mean(ctx0, semantic_tensor);  // [hidden]

        // ===== 融合: enhancement = ring_mean + 0.5 * (new_field + semantic_mean) =====
        ggml_tensor* field_semantic_sum = ggml_add(ctx0, new_field, semantic_mean);
        ggml_tensor* scaled_sum = ggml_scale(ctx0, field_semantic_sum, 0.5f);
        ggml_tensor* enhancement = ggml_add(ctx0, ring_mean, scaled_sum);

        // ===== Clip =====
        enhancement = ggml_clamp(ctx0, enhancement, -SFA_ENHANCEMENT_CLIP, SFA_ENHANCEMENT_CLIP);

        // ===== Scale by alpha =====
        float alpha_eff = sfa_context::layer_alpha(ctx.alpha_base, ctx.cross_decay, layer_idx, nl);
        enhancement = ggml_scale(ctx0, enhancement, alpha_eff);

        // ===== Add to attention output =====
        ggml_tensor* enhanced = ggml_add(ctx0, attn_out, enhancement);

        return enhanced;
    }

    // 辅助函数：创建 1D host view tensor
    ggml_tensor* make_host_view_1d(ggml_context* ctx0, void* host_data, int d0) {
        ggml_tensor* t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d0);
        t->data = host_data;
        t->buffer = nullptr;
        return t;
    }

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

void sfa_llama_init(struct llama_model *model) {
    g_sfa_bridge.on_model_load(model);
}

void sfa_llama_seq_start(int seq_id) {
    g_sfa_bridge.on_seq_start(seq_id);
}

void sfa_llama_seq_copy(int dst_seq_id, int src_seq_id) {
    g_sfa_bridge.on_seq_copy(dst_seq_id, src_seq_id);
}

void sfa_llama_seq_remove(int seq_id) {
    g_sfa_bridge.on_seq_remove(seq_id);
}

void sfa_llama_free() {
    g_sfa_bridge.on_ctx_free();
}

ggml_tensor * sfa_llama_build_enhance(
    ggml_context *ctx0,
    ggml_tensor * attn_out,
    int layer_idx,
    int seq_id,
    const ggml_cgraph *gf) {

    return g_sfa_bridge.build_sfa_enhance_for_layer(
        ctx0, attn_out, layer_idx, seq_id, gf);
}

void sfa_llama_enable(bool on) {
    g_sfa_bridge.get_context()->enabled = on;
}

void sfa_llama_set_alpha(float alpha) {
    g_sfa_bridge.get_context()->alpha_base = alpha;
}

} // extern "C"
