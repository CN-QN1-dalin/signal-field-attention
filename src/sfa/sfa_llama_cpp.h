#pragma once

// SFA (Signal Field Attention) integration for llama.cpp
// Model-layer implementation — NOT custom ggml operators
// Three signal channels: Ring Buffer, EMA Field, Semantic Pool

#include "ggml.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-graph.h"

#include <cstring>
#include <cmath>

// ─── Configuration ───────────────────────────────────────────────

#define SFA_RING_SIZE 16
#define SFA_SEMANTIC_SLOTS 64
#define SFA_EMA_GAMMA 0.98f
#define SFA_GAUSS_GAMMA 0.951229f  // exp(-0.05)
#define SFA_CROSS_DECAY_DEFAULT 0.7f
#define SFA_ENHANCEMENT_CLIP 0.01f

// ─── SFA State per layer ─────────────────────────────────────────

struct sfa_layer_state {
    // Near-field: Ring buffer of recent attention outputs
    float ring_buffer[SFA_RING_SIZE];  // flattened [k][hidden_size]
    int ring_pos;
    int ring_count;

    // Far-field: EMA of hidden state means
    float field_state;  // scalar EMA accumulator (per-head, aggregated)

    // Semantic memory pool
    float semantic_tokens[SFA_SEMANTIC_SLOTS];  // flattened [m][hidden_size]
    float semantic_conf[SFA_SEMANTIC_SLOTS];
    float semantic_age;

    // Gaussian compression
    float gaussian_comp;

    // Per-layer alpha (computed from alpha_base * decay^layer_idx)
    float alpha_effective;

    void reset(int hidden_size) {
        std::memset(ring_buffer, 0, sizeof(ring_buffer));
        ring_pos = 0;
        ring_count = 0;
        field_state = 0.0f;
        std::memset(semantic_tokens, 0, sizeof(semantic_tokens));
        std::memset(semantic_conf, 0, sizeof(semantic_conf));
        semantic_age = 0.0f;
        gaussian_comp = 0.0f;
        alpha_effective = 0.0f;
    }
};

// ─── SFA Global Context ──────────────────────────────────────────

struct sfa_context {
    bool enabled;
    float alpha_base;
    float cross_decay;
    int ring_size;
    int n_layers;
    int hidden_size;
    int semantic_slots;

    // Per-layer states (allocated dynamically)
    sfa_layer_state * layer_states;

    sfa_context()
        : enabled(false), alpha_base(2.0f), cross_decay(SFA_CROSS_DECAY_DEFAULT),
          ring_size(SFA_RING_SIZE), n_layers(0), hidden_size(0),
          semantic_slots(SFA_SEMANTIC_SLOTS), layer_states(nullptr) {}

    ~sfa_context() {
        delete[] layer_states;
        layer_states = nullptr;
    }

    void init(int n_layers_, int hidden_size_) {
        n_layers = n_layers_;
        hidden_size = hidden_size_;
        layer_states = new sfa_layer_state[n_layers];
        for (int i = 0; i < n_layers; i++) {
            layer_states[i].reset(hidden_size_);
        }
    }

    void reset_all() {
        for (int i = 0; i < n_layers; i++) {
            layer_states[i].reset(hidden_size);
        }
    }

    sfa_layer_state& layer(int idx) {
        return layer_states[idx];
    }

    static float layer_alpha(float alpha_base, float cross_decay, int lidx, int n_layers) {
        float ratio = static_cast<float>(lidx) / std::max(n_layers - 1, 1);
        float alpha = alpha_base * (0.3f + ratio * 0.7f);
        return alpha * std::pow(cross_decay, lidx);
    }
};

// ─── Public API ──────────────────────────────────────────────────

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize SFA context for a model.
 * Must be called before building the graph.
 */
void sfa_init(sfa_context *ctx, int n_layers, int hidden_size,
              float alpha_base = 2.0f, float cross_decay = SFA_CROSS_DECAY_DEFAULT);

/**
 * Enable/disable SFA enhancement.
 */
void sfa_enable(sfa_context *ctx, bool on);

/**
 * Reset SFA state for a new sequence.
 */
void sfa_reset(sfa_context *ctx, int seq_id = 0);

/**
 * Set alpha base at runtime.
 */
void sfa_set_alpha(sfa_context *ctx, float alpha);

/**
 * Get SFA version string.
 */
const char* sfa_version();

#ifdef __cplusplus
}
#endif

// ─── Internal: ggml graph building helpers ───────────────────────

namespace sfa {

/**
 * Build SFA enhancement as a ggml_add node.
 *
 * The enhancement is computed as:
 *   enhancement = ring_mean + (semantic_pool + gaussian_comp) * 0.5
 * Then applied to attention output:
 *   output[:, :seq_len, :] += alpha_effective * enhancement
 *
 * This is inserted AFTER the standard attention output and BEFORE
 * the residual connection (ggml_add to inpSA).
 */
ggml_tensor * build_sfa_enhance(
    ggml_context *ctx0,
    ggml_tensor * attn_out,       // [batch, seq_len, hidden]
    ggml_tensor * ring_buffer,    // [n_layers, ring_size, hidden]
    ggml_tensor * field_state,    // [n_layers, hidden]
    ggml_tensor * semantic_pool,  // [slots, hidden]
    ggml_tensor * gaussian_comp,  // [hidden]
    float alpha_base,
    float cross_decay,
    int layer_idx,
    int n_layers,
    int hidden_size,
    int ring_size,
    int semantic_slots,
    const ggml_cgraph *gf = nullptr
);

} // namespace sfa
