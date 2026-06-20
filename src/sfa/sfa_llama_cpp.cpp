#include "sfa_llama_cpp.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// ─── Public API ──────────────────────────────────────────────────

void sfa_init(sfa_context *ctx, int n_layers, int hidden_size,
              float alpha_base, float cross_decay) {
    ctx->alpha_base = alpha_base;
    ctx->cross_decay = cross_decay;
    ctx->init(n_layers, hidden_size);
    ctx->enabled = true;
}

void sfa_enable(sfa_context *ctx, bool on) {
    ctx->enabled = on;
}

void sfa_reset(sfa_context *ctx, int /*seq_id*/) {
    ctx->reset_all();
}

void sfa_set_alpha(sfa_context *ctx, float alpha) {
    ctx->alpha_base = alpha;
}

const char* sfa_version() {
    return "1.0.0";
}

// ─── ggml Graph Building ─────────────────────────────────────────

namespace sfa {

// Helper: compute mean of ring buffer for a layer
// ring_buffer shape: [ring_size, hidden_size]
// Returns: [hidden_size] tensor (mean of valid entries)
static inline ggml_tensor * ring_buffer_mean(
    ggml_context *ctx0,
    ggml_tensor * ring_buffer,  // [ring_size, hidden_size]
    int ring_count,
    const ggml_cgraph *gf) {

    if (ring_count <= 0) {
        // Return zero tensor
        return ggml_zeros_in_place(ctx0, ring_buffer, GGML_TYPE_F32);
    }

    // Sum all valid ring entries along axis 0
    // ggml_sum_elements sums along the first dimension
    ggml_tensor * sum = ggml_sum_elements(ctx0, ring_buffer);
    return sum;
}

// Helper: EMA update for field state
// field_state shape: [hidden_size]
// new_val shape: [hidden_size]
// Returns: updated field_state = gamma * field_state + (1-gamma) * new_val
static inline ggml_tensor * ema_update(
    ggml_context *ctx0,
    ggml_tensor * field_state,
    ggml_tensor * new_val,
    float gamma,
    const ggml_cgraph *gf) {

    ggml_tensor * diff = ggml_sub(ctx0, new_val, field_state);
    ggml_tensor * scaled = ggml_scale(ctx0, diff, 1.0f - gamma);
    return ggml_add(ctx0, field_state, scaled);
}

// Helper: semantic pool attention (dot product between query and semantic slots)
// query: [hidden_size]
// semantic_pool: [slots, hidden_size]
// Returns: weighted sum of semantic tokens
static inline ggml_tensor * semantic_pool_attention(
    ggml_context *ctx0,
    ggml_tensor * query,      // [hidden_size]
    ggml_tensor * semantic_pool, // [slots, hidden_size]
    float temperature,
    const ggml_cgraph *gf) {

    // Compute dot products: query @ semantic_pool^T -> [slots]
    ggml_tensor * dots = ggml_mul_mat(ctx0, semantic_pool, query);

    // Softmax over slots
    dots = ggml_soft_max_ext(ctx0, dots, nullptr, temperature, 1.0f);

    // Weighted sum: semantic_pool^T @ dots -> [hidden_size]
    return ggml_mul_mat(ctx0, query, dots);
}

// Main SFA enhancement builder
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
    const ggml_cgraph *gf) {

    // 1. Extract current layer's ring buffer
    // ring_buffer shape: [n_layers, ring_size, hidden]
    // We need layer_idx slice: [ring_size, hidden]
    ggml_tensor * layer_ring = ggml_get_rows(ctx0, ring_buffer,
                                              ggml_new_i32(ctx0, layer_idx));

    // 2. Compute ring mean (approximation of recent attention output average)
    ggml_tensor * ring_mean = ring_buffer_mean(ctx0, layer_ring, ring_size / 2, gf);

    // 3. EMA field update (far-field memory)
    ggml_tensor * layer_field = ggml_get_rows(ctx0, field_state,
                                               ggml_new_i32(ctx0, layer_idx));
    ggml_tensor * field_update = ema_update(ctx0, layer_field, attn_out,
                                             SFA_EMA_GAMMA, gf);

    // 4. Semantic pool attention
    ggml_tensor * semantic_out = semantic_pool_attention(
        ctx0, attn_out, semantic_pool, 0.07f, gf);

    // 5. Gaussian compression (simple exponential smoothing)
    ggml_tensor * gauss_update = ema_update(ctx0, gaussian_comp, attn_out,
                                             SFA_GAUSS_GAMMA, gf);

    // 6. Combine: enhancement = ring_mean + (field_update + semantic_out) * 0.5
    ggml_tensor * combined = ggml_add(ctx0, field_update, semantic_out);
    combined = ggml_scale(ctx0, combined, 0.5f);
    ggml_tensor * enhancement = ggml_add(ctx0, ring_mean, combined);

    // 7. Compute effective alpha with cross-layer decay
    float alpha_eff = sfa_context::layer_alpha(alpha_base, cross_decay, layer_idx, n_layers);

    // 8. Clip enhancement to prevent instability
    enhancement = ggml_clamp(ctx0, enhancement, -SFA_ENHANCEMENT_CLIP, SFA_ENHANCEMENT_CLIP);

    // 9. Scale by alpha
    enhancement = ggml_scale(ctx0, enhancement, alpha_eff);

    // 10. Add to attention output (broadcast over seq_len)
    // attn_out: [batch, seq_len, hidden]
    // enhancement: [hidden]
    // ggml_add will broadcast automatically
    ggml_tensor * enhanced = ggml_add(ctx0, attn_out, enhancement);

    return enhanced;
}

} // namespace sfa
