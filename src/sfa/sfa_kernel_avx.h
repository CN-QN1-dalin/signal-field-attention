#pragma once

// ==========================================
// ⚡ SFA高性能内核 - SIMD优化版
// ==========================================
// 目标：利用Apple Silicon NEON指令集加速向量运算
// ==========================================

#include <cstring>
#include <cassert>

#ifdef __APPLE__
#include <arm_neon.h>
#endif

namespace sfa {

// NEON优化的向量加法
inline void neon_vec_add(const float* a, const float* b, float* out, int n) {
    int i = 0;
#ifdef __ARM_FEATURE_NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vo = vaddq_f32(va, vb);
        vst1q_f32(out + i, vo);
    }
#endif
    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

// NEON优化的向量缩放
inline void neon_vec_scale(const float* a, float scale, float* out, int n) {
    int i = 0;
#ifdef __ARM_FEATURE_NEON
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vo = vmulq_f32(va, vscale);
        vst1q_f32(out + i, vo);
    }
#endif
    for (; i < n; i++) {
        out[i] = a[i] * scale;
    }
}

// NEON优化的向量clamp
inline void neon_vec_clamp(float* vec, float min_val, float max_val, int n) {
    int i = 0;
#ifdef __ARM_FEATURE_NEON
    float32x4_t vmin = vdupq_n_f32(min_val);
    float32x4_t vmax = vdupq_f32(max_val);
    for (; i + 3 < n; i += 4) {
        float32x4_t vv = vld1q_f32(vec + i);
        float32x4_t vminned = vmaxq_f32(vv, vmin);
        float32x4_t vmaxned = vminq_f32(vminned, vmax);
        vst1q_f32(vec + i, vmaxned);
    }
#endif
    for (; i < n; i++) {
        if (vec[i] < min_val) vec[i] = min_val;
        if (vec[i] > max_val) vec[i] = max_val;
    }
}

// NEON优化的EMA更新
inline void neon_ema_update(float* state, const float* input, float gamma, int n) {
    float one_minus_gamma = 1.0f - gamma;
    int i = 0;
#ifdef __ARM_FEATURE_NEON
    float32x4_t vgamma = vdupq_n_f32(gamma);
    float32x4_t vinv_gamma = vdupq_n_f32(one_minus_gamma);
    for (; i + 3 < n; i += 4) {
        float32x4_t vs = vld1q_f32(state + i);
        float32x4_t vi = vld1q_f32(input + i);
        float32x4_t vo = vaddq_f32(
            vmulq_f32(vgamma, vs),
            vmulq_f32(vinv_gamma, vi)
        );
        vst1q_f32(state + i, vo);
    }
#endif
    for (; i < n; i++) {
        state[i] = gamma * state[i] + one_minus_gamma * input[i];
    }
}

// 向量化ring buffer均值计算
inline void neon_ring_mean(const float* ring, int ring_size, int hidden_size, float* out) {
    std::memset(out, 0, hidden_size * sizeof(float));
    
    int i = 0;
#ifdef __ARM_FEATURE_NEON
    for (int r = 0; r < ring_size; r++) {
        for (; i + 3 < hidden_size; i += 4) {
            float32x4_t vout = vld1q_f32(out + i);
            float32x4_t vring = vld1q_f32(ring + r * hidden_size + i);
            vst1q_f32(out + i, vaddq_f32(vout, vring));
        }
    }
    // 归一化
    float inv_size = 1.0f / ring_size;
    for (; i < hidden_size; i++) {
        out[i] *= inv_size;
    }
#else
    for (int r = 0; r < ring_size; r++) {
        for (i = 0; i < hidden_size; i++) {
            out[i] += ring[r * hidden_size + i];
        }
    }
    float inv_size = 1.0f / ring_size;
    for (i = 0; i < hidden_size; i++) {
        out[i] *= inv_size;
    }
#endif
}

} // namespace sfa
