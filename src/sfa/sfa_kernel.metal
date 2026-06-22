// ============================================================
// ⚡ SFA Metal GPU 内核实现
// ============================================================
// 目标：利用 Apple Silicon GPU 加速 SFA 向量运算
// 状态：初始版本 (2026-06-22)
// ============================================================

#include <metal_stdlib>
using namespace metal;

// SFA 配置常量
constant constexpr int SFA_RING_SIZE = 16;
constant constexpr int SFA_SEMANTIC_SLOTS = 64;
constant constexpr float SFA_EMA_GAMMA = 0.98f;
constant constexpr float SFA_ENHANCEMENT_CLIP = 0.01f;

// NEON 优化的向量加法 (Metal 等效)
kernel void neon_vec_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// NEON 优化的向量缩放
kernel void neon_vec_scale(
    device const float* a [[buffer(0)]],
    device const float* out [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    
    if (tid < n) {
        out[tid] = a[tid] * scale;
    }
}

// NEON 优化的向量 clamp
kernel void neon_vec_clamp(
    device float* vec [[buffer(0)]],
    constant float& min_val [[buffer(1)]],
    constant float& max_val [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    
    if (tid < n) {
        vec[tid] = max(min_val, min(vec[tid], max_val));
    }
}

// NEON 优化的 EMA 更新
kernel void neon_ema_update(
    device float* state [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant float& gamma [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    
    if (tid < n) {
        state[tid] = gamma * state[tid] + (1.0f - gamma) * input[tid];
    }
}

// 向量化 ring buffer 均值计算
kernel void neon_ring_mean(
    device const float* ring [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& ring_size [[buffer(2)]],
    constant int& hidden_size [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    
    int h = tid.x;
    if (h < hidden_size) {
        float sum = 0.0f;
        for (int r = 0; r < ring_size; r++) {
            sum += ring[r * hidden_size + h];
        }
        out[h] = sum / ring_size;
    }
}

// SFA 增强计算主 kernel
kernel void sfa_enhance_compute(
    device const float* attn_out [[buffer(0)]],
    device const float* ring_buf [[buffer(1)]],
    device const float* field_state [[buffer(2)]],
    device const float* semantic_pool [[buffer(3)]],
    device float* enhancement [[buffer(4)]],
    constant int& hidden_size [[buffer(5)]],
    constant int& layer_idx [[buffer(6)]],
    constant float& alpha_eff [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
    
    if (tid < hidden_size) {
        // 1. 计算 ring mean
        float ring_mean = 0.0f;
        for (int r = 0; r < SFA_RING_SIZE; r++) {
            ring_mean += ring_buf[(layer_idx * SFA_RING_SIZE + r) * hidden_size + tid];
        }
        ring_mean /= SFA_RING_SIZE;
        
        // 2. EMA field update
        float field = field_state[layer_idx * hidden_size + tid];
        field = SFA_EMA_GAMMA * field + (1.0f - SFA_EMA_GAMMA) * attn_out[tid];
        
        // 3. Semantic attention (简化版)
        float semantic = semantic_pool[layer_idx * hidden_size + tid];
        
        // 4. 计算 enhancement
        float enh = ring_mean + 0.5f * (field + semantic);
        
        // 5. Clip and scale
        enh = max(-SFA_ENHANCEMENT_CLIP, min(SFA_ENHANCEMENT_CLIP, enh));
        enh *= alpha_eff;
        
        enhancement[tid] = enh;
    }
}
