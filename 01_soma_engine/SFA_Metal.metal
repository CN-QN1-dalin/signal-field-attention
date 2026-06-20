//
//  SFA_Metal.metal
//  Signal Field Attention - Metal GPU Kernels
//
//  Three GPU-accelerated kernels for SFA:
//  1. near_field_attn — Softmax attention on ring buffer (O(k·d))
//  2. ema_update — Exponential moving average field state (O(d))
//  3. dual_path_fusion — Near + Alpha*Far fusion (O(d))
//
//  Compile:
//    metal -c SFA_Metal.metal -o SFA_Metal.air
//    metallic -S SFA_Metal.air -o SFA_Metal.metallib
//
//  Or use the convenience script:
//    bash build_metal_lib.sh
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Configuration Constants (set per-model)
// ============================================================================

// Maximum ring buffer size for threadgroup allocation
#define MAX_RING_SIZE 256
#define MAX_HEAD_DIM  128

// ============================================================================
// Kernel 1: Near-Field Softmax Attention
//
// Computes softmax(Q · K_ring^T / sqrt(d)) · V_ring
// Thread-per-head-dim: each thread computes one output dimension
// Uses threadgroup memory for score accumulation to minimize global memory traffic.
// ============================================================================

kernel void near_field_attn(
    constant float*   q        [[buffer(0)]],   // [batch * num_heads, head_dim]
    constant float*   keys     [[buffer(1)]],   // [ring_size * num_heads, head_dim]
    constant float*   values   [[buffer(2)]],   // [ring_size * num_heads, head_dim]
    device float*     out      [[buffer(3)]],   // [batch * num_heads, head_dim]
    constant uint&    ring_size[[buffer(4)]],
    constant uint&    num_heads[[buffer(5)]],
    constant uint&    head_dim [[buffer(6)]],
    constant uint&    batch    [[buffer(7)]],
    uint              tid      [[thread_position_in_grid]],
    uint              gid      [[threadgroup_position_in_grid]])
{
    uint total_threads = batch * num_heads * head_dim;
    if (tid >= total_threads) return;
    
    // Decompose thread index: (batch_idx, head_idx, dim_idx)
    uint dim_idx = tid % head_dim;
    uint temp = tid / head_dim;
    uint head_idx = temp % num_heads;
    uint batch_idx = temp / num_heads;
    
    // Load Q for this head into threadgroup memory
    threadgroup float q_local[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d += simd_lane_id()) {
        q_local[d] = q[(batch_idx * num_heads + head_idx) * head_dim + d];
    }
    simd_barrier();
    
    // Compute scores: q · k_j for all j in ring
    // We use a shared reduction: each SIMD lane handles one ring slot
    float max_score = -FLT_MAX;
    threadgroup float scores[MAX_RING_SIZE];
    
    for (uint j = 0; j < ring_size; j++) {
        float s = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            s += q_local[d] * keys[(j * num_heads + head_idx) * head_dim + d];
        }
        s *= rsqrt(sqrtf(128.0f)); // scale = 1/sqrt(head_dim), default 128
        scores[j] = s;
        if (s > max_score) max_score = s;
    }
    simd_barrier();
    
    // Parallel max reduction within SIMD group
    for (uint stride = simd_group_size() / 2; stride > 0; stride /= 2) {
        float other = simd_shuffle_down(max_score, stride);
        if (other > max_score) max_score = other;
    }
    max_score = simd_shuffle(max_score, 0);
    
    // Compute exp scores with numerical stability
    float exp_sum = 0.0f;
    for (uint j = 0; j < ring_size; j++) {
        float e = exp(scores[j] - max_score);
        scores[j] = e; // reuse scores array for exp values
        exp_sum += e;
    }
    simd_barrier();
    
    // Reduction: sum all exp scores
    for (uint stride = simd_group_size() / 2; stride > 0; stride /= 2) {
        exp_sum += simd_shuffle_down(exp_sum, stride);
    }
    exp_sum = simd_shuffle(exp_sum, 0);
    
    // Compute output for this head-dimension
    float result = 0.0f;
    for (uint j = 0; j < ring_size; j++) {
        float w = scores[j] / (exp_sum + 1e-8f);
        result += w * values[(j * num_heads + head_idx) * head_dim + dim_idx];
    }
    
    // Write output
    out[tid] = result;
}

// ============================================================================
// Kernel 2: EMA Field State Update
//
// state_out[i] = gamma * state_in[i] + (1-gamma) * k_mean[i]
// One thread per dimension — trivially parallel, no synchronization needed.
// ============================================================================

kernel void ema_update(
    constant float*   k_mean     [[buffer(0)]],   // [num_heads * head_dim]
    constant float*   state_in   [[buffer(1)]],   // [num_heads * head_dim]
    device float*     state_out  [[buffer(2)]],   // [num_heads * head_dim]
    constant uint&    total_dim  [[buffer(3)]],
    constant float&   gamma      [[buffer(4)]],
    uint              tid        [[thread_position_in_grid]])
{
    if (tid >= total_dim) return;
    
    float one_minus_gamma = 1.0f - gamma;
    state_out[tid] = gamma * state_in[tid] + one_minus_gamma * k_mean[tid];
}

// ============================================================================
// Kernel 3: Dual-Path Fusion
//
// out[i] = near[i] + alpha * far[i]
// Simple element-wise — ideal for GPU parallelism.
// ============================================================================

kernel void dual_path_fusion(
    constant float*   near       [[buffer(0)]],   // [num_heads * head_dim]
    constant float*   far        [[buffer(1)]],   // [num_heads * head_dim]
    device float*     out        [[buffer(2)]],   // [num_heads * head_dim]
    constant uint&    total_dim  [[buffer(3)]],
    constant float&   alpha      [[buffer(4)]],
    uint              tid        [[thread_position_in_grid]])
{
    if (tid >= total_dim) return;
    out[tid] = near[tid] + alpha * far[tid];
}

// ============================================================================
// Kernel 4: QKV Projection (Full Prefill)
//
// qkv[t, h, d] = x[t, :] · W_qkv[h, d, :]
// Optimized for small batch (B=1) with sequential token access.
// ============================================================================

kernel void qkv_project(
    constant float*   x          [[buffer(0)]],   // [seq_len * dims]
    constant float*   W          [[buffer(1)]],   // [3 * dims * dims]
    device float*     q_out      [[buffer(2)]],   // [seq_len * num_heads * head_dim]
    device float*     k_out      [[buffer(3)]],   // [seq_len * num_heads * head_dim]
    device float*     v_out      [[buffer(4)]],   // [seq_len * num_heads * head_dim]
    constant uint&    seq_len    [[buffer(5)]],
    constant uint&    dims       [[buffer(6)]],
    constant uint&    num_heads  [[buffer(7)]],
    constant uint&    head_dim   [[buffer(8)]],
    uint              tid        [[thread_position_in_grid]])
{
    // Each thread handles one (token, head, dim) triple
    uint total = seq_len * num_heads * head_dim;
    if (tid >= total) return;
    
    uint token = tid / (num_heads * head_dim);
    uint rem = tid % (num_heads * head_dim);
    uint head = rem / head_dim;
    uint d = rem % head_dim;
    
    float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
    constant float* x_row = x + token * dims;
    
    for (uint wi = 0; wi < dims; wi++) {
        float xv = x_row[wi];
        q_val += xv * W[head * dims * head_dim + wi * head_dim + d];
        k_val += xv * W[dims * dims + head * dims * head_dim + wi * head_dim + d];
        v_val += xv * W[2 * dims * dims + head * dims * head_dim + wi * head_dim + d];
    }
    
    q_out[tid] = q_val;
    k_out[tid] = k_val;
    v_out[tid] = v_val;
}

// ============================================================================
// Kernel 5: Output Projection
//
// out[d] = sum_h,dh( attn[h,dh] * W_o[(h,dh),d] )
// ============================================================================

kernel void output_project(
    constant float*   attn       [[buffer(0)]],   // [num_heads * head_dim]
    constant float*   W          [[buffer(1)]],   // [num_heads * head_dim * dims]
    device float*     out        [[buffer(2)]],   // [dims]
    constant uint&    dims       [[buffer(3)]],
    constant uint&    num_heads  [[buffer(4)]],
    constant uint&    head_dim   [[buffer(5)]],
    uint              tid        [[thread_position_in_grid]])
{
    if (tid >= dims) return;
    
    float sum = 0.0f;
    for (uint h = 0; h < num_heads; h++) {
        for (uint hd = 0; hd < head_dim; hd++) {
            int32_t attn_idx = h * head_dim + hd;
            int32_t w_idx = (h * head_dim + hd) * dims + tid;
            sum += attn[attn_idx] * W[w_idx];
        }
    }
    out[tid] = sum;
}

// ============================================================================
// Kernel 6: Ring Buffer Write (Circular Append)
//
// Writes k/v for a single token into the ring buffer at the correct position.
// Used during decode step for O(1) ring update.
// ============================================================================

kernel void ring_write(
    device float*   ring        [[buffer(0)]],   // [k * num_heads * head_dim]
    constant float* data        [[buffer(1)]],   // [num_heads * head_dim]
    constant uint&  pos         [[buffer(2)]],
    constant uint&  k_size      [[buffer(3)]],
    constant uint&  total_dim   [[buffer(4)]],  // num_heads * head_dim
    uint            tid         [[thread_position_in_grid]])
{
    if (tid >= total_dim) return;
    
    uint write_idx = ((pos % k_size) * total_dim) + tid;
    ring[write_idx] = data[tid];
}
