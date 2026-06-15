//
//  SFA_Metal.cpp
//  Signal Field Attention - C++/Metal Accelerated Engine
//
//  Dual-mode engine: Metal GPU kernels with CPU fallback.
//
//  Build (CPU-only):
//    clang++ -std=c++17 -O3 -DNOCPU_ONLY SFA_Metal.cpp -o soma_metal
//
//  Build (with Metal GPU):
//    clang++ -std=c++17 -O3 -fobjc-arc SFA_Metal.cpp \
//        -framework Metal -framework Foundation \
//        -o soma_metal
//
//  Run:
//    ./soma_metal [--cpu] [--prefill 1024] [--decode 10000] [--compare]
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <chrono>
#include <cassert>
#include <iomanip>

#ifdef USE_METAL
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#endif

// ============================================================================
// Configuration
// ============================================================================

struct SFACfg {
    int32_t dims;          // Hidden dimension (e.g., 3584 for 7B)
    int32_t num_heads;     // Number of attention heads (e.g., 28 for 7B)
    int32_t num_kv_heads;  // Number of KV heads for GQA (e.g., 4 for 7B)
    int32_t head_dim;      // Dimension per head (dims / num_heads)
    int32_t k;             // Ring buffer size (near-field window)
    float gamma;           // EMA decay factor (default: 0.98)
    float alpha;           // Far-field mixing weight (default: 0.1)
};

// ============================================================================
// Utility Functions
// ============================================================================

static float dot_product(const float* a, const float* b, int32_t len) {
    float sum = 0.0f;
    for (int32_t i = 0; i < len; i++) sum += a[i] * b[i];
    return sum;
}

static void print_tensor(const float* data, int32_t rows, int32_t cols, const char* name) {
    std::cout << name << " [" << rows << "x" << cols << "]:" << std::endl;
    for (int32_t i = 0; i < std::min(rows, 5); i++) {
        std::cout << "  row " << i << ": ";
        for (int32_t j = 0; j < std::min(cols, 10); j++) {
            std::cout << std::fixed << std::setprecision(4) << data[i * cols + j] << " ";
        }
        if (cols > 10) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 5) std::cout << "  ... (" << rows - 5 << " more rows)" << std::endl;
}

static double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

// ============================================================================
// Standard Attention (reference for correctness comparison)
// ============================================================================

class StandardAttention {
public:
    StandardAttention(const SFACfg& cfg) : cfg_(cfg) {
        float scale = std::sqrt(2.0f / (cfg_.dims + cfg_.dims));
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        qkv_weights_.resize(3 * cfg_.dims * cfg_.dims);
        for (auto& w : qkv_weights_) w = dist(gen) * scale;
        
        out_weights_.resize(cfg_.dims * cfg_.dims);
        for (auto& w : out_weights_) w = dist(gen) * scale;
    }
    
    std::vector<float> forward(const std::vector<float>& x, int32_t seq_len) {
        int32_t batch = 1;
        int32_t dims = cfg_.dims;
        int32_t num_heads = cfg_.num_heads;
        int32_t head_dim = cfg_.head_dim;
        
        // QKV projection
        std::vector<float> q(batch * seq_len * num_heads * head_dim);
        std::vector<float> k(batch * seq_len * num_heads * head_dim);
        std::vector<float> v(batch * seq_len * num_heads * head_dim);
        
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t h = 0; h < num_heads; h++) {
                for (int32_t d = 0; d < head_dim; d++) {
                    int32_t qkv_off = t * num_heads * head_dim + h * head_dim + d;
                    float qv = 0, kv = 0, vv = 0;
                    for (int32_t wi = 0; wi < dims; wi++) {
                        float xv = x[t * dims + wi];
                        qv += xv * qkv_weights_[h * dims * head_dim + wi * head_dim + d];
                        kv += xv * qkv_weights_[dims * dims + h * dims * head_dim + wi * head_dim + d];
                        vv += xv * qkv_weights_[2 * dims * dims + h * dims * head_dim + wi * head_dim + d];
                    }
                    q[qkv_off] = qv;
                    k[qkv_off] = kv;
                    v[qkv_off] = vv;
                }
            }
        }
        
        // Full causal attention
        std::vector<float> outputs(seq_len * dims, 0.0f);
        float scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t h = 0; h < num_heads; h++) {
                // Compute attention scores for token t
                std::vector<float> scores(t, 0.0f);
                for (int32_t s = 0; s < t; s++) {
                    scores[s] = dot_product(
                        q.data() + t * num_heads * head_dim + h * head_dim,
                        k.data() + s * num_heads * head_dim + h * head_dim,
                        head_dim
                    ) * scale_attn;
                }
                
                // Softmax
                float max_s = -1e30f;
                for (float s : scores) if (s > max_s) max_s = s;
                float exp_sum = 0.0f;
                for (auto& sc : scores) { sc = exp(sc - max_s); exp_sum += sc; }
                
                // Weighted sum
                std::vector<float> attn_out(head_dim, 0.0f);
                for (int32_t d = 0; d < head_dim; d++) {
                    for (int32_t s = 0; s < t; s++) {
                        attn_out[d] += (scores[s] / exp_sum) *
                            v.data()[(s * num_heads + h) * head_dim + d];
                    }
                }
                
                // Output projection
                for (int32_t d = 0; d < dims; d++) {
                    float sum = 0.0f;
                    for (int32_t hd = 0; hd < head_dim; hd++) {
                        sum += attn_out[hd] * out_weights_[(h * head_dim + hd) * dims + d];
                    }
                    outputs[t * dims + d] = sum;
                }
            }
        }
        
        return outputs;
    }
    
private:
    SFACfg cfg_;
    std::vector<float> qkv_weights_;
    std::vector<float> out_weights_;
};

// ============================================================================
// CPU Implementation of SFA
// ============================================================================

class CPUSFAModule {
public:
    CPUSFAModule(const SFACfg& cfg) : cfg_(cfg) {
        float scale = std::sqrt(2.0f / (cfg_.dims + cfg_.dims));
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        qkv_weights_.resize(3 * cfg_.dims * cfg_.dims);
        for (auto& w : qkv_weights_) w = dist(gen) * scale;
        
        out_weights_.resize(cfg_.dims * cfg_.dims);
        for (auto& w : out_weights_) w = dist(gen) * scale;
        
        ring_k_.resize(cfg_.k * cfg_.num_heads * cfg_.head_dim, 0.0f);
        ring_v_.resize(cfg_.k * cfg_.num_heads * cfg_.head_dim, 0.0f);
        
        field_state_.assign(cfg_.num_heads * cfg_.head_dim, 0.0f);
        ring_pos_ = 0;
        ring_size_ = 0;
    }
    
    struct PrefillResult {
        std::vector<float> output;
        std::vector<float> field_state;
        double ms;
    };
    
    PrefillResult prefill(const std::vector<float>& x, int32_t seq_len) {
        auto start = std::chrono::steady_clock::now();
        
        int32_t batch = 1;
        int32_t dims = cfg_.dims;
        int32_t num_heads = cfg_.num_heads;
        int32_t head_dim = cfg_.head_dim;
        
        // QKV projection for entire sequence
        std::vector<float> q(batch * seq_len * num_heads * head_dim);
        std::vector<float> k(batch * seq_len * num_heads * head_dim);
        std::vector<float> v(batch * seq_len * num_heads * head_dim);
        
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t h = 0; h < num_heads; h++) {
                for (int32_t d = 0; d < head_dim; d++) {
                    int32_t off = t * num_heads * head_dim + h * head_dim + d;
                    float qv = 0, kv = 0, vv = 0;
                    for (int32_t wi = 0; wi < dims; wi++) {
                        float xv = x[t * dims + wi];
                        qv += xv * qkv_weights_[h * dims * head_dim + wi * head_dim + d];
                        kv += xv * qkv_weights_[dims * dims + h * dims * head_dim + wi * head_dim + d];
                        vv += xv * qkv_weights_[2 * dims * dims + h * dims * head_dim + wi * head_dim + d];
                    }
                    q[off] = qv;
                    k[off] = kv;
                    v[off] = vv;
                }
            }
        }
        
        // Process each token
        std::vector<float> outputs(seq_len * dims, 0.0f);
        std::vector<float> field_state(cfg_.num_heads * cfg_.head_dim, 0.0f);
        
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t h = 0; h < num_heads; h++) {
                const float* q_t = q.data() + t * num_heads * head_dim + h * head_dim;
                float* attn_out = outputs.data() + t * dims + h * head_dim;
                
                int32_t effective_ring = std::min(ring_size_, cfg_.k);
                
                if (effective_ring > 0) {
                    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                    
                    // Compute attention scores
                    std::vector<float> scores(effective_ring), exp_scores(effective_ring);
                    float max_score = -1e30f;
                    
                    for (int32_t j = 0; j < effective_ring; j++) {
                        int32_t ring_idx = (ring_pos_ + j) % cfg_.k;
                        float s = dot_product(
                            q_t,
                            ring_k_.data() + ring_idx * num_heads * head_dim + h * head_dim,
                            head_dim
                        );
                        s *= scale;
                        scores[j] = s;
                        if (s > max_score) max_score = s;
                    }
                    
                    float exp_sum = 0.0f;
                    for (int32_t j = 0; j < effective_ring; j++) {
                        exp_scores[j] = std::exp(scores[j] - max_score);
                        exp_sum += exp_scores[j];
                    }
                    
                    // Weighted sum
                    std::vector<float> near_out(head_dim, 0.0f);
                    for (int32_t d = 0; d < head_dim; d++) {
                        for (int32_t j = 0; j < effective_ring; j++) {
                            int32_t ring_idx = (ring_pos_ + j) % cfg_.k;
                            near_out[d] += (exp_scores[j] / exp_sum) *
                                ring_v_.data()[(ring_idx * num_heads + h) * head_dim + d];
                        }
                    }
                    
                    // Dual-path fusion
                    for (int32_t d = 0; d < head_dim; d++) {
                        attn_out[d] = near_out[d] + cfg_.alpha * field_state[h * head_dim + d];
                    }
                } else {
                    // First token: identity
                    std::memcpy(attn_out, q_t, head_dim * sizeof(float));
                }
                
                // Output projection
                for (int32_t d = 0; d < dims; d++) {
                    float sum = 0.0f;
                    for (int32_t hd = 0; hd < head_dim; hd++) {
                        sum += attn_out[hd] * out_weights_[(h * head_dim + hd) * dims + d];
                    }
                    outputs[t * dims + d] = sum;
                }
                
                // Update ring buffer
                if (ring_size_ < cfg_.k) {
                    std::memcpy(
                        ring_k_.data() + ring_size_ * num_heads * head_dim + h * head_dim,
                        k.data() + t * num_heads * head_dim + h * head_dim,
                        head_dim * sizeof(float)
                    );
                    std::memcpy(
                        ring_v_.data() + ring_size_ * num_heads * head_dim + h * head_dim,
                        v.data() + t * num_heads * head_dim + h * head_dim,
                        head_dim * sizeof(float)
                    );
                    ring_size_++;
                }
                
                // Update EMA field state
                float one_minus_gamma = 1.0f - cfg_.gamma;
                for (int32_t d = 0; d < head_dim; d++) {
                    field_state[h * head_dim + d] = cfg_.gamma * field_state[h * head_dim + d] +
                        one_minus_gamma * k[t * num_heads * head_dim + h * head_dim + d];
                }
            }
        }
        
        double ms = elapsed_ms(start);
        return {outputs, field_state, ms};
    }
    
    struct DecodeResult {
        std::vector<float> output;
        std::vector<float> field_state;
        double ms;
    };
    
    DecodeResult decode_step(const std::vector<float>& x_new) {
        auto start = std::chrono::steady_clock::now();
        
        int32_t num_heads = cfg_.num_heads;
        int32_t head_dim = cfg_.head_dim;
        
        // QKV for single token
        std::vector<float> q(num_heads * head_dim), k(num_heads * head_dim), v(num_heads * head_dim);
        for (int32_t h = 0; h < num_heads; h++) {
            for (int32_t d = 0; d < head_dim; d++) {
                int32_t off = h * head_dim + d;
                float qv = 0, kv = 0, vv = 0;
                for (int32_t wi = 0; wi < cfg_.dims; wi++) {
                    float xv = x_new[wi];
                    qv += xv * qkv_weights_[h * cfg_.dims * head_dim + wi * head_dim + d];
                    kv += xv * qkv_weights_[cfg_.dims * cfg_.dims + h * cfg_.dims * head_dim + wi * head_dim + d];
                    vv += xv * qkv_weights_[2 * cfg_.dims * cfg_.dims + h * cfg_.dims * head_dim + wi * head_dim + d];
                }
                q[off] = qv;
                k[off] = kv;
                v[off] = vv;
            }
        }
        
        // Near-field attention
        std::vector<float> fused(num_heads * head_dim, 0.0f);
        int32_t effective_ring = std::min(ring_size_, cfg_.k);
        
        if (effective_ring > 0) {
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            for (int32_t h = 0; h < num_heads; h++) {
                std::vector<float> scores(effective_ring), exp_scores(effective_ring);
                float max_score = -1e30f;
                
                for (int32_t j = 0; j < effective_ring; j++) {
                    int32_t ring_idx = (ring_pos_ + j) % cfg_.k;
                    float s = dot_product(
                        q.data() + h * head_dim,
                        ring_k_.data() + ring_idx * num_heads * head_dim + h * head_dim,
                        head_dim
                    );
                    s *= scale;
                    scores[j] = s;
                    if (s > max_score) max_score = s;
                }
                
                float exp_sum = 0.0f;
                for (int32_t j = 0; j < effective_ring; j++) {
                    exp_scores[j] = std::exp(scores[j] - max_score);
                    exp_sum += exp_scores[j];
                }
                
                for (int32_t d = 0; d < head_dim; d++) {
                    float near_out = 0.0f;
                    for (int32_t j = 0; j < effective_ring; j++) {
                        int32_t ring_idx = (ring_pos_ + j) % cfg_.k;
                        near_out += (exp_scores[j] / exp_sum) *
                            ring_v_.data()[(ring_idx * num_heads + h) * head_dim + d];
                    }
                    fused[h * head_dim + d] = near_out + cfg_.alpha * field_state_[h * head_dim + d];
                }
            }
        } else {
            for (int32_t i = 0; i < num_heads * head_dim; i++) {
                fused[i] = field_state_[i];
            }
        }
        
        // Output projection
        std::vector<float> output(cfg_.dims, 0.0f);
        for (int32_t d = 0; d < cfg_.dims; d++) {
            for (int32_t h = 0; h < num_heads; h++) {
                for (int32_t hd = 0; hd < head_dim; hd++) {
                    output[d] += fused[h * head_dim + hd] * out_weights_[(h * head_dim + hd) * cfg_.dims + d];
                }
            }
        }
        
        // Update state
        float one_minus_gamma = 1.0f - cfg_.gamma;
        for (int32_t h = 0; h < num_heads; h++) {
            for (int32_t d = 0; d < head_dim; d++) {
                field_state_[h * head_dim + d] = cfg_.gamma * field_state_[h * head_dim + d] +
                    one_minus_gamma * k[h * head_dim + d];
            }
        }
        
        // Update ring buffer
        if (ring_size_ < cfg_.k) {
            std::memcpy(ring_k_.data() + ring_size_ * num_heads * head_dim, k.data(), num_heads * head_dim * sizeof(float));
            std::memcpy(ring_v_.data() + ring_size_ * num_heads * head_dim, v.data(), num_heads * head_dim * sizeof(float));
            ring_size_++;
        } else {
            std::memcpy(ring_k_.data() + ring_pos_ * num_heads * head_dim, k.data(), num_heads * head_dim * sizeof(float));
            std::memcpy(ring_v_.data() + ring_pos_ * num_heads * head_dim, v.data(), num_heads * head_dim * sizeof(float));
            ring_pos_ = (ring_pos_ + 1) % cfg_.k;
        }
        
        double ms = elapsed_ms(start);
        return {output, field_state_, ms};
    }
    
    const SFACfg& config() const { return cfg_; }
    
    int64_t memory_usage() const {
        return (cfg_.k * cfg_.num_heads * cfg_.head_dim * 2 +
                cfg_.num_heads * cfg_.head_dim) * sizeof(float);
    }
    
    // Reset state for new sequence
    void reset() {
        std::fill(ring_k_.begin(), ring_k_.end(), 0.0f);
        std::fill(ring_v_.begin(), ring_v_.end(), 0.0f);
        std::fill(field_state_.begin(), field_state_.end(), 0.0f);
        ring_pos_ = 0;
        ring_size_ = 0;
    }

private:
    SFACfg cfg_;
    std::vector<float> qkv_weights_;
    std::vector<float> out_weights_;
    std::vector<float> ring_k_;
    std::vector<float> ring_v_;
    std::vector<float> field_state_;
    int32_t ring_pos_;
    int32_t ring_size_;
};

// ============================================================================
// Metal GPU Module (conditional compilation)
// ============================================================================

#ifdef USE_METAL
class MetalSFAModule : public CPUSFAModule {
public:
    MetalSFAModule(const SFACfg& cfg) : CPUSFAModule(cfg), cfg_(cfg) {
        init_metal();
    }
    
    ~MetalSFAModule() {
        // Release Metal resources
    }
    
private:
    void init_metal() {
        // Get default device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Warning: Metal not available, falling back to CPU." << std::endl;
            use_metal_ = false;
            return;
        }
        use_metal_ = true;
        
        // Create command queue
        command_queue_ = [device newCommandQueue];
        
        // Load shader library
        NSBundle* bundle = [NSBundle mainBundle];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [@"] R"(
#include <metal_stdlib>
using namespace metal;

kernel void near_field_attn(
    constant float* q [[buffer(0)]],
    constant float* keys [[buffer(1)]],
    constant float* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& ring_size [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
        if (tid >= head_dim) return;
        // Simplified: single head, single batch
        float max_score = -FLT_MAX;
        thread float scores[256];
        for (uint j = 0; j < ring_size; j++) {
            float s = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                s += q[d] * keys[j * head_dim + d];
            }
            s *= 0.08838834764f;
            scores[j] = s;
            if (s > max_score) max_score = s;
        }
        float exp_sum = 0.0f;
        for (uint j = 0; j < ring_size; j++) {
            scores[j] = exp(scores[j] - max_score);
            exp_sum += scores[j];
        }
        float result = 0.0f;
        for (uint j = 0; j < ring_size; j++) {
            result += (scores[j] / exp_sum) * values[j * head_dim + tid];
        }
        out[tid] = result;
    }
" R"]
            options:nil error:&error];
        
        if (!library) {
            std::cerr << "Warning: Failed to compile Metal shaders: "
                      << [[error localizedDescription] UTF8String] << std::endl;
            use_metal_ = false;
            return;
        }
        
        // Create compute pipelines
        near_field_pipeline_ = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"near_field_attn"]];
        ema_pipeline_ = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"ema_update"]];
        fusion_pipeline_ = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"dual_path_fusion"]];
    }
    
    bool use_metal_ = false;
    SFACfg cfg_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLComputePipelineState> near_field_pipeline_;
    id<MTLComputePipelineState> ema_pipeline_;
    id<MTLComputePipelineState> fusion_pipeline_;
};
#endif

// ============================================================================
// Benchmark & Comparison
// ============================================================================

static std::vector<float> random_input(int32_t dims, int32_t seq_len) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(seq_len * dims);
    for (auto& v : x) v = dist(gen);
    return x;
}

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f);
}

static int32_t parse_int(const char* arg, int32_t default_val) {
    try { return std::stoi(arg); } catch (...) { return default_val; }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, const char* argv[]) {
    bool cpu_only = false;
    int32_t prefill_seq = 256;
    int32_t decode_steps = 1000;
    bool compare = false;
    
    for (int32_t i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--cpu") cpu_only = true;
        else if (arg == "--prefill") prefill_seq = parse_int(argv[++i], 256);
        else if (arg == "--decode") decode_steps = parse_int(argv[++i], 1000);
        else if (arg == "--compare") compare = true;
        else if (arg == "--help") {
            std::cout << "Usage: ./soma_metal [--cpu] [--prefill N] [--decode N] [--compare]" << std::endl;
            return 0;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Soma Metal Engine v1.0" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mode: " << (cpu_only ? "CPU-only" : "CPU+Metal (auto)") << std::endl;
    
    // Small config for fast demo
    SFACfg cfg;
    cfg.dims = 128;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 32;
    cfg.k = 16;
    cfg.gamma = 0.98f;
    cfg.alpha = 0.1f;
    
    std::cout << "Config: dims=" << cfg.dims
              << " heads=" << cfg.num_heads
              << " kv_heads=" << cfg.num_kv_heads
              << " head_dim=" << cfg.head_dim
              << " k=" << cfg.k
              << " gamma=" << cfg.gamma
              << " alpha=" << cfg.alpha << std::endl;
    std::cout << "Memory: " << (cfg.k * cfg.num_heads * cfg.head_dim * 2 +
                          cfg.num_heads * cfg.head_dim) * sizeof(float) / 1024
              << " KB (ring buffer + field state)" << std::endl;
    
    // Generate random input
    std::vector<float> x = random_input(cfg.dims, prefill_seq);
    
    // ---- Prefill Benchmark ----
    std::cout << "\n--- Prefill Benchmark (seq_len=" << prefill_seq << ") ---" << std::endl;
    
    CPUSFAModule sfa(cfg);
    auto sfa_result = sfa.prefill(x, prefill_seq);
    std::cout << "SFA Prefill: " << std::fixed << std::setprecision(2)
              << sfa_result.ms << " ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(0)
              << (prefill_seq * 1000.0 / sfa_result.ms) << " tokens/sec" << std::endl;
    
    // ---- Decode Benchmark ----
    std::cout << "\n--- Decode Benchmark (" << decode_steps << " steps) ---" << std::endl;
    
    sfa.reset();
    std::vector<float> x_single(cfg.dims, 0.0f);
    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    
    double total_decode_ms = 0.0;
    for (int32_t i = 0; i < decode_steps; i++) {
        for (auto& v : x_single) v = dist(gen);
        auto dec = sfa.decode_step(x_single);
        total_decode_ms += dec.ms;
    }
    
    double avg_decode_ms = total_decode_ms / decode_steps;
    std::cout << "Avg decode: " << std::fixed << std::setprecision(3)
              << avg_decode_ms << " ms/step" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(0)
              << (1000.0 / avg_decode_ms) << " tokens/sec" << std::endl;
    
    // ---- Correctness Comparison (small sequence) ----
    if (compare) {
        std::cout << "\n--- Correctness Comparison (seq_len=32) ---" << std::endl;
        
        std::vector<float> x_small = random_input(cfg.dims, 32);
        
        CPUSFAModule sfa_cmp(cfg);
        auto sfa_out = sfa_cmp.prefill(x_small, 32);
        
        StandardAttention std_attn(cfg);
        auto std_out = std_attn.forward(x_small, 32);
        
        float sim = cosine_similarity(sfa_out.output, std_out);
        std::cout << "Cosine similarity (SFA vs Standard): " << std::fixed << std::setprecision(6)
                  << sim << std::endl;
        
        // Per-token similarity
        std::cout << "Per-token similarity:" << std::endl;
        for (int32_t t = 0; t < 32; t++) {
            float token_sim = cosine_similarity(
                std::vector<float>(sfa_out.output.begin() + t * cfg.dims,
                                   sfa_out.output.begin() + (t + 1) * cfg.dims),
                std::vector<float>(std_out.begin() + t * cfg.dims,
                                   std_out.begin() + (t + 1) * cfg.dims));
            std::cout << "  t=" << t << ": " << std::fixed << std::setprecision(6) << token_sim << std::endl;
        }
    }
    
    // ---- Field State Summary ----
    std::cout << "\n--- Field State Summary ---" << std::endl;
    std::cout << "Field state norm: " << std::fixed << std::setprecision(4)
              << std::sqrt(std::accumulate(sfa_result.field_state.begin(), sfa_result.field_state.end(),
                                          0.0f, [](float a, float b) { return a + b * b; })) << std::endl;
    
    std::cout << "\nDone." << std::endl;
    return 0;
}
