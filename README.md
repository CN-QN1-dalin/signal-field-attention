# QN1 Engine: Signal Field Attention & Efficient Inference Suite

A suite of eight efficient attention and inference acceleration techniques, anchored by Signal Field Attention (SFA). Replaces O(n²) self-attention with an O(k·n) dual-channel approach. Achieves 4× decoding speedup and 248× memory compression on 7B models while maintaining competitive perplexity.

## Overview

Signal Field Attention (SFA) decomposes standard self-attention into two channels:

- **Near-field channel**: Standard softmax attention on the most recent k tokens (precise)
- **Far-field channel**: Exponentially weighted moving average (EWMA) state that compresses historical key-value pairs into a fixed-size vector (efficient)

The two channels are fused with a learnable mixing coefficient, preserving exact attention for recent tokens while compressing historical context into O(k) memory regardless of sequence length.

## Key Results

| Metric | Standard Attention | Signal Field | Improvement |
|--------|-------------------|-------------|-------------|
| Decoding speed (7B) | 1× | 4.16× | 4× faster |
| KV cache (64K seq) | 2.1 GB | 8.6 MB | 248× compression |
| Additional parameters | 0 | ~8 KB | Negligible |
| PPL (0.5B, shallow) | 22.375 | 23.062 | +3.07% |
| PPL (0.5B, deep) | 22.375 | 20.011 | −10.57% |

## Quick Start

### Requirements

- Python 3.8+
- [MLX](https://ml-explore.github.io/mlx/) (Apple Silicon) — for full experiments
- PyTorch — for experiment scripts (no external dependencies required)

### Installation

```bash
# Clone the repository
git clone https://github.com/CN-QN1-dalin/signal-field-attention.git
cd signal-field-attention

# Install MLX (Apple Silicon only, for full model experiments)
pip install mlx

# Experiment scripts use only standard library — no pip install needed
```

### Running Experiments

```bash
# Run all 8 experiment suites
python3 run_all.py

# Run individual experiments
python3 01-signal-field/signal_field.py
python3 02-huayue/huayue.py
python3 03-guiyuan/guiyuan.py
python3 04-lingya/lingya.py
python3 05-ring-buffer/ring_buffer.py
python3 06-rca/rca.py
python3 07-metal-kernel/metal_kernel.py
python3 08-ultra/ultra.py
```

## Architecture

### Signal Field Attention

```
Input X
│
├──→ [QKV Projection]
│     │
│     ├── Near-field channel:
│     │   Standard softmax(Q, K_near, V_near)  [k tokens]
│     │
│     ├── Far-field channel:
│     │   S_t = γ·S_{t-1} + (1-γ)·mean(K_hist)  [fixed state]
│     │
│     └──→ Fusion: near + α · far
│
└──→ Output Projection → Y
```

**Core equations:**

```
output = softmax(Q · K_near^T / √d) · V_near + α · S_far
S_t = γ · S_{t-1} + (1-γ) · mean(K_hist)
```

### MLX Implementation

For full model experiments on Apple Silicon:

```bash
# Run incremental inference benchmark (requires MLX)
cd src
python3 taicu_sf_v2.py

# Run distillation training on 0.5B model
python3 taicu_0.5b_distill_v2.py

# Run 7B benchmark
python3 taicu_7b_benchmark.py

# Run PPL parameter search and validation
python3 taicu_7b_ppl_search.py
python3 taicu_7b_ppl_verify.py
```

## Project Structure

```
├── README.md                    # This file
├── LICENSE                      # MIT License
├── run_all.py                   # Experiment runner (all 8 suites)
│
├── 01-signal-field/             # Signal Field Attention core
│   ├── signal_field.py          # Dual-channel attention implementation
│   └── test_signal_field.py     # Unit tests
│
├── 02-huayue/                   # Hybrid architecture (Attention + SSM)
│   └── huayue.py
│
├── 03-guiyuan/                  # SSM KV compression
│   └── guiyuan.py
│
├── 04-lingya/                   # Orthogonal basis fine-tuning
│   └── lingya.py
│
├── 05-ring-buffer/              # O(1) KV cache with ring buffer
│   └── ring_buffer.py
│
├── 06-rca/                      # Frequency-domain attention (RFF)
│   └── rca.py
│
├── 07-metal-kernel/             # Metal GPU kernel implementations
│   └── metal_kernel.py
│
├── 08-ultra/                    # Ultra-efficient model deployment
│   └── ultra.py
│
├── src/                         # Full model MLX implementations
│   ├── taicu_sf_v2.py           # Incremental inference engine
│   ├── taicu_0.5b_distill.py    # 0.5B distillation (original)
│   ├── taicu_0.5b_distill_v2.py # 0.5B distillation (improved)
│   ├── taicu_7b_benchmark.py    # 7B inference benchmark
│   ├── taicu_7b_ppl_search.py   # PPL hyperparameter search
│   └── taicu_7b_ppl_verify.py   # PPL validation
│
├── OPEN_SOURCE.md               # Detailed open-source declaration
├── TECHNICAL_REPORT.md          # Full technical report
└── LICENSE                      # MIT License
```

## Experiment Suite Details

### 1. Signal Field Attention (v5d)
- **Purpose**: Core dual-channel attention mechanism
- **Tests**: Stability, compression ratio, throughput, layer analysis
- **Key metric**: Memory O(k·d) fixed, independent of sequence length

### 2. Huayue Hybrid Architecture
- **Purpose**: Attention + SSM layered hybrid
- **Tests**: Architecture construction, performance simulation, S-curve allocation
- **Key metric**: 75% SSM replacement with <1% PPL impact

### 3. Guiyuan KV Compression
- **Purpose**: Gaussian-decay KV cache compression
- **Tests**: Memory compression rate, prefill/decode consistency
- **Key metric**: ≥99% compression with minimal accuracy loss

### 4. LingYa Orthogonal Fine-tuning
- **Purpose**: Parameter-efficient fine-tuning via orthogonal basis
- **Tests**: Parameter count vs LoRA, orthogonality verification
- **Key metric**: 50% parameter reduction vs LoRA

### 5. RingBuffer KV Cache
- **Purpose**: Fixed-size circular buffer for O(1) memory
- **Tests**: Write/read performance, memory constancy
- **Key metric**: O(1) memory regardless of sequence length

### 6. RCA Frequency-Domain Attention
- **Purpose**: Random Fourier Feature approximation
- **Tests**: FFT reconstruction error, RFF attention accuracy
- **Key metric**: O(n·M·d) complexity with minimal error

### 7. Metal GPU Kernel
- **Purpose**: Direct Metal shader GPU acceleration
- **Tests**: Custom kernels for dequantization and attention fusion
- **Key metric**: ~25× theoretical GPU speedup

### 8. Ultra Deployment
- **Purpose**: Ultra-efficient model deployment
- **Tests**: Layer-wise loading, INT4 quantization, swap optimization
- **Key metric**: 75B model on 16GB with minimal swap

## Implementation Details

### Pure Python Experiments
All 8 experiment suites (`01-` through `08-`) are implemented in pure Python with zero external dependencies (only the standard library). They can run on any machine with Python 3.8+.

### Full Model Experiments
The `src/` directory contains MLX implementations for full model experiments on Apple Silicon:
- Incremental prefill + decode pipeline
- Layer-wise distillation training
- Benchmark and PPL validation

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

Free for personal, academic, and educational use. For commercial licensing inquiries, contact: 362118251@qq.com

## Citation

If you find this work useful, please cite:

```bibtex
@misc{signal_field_attention_2026,
  author = {Taicu Team},
  title = {Signal Field Attention: Learning to Compress Attention for Efficient Inference},
  year = {2026},
  howpublished = {\url{https://github.com/CN-QN1-dalin/signal-field-attention}},
  note = {Technical Report v1.0}
}
```

## Acknowledgments

We build upon the shoulders of giants:
- [Transformer](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [Linformer](https://arxiv.org/abs/2006.04768) (Wang et al., 2020)
- [Performer](https://arxiv.org/abs/2009.14794) (Choromanski et al., 2021)
- [Mamba](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- [StreamingLLM](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023)
- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2022)
- [MLX](https://github.com/ml-explore/mlx) (Apple)
- [Qwen](https://github.com/QwenLM/Qwen) (Alibaba Cloud)

## Disclaimer

This code is provided for research and educational purposes. The experiments described use synthetic data and benchmark models. Results may vary with different datasets, model architectures, and hardware configurations.

Signal Field Attention is designed to work alongside existing Transformer architectures — it is a drop-in replacement for attention layers, not a complete model architecture replacement.
