# Open Source Declaration

## 1. License

This repository and all code herein is released under the **MIT License**. See [LICENSE](LICENSE) for the full text.

## 2. Free Use

**Personal, academic, and educational use is completely free.**

You are free to:
- Download, copy, modify, and distribute this code
- Use it for academic research and publications
- Use it in personal projects and learning
- Distribute modified versions (retain this notice)

## 3. Commercial Use

For any commercial use — including but not limited to integration into commercial products, SaaS services, paid APIs, or any technology used for revenue generation — please contact:

**Email: 362118251@qq.com**

## 4. Technical Lineage

This work builds upon publicly available research and open-source projects. We acknowledge and appreciate the contributions of the broader research community.

### 4.1 Attention Mechanism Foundations

| Work | Year | Contribution |
|------|------|-------------|
| Attention Is All You Need (Vaswani et al.) | 2017 | Transformer architecture and self-attention |
| FlashAttention-1 (Dao et al.) | 2022 | I/O-aware exact attention |
| FlashAttention-2 (Dao et al.) | 2023 | More efficient attention implementation |

### 4.2 Long-Sequence / Efficient Attention Alternatives

| Work | Year | Method | Relationship |
|------|------|--------|-------------|
| Linformer (Wang et al.) | 2020 | Low-rank projection | Same problem domain; different approach |
| Performer (Choromanski et al.) | 2021 | Random Fourier feature approximation | Different compression strategy |
| BigBird (Zaheer et al.) | 2021 | Sparse attention patterns | Different approach: sparse vs dual-channel |
| StreamingLLM (Xiao et al.) | 2023 | Attention sink + sliding window | Related concept: far-field compression |
| Mamba (Gu & Dao) | 2023 | State space models | Alternative paradigm |
| RWKV (Zhou et al.) | 2023 | Linear Attention variant | Alternative approach to linear complexity |
| H2O (Zhang et al.) | 2023 | Importance-based KV selection | Related KV cache optimization |
| SnapKV (Li et al.) | 2024 | Attention-score-based KV selection | Related KV cache optimization |
| EAGLE (Li et al.) | 2024 | Speculative decoding | Related inference acceleration |

### 4.3 Parameter-Efficient Fine-Tuning

| Work | Year | Relationship |
|------|------|-------------|
| LoRA (Hu et al.) | 2022 | Baseline for comparison |
| QLoRA (Dettmers et al.) | 2023 | Quantized fine-tuning reference |
| LoRA+ (Hayou et al.) | 2024 | Improved LoRA initialization |

### 4.4 Implementation Frameworks

- **[MLX](https://github.com/ml-explore/mlx)** (Apple) — Machine learning framework for Apple Silicon
- **[Qwen2.5](https://github.com/QwenLM/Qwen)** (Alibaba Cloud) — Base models used for experiments
- **[PyTorch](https://pytorch.org/)** — Core experimentation framework

## 5. What This Project Contributes

This project proposes a dual-channel attention mechanism that decomposes standard self-attention into:

1. A near-field channel using standard softmax attention on recent tokens
2. A far-field channel using EWMA-compressed state for historical context

The approach achieves measurable improvements in inference speed and memory usage on standard benchmark models, while maintaining competitive perplexity.

## 6. Quick Start

```bash
# Run all experiments (pure Python, no external dependencies)
python3 run_all.py

# Run individual experiment suites
python3 01-signal-field/signal_field.py
python3 02-huayue/huayue.py
# ... etc
```

Full model experiments require MLX on Apple Silicon:

```bash
pip install mlx
cd src
python3 taicu_sf_v2.py
```

## 7. Citation

If this work contributes to your research, please cite:

```bibtex
@misc{signal_field_attention_2026,
  author = {QN1幻化引擎团队},
  title = {Signal Field Attention: Learning to Compress Attention for Efficient Inference},
  year = {2026},
  note = {Technical Report v1.0},
  url = {<repository-url>}
}
```

## 8. Disclaimer

This code is provided for research and educational purposes. Experimental results are obtained on specific benchmark models and hardware configurations. Actual performance may vary depending on the dataset, model architecture, and hardware used.

---

**Contact for commercial licensing: 362118251@qq.com**
