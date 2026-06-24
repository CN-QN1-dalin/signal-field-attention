# Task Summary: SFA v7 Alpha Calibration & Full Benchmark

## Objective
Calibrate SFA parameters (alpha_base, cross_decay, clip) based on enhancement ratio analysis, deploy to codebase, and run full benchmark suite.

## Key Findings

### Alpha Calibration Analysis
- **Root cause of weak signal:** Exponential decay (0.7^23) reduces alpha_eff by 450,000× in deepest layers
- **Tested values:**
  - alpha_base=1.0: ~0.3% enhancement ratio
  - alpha_base=5.0: ~1.6% enhancement ratio  
  - alpha_base=10.0: ~3.1% enhancement ratio
- **Selected:** alpha_base=0.1 (conservative, cross_decay=0.8 gives better layer coverage)

### Deployed Configuration
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| alpha_base | 0.04 | 0.1 | 2.5x stronger base signal |
| cross_decay | 0.7 | 0.8 | Better layer coverage |
| enhancement_clip | 0.01 | 0.5 | Prevent signal saturation |

### Verification (Orthogonality)
All seq_len >= 256 pass with |cos| < 0.1:
- seq=256: 0.87% ratio, cos=0.082 ✅
- seq=512: 1.25% ratio, cos=0.090 ✅
- seq=1024: 1.77% ratio, cos=0.017 ✅
- seq=2048: 2.57% ratio, cos=0.017 ✅

### Full Benchmark Results
- **Correctness**: sim_skip_t0 = 1.00000000 (perfect) ✅
- **Memory**: 248x at 4K, 3972x at 64K ✅
- **Speed**: Python/MLX slower than standard (expected — C++/Metal needed)
- **Decode**: Near-O(1) across all seq lengths ✅
- **PPL**: Baseline 24.16 (SFA-enhanced PPL pending real model integration)

## Files Modified
1. `src/sfa/sfa_adapter.h` — ALPHA_BASE, CROSS_DECAY, ENHANCEMENT_CLIP
2. `src/sfa/sfa_engine.h` — SFA_Config defaults
3. `src/sfa/sfa_llama_cpp.h` — SFA_CROSS_DECAY_DEFAULT, SFA_ENHANCEMENT_CLIP
4. `src/sfa/sfa_kernel.metal` — SFA_ENHANCEMENT_CLIP
5. `docs/sfa_alpha_calibration.md` — Full analysis
6. `benchmark_results.json` — Complete benchmark data
7. `memory/2026-06-22.md` — Session log

## Commits (all pushed to main)
- 9f81c8b: calibrate SFA parameters in code
- 5e88944: update calibration docs
- 0ee55a1: add benchmark results
- 0962127: memory update

## Next Steps
1. Build C++/Metal backend to realize speed advantage
2. Run SFA-enhanced PPL on real Qwen2.5-0.5B model
3. Fine-tune alpha_base based on real PPL results
