# SFA Alpha Calibration Analysis

**Date:** 2026-06-22  
**Author:** QN1幻化引擎团队  
**Status:** ✅ APPLIED — Calibrated configuration deployed

## Problem Statement

The SFA enhancement signal was extremely weak relative to attention output, making it unlikely to have meaningful impact on PPL.

## Root Cause Analysis

### Original Configuration (DEPRECATED)
- `alpha_base = 0.04`
- `cross_decay = 0.7`
- `enhancement_clip = 0.01`

### Measured Values (Original)
| Metric | Value |
|--------|-------|
| Raw enhancement norm (per layer) | 0.183 |
| Attention output norm | 2.993 |
| Scaled enhancement norm (alpha=0.04) | 0.000377 |
| Ratio | 0.013% |

### Why It Was Too Weak

The alpha scaling formula:
```
alpha_eff = alpha_base × (0.3 + 0.7 × layer_ratio) × decay^layer
```

At layer 23 (deepest):
- `alpha_eff = 0.04 × 1.0 × 0.7^23 = 0.04 × 0.000011 = 0.00000044`
- Enhancement is reduced by **450,000×** from raw value

## Applied Configuration (2026-06-22)

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| alpha_base | 0.04 | **0.1** | 2.5× increase |
| cross_decay | 0.7 | **0.8** | Much gentler decay |
| enhancement_clip | 0.01 | **0.5** | 50× increase, prevents saturation |

### Verification Results

| Seq Length | Enhancement Ratio | |cos| | Status |
|------------|------------------|------|--------|
| 64 | 0.44% | 0.220 | ⚠️ Short seq |
| 128 | 0.61% | 0.109 | ⚠️ Borderline |
| 256 | 0.87% | 0.082 | ✅ PASS |
| 512 | 1.25% | 0.090 | ✅ PASS |
| 1024 | 1.77% | 0.017 | ✅ PASS |
| 2048 | 2.57% | 0.017 | ✅ PASS |

### Memory Overhead
| Component | Size |
|-----------|------|
| RingBuffer (24×16×896) | 1.31 MB |
| Field State (24×896) | 0.08 MB |
| Semantic Pool (64×896) | 0.22 MB |
| **Total** | **1.61 MB** |

### Orthogonality
- Cosine similarity remains ~0.03 regardless of alpha (scalar multiplication preserves direction)
- All configurations pass |cos| < 0.1 threshold for seq ≥ 256

## Files Modified
- `src/sfa/sfa_adapter.h` — ALPHA_BASE, CROSS_DECAY, ENHANCEMENT_CLIP
- `src/sfa/sfa_engine.h` — SFA_Config defaults
- `src/sfa/sfa_llama_cpp.h` — SFA_CROSS_DECAY_DEFAULT, SFA_ENHANCEMENT_CLIP, alpha_base
- `src/sfa/sfa_kernel.metal` — SFA_ENHANCEMENT_CLIP

## Next Steps

1. **Immediate:** Test with calibrated config on Qwen2.5-0.5B real model
2. **Monitor:** PPL change, enhancement magnitude, orthogonality
3. **Iterate:** Fine-tune alpha_base based on real PPL results
4. **Document:** Record results for TECHNICAL_REPORT.md revision

## References

- Previous orthogonality test: v4 random projection fix (cosine reduced from 0.65 to 0.007)
- Original alpha_base=0.04 derived from earlier experiments with different clip_val
- The clip_val=0.01 was appropriate for alpha_base=2.0 but not for 0.04
