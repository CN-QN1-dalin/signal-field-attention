# SFA Alpha Calibration Analysis

**Date:** 2026-06-22  
**Author:** QN1幻化引擎团队  
**Status:** Critical Finding — alpha_base too small

## Problem Statement

The SFA enhancement signal is extremely weak relative to attention output, making it unlikely to have meaningful impact on PPL.

## Root Cause Analysis

### Current Configuration
- `alpha_base = 0.04`
- `cross_decay = 0.7`
- `enhancement_clip = 0.01`

### Measured Values
| Metric | Value |
|--------|-------|
| Raw enhancement norm (per layer) | 0.183 |
| Attention output norm | 2.993 |
| Scaled enhancement norm (alpha=0.04) | 0.000377 |
| Ratio | 0.013% |

### Why It's Too Weak

The alpha scaling formula:
```
alpha_eff = alpha_base × (0.3 + 0.7 × layer_ratio) × decay^layer
```

At layer 23 (deepest):
- `alpha_eff = 0.04 × 1.0 × 0.7^23 = 0.04 × 0.000011 = 0.00000044`
- Enhancement is reduced by **450,000×** from raw value

Even at layer 0:
- `alpha_eff = 0.04 × 0.3 × 1.0 = 0.012`
- Enhancement is reduced by **83×** from raw value

## Recommended Configuration

### Option A: Conservative (Recommended for initial testing)
| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| alpha_base | 0.04 | **1.0** |
| cross_decay | 0.7 | **0.9** |
| enhancement_clip | 0.01 | **0.5** |

Expected enhancement ratio: ~0.3% of attention output

### Option B: Moderate
| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| alpha_base | 0.04 | **5.0** |
| cross_decay | 0.7 | **0.95** |
| enhancement_clip | 0.01 | **1.0** |

Expected enhancement ratio: ~1.6% of attention output

### Option C: Aggressive
| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| alpha_base | 0.04 | **10.0** |
| cross_decay | 0.7 | **0.98** |
| enhancement_clip | 0.01 | **2.0** |

Expected enhancement ratio: ~3.1% of attention output

## Orthogonality Under Different Alphas

Tested with random projection orthogonality fix:

| alpha_base | avg_cosine | Result |
|------------|-----------|--------|
| 0.04 | 0.029 | ✅ PASS |
| 1.0 | 0.029 | ✅ PASS (unchanged, alpha is scalar) |
| 5.0 | 0.029 | ✅ PASS |
| 10.0 | 0.029 | ✅ PASS |

Orthogonality is unaffected by alpha scaling (scalar multiplication preserves direction).

## Next Steps

1. **Immediate:** Test with alpha_base=1.0 on Qwen2.5-0.5B
2. **Monitor:** PPL change, enhancement magnitude, orthogonality
3. **Iterate:** If PPL degrades, reduce alpha_base; if no improvement, increase
4. **Document:** Record results for TECHNICAL_REPORT.md revision

## References

- Previous orthogonality test: v4 random projection fix (cosine reduced from 0.65 to 0.007)
- Current alpha_base=0.04 derived from earlier experiments with different clip_val
- The clip_val=0.01 was appropriate for alpha_base=2.0 but not for 0.04
