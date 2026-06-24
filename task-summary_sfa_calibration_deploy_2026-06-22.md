# Task Summary: SFA v7 Parameter Calibration & Deployment

## Objective
Diagnose why SFA enhancement has minimal PPL impact and deploy calibrated parameters.

## Key Findings

### Problem: Enhancement Signal Too Weak
- **Old config:** alpha_base=0.04, cross_decay=0.7, clip=0.01
- Enhancement ratio: 0.013% of attention output — far too weak
- Root cause: 0.7^23 ≈ 0.000011 reduces signal by 450,000× in deepest layers

### Applied Solution
- **New config:** alpha_base=0.1, cross_decay=0.8, clip=0.5
- Enhancement ratio: 1.25% at seq=512, scaling to 2.57% at seq=2048
- Orthogonality maintained: |cos| < 0.1 for seq ≥ 256
- Memory overhead: 1.61 MB total

### Verification Matrix
| Seq | Ratio | |cos| | Status |
|-----|-------|------|--------|
| 64 | 0.44% | 0.220 | ⚠️ |
| 128 | 0.61% | 0.109 | ⚠️ |
| 256 | 0.87% | 0.082 | ✅ |
| 512 | 1.25% | 0.090 | ✅ |
| 1024 | 1.77% | 0.017 | ✅ |
| 2048 | 2.57% | 0.017 | ✅ |

## Files Modified
- `src/sfa/sfa_adapter.h` — ALPHA_BASE 2.0→0.1, CROSS_DECAY 0.7→0.8, ENHANCEMENT_CLIP 0.01→0.5
- `src/sfa/sfa_engine.h` — Same defaults in SFA_Config
- `src/sfa/sfa_llama_cpp.h` — Same defaults
- `src/sfa/sfa_kernel.metal` — ENHANCEMENT_CLIP 0.01→0.5
- `docs/sfa_alpha_calibration.md` — Updated with verification results

## Commits
- 9f81c8b: calibrate SFA parameters in code
- 5e88944: update calibration docs with verification

## Next Step
Test calibrated config on real Qwen2.5-0.5B model to measure actual PPL impact.
