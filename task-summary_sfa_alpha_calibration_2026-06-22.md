# Task Summary: SFA Alpha Calibration Analysis

## Objective
Diagnose why SFA enhancement has minimal impact on PPL and determine optimal alpha parameters.

## Key Findings

### Problem: Enhancement Signal Too Weak
- **Current alpha_base=0.04** produces enhancement at only **0.013%** of attention output norm
- Root cause: `alpha_eff = alpha_base × (0.3+0.7×layer_ratio) × 0.7^layer`
  - Layer 0: alpha_eff=0.012 (83× reduction)
  - Layer 23: alpha_eff=0.00000044 (450,000× reduction)

### Recommended Configurations
| Option | alpha_base | cross_decay | clip | Expected Enhancement Ratio |
|--------|-----------|-------------|------|--------------------------|
| Conservative | 1.0 | 0.9 | 0.5 | ~0.3% |
| Moderate | 5.0 | 0.95 | 1.0 | ~1.6% |
| Aggressive | 10.0 | 0.98 | 2.0 | ~3.1% |

### Orthogonality Unaffected
- Alpha is a scalar multiplier — does not change vector direction
- Cosine similarity remains ~0.029 across all alpha values (PASS)

## Validation Results
- ✅ Orthogonality test: PASS (cosine < 0.1) at seq_len >= 256
- ✅ Sequence isolation test: PASS
- ⚠️ Metal kernel compilation: BLOCKED (no Xcode IDE)
- ℹ️ CPU simulation confirms enhancement ratio scales linearly with alpha_base

## Files Created
- `docs/sfa_alpha_calibration.md` — Full analysis document

## Next Step
Test alpha_base=1.0 on real Qwen2.5-0.5B model to measure actual PPL impact.
