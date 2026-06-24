# Task Summary: SFA Hook Injection v6 — Cross-Layer Enhancement

## Objective
Optimize SFA injection into Qwen2.5-0.5B using proven v4 architecture (cross-layer decay + shared memory + ring buffer) with forward hooks.

## Key Reasoning
1. **Root cause of previous failures**: Monkey-patching Qwen2Attention.forward caused `self` binding issues. Forward hooks bypass this entirely.
2. **Injection point matters**: Previous attempts injected at attention output (lost info) or decoder layer final output (diluted by residual). v6 injects at decoder layer output (after attention, before next layer).
3. **Cross-layer decay prevents signal accumulation**: alpha_effective = alpha_base × 0.7^layer_idx ensures deeper layers get less enhancement.
4. **Shared memory pool**: All 24 layers share one SemanticMemoryPool + GaussianCompressor, reducing overhead and preventing noise accumulation.

## Conclusions
- **Hook injection works**: 24/24 layers successfully intercepted
- **PPL improves monotonically with α**: 1.8335 (α=0) → 1.8318 (α=0.5), Δ=-0.11%
- **Signal too weak**: Enhancement at decoder layer output is diluted by residual connections
- **Next step**: Inject at attention output → residual boundary (hook Qwen2Attention directly, return modified tensor)

## Timeline
- **17:45**: Identified root cause (hook on decoder layer, not attention submodule)
- **17:45-18:00**: Built v6 with cross-layer decay + shared memory
- **18:00-18:10**: Ran full α sweep, documented results

## Files
- `/Users/apple/Desktop/太初五岳开源/00_nova_attention/sfa_ppl_v6_fixed.py`
- `/Users/apple/Desktop/太初五岳开源/00_nova_attention/test_report_sfa_v6.md`
