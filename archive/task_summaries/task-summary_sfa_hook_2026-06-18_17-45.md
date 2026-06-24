# Task Summary: SFA Hook Injection on Qwen2.5-0.5B

## Objective
Inject SFA (Signal Field Attention) logic into Qwen2.5-0.5B-Instruct's attention modules and measure PPL impact.

## Key Reasoning
1. **Multiple injection approaches failed**:
   - `isinstance(module, torch.nn.MultiheadAttention)` — Qwen2.5 uses `Qwen2Attention`, not `MultiheadAttention`
   - `types.MethodType(wrapper, module)` — caused `self` to be passed twice
   - Direct `module.forward = wrapper` — wrapper called `orig_forward(*args)` but `args[0]` was already `self`

2. **Forward hook breakthrough**: Using `module.register_forward_hook(hook_fn)` bypasses all method-binding issues. Hook receives `(module, input_args, output)` cleanly.

3. **Proxy limitation**: Using `attn_output` as proxy for `hidden_states` loses information, resulting in weak enhancement signal.

## Conclusions
- **Hook injection works**: 24/24 Qwen2Attention modules successfully intercepted
- **PPL impact is minimal**: α=0.2 gives only 0.76% improvement (2.0507 → 2.0350)
- **Next steps**: Get real `hidden_states` from kwargs, add cross-layer decay, test on longer sequences

## Timeline
- **17:00-17:20**: Discovered root cause of monkey-patch failures (self binding issue)
- **17:20-17:30**: Implemented forward hook approach, confirmed injection works
- **17:30-17:45**: Ran full α sweep, documented results

## Files
- `/Users/apple/Desktop/太初五岳开源/00_nova_attention/sfa_ppl_final.py` — Working injection
- `/Users/apple/Desktop/太初五岳开源/00_nova_attention/test_report_sfa_injection.md` — Report
