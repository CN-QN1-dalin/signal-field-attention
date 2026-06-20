# Task Summary: v3 Strict Peer Review of Soma Papers

## Objective
Perform an even stricter, deeper peer review of Soma Engine, Heritage, Native, LingYa papers from the perspective of top-tier conference Area Chairs and industry chief researchers.

## Key Reasoning & Findings

### New Critical Findings Beyond v2 Review

1. **F2 - Memory Calculation Systematic Error**: The paper claims 462KB for 7B model at 64K sequence, but benchmark_results.json shows dims=896 giving 115.5KB. Scaling to 7B (dims=3584): 115.5 * (3584/896)^2 = 1.8MB, not 462KB. The paper's 462KB is off by ~4x.

2. **M2 - Lemma 1 Conceptual Error**: The paper claims "EMA decay γ^|i-j| approximates standard attention kernel K(i,j) = exp(-|i-j|/σ)". But standard attention kernel K(i,j) = exp(q_i^T k_j / √d) is determined by **QK dot product similarity**, NOT by distance |i-j|. Standard attention is **global**, not distance-decaying. The paper is comparing EMA decay to a "distance kernel" that doesn't exist in standard attention. This is a fundamental conceptual error.

3. **E1 - Correctness Test Setting**: The benchmark sets `soma.alpha = 0.0` (disabling far-field channel) and tests only Ring Buffer vs Standard Attention, NOT the complete SFA mechanism.

4. **E3 - PPL Testing Gap**: The PPL benchmark only tests the original HuggingFace model, NOT the SFA-replaced model. All claimed PPL numbers in the papers are simulated.

5. **N1 - Paper Overlap**: Engine and Native share correctness/ memory/speed data. Heritage and Native share PPL and hyperparameter data. Question: can these be merged into one paper?

6. **N3 - Inconsistent Technical Narrative**: Engine says "complete replacement of self-attention", Heritage says "distillation to student model", Native says "native architecture designed from scratch", Convergence says "convergence mechanism". Same technology described differently.

### Updated Desk-Reject Triggers

| # | Trigger | Papers Affected |
|---|---------|----------------|
| F1 | >50% key data points are simulated | Heritage, Native, LingYa |
| F2 | Systematic memory calculation error (4x off) | Engine, Native |
| F3 | "Soma Labs" creates false impression | All |
| F4 | Missing SOTA comparisons (QLoRA, Mamba-2) | Heritage, Native, LingYa |
| M2 | Lemma 1 compares to non-existent kernel | Heritage |
| M1 | Inequality direction error in Theorem 1 | Heritage |

### Submission Strategy Recommendation

**Plan A (Preferred)**: Merge Engine + Native + Heritage + LingYa into ONE paper titled "Soma: A Signal Field Native Architecture for Efficient Long-Sequence Processing" targeting ICLR/NeurIPS 2027.

**Plan B**: Split into Architecture paper (Engine+Native) + Applications paper (Heritage+LingYa).

**Plan C (Pragmatic)**: Publish on arXiv first with clear disclaimer "preliminary results, simulation-based experiments", then supplement with real experiments.

## Files Created

- `顶会审查报告_v3_strict.md` (9.2KB) - Comprehensive strict review from Area Chair perspective
