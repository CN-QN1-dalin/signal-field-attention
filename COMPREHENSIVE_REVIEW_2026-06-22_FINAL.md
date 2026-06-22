# Dalin Soma Project: Comprehensive Review & Strategic Planning Summary

## Objective
To conduct a full-scale review of the "Tai Chu Wu Yue" (Dalin Soma) project assets, identify critical bottlenecks (specifically session concurrency limits and memory constraints), and formulate a consolidated "Assault Plan" based on 12 core expert reports to solve high-priority technical challenges in distributed inference, extreme quantization, and SFA-based routing.

## Key Reasoning & Progress

### 1. Project Status & Bottlenecks
* **Concurrency Limit:** The primary operational bottleneck is the `sessions_spawn` limit of 5/5, causing many sub-agent tasks to return "Forbidden."
* **Context Loss:** `MEMORY.md` suffers from truncation due to volume, leading to lost context.
* **Asset Inventory:** Identified 12 core expert reports in `docs/` (e.g., `distributed_inference.md`, `extreme_quantization.md`, `wanxiang_loading_strategy.md`).
* **Cleanup Needed:** Root directory and `memory/` contain non-core artifacts (dream diaries, stray task files) that need archiving to free up context space.

### 2. Technical Core: The "Assault" Plan
The strategy shifts from parallel spawning to **serial execution** of key experts to bypass concurrency limits, leveraging the following synthesized technical pillars:

* **SFA Pre-routing (DeepSeek-R1-Distill-Qwen-14B):**
  * **Problem:** 256 Experts, only 6 active (2.34% hit rate).
  * **Solution:** SFA Signal Field Attention (EMA-based) predicts expert activation.
  * **Architecture:** Semantic Feature Extractor (SFE) → RouterNet (332k params, low overhead) → GateVerifier (confidence-based triage).
  * **Strategy:** Three-tier prefetching (Hot/Warm/Cold) based on confidence scores.

* **Extreme Quantization & Compression:**
  * **IQ1_S for 671B Models:** Achieves ~44GB size for 671B params.
  * **Memory Budget:** Theoretical minimum ~52-60GB RAM required.
  * **mmap Strategy:** Dynamic expert loading via OS memory mapping. Only active experts (top-8) reside in RAM; others stay on SSD.
  * **Hybrid Precision:** Embeddings/Shared layers FP16/FP8; Experts NF4/INT4.

* **Liquefaction v2.0 (Storage Paradigm):**
  * **Concept:** Parameter lifecycle managed by SFA signals across three tiers: SSD(INT4) → RAM(FP8 Hot) → GPU(FP8 Compute).
  * **Mechanism:** "Phase transitions" driven by 5D SFA signal field ($\alpha, \beta, \gamma, \delta, \rho$).
  * **Performance:** ~22x compression ratio, ~3-5ms/token latency, minimal accuracy loss (~0.7 MMLU points).

* **Distributed Inference (Apple Silicon Cluster):**
  * **Topology:** Multi-Mac cluster via Thunderbolt 4/USB4 (<1ms latency target).
  * **Stack:** Flatbuffers, UDP zero-copy, Metal pipelines.
  * **Strategy:** MoE expert distribution + Sequence/Tensor parallelism.

### 3. Critical Bugs Found (P0)
* **llama.cpp Integration:**
  1. `field_state` name matching is fragile/sync-vulnerable.
  2. `n_sfa_layers` incorrectly used instead of `n_swa`.
  3. `seq_cp`/`seq_rm` index comparison logic is inverted.

## Conclusions & Next Steps

1. **Immediate Action:** Archive/clean root directory junk to reduce context noise.
2. **Execution:** Re-run expert tasks **sequentially** to avoid 5/5 spawn limit. Prioritize fixing P0 bugs in `distributed_inference.md` and `extreme_quantization.md`.
3. **Integration:** Merge SFA pre-routing logic with the Liquefaction v2.0 manager to optimize the `mmap` swap-in/out cycles.
4. **Documentation:** Update `MEMORY.md` with only the high-level parameters of the 12 reports to prevent truncation.

## Timeline

* **[2026-06-21 01:09 - 2026-06-22 10:01]**: Initial assessment. Identified 12 docs, concurrency limits, and context truncation issues.
* **[2026-06-22 10:01 - 10:03]**: Generation of core technical documents: `distributed_inference.md`, `extreme_quantization.md`, `wanxiang_loading_strategy.md`.
* **[2026-06-22 10:03 - 10:05]**: Deep dive into SFA Pre-routing (RouterNet design) and IQ1_S memory calculations for 671B models.
* **[2026-06-22 10:05 - 10:07]**: Refinement of Liquefaction v2.0 (SSD/RAM/GPU tiering) and Distributed Inference topology.
* **[2026-06-22 10:07 - 10:10]**: User directive: "Lead all experts to review and then assault/tackle hard problems."
* **[2026-06-22 10:10 - 10:17]**: Comprehensive review of `ALL_EXPERTS_ASSAULT_PLAN.md` and `FORWARD_PLAN.md`. Identification of P0 bugs. Creation of `COMPREHENSIVE_REVIEW_2026-06-22.md`.

Expand for details about: Exact file listing of docs/, list of rejected subagent UUIDs, MEMORY.md truncation percentage, specific content of distributed_inference.md and extreme_quantization.md, plan for serializing expert tasks
