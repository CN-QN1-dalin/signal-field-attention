# TECHNICAL_REPORT.md Update Log

**Date**: 2026-06-22 11:32 GMT+8  
**Previous Version**: v1.1 (June 2026, simulator-only)  
**New Version**: v2.0 (June 22, 2026)

## Changes Made

### 1. Version & Status Update
- Updated version from v1.1 to v2.0
- Changed status from "Prototype implementation with simulator-based evaluation" to "Multi-version prototype ecosystem with SFA v7 end-to-end validation, llama.cpp integration, and Metal kernel pipeline"

### 2. Abstract Enhancement
- Added SFA v7 verified results (+19% speedup, +34% at 32K, 0.9% PPL loss, 0 memory delta)
- Added llama.cpp integration status
- Added Metal kernel pipeline status
- Clearly separated verified data from theoretical estimates

### 3. New Section: SFA v7 Multi-Layer Architecture (2.4)
- Documented anchor-based far-field (K_ANCHORS=8)
- Documented adaptive α formula
- Documented v7a (conservative, 8/24 layers) and v7b (aggressive, 24/32 layers) modes
- Documented near-field window size kn=256

### 4. New Section: llama.cpp Integration Architecture (2.5)
- Documented three-signal-channel architecture (Ring Buffer, EMA Field, Semantic Pool)
- Documented Gaussian compression channel
- Documented cross-layer α decay formula
- Documented enhancement clipping mechanism

### 5. New Section 3.2: SFA v7 End-to-End Validation
- **VERIFIED DATA** from real Qwen2.5-7B-4bit inference
- PPL results: -1.61% to -5.79% improvement (net gain, contrary to simulator predictions)
- Speedup: +19% average, +34% at 32K long sequences
- Memory delta: 0%

### 6. Updated Section 3.3 (Legacy PPL Results)
- Retained simulator data but added clear warning label
- Added updated interpretation referencing v7 real-model results

### 7. New Section 3.5: Metal Engine Performance
- Verified data from C++/Metal engine
- Prefill: 7.31ms (35,021 tok/s)
- Decode: 0.036ms (27,884 tok/s)
- O(1) decode complexity verified across 10 sequence lengths (128–65,536)

### 8. Updated Section 4.3 (Limitations)
- Added mixed data authenticity clarification
- Added llama.cpp prototype-level warning
- Added α=0.1 full enhancement testing gap

### 9. Updated Section 4.6 (Future Work)
- Added llama.cpp end-to-end testing
- Added α=0.1 full SFA enhancement testing
- Added ablation study requirement

### 10. New Section 4.7: arXiv Submission Blocking Factors
- Six specific blockers identified for arXiv readiness
- Mixed data provenance, missing theoretical foundation, no ablation study, comparison gap, reviewer concern risk, code reproducibility

### 11. New Section 6: Project Structure Overview
- Comprehensive table of all project directories with status indicators

### 12. Updated References
- Added MiniMax Sparse Attention (Lai et al. 2026)

## Key Data Classification

| Data Type | Sections | Examples |
|-----------|----------|----------|
| **Verified** | 3.2, 3.5 | SFA v7 PPL, speedup, Metal engine latency |
| **Simulator** | 3.3, 3.4, 3.7 | 0.5B PPL, theoretical speedup, memory compression |
| **Theoretical** | 3.4, Abstract | 4.16× speedup target, 248× compression target |

## Next Steps (Priority Order)

1. **llama.cpp compilation test** — Build the SFA integration with llama.cpp and run real inference
2. **Metal SDK installation** — Resolve Xcode dependency for GPU kernel compilation
3. **α=0.1 full enhancement test** — Systematically test α=0.1 across all layers
4. **Ablation study** — Isolate impact of near-field only, far-field only, adaptive α, anchor count
5. **Direct comparison** — Benchmark against StreamingLLM, H2O, SnapKV on same dataset
6. **arXiv manuscript preparation** — Address six blocking factors in Section 4.7
