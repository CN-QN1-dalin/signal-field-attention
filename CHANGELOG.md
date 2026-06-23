# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-06-23

### Added
- SFA v7 integration test suite (`test_sfa_integration.py`)
- Orthogonality correctness test (`test_sfa_correctness.py`)
- Metal GPU kernel (`src/sfa/sfa_kernel.metal`)
- C++ bridge layer (`src/sfa/sfa_llama_bridge.cpp`)
- Universal adapter pattern for llama.cpp integration (`src/sfa/sfa_llama_cpp.cpp`)
- Integration guide (`docs/INTEGRATION_GUIDE.md`)
- LLaMA.cpp integration checklist (`docs/LLAMA_CPP_INTEGRATION_CHECKLIST.md`)

### Changed
- Calibrated SFA constants: ALPHA_BASE=0.1, ENHANCEMENT_CLIP=0.5, CROSS_DECAY=0.8
- Replaced `ggml_new_i32` with `ggml_new_tensor_1d + ggml_set_f32` for API compliance
- Fixed field_state tensor shape handling in `build_sfa_enhance`
- Implemented proper sequence lifecycle hooks (seq_cp/seq_rm)

### Fixed
- Removed hardcoded paths (`/tmp/llama.cpp/ggml/include/ggml.h`)
- Resolved stale constant values in `sfa_lockfree.h`
- Thread safety improvements in global state management

### Test Results
- Qwen2.5-0.5B: Orthogonality cosine ≈ 0.0 (target < 0.1)
- Memory compression: 248x at seq=4096, 3972x at seq=65536
- Enhancement ratio: ~3.3% of attention output
