# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [2026-06-22] - SFA v7 Integration Complete

### Added
- SFA v7 random projection orthogonality fix (v4)
- llama.cpp integration bridge (`sfa_llama_bridge.cpp`)
- Metal GPU kernel implementation (`sfa_kernel.metal`)
- Integration test suite (`test_sfa_integration.py`, `test_sfa_correctness.py`)
- Technical report appendix on orthogonality fix
- Integration guide documentation

### Changed
- Updated TECHNICAL_REPORT.md with latest results
- Fixed ggml API usage in `sfa_llama_cpp.cpp`
- Improved multi-sequence state isolation

### Fixed
- P0 Bug 1: field_state tensor shape handling
- P0 Bug 2: n_sfa_layers misuse in layer_alpha calculation
- P0 Bug 3: seq_cp/seq_rm lifecycle hooks
- Ring buffer mean calculation using correct ggml_mean API
- Semantic pool attention matrix multiplication order

### Performance
- Cosine similarity reduced from 0.65 to -0.042 ~ 0.007
- PPL improvement: -1.61% to -5.79% on real models
- Memory compression: 248x at 64K sequence

## [2026-06-21] - Project Renamed to Dalin Soma

### Changed
- Project renamed from "Taicu/QN1" to "Dalin Soma"
- Repository structure cleaned and organized
- License changed to MIT

## [2026-06-16] - Initial Release

### Added
- SFA v7 multi-layer end-to-end validation
- Metal GPU kernel pipeline
- llama.cpp integration prototype
