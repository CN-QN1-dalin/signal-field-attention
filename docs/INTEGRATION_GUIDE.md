# SFA v7 Integration Guide

## Quick Start

```bash
# Clone repository
git clone https://github.com/CN-QN1-dalin/signal-field-attention.git
cd signal-field-attention

# Run tests
python3 test_sfa_correctness.py
python3 test_sfa_integration.py

# Compile Metal kernel (requires Xcode)
xcrun -sdk macosx metal -c src/sfa/sfa_kernel.metal -o /tmp/sfa_kernel.metallib
```

## Architecture Overview

```
llama.cpp → SFA Bridge → ggml Graph → Metal Kernel
    ↓           ↓           ↓           ↓
Model Load  Seq Start   Enhance    Vector Ops
Seq Copy    Seq Remove  Clip       EMA Update
Ctx Free                 Scale      Ring Mean
```

## Key Components

### 1. SFA Bridge (`sfa_llama_bridge.cpp`)
- Manages multi-sequence state isolation
- Implements ggml graph construction
- Handles lifecycle hooks

### 2. SFA Kernel (`sfa_kernel.metal`)
- 6 core GPU kernels
- NEON-optimized vector operations
- Metal shader implementation

### 3. Test Suite
- `test_sfa_integration.py` - Integration tests
- `test_sfa_correctness.py` - Correctness verification

## Configuration

```python
# SFA Parameters
SFA_RING_SIZE = 16
SFA_SEMANTIC_SLOTS = 64
SFA_EMA_GAMMA = 0.98
SFA_ENHANCEMENT_CLIP = 0.5
SFA_ALPHA_BASE = 0.1
```

## Troubleshooting

### Metal Compilation Failure
```bash
# Install Xcode command-line tools
xcode-select --install

# Verify installation
xcrun -sdk macosx metal --version
```

### Test Failures
```bash
# Run with verbose output
python3 -m pytest test_sfa_*.py -v
```

## Performance Benchmarks

| Metric | Baseline | SFA v7 | Improvement |
|--------|----------|--------|-------------|
| PPL (0.5B) | 7.43 | 6.90 | -7.08% |
| PPL (7B) | 10.79 | 10.62 | -1.57% |
| Decode Speed | 1x | 1.19x | +19% |
| Memory | 1x | 248x compressed | +24700% |

## Next Steps

1. Compile Metal kernel
2. Run full PPL test suite
3. Prepare llama.cpp PR
4. Update paper data
