#!/bin/bash
# Build .metallib from SFA_Metal.metal
# Requires Xcode Command Line Tools with Metal SDK
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/SFA_Metal.metal"
AIR="${SCRIPT_DIR}/SFA_Metal.air"
METALLIB="${SCRIPT_DIR}/SFA_Metal.metallib"

if [ ! -f "$SRC" ]; then
    echo "Error: $SRC not found"
    exit 1
fi

echo "=== Building SFA Metal GPU Kernels ==="
echo "Source: $SRC"

# Check for metal compiler
if ! command -v metal &>/dev/null; then
    echo ""
    echo "⚠ Metal compiler not found in PATH."
    echo "  Install Xcode Command Line Tools:"
    echo "    xcode-select --install"
    echo ""
    echo "  Or download Xcode from App Store."
    echo ""
    echo "  Meanwhile, CPU fallback is ready to use."
    echo "  Build CPU-only binary: bash build_metal.sh cpu"
    exit 1
fi

# Compile to .air
echo "[1/2] Compiling to .air..."
metal -std=metal2.0 -o "$AIR" "$SRC"
echo "  → $AIR"

# Link to .metallib
echo "[2/2] Linking to .metallib..."
metallic -S "$AIR" -o "$METALLIB"
echo "  → $METALLIB"

# Clean up .air
rm -f "$AIR"

echo ""
echo "✅ Built successfully: $METALLIB"
echo "  Size: $(du -h "$METALLIB" | cut -f1)"
echo ""
echo "Usage in C++ code:"
echo "  Load SFA_Metal.metallib via MTLLibrary"
echo "  Dispatch kernels via MTLComputeCommandEncoder"
