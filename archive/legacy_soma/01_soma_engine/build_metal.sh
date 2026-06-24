#!/bin/bash
set -e

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/SFA_Metal.cpp"

case "${1:-metal}" in
    metal)
        echo "Building with Metal GPU acceleration..."
        clang++ -std=c++17 -O3 -fobjc-arc "${SRC}" \
            -framework Metal -framework Foundation \
            -o "$(dirname "${SRC}")/soma_metal"
        echo "Built: $(dirname "${SRC}")/soma_metal"
        ;;
    cpu)
        echo "Building CPU-only..."
        clang++ -std=c++17 -O3 "${SRC}" \
            -o "$(dirname "${SRC}")/soma_metal"
        echo "Built: $(dirname "${SRC}")/soma_metal"
        ;;
    clean)
        rm -f "$(dirname "${SRC}")/soma_metal"
        echo "Cleaned."
        ;;
esac
