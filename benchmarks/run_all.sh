#!/bin/bash
# Run all benchmarks: zig-nn vs PyTorch (CPU & CUDA)
set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo " zig-nn vs PyTorch Benchmark"
echo "============================================"
echo ""

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/nix/store/5f5w8ysq8b350hfb3f9yanq2f3wjy2n4-gcc-14.3.0-lib/lib:${LD_LIBRARY_PATH}

# Auto-detect CUDA nix paths if not already set
if [ -z "$CUDA_PATH" ]; then
    CUDA_PATH=$(ls -d /nix/store/*cuda*cudart-12* 2>/dev/null | grep -v drv | grep -v source | head -1)
    export CUDA_PATH
fi
if [ -z "$CUBLAS_PATH" ]; then
    CUBLAS_PATH=$(ls -d /nix/store/*libcublas-*-lib 2>/dev/null | head -1)
    export CUBLAS_PATH
fi

# Check CUDA availability
CUDA_AVAILABLE=false
if [ -n "$CUDA_PATH" ] && [ -n "$CUBLAS_PATH" ]; then
    CUDA_AVAILABLE=true
    echo "CUDA_PATH=$CUDA_PATH"
    echo "CUBLAS_PATH=$CUBLAS_PATH"
    echo ""
fi

# Python: prefer CUDA-enabled venv, fall back to system python
PYTHON_CUDA="/tmp/pytorch-cuda-venv/bin/python3"
PYTHON_CPU="python"
if [ ! -x "$PYTHON_CUDA" ]; then
    PYTHON_CUDA="$PYTHON_CPU"
fi

for example in xor spiral charlm diffusion; do
    echo "########################################"
    echo "# ${example}"
    echo "########################################"
    echo ""

    if [ "$CUDA_AVAILABLE" = true ]; then
        echo "--- zig-nn CPU (ReleaseFast) ---"
        zig build ${example} -Dcuda=true -Doptimize=ReleaseFast -- cpu 2>&1
        echo ""
        echo "--- zig-nn CUDA (ReleaseFast) ---"
        zig build ${example} -Dcuda=true -Doptimize=ReleaseFast -- cuda 2>&1
        echo ""
    else
        echo "--- zig-nn CPU (ReleaseFast, CUDA skipped: paths not found) ---"
        zig build ${example} -Doptimize=ReleaseFast -- cpu 2>&1
        echo ""
    fi

    echo "--- PyTorch CPU ---"
    $PYTHON_CPU benchmarks/bench_${example}.py cpu
    echo ""

    echo "--- PyTorch CUDA ---"
    $PYTHON_CUDA benchmarks/bench_${example}.py cuda
    echo ""
done

echo "============================================"
echo " Done!"
echo "============================================"
