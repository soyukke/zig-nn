#!/bin/bash
# Zig DDPM batch size scaling benchmark
set -euo pipefail
cd "$(dirname "$0")/.."

for bs in 256 512 1024 2048; do
    echo "--- batch=$bs ---"
    NIX_CFLAGS_COMPILE="" zig build -Doptimize=ReleaseFast 2>/dev/null
    ./zig-out/bin/nn diffusion 2>&1 | grep -E "Training time"
done
