# Zig NN Library

# デフォルト: ヘルプ表示
default:
    @just --list

# ビルド (デバッグ)
build:
    zig build

# ビルド (リリース)
build-release:
    zig build -Doptimize=ReleaseFast

# テスト実行
test:
    zig build test

# CPU 微分ランタイムテスト
test-diff-cpu:
    NIX_CFLAGS_COMPILE="" zig build test-diff-cpu

# Metal 微分ランタイムテスト (macOS only)
test-diff-mps:
    NIX_CFLAGS_COMPILE="" zig build test-diff-mps

# CUDA 微分ランタイムテスト (Linux only, requires GPU)
test-diff-cuda:
    zig build test-diff-cuda -Dcuda=true

# 全デモ実行
run-all:
    zig build run

# XOR デモ
xor:
    zig build run -- xor

# Spiral 分類デモ
spiral:
    zig build run -- spiral

# 文字レベル言語モデルデモ
charlm:
    zig build run -- charlm

# GPT-2 テキスト生成 (リリース, 高速)
gpt2:
    zig build run -Doptimize=ReleaseFast -- gpt2

# Gemma 3 1B テキスト生成 (リリース, 高速)
gemma3:
    zig build run -Doptimize=ReleaseFast -- gemma3

# GPT-2 モデルをダウンロード
download-model:
    mkdir -p models
    curl -L -o models/gpt2.gguf "https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.Q4_0.gguf"

# クリーン
clean:
    rm -rf .zig-cache zig-out
