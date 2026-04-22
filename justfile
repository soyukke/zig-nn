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

# --- Zig style checker (installed from ~/dotfiles/zig-tools) ---
#
# `just lint` は src/ の現状を scripts/style_baseline.txt と比較し、
# 増えた違反だけで失敗する (ratcheting baseline)。既存違反を直した後は
# `just lint-update-baseline` で baseline を更新する。
# ZIG_STYLE_CHECKER 環境変数で checker 本体のパスを差し替え可能。

style_checker := env("ZIG_STYLE_CHECKER", env("HOME") + "/dotfiles/zig-tools/check_style.zig")

# zig fmt でフォーマット崩れを検出 (書き換えはしない)
fmt-check:
    NIX_CFLAGS_COMPILE="" zig fmt --check src

# zig fmt でフォーマットを書き換え
fmt:
    NIX_CFLAGS_COMPILE="" zig fmt src

# baseline と比較して新規違反があれば fail
lint:
    NIX_CFLAGS_COMPILE="" zig run {{style_checker}} -- --root src

# baseline を無視して全違反を列挙
lint-strict:
    NIX_CFLAGS_COMPILE="" zig run {{style_checker}} -- --root src --strict

# baseline を現在の違反で再スナップショット
lint-update-baseline:
    NIX_CFLAGS_COMPILE="" zig run {{style_checker}} -- --root src --update-baseline

# fmt-check + lint をまとめて実行 (PR 前の最低限チェック)
check: fmt-check lint
# --- end zig-tools linter ---
