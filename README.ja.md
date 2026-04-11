# nn

[English](README.md)

Zig で書かれたニューラルネットワークライブラリ。

コンパイル時型安全なテンソル演算、自動微分、Apple Silicon 上の GPU アクセラレーション (Metal/MPS) を提供します。

## 特徴

- **Tensor**: `Tensor(T, shape)` - コンパイル時 shape、f16/f32/f64 対応
- **Autograd**: 逆伝播による自動微分
- **レイヤー**: Linear, Conv2D, Pool, LayerNorm, Dropout, Embedding, RNN, Attention, Transformer
- **オプティマイザ**: SGD, Adam, RMSProp
- **GPU**: Metal compute shaders / MPSGraph による学習・推論 (Apple Silicon)
- **GGUF**: 量子化モデルの読み込みと実行 (Q4_0, Q4_1, Q8_0)
- **トークナイザ**: BPE (GPT-2), SentencePiece (Gemma)

## クイックスタート

### Nix を使う場合 (推奨)

```bash
git clone <repository-url>
cd nn
nix develop   # direnv を使っている場合は `direnv allow`

zig build test     # テスト実行
zig build xor      # XOR デモ実行
```

### Nix なしの場合

[Zig 0.15](https://ziglang.org/download/) が必要です。

```bash
zig build test
zig build xor
```

## サンプル

| サンプル | コマンド | 説明 |
|---------|---------|------|
| XOR | `zig build xor` | XOR 分類 |
| Spiral | `zig build spiral` | 2D スパイラル分類 |
| CharLM | `zig build charlm` | 文字レベル言語モデル |
| Diffusion | `zig build diffusion` | DDPM 拡散モデル |
| GPT-2 | `zig build gpt2 -Doptimize=ReleaseFast` | GPT-2 テキスト生成 (GGUF モデルが必要) |
| Gemma 3 | `zig build gemma3 -Doptimize=ReleaseFast` | Gemma 3 1B テキスト生成 (GGUF モデルが必要) |

### モデルのダウンロード

```bash
# GPT-2 Q4_0
mkdir -p models
curl -L -o models/gpt2.gguf "https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.Q4_0.gguf"
```

## ベンチマーク

PyTorch との比較スクリプトは `benchmarks/` にあります:

```bash
python benchmarks/bench_ddpm.py          # DDPM 学習ベンチマーク
python benchmarks/bench_qlora_pytorch.py # QLoRA ベンチマーク (transformers, peft が必要)
```

## ライセンス

MIT
