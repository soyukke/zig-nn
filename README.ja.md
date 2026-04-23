# nn

[English](README.md)

Zig で書かれたニューラルネットワークライブラリ。

コンパイル時型安全なテンソル演算、統一された autograd ランタイム、
GPU アクセラレーション (macOS: Metal/MPS, Linux: CUDA) を提供します。

## 特徴

- **Tensor**: `Tensor(T, shape)` - コンパイル時 shape、f16/f32/f64 対応
- **統一ランタイム**: `nn.unified.*` が唯一の公開 API。
  `DiffCpuRuntime` / `DiffMpsRuntime` / `DiffCudaRuntime` を comptime で
  ディスパッチし (vtable・実行時コストなし)、
  `trainer(Model, .cpu|.cuda)` でバックエンドを切り替え可能。
- **Autograd**: 逆伝播による自動微分
- **レイヤー**: Linear, Conv2D, MaxPool2D, LayerNorm, Dropout,
  Embedding, (Causal / Cross / Multi-head) Attention,
  Transformer encoder/decoder
- **オプティマイザ**: weight decay 付き Adam、LR スケジューラ
  (cosine annealing / linear warmup / warmup-cosine)
- **GPU バックエンド**:
  - **Metal / MPSGraph** (macOS Apple Silicon)
  - **CUDA / cuBLAS** (Linux, `-Dcuda=true` で opt-in)
- **GGUF**: 量子化モデルの読み込みと実行 (Q4_0, Q4_1, Q8_0)
- **トークナイザ**: BPE (GPT-2), SentencePiece (Gemma)
- **チェックポイント**: `save_checkpoint` / `load_checkpoint` で
  パラメータ保存/復元
- **決定論性**: `Trainer.Config{ .seed = 42 }` で
  パラメータ初期化・dropout・`BatchIterator` を bitwise 固定
- **ロギング / プロファイル**: scoped logger と
  stderr / stdout / profile-file の 3 sink 分離。環境変数で制御
  (詳細は [docs/logging.md](docs/logging.md))

## 必要環境

- [Zig 0.16.0](https://ziglang.org/download/)
- macOS: Xcode Command Line Tools (Metal / MPSGraph / Accelerate)
- Linux: OpenBLAS (`OPENBLAS_PATH`)、CUDA 利用時は
  CUDA toolkit + cuBLAS (`CUDA_PATH`, `CUBLAS_PATH`)

Nix flake を使えば上記は自動で揃います。

## クイックスタート

### Nix を使う場合 (推奨)

```bash
git clone https://github.com/soyukke/zig-nn.git nn
cd nn
nix develop   # direnv を使っている場合は `direnv allow` / `direnv reload`

zig build test     # テスト実行
zig build xor      # XOR デモ
```

### Nix なしの場合

```bash
zig build test
zig build xor
```

> Nix 環境では `zig build` の前に `NIX_CFLAGS_COMPILE=""` を付ける必要が
> あります (`justfile` には設定済み)。

## サンプル

各学習サンプルはバックエンド引数を受け取れます
(デフォルト `cpu`、Linux では `cuda`、
`gemma3` はさらに `metal` / `qlora` を追加)。

| サンプル | コマンド | 説明 | コード |
|---------|---------|------|--------|
| XOR | `zig build xor` (または `-- cuda`) | XOR 分類 | [examples/xor/main.zig](examples/xor/main.zig) |
| Spiral | `zig build spiral` | 2D スパイラル分類 | [examples/spiral/main.zig](examples/spiral/main.zig) |
| CharLM | `zig build charlm` (または `-- cuda`) | 文字レベル Transformer LM | [examples/charlm/main.zig](examples/charlm/main.zig) |
| Diffusion | `zig build diffusion` | DDPM 拡散モデル | [examples/diffusion/main.zig](examples/diffusion/main.zig) |
| GPT-2 | `zig build gpt2 -Doptimize=ReleaseFast` | GPT-2 テキスト生成 (GGUF モデル必要) | [examples/gpt2/main.zig](examples/gpt2/main.zig) |
| Gemma 3 | `zig build gemma3 -Doptimize=ReleaseFast -- [cpu\|metal\|qlora]` | Gemma 3 1B 推論 / QLoRA ファインチューニング | [examples/gemma3/main.zig](examples/gemma3/main.zig) |

### モデルのダウンロード

```bash
# GPT-2 Q4_0
mkdir -p models
curl -L -o models/gpt2.gguf "https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.Q4_0.gguf"
```

Gemma 3 サンプルは `models/gemma3-1b.gguf` を想定しています。

## 再現性 (Seed 固定)

`seed` を指定すると、パラメータ初期化・dropout・データシャッフルが
実行間・バックエンド間で bitwise に一致します:

```zig
var t = try nn.unified.trainer(Model, .cpu)
    .init(allocator, {}, .{ .lr = 1e-3, .seed = 42 });
```

データ側は `BatchIterator.init_with_seed(...)` を利用できます。

## ログ / プロファイル

`NN_LOG_LEVEL`, `NN_LOG_SCOPES`, `NN_PROFILE`, `NN_PROFILE_DIR` の
環境変数で scoped logger と profile artifact を制御します。詳細は
[docs/logging.md](docs/logging.md) を参照。

```bash
NN_LOG_LEVEL=debug NN_LOG_SCOPES=metal,cuda zig build gemma3
```

## ベンチマーク

PyTorch との比較スクリプトは `benchmarks/` にあります:

```bash
python benchmarks/bench_xor.py
python benchmarks/bench_spiral.py
python benchmarks/bench_charlm.py
python benchmarks/bench_ddpm.py
python benchmarks/bench_qlora_pytorch.py   # transformers, peft が必要
bash   benchmarks/run_all.sh
```

## ライセンス

MIT
