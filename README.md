# nn

[日本語](README.ja.md)

A neural network library written in Zig.

Comptime-shaped tensors, autograd, and GPU acceleration (Metal/MPS) on Apple Silicon.

## Features

- **Tensor**: `Tensor(T, shape)` with comptime shape and f16/f32/f64 support
- **Autograd**: reverse-mode automatic differentiation
- **Layers**: Linear, Conv2D, Pool, LayerNorm, Dropout, Embedding, RNN, Attention, Transformer
- **Optimizers**: SGD, Adam, RMSProp
- **GPU**: Metal compute shaders and MPSGraph for training/inference on Apple Silicon
- **GGUF**: load and run quantized models (Q4_0, Q4_1, Q8_0)
- **Tokenizers**: BPE (GPT-2), SentencePiece (Gemma)

## Quick Start

### With Nix (recommended)

```bash
git clone <repository-url>
cd nn
nix develop   # or `direnv allow` if you use direnv

zig build test     # run tests
zig build xor      # run XOR demo
```

### Without Nix

Requires [Zig 0.15](https://ziglang.org/download/).

```bash
zig build test
zig build xor
```

## Examples

| Example | Command | Description |
|---------|---------|-------------|
| XOR | `zig build xor` | Basic XOR classification |
| Spiral | `zig build spiral` | 2D spiral classification |
| CharLM | `zig build charlm` | Character-level language model |
| Diffusion | `zig build diffusion` | DDPM diffusion model |
| GPT-2 | `zig build gpt2 -Doptimize=ReleaseFast` | GPT-2 text generation (requires GGUF model) |
| Gemma 3 | `zig build gemma3 -Doptimize=ReleaseFast` | Gemma 3 1B text generation (requires GGUF model) |

### Downloading Models

```bash
# GPT-2 Q4_0
mkdir -p models
curl -L -o models/gpt2.gguf "https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.Q4_0.gguf"
```

## Benchmarks

PyTorch comparison scripts are in `benchmarks/`:

```bash
python benchmarks/bench_ddpm.py          # DDPM training benchmark
python benchmarks/bench_qlora_pytorch.py # QLoRA benchmark (requires transformers, peft)
```

## License

MIT
