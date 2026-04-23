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
nix develop   # or `direnv allow` / `direnv reload` if you use direnv

zig build test     # run tests
zig build xor      # run XOR demo
```

### Without Nix

Requires [Zig 0.16.0](https://ziglang.org/download/).

```bash
zig build test
zig build xor
```

## Examples

| Example | Command | Description | Code |
|---------|---------|-------------|------|
| XOR | `zig build xor` | Basic XOR classification | [examples/xor/main.zig](examples/xor/main.zig) |
| Spiral | `zig build spiral` | 2D spiral classification | [examples/spiral/main.zig](examples/spiral/main.zig) |
| CharLM | `zig build charlm` | Character-level language model | [examples/charlm/main.zig](examples/charlm/main.zig) |
| Diffusion | `zig build diffusion` | DDPM diffusion model | [examples/diffusion/main.zig](examples/diffusion/main.zig) |
| GPT-2 | `zig build gpt2 -Doptimize=ReleaseFast` | GPT-2 text generation (requires GGUF model) | [examples/gpt2/main.zig](examples/gpt2/main.zig) |
| Gemma 3 | `zig build gemma3 -Doptimize=ReleaseFast` | Gemma 3 1B text generation (requires GGUF model) | [examples/gemma3/main.zig](examples/gemma3/main.zig) |

### Downloading Models

```bash
# GPT-2 Q4_0
mkdir -p models
curl -L -o models/gpt2.gguf "https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.Q4_0.gguf"
```

## Logging / Profiling

`NN_LOG_LEVEL`, `NN_LOG_SCOPES`, `NN_PROFILE`, `NN_PROFILE_DIR` env vars
control scoped logging and profile artifacts. See
[docs/logging.md](docs/logging.md) for the full spec.

```bash
NN_LOG_LEVEL=debug NN_LOG_SCOPES=metal,cuda zig build gemma3
```

## Benchmarks

PyTorch comparison scripts are in `benchmarks/`:

```bash
python benchmarks/bench_ddpm.py          # DDPM training benchmark
python benchmarks/bench_qlora_pytorch.py # QLoRA benchmark (requires transformers, peft)
```

## License

MIT
