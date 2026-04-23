# nn

[日本語](README.ja.md)

A neural network library written in Zig.

Comptime-shaped tensors, a unified autograd runtime, and GPU acceleration
(Metal/MPS on macOS, CUDA on Linux).

## Features

- **Tensor**: `Tensor(T, shape)` with comptime shape and f16/f32/f64 support
- **Unified runtime**: `nn.unified.*` exposes a single API that works over
  `DiffCpuRuntime` / `DiffMpsRuntime` / `DiffCudaRuntime` via comptime dispatch
  (no vtable, no runtime cost). Switch backends by `trainer(Model, .cpu|.cuda)`.
- **Autograd**: reverse-mode automatic differentiation
- **Layers**: Linear, Conv2D, MaxPool2D, LayerNorm, Dropout, Embedding,
  (Causal / Cross / Multi-head) Attention, Transformer encoder/decoder
- **Optimizers**: Adam with weight decay, LR schedules (cosine annealing,
  linear warmup, warmup-cosine)
- **GPU backends**:
  - **Metal / MPSGraph** (macOS, Apple Silicon)
  - **CUDA / cuBLAS** (Linux, opt-in via `-Dcuda=true`)
- **GGUF**: load and run quantized models (Q4_0, Q4_1, Q8_0)
- **Tokenizers**: BPE (GPT-2), SentencePiece (Gemma)
- **Checkpoints**: `save_checkpoint` / `load_checkpoint` for parameter I/O
- **Determinism**: fix every stochastic path (parameter init, dropout,
  `BatchIterator`) via `Trainer.Config{ .seed = 42 }`
- **Logging / Profiling**: scoped logger, stderr/stdout/profile-file sinks,
  controlled by env vars (see [docs/logging.md](docs/logging.md))

## Requirements

- [Zig 0.16.0](https://ziglang.org/download/)
- macOS: Xcode Command Line Tools (Metal / MPSGraph / Accelerate)
- Linux: OpenBLAS (`OPENBLAS_PATH`) for CPU, CUDA toolkit + cuBLAS
  (`CUDA_PATH`, `CUBLAS_PATH`) for the optional CUDA backend

The Nix flake wires all of the above up for you.

## Quick Start

### With Nix (recommended)

```bash
git clone https://github.com/soyukke/zig-nn.git nn
cd nn
nix develop   # or `direnv allow` / `direnv reload` if you use direnv

zig build test     # run tests
zig build xor      # run the XOR demo
```

### Without Nix

```bash
zig build test
zig build xor
```

> On Nix, prepend `NIX_CFLAGS_COMPILE=""` to `zig build` commands — the
> `justfile` already handles this.

## Examples

Every training example accepts an optional backend argument
(`cpu` default; `cuda` on Linux; `gemma3` additionally accepts `metal` and
`qlora`).

| Example | Command | Description | Code |
|---------|---------|-------------|------|
| XOR | `zig build xor` (or `-- cuda`) | Basic XOR classification | [examples/xor/main.zig](examples/xor/main.zig) |
| Spiral | `zig build spiral` | 2D spiral classification | [examples/spiral/main.zig](examples/spiral/main.zig) |
| CharLM | `zig build charlm` (or `-- cuda`) | Character-level Transformer LM | [examples/charlm/main.zig](examples/charlm/main.zig) |
| Diffusion | `zig build diffusion` | DDPM diffusion model | [examples/diffusion/main.zig](examples/diffusion/main.zig) |
| GPT-2 | `zig build gpt2 -Doptimize=ReleaseFast` | GPT-2 text generation (requires GGUF model) | [examples/gpt2/main.zig](examples/gpt2/main.zig) |
| Gemma 3 | `zig build gemma3 -Doptimize=ReleaseFast -- [cpu\|metal\|qlora]` | Gemma 3 1B inference / QLoRA fine-tuning | [examples/gemma3/main.zig](examples/gemma3/main.zig) |

### Downloading Models

```bash
# GPT-2 Q4_0
mkdir -p models
curl -L -o models/gpt2.gguf "https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/main/gpt2.Q4_0.gguf"
```

Gemma 3 examples expect `models/gemma3-1b.gguf`.

## Reproducibility

Fix the seed to make parameter init, dropout, and shuffling bitwise
deterministic across runs and backends:

```zig
var t = try nn.unified.trainer(Model, .cpu)
    .init(allocator, {}, .{ .lr = 1e-3, .seed = 42 });
```

`BatchIterator.init_with_seed(...)` offers the same knob for data shuffling.

## Logging / Profiling

`NN_LOG_LEVEL`, `NN_LOG_SCOPES`, `NN_PROFILE`, `NN_PROFILE_DIR` control
scoped logging and profile artifacts. See [docs/logging.md](docs/logging.md)
for the full spec.

```bash
NN_LOG_LEVEL=debug NN_LOG_SCOPES=metal,cuda zig build gemma3
```

## Benchmarks

PyTorch comparison scripts are in `benchmarks/`:

```bash
python benchmarks/bench_xor.py
python benchmarks/bench_spiral.py
python benchmarks/bench_charlm.py
python benchmarks/bench_ddpm.py
python benchmarks/bench_qlora_pytorch.py   # requires transformers, peft
bash   benchmarks/run_all.sh
```

## License

MIT
