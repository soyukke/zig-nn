/// Zig Neural Network Library
///
/// コンパイル時型安全なテンソル演算と自動微分を提供する。
/// バックエンド: SIMD (ARM NEON / x86 AVX), CPU, Metal (macOS), CUDA (Linux)
///
/// ## API
///
/// `nn.unified.*` が唯一の公開 API。
/// `DiffCpuRuntime` / `DiffCudaRuntime` / `DiffMpsRuntime` の duck-typed ops を
/// `Trainer(Model, .cpu|.cuda)` で統一的に利用できる。

const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;
const is_linux = builtin.os.tag == .linux;

// Logging infrastructure (scoped loggers + std_options)
pub const log = @import("log.zig");
pub const std_options = log.std_options;

// Backend primitives
pub const backend = @import("backend/backend.zig");
pub const Backend = backend.Backend;
pub const BackendType = backend.BackendType;
pub const cpu = @import("backend/cpu.zig");
pub const simd = @import("backend/simd.zig");
pub const metal = if (is_macos) @import("backend/metal.zig") else struct {};
pub const cuda = if (is_linux) @import("backend/cuda.zig") else struct {};

// Shared runtime infrastructure
pub const compute = @import("compute.zig");
pub const diff_node = @import("diff_node.zig");
pub const runtime_kernels = @import("runtime_kernels.zig");

// GGUF / tokenizer / data utilities
pub const gguf = @import("gguf/gguf.zig");
pub const dequant = @import("gguf/dequant.zig");
pub const simd_dot = @import("gguf/simd_dot.zig");
pub const thread_pool = @import("gguf/thread_pool.zig");
pub const bpe = @import("tokenizer/bpe.zig");
pub const sentencepiece = @import("tokenizer/sentencepiece.zig");
pub const BatchIterator = @import("data/dataloader.zig").BatchIterator;

// Inference-only models (GGUF loader based)
pub const gpt2 = @import("models/gpt2.zig");
pub const gemma3 = @import("models/gemma3.zig");
pub const gemma3_metal = if (is_macos) @import("models/gemma3_metal.zig") else struct {};
pub const gemma3_qlora = if (is_macos) @import("models/gemma3_qlora.zig") else struct {};

// Unified NN layers + runtimes (CPU/CUDA/MPS backend-agnostic)
pub const unified = struct {
    pub const Module = compute.Module;
    pub const ParamHandle = compute.ParamHandle;
    pub const ParamInit = compute.ParamInit;
    pub const AdamState = compute.AdamState;
    pub const adamStep = compute.adamStep;
    pub const cosineAnnealingLR = compute.cosineAnnealingLR;
    pub const linearWarmupLR = compute.linearWarmupLR;
    pub const warmupCosineDecayLR = compute.warmupCosineDecayLR;
    pub const saveCheckpoint = compute.saveCheckpoint;
    pub const loadCheckpoint = compute.loadCheckpoint;

    // Layers
    pub const Linear = @import("nn/graph_linear.zig").Linear;
    pub const LayerNorm = @import("nn/graph_norm.zig").LayerNorm;
    pub const SelfAttention = @import("nn/graph_attention.zig").SelfAttention;
    pub const MultiHeadSelfAttention = @import("nn/graph_attention.zig").MultiHeadSelfAttention;
    pub const CausalSelfAttention = @import("nn/graph_attention.zig").CausalSelfAttention;
    pub const MultiHeadCausalSelfAttention = @import("nn/graph_attention.zig").MultiHeadCausalSelfAttention;
    pub const CrossAttention = @import("nn/graph_attention.zig").CrossAttention;
    pub const MultiHeadCrossAttention = @import("nn/graph_attention.zig").MultiHeadCrossAttention;
    pub const DynamicCausalSelfAttention = @import("nn/graph_attention.zig").DynamicCausalSelfAttention;
    pub const TransformerEncoderLayer = @import("nn/graph_transformer.zig").TransformerEncoderLayer;
    pub const TransformerDecoderLayer = @import("nn/graph_transformer.zig").TransformerDecoderLayer;
    pub const Embedding = @import("nn/graph_embedding.zig").Embedding;
    pub const Dropout = @import("nn/graph_dropout.zig").Dropout;
    pub const Sequential = @import("nn/graph_sequential.zig").Sequential;
    pub const ReLU = @import("nn/graph_sequential.zig").ReLU;
    pub const GELU = @import("nn/graph_sequential.zig").GELU;
    pub const SiLU = @import("nn/graph_sequential.zig").SiLU;
    pub const Sigmoid = @import("nn/graph_sequential.zig").Sigmoid;
    pub const Tanh = @import("nn/graph_sequential.zig").Tanh;
    pub const Conv2d = @import("nn/graph_conv2d.zig").Conv2d;
    pub const MaxPool2d = @import("nn/graph_conv2d.zig").MaxPool2d;

    // Trainer / Runtimes
    pub const Device = @import("trainer.zig").Device;
    pub const Trainer = @import("trainer.zig").Trainer;
    pub const DiffCpuRuntime = @import("diff_cpu_runtime.zig").DiffCpuRuntime;
    pub const DiffTensor = @import("diff_cpu_runtime.zig").DiffTensor;
    pub const DiffMpsRuntime = if (is_macos) @import("diff_mps_runtime.zig").DiffMpsRuntime else struct {};
    pub const DiffMpsTensor = if (is_macos) @import("diff_mps_runtime.zig").DiffMpsTensor else struct {};
    pub const DiffCudaRuntime = if (is_linux) @import("diff_cuda_runtime.zig").DiffCudaRuntime else struct {};
    pub const DiffCudaTensor = if (is_linux) @import("diff_cuda_runtime.zig").DiffCudaTensor else struct {};
    pub const GpuAdamState = if (is_linux) @import("diff_cuda_runtime.zig").GpuAdamState else struct {};
};

// Diffusion utilities (pure helpers, used by examples/diffusion and seqdiffuseq)
pub const diffusion = @import("nn/diffusion.zig");

// Timer shim (std.time.Timer が Zig 0.16 で削除されたため)
pub const Timer = @import("util/timer.zig").Timer;
pub const nowNanos = @import("util/timer.zig").nowNanos;

test {
    @import("std").testing.refAllDecls(@This());
}
