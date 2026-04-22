/// Zig Neural Network Library
///
/// コンパイル時型安全なテンソル演算と自動微分を提供する。
/// バックエンド: SIMD (ARM NEON / x86 AVX), CPU, Metal (macOS), CUDA (Linux)
///
/// ## API
///
/// `nn.unified.*` が唯一の公開 API。
/// `DiffCpuRuntime` / `DiffCudaRuntime` / `DiffMpsRuntime` の duck-typed ops を
/// `trainer(Model, .cpu|.cuda)` で統一的に利用できる。
const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;
const is_linux = builtin.os.tag == .linux;

// Logging infrastructure (scoped loggers + std_options)
pub const log = @import("log.zig");
pub const std_options = log.std_options;

// Backend primitives
const backend_mod = @import("backend/backend.zig");
pub const backend = backend_mod.backend;
pub const BackendType = backend_mod.BackendType;
pub const cpu = @import("backend/cpu.zig");
pub const simd = @import("backend/simd.zig");
pub const metal = if (is_macos) @import("backend/metal.zig") else struct {};
pub const cuda = if (is_linux) @import("backend/cuda.zig") else struct {};

// Shared runtime infrastructure
pub const compute = @import("compute.zig");
pub const diff_node = @import("diff/node.zig");
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
    pub const adam_step = compute.adam_step;
    pub const cosine_annealing_lr = compute.cosine_annealing_lr;
    pub const linear_warmup_lr = compute.linear_warmup_lr;
    pub const warmup_cosine_decay_lr = compute.warmup_cosine_decay_lr;
    pub const save_checkpoint = compute.save_checkpoint;
    pub const load_checkpoint = compute.load_checkpoint;

    // Layers
    pub const linear = @import("nn/graph_linear.zig").linear;
    pub const layer_norm = @import("nn/graph_norm.zig").layer_norm;
    pub const self_attention = @import("nn/graph_attention.zig").self_attention;
    pub const multi_head_self_attention =
        @import("nn/graph_attention.zig").multi_head_self_attention;
    pub const causal_self_attention = @import("nn/graph_attention.zig").causal_self_attention;
    pub const multi_head_causal_self_attention =
        @import("nn/graph_attention.zig").multi_head_causal_self_attention;
    pub const cross_attention = @import("nn/graph_attention.zig").cross_attention;
    pub const multi_head_cross_attention =
        @import("nn/graph_attention.zig").multi_head_cross_attention;
    pub const dynamic_causal_self_attention =
        @import("nn/graph_attention.zig").dynamic_causal_self_attention;
    pub const transformer_encoder_layer =
        @import("nn/graph_transformer.zig").transformer_encoder_layer;
    pub const transformer_decoder_layer =
        @import("nn/graph_transformer.zig").transformer_decoder_layer;
    pub const embedding = @import("nn/graph_embedding.zig").embedding;
    pub const dropout = @import("nn/graph_dropout.zig").dropout;
    pub const sequential = @import("nn/graph_sequential.zig").sequential;
    pub const ReLU = @import("nn/graph_sequential.zig").ReLU;
    pub const GELU = @import("nn/graph_sequential.zig").GELU;
    pub const SiLU = @import("nn/graph_sequential.zig").SiLU;
    pub const Sigmoid = @import("nn/graph_sequential.zig").Sigmoid;
    pub const Tanh = @import("nn/graph_sequential.zig").Tanh;
    pub const conv2d = @import("nn/graph_conv2d.zig").conv2d;
    pub const max_pool2d = @import("nn/graph_conv2d.zig").max_pool2d;

    // Trainer / Runtimes
    pub const Device = @import("trainer.zig").Device;
    pub const trainer = @import("trainer.zig").trainer;
    pub const DiffCpuRuntime = @import("diff/cpu_runtime.zig").DiffCpuRuntime;
    pub const DiffTensor = @import("diff/cpu_runtime.zig").DiffTensor;
    pub const DiffMpsRuntime = if (is_macos)
        @import("diff/mps_runtime.zig").DiffMpsRuntime
    else
        struct {};
    pub const DiffMpsTensor = if (is_macos)
        @import("diff/mps_runtime.zig").DiffMpsTensor
    else
        struct {};
    pub const DiffCudaRuntime = if (is_linux)
        @import("diff/cuda_runtime.zig").DiffCudaRuntime
    else
        struct {};
    pub const DiffCudaTensor = if (is_linux)
        @import("diff/cuda_runtime.zig").DiffCudaTensor
    else
        struct {};
    pub const GpuAdamState = if (is_linux)
        @import("diff/cuda_runtime.zig").GpuAdamState
    else
        struct {};
};

// Diffusion utilities (pure helpers, used by examples/diffusion and seqdiffuseq)
pub const diffusion = @import("nn/diffusion.zig");

// Timer shim (std.time.Timer が Zig 0.16 で削除されたため)
pub const Timer = @import("util/timer.zig").Timer;
pub const now_nanos = @import("util/timer.zig").now_nanos;

test {
    @import("std").testing.refAllDecls(@This());
}
