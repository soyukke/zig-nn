/// Zig Neural Network Library
///
/// コンパイル時型安全なテンソル演算と自動微分を提供する。
/// バックエンド: SIMD (ARM NEON / x86 AVX), CPU, Metal (macOS), CUDA (Linux)
///
/// ## API パターン
///
/// 本ライブラリには3つのAPIレイヤーがある:
///
/// **1. unified (推奨 / アクティブ)**
///   - `nn.unified.Linear`, `nn.unified.CausalSelfAttention` 等
///   - `DiffCpuRuntime` (CPU自動微分) / `MpsRuntime` (Metal GPU) で実行
///   - 型パラメータ不要: `Linear(in_dim, out_dim)` (T指定なし)
///   - 新規開発はこのAPIを使用すること
///
/// **2. Variable ベース (レガシー)**
///   - `nn.Linear`, `nn.MultiHeadAttention` 等 (nn/*.zig)
///   - `Variable(T, shape)` + `GradEngine` による動的計算グラフ
///   - 古いデモで使用。新規開発では非推奨
///
/// **3. graph_nn (レガシー / macOS only)**
///   - `nn.graph_nn.GraphLinear` 等 (Graph* prefix)
///   - MPSGraph 固有の抽象化。unified に統合済み

const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;
const is_linux = builtin.os.tag == .linux;

// Core
pub const Tensor = @import("core/tensor.zig").Tensor;
pub const shape = @import("core/shape.zig");
pub const GraphNode = @import("core/graph.zig").GraphNode;

// Backend
pub const backend = @import("backend/backend.zig");
pub const Backend = backend.Backend;
pub const BackendType = backend.BackendType;
pub const cpu = @import("backend/cpu.zig");
pub const simd = @import("backend/simd.zig");

// Metal (macOS only)
pub const metal = if (is_macos) @import("backend/metal.zig") else struct {};
pub const mps_graph = if (is_macos) @import("backend/mps_graph.zig") else struct {};

// CUDA (Linux only)
pub const cuda = if (is_linux) @import("backend/cuda.zig") else struct {};
pub const cuda_runtime = if (is_linux) @import("cuda_runtime.zig") else struct {};

// Autograd
pub const Variable = @import("autograd/variable.zig").Variable;
pub const GradEngine = @import("autograd/engine.zig").GradEngine;
pub const ops = @import("autograd/ops.zig");

// GPU Autograd (Metal Training - macOS only)
pub const GpuVariable = if (is_macos) @import("autograd/gpu_variable.zig").GpuVariable else struct {};
pub const gpu_ops = if (is_macos) @import("autograd/gpu_ops.zig") else struct {};
pub const GpuAdam = if (is_macos) @import("autograd/gpu_optimizer.zig").GpuAdam else struct {};

// Module mixin (autograd parameter management)
pub const Module = @import("nn/module.zig").Module;

// Neural Network Layers
pub const Linear = @import("nn/linear.zig").Linear;
pub const Conv2D = @import("nn/conv.zig").Conv2D;
pub const pool = @import("nn/pool.zig");
pub const normalization = @import("nn/normalization.zig");
pub const BatchNorm1d = normalization.BatchNorm1d;
pub const LayerNorm = normalization.LayerNorm;
pub const dropout = @import("nn/dropout.zig");
pub const Embedding = @import("nn/embedding.zig").Embedding;
pub const recurrent = @import("nn/recurrent.zig");
pub const LSTM = recurrent.LSTM;
pub const GRU = recurrent.GRU;
pub const attention = @import("nn/attention.zig");
pub const diffusion = @import("nn/diffusion.zig");
pub const MultiHeadAttention = attention.MultiHeadAttention;
pub const CausalSelfAttention = attention.CausalSelfAttention;
pub const CrossAttention = attention.CrossAttention;
pub const transformer = @import("nn/transformer.zig");
pub const TransformerEncoder = transformer.TransformerEncoder;
pub const TransformerDecoder = transformer.TransformerDecoder;
pub const TransformerEncoderBlock = transformer.TransformerEncoderBlock;
pub const TransformerDecoderBlock = transformer.TransformerDecoderBlock;

// Loss Functions
pub const loss = @import("loss/loss.zig");
pub const crossEntropyLoss = loss.crossEntropyLoss;
pub const bceLossWithLogits = loss.bceLossWithLogits;

// Optimizers
pub const optim_common = @import("optim/common.zig");
pub const SGD = @import("optim/sgd.zig").SGD;
pub const Adam = @import("optim/adam.zig").Adam;
pub const RMSProp = @import("optim/rmsprop.zig").RMSProp;

// GGUF Loader
pub const gguf = @import("gguf/gguf.zig");
pub const dequant = @import("gguf/dequant.zig");
pub const simd_dot = @import("gguf/simd_dot.zig");
pub const thread_pool = @import("gguf/thread_pool.zig");

// Models
pub const gpt2 = @import("models/gpt2.zig");
pub const gemma3 = @import("models/gemma3.zig");
pub const gemma3_metal = if (is_macos) @import("models/gemma3_metal.zig") else struct {};
pub const gemma3_qlora = if (is_macos) @import("models/gemma3_qlora.zig") else struct {};

// Unified compute (backend-agnostic)
pub const compute = @import("compute.zig");
pub const cpu_runtime = @import("cpu_runtime.zig");
pub const diff_cpu_runtime = @import("diff_cpu_runtime.zig");
pub const mps_runtime = if (is_macos) @import("mps_runtime.zig") else struct {};

// Unified NN layers (CPU/GPU backend-agnostic)
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
    pub const Trainer = @import("trainer.zig").Trainer;
    pub const MpsRuntime = if (is_macos) @import("mps_runtime.zig").MpsRuntime else struct {};
    pub const CpuRuntime = @import("cpu_runtime.zig").CpuRuntime;
    pub const CpuTensor = @import("cpu_runtime.zig").CpuTensor;
    pub const TensorInfo = @import("cpu_runtime.zig").TensorInfo;
    pub const DiffCpuRuntime = @import("diff_cpu_runtime.zig").DiffCpuRuntime;
    pub const DiffTensor = @import("diff_cpu_runtime.zig").DiffTensor;
    pub const BatchIterator = @import("data/dataloader.zig").BatchIterator;
    pub const Conv2d = @import("nn/graph_conv2d.zig").Conv2d;
    pub const MaxPool2d = @import("nn/graph_conv2d.zig").MaxPool2d;
};

// Graph NN layers (legacy aliases, MPSGraph backend - macOS only)
pub const graph_nn = if (is_macos) struct {
    pub const GraphModule = @import("nn/graph_module.zig").GraphModule;
    pub const GraphParam = @import("nn/graph_module.zig").GraphParam;
    pub const AdamState = @import("nn/graph_module.zig").AdamState;
    pub const saveCheckpoint = @import("nn/graph_module.zig").saveCheckpoint;
    pub const loadCheckpoint = @import("nn/graph_module.zig").loadCheckpoint;
    pub const GraphLinear = @import("nn/graph_linear.zig").GraphLinear;
    pub const GraphLayerNorm = @import("nn/graph_norm.zig").GraphLayerNorm;
    pub const GraphSelfAttention = @import("nn/graph_attention.zig").GraphSelfAttention;
    pub const GraphCausalSelfAttention = @import("nn/graph_attention.zig").GraphCausalSelfAttention;
    pub const GraphCrossAttention = @import("nn/graph_attention.zig").GraphCrossAttention;
    pub const GraphTransformerEncoderLayer = @import("nn/graph_transformer.zig").GraphTransformerEncoderLayer;
    pub const GraphTransformerDecoderLayer = @import("nn/graph_transformer.zig").GraphTransformerDecoderLayer;
} else struct {};

// Shared runtime computation kernels
pub const runtime_kernels = @import("runtime_kernels.zig");

// Data utilities
pub const BatchIterator = @import("data/dataloader.zig").BatchIterator;

// Tokenizer
pub const bpe = @import("tokenizer/bpe.zig");
pub const sentencepiece = @import("tokenizer/sentencepiece.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
