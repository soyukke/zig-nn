/// diff/cuda_runtime_test.zig: DiffCudaRuntime の数値勾配テスト
///
/// 共通テスト基盤 (test_helpers.zig) を CUDA Adapter 経由で使用する。
/// GPU 上で forward/backward を実行し、結果を CPU にコピーして数値勾配と比較する。
///
/// ビルド: zig build test-diff-cuda -Dcuda=true
const std = @import("std");
pub const std_options = @import("../log.zig").std_options;
const compute = @import("../compute.zig");
const Module = compute.Module;
const diff_cuda = @import("cuda_runtime.zig");
const DiffCudaRuntime = diff_cuda.DiffCudaRuntime;
const DiffCudaTensor = diff_cuda.DiffCudaTensor;
const cuda = @import("../backend/cuda.zig");
const CudaContext = cuda.CudaContext;
const helpers = @import("test_helpers.zig");

const testing = std.testing;

// ════════════════════════════════════════════════════════════════
// CUDA Adapter: GPU ↔ CPU データ転送を介してアクセス
// ════════════════════════════════════════════════════════════════
const CudaAdapter = struct {
    pub const Runtime = DiffCudaRuntime;
    pub const Tensor = DiffCudaTensor;

    /// ホストデータを GPU にコピーしてノードを作成
    pub fn make_input(rt: *Runtime, data: []f32, shape: []const usize, requires_grad: bool) Tensor {
        const node = rt.make_tensor(data, shape);
        node.requires_grad = requires_grad;
        return node;
    }

    /// GPU テンソルデータをホストバッファにコピー
    pub fn read_data(rt: *Runtime, tensor: Tensor, dst: []f32) void {
        rt.copy_to_host(tensor, dst);
    }

    /// GPU 上の勾配をホストバッファにコピー
    pub fn read_grad(rt: *Runtime, tensor: Tensor, dst: []f32) ?[]f32 {
        if (tensor.grad) |grad_dptr| {
            const total = tensor.total_elements();
            rt.cuda_ctx.copy_device_to_host(
                @ptrCast(dst.ptr),
                grad_dptr,
                total * @sizeOf(f32),
            ) catch unreachable;
            return dst;
        }
        return null;
    }
};

/// テスト全体で共有する CudaContext (1回だけ初期化)
var global_cuda_ctx: ?CudaContext = null;

fn get_or_init_cuda_ctx() !*CudaContext {
    if (global_cuda_ctx) |*ctx| return ctx;
    global_cuda_ctx = try CudaContext.init(0);
    return &global_cuda_ctx.?;
}

// ════════════════════════════════════════════════════════════════
// パラメータ不要な単項演算テスト
// ════════════════════════════════════════════════════════════════

test "diff_cuda_runtime: negative gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_negative_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: gelu gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_gelu_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: silu gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_silu_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: square gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_square_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: reductionMean gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_reduction_mean_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: tanh gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_tanh_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: sigmoid gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_sigmoid_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: softmax gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_softmax_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: relu gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_relu_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: reductionSum gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_reduction_sum_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: transpose 3D gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_transpose3_d_grad(CudaAdapter, &rt);
}

test "diff_cuda_runtime: reshape gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    try helpers.test_reshape_grad(CudaAdapter, &rt);
}

// ════════════════════════════════════════════════════════════════
// パラメータ依存テスト
// ════════════════════════════════════════════════════════════════
const check_grad = helpers.gradient_checker(CudaAdapter).check_grad;

test "diff_cuda_runtime: matmul 2D gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    _ = module.add_param(&.{ 3, 2 }, .xavier);
    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    rt.init_params();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try check_grad(struct {
        fn f(ctx: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
            const w = ctx.param(.{ .index = 0 });
            return ctx.matmul(x, w);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cuda_runtime: layerNorm gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    _ = module.add_param(&.{4}, .ones);
    _ = module.add_param(&.{4}, .zeros);
    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    rt.init_params();

    var data = [_]f32{ 0.5, 1.3, 2.7, 0.1, 3.2, 0.8, 1.5, 2.1 };
    try check_grad(struct {
        fn f(ctx: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
            const gamma = ctx.param(.{ .index = 0 });
            const beta = ctx.param(.{ .index = 1 });
            return ctx.gelu(ctx.layer_norm(x, gamma, beta, 1e-5, -1));
        }
    }.f, &rt, &data, &.{ 2, 4 }, 1e-4, 2e-2);
}

test "diff_cuda_runtime: add broadcast gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    _ = module.add_param(&.{3}, .xavier);
    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    rt.init_params();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try check_grad(struct {
        fn f(ctx: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
            const b = ctx.param(.{ .index = 0 });
            return ctx.add(x, b);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cuda_runtime: linear forward+backward" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    _ = module.add_param(&.{ 3, 2 }, .xavier);
    _ = module.add_param(&.{2}, .zeros);
    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    rt.init_params();

    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input_node = CudaAdapter.make_input(&rt, &input_data, &.{ 2, 3 }, true);

    const w = rt.param(.{ .index = 0 });
    const b = rt.param(.{ .index = 1 });
    const xw = rt.matmul(input_node, w);
    const y = rt.add(xw, b);

    const loss_node = rt.reduction_sum(y, -1);
    const total_loss = rt.reduction_sum(loss_node, 0);

    rt.backward(total_loss);

    // パラメータ勾配が非ゼロであることを確認
    const w_grad = rt.param_grad(0);
    const b_grad = rt.param_grad(1);
    var w_norm: f32 = 0;
    for (w_grad) |g| w_norm += g * g;
    var b_norm: f32 = 0;
    for (b_grad) |g| b_norm += g * g;

    try testing.expect(w_norm > 0);
    try testing.expect(b_norm > 0);
}

// ════════════════════════════════════════════════════════════════
// 損失関数テスト
// ════════════════════════════════════════════════════════════════

test "diff_cuda_runtime: mseLoss gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    var pred_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const target_data = [_]f32{ 1.5, 2.5, 2.5, 3.5 };

    rt.reset_arena();
    const pred = CudaAdapter.make_input(&rt, &pred_data, &.{ 2, 2 }, true);
    const loss = rt.mse_loss(pred, &target_data);

    rt.backward(loss);

    var ga: [4]f32 = undefined;
    _ = CudaAdapter.read_grad(&rt, pred, &ga) orelse unreachable;
    const n: f32 = 4.0;
    for (0..4) |i| {
        const expected = 2.0 * (pred_data[i] - target_data[i]) / n;
        try testing.expectApproxEqAbs(expected, ga[i], 1e-5);
    }
}

test "diff_cuda_runtime: crossEntropyLossWithIndices gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    var logits_data = [_]f32{ 1.0, 2.0, 3.0, 1.0, 3.0, 2.0 };
    const indices = [_]u32{ 0, 2 };

    rt.reset_arena();
    const logits = CudaAdapter.make_input(&rt, &logits_data, &.{ 2, 3 }, true);
    const loss = rt.cross_entropy_loss_with_indices(logits, &indices);

    rt.backward(loss);

    var ga: [6]f32 = undefined;
    _ = CudaAdapter.read_grad(&rt, logits, &ga) orelse unreachable;
    for (0..2) |i| {
        var row_sum: f32 = 0;
        for (0..3) |j| row_sum += ga[i * 3 + j];
        try testing.expectApproxEqAbs(@as(f32, 0.0), row_sum, 1e-5);
    }
}

test "diff_cuda_runtime: bceLossWithLogits gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    var logits_data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const target_data = [_]f32{ 0.0, 1.0, 1.0, 0.0 };

    rt.reset_arena();
    const logits = CudaAdapter.make_input(&rt, &logits_data, &.{ 2, 2 }, true);
    const loss = rt.bce_loss_with_logits(logits, &target_data);

    rt.backward(loss);

    var ga: [4]f32 = undefined;
    _ = CudaAdapter.read_grad(&rt, logits, &ga) orelse unreachable;
    const n: f32 = 4.0;
    for (0..4) |i| {
        const sig = 1.0 / (1.0 + @exp(-logits_data[i]));
        const expected = (sig - target_data[i]) / n;
        try testing.expectApproxEqAbs(expected, ga[i], 1e-5);
    }
}

test "diff_cuda_runtime: gather gradient" {
    const cuda_ctx = try get_or_init_cuda_ctx();
    var module = Module.init(testing.allocator);
    defer module.deinit();

    _ = module.add_param(&.{ 4, 3 }, .xavier);
    var rt = try DiffCudaRuntime.init(&module, cuda_ctx, testing.allocator);
    defer rt.deinit();

    rt.init_params();

    const indices = [_]u32{ 1, 3, 0, 1 };

    rt.reset_arena();
    rt.zero_grad();
    const table = rt.param(.{ .index = 0 });
    const embedded = rt.gather(table, &indices);

    const loss = rt.reduction_sum(rt.reduction_sum(embedded, -1), 0);
    rt.backward(loss);

    const tg = rt.param_grad(0);
    // Row 1 は index 0, 3 で2回参照 → grad=2.0
    for (0..3) |j| {
        try testing.expectApproxEqAbs(@as(f32, 2.0), tg[1 * 3 + j], 1e-5);
    }
    // Row 2 は未参照 → grad=0.0
    for (0..3) |j| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), tg[2 * 3 + j], 1e-5);
    }
}
