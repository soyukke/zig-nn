/// diff_mps_runtime_test.zig: DiffMpsRuntime の数値勾配テスト
///
/// 共通テスト基盤 (diff_runtime_test_helpers.zig) を MPS Adapter 経由で使用する。
/// Metal compute カーネルの forward 正当性を数値勾配チェックで検証する。
///
/// ビルド: zig build test-diff-mps (macOS only)
const std = @import("std");
const compute = @import("compute.zig");
const Module = compute.Module;
const diff_mps = @import("diff_mps_runtime.zig");
const DiffMpsRuntime = diff_mps.DiffMpsRuntime;
const DiffMpsTensor = diff_mps.DiffMpsTensor;
const metal = @import("backend/metal.zig");
const MetalContext = metal.MetalContext;
const helpers = @import("diff_runtime_test_helpers.zig");

const testing = std.testing;

// ════════════════════════════════════════════════════════════════
// MPS Adapter: UMA 経由でデータにアクセス
// ════════════════════════════════════════════════════════════════
const MpsAdapter = struct {
    pub const Runtime = DiffMpsRuntime;
    pub const Tensor = DiffMpsTensor;

    /// ホストデータを MTLBuffer にコピーしてノードを作成
    pub fn makeInput(rt: *Runtime, data: []f32, shape: []const usize, requires_grad: bool) Tensor {
        const node = rt.makeTensor(data, shape);
        node.requires_grad = requires_grad;
        return node;
    }

    /// UMA: bufferContents で直接読み取り
    pub fn readData(_: *Runtime, tensor: Tensor, dst: []f32) void {
        @memcpy(dst, MetalContext.bufferContents(f32, tensor.data)[0..dst.len]);
    }

    /// UMA: 勾配バッファを直接読み取り
    pub fn readGrad(_: *Runtime, tensor: Tensor, dst: []f32) ?[]f32 {
        if (tensor.grad) |grad_buf| {
            @memcpy(dst, MetalContext.bufferContents(f32, grad_buf)[0..dst.len]);
            return dst;
        }
        return null;
    }
};

/// テスト全体で共有する MetalContext (1回だけ初期化)
var global_metal_ctx: ?MetalContext = null;

fn getOrInitMetalCtx() !*MetalContext {
    if (global_metal_ctx) |*ctx| return ctx;
    global_metal_ctx = try MetalContext.init();
    return &global_metal_ctx.?;
}

// ════════════════════════════════════════════════════════════════
// パラメータ不要な単項演算テスト (helpers の共通関数を使用)
// ════════════════════════════════════════════════════════════════

test "diff_mps_runtime: gelu gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testGeluGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: silu gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testSiluGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: square gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testSquareGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: tanh gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testTanhGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: sigmoid gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testSigmoidGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: relu gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testReluGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: negative gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testNegativeGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: softmax gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testSoftmaxGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: reductionSum gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testReductionSumGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: reductionMean gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testReductionMeanGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: transpose 3D gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testTranspose3DGrad(MpsAdapter, &rt);
}

test "diff_mps_runtime: reshape gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testReshapeGrad(MpsAdapter, &rt);
}

// ════════════════════════════════════════════════════════════════
// パラメータ依存テスト (checkGrad を直接使用)
// ════════════════════════════════════════════════════════════════
const checkGrad = helpers.GradientChecker(MpsAdapter).checkGrad;

test "diff_mps_runtime: matmul 2D gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 3, 2 }, .xavier);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
            const w = ctx.param(.{ .index = 0 });
            return ctx.matmul(x, w);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_mps_runtime: layerNorm gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{4}, .ones);
    _ = module.addParam(&.{4}, .zeros);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 0.5, 1.3, 2.7, 0.1, 3.2, 0.8, 1.5, 2.1 };
    try checkGrad(struct {
        fn f(ctx: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
            const gamma = ctx.param(.{ .index = 0 });
            const beta = ctx.param(.{ .index = 1 });
            return ctx.gelu(ctx.layerNorm(x, gamma, beta, 1e-5, -1));
        }
    }.f, &rt, &data, &.{ 2, 4 }, 1e-4, 2e-2);
}

test "diff_mps_runtime: add broadcast gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{3}, .xavier);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
            const b = ctx.param(.{ .index = 0 });
            return ctx.add(x, b);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_mps_runtime: mul broadcast gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
            var scalar_host = [_]f32{2.5};
            const scalar = ctx.makeTensor(&scalar_host, &.{1});
            return ctx.mul(x, scalar);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_mps_runtime: linear forward+backward" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 3, 2 }, .xavier);
    _ = module.addParam(&.{2}, .zeros);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    // loss = sum(x @ W + b) → dW = x^T @ ones, db = sum(ones, axis=0)
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input_node = MpsAdapter.makeInput(&rt, &input_data, &.{ 2, 3 }, true);

    const w = rt.param(.{ .index = 0 });
    const b = rt.param(.{ .index = 1 });
    const xw = rt.matmul(input_node, w);
    const y = rt.add(xw, b);

    const loss_node = rt.reductionSum(y, -1);
    const total_loss = rt.reductionSum(loss_node, 0);

    rt.backward(total_loss);

    // db = [2.0, 2.0] (2 rows の ones を axis=0 で sum)
    const b_grad = rt.paramGrad(1);
    try testing.expectApproxEqAbs(@as(f32, 2.0), b_grad[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), b_grad[1], 1e-5);

    // dW = x^T @ ones(2,2) = [[1,4],[2,5],[3,6]]^T @ [[1,1],[1,1]] = [[5,5],[7,7],[9,9]]
    const w_grad = rt.paramGrad(0);
    try testing.expectApproxEqAbs(@as(f32, 5.0), w_grad[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 5.0), w_grad[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 7.0), w_grad[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 7.0), w_grad[3], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 9.0), w_grad[4], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 9.0), w_grad[5], 1e-4);
}

// ════════════════════════════════════════════════════════════════
// 損失関数テスト
// ════════════════════════════════════════════════════════════════

test "diff_mps_runtime: mseLoss gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();

    var pred_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const target_data = [_]f32{ 1.5, 2.5, 2.5, 3.5 };

    rt.resetArena();
    const pred = MpsAdapter.makeInput(&rt, &pred_data, &.{ 2, 2 }, true);
    const loss = rt.mseLoss(pred, &target_data);

    rt.backward(loss);

    var ga: [4]f32 = undefined;
    _ = MpsAdapter.readGrad(&rt, pred, &ga) orelse unreachable;
    const n: f32 = 4.0;
    for (0..4) |i| {
        const expected = 2.0 * (pred_data[i] - target_data[i]) / n;
        try testing.expectApproxEqAbs(expected, ga[i], 1e-5);
    }
}

test "diff_mps_runtime: crossEntropyLossWithIndices gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();

    var logits_data = [_]f32{ 1.0, 2.0, 3.0, 1.0, 3.0, 2.0 };
    const indices = [_]u32{ 0, 2 };

    rt.resetArena();
    const logits = MpsAdapter.makeInput(&rt, &logits_data, &.{ 2, 3 }, true);
    const loss = rt.crossEntropyLossWithIndices(logits, &indices);

    rt.backward(loss);

    var ga: [6]f32 = undefined;
    _ = MpsAdapter.readGrad(&rt, logits, &ga) orelse unreachable;

    // grad = (softmax - one_hot) / batch で exact check
    const batch: f32 = 2.0;
    for (0..2) |i| {
        // softmax を手計算
        var max_val: f32 = -std.math.inf(f32);
        for (0..3) |j| {
            if (logits_data[i * 3 + j] > max_val) max_val = logits_data[i * 3 + j];
        }
        var sum_exp: f32 = 0;
        var sm: [3]f32 = undefined;
        for (0..3) |j| {
            sm[j] = @exp(logits_data[i * 3 + j] - max_val);
            sum_exp += sm[j];
        }
        for (0..3) |j| {
            sm[j] /= sum_exp;
            var expected = sm[j];
            if (j == indices[i]) expected -= 1.0;
            expected /= batch;
            try testing.expectApproxEqAbs(expected, ga[i * 3 + j], 1e-5);
        }
    }
}

test "diff_mps_runtime: bceLossWithLogits gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();

    var logits_data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const target_data = [_]f32{ 0.0, 1.0, 1.0, 0.0 };

    rt.resetArena();
    const logits = MpsAdapter.makeInput(&rt, &logits_data, &.{ 2, 2 }, true);
    const loss = rt.bceLossWithLogits(logits, &target_data);

    rt.backward(loss);

    var ga: [4]f32 = undefined;
    _ = MpsAdapter.readGrad(&rt, logits, &ga) orelse unreachable;
    const n: f32 = 4.0;
    for (0..4) |i| {
        const sig = 1.0 / (1.0 + @exp(-logits_data[i]));
        const expected = (sig - target_data[i]) / n;
        try testing.expectApproxEqAbs(expected, ga[i], 1e-5);
    }
}

test "diff_mps_runtime: gather gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 4, 3 }, .xavier);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    const indices = [_]u32{ 1, 3, 0, 1 };

    rt.resetArena();
    rt.zeroGrad();
    const table = rt.param(.{ .index = 0 });
    const embedded = rt.gather(table, &indices);

    const loss = rt.reductionSum(rt.reductionSum(embedded, -1), 0);
    rt.backward(loss);

    const tg = rt.paramGrad(0);
    // Row 1 は index 0, 3 で2回参照 → grad=2.0
    for (0..3) |j| {
        try testing.expectApproxEqAbs(@as(f32, 2.0), tg[1 * 3 + j], 1e-5);
    }
    // Row 2 は未参照 → grad=0.0
    for (0..3) |j| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), tg[2 * 3 + j], 1e-5);
    }
}

// ════════════════════════════════════════════════════════════════
// 境界サイズテスト (共通ヘルパー使用)
// ════════════════════════════════════════════════════════════════

test "diff_mps_runtime: gelu n=33 (BM=64 boundary)" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testGeluBoundary(MpsAdapter, &rt, 33);
}

test "diff_mps_runtime: gelu n=65 (BM=64+1)" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testGeluBoundary(MpsAdapter, &rt, 65);
}

test "diff_mps_runtime: gelu n=257 (threadgroup 256+1)" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testGeluBoundary(MpsAdapter, &rt, 257);
}

test "diff_mps_runtime: matmul 65x33 (tiling boundary)" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 65, 33 }, .xavier);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();
    try helpers.testMatmulBoundary(MpsAdapter, &rt, 5, 65, 33);
}

test "diff_mps_runtime: matmul 7x5 (non-power-of-2)" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 7, 5 }, .xavier);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();
    try helpers.testMatmulBoundary(MpsAdapter, &rt, 3, 7, 5);
}

test "diff_mps_runtime: softmax 4x7 (non-power-of-2 cols)" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testSoftmaxBoundary(MpsAdapter, &rt, 4, 7);
}

test "diff_mps_runtime: softmax 3x33 (non-power-of-2)" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    try helpers.testSoftmaxBoundary(MpsAdapter, &rt, 3, 33);
}

// ════════════════════════════════════════════════════════════════
// Phase 2: rmsNorm / causalSoftmax / rope
// ════════════════════════════════════════════════════════════════

test "diff_mps_runtime: rmsNorm gradient (x)" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{4}, .ones);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 0.5, 1.3, 2.7, 0.1, 3.2, 0.8, 1.5, 2.1 };
    try checkGrad(struct {
        fn f(ctx: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
            const w = ctx.param(.{ .index = 0 });
            return ctx.rmsNorm(x, w, 1e-5);
        }
    }.f, &rt, &data, &.{ 2, 4 }, 1e-4, 2e-2);
}

test "diff_mps_runtime: rmsNorm forward sanity" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{3}, .ones);
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    // x = [[1, 2, 3]], inv_rms = 1/sqrt((1+4+9)/3) = sqrt(3/14) ≈ 0.4629
    // out = x * inv_rms * w = [0.4629, 0.9258, 1.3887]
    var x_data = [_]f32{ 1.0, 2.0, 3.0 };

    rt.resetArena();
    const x = MpsAdapter.makeInput(&rt, &x_data, &.{ 1, 3 }, false);
    const w = rt.param(.{ .index = 0 });
    const y = rt.rmsNorm(x, w, 0);

    var out: [3]f32 = undefined;
    MpsAdapter.readData(&rt, y, &out);
    const inv_rms: f32 = 1.0 / @sqrt((1.0 + 4.0 + 9.0) / 3.0);
    try testing.expectApproxEqAbs(1.0 * inv_rms, out[0], 1e-4);
    try testing.expectApproxEqAbs(2.0 * inv_rms, out[1], 1e-4);
    try testing.expectApproxEqAbs(3.0 * inv_rms, out[2], 1e-4);
}

test "diff_mps_runtime: causalSoftmax upper-triangular mask" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();

    const seq: u32 = 4;
    const num_heads: u32 = 1;
    var scores = [_]f32{
        1.0, 2.0, 3.0, 4.0,
        0.5, 1.5, 2.5, 3.5,
        0.1, 0.2, 0.3, 0.4,
        2.0, 2.0, 2.0, 2.0,
    };

    rt.resetArena();
    const x = MpsAdapter.makeInput(&rt, &scores, &.{ @as(usize, seq), @as(usize, seq) }, false);
    const y = rt.causalSoftmax(x, num_heads, seq);

    var out: [16]f32 = undefined;
    MpsAdapter.readData(&rt, y, &out);

    // 上三角 (j > i) は 0、各行の合計は 1.0
    for (0..seq) |i| {
        var row_sum: f32 = 0;
        for (0..seq) |j| {
            const v = out[i * seq + j];
            if (j > i) try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-6);
            row_sum += v;
        }
        try testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 1e-5);
    }
}

test "diff_mps_runtime: causalSoftmax gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{
        0.5, 1.3, 2.7, 0.1,
        3.2, 0.8, 1.5, 2.1,
        0.3, 0.6, 1.2, 2.4,
        1.1, 0.4, 0.9, 1.8,
    };
    try checkGrad(struct {
        fn f(ctx: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
            return ctx.causalSoftmax(x, 1, 4);
        }
    }.f, &rt, &data, &.{ 4, 4 }, 1e-4, 2e-2);
}

test "diff_mps_runtime: rope gradient" {
    const metal_ctx = try getOrInitMetalCtx();
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffMpsRuntime.init(&module, metal_ctx, testing.allocator);
    defer rt.deinit();

    // x: [seq_len=3, n_heads=2, head_dim=4] = 24
    var data = [_]f32{
        0.1,  0.2,  0.3, 0.4,
        0.5,  0.6,  0.7, 0.8,
        -0.1, -0.2, 0.3, 0.4,
        0.5,  -0.6, 0.7, -0.8,
        0.9,  0.1,  -0.2, 0.3,
        -0.4, 0.5,  0.6, -0.7,
    };

    try checkGrad(struct {
        const SEQ: u32 = 3;
        const HEADS: u32 = 2;
        const HALF: u32 = 2;
        // freqs are recreated in f so checkGrad's resetArena between probes stays safe.
        fn f(ctx: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
            var freqs_host = [_]f32{ 1.0, 0.01 };
            const fr = ctx.makeTensor(&freqs_host, &.{HALF});
            fr.requires_grad = false;
            return ctx.rope(x, fr, HEADS, SEQ, HALF);
        }
    }.f, &rt, &data, &.{ 3, 2, 4 }, 1e-3, 3e-2);
}
