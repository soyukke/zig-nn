/// diff_cpu_runtime_test.zig: DiffCpuRuntime の数値勾配テスト
///
/// 共通テスト基盤 (diff_runtime_test_helpers.zig) を CPU Adapter 経由で使用する。
const std = @import("std");
const compute = @import("compute.zig");
const Module = compute.Module;
const diff_cpu = @import("diff_cpu_runtime.zig");
const DiffCpuRuntime = diff_cpu.DiffCpuRuntime;
const DiffTensor = diff_cpu.DiffTensor;
const helpers = @import("diff_runtime_test_helpers.zig");

const testing = std.testing;

// ════════════════════════════════════════════════════════════════
// CPU Adapter: ホストメモリ上で直接データにアクセス
// ════════════════════════════════════════════════════════════════
const CpuAdapter = struct {
    pub const Runtime = DiffCpuRuntime;
    pub const Tensor = DiffTensor;

    pub fn makeInput(rt: *Runtime, data: []f32, shape: []const usize, requires_grad: bool) Tensor {
        return rt.makeNode(data, shape, requires_grad);
    }

    pub fn readData(_: *Runtime, tensor: Tensor, dst: []f32) void {
        @memcpy(dst, tensor.data[0..dst.len]);
    }

    pub fn readGrad(_: *Runtime, tensor: Tensor, dst: []f32) ?[]f32 {
        if (tensor.grad) |g| {
            @memcpy(dst, g[0..dst.len]);
            return dst;
        }
        return null;
    }
};

// ════════════════════════════════════════════════════════════════
// パラメータ不要な単項演算テスト (helpers の共通関数を使用)
// ════════════════════════════════════════════════════════════════

test "diff_cpu_runtime: negative gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testNegativeGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: gelu gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testGeluGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: silu gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testSiluGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: square gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testSquareGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: reductionMean gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testReductionMeanGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: tanh gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testTanhGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: sigmoid gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testSigmoidGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: softmax gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testSoftmaxGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: relu gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testReluGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: reductionSum gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testReductionSumGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: transpose 3D gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testTranspose3DGrad(CpuAdapter, &rt);
}

test "diff_cpu_runtime: reshape gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    try helpers.testReshapeGrad(CpuAdapter, &rt);
}

// ════════════════════════════════════════════════════════════════
// パラメータ依存テスト (checkGrad を直接使用)
// ════════════════════════════════════════════════════════════════
const checkGrad = helpers.GradientChecker(CpuAdapter).checkGrad;

test "diff_cpu_runtime: matmul 2D gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 3, 2 }, .xavier);
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const w = ctx.param(.{ .index = 0 });
            return ctx.matmul(x, w);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: layerNorm gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{4}, .ones);
    _ = module.addParam(&.{4}, .zeros);
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 0.5, 1.3, 2.7, 0.1, 3.2, 0.8, 1.5, 2.1 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const gamma = ctx.param(.{ .index = 0 });
            const beta = ctx.param(.{ .index = 1 });
            return ctx.gelu(ctx.layerNorm(x, gamma, beta, 1e-5, -1));
        }
    }.f, &rt, &data, &.{ 2, 4 }, 1e-4, 2e-2);
}

test "diff_cpu_runtime: add broadcast gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{3}, .xavier);
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const b = ctx.param(.{ .index = 0 });
            return ctx.add(x, b);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: mul broadcast gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const scalar_data = ctx.allocData(1);
            scalar_data[0] = 2.5;
            const scalar = ctx.makeNode(scalar_data, &.{1}, false);
            return ctx.mul(x, scalar);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: linear forward+backward" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 3, 2 }, .xavier);
    _ = module.addParam(&.{2}, .zeros);
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input_node = rt.makeNode(&input_data, &.{ 2, 3 }, true);

    const w = rt.param(.{ .index = 0 });
    const b = rt.param(.{ .index = 1 });
    const xw = rt.matmul(input_node, w);
    const y = rt.add(xw, b);

    const loss_node = rt.reductionSum(y, -1);
    const total_loss = rt.reductionSum(loss_node, 0);

    rt.backward(total_loss);

    const w_grad = rt.paramGrad(0);
    const b_grad = rt.paramGrad(1);
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

test "diff_cpu_runtime: mseLoss gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var pred_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const target_data = [_]f32{ 1.5, 2.5, 2.5, 3.5 };

    rt.resetArena();
    const pred = rt.makeNode(&pred_data, &.{ 2, 2 }, true);
    const loss = rt.mseLoss(pred, &target_data);

    rt.backward(loss);

    const ga = pred.grad.?;
    const n: f32 = 4.0;
    for (0..4) |i| {
        const expected = 2.0 * (pred_data[i] - target_data[i]) / n;
        try testing.expectApproxEqAbs(expected, ga[i], 1e-5);
    }
}

test "diff_cpu_runtime: crossEntropyLossWithIndices gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var logits_data = [_]f32{ 1.0, 2.0, 3.0, 1.0, 3.0, 2.0 };
    const indices = [_]u32{ 0, 2 };

    rt.resetArena();
    const logits = rt.makeNode(&logits_data, &.{ 2, 3 }, true);
    const loss = rt.crossEntropyLossWithIndices(logits, &indices);

    rt.backward(loss);

    const ga = logits.grad.?;
    for (0..2) |i| {
        var row_sum: f32 = 0;
        for (0..3) |j| row_sum += ga[i * 3 + j];
        try testing.expectApproxEqAbs(@as(f32, 0.0), row_sum, 1e-5);
    }
}

test "diff_cpu_runtime: gather gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 4, 3 }, .xavier);
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
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
    for (0..3) |j| {
        try testing.expectApproxEqAbs(@as(f32, 2.0), tg[1 * 3 + j], 1e-5);
    }
    for (0..3) |j| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), tg[2 * 3 + j], 1e-5);
    }
}

test "diff_cpu_runtime: bceLossWithLogits gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var logits_data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const target_data = [_]f32{ 0.0, 1.0, 1.0, 0.0 };

    rt.resetArena();
    const logits = rt.makeNode(&logits_data, &.{ 2, 2 }, true);
    const loss = rt.bceLossWithLogits(logits, &target_data);

    rt.backward(loss);

    const ga = logits.grad.?;
    const n: f32 = 4.0;
    for (0..4) |i| {
        const sig = 1.0 / (1.0 + @exp(-logits_data[i]));
        const expected = (sig - target_data[i]) / n;
        try testing.expectApproxEqAbs(expected, ga[i], 1e-5);
    }
}

// ════════════════════════════════════════════════════════════════
// CPU 固有テスト (concat, split, conv2d, maxPool2d)
// ════════════════════════════════════════════════════════════════

test "diff_cpu_runtime: concat last-axis gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const parts = ctx.split(x, 2, -1);
            return ctx.concatAxis(parts[0], parts[1], -1);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: concat axis=0 gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const parts = ctx.split(x, 1, 0);
            return ctx.concatAxis(parts[0], parts[1], 0);
        }
    }.f, &rt, &data, &.{ 3, 2 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: split last-axis gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.split(x, 2, -1)[0];
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: split+transform gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const parts = ctx.split(x, 1, -1);
            const doubled = ctx.add(parts[0], parts[0]);
            return ctx.concatAxis(doubled, parts[1], -1);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: conv2d gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 1, 4 }, .xavier);
    _ = module.addParam(&.{1}, .zeros);
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const in_ch = 1;
    const out_ch = 1;
    const k = 2;
    const h = 3;
    const w_dim = 3;

    rt.resetArena();
    const input = rt.makeNode(&data, &.{ 1, in_ch * h * w_dim }, true);
    const weight_t = rt.param(.{ .index = 0 });
    const bias_t = rt.param(.{ .index = 1 });
    const output = rt.conv2d(input, weight_t, bias_t, 1, 0, k, in_ch, out_ch, h, w_dim);

    try testing.expectEqual(@as(usize, 1), output.shape[0]);
    try testing.expectEqual(@as(usize, 4), output.shape[1]);

    rt.backward(output);
    try testing.expect(input.grad != null);
    try testing.expect(weight_t.grad != null);
}

test "diff_cpu_runtime: maxPool2d gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    rt.resetArena();
    const input = rt.makeNode(&data, &.{ 1, 16 }, true);
    const output = rt.maxPool2d(input, 2, 2, 1, 4, 4);

    try testing.expectEqual(@as(usize, 1), output.shape[0]);
    try testing.expectEqual(@as(usize, 4), output.shape[1]);

    try testing.expectApproxEqAbs(@as(f32, 6), output.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 8), output.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 14), output.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 16), output.data[3], 1e-5);

    rt.backward(output);
    const ga = input.grad.?;

    try testing.expect(ga[5] != 0);
    try testing.expect(ga[7] != 0);
    try testing.expect(ga[13] != 0);
    try testing.expect(ga[15] != 0);
    try testing.expectApproxEqAbs(@as(f32, 0), ga[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), ga[1], 1e-5);
}

// logSoftmax は CPU 固有
test "diff_cpu_runtime: logSoftmax gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.logSoftmax(x, -1);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 2e-2);
}
