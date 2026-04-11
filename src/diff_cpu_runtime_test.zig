/// diff_cpu_runtime_test.zig: DiffCpuRuntime の数値勾配テスト
const std = @import("std");
const compute = @import("compute.zig");
const Module = compute.Module;
const diff_cpu = @import("diff_cpu_runtime.zig");
const DiffCpuRuntime = diff_cpu.DiffCpuRuntime;
const DiffTensor = diff_cpu.DiffTensor;
const DiffNode = diff_cpu.DiffNode;

const testing = std.testing;

fn checkGrad(
    comptime f: fn (*DiffCpuRuntime, DiffTensor) DiffTensor,
    runtime: *DiffCpuRuntime,
    x_data: []f32,
    shape: []const usize,
    eps: f32,
    tol: f32,
) !void {
    const n = x_data.len;
    const num_grad = try runtime.allocator.alloc(f32, n);
    defer runtime.allocator.free(num_grad);

    // First run to determine output size and generate random weights
    runtime.resetArena();
    const probe = f(runtime, runtime.makeNode(x_data, shape, true));
    const out_n = probe.totalElements();
    const weights = try runtime.allocator.alloc(f32, out_n);
    defer runtime.allocator.free(weights);
    // Use deterministic non-uniform weights so sum(w*y) has non-trivial gradients
    for (weights, 0..) |*w, idx| {
        w.* = @as(f32, @floatFromInt(idx + 1)) * 0.3 + 0.1;
    }

    // Numerical gradient: loss = sum(weights * f(x)), computed in f64 for precision
    for (0..n) |i| {
        const orig = x_data[i];

        x_data[i] = orig + eps;
        runtime.resetArena();
        const y_plus = f(runtime, runtime.makeNode(x_data, shape, true));
        var loss_plus: f64 = 0;
        for (y_plus.data, 0..) |v, j| loss_plus += @as(f64, v) * @as(f64, weights[j]);

        x_data[i] = orig - eps;
        runtime.resetArena();
        const y_minus = f(runtime, runtime.makeNode(x_data, shape, true));
        var loss_minus: f64 = 0;
        for (y_minus.data, 0..) |v, j| loss_minus += @as(f64, v) * @as(f64, weights[j]);

        num_grad[i] = @floatCast((loss_plus - loss_minus) / (2.0 * @as(f64, eps)));
        x_data[i] = orig;
    }

    // Analytical gradient via backward()
    // Construct: loss = sum(weights * f(x))
    runtime.resetArena();
    const x_node = runtime.makeNode(x_data, shape, true);
    const y_node = f(runtime, x_node);

    // Weighted sum node
    const ws_data = runtime.allocData(1);
    ws_data[0] = 0;
    for (y_node.data, 0..) |v, j| ws_data[0] += v * weights[j];
    const loss_node = runtime.makeNode(ws_data, &.{1}, false);
    // backward: d(loss)/d(y_i) = weights[i]
    const WeightCtx = struct { w: []const f32 };
    const wctx = runtime.arena.allocator().create(WeightCtx) catch unreachable;
    wctx.* = .{ .w = weights };
    loss_node.context = wctx;
    loss_node.backward_fn = struct {
        fn backward(self: *DiffNode) void {
            const parent = self.parents[0].?;
            if (parent.grad == null) return;
            const go = self.grad.?[0];
            const ctx: *const WeightCtx = @ptrCast(@alignCast(self.context.?));
            for (parent.grad.?, 0..) |*g, j| g.* += go * ctx.w[j];
        }
    }.backward;
    loss_node.parents[0] = y_node;

    runtime.backward(loss_node);

    const ana_grad = x_node.grad.?;
    for (0..n) |i| {
        const diff = @abs(ana_grad[i] - num_grad[i]);
        // Use combined tolerance: max(relative_error, absolute_error)
        // This handles near-zero gradients gracefully
        const scale = @max(@abs(ana_grad[i]), @abs(num_grad[i]));
        const abs_tol: f32 = 3e-3; // absolute tolerance for near-zero gradients (f32 precision)
        if (diff > @max(tol * scale, abs_tol)) {
            std.debug.print("Gradient mismatch at [{d}]: analytical={d:.6}, numerical={d:.6}, diff={d:.6}\n", .{ i, ana_grad[i], num_grad[i], diff });
            return error.TestExpectedApproxEqAbs;
        }
    }
}

test "diff_cpu_runtime: gelu gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ -1.0, 0.0, 0.5, 1.0, 2.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.gelu(x);
        }
    }.f, &rt, &data, &.{5}, 1e-4, 1e-2);
}

test "diff_cpu_runtime: silu gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.silu(x);
        }
    }.f, &rt, &data, &.{5}, 1e-4, 1e-2);
}

test "diff_cpu_runtime: square gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ -2.0, -1.0, 0.5, 1.0, 2.0, 3.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.square(x);
        }
    }.f, &rt, &data, &.{6}, 1e-4, 1e-2);
}

test "diff_cpu_runtime: reductionMean gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.reductionMean(x, -1);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: tanh gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ -2.0, -1.0, 0.0, 0.5, 1.0, 2.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.tanh_(x);
        }
    }.f, &rt, &data, &.{6}, 1e-4, 1e-2);
}

test "diff_cpu_runtime: sigmoid gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ -2.0, -1.0, 0.0, 0.5, 1.0, 2.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.sigmoid(x);
        }
    }.f, &rt, &data, &.{6}, 1e-4, 1e-2);
}

test "diff_cpu_runtime: negative gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, -2.0, 3.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.negative(x);
        }
    }.f, &rt, &data, &.{3}, 1e-4, 1e-2);
}

test "diff_cpu_runtime: softmax gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.softmax(x, -1);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

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

test "diff_cpu_runtime: matmul 2D gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 3, 2 }, .xavier);
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    // Test gradient w.r.t. input (through matmul with param)
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // [2, 3]
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
    _ = module.addParam(&.{4}, .ones); // gamma
    _ = module.addParam(&.{4}, .zeros); // beta
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    // Use non-uniform input to avoid symmetric cancellation in gradients
    var data = [_]f32{ 0.5, 1.3, 2.7, 0.1, 3.2, 0.8, 1.5, 2.1 }; // [2, 4]
    // Apply gelu after layerNorm to break the sum-to-zero gradient symmetry
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
    _ = module.addParam(&.{3}, .xavier); // bias
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // [2, 3]
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

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // [2, 3]
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const scalar_data = ctx.allocData(1);
            scalar_data[0] = 2.5;
            const scalar = ctx.makeNode(scalar_data, &.{1}, false);
            return ctx.mul(x, scalar);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: reductionSum gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // [2, 3]
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.reductionSum(x, -1);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: transpose 3D gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // [1, 2, 3]
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.transpose(x, 1, 2);
        }
    }.f, &rt, &data, &.{ 1, 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: reshape gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.reshape(x, &.{ 2, 3 });
        }
    }.f, &rt, &data, &.{6}, 1e-4, 1e-2);
}

test "diff_cpu_runtime: linear forward+backward" {
    // Test a full linear layer: y = x @ W + b
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 3, 2 }, .xavier); // W [3, 2]
    _ = module.addParam(&.{2}, .zeros); // b [2]
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    // Input [2, 3]
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input_node = rt.makeNode(&input_data, &.{ 2, 3 }, true);

    // Forward: y = x @ W + b
    const w = rt.param(.{ .index = 0 });
    const b = rt.param(.{ .index = 1 });
    const xw = rt.matmul(input_node, w);
    const y = rt.add(xw, b);

    // Loss = sum(y)
    const loss_node = rt.reductionSum(y, -1);
    const total_loss = rt.reductionSum(loss_node, 0);

    rt.backward(total_loss);

    // Check that param gradients are non-zero
    const w_grad = rt.paramGrad(0);
    const b_grad = rt.paramGrad(1);
    var w_norm: f32 = 0;
    for (w_grad) |g| w_norm += g * g;
    var b_norm: f32 = 0;
    for (b_grad) |g| b_norm += g * g;

    try testing.expect(w_norm > 0);
    try testing.expect(b_norm > 0);
}

test "diff_cpu_runtime: relu gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ -2.0, -0.5, 0.1, 0.5, 1.0, 2.0 };
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            return ctx.relu(x);
        }
    }.f, &rt, &data, &.{6}, 1e-4, 1e-2);
}

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

    // Check gradient exists and is correct: 2*(pred-target)/n
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

    // logits [2, 3], targets [2]
    var logits_data = [_]f32{ 1.0, 2.0, 3.0, 1.0, 3.0, 2.0 };
    const indices = [_]u32{ 0, 2 };

    rt.resetArena();
    const logits = rt.makeNode(&logits_data, &.{ 2, 3 }, true);
    const loss = rt.crossEntropyLossWithIndices(logits, &indices);

    rt.backward(loss);

    // Verify gradient is non-zero and has expected properties
    const ga = logits.grad.?;
    // For each row: grad = (softmax - one_hot) / batch
    // Sum of gradients per row should be ~0 (softmax sums to 1, one_hot sums to 1)
    for (0..2) |i| {
        var row_sum: f32 = 0;
        for (0..3) |j| row_sum += ga[i * 3 + j];
        try testing.expectApproxEqAbs(@as(f32, 0.0), row_sum, 1e-5);
    }
}

test "diff_cpu_runtime: gather gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    _ = module.addParam(&.{ 4, 3 }, .xavier); // embedding table [4, 3]
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    const indices = [_]u32{ 1, 3, 0, 1 }; // 4 lookups

    rt.resetArena();
    rt.zeroGrad();
    const table = rt.param(.{ .index = 0 });
    const embedded = rt.gather(table, &indices); // [4, 3]

    // Sum all for scalar loss
    const loss = rt.reductionSum(rt.reductionSum(embedded, -1), 0);
    rt.backward(loss);

    // Check that table gradient has accumulation for repeated index (1 appears twice)
    const tg = rt.paramGrad(0);
    // Row 1 should have grad=2.0 for each element (appears in index 0 and 3)
    for (0..3) |j| {
        try testing.expectApproxEqAbs(@as(f32, 2.0), tg[1 * 3 + j], 1e-5);
    }
    // Row 2 should have grad=0.0 (not referenced)
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

    // Verify gradient: (sigmoid(x) - target) / n
    const ga = logits.grad.?;
    const n: f32 = 4.0;
    for (0..4) |i| {
        const sig = 1.0 / (1.0 + @exp(-logits_data[i]));
        const expected = (sig - target_data[i]) / n;
        try testing.expectApproxEqAbs(expected, ga[i], 1e-5);
    }
}

test "diff_cpu_runtime: concat last-axis gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    // Test: f(x) = concat(split(x)) should pass through gradients correctly
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [2, 3]
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

    var data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [3, 2]
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

    // f(x) = split(x, 2, -1)[0] — gradient only flows to first part
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [2, 3]
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

    // f(x) = concat(split(x)[0] + split(x)[0], split(x)[1])
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [2, 3]
    try checkGrad(struct {
        fn f(ctx: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
            const parts = ctx.split(x, 1, -1);
            const doubled = ctx.add(parts[0], parts[0]); // 2x first col
            return ctx.concatAxis(doubled, parts[1], -1);
        }
    }.f, &rt, &data, &.{ 2, 3 }, 1e-4, 1e-2);
}

test "diff_cpu_runtime: conv2d gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    // weight [1, 1*2*2] = [1, 4], bias [1]
    _ = module.addParam(&.{ 1, 4 }, .xavier);
    _ = module.addParam(&.{1}, .zeros);
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    // input: [1, 1*3*3] = [1, 9] — 1 batch, 1 channel, 3x3
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const in_ch = 1;
    const out_ch = 1;
    const k = 2;
    const h = 3;
    const w_dim = 3;

    // Forward pass
    rt.resetArena();
    const input = rt.makeNode(&data, &.{ 1, in_ch * h * w_dim }, true);
    const weight_t = rt.param(.{ .index = 0 });
    const bias_t = rt.param(.{ .index = 1 });
    const output = rt.conv2d(input, weight_t, bias_t, 1, 0, k, in_ch, out_ch, h, w_dim);

    // Output shape: OH = (3-2)/1+1 = 2, OW = 2, so [1, 1*2*2] = [1, 4]
    try testing.expectEqual(@as(usize, 1), output.shape[0]);
    try testing.expectEqual(@as(usize, 4), output.shape[1]);

    // Backward: verify no crash and gradients exist
    rt.backward(output);
    try testing.expect(input.grad != null);
    try testing.expect(weight_t.grad != null);
}

test "diff_cpu_runtime: maxPool2d gradient" {
    var module = Module.init(testing.allocator);
    defer module.deinit();
    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    // input: [1, 1*4*4] = [1, 16] — 1 batch, 1 channel, 4x4
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    rt.resetArena();
    const input = rt.makeNode(&data, &.{ 1, 16 }, true);
    const output = rt.maxPool2d(input, 2, 2, 1, 4, 4);

    // Output: OH = (4-2)/2+1 = 2, OW = 2, so [1, 4]
    try testing.expectEqual(@as(usize, 1), output.shape[0]);
    try testing.expectEqual(@as(usize, 4), output.shape[1]);

    // Max values should be 6, 8, 14, 16
    try testing.expectApproxEqAbs(@as(f32, 6), output.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 8), output.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 14), output.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 16), output.data[3], 1e-5);

    // Backward
    rt.backward(output);
    const ga = input.grad.?;

    // Only the max positions should have gradient
    // Position 5 (val 6), 7 (val 8), 13 (val 14), 15 (val 16)
    try testing.expect(ga[5] != 0); // max of [1,2,5,6] = 6
    try testing.expect(ga[7] != 0); // max of [3,4,7,8] = 8
    try testing.expect(ga[13] != 0); // max of [9,10,13,14] = 14
    try testing.expect(ga[15] != 0); // max of [11,12,15,16] = 16
    // Non-max positions should be 0
    try testing.expectApproxEqAbs(@as(f32, 0), ga[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), ga[1], 1e-5);
}
