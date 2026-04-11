const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");

// ============================================================
// Cross Entropy Loss
// ============================================================

/// Cross Entropy Loss (Softmax + NLL)。
/// 多クラス分類用。
///
/// input: (batch, num_classes) - logits (softmax前の生の値)
/// target: [batch]usize - 正解クラスのインデックス
///
/// L = -mean(log(softmax(x)[target]))
///
/// backward:
///   dL/dx_i = softmax(x)_i - 1{i == target}  (バッチ平均)
pub fn crossEntropyLoss(
    comptime T: type,
    comptime batch: usize,
    comptime num_classes: usize,
    pred: *VariableMod.Variable(T, .{ batch, num_classes }),
    target: *const [batch]usize,
    allocator: Allocator,
) !VariableMod.Variable(T, .{1}) {
    const Node = GraphNodeMod.GraphNode(T);
    const in_data = pred.constData();

    // Softmax + NLL forward
    const softmax_buf = try allocator.alloc(T, batch * num_classes);
    var loss_val: T = 0;

    for (0..batch) |b| {
        const row = in_data[b * num_classes .. (b + 1) * num_classes];
        const sm = softmax_buf[b * num_classes .. (b + 1) * num_classes];

        // Numerically stable softmax
        var max_val: T = row[0];
        for (row[1..]) |v| {
            if (v > max_val) max_val = v;
        }
        var sum_exp: T = 0;
        for (row, sm) |v, *s| {
            s.* = @exp(v - max_val);
            sum_exp += s.*;
        }
        for (sm) |*s| s.* /= sum_exp;

        // NLL: -log(softmax[target])
        const t_class = target[b];
        const p = sm[t_class];
        // Clamp to avoid log(0)
        const p_clamped = @max(p, 1e-12);
        loss_val -= @log(p_clamped);
    }
    loss_val /= @as(T, @floatFromInt(batch));

    const result_tensor = try TensorMod.Tensor(T, .{1}).init(allocator);
    result_tensor.data[0] = loss_val;

    // Context for backward
    const Ctx = struct {
        softmax: []const T,
        target: [batch]usize,
        pred_node: *Node,
    };
    const ctx = try allocator.create(Ctx);
    ctx.* = .{
        .softmax = softmax_buf,
        .target = target.*,
        .pred_node = pred.node,
    };

    const node = try allocator.create(Node);
    node.* = Node.init(1, true);
    node.parents[0] = pred.node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const c: *const Ctx = @ptrCast(@alignCast(self.context.?));
            const inv_batch: T = 1.0 / @as(T, @floatFromInt(batch));

            if (c.pred_node.grad) |g| {
                for (0..batch) |b| {
                    for (0..num_classes) |j| {
                        const idx = b * num_classes + j;
                        var grad_val = c.softmax[idx];
                        if (j == c.target[b]) grad_val -= 1.0;
                        g[idx] += grad_out[0] * grad_val * inv_batch;
                    }
                }
            }
        }
    }.backward;

    return .{
        .tensor = result_tensor,
        .node = node,
        .owns_node = true,
        .allocator = allocator,
    };
}

// ============================================================
// Binary Cross Entropy Loss (with logits)
// ============================================================

/// BCE with Logits Loss。
/// 二値分類用。内部でsigmoidを適用してから BCE を計算する。
///
/// input: (n) - logits
/// target: [n]T - 正解ラベル (0 or 1)
///
/// L = -mean(target * log(σ(x)) + (1-target) * log(1-σ(x)))
///
/// 数値安定版:
///   L = mean(max(x, 0) - x*target + log(1 + exp(-|x|)))
///
/// backward:
///   dL/dx = (σ(x) - target) / n
pub fn bceLossWithLogits(
    comptime T: type,
    comptime shape: anytype,
    pred: *VariableMod.Variable(T, shape),
    target: []const T,
    allocator: Allocator,
) !VariableMod.Variable(T, .{1}) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = VariableMod.Variable(T, shape).num_elements;
    const in_data = pred.constData();

    // Forward: numerically stable BCE
    var loss_val: T = 0;
    for (0..n) |i| {
        const x = in_data[i];
        const t = target[i];
        // max(x, 0) - x*t + log(1 + exp(-|x|))
        const max_x: T = @max(x, 0);
        loss_val += max_x - x * t + @log(1.0 + @exp(-@abs(x)));
    }
    loss_val /= @as(T, @floatFromInt(n));

    const result_tensor = try TensorMod.Tensor(T, .{1}).init(allocator);
    result_tensor.data[0] = loss_val;

    // Save sigmoid for backward
    const sig_buf = try allocator.alloc(T, n);
    for (0..n) |i| {
        sig_buf[i] = 1.0 / (1.0 + @exp(-in_data[i]));
    }

    const Ctx = struct {
        sigmoid: []const T,
        target: []const T,
        pred_node: *Node,
    };
    const ctx = try allocator.create(Ctx);
    ctx.* = .{
        .sigmoid = sig_buf,
        .target = target,
        .pred_node = pred.node,
    };

    const node = try allocator.create(Node);
    node.* = Node.init(1, true);
    node.parents[0] = pred.node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const c: *const Ctx = @ptrCast(@alignCast(self.context.?));
            const inv_n: T = 1.0 / @as(T, @floatFromInt(n));

            if (c.pred_node.grad) |g| {
                for (0..n) |i| {
                    g[i] += grad_out[0] * (c.sigmoid[i] - c.target[i]) * inv_n;
                }
            }
        }
    }.backward;

    return .{
        .tensor = result_tensor,
        .node = node,
        .owns_node = true,
        .allocator = allocator,
    };
}

// ============================================================
// テスト
// ============================================================

test "CrossEntropy forward" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // batch=2, num_classes=3
    // logits: [[2, 1, 0.1], [0.5, 2, 0.3]]
    // targets: [0, 1]
    var pred = try VariableMod.Variable(f64, .{ 2, 3 }).fromSlice(temp, &[_]f64{
        2.0, 1.0, 0.1,
        0.5, 2.0, 0.3,
    }, true);

    const target = [_]usize{ 0, 1 };
    const loss = try crossEntropyLoss(f64, 2, 3, &pred, &target, temp);

    // softmax([2, 1, 0.1]) → p[0]=0.6590 → -log = 0.4170
    // softmax([0.5, 2, 0.3]) → p[1]=0.7113 → -log = 0.3410
    // loss = mean(0.4170 + 0.3410) = 0.3790
    try std.testing.expect(loss.constData()[0] > 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.379), loss.constData()[0], 0.01);
}

test "CrossEntropy perfect prediction" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // Very confident correct prediction → loss ≈ 0
    var pred = try VariableMod.Variable(f64, .{ 1, 3 }).fromSlice(temp, &[_]f64{
        100.0, -100.0, -100.0,
    }, true);
    const target = [_]usize{0};
    const loss = try crossEntropyLoss(f64, 1, 3, &pred, &target, temp);
    try std.testing.expectApproxEqAbs(@as(f64, 0), loss.constData()[0], 1e-6);
}

test "CrossEntropy backward" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var pred = try VariableMod.Variable(f64, .{ 2, 3 }).fromSlice(temp, &[_]f64{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
    }, true);
    pred.node.grad = try temp.alloc(f64, 6);
    @memset(pred.node.grad.?, 0);

    const target = [_]usize{ 2, 0 };
    var loss = try crossEntropyLoss(f64, 2, 3, &pred, &target, temp);
    loss.node.grad = try temp.alloc(f64, 1);
    loss.node.grad.?[0] = 1.0;

    if (loss.node.backward_fn) |bfn| bfn(loss.node);

    // Gradient: (softmax - one_hot) / batch
    // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
    // Sample 0 target=2: [0.0900, 0.2447, -0.3348] / 2
    // Sample 1 target=0: [-0.9100, 0.2447, 0.6652] / 2
    const g = pred.node.grad.?;
    // Sum of gradients per sample should be 0
    const sum0 = g[0] + g[1] + g[2];
    const sum1 = g[3] + g[4] + g[5];
    try std.testing.expectApproxEqAbs(@as(f64, 0), sum0, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), sum1, 1e-10);

    // Target class should have negative gradient
    try std.testing.expect(g[2] < 0); // sample 0, target=2
    try std.testing.expect(g[3] < 0); // sample 1, target=0
}

test "BCE with logits forward" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // logits = [0] → σ(0)=0.5
    // target = [1]
    // loss = -(1*log(0.5) + 0*log(0.5)) = log(2) ≈ 0.693
    var pred = try VariableMod.Variable(f64, .{1}).fromSlice(temp, &[_]f64{0.0}, true);
    const target = [_]f64{1.0};
    const loss = try bceLossWithLogits(f64, .{1}, &pred, &target, temp);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6931), loss.constData()[0], 1e-3);
}

test "BCE with logits confident prediction" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // Very confident correct: logit=10, target=1 → σ(10)≈1 → loss≈0
    var pred = try VariableMod.Variable(f64, .{1}).fromSlice(temp, &[_]f64{10.0}, true);
    const target = [_]f64{1.0};
    const loss = try bceLossWithLogits(f64, .{1}, &pred, &target, temp);
    try std.testing.expectApproxEqAbs(@as(f64, 0), loss.constData()[0], 1e-4);
}

test "BCE with logits backward" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var pred = try VariableMod.Variable(f64, .{3}).fromSlice(temp, &[_]f64{ 0, 1, -1 }, true);
    pred.node.grad = try temp.alloc(f64, 3);
    @memset(pred.node.grad.?, 0);

    const target = [_]f64{ 1, 1, 0 };
    var loss = try bceLossWithLogits(f64, .{3}, &pred, &target, temp);
    loss.node.grad = try temp.alloc(f64, 1);
    loss.node.grad.?[0] = 1.0;

    if (loss.node.backward_fn) |bfn| bfn(loss.node);

    // dL/dx = (σ(x) - target) / n
    // σ(0)=0.5, σ(1)≈0.731, σ(-1)≈0.269
    // grad = [(0.5-1)/3, (0.731-1)/3, (0.269-0)/3]
    const g = pred.node.grad.?;
    try std.testing.expectApproxEqAbs(@as(f64, -0.5 / 3.0), g[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, (0.7310585786 - 1.0) / 3.0), g[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2689414214 / 3.0), g[2], 1e-4);
}

test "BCE with logits symmetry" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // logit=0, target=0 should equal logit=0, target=1 (both = log(2))
    var p1 = try VariableMod.Variable(f64, .{1}).fromSlice(temp, &[_]f64{0.0}, false);
    var p2 = try VariableMod.Variable(f64, .{1}).fromSlice(temp, &[_]f64{0.0}, false);
    const l1 = try bceLossWithLogits(f64, .{1}, &p1, &[_]f64{0.0}, temp);
    const l2 = try bceLossWithLogits(f64, .{1}, &p2, &[_]f64{1.0}, temp);
    try std.testing.expectApproxEqAbs(l1.constData()[0], l2.constData()[0], 1e-10);
}
