const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");

/// Dropout: training 時にランダムに要素をゼロ化する正則化手法。
/// - training=true: 確率 p で要素を 0 にし、残りを 1/(1-p) でスケーリング
/// - training=false: 入力をそのままコピー (identity)
///
/// backward: マスクをそのまま勾配に掛ける
pub fn dropoutForward(
    comptime T: type,
    comptime shape: anytype,
    input: *VariableMod.Variable(T, shape),
    p: T,
    training: bool,
    random: std.Random,
    allocator: Allocator,
) !VariableMod.Variable(T, shape) {
    const Var = VariableMod.Variable(T, shape);
    const Node = GraphNodeMod.GraphNode(T);
    const n = Var.num_elements;

    const out_tensor = try TensorMod.Tensor(T, shape).init(allocator);
    const out_data = out_tensor.slice();
    const in_data = input.constData();

    // マスク生成 (backward用に保存)
    const mask = try allocator.alloc(T, n);

    if (training and p > 0) {
        const inv_keep: T = 1.0 / (1.0 - p);
        for (0..n) |i| {
            if (random.float(T) < p) {
                mask[i] = 0;
                out_data[i] = 0;
            } else {
                mask[i] = inv_keep;
                out_data[i] = in_data[i] * inv_keep;
            }
        }
    } else {
        // eval mode or p=0: identity
        @memcpy(out_data, in_data);
        @memset(mask, 1);
    }

    const Ctx = struct {
        mask_data: []const T,
        input_parent: *Node,
    };
    const ctx = try allocator.create(Ctx);
    ctx.* = .{ .mask_data = mask, .input_parent = input.node };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = input.node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const c: *const Ctx = @ptrCast(@alignCast(self.context.?));

            if (c.input_parent.grad) |in_grad| {
                for (in_grad, grad_out, c.mask_data) |*dst, go, m| {
                    dst.* += go * m;
                }
            }
        }
    }.backward;

    return .{
        .tensor = out_tensor,
        .node = node,
        .owns_node = true,
        .allocator = allocator,
    };
}

// ============================================================
// テスト
// ============================================================

test "Dropout training mode" {
    const alloc = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{100}).fromSlice(temp, blk: {
        var data: [100]f64 = undefined;
        for (&data) |*v| v.* = 1.0;
        break :blk &data;
    }, false);

    var prng = std.Random.DefaultPrng.init(42);

    const output = try dropoutForward(f64, .{100}, &input, 0.5, true, prng.random(), temp);
    const out = output.constData();

    // 約50%の要素が0、残りは2.0 (= 1.0 * 1/(1-0.5))
    var zeros: usize = 0;
    var twos: usize = 0;
    for (out) |v| {
        if (v == 0) zeros += 1;
        if (@abs(v - 2.0) < 1e-10) twos += 1;
    }
    // 確率的なので厳密ではないが、大まかに50%前後
    try std.testing.expect(zeros > 20 and zeros < 80);
    try std.testing.expect(twos > 20 and twos < 80);
    try std.testing.expectEqual(@as(usize, 100), zeros + twos);
}

test "Dropout eval mode - identity" {
    const alloc = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{4}).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4 },
        false,
    );

    var prng = std.Random.DefaultPrng.init(42);
    const output = try dropoutForward(f64, .{4}, &input, 0.5, false, prng.random(), temp);

    // eval mode: 入力そのまま
    try std.testing.expectApproxEqAbs(@as(f64, 1), output.constData()[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2), output.constData()[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3), output.constData()[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4), output.constData()[3], 1e-10);
}

test "Dropout backward" {
    const alloc = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{4}).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4 },
        true,
    );
    input.node.grad = try temp.alloc(f64, 4);
    @memset(input.node.grad.?, 0);

    // p=0: 全要素を保持 (mask=1)
    var prng = std.Random.DefaultPrng.init(42);
    var output = try dropoutForward(f64, .{4}, &input, 0, true, prng.random(), temp);

    output.node.grad = try temp.alloc(f64, 4);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| bfn(output.node);

    // p=0なのでmask=1, 勾配はそのまま通過
    for (input.node.grad.?) |g| {
        try std.testing.expectApproxEqAbs(@as(f64, 1), g, 1e-10);
    }
}
