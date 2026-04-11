const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");

/// MaxPool2D forward。
/// 各プーリングウィンドウ内の最大値を取る。
/// backwardではmax位置に勾配をルーティングする。
///
/// 入力: (batch, channels, H, W)
/// 出力: (batch, channels, OH, OW)
///   OH = (H - pool_size) / pool_stride + 1
pub fn maxPool2dForward(
    comptime T: type,
    comptime batch: usize,
    comptime channels: usize,
    comptime H: usize,
    comptime W: usize,
    comptime pool_size: usize,
    comptime pool_stride: usize,
    input: *VariableMod.Variable(T, .{ batch, channels, H, W }),
    allocator: Allocator,
) !VariableMod.Variable(T, .{ batch, channels, (H - pool_size) / pool_stride + 1, (W - pool_size) / pool_stride + 1 }) {
    const OH = comptime (H - pool_size) / pool_stride + 1;
    const OW = comptime (W - pool_size) / pool_stride + 1;
    const out_n = batch * channels * OH * OW;
    const Node = GraphNodeMod.GraphNode(T);

    const OutTensor = TensorMod.Tensor(T, .{ batch, channels, OH, OW });
    const out_tensor = try OutTensor.init(allocator);
    const out_data = out_tensor.slice();

    const in_data = input.constData();

    // max indices を保存 (backward用)
    const Ctx = struct {
        max_indices: []usize,
        input_parent: *Node,
    };

    const ctx = try allocator.create(Ctx);
    ctx.max_indices = try allocator.alloc(usize, out_n);
    ctx.input_parent = input.node;

    for (0..batch) |b| {
        for (0..channels) |ch| {
            const in_base = b * channels * H * W + ch * H * W;
            const out_base = b * channels * OH * OW + ch * OH * OW;

            for (0..OH) |oh| {
                for (0..OW) |ow| {
                    var max_val: T = -std.math.inf(T);
                    var max_flat_idx: usize = 0;

                    for (0..pool_size) |ph| {
                        for (0..pool_size) |pw| {
                            const ih = oh * pool_stride + ph;
                            const iw = ow * pool_stride + pw;
                            const flat_idx = ih * W + iw;
                            if (in_data[in_base + flat_idx] > max_val) {
                                max_val = in_data[in_base + flat_idx];
                                max_flat_idx = flat_idx;
                            }
                        }
                    }

                    const out_idx = out_base + oh * OW + ow;
                    out_data[out_idx] = max_val;
                    ctx.max_indices[out_idx] = max_flat_idx;
                }
            }
        }
    }

    const OutVar = VariableMod.Variable(T, .{ batch, channels, OH, OW });
    var result = try OutVar.init(out_tensor, allocator, true);
    result.node.parents[0] = input.node;
    result.node.context = @ptrCast(ctx);

    result.node.backward_fn = struct {
        fn backward(node: *Node) void {
            const grad_out = node.grad orelse return;
            const c: *const Ctx = @ptrCast(@alignCast(node.context.?));

            if (c.input_parent.grad) |in_grad| {
                for (0..batch) |b| {
                    for (0..channels) |ch| {
                        const in_base = b * channels * H * W + ch * H * W;
                        const out_base = b * channels * OH * OW + ch * OH * OW;

                        for (0..OH) |oh| {
                            for (0..OW) |ow| {
                                const out_idx = out_base + oh * OW + ow;
                                const max_idx = c.max_indices[out_idx];
                                in_grad[in_base + max_idx] += grad_out[out_idx];
                            }
                        }
                    }
                }
            }
        }
    }.backward;

    return result;
}

// ============================================================
// テスト
// ============================================================

test "MaxPool2D forward" {
    const alloc = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // Input (1, 1, 4, 4): 1..16
    var input = try VariableMod.Variable(f64, .{ 1, 1, 4, 4 }).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 },
        false,
    );

    const output = try maxPool2dForward(f64, 1, 1, 4, 4, 2, 2, &input, temp);

    // Pool(2,2) stride=2: OH=2, OW=2
    // (0,0): max(1,2,5,6) = 6
    // (0,1): max(3,4,7,8) = 8
    // (1,0): max(9,10,13,14) = 14
    // (1,1): max(11,12,15,16) = 16
    try std.testing.expectApproxEqAbs(@as(f64, 6), output.constData()[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8), output.constData()[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 14), output.constData()[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 16), output.constData()[3], 1e-10);
}

test "MaxPool2D backward" {
    const alloc = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 1, 4, 4 }).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 },
        true,
    );
    input.node.grad = try temp.alloc(f64, 16);
    @memset(input.node.grad.?, 0);

    var output = try maxPool2dForward(f64, 1, 1, 4, 4, 2, 2, &input, temp);

    // loss = sum(output), dL/dout = 1
    output.node.grad = try temp.alloc(f64, 4);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| {
        bfn(output.node);
    }

    // Gradient goes to max positions only:
    // max at: (1,1)=5, (1,3)=7, (3,1)=13, (3,3)=15 (flat indices)
    const expected = [_]f64{ 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1 };
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(expected[i], input.node.grad.?[i], 1e-10);
    }
}

test "MaxPool2D multi-channel" {
    const alloc = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // (1, 2, 4, 4): 2 channels
    const ch0 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const ch1 = [_]f64{ 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    var input = try VariableMod.Variable(f64, .{ 1, 2, 4, 4 }).fromSlice(
        temp,
        &(ch0 ++ ch1),
        false,
    );

    const output = try maxPool2dForward(f64, 1, 2, 4, 4, 2, 2, &input, temp);

    // Channel 0: max(1,2,5,6)=6, max(3,4,7,8)=8, max(9,10,13,14)=14, max(11,12,15,16)=16
    try std.testing.expectApproxEqAbs(@as(f64, 6), output.constData()[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8), output.constData()[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 14), output.constData()[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 16), output.constData()[3], 1e-10);

    // Channel 1: max(16,15,12,11)=16, max(14,13,10,9)=14, max(8,7,4,3)=8, max(6,5,2,1)=6
    try std.testing.expectApproxEqAbs(@as(f64, 16), output.constData()[4], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 14), output.constData()[5], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8), output.constData()[6], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6), output.constData()[7], 1e-10);
}
