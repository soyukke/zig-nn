const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");
const ModuleMixin = @import("module.zig").Module;
const cpu_backend = @import("../backend/cpu.zig");

/// 2D畳み込みレイヤー。
/// output = conv2d(input, weight) + bias
///
/// - T: スカラー型 (f16, f32, f64)
/// - in_ch: 入力チャンネル数
/// - out_ch: 出力チャンネル数
/// - kernel_size: カーネルサイズ (正方形)
/// - stride_val: ストライド
/// - pad_val: ゼロパディング
///
/// 入力形状: (batch, in_ch, H, W) — NCHW
/// 出力形状: (batch, out_ch, OH, OW)
///   OH = (H + 2*pad - kernel_size) / stride + 1
pub fn Conv2D(
    comptime T: type,
    comptime in_ch: usize,
    comptime out_ch: usize,
    comptime kernel_size: usize,
    comptime stride_val: usize,
    comptime pad_val: usize,
) type {
    const col_ch = in_ch * kernel_size * kernel_size;

    return struct {
        const Self = @This();
        const M = ModuleMixin(Self);

        weight: VariableMod.Variable(T, .{ out_ch, col_ch }),
        bias: VariableMod.Variable(T, .{out_ch}),

        /// Kaiming He 初期化
        pub fn init(allocator: Allocator) !Self {
            const fan_in: T = @floatFromInt(col_ch);
            const limit = @sqrt(6.0 / fan_in);

            const weight_tensor = try TensorMod.Tensor(T, .{ out_ch, col_ch }).init(allocator);
            var prng = std.Random.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                std.posix.getrandom(std.mem.asBytes(&seed)) catch {
                    seed = 42;
                };
                break :blk seed;
            });
            const rng = prng.random();
            const w_data = weight_tensor.slice();
            for (w_data) |*v| {
                v.* = (rng.float(T) * 2.0 - 1.0) * limit;
            }

            const weight = try VariableMod.Variable(T, .{ out_ch, col_ch }).init(
                weight_tensor,
                allocator,
                true,
            );

            const bias_tensor = try TensorMod.Tensor(T, .{out_ch}).zeros(allocator);
            const bias = try VariableMod.Variable(T, .{out_ch}).init(
                bias_tensor,
                allocator,
                true,
            );

            return .{ .weight = weight, .bias = bias };
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }

        pub fn outHeight(comptime H: usize) usize {
            return (H + 2 * pad_val - kernel_size) / stride_val + 1;
        }

        pub fn outWidth(comptime W: usize) usize {
            return (W + 2 * pad_val - kernel_size) / stride_val + 1;
        }

        /// forward: conv2d(input, weight) + bias  (im2col + BLAS)
        /// input: (batch, in_ch, H, W) => output: (batch, out_ch, OH, OW)
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime H: usize,
            comptime W: usize,
            input: *VariableMod.Variable(T, .{ batch, in_ch, H, W }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, out_ch, outHeight(H), outWidth(W) }) {
            const OH = comptime outHeight(H);
            const OW = comptime outWidth(W);
            const out_spatial = OH * OW;
            const Node = GraphNodeMod.GraphNode(T);

            const OutTensor = TensorMod.Tensor(T, .{ batch, out_ch, OH, OW });
            const out_tensor = try OutTensor.init(allocator);
            const out_data = out_tensor.slice();

            const weight_slice = self.weight.constData();
            const bias_slice = self.bias.constData();
            const in_slice = input.constData();

            // im2col + BLAS matmul forward (col_buf + d_col in single alloc)
            const fwd_buf = try allocator.alloc(T, (batch + 1) * col_ch * out_spatial);
            const col_buf = fwd_buf[0 .. batch * col_ch * out_spatial];
            const d_col_buf = fwd_buf[batch * col_ch * out_spatial ..][0 .. col_ch * out_spatial];

            for (0..batch) |b| {
                const col_off = b * col_ch * out_spatial;
                const in_off = b * in_ch * H * W;

                // im2col: input patches → column matrix (col_ch, out_spatial)
                for (0..out_spatial) |s| {
                    const oh = s / OW;
                    const ow = s % OW;
                    for (0..in_ch) |ic| {
                        for (0..kernel_size) |kh| {
                            for (0..kernel_size) |kw| {
                                const ih = oh * stride_val + kh;
                                const iw = ow * stride_val + kw;
                                const col_row = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                if (ih >= pad_val and ih - pad_val < H and
                                    iw >= pad_val and iw - pad_val < W)
                                {
                                    col_buf[col_off + col_row * out_spatial + s] =
                                        in_slice[in_off + ic * H * W + (ih - pad_val) * W + (iw - pad_val)];
                                } else {
                                    col_buf[col_off + col_row * out_spatial + s] = 0;
                                }
                            }
                        }
                    }
                }

                // matmul: weight(out_ch, col_ch) @ col(col_ch, out_spatial) → out(out_ch, out_spatial)
                const out_off = b * out_ch * out_spatial;
                cpu_backend.matmul(T, weight_slice.ptr, col_buf.ptr + col_off, out_data.ptr + out_off, out_ch, col_ch, out_spatial);

                // add bias
                for (0..out_ch) |oc| {
                    for (0..out_spatial) |s| {
                        out_data[out_off + oc * out_spatial + s] += bias_slice[oc];
                    }
                }
            }

            // Backward context
            const Ctx = struct {
                col_buf: []const T,
                weight_data: []const T,
                d_col: []T,
                input_parent: *Node,
                weight_parent: *Node,
                bias_parent: *Node,
            };

            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .col_buf = col_buf,
                .weight_data = weight_slice,
                .d_col = d_col_buf,
                .input_parent = input.node,
                .weight_parent = self.weight.node,
                .bias_parent = self.bias.node,
            };

            const OutVar = VariableMod.Variable(T, .{ batch, out_ch, OH, OW });
            var result = try OutVar.init(out_tensor, allocator, true);
            result.node.parents[0] = input.node;
            result.node.parents[1] = self.weight.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));

                    // dL/dbias = sum over batch and spatial
                    if (c.bias_parent.grad) |bias_grad| {
                        for (0..batch) |b| {
                            const g_off = b * out_ch * out_spatial;
                            for (0..out_ch) |oc| {
                                for (0..out_spatial) |s| {
                                    bias_grad[oc] += grad_out[g_off + oc * out_spatial + s];
                                }
                            }
                        }
                    }

                    // dL/dweight: dW += grad_out_b(out_ch, out_spatial) @ col_b^T(out_spatial, col_ch)
                    if (c.weight_parent.grad) |w_grad| {
                        for (0..batch) |b| {
                            cpu_backend.matmulTransBAccum(T,
                                grad_out.ptr + b * out_ch * out_spatial,
                                c.col_buf.ptr + b * col_ch * out_spatial,
                                w_grad.ptr,
                                out_ch, out_spatial, col_ch);
                        }
                    }

                    // dL/dinput via col2im: d_col = weight^T @ grad_out_b, then scatter
                    if (c.input_parent.grad) |in_grad| {
                        const d_col = c.d_col;
                        for (0..batch) |b| {
                            // d_col(col_ch, out_spatial) = weight^T(col_ch, out_ch) @ grad_out_b(out_ch, out_spatial)
                            cpu_backend.matmulTransA(T,
                                c.weight_data.ptr,
                                grad_out.ptr + b * out_ch * out_spatial,
                                d_col.ptr,
                                col_ch, out_ch, out_spatial);

                            // col2im: scatter d_col back to in_grad
                            const in_off = b * in_ch * H * W;
                            for (0..out_spatial) |s| {
                                const oh = s / OW;
                                const ow = s % OW;
                                for (0..in_ch) |ic| {
                                    for (0..kernel_size) |kh| {
                                        for (0..kernel_size) |kw| {
                                            const ih = oh * stride_val + kh;
                                            const iw = ow * stride_val + kw;
                                            if (ih >= pad_val and ih - pad_val < H and
                                                iw >= pad_val and iw - pad_val < W)
                                            {
                                                const col_row = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                                in_grad[in_off + ic * H * W + (ih - pad_val) * W + (iw - pad_val)] +=
                                                    d_col[col_row * out_spatial + s];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }.backward;

            return result;
        }

        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }
    };
}

// ============================================================
// テスト
// ============================================================

test "Conv2D forward - no padding" {
    const alloc = std.testing.allocator;

    var conv = try Conv2D(f64, 1, 1, 2, 1, 0).init(alloc);
    defer conv.deinit();

    // 手動でweightをセット: [1, 0, 0, 1] (対角カーネル)
    const w = conv.weight.data();
    w[0] = 1;
    w[1] = 0;
    w[2] = 0;
    w[3] = 1;
    conv.bias.data()[0] = 0;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // Input (1, 1, 3, 3): [[1,2,3],[4,5,6],[7,8,9]]
    var input = try VariableMod.Variable(f64, .{ 1, 1, 3, 3 }).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
        false,
    );

    const output = try conv.forward(1, 3, 3, &input, temp);

    // Expected: OH=2, OW=2
    // (0,0): 1*1 + 0*2 + 0*4 + 1*5 = 6
    // (0,1): 1*2 + 0*3 + 0*5 + 1*6 = 8
    // (1,0): 1*4 + 0*5 + 0*7 + 1*8 = 12
    // (1,1): 1*5 + 0*6 + 0*8 + 1*9 = 14
    try std.testing.expectApproxEqAbs(@as(f64, 6), output.constData()[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8), output.constData()[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 12), output.constData()[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 14), output.constData()[3], 1e-10);
}

test "Conv2D backward - analytical check" {
    const alloc = std.testing.allocator;

    var conv = try Conv2D(f64, 1, 1, 2, 1, 0).init(alloc);
    defer conv.deinit();

    const w = conv.weight.data();
    w[0] = 1;
    w[1] = 0;
    w[2] = 0;
    w[3] = 1;
    conv.bias.data()[0] = 0;

    try conv.allocGrad(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 1, 3, 3 }).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
        true,
    );
    input.node.grad = try temp.alloc(f64, 9);
    @memset(input.node.grad.?, 0);

    var output = try conv.forward(1, 3, 3, &input, temp);

    // loss = sum(output), dL/dout = 1
    output.node.grad = try temp.alloc(f64, 4);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| {
        bfn(output.node);
    }

    // dL/dweight = [12, 16, 24, 28]
    try std.testing.expectApproxEqAbs(@as(f64, 12), conv.weight.node.grad.?[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 16), conv.weight.node.grad.?[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 24), conv.weight.node.grad.?[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 28), conv.weight.node.grad.?[3], 1e-10);

    // dL/dbias = 4
    try std.testing.expectApproxEqAbs(@as(f64, 4), conv.bias.node.grad.?[0], 1e-10);

    // dL/dinput = [1, 1, 0, 1, 2, 1, 0, 1, 1]
    const expected_din = [_]f64{ 1, 1, 0, 1, 2, 1, 0, 1, 1 };
    for (0..9) |i| {
        try std.testing.expectApproxEqAbs(expected_din[i], input.node.grad.?[i], 1e-10);
    }
}

test "Conv2D backward - numerical gradient check" {
    const alloc = std.testing.allocator;

    var conv = try Conv2D(f64, 1, 1, 2, 1, 0).init(alloc);
    defer conv.deinit();

    // 特定のweight/biasを設定
    const w = conv.weight.data();
    w[0] = 0.5;
    w[1] = -0.3;
    w[2] = 0.7;
    w[3] = -0.1;
    conv.bias.data()[0] = 0.2;

    try conv.allocGrad(alloc);

    const input_data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    // Analytical gradient
    {
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const temp = arena.allocator();

        var input = try VariableMod.Variable(f64, .{ 1, 1, 3, 3 }).fromSlice(temp, &input_data, true);
        input.node.grad = try temp.alloc(f64, 9);
        @memset(input.node.grad.?, 0);

        var output = try conv.forward(1, 3, 3, &input, temp);
        output.node.grad = try temp.alloc(f64, 4);
        @memset(output.node.grad.?, 1.0);

        if (output.node.backward_fn) |bfn| bfn(output.node);
    }

    // Save analytical gradients
    var analytical_w_grad: [4]f64 = undefined;
    for (0..4) |i| analytical_w_grad[i] = conv.weight.node.grad.?[i];
    const analytical_b_grad = conv.bias.node.grad.?[0];

    // Numerical gradient for weight
    const eps: f64 = 1e-5;
    for (0..4) |i| {
        const original = conv.weight.data()[i];

        conv.weight.data()[i] = original + eps;
        const loss_plus = blk: {
            var arena = std.heap.ArenaAllocator.init(alloc);
            defer arena.deinit();
            const temp = arena.allocator();
            var input = try VariableMod.Variable(f64, .{ 1, 1, 3, 3 }).fromSlice(temp, &input_data, false);
            const output = try conv.forward(1, 3, 3, &input, temp);
            var s: f64 = 0;
            for (output.constData()) |v| s += v;
            break :blk s;
        };

        conv.weight.data()[i] = original - eps;
        const loss_minus = blk: {
            var arena = std.heap.ArenaAllocator.init(alloc);
            defer arena.deinit();
            const temp = arena.allocator();
            var input = try VariableMod.Variable(f64, .{ 1, 1, 3, 3 }).fromSlice(temp, &input_data, false);
            const output = try conv.forward(1, 3, 3, &input, temp);
            var s: f64 = 0;
            for (output.constData()) |v| s += v;
            break :blk s;
        };

        conv.weight.data()[i] = original;
        const numerical_grad = (loss_plus - loss_minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(analytical_w_grad[i], numerical_grad, 1e-5);
    }

    // Numerical gradient for bias
    {
        const original = conv.bias.data()[0];

        conv.bias.data()[0] = original + eps;
        const loss_plus = blk: {
            var arena = std.heap.ArenaAllocator.init(alloc);
            defer arena.deinit();
            const temp = arena.allocator();
            var input = try VariableMod.Variable(f64, .{ 1, 1, 3, 3 }).fromSlice(temp, &input_data, false);
            const output = try conv.forward(1, 3, 3, &input, temp);
            var s: f64 = 0;
            for (output.constData()) |v| s += v;
            break :blk s;
        };

        conv.bias.data()[0] = original - eps;
        const loss_minus = blk: {
            var arena = std.heap.ArenaAllocator.init(alloc);
            defer arena.deinit();
            const temp = arena.allocator();
            var input = try VariableMod.Variable(f64, .{ 1, 1, 3, 3 }).fromSlice(temp, &input_data, false);
            const output = try conv.forward(1, 3, 3, &input, temp);
            var s: f64 = 0;
            for (output.constData()) |v| s += v;
            break :blk s;
        };

        conv.bias.data()[0] = original;
        const numerical_grad = (loss_plus - loss_minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(analytical_b_grad, numerical_grad, 1e-5);
    }
}

test "Conv2D forward - with padding" {
    const alloc = std.testing.allocator;

    // kernel=3, stride=1, padding=1 → same spatial size
    var conv = try Conv2D(f64, 1, 1, 3, 1, 1).init(alloc);
    defer conv.deinit();

    // All-ones kernel
    for (conv.weight.data()) |*v| v.* = 1;
    conv.bias.data()[0] = 0;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 1, 3, 3 }).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
        false,
    );

    const output = try conv.forward(1, 3, 3, &input, temp);

    // OH=3, OW=3 (same padding)
    // (0,0): 0+0+0 + 0+1+2 + 0+4+5 = 12
    // (0,1): 0+0+0 + 1+2+3 + 4+5+6 = 21
    // (1,1): 1+2+3 + 4+5+6 + 7+8+9 = 45
    try std.testing.expectApproxEqAbs(@as(f64, 12), output.constData()[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 21), output.constData()[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 45), output.constData()[4], 1e-10);
}
