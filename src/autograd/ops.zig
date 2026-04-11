const std = @import("std");
const Allocator = std.mem.Allocator;
const GraphNodeMod = @import("../core/graph.zig");
const VariableMod = @import("variable.zig");
const TensorMod = @import("../core/tensor.zig");
const shape_utils = @import("../core/shape.zig");
const cpu_backend = @import("../backend/cpu.zig");

/// 微分可能な演算を提供する。
/// 各演算はforward計算を行い、計算グラフにbackward関数を登録する。

/// element-wise add: z = a + b
/// dz/da = 1, dz/db = 1
pub fn add(
    comptime T: type,
    comptime shape: anytype,
    a: *VariableMod.Variable(T, shape),
    b: *VariableMod.Variable(T, shape),
    allocator: Allocator,
) !VariableMod.Variable(T, shape) {
    const Var = VariableMod.Variable(T, shape);
    const Node = GraphNodeMod.GraphNode(T);
    const n = Var.num_elements;

    // Forward
    const result_tensor = try TensorMod.Tensor(T, shape).init(allocator);
    cpu_backend.add(T, a.tensor.data, b.tensor.data, result_tensor.data, n);

    // GraphNode構築
    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents = .{ a.node, b.node, null };

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            // dz/da = 1: cblas_saxpy で高速累積
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        cpu_backend.addAccum(T, grad_out.ptr, g.ptr, n);
                    }
                }
            }
            // dz/db = 1
            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |g| {
                        cpu_backend.addAccum(T, grad_out.ptr, g.ptr, n);
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

/// element-wise mul: z = a * b
/// dz/da = b, dz/db = a
pub fn mul(
    comptime T: type,
    comptime shape: anytype,
    a: *VariableMod.Variable(T, shape),
    b: *VariableMod.Variable(T, shape),
    allocator: Allocator,
) !VariableMod.Variable(T, shape) {
    const Var = VariableMod.Variable(T, shape);
    const Node = GraphNodeMod.GraphNode(T);
    const n = Var.num_elements;

    // Forward
    const result_tensor = try TensorMod.Tensor(T, shape).init(allocator);
    cpu_backend.mul(T, a.tensor.data, b.tensor.data, result_tensor.data, n);

    // backward用にa,bのデータポインタを保存 (arena内で安定)
    const Context = struct {
        a_data: [*]const T,
        b_data: [*]const T,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .a_data = a.tensor.data, .b_data = b.tensor.data };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents = .{ a.node, b.node, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const context: *Context = @ptrCast(@alignCast(self.context.?));

            // dz/da = b
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        for (g, grad_out, context.b_data[0..n]) |*dst, go, bv| dst.* += go * bv;
                    }
                }
            }
            // dz/db = a
            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |g| {
                        for (g, grad_out, context.a_data[0..n]) |*dst, go, av| dst.* += go * av;
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

/// 行列積: z = a @ b
/// A: (M, K), B: (K, N) => Z: (M, N)
/// dz/dA = grad_out @ B^T
/// dz/dB = A^T @ grad_out
pub fn matmul(
    comptime T: type,
    comptime shape_a: anytype,
    comptime shape_b: anytype,
    a: *VariableMod.Variable(T, shape_a),
    b: *VariableMod.Variable(T, shape_b),
    allocator: Allocator,
) !VariableMod.Variable(T, shape_utils.matmulShape(shape_a, shape_b)) {
    const result_shape = comptime shape_utils.matmulShape(shape_a, shape_b);
    const Node = GraphNodeMod.GraphNode(T);
    const M = shape_a[0];
    const K = shape_a[1];
    const N = shape_b[1];
    const result_n = M * N;

    // Forward
    const result_tensor = try TensorMod.Tensor(T, result_shape).init(allocator);
    cpu_backend.matmul(T, a.tensor.data, b.tensor.data, result_tensor.data, M, K, N);

    // backward用キャッシュ (コピー不要: arena内でデータは安定)
    const Context = struct {
        a_data: [*]const T,
        b_data: [*]const T,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .a_data = a.tensor.data, .b_data = b.tensor.data };

    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);
    node.parents = .{ a.node, b.node, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const context: *Context = @ptrCast(@alignCast(self.context.?));

            // dz/dA = grad_out @ B^T  : (M,N) @ (K,N)^T => (M,K)
            // beta=1.0 で直接 grad バッファに累積 (temp alloc + loop 不要)
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        cpu_backend.matmulTransBAccum(T, grad_out.ptr, context.b_data, g.ptr, M, N, K);
                    }
                }
            }

            // dz/dB = A^T @ grad_out  : (M,K)^T @ (M,N) => (K,N)
            // beta=1.0 で直接 grad バッファに累積
            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |g| {
                        cpu_backend.matmulTransAAccum(T, context.a_data, grad_out.ptr, g.ptr, K, M, N);
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

/// ReLU: z = max(0, x)
/// dz/dx = 1 if x > 0, 0 otherwise
pub fn relu(
    comptime T: type,
    comptime shape: anytype,
    x: *VariableMod.Variable(T, shape),
    allocator: Allocator,
) !VariableMod.Variable(T, shape) {
    const Var = VariableMod.Variable(T, shape);
    const Node = GraphNodeMod.GraphNode(T);
    const n = Var.num_elements;

    // Forward
    const result_tensor = try TensorMod.Tensor(T, shape).init(allocator);
    cpu_backend.relu(T, x.tensor.data, result_tensor.data, n);

    // backward用: 入力データポインタを保存 (arena内で安定)
    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x.node;
    node.context = @ptrCast(x.tensor.data);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const x_data: [*]const T = @ptrCast(@alignCast(self.context.?));

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const vec_len = std.simd.suggestVectorLength(T) orelse 4;
                        const vec_count = n / vec_len;
                        const remainder = n % vec_len;
                        for (0..vec_count) |vi| {
                            const off = vi * vec_len;
                            const xv: @Vector(vec_len, T) = x_data[off..][0..vec_len].*;
                            const go_v: @Vector(vec_len, T) = grad_out[off..][0..vec_len].*;
                            const zeros: @Vector(vec_len, T) = @splat(0);
                            const mask = xv > zeros;
                            const gv: @Vector(vec_len, T) = g[off..][0..vec_len].*;
                            g[off..][0..vec_len].* = gv + @select(T, mask, go_v, zeros);
                        }
                        for (0..remainder) |ri| {
                            const idx = vec_count * vec_len + ri;
                            g[idx] += if (x_data[idx] > 0) grad_out[idx] else 0;
                        }
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

/// Sigmoid: z = 1 / (1 + exp(-x))
/// dz/dx = z * (1 - z)
pub fn sigmoid(
    comptime T: type,
    comptime shape: anytype,
    x: *VariableMod.Variable(T, shape),
    allocator: Allocator,
) !VariableMod.Variable(T, shape) {
    const Var = VariableMod.Variable(T, shape);
    const Node = GraphNodeMod.GraphNode(T);
    const n = Var.num_elements;

    // Forward (SIMD vectorized)
    const result_tensor = try TensorMod.Tensor(T, shape).init(allocator);
    const x_data = x.tensor.data;
    const out_data = result_tensor.data;
    const vec_len = std.simd.suggestVectorLength(T) orelse 4;
    const vec_count = n / vec_len;
    const remainder = n % vec_len;
    for (0..vec_count) |vi| {
        const offset = vi * vec_len;
        const v: @Vector(vec_len, T) = x_data[offset..][0..vec_len].*;
        const neg_v = -v;
        const exp_v = @exp(neg_v);
        const ones: @Vector(vec_len, T) = @splat(1.0);
        const sig = ones / (ones + exp_v);
        out_data[offset..][0..vec_len].* = sig;
    }
    for (0..remainder) |i| {
        const idx = vec_count * vec_len + i;
        out_data[idx] = 1.0 / (1.0 + @exp(-x_data[idx]));
    }

    // backward用: sigmoid出力ポインタを保存 (arena内で安定)
    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x.node;
    node.context = @ptrCast(result_tensor.data);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const sig_out: [*]const T = @ptrCast(@alignCast(self.context.?));

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        // dz/dx = z * (1 - z), SIMD vectorized
                        const vl = std.simd.suggestVectorLength(T) orelse 4;
                        const vc = n / vl;
                        const rem = n % vl;
                        for (0..vc) |vi| {
                            const off = vi * vl;
                            const sv: @Vector(vl, T) = sig_out[off..][0..vl].*;
                            const go_v: @Vector(vl, T) = grad_out[off..][0..vl].*;
                            const ones: @Vector(vl, T) = @splat(1.0);
                            const gv: @Vector(vl, T) = g[off..][0..vl].*;
                            g[off..][0..vl].* = gv + go_v * sv * (ones - sv);
                        }
                        for (0..rem) |ri| {
                            const idx = vc * vl + ri;
                            g[idx] += grad_out[idx] * sig_out[idx] * (1.0 - sig_out[idx]);
                        }
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

/// Tanh: z = tanh(x)
/// dz/dx = 1 - z^2
pub fn tanh(
    comptime T: type,
    comptime shape: anytype,
    x: *VariableMod.Variable(T, shape),
    allocator: Allocator,
) !VariableMod.Variable(T, shape) {
    const Var = VariableMod.Variable(T, shape);
    const Node = GraphNodeMod.GraphNode(T);
    const n = Var.num_elements;

    // Forward (SIMD vectorized): tanh(x) = 2*sigmoid(2x) - 1
    const result_tensor = try TensorMod.Tensor(T, shape).init(allocator);
    const x_data = x.tensor.data;
    const out_data = result_tensor.data;
    const vec_len = std.simd.suggestVectorLength(T) orelse 4;
    const vec_count = n / vec_len;
    const remainder = n % vec_len;
    for (0..vec_count) |vi| {
        const offset = vi * vec_len;
        const v: @Vector(vec_len, T) = x_data[offset..][0..vec_len].*;
        const twos: @Vector(vec_len, T) = @splat(2.0);
        const ones: @Vector(vec_len, T) = @splat(1.0);
        const exp_v = @exp(-twos * v);
        const sig2 = ones / (ones + exp_v); // sigmoid(2x)
        out_data[offset..][0..vec_len].* = twos * sig2 - ones;
    }
    for (0..remainder) |i| {
        const idx = vec_count * vec_len + i;
        const s2 = 1.0 / (1.0 + @exp(-2.0 * x_data[idx]));
        out_data[idx] = 2.0 * s2 - 1.0;
    }

    // backward用: tanh出力ポインタを保存 (arena内で安定)
    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x.node;
    node.context = @ptrCast(result_tensor.data);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const tanh_out: [*]const T = @ptrCast(@alignCast(self.context.?));

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        // dz/dx = 1 - z^2, SIMD vectorized
                        const vl = std.simd.suggestVectorLength(T) orelse 4;
                        const vc = n / vl;
                        const rem = n % vl;
                        for (0..vc) |vi| {
                            const off = vi * vl;
                            const tv: @Vector(vl, T) = tanh_out[off..][0..vl].*;
                            const go_v: @Vector(vl, T) = grad_out[off..][0..vl].*;
                            const ones: @Vector(vl, T) = @splat(1.0);
                            const gv: @Vector(vl, T) = g[off..][0..vl].*;
                            g[off..][0..vl].* = gv + go_v * (ones - tv * tv);
                        }
                        for (0..rem) |ri| {
                            const idx = vc * vl + ri;
                            g[idx] += grad_out[idx] * (1.0 - tanh_out[idx] * tanh_out[idx]);
                        }
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

/// SiLU (Swish): z = x * sigmoid(x)
/// dz/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
pub fn silu(
    comptime T: type,
    comptime shape: anytype,
    x: *VariableMod.Variable(T, shape),
    allocator: Allocator,
) !VariableMod.Variable(T, shape) {
    const Var = VariableMod.Variable(T, shape);
    const Node = GraphNodeMod.GraphNode(T);
    const n = Var.num_elements;

    // Forward: z = x * sigmoid(x), SIMD vectorized, sigmoid出力を保存して backward での exp() 再計算を回避
    const result_tensor = try TensorMod.Tensor(T, shape).init(allocator);
    const sig_buf = try allocator.alloc(T, n);
    const x_data = x.tensor.data;
    const out_data = result_tensor.data;
    const vec_len = std.simd.suggestVectorLength(T) orelse 4;
    const vec_count = n / vec_len;
    const remainder = n % vec_len;
    for (0..vec_count) |vi| {
        const offset = vi * vec_len;
        const v: @Vector(vec_len, T) = x_data[offset..][0..vec_len].*;
        const neg_v = -v;
        const exp_v = @exp(neg_v);
        const ones: @Vector(vec_len, T) = @splat(1.0);
        const sig = ones / (ones + exp_v);
        sig_buf[offset..][0..vec_len].* = sig;
        out_data[offset..][0..vec_len].* = v * sig;
    }
    for (0..remainder) |i| {
        const idx = vec_count * vec_len + i;
        const sig = 1.0 / (1.0 + @exp(-x_data[idx]));
        sig_buf[idx] = sig;
        out_data[idx] = x_data[idx] * sig;
    }

    // backward用: x ポインタ + sigmoid 出力を保存
    const Context = struct {
        x_data: [*]const T,
        sig_data: [*]const T,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .x_data = x.tensor.data, .sig_data = sig_buf.ptr };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x.node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const context: *Context = @ptrCast(@alignCast(self.context.?));

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        // dz/dx = sig * (1 + x * (1 - sig)), SIMD vectorized
                        const vl = std.simd.suggestVectorLength(T) orelse 4;
                        const vc = n / vl;
                        const rem = n % vl;
                        for (0..vc) |vi| {
                            const off = vi * vl;
                            const sv: @Vector(vl, T) = context.sig_data[off..][0..vl].*;
                            const xv: @Vector(vl, T) = context.x_data[off..][0..vl].*;
                            const go_v: @Vector(vl, T) = grad_out[off..][0..vl].*;
                            const ones: @Vector(vl, T) = @splat(1.0);
                            const gv: @Vector(vl, T) = g[off..][0..vl].*;
                            g[off..][0..vl].* = gv + go_v * sv * (ones + xv * (ones - sv));
                        }
                        for (0..rem) |ri| {
                            const idx = vc * vl + ri;
                            const sig = context.sig_data[idx];
                            g[idx] += grad_out[idx] * sig * (1.0 + context.x_data[idx] * (1.0 - sig));
                        }
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

/// MSE Loss: L = mean((pred - target)^2)
/// dL/dpred = 2 * (pred - target) / n
pub fn mseLoss(
    comptime T: type,
    comptime shape: anytype,
    pred: *VariableMod.Variable(T, shape),
    target: []const T,
    allocator: Allocator,
) !VariableMod.Variable(T, .{1}) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = VariableMod.Variable(T, shape).num_elements;

    // Forward: mean squared error
    var loss_val: T = 0;
    const pred_data = pred.tensor.data;
    for (0..n) |i| {
        const diff = pred_data[i] - target[i];
        loss_val += diff * diff;
    }
    loss_val /= @floatFromInt(n);

    const result_tensor = try TensorMod.Tensor(T, .{1}).init(allocator);
    result_tensor.data[0] = loss_val;

    // backward用: pred - target を保存
    const Context = struct {
        diff: []T,
        n: usize,
    };
    const ctx = try allocator.create(Context);
    const diff = try allocator.alloc(T, n);
    for (0..n) |i| {
        diff[i] = pred_data[i] - target[i];
    }
    ctx.* = .{ .diff = diff, .n = n };

    const node = try allocator.create(Node);
    node.* = Node.init(1, true);
    node.parents[0] = pred.node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const mse_scale: T = 2.0 / @as(T, @floatFromInt(context.n));

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        for (g, context.diff) |*dst, d| {
                            dst.* += grad_out[0] * mse_scale * d;
                        }
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

/// バイアス加算 (broadcast): z[i,j] = a[i,j] + bias[j]
/// a: (rows, cols), bias: (cols) => z: (rows, cols)
/// dz/da = 1, dz/dbias = sum over rows of grad_out
pub fn addBias(
    comptime T: type,
    comptime rows: usize,
    comptime cols: usize,
    a: *VariableMod.Variable(T, .{ rows, cols }),
    bias: *VariableMod.Variable(T, .{cols}),
    allocator: Allocator,
) !VariableMod.Variable(T, .{ rows, cols }) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = rows * cols;

    // Forward
    const result_tensor = try TensorMod.Tensor(T, .{ rows, cols }).init(allocator);
    for (0..rows) |i| {
        for (0..cols) |j| {
            result_tensor.data[i * cols + j] = a.tensor.data[i * cols + j] + bias.tensor.data[j];
        }
    }

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents = .{ a.node, bias.node, null };

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            // dz/da = 1: cblas_saxpy で高速累積
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        cpu_backend.addAccum(T, grad_out.ptr, g.ptr, n);
                    }
                }
            }
            // dz/dbias = sum over rows (saxpy per row)
            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |g| {
                        for (0..rows) |i| {
                            cpu_backend.addAccum(T, grad_out.ptr + i * cols, g.ptr, cols);
                        }
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

/// Flatten 4D → 2D: (batch, C, H, W) → (batch, C*H*W)
/// CNN出力をLinear層に接続するための操作。
/// backward: パススルー (データレイアウトは同一)
pub fn flatten4dTo2d(
    comptime T: type,
    comptime batch: usize,
    comptime C: usize,
    comptime H: usize,
    comptime W: usize,
    input: *VariableMod.Variable(T, .{ batch, C, H, W }),
    allocator: Allocator,
) !VariableMod.Variable(T, .{ batch, C * H * W }) {
    const n = batch * C * H * W;
    const Node = GraphNodeMod.GraphNode(T);

    const OutTensor = TensorMod.Tensor(T, .{ batch, C * H * W });
    const out_tensor = try OutTensor.init(allocator);
    @memcpy(out_tensor.slice(), input.constData());

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = input.node;

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        for (g, grad_out) |*dst, src| dst.* += src;
                    }
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

/// Flatten 3D → 2D: (d0, d1, d2) → (d0*d1, d2)
/// RNN出力をLinear層に接続するための操作。
/// backward: パススルー (データレイアウトは同一)
pub fn flatten3dTo2d(
    comptime T: type,
    comptime d0: usize,
    comptime d1: usize,
    comptime d2: usize,
    input: *VariableMod.Variable(T, .{ d0, d1, d2 }),
    allocator: Allocator,
) !VariableMod.Variable(T, .{ d0 * d1, d2 }) {
    const n = d0 * d1 * d2;
    const Node = GraphNodeMod.GraphNode(T);

    const out_tensor = try TensorMod.Tensor(T, .{ d0 * d1, d2 }).init(allocator);
    @memcpy(out_tensor.slice(), input.constData());

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = input.node;

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        for (g, grad_out) |*dst, src| dst.* += src;
                    }
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

/// Scale: z = alpha * x (スカラー乗算)
/// dz/dx = alpha
pub fn scale(
    comptime T: type,
    comptime shape: anytype,
    x: *VariableMod.Variable(T, shape),
    alpha: T,
    allocator: Allocator,
) !VariableMod.Variable(T, shape) {
    const Var = VariableMod.Variable(T, shape);
    const Node = GraphNodeMod.GraphNode(T);
    const n = Var.num_elements;

    // Forward (BLAS)
    const result_tensor = try TensorMod.Tensor(T, shape).init(allocator);
    cpu_backend.scale(T, x.tensor.data, alpha, result_tensor.data, n);

    // backward用: alpha値を保存
    const Context = struct {
        alpha_val: T,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .alpha_val = alpha };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x.node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            const context: *Context = @ptrCast(@alignCast(self.context.?));

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        for (g, grad_out) |*dst, go| {
                            dst.* += context.alpha_val * go;
                        }
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

/// Concat last dim: z = concat(a, b, axis=-1)
/// a: (rows, cols_a), b: (rows, cols_b) → z: (rows, cols_a + cols_b)
/// dz/da = grad_out[:, :cols_a], dz/db = grad_out[:, cols_a:]
pub fn concatLastDim(
    comptime T: type,
    comptime rows: usize,
    comptime cols_a: usize,
    comptime cols_b: usize,
    a: *VariableMod.Variable(T, .{ rows, cols_a }),
    b: *VariableMod.Variable(T, .{ rows, cols_b }),
    allocator: Allocator,
) !VariableMod.Variable(T, .{ rows, cols_a + cols_b }) {
    const cols_out = cols_a + cols_b;
    const Node = GraphNodeMod.GraphNode(T);

    // Forward
    const result_tensor = try TensorMod.Tensor(T, .{ rows, cols_out }).init(allocator);
    const out_data = result_tensor.data;
    for (0..rows) |i| {
        for (0..cols_a) |j| {
            out_data[i * cols_out + j] = a.tensor.data[i * cols_a + j];
        }
        for (0..cols_b) |j| {
            out_data[i * cols_out + cols_a + j] = b.tensor.data[i * cols_b + j];
        }
    }

    const node = try allocator.create(Node);
    node.* = Node.init(rows * cols_out, true);
    node.parents = .{ a.node, b.node, null };

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            // dz/da: grad_out の最初の cols_a 列をスライス
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        for (0..rows) |i| {
                            for (0..cols_a) |j| {
                                g[i * cols_a + j] += grad_out[i * cols_out + j];
                            }
                        }
                    }
                }
            }
            // dz/db: grad_out の残りの cols_b 列をスライス
            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |g| {
                        for (0..rows) |i| {
                            for (0..cols_b) |j| {
                                g[i * cols_b + j] += grad_out[i * cols_out + cols_a + j];
                            }
                        }
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

/// Unflatten 2D → 3D: (d0*d1, d2) → (d0, d1, d2)
/// flatten3dTo2d の逆操作。
/// backward: パススルー (データレイアウトは同一)
pub fn unflatten2dTo3d(
    comptime T: type,
    comptime d0: usize,
    comptime d1: usize,
    comptime d2: usize,
    input: *VariableMod.Variable(T, .{ d0 * d1, d2 }),
    allocator: Allocator,
) !VariableMod.Variable(T, .{ d0, d1, d2 }) {
    const n = d0 * d1 * d2;
    const Node = GraphNodeMod.GraphNode(T);

    const out_tensor = try TensorMod.Tensor(T, .{ d0, d1, d2 }).init(allocator);
    @memcpy(out_tensor.slice(), input.constData());

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = input.node;

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        for (g, grad_out) |*dst, src| dst.* += src;
                    }
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
// ヘルパー
// ============================================================

/// 行列転置 (row-major)
/// src: (rows, cols) => dst: (cols, rows)
fn transpose(comptime T: type, src: []const T, dst: []T, rows: usize, cols: usize) void {
    for (0..rows) |i| {
        for (0..cols) |j| {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// ============================================================
// テスト: 数値微分(gradient checking)で検証
// ============================================================

fn numericalGradient(
    comptime T: type,
    comptime shape: anytype,
    comptime f: fn (
        *VariableMod.Variable(T, shape),
        Allocator,
    ) anyerror!VariableMod.Variable(T, .{1}),
    x: *VariableMod.Variable(T, shape),
    allocator: Allocator,
) ![]T {
    const n = VariableMod.Variable(T, shape).num_elements;
    const eps: T = 1e-5;
    const grad_numerical = try allocator.alloc(T, n);

    for (0..n) |i| {
        const original = x.data()[i];

        // f(x + eps)
        x.data()[i] = original + eps;
        var fp = try f(x, allocator);
        const fp_val = fp.constData()[0];
        fp.deinit();

        // f(x - eps)
        x.data()[i] = original - eps;
        var fm = try f(x, allocator);
        const fm_val = fm.constData()[0];
        fm.deinit();

        grad_numerical[i] = (fp_val - fm_val) / (2.0 * eps);
        x.data()[i] = original;
    }

    return grad_numerical;
}

test "ops add backward" {
    const allocator = std.testing.allocator;

    var a = try VariableMod.Variable(f32, .{3}).fromSlice(allocator, &.{ 1, 2, 3 }, true);
    defer a.deinit();
    var b = try VariableMod.Variable(f32, .{3}).fromSlice(allocator, &.{ 4, 5, 6 }, true);
    defer b.deinit();

    // 勾配バッファを事前確保
    a.node.grad = try allocator.alloc(f32, 3);
    @memset(a.node.grad.?, 0);
    b.node.grad = try allocator.alloc(f32, 3);
    @memset(b.node.grad.?, 0);

    var z = try add(f32, .{3}, &a, &b, allocator);
    defer z.deinit();

    // manual backward
    z.node.grad = try allocator.alloc(f32, 3);
    @memset(z.node.grad.?, 1.0); // dL/dz = 1

    if (z.node.backward_fn) |bfn| {
        bfn(z.node);
    }

    // dz/da = 1, dz/db = 1
    for (a.node.grad.?) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 1e-6);
    }
    for (b.node.grad.?) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 1e-6);
    }
}

test "ops mul backward" {
    const allocator = std.testing.allocator;

    var a = try VariableMod.Variable(f32, .{3}).fromSlice(allocator, &.{ 2, 3, 4 }, true);
    defer a.deinit();
    var b = try VariableMod.Variable(f32, .{3}).fromSlice(allocator, &.{ 5, 6, 7 }, true);
    defer b.deinit();

    a.node.grad = try allocator.alloc(f32, 3);
    @memset(a.node.grad.?, 0);
    b.node.grad = try allocator.alloc(f32, 3);
    @memset(b.node.grad.?, 0);

    var z = try mul(f32, .{3}, &a, &b, allocator);
    // mul の Context は a_data, b_data ([*]const f32) を保持するが、
    // これらは a.tensor.data / b.tensor.data へのポインタであり a.deinit()/b.deinit() で解放される。
    // ここでは Context 構造体自体のみ解放する。
    const MulContext = struct { a_data: [*]const f32, b_data: [*]const f32 };
    const ctx: *MulContext = @ptrCast(@alignCast(z.node.context.?));
    defer z.deinit();
    defer allocator.destroy(ctx);

    z.node.grad = try allocator.alloc(f32, 3);
    @memset(z.node.grad.?, 1.0);

    if (z.node.backward_fn) |bfn| {
        bfn(z.node);
    }

    // dz/da = b = {5, 6, 7}
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), a.node.grad.?[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), a.node.grad.?[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), a.node.grad.?[2], 1e-6);

    // dz/db = a = {2, 3, 4}
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), b.node.grad.?[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), b.node.grad.?[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), b.node.grad.?[2], 1e-6);
}

test "ops relu backward" {
    const allocator = std.testing.allocator;

    var x = try VariableMod.Variable(f32, .{4}).fromSlice(allocator, &.{ -1, 0, 2, -3 }, true);
    defer x.deinit();

    x.node.grad = try allocator.alloc(f32, 4);
    @memset(x.node.grad.?, 0);

    var z = try relu(f32, .{4}, &x, allocator);
    // relu の context は x.tensor.data ([*]T) へのポインタ。
    // x.deinit() で解放されるため、ここで free してはいけない。
    defer z.deinit();

    // forward: {0, 0, 2, 0}
    try std.testing.expectEqual(@as(f32, 0), z.constData()[0]);
    try std.testing.expectEqual(@as(f32, 0), z.constData()[1]);
    try std.testing.expectEqual(@as(f32, 2), z.constData()[2]);
    try std.testing.expectEqual(@as(f32, 0), z.constData()[3]);

    z.node.grad = try allocator.alloc(f32, 4);
    @memset(z.node.grad.?, 1.0);

    if (z.node.backward_fn) |bfn| {
        bfn(z.node);
    }

    // dz/dx: {0, 0, 1, 0}
    try std.testing.expectEqual(@as(f32, 0), x.node.grad.?[0]);
    try std.testing.expectEqual(@as(f32, 0), x.node.grad.?[1]);
    try std.testing.expectEqual(@as(f32, 1), x.node.grad.?[2]);
    try std.testing.expectEqual(@as(f32, 0), x.node.grad.?[3]);
}

test "ops sigmoid backward - gradient check" {
    const allocator = std.testing.allocator;

    var x = try VariableMod.Variable(f32, .{3}).fromSlice(allocator, &.{ -1.0, 0.0, 1.0 }, true);
    defer x.deinit();

    // Analytical gradient
    x.node.grad = try allocator.alloc(f32, 3);
    @memset(x.node.grad.?, 0);

    var z = try sigmoid(f32, .{3}, &x, allocator);
    // sigmoid の context は result_tensor.data ([*]T) = z.tensor.data へのポインタ。
    // z.deinit() で解放されるため、ここで free してはいけない。
    defer z.deinit();

    z.node.grad = try allocator.alloc(f32, 3);
    @memset(z.node.grad.?, 1.0);

    if (z.node.backward_fn) |bfn| {
        bfn(z.node);
    }

    // Analytical gradient check: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    const x_vals = [_]f32{ -1.0, 0.0, 1.0 };
    for (0..3) |i| {
        const s = 1.0 / (1.0 + @exp(-x_vals[i]));
        const expected_grad = s * (1.0 - s);
        try std.testing.expectApproxEqAbs(expected_grad, x.node.grad.?[i], 1e-6);
    }
}

test "ops matmul backward - gradient check" {
    const allocator = std.testing.allocator;

    // A: 2x3, B: 3x2 => C: 2x2
    var a = try VariableMod.Variable(f32, .{ 2, 3 }).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, true);
    defer a.deinit();
    var b = try VariableMod.Variable(f32, .{ 3, 2 }).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, true);
    defer b.deinit();

    a.node.grad = try allocator.alloc(f32, 6);
    @memset(a.node.grad.?, 0);
    b.node.grad = try allocator.alloc(f32, 6);
    @memset(b.node.grad.?, 0);

    var z = try matmul(f32, .{ 2, 3 }, .{ 3, 2 }, &a, &b, allocator);
    // matmul の Context は a_data, b_data ([*]const f32) を保持するが、
    // これらは a.tensor.data / b.tensor.data へのポインタであり a.deinit()/b.deinit() で解放される。
    // Context 構造体自体のみ解放する。
    const MatmulContext = struct { a_data: [*]const f32, b_data: [*]const f32 };
    const matmul_ctx: *MatmulContext = @ptrCast(@alignCast(z.node.context.?));
    defer z.deinit();
    defer allocator.destroy(matmul_ctx);

    // z = [[58,64],[139,154]]
    try std.testing.expectApproxEqAbs(@as(f32, 58), z.constData()[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 64), z.constData()[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 139), z.constData()[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 154), z.constData()[3], 1e-4);

    // sum(z) を loss として backward
    z.node.grad = try allocator.alloc(f32, 4);
    @memset(z.node.grad.?, 1.0);

    if (z.node.backward_fn) |bfn| {
        bfn(z.node);
    }

    // grad_A = grad_out @ B^T = [[1,1],[1,1]] @ [[7,9,11],[8,10,12]] = [[15,19,23],[15,19,23]]
    try std.testing.expectApproxEqAbs(@as(f32, 15), a.node.grad.?[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 19), a.node.grad.?[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 23), a.node.grad.?[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 15), a.node.grad.?[3], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 19), a.node.grad.?[4], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 23), a.node.grad.?[5], 1e-4);

    // grad_B = A^T @ grad_out = [[1,4],[2,5],[3,6]] @ [[1,1],[1,1]] = [[5,5],[7,7],[9,9]]
    try std.testing.expectApproxEqAbs(@as(f32, 5), b.node.grad.?[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5), b.node.grad.?[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 7), b.node.grad.?[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 7), b.node.grad.?[3], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 9), b.node.grad.?[4], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 9), b.node.grad.?[5], 1e-4);
}

test "ops mse_loss backward" {
    const allocator = std.testing.allocator;

    var pred = try VariableMod.Variable(f32, .{3}).fromSlice(allocator, &.{ 1.0, 2.0, 3.0 }, true);
    defer pred.deinit();
    const target = [_]f32{ 1.5, 2.5, 3.5 };

    pred.node.grad = try allocator.alloc(f32, 3);
    @memset(pred.node.grad.?, 0);

    var loss = try mseLoss(f32, .{3}, &pred, &target, allocator);
    const loss_ctx: *struct { diff: []f32, n: usize } = @ptrCast(@alignCast(loss.node.context.?));
    const loss_diff = loss_ctx.diff;
    defer loss.deinit();
    defer allocator.destroy(loss_ctx);
    defer allocator.free(loss_diff);

    // loss = mean((-0.5)^2 + (-0.5)^2 + (-0.5)^2) = 0.25
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), loss.constData()[0], 1e-6);

    // backward
    const engine_mod = @import("engine.zig");
    var engine = engine_mod.GradEngine(f32).init(allocator);
    defer engine.deinit();

    try engine.backward(loss.node);

    // dL/dpred = 2 * (pred - target) / n = 2 * (-0.5) / 3 = -1/3
    for (pred.node.grad.?) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, -1.0 / 3.0), g, 1e-5);
    }
}

test "ops silu backward - gradient check" {
    const allocator = std.testing.allocator;

    var x = try VariableMod.Variable(f32, .{4}).fromSlice(allocator, &.{ -2.0, -0.5, 0.0, 1.5 }, true);
    defer x.deinit();

    // Analytical gradient
    x.node.grad = try allocator.alloc(f32, 4);
    @memset(x.node.grad.?, 0);

    var z = try silu(f32, .{4}, &x, allocator);
    // silu の Context は x_data ([*]const T, x所有) と sig_data ([*]const T, 別途alloc) を保持。
    // sig_data と Context 構造体を解放する。x_data は x.deinit() で解放される。
    const SiluContext = struct { x_data: [*]const f32, sig_data: [*]const f32 };
    const silu_ctx: *SiluContext = @ptrCast(@alignCast(z.node.context.?));
    defer allocator.free(silu_ctx.sig_data[0..4]);
    defer allocator.destroy(silu_ctx);
    defer z.deinit();

    // Forward check: silu(0) = 0, silu(x) = x * sigmoid(x)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), z.constData()[2], 1e-6);

    z.node.grad = try allocator.alloc(f32, 4);
    @memset(z.node.grad.?, 1.0);

    if (z.node.backward_fn) |bfn| {
        bfn(z.node);
    }

    // Numerical gradient check
    const x_vals = [_]f32{ -2.0, -0.5, 0.0, 1.5 };
    const eps: f32 = 1e-5;
    for (0..4) |i| {
        const xv = x_vals[i];
        const f_plus = (xv + eps) * (1.0 / (1.0 + @exp(-(xv + eps))));
        const f_minus = (xv - eps) * (1.0 / (1.0 + @exp(-(xv - eps))));
        const numerical_grad = (f_plus - f_minus) / (2.0 * eps);
        try std.testing.expectApproxEqAbs(numerical_grad, x.node.grad.?[i], 1e-2);
    }
}

test "ops scale backward" {
    const allocator = std.testing.allocator;

    var x = try VariableMod.Variable(f32, .{3}).fromSlice(allocator, &.{ 1.0, 2.0, 3.0 }, true);
    defer x.deinit();

    x.node.grad = try allocator.alloc(f32, 3);
    @memset(x.node.grad.?, 0);

    var z = try scale(f32, .{3}, &x, 2.5, allocator);
    // scale の Context は allocator.create で割り当て。z.deinit() では解放されないため手動で解放。
    const ScaleContext = struct { alpha_val: f32 };
    const scale_ctx: *ScaleContext = @ptrCast(@alignCast(z.node.context.?));
    defer allocator.destroy(scale_ctx);
    defer z.deinit();

    // Forward: z = 2.5 * x = {2.5, 5.0, 7.5}
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), z.constData()[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), z.constData()[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), z.constData()[2], 1e-6);

    z.node.grad = try allocator.alloc(f32, 3);
    @memset(z.node.grad.?, 1.0);

    if (z.node.backward_fn) |bfn| bfn(z.node);

    // dz/dx = alpha = 2.5
    for (x.node.grad.?) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), g, 1e-6);
    }
}

test "ops concatLastDim forward and backward" {
    const allocator = std.testing.allocator;

    // a: (2, 3), b: (2, 2)
    var a = try VariableMod.Variable(f32, .{ 2, 3 }).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, true);
    defer a.deinit();
    var b = try VariableMod.Variable(f32, .{ 2, 2 }).fromSlice(allocator, &.{ 7, 8, 9, 10 }, true);
    defer b.deinit();

    a.node.grad = try allocator.alloc(f32, 6);
    @memset(a.node.grad.?, 0);
    b.node.grad = try allocator.alloc(f32, 4);
    @memset(b.node.grad.?, 0);

    var z = try concatLastDim(f32, 2, 3, 2, &a, &b, allocator);
    defer z.deinit();

    // Forward: z = [[1,2,3,7,8], [4,5,6,9,10]]
    const out = z.constData();
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7), out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8), out[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), out[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5), out[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), out[7], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9), out[8], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10), out[9], 1e-6);

    z.node.grad = try allocator.alloc(f32, 10);
    // grad_out = [[1,1,1,1,1], [1,1,1,1,1]]
    @memset(z.node.grad.?, 1.0);

    if (z.node.backward_fn) |bfn| bfn(z.node);

    // dz/da: all 1s
    for (a.node.grad.?) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 1e-6);
    }
    // dz/db: all 1s
    for (b.node.grad.?) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 1e-6);
    }
}

test "ops unflatten2dTo3d" {
    const allocator = std.testing.allocator;

    // input: (6, 2) → output: (2, 3, 2)
    var input = try VariableMod.Variable(f32, .{ 6, 2 }).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, true);
    defer input.deinit();

    input.node.grad = try allocator.alloc(f32, 12);
    @memset(input.node.grad.?, 0);

    var z = try unflatten2dTo3d(f32, 2, 3, 2, &input, allocator);
    defer z.deinit();

    // Data should be identical
    for (0..12) |i| {
        try std.testing.expectApproxEqAbs(input.constData()[i], z.constData()[i], 1e-6);
    }

    z.node.grad = try allocator.alloc(f32, 12);
    @memset(z.node.grad.?, 1.0);

    if (z.node.backward_fn) |bfn| bfn(z.node);

    for (input.node.grad.?) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 1e-6);
    }
}
