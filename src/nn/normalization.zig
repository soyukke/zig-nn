const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");
const ModuleMixin = @import("module.zig").Module;

/// BatchNorm1d: 入力 (batch, features) のバッチ正規化。
/// 各 feature について batch 方向の平均・分散で正規化する。
/// output = gamma * (x - mean) / sqrt(var + eps) + beta
pub fn BatchNorm1d(comptime T: type, comptime num_features: usize) type {
    return struct {
        const Self = @This();
        const M = ModuleMixin(Self);

        gamma: VariableMod.Variable(T, .{num_features}),
        beta: VariableMod.Variable(T, .{num_features}),
        running_mean: [num_features]T,
        running_var: [num_features]T,
        training: bool,
        momentum: T,
        eps: T,

        pub fn init(allocator: Allocator) !Self {
            const gamma_tensor = try TensorMod.Tensor(T, .{num_features}).fill(allocator, 1);
            const gamma = try VariableMod.Variable(T, .{num_features}).init(gamma_tensor, allocator, true);
            const beta_tensor = try TensorMod.Tensor(T, .{num_features}).zeros(allocator);
            const beta = try VariableMod.Variable(T, .{num_features}).init(beta_tensor, allocator, true);

            var rm: [num_features]T = undefined;
            @memset(&rm, 0);
            var rv: [num_features]T = undefined;
            @memset(&rv, 1);

            return .{
                .gamma = gamma,
                .beta = beta,
                .running_mean = rm,
                .running_var = rv,
                .training = true,
                .momentum = 0.1,
                .eps = 1e-5,
            };
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }

        pub fn forward(
            self: *Self,
            comptime batch: usize,
            input: *VariableMod.Variable(T, .{ batch, num_features }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, num_features }) {
            const m: T = @floatFromInt(batch);
            const Node = GraphNodeMod.GraphNode(T);

            const out_tensor = try TensorMod.Tensor(T, .{ batch, num_features }).init(allocator);
            const out_data = out_tensor.slice();
            const in_data = input.constData();
            const gamma_data = self.gamma.constData();
            const beta_data = self.beta.constData();

            var mean: [num_features]T = undefined;
            var invstd: [num_features]T = undefined;

            if (self.training) {
                for (0..num_features) |f| {
                    var sum: T = 0;
                    for (0..batch) |b| sum += in_data[b * num_features + f];
                    mean[f] = sum / m;

                    var var_sum: T = 0;
                    for (0..batch) |b| {
                        const diff = in_data[b * num_features + f] - mean[f];
                        var_sum += diff * diff;
                    }
                    const variance = var_sum / m;
                    invstd[f] = 1.0 / @sqrt(variance + self.eps);

                    self.running_mean[f] = (1.0 - self.momentum) * self.running_mean[f] + self.momentum * mean[f];
                    self.running_var[f] = (1.0 - self.momentum) * self.running_var[f] + self.momentum * variance;
                }
            } else {
                for (0..num_features) |f| {
                    mean[f] = self.running_mean[f];
                    invstd[f] = 1.0 / @sqrt(self.running_var[f] + self.eps);
                }
            }

            for (0..batch) |b| {
                for (0..num_features) |f| {
                    const x_hat = (in_data[b * num_features + f] - mean[f]) * invstd[f];
                    out_data[b * num_features + f] = gamma_data[f] * x_hat + beta_data[f];
                }
            }

            const Ctx = struct {
                in_data: []const T,
                mean: [num_features]T,
                invstd: [num_features]T,
                gamma_data: []const T,
            };

            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .in_data = in_data,
                .mean = mean,
                .invstd = invstd,
                .gamma_data = gamma_data,
            };

            const OutVar = VariableMod.Variable(T, .{ batch, num_features });
            var result = try OutVar.init(out_tensor, allocator, true);
            result.node.parents[0] = input.node;
            result.node.parents[1] = self.gamma.node;
            result.node.parents[2] = self.beta.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));

                    if (node.parents[1].?.grad) |gamma_grad| {
                        for (0..num_features) |f| {
                            var sum_dy_xhat: T = 0;
                            for (0..batch) |b| {
                                const x_hat = (c.in_data[b * num_features + f] - c.mean[f]) * c.invstd[f];
                                sum_dy_xhat += grad_out[b * num_features + f] * x_hat;
                            }
                            gamma_grad[f] += sum_dy_xhat;
                        }
                    }

                    if (node.parents[2].?.grad) |beta_grad| {
                        for (0..num_features) |f| {
                            var sum_dy: T = 0;
                            for (0..batch) |b| sum_dy += grad_out[b * num_features + f];
                            beta_grad[f] += sum_dy;
                        }
                    }

                    if (node.parents[0].?.grad) |in_grad| {
                        for (0..num_features) |f| {
                            var x_hat_buf: [batch]T = undefined;
                            var s1: T = 0;
                            var s2: T = 0;
                            for (0..batch) |b| {
                                const x_hat = (c.in_data[b * num_features + f] - c.mean[f]) * c.invstd[f];
                                x_hat_buf[b] = x_hat;
                                const dx_hat = grad_out[b * num_features + f] * c.gamma_data[f];
                                s1 += dx_hat;
                                s2 += dx_hat * x_hat;
                            }
                            for (0..batch) |b| {
                                const dx_hat = grad_out[b * num_features + f] * c.gamma_data[f];
                                in_grad[b * num_features + f] += c.invstd[f] * (dx_hat - s1 / m - x_hat_buf[b] * s2 / m);
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

/// LayerNorm: 入力 (batch, features) のレイヤー正規化。
/// 各サンプルについて features 方向で正規化する。
pub fn LayerNorm(comptime T: type, comptime num_features: usize) type {
    return struct {
        const Self = @This();
        const M = ModuleMixin(Self);

        gamma: VariableMod.Variable(T, .{num_features}),
        beta: VariableMod.Variable(T, .{num_features}),
        eps: T,

        pub fn init(allocator: Allocator) !Self {
            const gamma_tensor = try TensorMod.Tensor(T, .{num_features}).fill(allocator, 1);
            const gamma = try VariableMod.Variable(T, .{num_features}).init(gamma_tensor, allocator, true);
            const beta_tensor = try TensorMod.Tensor(T, .{num_features}).zeros(allocator);
            const beta = try VariableMod.Variable(T, .{num_features}).init(beta_tensor, allocator, true);

            return .{ .gamma = gamma, .beta = beta, .eps = 1e-5 };
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }

        pub fn forward(
            self: *Self,
            comptime batch: usize,
            input: *VariableMod.Variable(T, .{ batch, num_features }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, num_features }) {
            const n_f: T = @floatFromInt(num_features);
            const Node = GraphNodeMod.GraphNode(T);

            const out_tensor = try TensorMod.Tensor(T, .{ batch, num_features }).init(allocator);
            const out_data = out_tensor.slice();
            const in_data = input.constData();
            const gamma_data = self.gamma.constData();
            const beta_data = self.beta.constData();

            // Per-sample mean & invstd
            const Ctx = struct {
                in_data: []const T,
                mean: [batch]T,
                invstd: [batch]T,
                gamma_data: []const T,
            };

            var mean: [batch]T = undefined;
            var invstd: [batch]T = undefined;

            for (0..batch) |b| {
                var sum: T = 0;
                for (0..num_features) |f| sum += in_data[b * num_features + f];
                mean[b] = sum / n_f;

                var var_sum: T = 0;
                for (0..num_features) |f| {
                    const diff = in_data[b * num_features + f] - mean[b];
                    var_sum += diff * diff;
                }
                invstd[b] = 1.0 / @sqrt(var_sum / n_f + self.eps);
            }

            for (0..batch) |b| {
                for (0..num_features) |f| {
                    const x_hat = (in_data[b * num_features + f] - mean[b]) * invstd[b];
                    out_data[b * num_features + f] = gamma_data[f] * x_hat + beta_data[f];
                }
            }

            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .in_data = in_data,
                .mean = mean,
                .invstd = invstd,
                .gamma_data = gamma_data,
            };

            const OutVar = VariableMod.Variable(T, .{ batch, num_features });
            var result = try OutVar.init(out_tensor, allocator, true);
            result.node.parents[0] = input.node;
            result.node.parents[1] = self.gamma.node;
            result.node.parents[2] = self.beta.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));

                    // dL/dgamma, dL/dbeta
                    if (node.parents[1].?.grad) |gamma_grad| {
                        for (0..num_features) |f| {
                            var s: T = 0;
                            for (0..batch) |b| {
                                const x_hat = (c.in_data[b * num_features + f] - c.mean[b]) * c.invstd[b];
                                s += grad_out[b * num_features + f] * x_hat;
                            }
                            gamma_grad[f] += s;
                        }
                    }

                    if (node.parents[2].?.grad) |beta_grad| {
                        for (0..num_features) |f| {
                            var s: T = 0;
                            for (0..batch) |b| s += grad_out[b * num_features + f];
                            beta_grad[f] += s;
                        }
                    }

                    // dL/dinput
                    if (node.parents[0].?.grad) |in_grad| {
                        for (0..batch) |b| {
                            var x_hat_buf: [num_features]T = undefined;
                            var s1: T = 0;
                            var s2: T = 0;
                            for (0..num_features) |f| {
                                const x_hat = (c.in_data[b * num_features + f] - c.mean[b]) * c.invstd[b];
                                x_hat_buf[f] = x_hat;
                                const dx_hat = grad_out[b * num_features + f] * c.gamma_data[f];
                                s1 += dx_hat;
                                s2 += dx_hat * x_hat;
                            }
                            for (0..num_features) |f| {
                                const dx_hat = grad_out[b * num_features + f] * c.gamma_data[f];
                                in_grad[b * num_features + f] += c.invstd[b] * (dx_hat - s1 / n_f - x_hat_buf[f] * s2 / n_f);
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

test "BatchNorm1d forward - zero mean unit variance" {
    const alloc = std.testing.allocator;

    var bn = try BatchNorm1d(f64, 2).init(alloc);
    defer bn.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // (batch=3, features=2)
    var input = try VariableMod.Variable(f64, .{ 3, 2 }).fromSlice(
        temp,
        &[_]f64{ 1, 10, 2, 20, 3, 30 },
        false,
    );

    const output = try bn.forward(3, &input, temp);
    const out = output.constData();

    // Feature 0: mean=2, std=sqrt(2/3) → normalized should be {-1.2247, 0, 1.2247}
    // Feature 1: mean=20, std=sqrt(200/3) → same normalized values
    // With gamma=1, beta=0: output = normalized
    const expected_0 = [_]f64{ -1.2247448713916, 0, 1.2247448713916 };
    for (0..3) |b| {
        try std.testing.expectApproxEqAbs(expected_0[b], out[b * 2 + 0], 1e-4);
    }
}

test "BatchNorm1d backward" {
    const alloc = std.testing.allocator;

    var bn = try BatchNorm1d(f64, 2).init(alloc);
    defer bn.deinit();
    try bn.allocGrad(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 3, 2 }).fromSlice(
        temp,
        &[_]f64{ 1, 10, 2, 20, 3, 30 },
        true,
    );
    input.node.grad = try temp.alloc(f64, 6);
    @memset(input.node.grad.?, 0);

    var output = try bn.forward(3, &input, temp);
    output.node.grad = try temp.alloc(f64, 6);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| bfn(output.node);

    // dL/dbeta should be batch_size = 3 for each feature
    try std.testing.expectApproxEqAbs(@as(f64, 3), bn.beta.node.grad.?[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 3), bn.beta.node.grad.?[1], 1e-6);

    // dL/dgamma should be sum of x_hat * 1 = sum of x_hat ≈ 0 (normalized values sum to ~0)
    try std.testing.expectApproxEqAbs(@as(f64, 0), bn.gamma.node.grad.?[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0), bn.gamma.node.grad.?[1], 1e-6);
}

test "LayerNorm forward" {
    const alloc = std.testing.allocator;

    var ln = try LayerNorm(f64, 3).init(alloc);
    defer ln.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // (batch=2, features=3)
    var input = try VariableMod.Variable(f64, .{ 2, 3 }).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4, 5, 6 },
        false,
    );

    const output = try ln.forward(2, &input, temp);
    const out = output.constData();

    // Sample 0: mean=2, var=2/3 → invstd = 1/sqrt(2/3 + 1e-5)
    // x_hat = {-1.2247, 0, 1.2247} (approx)
    // With gamma=1, beta=0
    try std.testing.expectApproxEqAbs(@as(f64, -1.2247448713916), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2247448713916), out[2], 1e-5);
}

test "LayerNorm backward" {
    const alloc = std.testing.allocator;

    var ln = try LayerNorm(f64, 3).init(alloc);
    defer ln.deinit();
    try ln.allocGrad(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 2, 3 }).fromSlice(
        temp,
        &[_]f64{ 1, 2, 3, 4, 5, 6 },
        true,
    );
    input.node.grad = try temp.alloc(f64, 6);
    @memset(input.node.grad.?, 0);

    var output = try ln.forward(2, &input, temp);
    output.node.grad = try temp.alloc(f64, 6);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| bfn(output.node);

    // dL/dbeta = sum over batch = 2 for each feature
    try std.testing.expectApproxEqAbs(@as(f64, 2), ln.beta.node.grad.?[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 2), ln.beta.node.grad.?[1], 1e-6);
}
