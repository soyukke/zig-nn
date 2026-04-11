const std = @import("std");
const Allocator = std.mem.Allocator;
const optim_common = @import("common.zig");

/// Adam / AdamW オプティマイザ。
///
/// Adam: adaptive learning rate with first & second moment estimates.
/// AdamW: weight_decay > 0 の場合、decoupled weight decay を適用。
pub fn Adam(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Param = optim_common.Param(T);

        /// デフォルト値付き設定 struct
        pub const Config = struct {
            lr: T = 1e-3,
            beta1: T = 0.9,
            beta2: T = 0.999,
            epsilon: T = 1e-8,
            weight_decay: T = 0.0,
        };

        params: []const Param,
        lr: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        t: usize,
        // first moment (m) and second moment (v) for each param
        m: [][]T,
        v: [][]T,
        allocator: Allocator,

        pub fn init(
            params: []const Param,
            allocator: Allocator,
            lr: T,
            beta1: T,
            beta2: T,
            epsilon: T,
            weight_decay: T,
        ) !Self {
            const m = try allocator.alloc([]T, params.len);
            const v = try allocator.alloc([]T, params.len);
            for (params, m, v) |p, *mi, *vi| {
                mi.* = try allocator.alloc(T, p.data.len);
                @memset(mi.*, 0);
                vi.* = try allocator.alloc(T, p.data.len);
                @memset(vi.*, 0);
            }
            return .{
                .params = params,
                .lr = lr,
                .beta1 = beta1,
                .beta2 = beta2,
                .epsilon = epsilon,
                .weight_decay = weight_decay,
                .t = 0,
                .m = m,
                .v = v,
                .allocator = allocator,
            };
        }

        /// Config struct からの初期化 (デフォルト値を活用できる)
        pub fn initWithConfig(params: []const Param, allocator: Allocator, config: Config) !Self {
            return init(params, allocator, config.lr, config.beta1, config.beta2, config.epsilon, config.weight_decay);
        }

        pub fn deinit(self: *Self) void {
            for (self.m) |mi| self.allocator.free(mi);
            for (self.v) |vi| self.allocator.free(vi);
            self.allocator.free(self.m);
            self.allocator.free(self.v);
        }

        pub fn step(self: *Self) void {
            self.t += 1;
            const t_f: T = @floatFromInt(self.t);
            const bc1 = 1.0 - std.math.pow(T, self.beta1, t_f);
            const bc2 = 1.0 - std.math.pow(T, self.beta2, t_f);

            for (self.params, self.m, self.v) |p, mi, vi| {
                const g = p.grad.* orelse continue;

                for (p.data, g, mi, vi) |*w, gi, *m_j, *v_j| {
                    // AdamW: decoupled weight decay
                    if (self.weight_decay > 0) {
                        w.* -= self.lr * self.weight_decay * w.*;
                    }

                    // Update moments
                    m_j.* = self.beta1 * m_j.* + (1.0 - self.beta1) * gi;
                    v_j.* = self.beta2 * v_j.* + (1.0 - self.beta2) * gi * gi;

                    // Bias correction
                    const m_hat = m_j.* / bc1;
                    const v_hat = v_j.* / bc2;

                    // Update weight
                    w.* -= self.lr * m_hat / (@sqrt(v_hat) + self.epsilon);
                }
            }
        }

        pub fn zeroGrad(self: *Self) void {
            optim_common.zeroGrad(T, self.params);
        }
    };
}

test "Adam step" {
    const allocator = std.testing.allocator;

    var data = [_]f32{ 1.0, 2.0 };
    var grad_storage = [_]f32{ 0.5, -0.5 };
    var grad: ?[]f32 = &grad_storage;

    const params = [_]Adam(f32).Param{
        .{ .data = &data, .grad = &grad },
    };

    var adam = try Adam(f32).init(&params, allocator, 0.001, 0.9, 0.999, 1e-8, 0);
    defer adam.deinit();

    // Run a few steps
    adam.step();

    // After 1 step, weights should have changed
    try std.testing.expect(data[0] != 1.0);
    try std.testing.expect(data[1] != 2.0);

    // data[0] should decrease (positive gradient), data[1] should increase (negative gradient)
    try std.testing.expect(data[0] < 1.0);
    try std.testing.expect(data[1] > 2.0);
}

test "Adam initWithConfig" {
    const allocator = std.testing.allocator;

    var data = [_]f32{ 1.0, 2.0 };
    var grad_storage = [_]f32{ 0.5, -0.5 };
    var grad: ?[]f32 = &grad_storage;

    const params = [_]Adam(f32).Param{
        .{ .data = &data, .grad = &grad },
    };

    // Only specify lr and weight_decay, use defaults for the rest
    var adam = try Adam(f32).initWithConfig(&params, allocator, .{ .lr = 1e-3, .weight_decay = 0.01 });
    defer adam.deinit();

    try std.testing.expectEqual(@as(f32, 1e-3), adam.lr);
    try std.testing.expectEqual(@as(f32, 0.9), adam.beta1);
    try std.testing.expectEqual(@as(f32, 0.999), adam.beta2);
    try std.testing.expectEqual(@as(f32, 1e-8), adam.epsilon);
    try std.testing.expectEqual(@as(f32, 0.01), adam.weight_decay);

    adam.step();
    try std.testing.expect(data[0] < 1.0);
}
