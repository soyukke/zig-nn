const std = @import("std");
const Allocator = std.mem.Allocator;
const optim_common = @import("common.zig");

/// RMSProp オプティマイザ。
/// 二乗勾配の移動平均で適応的学習率を実現する。
///
/// v_t = rho * v_{t-1} + (1 - rho) * g_t^2
/// w_t = w_{t-1} - lr * g_t / (sqrt(v_t) + eps)
pub fn RMSProp(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Param = optim_common.Param(T);

        params: []const Param,
        lr: T,
        rho: T,
        epsilon: T,
        v: [][]T,
        allocator: Allocator,

        pub fn init(
            params: []const Param,
            allocator: Allocator,
            lr: T,
            rho: T,
            epsilon: T,
        ) !Self {
            const v = try allocator.alloc([]T, params.len);
            for (params, v) |p, *vi| {
                vi.* = try allocator.alloc(T, p.data.len);
                @memset(vi.*, 0);
            }
            return .{
                .params = params,
                .lr = lr,
                .rho = rho,
                .epsilon = epsilon,
                .v = v,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.v) |vi| self.allocator.free(vi);
            self.allocator.free(self.v);
        }

        pub fn step(self: *Self) void {
            for (self.params, self.v) |p, vi| {
                const g = p.grad.* orelse continue;
                for (p.data, g, vi) |*w, gi, *v_j| {
                    v_j.* = self.rho * v_j.* + (1.0 - self.rho) * gi * gi;
                    w.* -= self.lr * gi / (@sqrt(v_j.*) + self.epsilon);
                }
            }
        }

        pub fn zeroGrad(self: *Self) void {
            optim_common.zeroGrad(T, self.params);
        }
    };
}

// ============================================================
// テスト
// ============================================================

test "RMSProp step" {
    const allocator = std.testing.allocator;

    var data = [_]f32{ 1.0, 2.0 };
    var grad_storage = [_]f32{ 0.5, -0.5 };
    var grad: ?[]f32 = &grad_storage;

    const params = [_]RMSProp(f32).Param{
        .{ .data = &data, .grad = &grad },
    };

    var rmsprop = try RMSProp(f32).init(&params, allocator, 0.01, 0.99, 1e-8);
    defer rmsprop.deinit();

    rmsprop.step();

    // After 1 step: v = 0.01 * g^2 = 0.01 * 0.25 = 0.0025
    // w -= lr * g / (sqrt(v) + eps) = 0.01 * 0.5 / (sqrt(0.0025) + 1e-8)
    //    = 0.01 * 0.5 / 0.05 = 0.1
    // data[0] should decrease (positive gradient)
    try std.testing.expect(data[0] < 1.0);
    // data[1] should increase (negative gradient)
    try std.testing.expect(data[1] > 2.0);

    // Verify specific values
    // v = (1-0.99) * 0.25 = 0.0025, sqrt = 0.05
    // update = 0.01 * 0.5 / 0.05 = 0.1
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.1), data[1], 1e-5);
}

test "RMSProp multiple steps" {
    const allocator = std.testing.allocator;

    var data = [_]f32{5.0};
    var grad_storage = [_]f32{1.0};
    var grad: ?[]f32 = &grad_storage;

    const params = [_]RMSProp(f32).Param{
        .{ .data = &data, .grad = &grad },
    };

    var rmsprop = try RMSProp(f32).init(&params, allocator, 0.01, 0.9, 1e-8);
    defer rmsprop.deinit();

    // Multiple steps should converge (v grows, update shrinks)
    const initial = data[0];
    for (0..10) |_| rmsprop.step();

    try std.testing.expect(data[0] < initial);
}
