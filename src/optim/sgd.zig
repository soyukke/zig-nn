const std = @import("std");
const Allocator = std.mem.Allocator;
const optim_common = @import("common.zig");

/// SGD (Stochastic Gradient Descent) オプティマイザ。
/// momentum 付きもサポート。
///
/// 使用例:
///   var sgd = SGD(f32).init(&.{ .{ .data = layer.weight.data(), .grad = &layer.weight.node.grad }, ... }, 0.01, 0.9);
///   sgd.step();
///   sgd.zeroGrad();
pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Param = optim_common.Param(T);

        params: []const Param,
        lr: T,
        momentum: T,
        velocities: ?[][]T,
        allocator: ?Allocator,

        /// 初期化
        pub fn init(params: []const Param, lr: T, momentum: T) Self {
            return .{
                .params = params,
                .lr = lr,
                .momentum = momentum,
                .velocities = null,
                .allocator = null,
            };
        }

        /// momentum用バッファを確保 (momentum > 0 の場合のみ必要)
        pub fn initMomentum(self: *Self, allocator: Allocator) !void {
            if (self.momentum == 0) return;
            self.allocator = allocator;
            const vels = try allocator.alloc([]T, self.params.len);
            for (self.params, vels) |p, *v| {
                v.* = try allocator.alloc(T, p.data.len);
                @memset(v.*, 0);
            }
            self.velocities = vels;
        }

        /// メモリ解放
        pub fn deinit(self: *Self) void {
            if (self.velocities) |vels| {
                if (self.allocator) |alloc| {
                    for (vels) |v| alloc.free(v);
                    alloc.free(vels);
                }
            }
        }

        /// パラメータ更新
        pub fn step(self: *Self) void {
            for (self.params, 0..) |p, pi| {
                const g = p.grad.* orelse continue;

                if (self.momentum > 0 and self.velocities != null) {
                    const vel = self.velocities.?[pi];
                    for (p.data, g, vel) |*w, gi, *vi| {
                        vi.* = self.momentum * vi.* + gi;
                        w.* -= self.lr * vi.*;
                    }
                } else {
                    for (p.data, g) |*w, gi| {
                        w.* -= self.lr * gi;
                    }
                }
            }
        }

        /// 全パラメータの勾配をゼロにリセット
        pub fn zeroGrad(self: *Self) void {
            optim_common.zeroGrad(T, self.params);
        }
    };
}

test "SGD step" {
    const allocator = std.testing.allocator;

    var data = [_]f32{ 1.0, 2.0, 3.0 };
    var grad_storage = [_]f32{ 0.1, 0.2, 0.3 };
    var grad: ?[]f32 = &grad_storage;

    const params = [_]SGD(f32).Param{
        .{ .data = &data, .grad = &grad },
    };

    var sgd = SGD(f32).init(&params, 0.1, 0);
    _ = allocator;

    sgd.step();

    // w -= lr * grad => 1.0 - 0.1*0.1 = 0.99
    try std.testing.expectApproxEqAbs(@as(f32, 0.99), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.98), data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.97), data[2], 1e-6);

    sgd.zeroGrad();
    for (grad_storage) |g| {
        try std.testing.expectEqual(@as(f32, 0), g);
    }
}
