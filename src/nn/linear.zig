const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const ops = @import("../autograd/ops.zig");
const TensorMod = @import("../core/tensor.zig");
const ModuleMixin = @import("module.zig").Module;

/// 全結合 (Dense) レイヤー。
/// output = input @ weight + bias
///
/// - T: スカラー型 (f16, f32, f64)
/// - in_features: 入力次元数
/// - out_features: 出力次元数
///
/// 使用例:
///   var layer = try Linear(f32, 784, 128).init(allocator);
///   defer layer.deinit();
///   var output = try layer.forward(4, &input, arena_allocator);
pub fn Linear(comptime T: type, comptime in_features: usize, comptime out_features: usize) type {
    return struct {
        const Self = @This();
        const M = ModuleMixin(Self);

        /// 重み: (in_features, out_features)
        weight: VariableMod.Variable(T, .{ in_features, out_features }),
        /// バイアス: (out_features)
        bias: VariableMod.Variable(T, .{out_features}),

        /// Xavier 初期化で作成
        pub fn init(allocator: Allocator) !Self {
            const fan_in: T = @floatFromInt(in_features);
            const fan_out: T = @floatFromInt(out_features);
            const limit = @sqrt(6.0 / (fan_in + fan_out));

            const weight_tensor = try TensorMod.Tensor(T, .{ in_features, out_features }).init(allocator);
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

            const weight = try VariableMod.Variable(T, .{ in_features, out_features }).init(
                weight_tensor,
                allocator,
                true,
            );

            const bias_tensor = try TensorMod.Tensor(T, .{out_features}).zeros(allocator);
            const bias = try VariableMod.Variable(T, .{out_features}).init(
                bias_tensor,
                allocator,
                true,
            );

            return .{ .weight = weight, .bias = bias };
        }

        /// メモリ解放
        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }

        /// forward: output = input @ weight + bias
        /// input: (batch, in_features) => output: (batch, out_features)
        /// allocator にはarena allocatorを渡すこと (一時バッファの管理用)
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            input: *VariableMod.Variable(T, .{ batch, in_features }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, out_features }) {
            var mm = try ops.matmul(T, .{ batch, in_features }, .{ in_features, out_features }, input, &self.weight, allocator);
            errdefer mm.deinit();
            return ops.addBias(T, batch, out_features, &mm, &self.bias, allocator);
        }

        /// 勾配をゼロにリセット
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        /// 勾配バッファを確保
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
    };
}

test "Linear init" {
    const allocator = std.testing.allocator;
    var layer = try Linear(f32, 3, 2).init(allocator);
    defer layer.deinit();

    // Weight should have values (Xavier initialized, not all zero)
    var has_nonzero = false;
    for (layer.weight.constData()) |v| {
        if (v != 0) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);

    // Bias should be zero
    for (layer.bias.constData()) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}
