/// Sequential: comptime レイヤー合成 (PyTorch nn.Sequential 相当)
///
/// 全てコンパイル時に解決され、手書きと完全に同じコードが生成される。
/// ゼロオーバーヘッド。
///
/// Usage:
///   const Model = Sequential(.{ Linear_(128, 64), ReLU, Linear_(64, 10) });
///   const model = Model.init(&module);
///   const out = model.forward(&rt, input);
const std = @import("std");

// ── Activation function wrappers (zero-sized types) ──

pub const ReLU = struct {
    pub fn forward(_: @This(), ctx: anytype, x: anytype) @TypeOf(x) {
        return ctx.relu(x);
    }
};

pub const GELU = struct {
    pub fn forward(_: @This(), ctx: anytype, x: anytype) @TypeOf(x) {
        return ctx.gelu(x);
    }
};

pub const SiLU = struct {
    pub fn forward(_: @This(), ctx: anytype, x: anytype) @TypeOf(x) {
        return ctx.silu(x);
    }
};

pub const Sigmoid = struct {
    pub fn forward(_: @This(), ctx: anytype, x: anytype) @TypeOf(x) {
        return ctx.sigmoid(x);
    }
};

pub const Tanh = struct {
    pub fn forward(_: @This(), ctx: anytype, x: anytype) @TypeOf(x) {
        return ctx.tanh_(x);
    }
};

// ── Sequential ──

pub fn Sequential(comptime layer_types: anytype) type {
    return struct {
        layers: LayersStruct(layer_types),

        pub fn init(module: anytype) @This() {
            var result: @This() = undefined;
            inline for (0..layer_types.len) |i| {
                const name = comptime std.fmt.comptimePrint("layer_{d}", .{i});
                if (comptime @hasDecl(layer_types[i], "init")) {
                    @field(result.layers, name) = layer_types[i].init(module);
                } else {
                    @field(result.layers, name) = .{};
                }
            }
            return result;
        }

        pub fn forward(self: @This(), ctx: anytype, x: anytype) @TypeOf(x) {
            var h = x;
            inline for (0..layer_types.len) |i| {
                const name = comptime std.fmt.comptimePrint("layer_{d}", .{i});
                h = @field(self.layers, name).forward(ctx, h);
            }
            return h;
        }
    };
}

fn LayersStruct(comptime layer_types: anytype) type {
    const N = layer_types.len;
    comptime var field_names: [N][:0]const u8 = undefined;
    comptime var field_types: [N]type = undefined;
    inline for (0..N) |i| {
        field_names[i] = std.fmt.comptimePrint("layer_{d}", .{i});
        field_types[i] = layer_types[i];
    }
    return @Struct(.auto, null, &field_names, &field_types, &@splat(.{}));
}

// ── Tests ──

const testing = std.testing;
const compute = @import("../compute.zig");
const DiffCpuRuntime = @import("../diff_cpu_runtime.zig").DiffCpuRuntime;
const Linear_ = @import("graph_linear.zig").Linear;

test "Sequential: Linear + ReLU + Linear forward+backward" {
    const Model = Sequential(.{
        Linear_(3, 4),
        ReLU,
        Linear_(4, 2),
    });

    var module = compute.Module.init(testing.allocator);
    defer module.deinit();
    const model = Model.init(&module);

    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    // Input [2, 3]
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input = rt.makeNode(&input_data, &.{ 2, 3 }, true);

    const out = model.forward(&rt, input);
    // Output should be [2, 2]
    try testing.expectEqual(@as(usize, 2), out.shape[0]);
    try testing.expectEqual(@as(usize, 2), out.shape[1]);

    // Backward
    const loss = rt.reductionSum(rt.reductionSum(out, -1), 0);
    rt.backward(loss);

    // Verify param gradients are non-zero
    for (0..module.paramCount()) |i| {
        const grad = rt.paramGrad(i);
        var norm: f32 = 0;
        for (grad) |g| norm += g * g;
        try testing.expect(norm > 0);
    }
}

test "Sequential: single layer" {
    const Model = Sequential(.{Linear_(4, 2)});

    var module = compute.Module.init(testing.allocator);
    defer module.deinit();
    const model = Model.init(&module);

    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();
    rt.initParams();

    var data = [_]f32{ 1, 2, 3, 4 };
    const input = rt.makeNode(&data, &.{ 1, 4 }, true);
    const out = model.forward(&rt, input);
    try testing.expectEqual(@as(usize, 2), out.shape[1]);
}

test "Sequential: activations only" {
    const Model = Sequential(.{ GELU, Tanh });

    var module = compute.Module.init(testing.allocator);
    defer module.deinit();
    const model = Model.init(&module);

    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var data = [_]f32{ -1.0, 0.0, 0.5, 1.0 };
    const input = rt.makeNode(&data, &.{4}, true);
    const out = model.forward(&rt, input);

    // tanh(gelu(x))
    for (0..4) |i| {
        const v = data[i];
        const sqrt_2_over_pi: f32 = 0.7978845608028654;
        const inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
        const gelu_val = 0.5 * v * (1.0 + std.math.tanh(inner));
        const expected = std.math.tanh(gelu_val);
        try testing.expectApproxEqAbs(expected, out.data[i], 1e-5);
    }
}
