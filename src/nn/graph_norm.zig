/// Unified LayerNorm layer (CPU/GPU backend-agnostic)
///
/// compute.Module にパラメータを登録し、forward は anytype ctx で layerNorm を実行する。
const compute = @import("../compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;

pub fn LayerNorm(comptime dim: usize) type {
    return struct {
        gamma: ParamHandle,
        beta: ParamHandle,

        pub fn init(module: anytype) @This() {
            return .{
                .gamma = module.addParam(&.{dim}, .ones),
                .beta = module.addParam(&.{dim}, .zeros),
            };
        }

        pub fn forward(self: @This(), ctx: anytype, input: anytype) @TypeOf(input) {
            return ctx.layerNorm(input, ctx.param(self.gamma), ctx.param(self.beta), 1e-5, -1);
        }
    };
}

/// Backward-compatible alias
pub const GraphLayerNorm = LayerNorm;
