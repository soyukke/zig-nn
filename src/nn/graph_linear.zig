/// Unified Linear layer (CPU/GPU backend-agnostic)
///
/// compute.Module にパラメータを登録し、forward は anytype ctx で matmul+add を実行する。
/// ctx が MpsRuntime なら graph node (metal_id) を返し、
/// ctx が CpuRuntime なら即時計算 (CpuTensor) を返す。
const compute = @import("../compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;

pub fn Linear(comptime in_dim: usize, comptime out_dim: usize) type {
    return struct {
        w: ParamHandle,
        b: ParamHandle,

        pub fn init(module: anytype) @This() {
            return .{
                .w = module.addParam(&.{ in_dim, out_dim }, .xavier),
                .b = module.addParam(&.{out_dim}, .zeros),
            };
        }

        pub fn forward(self: @This(), ctx: anytype, input: anytype) @TypeOf(input) {
            return ctx.add(ctx.matmul(input, ctx.param(self.w)), ctx.param(self.b));
        }
    };
}

/// Backward-compatible alias
pub const GraphLinear = Linear;
