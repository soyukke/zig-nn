/// Unified Embedding layer (CPU/GPU backend-agnostic)
///
/// compute.Module にパラメータを登録し、forward は anytype ctx で gather を実行する。
/// ctx が MpsRuntime なら graph node を返し、
/// ctx が DiffCpuRuntime なら自動微分対応テンソルを返す。
const compute = @import("../compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;

pub fn Embedding(comptime vocab_size: usize, comptime embed_dim: usize) type {
    return struct {
        w: ParamHandle,

        pub fn init(module: anytype) @This() {
            return .{
                .w = module.addParam(&.{ vocab_size, embed_dim }, .xavier),
            };
        }

        pub fn forward(
            self: @This(),
            ctx: anytype,
            indices: []const u32,
        ) @TypeOf(ctx.param(self.w)) {
            return ctx.gather(ctx.param(self.w), indices);
        }
    };
}
