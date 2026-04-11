/// Dropout: training 時にランダムに要素をゼロ化する正則化レイヤー
///
/// PyTorch 互換: training=true で inverted dropout (scale by 1/(1-rate))
/// DiffCpuRuntime / MpsRuntime の dropout op にデリゲートする。
pub fn Dropout(comptime rate: comptime_float) type {
    return struct {
        const Self = @This();

        pub fn forward(_: Self, ctx: anytype, x: anytype) @TypeOf(x) {
            return ctx.dropout(x, rate);
        }
    };
}
