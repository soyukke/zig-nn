// diff/common/reduce.zig: Reduction (sum / mean) backward の数式一極集中
//
// reductionSum / reductionMean の backward はいずれも「削減された軸に沿って
// go を parent shape に scatter する」操作で、scale factor のみ異なる:
//   sum  → scale = 1
//   mean → scale = 1 / reduction_size
//
// 各 backend runtime は、forward 時に Case (削減パターン) と scale を context に
// 詰めておき、backward で共通の scatter() を呼ぶだけで済む。

/// Reduction 軸の組み合わせを識別する。
/// Zig 0.16 では @enumFromInt(u8) で来るので u8 を明示。
pub const Case = enum(u8) {
    /// 全要素を 1 スカラーに縮約 (1D 入力). ga[i] += go[0] * scale
    all,
    /// 2D 入力の axis=0 縮約 ([rows, cols] → [1, cols]). ga[i,j] += go[j] * scale
    axis0_2d,
    /// 2D 入力の axis=1 縮約 ([rows, cols] → [rows, 1]). ga[i,j] += go[i] * scale
    axis1_2d,
};

/// 削減方式ごとの scale factor:
///   sum  = 1.0
///   mean = 1.0 / reduction_size
pub fn scaleSum() f32 {
    return 1.0;
}
pub fn scaleMean(reduction_size: usize) f32 {
    return 1.0 / @as(f32, @floatFromInt(reduction_size));
}

/// Scatter backward: go を parent の shape に散布し、scale 倍して accumulate。
///
/// 引数:
///   ga            : parent の勾配バッファ先頭 ([*]f32)。+= の左辺。
///   go            : 出力勾配バッファ先頭 ([*]const f32)。
///   parent_shape  : parent tensor の shape slice (case に応じて 1〜2 次元を参照)。
///   case          : 削減軸パターン。
///   scale         : sum なら 1.0、mean なら 1.0 / reduction_size。
///
/// CPU runtime は `[]f32` の `.ptr` を、MPS runtime は `bufPtr(...)` を渡せば
/// いずれも同じ実装で backward が走る。
pub fn scatter(
    ga: [*]f32,
    go: [*]const f32,
    parent_shape: []const usize,
    case: Case,
    scale: f32,
) void {
    switch (case) {
        .all => {
            var total: usize = 1;
            for (parent_shape) |s| total *= s;
            const gv = go[0] * scale;
            for (0..total) |i| ga[i] += gv;
        },
        .axis0_2d => {
            const rows = parent_shape[0];
            const cols = parent_shape[1];
            for (0..rows) |i| {
                for (0..cols) |j| ga[i * cols + j] += go[j] * scale;
            }
        },
        .axis1_2d => {
            const rows = parent_shape[0];
            const cols = parent_shape[1];
            for (0..rows) |i| {
                const gv = go[i] * scale;
                for (0..cols) |j| ga[i * cols + j] += gv;
            }
        },
    }
}

// ── tests: scatter の各 case が期待通りに書く ──

test "scatter axis1_2d with sum" {
    const testing = @import("std").testing;
    var ga = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const go = [_]f32{ 10, 20 };
    scatter(&ga, &go, &.{ 2, 3 }, .axis1_2d, scaleSum());
    try testing.expectEqualSlices(f32, &.{ 10, 10, 10, 20, 20, 20 }, &ga);
}

test "scatter axis0_2d with sum" {
    const testing = @import("std").testing;
    var ga = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const go = [_]f32{ 1, 2, 3 };
    scatter(&ga, &go, &.{ 2, 3 }, .axis0_2d, scaleSum());
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 1, 2, 3 }, &ga);
}

test "scatter axis1_2d with mean" {
    const testing = @import("std").testing;
    var ga = [_]f32{ 0, 0, 0, 0 };
    const go = [_]f32{ 10, 20 };
    // scale = 1/2 (cols=2)
    scatter(&ga, &go, &.{ 2, 2 }, .axis1_2d, scaleMean(2));
    try testing.expectEqualSlices(f32, &.{ 5, 5, 10, 10 }, &ga);
}

test "scatter all (1D)" {
    const testing = @import("std").testing;
    var ga = [_]f32{ 0, 0, 0, 0 };
    const go = [_]f32{7};
    scatter(&ga, &go, &.{4}, .all, scaleSum());
    try testing.expectEqualSlices(f32, &.{ 7, 7, 7, 7 }, &ga);
}
