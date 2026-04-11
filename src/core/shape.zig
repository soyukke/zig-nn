const std = @import("std");

/// コンパイル時のShape演算ユーティリティ。
/// Tensor型やopsでshapeの検証・変換に使用する。

/// タプルの要素数の積を返す
pub fn numElements(comptime shape: anytype) usize {
    comptime {
        var n: usize = 1;
        for (0..shape.len) |i| {
            n *= shape[i];
        }
        return n;
    }
}

/// タプルの次元数を返す
pub fn ndim(comptime shape: anytype) usize {
    return shape.len;
}

/// row-major stridesを計算する
pub fn strides(comptime shape: anytype) [shape.len]usize {
    comptime {
        var s: [shape.len]usize = undefined;
        var stride: usize = 1;
        var i: usize = shape.len;
        while (i > 0) {
            i -= 1;
            s[i] = stride;
            stride *= shape[i];
        }
        return s;
    }
}

/// matmulの結果shapeを計算する。次元不一致はコンパイルエラー。
/// A: .{M, K}, B: .{K, N} => .{M, N}
pub fn matmulShape(comptime a: anytype, comptime b: anytype) @TypeOf(.{ a[0], b[1] }) {
    if (a.len != 2) @compileError("matmul requires 2D tensor for A, got " ++ std.fmt.comptimePrint("{d}", .{a.len}) ++ "D");
    if (b.len != 2) @compileError("matmul requires 2D tensor for B, got " ++ std.fmt.comptimePrint("{d}", .{b.len}) ++ "D");
    if (a[1] != b[0]) @compileError(
        "matmul dimension mismatch: A columns (" ++
            std.fmt.comptimePrint("{d}", .{a[1]}) ++
            ") != B rows (" ++
            std.fmt.comptimePrint("{d}", .{b[0]}) ++ ")",
    );
    return .{ a[0], b[1] };
}

/// 転置のshapeを返す (2D限定)
pub fn transposeShape(comptime shape: anytype) @TypeOf(.{ shape[1], shape[0] }) {
    if (shape.len != 2) @compileError("transpose requires 2D shape");
    return .{ shape[1], shape[0] };
}

/// Conv2Dの出力shapeを計算する
/// input: .{C_in, H, W}
/// kernel: .{C_out, C_in, kH, kW}
/// => .{C_out, out_H, out_W}
pub fn conv2dOutputShape(
    comptime input: anytype,
    comptime kernel: anytype,
    comptime stride_h: usize,
    comptime stride_w: usize,
    comptime pad_h: usize,
    comptime pad_w: usize,
) @TypeOf(.{
    kernel[0],
    (input[1] + 2 * pad_h - kernel[2]) / stride_h + 1,
    (input[2] + 2 * pad_w - kernel[3]) / stride_w + 1,
}) {
    if (input.len != 3) @compileError("conv2d input requires 3D shape (C, H, W)");
    if (kernel.len != 4) @compileError("conv2d kernel requires 4D shape (C_out, C_in, kH, kW)");
    if (input[0] != kernel[1]) @compileError("conv2d channel mismatch");
    return .{
        kernel[0],
        (input[1] + 2 * pad_h - kernel[2]) / stride_h + 1,
        (input[2] + 2 * pad_w - kernel[3]) / stride_w + 1,
    };
}

/// 2つのshapeが要素ごとの演算で互換性があるか検証する
pub fn assertSameShape(comptime a: anytype, comptime b: anytype) void {
    if (a.len != b.len) @compileError(
        "shape rank mismatch: " ++
            std.fmt.comptimePrint("{d}", .{a.len}) ++ "D vs " ++
            std.fmt.comptimePrint("{d}", .{b.len}) ++ "D",
    );
    inline for (0..a.len) |i| {
        if (a[i] != b[i]) @compileError(
            "shape mismatch at dim " ++ std.fmt.comptimePrint("{d}", .{i}) ++
                ": " ++ std.fmt.comptimePrint("{d}", .{a[i]}) ++
                " vs " ++ std.fmt.comptimePrint("{d}", .{b[i]}),
        );
    }
}

// ============================================================
// テスト
// ============================================================

test "numElements" {
    try std.testing.expectEqual(@as(usize, 6), comptime numElements(.{ 2, 3 }));
    try std.testing.expectEqual(@as(usize, 24), comptime numElements(.{ 2, 3, 4 }));
    try std.testing.expectEqual(@as(usize, 1), comptime numElements(.{1}));
}

test "strides" {
    const s = comptime strides(.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 12), s[0]);
    try std.testing.expectEqual(@as(usize, 4), s[1]);
    try std.testing.expectEqual(@as(usize, 1), s[2]);
}

test "matmulShape" {
    const result = comptime matmulShape(.{ 3, 784 }, .{ 784, 128 });
    try std.testing.expectEqual(@as(usize, 3), result[0]);
    try std.testing.expectEqual(@as(usize, 128), result[1]);
}

test "transposeShape" {
    const result = comptime transposeShape(.{ 3, 5 });
    try std.testing.expectEqual(@as(usize, 5), result[0]);
    try std.testing.expectEqual(@as(usize, 3), result[1]);
}

test "conv2dOutputShape" {
    // input: 3x28x28, kernel: 16x3x5x5, stride 1, pad 0
    const result = comptime conv2dOutputShape(.{ 3, 28, 28 }, .{ 16, 3, 5, 5 }, 1, 1, 0, 0);
    try std.testing.expectEqual(@as(usize, 16), result[0]);
    try std.testing.expectEqual(@as(usize, 24), result[1]);
    try std.testing.expectEqual(@as(usize, 24), result[2]);
}
