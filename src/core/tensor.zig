const std = @import("std");
const Allocator = std.mem.Allocator;
const shape_utils = @import("shape.zig");

/// tupleを[N]usize配列に変換するヘルパー。
/// これにより型の同一性が保証される（異なるソース位置のtupleでも同じ配列型になる）。
fn tupleToArray(comptime tuple: anytype) [tuple.len]usize {
    comptime {
        var arr: [tuple.len]usize = undefined;
        for (0..tuple.len) |i| {
            arr[i] = tuple[i];
        }
        return arr;
    }
}

/// コンパイル時に形状が確定するテンソル型。
/// tupleを受け取り、内部的に[N]usizeに正規化して TensorNd に委譲する。
///
/// 使用例:
///   const Mat = Tensor(f32, .{3, 784});
///   var m = try Mat.init(allocator);
///   defer m.deinit();
pub fn Tensor(comptime T: type, comptime shape: anytype) type {
    return TensorNd(T, shape.len, tupleToArray(shape));
}

/// [N]usize配列ベースの内部Tensor実装。
/// 型の同一性は (T, N, shape_arr) で決まるため、
/// 異なるモジュールから同じ次元でインスタンス化しても同一型になる。
pub fn TensorNd(comptime T: type, comptime N: usize, comptime shape_arr: [N]usize) type {
    comptime {
        if (T != f16 and T != f32 and T != f64)
            @compileError("Tensor only supports f16, f32, f64, got: " ++ @typeName(T));
        if (N == 0)
            @compileError("Tensor shape must have at least 1 dimension");
        for (shape_arr) |d| {
            if (d == 0) @compileError("Tensor shape dimensions must be > 0");
        }
    }

    return struct {
        const Self = @This();

        /// コンパイル時定数
        pub const tensor_shape = shape_arr;
        pub const Scalar = T;
        pub const num_dims = N;
        pub const num_elements: usize = blk: {
            var n: usize = 1;
            for (shape_arr) |d| n *= d;
            break :blk n;
        };
        pub const tensor_strides: [N]usize = blk: {
            var s: [N]usize = undefined;
            var stride: usize = 1;
            var i: usize = N;
            while (i > 0) {
                i -= 1;
                s[i] = stride;
                stride *= shape_arr[i];
            }
            break :blk s;
        };

        /// データ本体
        data: [*]T,
        /// メモリ管理用
        allocator: Allocator,

        // ============================================================
        // 構築・解放
        // ============================================================

        pub fn init(allocator: Allocator) !Self {
            const mem = try allocator.alloc(T, num_elements);
            return .{ .data = mem.ptr, .allocator = allocator };
        }

        pub fn zeros(allocator: Allocator) !Self {
            var t = try init(allocator);
            @memset(t.slice(), 0);
            return t;
        }

        pub fn fill(allocator: Allocator, value: T) !Self {
            var t = try init(allocator);
            @memset(t.slice(), value);
            return t;
        }

        pub fn fromSlice(allocator: Allocator, src_data: []const T) !Self {
            if (src_data.len != num_elements) {
                return error.ShapeMismatch;
            }
            var t = try init(allocator);
            @memcpy(t.slice(), src_data);
            return t;
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.data[0..num_elements]);
        }

        // ============================================================
        // アクセサ
        // ============================================================

        pub fn slice(self: Self) []T {
            return self.data[0..num_elements];
        }

        pub fn constSlice(self: Self) []const T {
            return self.data[0..num_elements];
        }

        fn flatIndex(indices: [N]usize) usize {
            var idx: usize = 0;
            inline for (0..N) |d| {
                idx += indices[d] * tensor_strides[d];
            }
            return idx;
        }

        pub fn at(self: Self, indices: [N]usize) T {
            return self.data[flatIndex(indices)];
        }

        pub fn set(self: Self, indices: [N]usize, value: T) void {
            self.data[flatIndex(indices)] = value;
        }

        // ============================================================
        // ユーティリティ
        // ============================================================

        pub fn clone(self: Self) !Self {
            return fromSlice(self.allocator, self.constSlice());
        }

        /// 形状を変更（要素数が同じである必要あり）
        pub fn reshape(self: Self, comptime new_shape: anytype) Tensor(T, new_shape) {
            const NewT = Tensor(T, new_shape);
            comptime {
                if (NewT.num_elements != num_elements) {
                    @compileError(
                        "reshape: element count mismatch: " ++
                            std.fmt.comptimePrint("{d}", .{num_elements}) ++
                            " vs " ++
                            std.fmt.comptimePrint("{d}", .{NewT.num_elements}),
                    );
                }
            }
            return .{ .data = self.data, .allocator = self.allocator };
        }

        pub fn shapeStr() []const u8 {
            return comptime blk: {
                var buf: []const u8 = "(";
                for (shape_arr, 0..) |d, i| {
                    buf = buf ++ std.fmt.comptimePrint("{d}", .{d});
                    if (i < N - 1) buf = buf ++ ", ";
                }
                buf = buf ++ ")";
                break :blk buf;
            };
        }

        pub fn sum(self: Self) T {
            var s: T = 0;
            for (self.constSlice()) |v| {
                s += v;
            }
            return s;
        }

        pub fn max(self: Self) T {
            var m: T = self.data[0];
            for (self.constSlice()[1..]) |v| {
                if (v > m) m = v;
            }
            return m;
        }
    };
}

// ============================================================
// テスト
// ============================================================

test "Tensor init and deinit" {
    const T = Tensor(f32, .{ 2, 3 });
    var t = try T.zeros(std.testing.allocator);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 6), T.num_elements);
    try std.testing.expectEqual(@as(usize, 2), T.num_dims);

    for (t.constSlice()) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "Tensor fill" {
    const T = Tensor(f32, .{ 2, 3 });
    var t = try T.fill(std.testing.allocator, 42.0);
    defer t.deinit();

    for (t.constSlice()) |v| {
        try std.testing.expectEqual(@as(f32, 42.0), v);
    }
}

test "Tensor at and set" {
    const T = Tensor(f32, .{ 2, 3 });
    var t = try T.zeros(std.testing.allocator);
    defer t.deinit();

    t.set(.{ 0, 2 }, 5.0);
    try std.testing.expectEqual(@as(f32, 5.0), t.at(.{ 0, 2 }));

    t.set(.{ 1, 1 }, 3.14);
    try std.testing.expectEqual(@as(f32, 3.14), t.at(.{ 1, 1 }));
}

test "Tensor fromSlice" {
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const T = Tensor(f32, .{ 2, 3 });
    var t = try T.fromSlice(std.testing.allocator, &data);
    defer t.deinit();

    try std.testing.expectEqual(@as(f32, 1), t.at(.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 6), t.at(.{ 1, 2 }));
}

test "Tensor reshape" {
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const T = Tensor(f32, .{ 2, 3 });
    var t = try T.fromSlice(std.testing.allocator, &data);
    defer t.deinit();

    const reshaped = t.reshape(.{ 3, 2 });
    try std.testing.expectEqual(@as(f32, 1), reshaped.at(.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 4), reshaped.at(.{ 1, 1 }));
}

test "Tensor f16" {
    const T = Tensor(f16, .{ 2, 3 });
    var t = try T.fill(std.testing.allocator, @as(f16, 1.5));
    defer t.deinit();

    try std.testing.expectEqual(@as(f16, 1.5), t.at(.{ 0, 0 }));
}

test "Tensor f64" {
    const T = Tensor(f64, .{ 2, 3 });
    var t = try T.fill(std.testing.allocator, 3.14159265358979);
    defer t.deinit();

    try std.testing.expectApproxEqRel(@as(f64, 3.14159265358979), t.at(.{ 0, 0 }), 1e-15);
}

test "Tensor sum" {
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const T = Tensor(f32, .{ 2, 3 });
    var t = try T.fromSlice(std.testing.allocator, &data);
    defer t.deinit();

    try std.testing.expectEqual(@as(f32, 21), t.sum());
}

test "Tensor 3D" {
    const T = Tensor(f32, .{ 2, 3, 4 });
    var t = try T.zeros(std.testing.allocator);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 24), T.num_elements);
    try std.testing.expectEqual(@as(usize, 3), T.num_dims);

    t.set(.{ 1, 2, 3 }, 99.0);
    try std.testing.expectEqual(@as(f32, 99.0), t.at(.{ 1, 2, 3 }));
}

test "Tensor strides" {
    const T = Tensor(f32, .{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 12), T.tensor_strides[0]);
    try std.testing.expectEqual(@as(usize, 4), T.tensor_strides[1]);
    try std.testing.expectEqual(@as(usize, 1), T.tensor_strides[2]);
}

test "Tensor shapeStr" {
    const s = Tensor(f32, .{ 2, 3, 4 }).shapeStr();
    try std.testing.expectEqualStrings("(2, 3, 4)", s);
}

test "Tensor clone" {
    const data = [_]f32{ 1, 2, 3, 4 };
    const T = Tensor(f32, .{ 2, 2 });
    var t = try T.fromSlice(std.testing.allocator, &data);
    defer t.deinit();

    var t2 = try t.clone();
    defer t2.deinit();

    t.set(.{ 0, 0 }, 999.0);
    try std.testing.expectEqual(@as(f32, 1), t2.at(.{ 0, 0 }));
}
