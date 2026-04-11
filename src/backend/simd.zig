/// SIMDバックエンド。
/// Zigの@Vectorを使い、ARM NEON / x86 AVX を自動選択する。
/// std.simd.suggestVectorLength で最適なベクトル幅を取得。

const std = @import("std");

/// SIMD演算のベクトル長を取得。SIMDが利用できない場合はスカラーfallback。
fn vecLen(comptime T: type) usize {
    return std.simd.suggestVectorLength(T) orelse 1;
}

/// 行列積: C = A @ B (row-major)
/// A: (m x k), B: (k x n), C: (m x n)
///
/// Aの各行に対し、Bの列方向をSIMDでvec_len個ずつ処理する。
/// A[i,p]をsplatし、B[p, j..j+vec_len]とベクトル積を取り累積する。
pub fn matmul(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, m: usize, k: usize, n: usize) void {
    const vl = comptime vecLen(T);

    if (vl == 1) {
        // SIMD利用不可: スカラーfallback
        @import("cpu.zig").matmul(T, a, b, c, m, k, n);
        return;
    }

    for (0..m) |i| {
        var j: usize = 0;

        // SIMD パス: vec_len 列ずつ処理
        while (j + vl <= n) : (j += vl) {
            var acc: @Vector(vl, T) = @splat(0);
            for (0..k) |p| {
                const a_val: @Vector(vl, T) = @splat(a[i * k + p]);
                const b_vec: @Vector(vl, T) = b[p * n + j ..][0..vl].*;
                acc += a_val * b_vec;
            }
            c[i * n + j ..][0..vl].* = acc;
        }

        // スカラー残り
        while (j < n) : (j += 1) {
            var sum: T = 0;
            for (0..k) |p| {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// element-wise add: c[i] = a[i] + b[i]
pub fn add(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, len: usize) void {
    const vl = comptime vecLen(T);
    var i: usize = 0;

    if (vl > 1) {
        while (i + vl <= len) : (i += vl) {
            const va: @Vector(vl, T) = a[i..][0..vl].*;
            const vb: @Vector(vl, T) = b[i..][0..vl].*;
            c[i..][0..vl].* = va + vb;
        }
    }

    while (i < len) : (i += 1) {
        c[i] = a[i] + b[i];
    }
}

/// element-wise sub: c[i] = a[i] - b[i]
pub fn sub(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, len: usize) void {
    const vl = comptime vecLen(T);
    var i: usize = 0;

    if (vl > 1) {
        while (i + vl <= len) : (i += vl) {
            const va: @Vector(vl, T) = a[i..][0..vl].*;
            const vb: @Vector(vl, T) = b[i..][0..vl].*;
            c[i..][0..vl].* = va - vb;
        }
    }

    while (i < len) : (i += 1) {
        c[i] = a[i] - b[i];
    }
}

/// element-wise mul: c[i] = a[i] * b[i]
pub fn mul(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, len: usize) void {
    const vl = comptime vecLen(T);
    var i: usize = 0;

    if (vl > 1) {
        while (i + vl <= len) : (i += vl) {
            const va: @Vector(vl, T) = a[i..][0..vl].*;
            const vb: @Vector(vl, T) = b[i..][0..vl].*;
            c[i..][0..vl].* = va * vb;
        }
    }

    while (i < len) : (i += 1) {
        c[i] = a[i] * b[i];
    }
}

/// ReLU: c[i] = max(0, a[i])
pub fn relu(comptime T: type, a: [*]const T, c: [*]T, len: usize) void {
    const vl = comptime vecLen(T);
    var i: usize = 0;

    if (vl > 1) {
        const zero: @Vector(vl, T) = @splat(0);
        while (i + vl <= len) : (i += vl) {
            const va: @Vector(vl, T) = a[i..][0..vl].*;
            c[i..][0..vl].* = @max(va, zero);
        }
    }

    while (i < len) : (i += 1) {
        c[i] = @max(a[i], 0);
    }
}

/// scale: c[i] = a[i] * scalar
pub fn scale(comptime T: type, a: [*]const T, scalar: T, c: [*]T, len: usize) void {
    const vl = comptime vecLen(T);
    var i: usize = 0;

    if (vl > 1) {
        const vs: @Vector(vl, T) = @splat(scalar);
        while (i + vl <= len) : (i += vl) {
            const va: @Vector(vl, T) = a[i..][0..vl].*;
            c[i..][0..vl].* = va * vs;
        }
    }

    while (i < len) : (i += 1) {
        c[i] = a[i] * scalar;
    }
}

// ============================================================
// テスト: CPU結果と比較してSIMDの正確性を検証
// ============================================================

const cpu = @import("cpu.zig");

test "simd matmul matches cpu" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
    var c_cpu: [18]f32 = undefined;
    var c_simd: [18]f32 = undefined;

    // A: 3x4, B: 4x6, C: 3x6
    cpu.matmul(f32, &a, &b, &c_cpu, 3, 4, 6);
    matmul(f32, &a, &b, &c_simd, 3, 4, 6);

    for (c_cpu, c_simd) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 1e-6);
    }
}

test "simd matmul f64 matches cpu" {
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f64{ 7, 8, 9, 10, 11, 12 };
    var c_cpu: [4]f64 = undefined;
    var c_simd: [4]f64 = undefined;

    cpu.matmul(f64, &a, &b, &c_cpu, 2, 3, 2);
    matmul(f64, &a, &b, &c_simd, 2, 3, 2);

    for (c_cpu, c_simd) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 1e-12);
    }
}

test "simd add matches cpu" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const b = [_]f32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    var c_cpu: [10]f32 = undefined;
    var c_simd: [10]f32 = undefined;

    cpu.add(f32, &a, &b, &c_cpu, 10);
    add(f32, &a, &b, &c_simd, 10);

    for (c_cpu, c_simd) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
}

test "simd relu matches cpu" {
    const a = [_]f32{ -5, -3, -1, 0, 1, 3, 5, -2, 4, -6 };
    var c_cpu: [10]f32 = undefined;
    var c_simd: [10]f32 = undefined;

    cpu.relu(f32, &a, &c_cpu, 10);
    relu(f32, &a, &c_simd, 10);

    for (c_cpu, c_simd) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
}

test "simd scale matches cpu" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var c_cpu: [10]f32 = undefined;
    var c_simd: [10]f32 = undefined;

    cpu.scale(f32, &a, 3.5, &c_cpu, 10);
    scale(f32, &a, 3.5, &c_simd, 10);

    for (c_cpu, c_simd) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 1e-6);
    }
}

test "simd mul matches cpu" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const b = [_]f32{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    var c_cpu: [10]f32 = undefined;
    var c_simd: [10]f32 = undefined;

    cpu.mul(f32, &a, &b, &c_cpu, 10);
    mul(f32, &a, &b, &c_simd, 10);

    for (c_cpu, c_simd) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
}
