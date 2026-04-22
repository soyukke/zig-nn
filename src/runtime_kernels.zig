/// runtime_kernels.zig: cpu_runtime / diff_cpu_runtime 共通の純計算関数
///
/// 両 runtime の forward 計算ロジックが完全に重複していたため、
/// 型に依存しない純関数として抽出。backward ロジックは各 runtime に残す。
const std = @import("std");

pub const MAX_NDIM = 4;

const vl = std.simd.suggestVectorLength(f32) orelse 4;

// ── Shape utility functions ──

/// comptime shape slice → [MAX_NDIM]usize 配列に変換 (残りは1埋め)
pub fn init_shape_array(shape_slice: []const usize) [MAX_NDIM]usize {
    var arr: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
    for (0..shape_slice.len) |j| arr[j] = shape_slice[j];
    return arr;
}

/// shape 配列の全要素の積
pub fn total_elements(shape: [MAX_NDIM]usize, ndim: usize) usize {
    var size: usize = 1;
    for (0..ndim) |i| size *= shape[i];
    return size;
}

/// 最後の次元のサイズ
pub fn last_dim(shape: [MAX_NDIM]usize, ndim: usize) usize {
    return shape[ndim - 1];
}

/// 最後の次元を除いた行数
pub fn num_rows(shape: [MAX_NDIM]usize, ndim: usize) usize {
    var rows: usize = 1;
    for (0..ndim - 1) |i| rows *= shape[i];
    return rows;
}

/// Softmax (in-place): out_data は入力のコピーを事前に受け取る前提
pub fn softmax_forward(out_data: []f32, rows: usize, cols: usize) void {
    for (0..rows) |i| {
        const row = out_data[i * cols ..][0..cols];
        // SIMD max
        var k: usize = 0;
        var max_vec: @Vector(vl, f32) = @splat(-std.math.inf(f32));
        while (k + vl <= cols) : (k += vl) {
            const v: @Vector(vl, f32) = row[k..][0..vl].*;
            max_vec = @max(max_vec, v);
        }
        var max_val: f32 = @reduce(.Max, max_vec);
        while (k < cols) : (k += 1) {
            if (row[k] > max_val) max_val = row[k];
        }
        // SIMD exp + sum
        const max_splat: @Vector(vl, f32) = @splat(max_val);
        var sum_vec: @Vector(vl, f32) = @splat(0);
        k = 0;
        while (k + vl <= cols) : (k += vl) {
            const v: @Vector(vl, f32) = row[k..][0..vl].*;
            const e = @exp(v - max_splat);
            row[k..][0..vl].* = e;
            sum_vec += e;
        }
        var sum_exp: f32 = @reduce(.Add, sum_vec);
        while (k < cols) : (k += 1) {
            row[k] = @exp(row[k] - max_val);
            sum_exp += row[k];
        }
        // SIMD normalize
        const inv_sum: @Vector(vl, f32) = @splat(1.0 / sum_exp);
        k = 0;
        while (k + vl <= cols) : (k += vl) {
            const v: @Vector(vl, f32) = row[k..][0..vl].*;
            row[k..][0..vl].* = v * inv_sum;
        }
        const inv_sum_s = 1.0 / sum_exp;
        while (k < cols) : (k += 1) {
            row[k] *= inv_sum_s;
        }
    }
}

/// LogSoftmax: in_data → out_data, オプションで softmax_cache にも書き出し
pub fn log_softmax_forward(
    in_data: []const f32,
    out_data: []f32,
    rows: usize,
    cols: usize,
    softmax_cache: ?[]f32,
) void {
    for (0..rows) |i| {
        const row = in_data[i * cols ..][0..cols];
        const out_row = out_data[i * cols ..][0..cols];
        // SIMD max
        var k: usize = 0;
        var max_vec: @Vector(vl, f32) = @splat(-std.math.inf(f32));
        while (k + vl <= cols) : (k += vl) {
            const v: @Vector(vl, f32) = row[k..][0..vl].*;
            max_vec = @max(max_vec, v);
        }
        var max_val: f32 = @reduce(.Max, max_vec);
        while (k < cols) : (k += 1) {
            if (row[k] > max_val) max_val = row[k];
        }
        // SIMD sum_exp
        const max_splat: @Vector(vl, f32) = @splat(max_val);
        var sum_vec: @Vector(vl, f32) = @splat(0);
        k = 0;
        while (k + vl <= cols) : (k += vl) {
            const v: @Vector(vl, f32) = row[k..][0..vl].*;
            sum_vec += @exp(v - max_splat);
        }
        var sum_exp: f32 = @reduce(.Add, sum_vec);
        while (k < cols) : (k += 1) {
            sum_exp += @exp(row[k] - max_val);
        }
        // SIMD output: x - log_sum (+ optional softmax cache)
        const log_sum = @log(sum_exp) + max_val;
        const log_sum_splat: @Vector(vl, f32) = @splat(log_sum);
        k = 0;
        if (softmax_cache) |cache| {
            const cache_row = cache[i * cols ..][0..cols];
            while (k + vl <= cols) : (k += vl) {
                const v: @Vector(vl, f32) = row[k..][0..vl].*;
                const ls: @Vector(vl, f32) = v - log_sum_splat;
                const ls_exp: @Vector(vl, f32) = @exp(ls);
                out_row[k..][0..vl].* = ls;
                cache_row[k..][0..vl].* = ls_exp;
            }
            while (k < cols) : (k += 1) {
                out_row[k] = row[k] - log_sum;
                cache[i * cols + k] = @exp(out_row[k]);
            }
        } else {
            while (k + vl <= cols) : (k += vl) {
                const v: @Vector(vl, f32) = row[k..][0..vl].*;
                out_row[k..][0..vl].* = v - log_sum_splat;
            }
            while (k < cols) : (k += 1) {
                out_row[k] = row[k] - log_sum;
            }
        }
    }
}

/// LayerNorm 2パス: pass1=SIMD sum+sum_sq, pass2=normalize+affine
/// x_norm_out/inv_stds_out が non-null なら backward 用にキャッシュ
pub fn layer_norm_forward(
    in_data: []const f32,
    out_data: []f32,
    gamma: []const f32,
    beta: []const f32,
    rows: usize,
    dim: usize,
    eps: f32,
    x_norm_out: ?[]f32,
    inv_stds_out: ?[]f32,
) void {
    const dim_f: f32 = @floatFromInt(dim);
    for (0..rows) |i| {
        const row = in_data[i * dim ..][0..dim];
        const out_row = out_data[i * dim ..][0..dim];
        // Pass 1: SIMD sum + sum_sq
        var sum_vec: @Vector(vl, f32) = @splat(0);
        var sq_vec: @Vector(vl, f32) = @splat(0);
        var k: usize = 0;
        while (k + vl <= dim) : (k += vl) {
            const v: @Vector(vl, f32) = row[k..][0..vl].*;
            sum_vec += v;
            sq_vec += v * v;
        }
        var sum_val: f32 = @reduce(.Add, sum_vec);
        var sq_val: f32 = @reduce(.Add, sq_vec);
        while (k < dim) : (k += 1) {
            sum_val += row[k];
            sq_val += row[k] * row[k];
        }
        const mean = sum_val / dim_f;
        const variance = sq_val / dim_f - mean * mean;
        const inv_std = 1.0 / @sqrt(variance + eps);
        if (inv_stds_out) |inv_stds| inv_stds[i] = inv_std;
        // Pass 2: normalize + affine
        const mean_splat: @Vector(vl, f32) = @splat(mean);
        const inv_std_splat: @Vector(vl, f32) = @splat(inv_std);
        k = 0;
        if (x_norm_out) |x_norm| {
            const norm_row = x_norm[i * dim ..][0..dim];
            while (k + vl <= dim) : (k += vl) {
                const v: @Vector(vl, f32) = row[k..][0..vl].*;
                const g: @Vector(vl, f32) = gamma[k..][0..vl].*;
                const b: @Vector(vl, f32) = beta[k..][0..vl].*;
                const xn = (v - mean_splat) * inv_std_splat;
                norm_row[k..][0..vl].* = xn;
                out_row[k..][0..vl].* = xn * g + b;
            }
            while (k < dim) : (k += 1) {
                const xn = (row[k] - mean) * inv_std;
                x_norm[i * dim + k] = xn;
                out_row[k] = xn * gamma[k] + beta[k];
            }
        } else {
            while (k + vl <= dim) : (k += vl) {
                const v: @Vector(vl, f32) = row[k..][0..vl].*;
                const g: @Vector(vl, f32) = gamma[k..][0..vl].*;
                const b: @Vector(vl, f32) = beta[k..][0..vl].*;
                out_row[k..][0..vl].* = (v - mean_splat) * inv_std_splat * g + b;
            }
            while (k < dim) : (k += 1) {
                out_row[k] = (row[k] - mean) * inv_std * gamma[k] + beta[k];
            }
        }
    }
}

/// ReductionSum: rows (axis=1) — 各行の合計
pub fn reduction_sum_rows(in_data: []const f32, out_data: []f32, rows: usize, cols: usize) void {
    for (0..rows) |i| {
        const row = in_data[i * cols ..][0..cols];
        var sum_vec: @Vector(vl, f32) = @splat(0);
        var k: usize = 0;
        while (k + vl <= cols) : (k += vl) {
            const v: @Vector(vl, f32) = row[k..][0..vl].*;
            sum_vec += v;
        }
        var s: f32 = @reduce(.Add, sum_vec);
        while (k < cols) : (k += 1) s += row[k];
        out_data[i] = s;
    }
}

/// ReductionSum: cols (axis=0) — 各列の合計
pub fn reduction_sum_cols(in_data: []const f32, out_data: []f32, rows: usize, cols: usize) void {
    @memset(out_data[0..cols], 0);
    for (0..rows) |i| {
        const row = in_data[i * cols ..][0..cols];
        var k: usize = 0;
        while (k + vl <= cols) : (k += vl) {
            const v: @Vector(vl, f32) = row[k..][0..vl].*;
            const o: @Vector(vl, f32) = out_data[k..][0..vl].*;
            out_data[k..][0..vl].* = o + v;
        }
        while (k < cols) : (k += 1) out_data[k] += row[k];
    }
}

/// ReductionSum: 1D — 全要素合計
pub fn reduction_sum1d(data: []const f32) f32 {
    var sum_vec: @Vector(vl, f32) = @splat(0);
    var k: usize = 0;
    while (k + vl <= data.len) : (k += vl) {
        const v: @Vector(vl, f32) = data[k..][0..vl].*;
        sum_vec += v;
    }
    var s: f32 = @reduce(.Add, sum_vec);
    while (k < data.len) : (k += 1) s += data[k];
    return s;
}

// ── Element-wise forward functions ──

/// GELU forward: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_forward(in: []const f32, out: []f32) void {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    for (in, out) |v, *o| {
        const inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
        o.* = 0.5 * v * (1.0 + std.math.tanh(inner));
    }
}

/// SiLU forward: out[i] = x * sigmoid(x), sig_cache[i] = sigmoid(x)
pub fn silu_forward(in: []const f32, out: []f32, sig_cache: []f32) void {
    for (in, out, sig_cache) |v, *o, *sc| {
        const sig = 1.0 / (1.0 + @exp(-v));
        sc.* = sig;
        o.* = v * sig;
    }
}

/// ReLU forward: out[i] = max(0, x)
pub fn relu_forward(in: []const f32, out: []f32) void {
    for (in, out) |v, *o| {
        o.* = if (v > 0) v else 0;
    }
}

/// Tanh forward: out[i] = tanh(x)
pub fn tanh_forward(in: []const f32, out: []f32) void {
    for (in, out) |v, *o| {
        o.* = std.math.tanh(v);
    }
}

/// Sigmoid forward: out[i] = 1 / (1 + exp(-x))
pub fn sigmoid_forward(in: []const f32, out: []f32) void {
    for (in, out) |v, *o| {
        o.* = 1.0 / (1.0 + @exp(-v));
    }
}

/// Square forward: out[i] = x^2
pub fn square_forward(in: []const f32, out: []f32) void {
    for (in, out) |v, *o| {
        o.* = v * v;
    }
}

/// Negative forward: out[i] = -x
pub fn negative_forward(in: []const f32, out: []f32) void {
    for (in, out) |v, *o| {
        o.* = -v;
    }
}

/// Exp forward: out[i] = exp(x)
pub fn exp_forward(in: []const f32, out: []f32) void {
    for (in, out) |v, *o| {
        o.* = @exp(v);
    }
}

// ── Tests ──

const testing = std.testing;

test "geluForward: 基本値" {
    var in = [_]f32{ -1.0, 0.0, 1.0 };
    var out: [3]f32 = undefined;
    gelu_forward(&in, &out);
    // gelu(0) = 0
    try testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-6);
    // gelu(x) ≈ x for large x, gelu(1) ≈ 0.8412
    try testing.expectApproxEqAbs(@as(f32, 0.8412), out[2], 1e-3);
    // gelu(-1) ≈ -0.1588
    try testing.expectApproxEqAbs(@as(f32, -0.1588), out[0], 1e-3);
}

test "siluForward: 基本値とキャッシュ" {
    var in = [_]f32{ 0.0, 1.0, -1.0 };
    var out: [3]f32 = undefined;
    var sig: [3]f32 = undefined;
    silu_forward(&in, &out, &sig);
    // silu(0) = 0 * 0.5 = 0, sigmoid(0) = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), sig[0], 1e-6);
    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    try testing.expectApproxEqAbs(@as(f32, 0.7311), out[1], 1e-3);
}

test "reluForward: 正負" {
    var in = [_]f32{ -2.0, 0.0, 3.0 };
    var out: [3]f32 = undefined;
    relu_forward(&in, &out);
    try testing.expectEqual(@as(f32, 0.0), out[0]);
    try testing.expectEqual(@as(f32, 0.0), out[1]);
    try testing.expectEqual(@as(f32, 3.0), out[2]);
}

test "tanhForward: 基本値" {
    var in = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    tanh_forward(&in, &out);
    try testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.7616), out[1], 1e-3);
}

test "sigmoidForward: 基本値" {
    var in = [_]f32{ 0.0, 100.0, -100.0 };
    var out: [3]f32 = undefined;
    sigmoid_forward(&in, &out);
    try testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 1e-6);
}

test "squareForward: 基本値" {
    var in = [_]f32{ -3.0, 0.0, 2.0 };
    var out: [3]f32 = undefined;
    square_forward(&in, &out);
    try testing.expectEqual(@as(f32, 9.0), out[0]);
    try testing.expectEqual(@as(f32, 0.0), out[1]);
    try testing.expectEqual(@as(f32, 4.0), out[2]);
}

test "negativeForward: 基本値" {
    var in = [_]f32{ 1.0, -2.0, 0.0 };
    var out: [3]f32 = undefined;
    negative_forward(&in, &out);
    try testing.expectEqual(@as(f32, -1.0), out[0]);
    try testing.expectEqual(@as(f32, 2.0), out[1]);
    try testing.expectEqual(@as(f32, 0.0), out[2]);
}

test "expForward: 基本値" {
    var in = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    exp_forward(&in, &out);
    try testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.71828), out[1], 1e-3);
}
