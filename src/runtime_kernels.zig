/// runtime_kernels.zig: cpu_runtime / diff_cpu_runtime 共通の純計算関数
///
/// 両 runtime の forward 計算ロジックが完全に重複していたため、
/// 型に依存しない純関数として抽出。backward ロジックは各 runtime に残す。
const std = @import("std");

pub const MAX_NDIM = 4;

const vl = std.simd.suggestVectorLength(f32) orelse 4;

// ── Shape utility functions ──

/// comptime shape slice → [MAX_NDIM]usize 配列に変換 (残りは1埋め)
pub fn initShapeArray(shape_slice: []const usize) [MAX_NDIM]usize {
    var arr: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
    for (0..shape_slice.len) |j| arr[j] = shape_slice[j];
    return arr;
}

/// shape 配列の全要素の積
pub fn totalElements(shape: [MAX_NDIM]usize, ndim: usize) usize {
    var size: usize = 1;
    for (0..ndim) |i| size *= shape[i];
    return size;
}

/// 最後の次元のサイズ
pub fn lastDim(shape: [MAX_NDIM]usize, ndim: usize) usize {
    return shape[ndim - 1];
}

/// 最後の次元を除いた行数
pub fn numRows(shape: [MAX_NDIM]usize, ndim: usize) usize {
    var rows: usize = 1;
    for (0..ndim - 1) |i| rows *= shape[i];
    return rows;
}

/// Softmax (in-place): out_data は入力のコピーを事前に受け取る前提
pub fn softmaxForward(out_data: []f32, rows: usize, cols: usize) void {
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
pub fn logSoftmaxForward(in_data: []const f32, out_data: []f32, rows: usize, cols: usize, softmax_cache: ?[]f32) void {
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
                const ls = v - log_sum_splat;
                out_row[k..][0..vl].* = ls;
                cache_row[k..][0..vl].* = @exp(ls);
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
pub fn layerNormForward(
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
pub fn reductionSumRows(in_data: []const f32, out_data: []f32, rows: usize, cols: usize) void {
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
pub fn reductionSumCols(in_data: []const f32, out_data: []f32, rows: usize, cols: usize) void {
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
pub fn reductionSum1D(data: []const f32) f32 {
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
