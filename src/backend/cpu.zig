/// CPUバックエンド。f32 は CBLAS (macOS: Accelerate, Linux: OpenBLAS) を使用。
/// f64 等は素朴なループ実装にフォールバック。
const builtin = @import("builtin");
const std = @import("std");
const Timer = @import("../util/timer.zig").Timer;
const log_mod = @import("../log.zig");
const log = log_mod.cpu;

// プロファイリング制御
pub var profiling_enabled: bool = true;

// プロファイリングカウンタ
pub var sgemm_count: u64 = 0;
pub var sgemm_nanos: u64 = 0;
pub var saxpy_count: u64 = 0;
pub var saxpy_nanos: u64 = 0;
pub var other_count: u64 = 0;
pub var other_nanos: u64 = 0;

inline fn timerStart() ?Timer {
    if (!profiling_enabled) return null;
    return Timer.start() catch null;
}

inline fn timerRead(timer: *?Timer) u64 {
    if (timer.*) |*t| return t.read();
    return 0;
}

pub fn printProfile() void {
    var maybe_artifact = log_mod.openProfileArtifact("cpu") catch null;
    if (maybe_artifact) |*artifact| {
        defer artifact.close();
        const sgemm_ms = @as(f64, @floatFromInt(sgemm_nanos)) / 1_000_000.0;
        const saxpy_ms = @as(f64, @floatFromInt(saxpy_nanos)) / 1_000_000.0;
        const other_ms = @as(f64, @floatFromInt(other_nanos)) / 1_000_000.0;
        const total_blas_ms = sgemm_ms + saxpy_ms + other_ms;
        const w = artifact.writer();
        w.print("=== cpu.zig profile ===\n", .{}) catch return;
        w.print("  sgemm: {d} calls, {d:.1}ms\n", .{ sgemm_count, sgemm_ms }) catch return;
        w.print("  saxpy: {d} calls, {d:.1}ms\n", .{ saxpy_count, saxpy_ms }) catch return;
        w.print(
            "  other (add/sub/mul/relu/scale/bias/transpose): {d} calls, {d:.1}ms\n",
            .{ other_count, other_ms },
        ) catch return;
        w.print("  --- total BLAS: {d:.1}ms ---\n", .{total_blas_ms}) catch return;
        log.info("profile written: {s}", .{artifact.path});
    }
}

pub fn resetProfile() void {
    sgemm_count = 0;
    sgemm_nanos = 0;
    saxpy_count = 0;
    saxpy_nanos = 0;
    other_count = 0;
    other_nanos = 0;
}

// macOS: Accelerate framework (vDSP 関数用)
// Zig 0.16 + Nix apple-sdk では Accelerate.h の cImport が vImage サブフレームワーク
// ヘッダ解決に失敗するため、必要な vDSP シンボルだけを extern 宣言して利用する。
const vDSP_Length = c_ulong;
const vDSP_Stride = c_long;

const accelerate = if (builtin.os.tag == .macos) struct {
    pub extern "c" fn vDSP_vadd(
        A: [*]const f32,
        IA: vDSP_Stride,
        B: [*]const f32,
        IB: vDSP_Stride,
        C: [*]f32,
        IC: vDSP_Stride,
        N: vDSP_Length,
    ) void;
    pub extern "c" fn vDSP_vsub(
        B: [*]const f32,
        IB: vDSP_Stride,
        A: [*]const f32,
        IA: vDSP_Stride,
        C: [*]f32,
        IC: vDSP_Stride,
        N: vDSP_Length,
    ) void;
    pub extern "c" fn vDSP_vmul(
        A: [*]const f32,
        IA: vDSP_Stride,
        B: [*]const f32,
        IB: vDSP_Stride,
        C: [*]f32,
        IC: vDSP_Stride,
        N: vDSP_Length,
    ) void;
    pub extern "c" fn vDSP_vthres(
        A: [*]const f32,
        IA: vDSP_Stride,
        B: *const f32,
        C: [*]f32,
        IC: vDSP_Stride,
        N: vDSP_Length,
    ) void;
    pub extern "c" fn vDSP_vsadd(
        A: [*]const f32,
        IA: vDSP_Stride,
        B: *const f32,
        C: [*]f32,
        IC: vDSP_Stride,
        N: vDSP_Length,
    ) void;
    pub extern "c" fn vDSP_vsmul(
        A: [*]const f32,
        IA: vDSP_Stride,
        B: *const f32,
        C: [*]f32,
        IC: vDSP_Stride,
        N: vDSP_Length,
    ) void;
    pub extern "c" fn vDSP_vma(
        A: [*]const f32,
        IA: vDSP_Stride,
        B: [*]const f32,
        IB: vDSP_Stride,
        C: [*]const f32,
        IC: vDSP_Stride,
        D: [*]f32,
        ID: vDSP_Stride,
        N: vDSP_Length,
    ) void;
    pub extern "c" fn vDSP_mtrans(
        A: [*]const f32,
        IA: vDSP_Stride,
        C: [*]f32,
        IC: vDSP_Stride,
        M: vDSP_Length,
        N: vDSP_Length,
    ) void;
} else struct {};

// CBLAS extern 宣言 (ヘッダファイル不要、リンク時に解決)
// OpenBLAS が USE64BITINT でビルドされている場合 blasint = i64
const blasint = i64;

const CblasRowMajor: c_int = 101;
const CblasNoTrans: c_int = 111;
const CblasTrans: c_int = 112;

extern "c" fn cblas_sgemm(
    order: c_int,
    transA: c_int,
    transB: c_int,
    M: blasint,
    N: blasint,
    K: blasint,
    alpha: f32,
    A: [*]const f32,
    lda: blasint,
    B: [*]const f32,
    ldb: blasint,
    beta: f32,
    C: [*]f32,
    ldc: blasint,
) void;

extern "c" fn cblas_saxpy(
    N: blasint,
    alpha: f32,
    X: [*]const f32,
    incX: blasint,
    Y: [*]f32,
    incY: blasint,
) void;

/// 行列積: C = A @ B (row-major)
/// A: (m x k), B: (k x n), C: (m x n)
pub fn matmul(
    comptime T: type,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    m: usize,
    k: usize,
    n: usize,
) void {
    if (T == f32) {
        var timer = timerStart();
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            @intCast(m),
            @intCast(n),
            @intCast(k),
            1.0,
            a,
            @intCast(k),
            b,
            @intCast(n),
            0.0,
            c,
            @intCast(n),
        );
        sgemm_count += 1;
        sgemm_nanos += timerRead(&timer);
    } else {
        matmulNaive(T, a, b, c, m, k, n);
    }
}

/// 転置なし行列積: C = A^T @ B (row-major)
/// A: (k x m) stored, but treated as (m x k)^T → (k x m)
/// B: (k x n), C: (m x n)
pub fn matmulTransA(
    comptime T: type,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    m: usize,
    k: usize,
    n: usize,
) void {
    if (T == f32) {
        var timer = timerStart();
        cblas_sgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            @intCast(m),
            @intCast(n),
            @intCast(k),
            1.0,
            a,
            @intCast(m),
            b,
            @intCast(n),
            0.0,
            c,
            @intCast(n),
        );
        sgemm_count += 1;
        sgemm_nanos += timerRead(&timer);
    } else {
        // fallback: naive transpose + matmul
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                for (0..k) |p| {
                    sum += a[p * m + i] * b[p * n + j]; // A^T[i,p] = A[p,i]
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// 転置なし行列積: C = A @ B^T (row-major)
/// A: (m x k), B: (n x k) stored, treated as (k x n)^T
/// C: (m x n)
pub fn matmulTransB(
    comptime T: type,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    m: usize,
    k: usize,
    n: usize,
) void {
    if (T == f32) {
        var timer = timerStart();
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            @intCast(m),
            @intCast(n),
            @intCast(k),
            1.0,
            a,
            @intCast(k),
            b,
            @intCast(k),
            0.0,
            c,
            @intCast(n),
        );
        sgemm_count += 1;
        sgemm_nanos += timerRead(&timer);
    } else {
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                for (0..k) |p| {
                    sum += a[i * k + p] * b[j * k + p]; // B^T[p,j] = B[j,p]
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// C += A^T @ B (accumulate, beta=1.0)
/// A: (k x m) stored, C: (m x n), B: (k x n)
pub fn matmulTransAAccum(
    comptime T: type,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    m: usize,
    k: usize,
    n: usize,
) void {
    if (T == f32) {
        var timer = timerStart();
        cblas_sgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            @intCast(m),
            @intCast(n),
            @intCast(k),
            1.0,
            a,
            @intCast(m),
            b,
            @intCast(n),
            1.0,
            c,
            @intCast(n),
        );
        sgemm_count += 1;
        sgemm_nanos += timerRead(&timer);
    } else {
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                for (0..k) |p| {
                    sum += a[p * m + i] * b[p * n + j];
                }
                c[i * n + j] += sum;
            }
        }
    }
}

/// C += A @ B^T (accumulate, beta=1.0)
/// A: (m x k), B: (n x k) stored, C: (m x n)
pub fn matmulTransBAccum(
    comptime T: type,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    m: usize,
    k: usize,
    n: usize,
) void {
    if (T == f32) {
        var timer = timerStart();
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            @intCast(m),
            @intCast(n),
            @intCast(k),
            1.0,
            a,
            @intCast(k),
            b,
            @intCast(k),
            1.0,
            c,
            @intCast(n),
        );
        sgemm_count += 1;
        sgemm_nanos += timerRead(&timer);
    } else {
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                for (0..k) |p| {
                    sum += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] += sum;
            }
        }
    }
}

fn matmulNaive(
    comptime T: type,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    m: usize,
    k: usize,
    n: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
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
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        accelerate.vDSP_vadd(a, 1, b, 1, c, 1, @intCast(len));
    } else {
        for (0..len) |i| c[i] = a[i] + b[i];
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// element-wise sub: c[i] = a[i] - b[i]
pub fn sub(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, len: usize) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        accelerate.vDSP_vsub(b, 1, a, 1, c, 1, @intCast(len));
    } else {
        for (0..len) |i| c[i] = a[i] - b[i];
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// element-wise mul: c[i] = a[i] * b[i]
pub fn mul(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, len: usize) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        accelerate.vDSP_vmul(a, 1, b, 1, c, 1, @intCast(len));
    } else {
        for (0..len) |i| c[i] = a[i] * b[i];
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// ReLU: c[i] = max(0, a[i])
pub fn relu(comptime T: type, a: [*]const T, c: [*]T, len: usize) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        var zero: f32 = 0;
        accelerate.vDSP_vthres(a, 1, &zero, c, 1, @intCast(len));
    } else {
        for (0..len) |i| c[i] = @max(a[i], 0);
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// in-place accumulate: dst[i] += src[i] (cblas_saxpy)
pub fn addAccum(comptime T: type, src: [*]const T, dst: [*]T, len: usize) void {
    if (T == f32) {
        var timer = timerStart();
        cblas_saxpy(@intCast(len), 1.0, src, 1, dst, 1);
        saxpy_count += 1;
        saxpy_nanos += timerRead(&timer);
    } else {
        for (0..len) |i| dst[i] += src[i];
    }
}

/// element-wise add with scalar: c[i] = a[i] + scalar (broadcast scalar to vector)
pub fn addScalar(comptime T: type, a: [*]const T, scalar: T, c: [*]T, len: usize) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        var s = scalar;
        accelerate.vDSP_vsadd(a, 1, &s, c, 1, @intCast(len));
    } else {
        for (0..len) |i| c[i] = a[i] + scalar;
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// scale: c[i] = a[i] * scalar
pub fn scale(comptime T: type, a: [*]const T, scalar: T, c: [*]T, len: usize) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        var s = scalar;
        accelerate.vDSP_vsmul(a, 1, &s, c, 1, @intCast(len));
    } else {
        for (0..len) |i| c[i] = a[i] * scalar;
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// Fused bias add: c[batch*S*D + s*D + d] = a[batch*S*D + s*D + d] + bias[batch*D + d]
/// Shape semantics: a is [B*S, D] or [B, S, D], bias is [B, D] or [rows, D]
/// When S=1, this is equivalent to per-row vadd (flat bias broadcast).
pub fn addBias(
    comptime T: type,
    a: [*]const T,
    bias: [*]const T,
    c: [*]T,
    batches: usize,
    seq: usize,
    dim: usize,
) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        for (0..batches) |b| {
            const bias_row = bias + b * dim;
            for (0..seq) |s| {
                const offset = (b * seq + s) * dim;
                accelerate.vDSP_vadd(a + offset, 1, bias_row, 1, c + offset, 1, @intCast(dim));
            }
        }
    } else {
        for (0..batches) |b| {
            for (0..seq) |s| {
                const offset = (b * seq + s) * dim;
                for (0..dim) |d| {
                    c[offset + d] = a[offset + d] + bias[b * dim + d];
                }
            }
        }
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// Fused broadcast multiply: c[batch*S*D + s*D + d] = a[batch*S*D + s*D + d] * scale[batch*D + d]
/// Shape semantics: a is [B, S, D], scale is [B, 1, D] (broadcast along dim 1)
/// When S=1, this is equivalent to per-row vmul (flat broadcast).
pub fn mulBroadcast(
    comptime T: type,
    a: [*]const T,
    b: [*]const T,
    c: [*]T,
    batches: usize,
    seq: usize,
    dim: usize,
) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        for (0..batches) |batch| {
            const b_row = b + batch * dim;
            for (0..seq) |s| {
                const offset = (batch * seq + s) * dim;
                accelerate.vDSP_vmul(a + offset, 1, b_row, 1, c + offset, 1, @intCast(dim));
            }
        }
    } else {
        for (0..batches) |batch| {
            for (0..seq) |s| {
                const offset = (batch * seq + s) * dim;
                for (0..dim) |d| {
                    c[offset + d] = a[offset + d] * b[batch * dim + d];
                }
            }
        }
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// Batched matrix transpose: B matrices of R×C → C×R
/// Uses vDSP_mtrans on macOS for optimized transpose.
pub fn transposeMatrices(
    comptime T: type,
    src: [*]const T,
    dst: [*]T,
    batches: usize,
    rows: usize,
    cols: usize,
) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        const mat_size = rows * cols;
        for (0..batches) |b| {
            accelerate.vDSP_mtrans(
                src + b * mat_size,
                1,
                dst + b * mat_size,
                1,
                @intCast(cols),
                @intCast(rows),
            );
        }
    } else {
        const mat_size = rows * cols;
        for (0..batches) |b| {
            const s = src + b * mat_size;
            const d = dst + b * mat_size;
            for (0..rows) |i| {
                for (0..cols) |j| {
                    d[j * rows + i] = s[i * cols + j];
                }
            }
        }
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

/// GELU using Accelerate vvtanhf: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// macOS only. 3パス:
/// (1) inner = sqrt(2/π) * (x + 0.044715*x³), (2) vvtanhf(inner), (3) 0.5*x*(1+tanh)
pub fn geluAccelerate(x: [*]const f32, out: [*]f32, len: usize, tmp1: [*]f32, tmp2: [*]f32) void {
    if (comptime builtin.os.tag == .macos) {
        var timer = timerStart();
        const n: c_int = @intCast(len);
        // Pass 1: tmp1 = x³
        accelerate.vDSP_vmul(x, 1, x, 1, tmp1, 1, @intCast(len)); // tmp1 = x²
        accelerate.vDSP_vmul(tmp1, 1, x, 1, tmp1, 1, @intCast(len)); // tmp1 = x³
        // tmp1 = 0.044715 * x³
        var coeff: f32 = 0.044715;
        accelerate.vDSP_vsmul(tmp1, 1, &coeff, tmp1, 1, @intCast(len));
        // tmp1 = x + 0.044715 * x³
        accelerate.vDSP_vadd(x, 1, tmp1, 1, tmp1, 1, @intCast(len));
        // tmp1 = sqrt(2/π) * (x + 0.044715 * x³)
        var sqrt_2_pi: f32 = 0.7978845608028654;
        accelerate.vDSP_vsmul(tmp1, 1, &sqrt_2_pi, tmp1, 1, @intCast(len));
        // Pass 2: tmp2 = tanh(tmp1) via vvtanhf
        accelerate.vvtanhf(tmp2, tmp1, &n);
        // Pass 3: out = 0.5 * x * (1 + tanh)
        var one: f32 = 1.0;
        accelerate.vDSP_vsadd(tmp2, 1, &one, tmp2, 1, @intCast(len)); // tmp2 = 1+tanh
        accelerate.vDSP_vmul(x, 1, tmp2, 1, out, 1, @intCast(len)); // out = x*(1+tanh)
        var half: f32 = 0.5;
        accelerate.vDSP_vsmul(out, 1, &half, out, 1, @intCast(len)); // out = 0.5*x*(1+tanh)
        other_count += 1;
        other_nanos += timerRead(&timer);
    } else {
        unreachable;
    }
}

/// Fused AdaLN: out[b,s,d] = norm[b,s,d] * scale[b,1,d] + beta[b,1,d]
/// vDSP_vma: C[i] = A[i] * B[i] + D[i] (multiply-add)
pub fn modulateAdaLN(
    comptime T: type,
    norm: [*]const T,
    scale_data: [*]const T,
    beta_data: [*]const T,
    out: [*]T,
    batches: usize,
    seq: usize,
    dim: usize,
) void {
    var timer = timerStart();
    if (T == f32 and builtin.os.tag == .macos) {
        for (0..batches) |b| {
            const s_row = scale_data + b * dim;
            const b_row = beta_data + b * dim;
            for (0..seq) |s| {
                const offset = (b * seq + s) * dim;
                accelerate.vDSP_vma(
                    norm + offset,
                    1,
                    s_row,
                    1,
                    b_row,
                    1,
                    out + offset,
                    1,
                    @intCast(dim),
                );
            }
        }
    } else {
        for (0..batches) |b| {
            for (0..seq) |s| {
                const offset = (b * seq + s) * dim;
                for (0..dim) |d| {
                    out[offset + d] =
                        norm[offset + d] * scale_data[b * dim + d] + beta_data[b * dim + d];
                }
            }
        }
    }
    other_count += 1;
    other_nanos += timerRead(&timer);
}

// ============================================================
// テスト
// ============================================================

test "cpu matmul 2x3 @ 3x2" {
    // A = [[1,2,3],[4,5,6]]
    // B = [[7,8],[9,10],[11,12]]
    // C = [[58,64],[139,154]]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var c: [4]f32 = undefined;

    matmul(f32, &a, &b, &c, 2, 3, 2);

    try std.testing.expectEqual(@as(f32, 58), c[0]);
    try std.testing.expectEqual(@as(f32, 64), c[1]);
    try std.testing.expectEqual(@as(f32, 139), c[2]);
    try std.testing.expectEqual(@as(f32, 154), c[3]);
}

test "cpu matmul f64" {
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    var c: [4]f64 = undefined;

    matmul(f64, &a, &b, &c, 2, 2, 2);

    try std.testing.expectEqual(@as(f64, 19), c[0]);
    try std.testing.expectEqual(@as(f64, 22), c[1]);
    try std.testing.expectEqual(@as(f64, 43), c[2]);
    try std.testing.expectEqual(@as(f64, 50), c[3]);
}

test "cpu add" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    var c: [3]f32 = undefined;

    add(f32, &a, &b, &c, 3);

    try std.testing.expectEqual(@as(f32, 5), c[0]);
    try std.testing.expectEqual(@as(f32, 7), c[1]);
    try std.testing.expectEqual(@as(f32, 9), c[2]);
}

test "cpu relu" {
    const a = [_]f32{ -2, -1, 0, 1, 2 };
    var c: [5]f32 = undefined;

    relu(f32, &a, &c, 5);

    try std.testing.expectEqual(@as(f32, 0), c[0]);
    try std.testing.expectEqual(@as(f32, 0), c[1]);
    try std.testing.expectEqual(@as(f32, 0), c[2]);
    try std.testing.expectEqual(@as(f32, 1), c[3]);
    try std.testing.expectEqual(@as(f32, 2), c[4]);
}

test "cpu scale" {
    const a = [_]f32{ 1, 2, 3 };
    var c: [3]f32 = undefined;

    scale(f32, &a, 2.5, &c, 3);

    try std.testing.expectEqual(@as(f32, 2.5), c[0]);
    try std.testing.expectEqual(@as(f32, 5.0), c[1]);
    try std.testing.expectEqual(@as(f32, 7.5), c[2]);
}

test "cpu addBias flat" {
    // a = [[1,2,3],[4,5,6]], bias = [10,20,30]
    // expected: [[11,22,33],[14,25,36]]
    // flat bias: batches=1, seq=2 (same bias for all rows)
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const bias = [_]f32{ 10, 20, 30 };
    var c: [6]f32 = undefined;

    addBias(f32, &a, &bias, &c, 1, 2, 3);

    try std.testing.expectEqual(@as(f32, 11), c[0]);
    try std.testing.expectEqual(@as(f32, 22), c[1]);
    try std.testing.expectEqual(@as(f32, 33), c[2]);
    try std.testing.expectEqual(@as(f32, 14), c[3]);
    try std.testing.expectEqual(@as(f32, 25), c[4]);
    try std.testing.expectEqual(@as(f32, 36), c[5]);
}

test "cpu addBias 3D broadcast" {
    // a = [2, 2, 3] (2 batches, 2 positions, 3 dims)
    // bias = [2, 3] (2 batches, 3 dims) - broadcast along dim 1
    // a:    [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]]
    // bias: [[100,200,300], [400,500,600]]
    // expected: [[[101,202,303],[104,205,306]], [[407,508,609],[410,511,612]]]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const bias = [_]f32{ 100, 200, 300, 400, 500, 600 };
    var c: [12]f32 = undefined;

    addBias(f32, &a, &bias, &c, 2, 2, 3);

    // batch 0, pos 0: [1+100, 2+200, 3+300]
    try std.testing.expectEqual(@as(f32, 101), c[0]);
    try std.testing.expectEqual(@as(f32, 202), c[1]);
    try std.testing.expectEqual(@as(f32, 303), c[2]);
    // batch 0, pos 1: [4+100, 5+200, 6+300]
    try std.testing.expectEqual(@as(f32, 104), c[3]);
    try std.testing.expectEqual(@as(f32, 205), c[4]);
    try std.testing.expectEqual(@as(f32, 306), c[5]);
    // batch 1, pos 0: [7+400, 8+500, 9+600]
    try std.testing.expectEqual(@as(f32, 407), c[6]);
    try std.testing.expectEqual(@as(f32, 508), c[7]);
    try std.testing.expectEqual(@as(f32, 609), c[8]);
    // batch 1, pos 1: [10+400, 11+500, 12+600]
    try std.testing.expectEqual(@as(f32, 410), c[9]);
    try std.testing.expectEqual(@as(f32, 511), c[10]);
    try std.testing.expectEqual(@as(f32, 612), c[11]);
}

test "cpu mulBroadcast 3D" {
    // a = [2, 2, 2], scale = [2, 2]
    // a:     [[[1,2],[3,4]], [[5,6],[7,8]]]
    // scale: [[10,20], [30,40]]
    // expected: [[[10,40],[30,80]], [[150,240],[210,320]]]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 10, 20, 30, 40 };
    var c: [8]f32 = undefined;

    mulBroadcast(f32, &a, &b, &c, 2, 2, 2);

    // batch 0, pos 0: [1*10, 2*20]
    try std.testing.expectEqual(@as(f32, 10), c[0]);
    try std.testing.expectEqual(@as(f32, 40), c[1]);
    // batch 0, pos 1: [3*10, 4*20]
    try std.testing.expectEqual(@as(f32, 30), c[2]);
    try std.testing.expectEqual(@as(f32, 80), c[3]);
    // batch 1, pos 0: [5*30, 6*40]
    try std.testing.expectEqual(@as(f32, 150), c[4]);
    try std.testing.expectEqual(@as(f32, 240), c[5]);
    // batch 1, pos 1: [7*30, 8*40]
    try std.testing.expectEqual(@as(f32, 210), c[6]);
    try std.testing.expectEqual(@as(f32, 320), c[7]);
}

test "cpu transposeMatrices" {
    // 2 batches of 2x3 matrices
    // batch 0: [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
    // batch 1: [[7,8,9],[10,11,12]] → [[7,10],[8,11],[9,12]]
    const src = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var dst: [12]f32 = undefined;

    transposeMatrices(f32, &src, &dst, 2, 2, 3);

    // batch 0 transposed (3x2)
    try std.testing.expectEqual(@as(f32, 1), dst[0]);
    try std.testing.expectEqual(@as(f32, 4), dst[1]);
    try std.testing.expectEqual(@as(f32, 2), dst[2]);
    try std.testing.expectEqual(@as(f32, 5), dst[3]);
    try std.testing.expectEqual(@as(f32, 3), dst[4]);
    try std.testing.expectEqual(@as(f32, 6), dst[5]);
    // batch 1 transposed (3x2)
    try std.testing.expectEqual(@as(f32, 7), dst[6]);
    try std.testing.expectEqual(@as(f32, 10), dst[7]);
    try std.testing.expectEqual(@as(f32, 8), dst[8]);
    try std.testing.expectEqual(@as(f32, 11), dst[9]);
    try std.testing.expectEqual(@as(f32, 9), dst[10]);
    try std.testing.expectEqual(@as(f32, 12), dst[11]);
}
