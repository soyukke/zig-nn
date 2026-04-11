/// cuda_ops.zig: CUDA カーネルディスパッチ関数
///
/// CudaRuntime のボイラープレートを free function として抽出。
/// 生の CUdeviceptr と CudaContext を操作する。
const cuda = @import("backend/cuda.zig");
const CudaContext = cuda.CudaContext;
const CUdeviceptr = cuda.CUdeviceptr;
const CUfunction = cuda.CUfunction;

pub const MAX_NDIM = 8;
pub const BLOCK_SIZE: c_uint = 256;

pub fn gridFor(n: usize) c_uint {
    return @intCast((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

pub fn nextPow2(v: c_uint) c_uint {
    if (v == 0) return 1;
    var x = v - 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

pub fn initShapeArray(shape_slice: []const usize) [MAX_NDIM]usize {
    var arr: [MAX_NDIM]usize = .{0} ** MAX_NDIM;
    for (shape_slice, 0..) |s, i| arr[i] = s;
    return arr;
}

/// Dispatch broadcast binary op (add_broadcast / mul_broadcast pattern).
/// Launches func(out, larger, smaller, larger_total, smaller_total).
pub fn dispatchBroadcastBinop(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    larger_dptr: CUdeviceptr,
    smaller_dptr: CUdeviceptr,
    larger_total: usize,
    smaller_total: usize,
) void {
    var a_i: c_int = @intCast(larger_total);
    var b_i: c_int = @intCast(smaller_total);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&larger_dptr)),
        @ptrCast(@constCast(&smaller_dptr)),
        @ptrCast(&a_i),
        @ptrCast(&b_i),
    };
    ctx.launchKernel(
        func,
        .{ gridFor(larger_total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch elementwise unary op (gelu_kernel / silu_kernel pattern).
/// Launches func(out, x, n).
pub fn dispatchElementwise(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launchKernel(
        func,
        .{ gridFor(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch in-place softmax: func(data, rows, cols) with shared memory.
pub fn dispatchSoftmax(
    ctx: *CudaContext,
    func: CUfunction,
    data_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    const block_pow2 = nextPow2(@intCast(@min(cols, 1024)));
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&data_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launchKernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_pow2, 1, 1 },
        block_pow2 * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// Dispatch layernorm: func(out, x, gamma, beta, rows, cols, eps) with shared memory.
pub fn dispatchLayerNorm(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    gamma_dptr: CUdeviceptr,
    beta_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
    eps: f32,
) void {
    const block_dim = nextPow2(@intCast(@min(cols, 1024)));
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var eps_v: f32 = eps;
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(@constCast(&gamma_dptr)),
        @ptrCast(@constCast(&beta_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
        @ptrCast(&eps_v),
    };
    ctx.launchKernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_dim, 1, 1 },
        block_dim * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// Dispatch 2D transpose: func(out, x, rows, cols).
pub fn dispatchTranspose2d(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    in_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&in_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launchKernel(
        func,
        .{ gridFor(rows * cols), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch AdaLN modulation: func(out, norm, scale, beta, B, S, D).
pub fn dispatchAdaLN(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    norm_dptr: CUdeviceptr,
    scale_dptr: CUdeviceptr,
    beta_dptr: CUdeviceptr,
    B: usize,
    S: usize,
    D: usize,
) void {
    const total = B * S * D;
    var b_i: c_int = @intCast(B);
    var s_i: c_int = @intCast(S);
    var d_i: c_int = @intCast(D);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&norm_dptr)),
        @ptrCast(@constCast(&scale_dptr)),
        @ptrCast(@constCast(&beta_dptr)),
        @ptrCast(&b_i),
        @ptrCast(&s_i),
        @ptrCast(&d_i),
    };
    ctx.launchKernel(
        func,
        .{ gridFor(total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch gather: func(out, table, indices, num_indices, embed_dim).
pub fn dispatchGather(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    table_dptr: CUdeviceptr,
    idx_dptr: CUdeviceptr,
    num_indices: usize,
    embed_dim: usize,
) void {
    const total = num_indices * embed_dim;
    var n_i: c_int = @intCast(num_indices);
    var ed_i: c_int = @intCast(embed_dim);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&table_dptr)),
        @ptrCast(@constCast(&idx_dptr)),
        @ptrCast(&n_i),
        @ptrCast(&ed_i),
    };
    ctx.launchKernel(
        func,
        .{ gridFor(total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch scale: func(out, x, scale, n).
pub fn dispatchScale(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    scale: f32,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var s: f32 = scale;
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(&s),
        @ptrCast(&n_i),
    };
    ctx.launchKernel(
        func,
        .{ gridFor(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}
