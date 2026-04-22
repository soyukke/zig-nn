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

pub fn grid_for(n: usize) c_uint {
    return @intCast((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

pub fn next_pow2(v: c_uint) c_uint {
    if (v == 0) return 1;
    var x = v - 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

pub fn init_shape_array(shape_slice: []const usize) [MAX_NDIM]usize {
    var arr: [MAX_NDIM]usize = .{0} ** MAX_NDIM;
    for (shape_slice, 0..) |s, i| arr[i] = s;
    return arr;
}

/// Dispatch broadcast binary op (add_broadcast / mul_broadcast pattern).
/// Launches func(out, larger, smaller, larger_total, smaller_total).
pub fn dispatch_broadcast_binop(
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
    ctx.launch_kernel(
        func,
        .{ grid_for(larger_total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch elementwise unary op (gelu_kernel / silu_kernel pattern).
/// Launches func(out, x, n).
pub fn dispatch_elementwise(
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
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch in-place softmax: func(data, rows, cols) with shared memory.
pub fn dispatch_softmax(
    ctx: *CudaContext,
    func: CUfunction,
    data_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    const block_pow2 = next_pow2(@intCast(@min(cols, 1024)));
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&data_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launch_kernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_pow2, 1, 1 },
        block_pow2 * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// Dispatch layernorm: func(out, x, gamma, beta, rows, cols, eps) with shared memory.
pub fn dispatch_layer_norm(
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
    const block_dim = next_pow2(@intCast(@min(cols, 1024)));
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
    ctx.launch_kernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_dim, 1, 1 },
        block_dim * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// Dispatch 2D transpose: func(out, x, rows, cols).
pub fn dispatch_transpose2d(
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
    ctx.launch_kernel(
        func,
        .{ grid_for(rows * cols), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch AdaLN modulation: func(out, norm, scale, beta, B, S, D).
pub fn dispatch_ada_ln(
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
    ctx.launch_kernel(
        func,
        .{ grid_for(total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch gather: func(out, table, indices, num_indices, embed_dim).
pub fn dispatch_gather(
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
    ctx.launch_kernel(
        func,
        .{ grid_for(total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dispatch scale: func(out, x, scale, n).
pub fn dispatch_scale(
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
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

// ════════════════════════════════════════════════════════════════════
// Backward dispatch functions
// ════════════════════════════════════════════════════════════════════

/// Backward elementwise (3-input): func(ga, go, cache, n).
/// Used for gelu/relu/log/square/abs backward where cache is x (input),
/// and for sigmoid/tanh/exp/sqrt backward where cache is out (output).
pub fn dispatch_backward_elementwise3(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    cache_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&cache_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// SiLU backward: func(ga, go, x, sig_cache, n).
pub fn dispatch_silu_backward(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    sig_cache_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(@constCast(&sig_cache_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Clamp backward: func(ga, go, x, min_val, max_val, n).
pub fn dispatch_clamp_backward(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    min_val: f32,
    max_val: f32,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var mn: f32 = min_val;
    var mx: f32 = max_val;
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(&mn),
        @ptrCast(&mx),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dropout backward: func(ga, go, mask, n).
pub fn dispatch_dropout_backward(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    mask_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&mask_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Mul backward same-shape: func(ga, gb, go, a, b, n).
pub fn dispatch_mul_backward_same(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    gb_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    a_dptr: CUdeviceptr,
    b_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&gb_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&a_dptr)),
        @ptrCast(@constCast(&b_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Mul backward broadcast (ga or gb): func(g, go, other, a_total, b_total).
pub fn dispatch_mul_backward_broadcast(
    ctx: *CudaContext,
    func: CUfunction,
    g_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    other_dptr: CUdeviceptr,
    a_total: usize,
    b_total: usize,
) void {
    var a_i: c_int = @intCast(a_total);
    var b_i: c_int = @intCast(b_total);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&g_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&other_dptr)),
        @ptrCast(&a_i),
        @ptrCast(&b_i),
    };
    // Grid over the larger (a_total) for ga, over b_total for gb reduction
    ctx.launch_kernel(
        func,
        .{ grid_for(a_total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Reduce add/sub to broadcast: func(gb, go, out_total, b_total).
pub fn dispatch_reduce_to_broadcast(
    ctx: *CudaContext,
    func: CUfunction,
    gb_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    out_total: usize,
    b_total: usize,
) void {
    var a_i: c_int = @intCast(out_total);
    var b_i: c_int = @intCast(b_total);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&gb_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(&a_i),
        @ptrCast(&b_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(b_total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Div backward: func(ga, gb, go, a, b, n).
pub fn dispatch_div_backward(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    gb_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    a_dptr: CUdeviceptr,
    b_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&gb_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&a_dptr)),
        @ptrCast(@constCast(&b_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Softmax backward: func(ga, go, s, rows, cols) with shared memory.
pub fn dispatch_softmax_backward(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    s_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    const block_pow2 = next_pow2(@intCast(@min(cols, 1024)));
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&s_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launch_kernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_pow2, 1, 1 },
        block_pow2 * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// Softmax out-of-place forward: func(out, x, rows, cols) with shared memory.
pub fn dispatch_softmax_out(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    const block_pow2 = next_pow2(@intCast(@min(cols, 1024)));
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launch_kernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_pow2, 1, 1 },
        block_pow2 * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// LayerNorm forward with cache: func(out, x_norm, inv_stds, x, gamma, beta, rows, cols, eps).
pub fn dispatch_layer_norm_fwd(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    x_norm_dptr: CUdeviceptr,
    inv_stds_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    gamma_dptr: CUdeviceptr,
    beta_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
    eps: f32,
) void {
    const block_dim = next_pow2(@intCast(@min(cols, 1024)));
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var eps_v: f32 = eps;
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&x_norm_dptr)),
        @ptrCast(@constCast(&inv_stds_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(@constCast(&gamma_dptr)),
        @ptrCast(@constCast(&beta_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
        @ptrCast(&eps_v),
    };
    ctx.launch_kernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_dim, 1, 1 },
        block_dim * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// LayerNorm backward dx: func(gx, go, gamma, x_norm, inv_stds, rows, cols).
pub fn dispatch_layer_norm_backward_dx(
    ctx: *CudaContext,
    func: CUfunction,
    gx_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    gamma_dptr: CUdeviceptr,
    x_norm_dptr: CUdeviceptr,
    inv_stds_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    const block_dim = next_pow2(@intCast(@min(cols, 1024)));
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&gx_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&gamma_dptr)),
        @ptrCast(@constCast(&x_norm_dptr)),
        @ptrCast(@constCast(&inv_stds_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launch_kernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_dim, 1, 1 },
        block_dim * 2 * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// LayerNorm backward dgamma/dbeta: func(ggamma, gbeta, go, x_norm, rows, cols).
pub fn dispatch_layer_norm_backward_dg_db(
    ctx: *CudaContext,
    func: CUfunction,
    ggamma_dptr: CUdeviceptr,
    gbeta_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    x_norm_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ggamma_dptr)),
        @ptrCast(@constCast(&gbeta_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&x_norm_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(cols), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Scatter-add (gather backward): func(ga_table, go, indices, num_indices, embed_dim).
pub fn dispatch_scatter_add(
    ctx: *CudaContext,
    func: CUfunction,
    ga_table_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    indices_dptr: CUdeviceptr,
    num_indices: usize,
    embed_dim: usize,
) void {
    const total = num_indices * embed_dim;
    var n_i: c_int = @intCast(num_indices);
    var ed_i: c_int = @intCast(embed_dim);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_table_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&indices_dptr)),
        @ptrCast(&n_i),
        @ptrCast(&ed_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

// ── Loss function dispatch ──

/// Cross-entropy forward: func(out_loss, softmax_cache, logits, indices, batch, num_classes).
pub fn dispatch_cross_entropy_forward(
    ctx: *CudaContext,
    func: CUfunction,
    out_loss_dptr: CUdeviceptr,
    softmax_cache_dptr: CUdeviceptr,
    logits_dptr: CUdeviceptr,
    indices_dptr: CUdeviceptr,
    batch: usize,
    num_classes: usize,
) void {
    var b_i: c_int = @intCast(batch);
    var c_i: c_int = @intCast(num_classes);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_loss_dptr)),
        @ptrCast(@constCast(&softmax_cache_dptr)),
        @ptrCast(@constCast(&logits_dptr)),
        @ptrCast(@constCast(&indices_dptr)),
        @ptrCast(&b_i),
        @ptrCast(&c_i),
    };
    ctx.launch_kernel(
        func,
        .{ @intCast(batch), 1, 1 },
        .{ 1, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Cross-entropy backward: func(ga, go, softmax_cache, indices, batch, num_classes).
pub fn dispatch_cross_entropy_backward(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    softmax_cache_dptr: CUdeviceptr,
    indices_dptr: CUdeviceptr,
    batch: usize,
    num_classes: usize,
) void {
    const total = batch * num_classes;
    var b_i: c_int = @intCast(batch);
    var c_i: c_int = @intCast(num_classes);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&softmax_cache_dptr)),
        @ptrCast(@constCast(&indices_dptr)),
        @ptrCast(&b_i),
        @ptrCast(&c_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// MSE/BCE forward: func(out_loss, pred/logits, target, n).
pub fn dispatch_loss_forward(
    ctx: *CudaContext,
    func: CUfunction,
    out_loss_dptr: CUdeviceptr,
    pred_dptr: CUdeviceptr,
    target_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_loss_dptr)),
        @ptrCast(@constCast(&pred_dptr)),
        @ptrCast(@constCast(&target_dptr)),
        @ptrCast(&n_i),
    };
    const block_size: c_uint = next_pow2(@intCast(@min(n, 1024)));
    ctx.launch_kernel(
        func,
        .{ 1, 1, 1 },
        .{ block_size, 1, 1 },
        block_size * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// MSE/BCE backward: func(ga, go, pred/logits, target, n).
pub fn dispatch_loss_backward(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    pred_dptr: CUdeviceptr,
    target_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&pred_dptr)),
        @ptrCast(@constCast(&target_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

// ── Utility dispatch ──

/// Fill: func(dst, val, n).
pub fn dispatch_fill(
    ctx: *CudaContext,
    func: CUfunction,
    dst_dptr: CUdeviceptr,
    val: f32,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var v: f32 = val;
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&dst_dptr)),
        @ptrCast(&v),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Accumulate gradients: func(dst, src, n).
pub fn dispatch_accum_grad(
    ctx: *CudaContext,
    func: CUfunction,
    dst_dptr: CUdeviceptr,
    src_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&dst_dptr)),
        @ptrCast(@constCast(&src_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// SiLU forward with sigmoid cache: func(out, sig_cache, x, n).
pub fn dispatch_silu_fwd_cache(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    sig_cache_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&sig_cache_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Fused Add+SiLU forward with cache: func(out, sig_cache, a, b, a_total, b_total).
pub fn dispatch_add_silu_fwd_cache(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    sig_cache_dptr: CUdeviceptr,
    a_dptr: CUdeviceptr,
    b_dptr: CUdeviceptr,
    a_total: usize,
    b_total: usize,
) void {
    var a_i: c_int = @intCast(a_total);
    var b_i: c_int = @intCast(b_total);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&sig_cache_dptr)),
        @ptrCast(@constCast(&a_dptr)),
        @ptrCast(@constCast(&b_dptr)),
        @ptrCast(&a_i),
        @ptrCast(&b_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(a_total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Fused Add+SiLU backward (same-size): func(ga, gb, go, sig_cache, a, b, n).
pub fn dispatch_add_silu_backward_same(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    gb_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    sig_cache_dptr: CUdeviceptr,
    a_dptr: CUdeviceptr,
    b_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&gb_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&sig_cache_dptr)),
        @ptrCast(@constCast(&a_dptr)),
        @ptrCast(@constCast(&b_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Fused Add+SiLU backward (broadcast b): func(ga, gb, go, sig_cache, a, b, a_total, b_total).
pub fn dispatch_add_silu_backward_bcast(
    ctx: *CudaContext,
    func: CUfunction,
    ga_dptr: CUdeviceptr,
    gb_dptr: CUdeviceptr,
    go_dptr: CUdeviceptr,
    sig_cache_dptr: CUdeviceptr,
    a_dptr: CUdeviceptr,
    b_dptr: CUdeviceptr,
    a_total: usize,
    b_total: usize,
) void {
    var a_i: c_int = @intCast(a_total);
    var b_i: c_int = @intCast(b_total);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&ga_dptr)),
        @ptrCast(@constCast(&gb_dptr)),
        @ptrCast(@constCast(&go_dptr)),
        @ptrCast(@constCast(&sig_cache_dptr)),
        @ptrCast(@constCast(&a_dptr)),
        @ptrCast(@constCast(&b_dptr)),
        @ptrCast(&a_i),
        @ptrCast(&b_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(a_total), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Dropout forward: func(out, mask, x, seed, rate, inv_keep, n).
pub fn dispatch_dropout(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    mask_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    seed: u64,
    rate: f32,
    inv_keep: f32,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var seed_v: u64 = seed;
    var rate_v: f32 = rate;
    var inv_v: f32 = inv_keep;
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&mask_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(&seed_v),
        @ptrCast(&rate_v),
        @ptrCast(&inv_v),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

// ── Reduction dispatch ──

/// Reduction sum rows: func(out, x, rows, cols).
pub fn dispatch_reduction_sum_rows(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    const block_pow2 = next_pow2(@intCast(@min(cols, 1024)));
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launch_kernel(
        func,
        .{ @intCast(rows), 1, 1 },
        .{ block_pow2, 1, 1 },
        block_pow2 * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// Reduction sum cols: func(out, x, rows, cols).
pub fn dispatch_reduction_sum_cols(
    ctx: *CudaContext,
    func: CUfunction,
    out_dptr: CUdeviceptr,
    x_dptr: CUdeviceptr,
    rows: usize,
    cols: usize,
) void {
    var rows_i: c_int = @intCast(rows);
    var cols_i: c_int = @intCast(cols);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&out_dptr)),
        @ptrCast(@constCast(&x_dptr)),
        @ptrCast(&rows_i),
        @ptrCast(&cols_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(cols), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Reduction sum 1D: func(out, x, n).
pub fn dispatch_reduction_sum1d(
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
    ctx.launch_kernel(
        func,
        .{ 1, 1, 1 },
        .{ next_pow2(@intCast(@min(n, 1024))), 1, 1 },
        next_pow2(@intCast(@min(n, 1024))) * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

// ── Optimizer dispatch ──

/// Adam step: func(param, grad, m, v, lr, beta1, beta2, eps, wd, bc1, bc2, n).
pub fn dispatch_adam_step(
    ctx: *CudaContext,
    func: CUfunction,
    param_dptr: CUdeviceptr,
    grad_dptr: CUdeviceptr,
    m_dptr: CUdeviceptr,
    v_dptr: CUdeviceptr,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    bc1: f32,
    bc2: f32,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var lr_v: f32 = lr;
    var b1_v: f32 = beta1;
    var b2_v: f32 = beta2;
    var eps_v: f32 = eps;
    var wd_v: f32 = wd;
    var bc1_v: f32 = bc1;
    var bc2_v: f32 = bc2;
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&param_dptr)),
        @ptrCast(@constCast(&grad_dptr)),
        @ptrCast(@constCast(&m_dptr)),
        @ptrCast(@constCast(&v_dptr)),
        @ptrCast(&lr_v),
        @ptrCast(&b1_v),
        @ptrCast(&b2_v),
        @ptrCast(&eps_v),
        @ptrCast(&wd_v),
        @ptrCast(&bc1_v),
        @ptrCast(&bc2_v),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}

/// Norm squared partial sums: func(partial_sums, data, n).
pub fn dispatch_norm_sq(
    ctx: *CudaContext,
    func: CUfunction,
    partial_sums_dptr: CUdeviceptr,
    data_dptr: CUdeviceptr,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    const grid = grid_for(n);
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&partial_sums_dptr)),
        @ptrCast(@constCast(&data_dptr)),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid, 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        BLOCK_SIZE * @sizeOf(f32),
        &params,
    ) catch unreachable;
}

/// Scale gradients: func(grad, scale, n).
pub fn dispatch_scale_grad(
    ctx: *CudaContext,
    func: CUfunction,
    grad_dptr: CUdeviceptr,
    scale: f32,
    n: usize,
) void {
    var n_i: c_int = @intCast(n);
    var s: f32 = scale;
    var params = [_]?*anyopaque{
        @ptrCast(@constCast(&grad_dptr)),
        @ptrCast(&s),
        @ptrCast(&n_i),
    };
    ctx.launch_kernel(
        func,
        .{ grid_for(n), 1, 1 },
        .{ BLOCK_SIZE, 1, 1 },
        0,
        &params,
    ) catch unreachable;
}
