/// diff/cuda_runtime.zig: 微分可能 CUDA ランタイム
///
/// DiffCpuRuntime と同じ duck-typed ops インターフェースを GPU (CUDA) 上で提供し、
/// forward 時に計算グラフを構築し backward() で GPU 上で自動微分する。
/// 統一モジュールの forward(ctx: anytype, ...) が DiffCudaRuntime を ctx として
/// 受け取れば、同じ forward コードで GPU training が可能。
const std = @import("std");
const Allocator = std.mem.Allocator;
const compute = @import("../compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const cuda = @import("../backend/cuda.zig");
const CudaContext = cuda.CudaContext;
const CUdeviceptr = cuda.CUdeviceptr;
const CUfunction = cuda.CUfunction;
const CUBLAS_OP_N = cuda.CUBLAS_OP_N;
const CUBLAS_OP_T = cuda.CUBLAS_OP_T;
const ops = @import("../cuda_ops.zig");
const diff_node = @import("node.zig");

pub const MAX_NDIM = ops.MAX_NDIM;

pub const DiffCudaNode = diff_node.diff_node_generic(CUdeviceptr, ops.MAX_NDIM);

// ── Backward function type ──
const BackwardFn = *const fn (*DiffCudaNode) void;

pub const DiffCudaTensor = *DiffCudaNode;

// ── GPU Adam state ──

pub const GpuAdamState = struct {
    m_dptrs: []CUdeviceptr, // 1次モーメント (GPU)
    v_dptrs: []CUdeviceptr, // 2次モーメント (GPU)
    step: u32,
    allocator: Allocator,
    cuda_ctx: *CudaContext,

    pub fn init(
        allocator: Allocator,
        cuda_ctx: *CudaContext,
        param_sizes: []const usize,
    ) !GpuAdamState {
        const count = param_sizes.len;
        const m_dptrs = try allocator.alloc(CUdeviceptr, count);
        const v_dptrs = try allocator.alloc(CUdeviceptr, count);
        for (param_sizes, 0..) |size, i| {
            m_dptrs[i] = try cuda_ctx.alloc_buffer(size * @sizeOf(f32));
            try cuda_ctx.memset_zero(m_dptrs[i], size);
            v_dptrs[i] = try cuda_ctx.alloc_buffer(size * @sizeOf(f32));
            try cuda_ctx.memset_zero(v_dptrs[i], size);
        }
        return .{
            .m_dptrs = m_dptrs,
            .v_dptrs = v_dptrs,
            .step = 0,
            .allocator = allocator,
            .cuda_ctx = cuda_ctx,
        };
    }

    pub fn deinit(self: *GpuAdamState) void {
        for (self.m_dptrs) |d| self.cuda_ctx.free_buffer(d);
        for (self.v_dptrs) |d| self.cuda_ctx.free_buffer(d);
        self.allocator.free(self.m_dptrs);
        self.allocator.free(self.v_dptrs);
    }
};

// ── Kernel handles ──

const KernelHandles = struct {
    // Forward elementwise
    fn_add: CUfunction,
    fn_mul: CUfunction,
    fn_sub: CUfunction,
    fn_gelu: CUfunction,
    fn_silu_fwd_cache: CUfunction,
    fn_add_silu_fwd_cache: CUfunction,
    fn_relu: CUfunction,
    fn_sigmoid: CUfunction,
    fn_tanh: CUfunction,
    fn_exp: CUfunction,
    fn_log: CUfunction,
    fn_square: CUfunction,
    fn_sqrt: CUfunction,
    fn_abs: CUfunction,
    fn_clamp: CUfunction,
    fn_negative: CUfunction,
    fn_scale: CUfunction,
    // Forward structured
    fn_softmax_out: CUfunction,
    fn_layernorm_fwd: CUfunction,
    fn_transpose: CUfunction,
    fn_gather: CUfunction,
    // Backward elementwise
    fn_gelu_bw: CUfunction,
    fn_silu_bw: CUfunction,
    fn_relu_bw: CUfunction,
    fn_sigmoid_bw: CUfunction,
    fn_tanh_bw: CUfunction,
    fn_exp_bw: CUfunction,
    fn_log_bw: CUfunction,
    fn_square_bw: CUfunction,
    fn_sqrt_bw: CUfunction,
    fn_abs_bw: CUfunction,
    fn_clamp_bw: CUfunction,
    fn_dropout_bw: CUfunction,
    fn_add_silu_bw_same: CUfunction,
    fn_add_silu_bw_bcast: CUfunction,
    // Backward structured
    fn_mul_bw_same: CUfunction,
    fn_mul_bw_bcast_ga: CUfunction,
    fn_mul_bw_bcast_gb: CUfunction,
    fn_reduce_add_bcast: CUfunction,
    fn_reduce_sub_bcast: CUfunction,
    fn_div_bw: CUfunction,
    fn_softmax_bw: CUfunction,
    fn_ln_bw_dx: CUfunction,
    fn_ln_bw_dg_db: CUfunction,
    fn_scatter_add: CUfunction,
    // Loss
    fn_ce_fwd: CUfunction,
    fn_ce_bw: CUfunction,
    fn_mse_fwd: CUfunction,
    fn_mse_bw: CUfunction,
    fn_bce_fwd: CUfunction,
    fn_bce_bw: CUfunction,
    // Utility
    fn_fill: CUfunction,
    fn_accum_grad: CUfunction,
    fn_dropout: CUfunction,
    // Reduction
    fn_reduce_sum_rows: CUfunction,
    fn_reduce_sum_cols: CUfunction,
    fn_reduce_sum_1d: CUfunction,
    // Optimizer
    fn_adam_step: CUfunction,
    fn_norm_sq: CUfunction,
    fn_scale_grad: CUfunction,
};

// ── DiffCudaRuntime init helpers ──

/// PTX にロード済みの cuda_ctx から KernelHandles を一括で取り出す。
/// グループ別のヘルパーに委譲して 70 行制限を守る。
fn load_kernel_handles(cuda_ctx: *CudaContext) !KernelHandles {
    var k: KernelHandles = undefined;
    try load_kernels_forward(cuda_ctx, &k);
    try load_kernels_backward(cuda_ctx, &k);
    try load_kernels_loss_and_misc(cuda_ctx, &k);
    return k;
}

fn load_kernels_forward(cuda_ctx: *CudaContext, k: *KernelHandles) !void {
    // Forward elementwise
    k.fn_add = try cuda_ctx.get_function("add_broadcast");
    k.fn_mul = try cuda_ctx.get_function("mul_broadcast");
    k.fn_sub = try cuda_ctx.get_function("sub_broadcast");
    k.fn_gelu = try cuda_ctx.get_function("gelu_kernel");
    k.fn_silu_fwd_cache = try cuda_ctx.get_function("silu_fwd_cache_kernel");
    k.fn_add_silu_fwd_cache = try cuda_ctx.get_function("add_silu_fwd_cache_kernel");
    k.fn_relu = try cuda_ctx.get_function("relu_kernel");
    k.fn_sigmoid = try cuda_ctx.get_function("sigmoid_kernel");
    k.fn_tanh = try cuda_ctx.get_function("tanh_kernel");
    k.fn_exp = try cuda_ctx.get_function("exp_kernel");
    k.fn_log = try cuda_ctx.get_function("log_kernel");
    k.fn_square = try cuda_ctx.get_function("square_kernel");
    k.fn_sqrt = try cuda_ctx.get_function("sqrt_kernel");
    k.fn_abs = try cuda_ctx.get_function("abs_kernel");
    k.fn_clamp = try cuda_ctx.get_function("clamp_kernel");
    k.fn_negative = try cuda_ctx.get_function("negative_kernel");
    k.fn_scale = try cuda_ctx.get_function("scale_kernel");
    // Forward structured
    k.fn_softmax_out = try cuda_ctx.get_function("softmax_out_kernel");
    k.fn_layernorm_fwd = try cuda_ctx.get_function("layernorm_fwd_kernel");
    k.fn_transpose = try cuda_ctx.get_function("transpose_2d_kernel");
    k.fn_gather = try cuda_ctx.get_function("gather_kernel");
}

fn load_kernels_backward(cuda_ctx: *CudaContext, k: *KernelHandles) !void {
    // Backward elementwise
    k.fn_gelu_bw = try cuda_ctx.get_function("gelu_backward_kernel");
    k.fn_silu_bw = try cuda_ctx.get_function("silu_backward_kernel");
    k.fn_relu_bw = try cuda_ctx.get_function("relu_backward_kernel");
    k.fn_sigmoid_bw = try cuda_ctx.get_function("sigmoid_backward_kernel");
    k.fn_tanh_bw = try cuda_ctx.get_function("tanh_backward_kernel");
    k.fn_exp_bw = try cuda_ctx.get_function("exp_backward_kernel");
    k.fn_log_bw = try cuda_ctx.get_function("log_backward_kernel");
    k.fn_square_bw = try cuda_ctx.get_function("square_backward_kernel");
    k.fn_sqrt_bw = try cuda_ctx.get_function("sqrt_backward_kernel");
    k.fn_abs_bw = try cuda_ctx.get_function("abs_backward_kernel");
    k.fn_clamp_bw = try cuda_ctx.get_function("clamp_backward_kernel");
    k.fn_dropout_bw = try cuda_ctx.get_function("dropout_backward_kernel");
    k.fn_add_silu_bw_same = try cuda_ctx.get_function("add_silu_backward_same_kernel");
    k.fn_add_silu_bw_bcast = try cuda_ctx.get_function("add_silu_backward_bcast_kernel");
    // Backward structured
    k.fn_mul_bw_same = try cuda_ctx.get_function("mul_backward_same_kernel");
    k.fn_mul_bw_bcast_ga = try cuda_ctx.get_function("mul_backward_broadcast_b_ga_kernel");
    k.fn_mul_bw_bcast_gb = try cuda_ctx.get_function("mul_backward_broadcast_b_gb_kernel");
    k.fn_reduce_add_bcast = try cuda_ctx.get_function("reduce_add_to_broadcast_kernel");
    k.fn_reduce_sub_bcast = try cuda_ctx.get_function("reduce_sub_to_broadcast_kernel");
    k.fn_div_bw = try cuda_ctx.get_function("div_backward_kernel");
    k.fn_softmax_bw = try cuda_ctx.get_function("softmax_backward_kernel");
    k.fn_ln_bw_dx = try cuda_ctx.get_function("layernorm_backward_dx_kernel");
    k.fn_ln_bw_dg_db = try cuda_ctx.get_function("layernorm_backward_dgamma_dbeta_kernel");
    k.fn_scatter_add = try cuda_ctx.get_function("scatter_add_kernel");
}

fn load_kernels_loss_and_misc(cuda_ctx: *CudaContext, k: *KernelHandles) !void {
    // Loss
    k.fn_ce_fwd = try cuda_ctx.get_function("cross_entropy_forward_kernel");
    k.fn_ce_bw = try cuda_ctx.get_function("cross_entropy_backward_kernel");
    k.fn_mse_fwd = try cuda_ctx.get_function("mse_forward_kernel");
    k.fn_mse_bw = try cuda_ctx.get_function("mse_backward_kernel");
    k.fn_bce_fwd = try cuda_ctx.get_function("bce_forward_kernel");
    k.fn_bce_bw = try cuda_ctx.get_function("bce_backward_kernel");
    // Utility
    k.fn_fill = try cuda_ctx.get_function("fill_kernel");
    k.fn_accum_grad = try cuda_ctx.get_function("accum_grad_kernel");
    k.fn_dropout = try cuda_ctx.get_function("dropout_kernel");
    // Reduction
    k.fn_reduce_sum_rows = try cuda_ctx.get_function("reduction_sum_rows_kernel");
    k.fn_reduce_sum_cols = try cuda_ctx.get_function("reduction_sum_cols_kernel");
    k.fn_reduce_sum_1d = try cuda_ctx.get_function("reduction_sum_1d_kernel");
    // Optimizer
    k.fn_adam_step = try cuda_ctx.get_function("adam_step_kernel");
    k.fn_norm_sq = try cuda_ctx.get_function("norm_sq_kernel");
    k.fn_scale_grad = try cuda_ctx.get_function("scale_grad_kernel");
}

/// allocParamBuffers の戻り値: パラメータノード + 連続勾配バッファの情報。
const ParamAllocResult = struct {
    param_nodes: []DiffCudaNode,
    param_grad_dptrs: []CUdeviceptr,
    grad_base_dptr: CUdeviceptr,
    total_grad_floats: usize,
};

/// パラメータノードと連続勾配バッファを GPU 上に確保する。
/// 呼び出し側 (init) が成功時の所有権を引き受ける。
fn alloc_param_buffers(
    cuda_ctx: *CudaContext,
    allocator: Allocator,
    module: *const Module,
) !ParamAllocResult {
    const count = module.param_count();
    const param_nodes = try allocator.alloc(DiffCudaNode, count);
    errdefer allocator.free(param_nodes);
    const param_grad_dptrs = try allocator.alloc(CUdeviceptr, count);
    errdefer allocator.free(param_grad_dptrs);

    // Compute total gradient size and allocate one contiguous buffer
    var total_grad_floats: usize = 0;
    for (0..count) |i| total_grad_floats += module.param_size(.{ .index = i });
    const grad_base_dptr = if (total_grad_floats > 0)
        try cuda_ctx.alloc_buffer(total_grad_floats * @sizeOf(f32))
    else
        @as(CUdeviceptr, 0);
    errdefer if (total_grad_floats > 0) cuda_ctx.free_buffer(grad_base_dptr);
    if (total_grad_floats > 0) try cuda_ctx.memset_zero(grad_base_dptr, total_grad_floats);

    var grad_offset: usize = 0;
    var alloc_count: usize = 0;
    errdefer for (param_nodes[0..alloc_count]) |node| cuda_ctx.free_buffer(node.data);
    for (module.params.items, 0..) |meta, i| {
        const size = module.param_size(.{ .index = i });
        const dptr = try cuda_ctx.alloc_buffer(size * @sizeOf(f32));
        const grad_dptr = grad_base_dptr + grad_offset * @sizeOf(f32);

        param_nodes[i] = .{
            .data = dptr,
            .shape = ops.init_shape_array(meta.shape),
            .ndim = meta.shape.len,
            .grad = grad_dptr,
            .backward_fn = null,
            .parents = .{ null, null, null },
            .context = null,
            .requires_grad = true,
            .visited = false,
            .is_param = true,
            .param_index = i,
        };
        param_grad_dptrs[i] = grad_dptr;
        grad_offset += size;
        alloc_count += 1;
    }

    return .{
        .param_nodes = param_nodes,
        .param_grad_dptrs = param_grad_dptrs,
        .grad_base_dptr = grad_base_dptr,
        .total_grad_floats = total_grad_floats,
    };
}

// ── DiffCudaRuntime ──

pub const DiffCudaRuntime = struct {
    allocator: Allocator,
    cuda_ctx: *CudaContext,
    module: *const Module,
    param_nodes: []DiffCudaNode, // パラメータノード (永続)
    param_grad_dptrs: []CUdeviceptr, // パラメータ勾配バッファ (各パラメータのオフセット)
    grad_base_dptr: CUdeviceptr, // 連続勾配バッファのベースポインタ
    total_grad_floats: usize, // 全パラメータ勾配の合計 float 数
    arena: std.heap.ArenaAllocator, // 中間ノード用 CPU arena
    arena_gpu_bufs: std.ArrayList(CUdeviceptr), // 中間 GPU バッファ
    arena_gpu_sizes: std.ArrayList(usize), // 対応するバッファサイズ (bytes)
    gpu_pool: cuda.GpuMemPool, // GPU メモリプール
    topo_buf: std.ArrayListUnmanaged(*DiffCudaNode),
    prng: std.Random.DefaultPrng, // dropout 用 (op で状態更新)
    init_seed: u64, // init_params 用の固定 seed
    training: bool,
    kernels: KernelHandles,

    pub const DEFAULT_SEED: u64 = 42;

    pub fn init(
        module: *const Module,
        cuda_ctx: *CudaContext,
        allocator: Allocator,
    ) !DiffCudaRuntime {
        // Load PTX module
        const ptx_data = @embedFile("backend/cuda_kernels.ptx");
        try cuda_ctx.load_module(@ptrCast(ptx_data.ptr));

        const k = try load_kernel_handles(cuda_ctx);
        const params = try alloc_param_buffers(cuda_ctx, allocator, module);

        return .{
            .allocator = allocator,
            .cuda_ctx = cuda_ctx,
            .module = module,
            .param_nodes = params.param_nodes,
            .param_grad_dptrs = params.param_grad_dptrs,
            .grad_base_dptr = params.grad_base_dptr,
            .total_grad_floats = params.total_grad_floats,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .arena_gpu_bufs = .{},
            .arena_gpu_sizes = .{},
            .gpu_pool = cuda.GpuMemPool.init(allocator),
            .topo_buf = .empty,
            .prng = std.Random.DefaultPrng.init(DEFAULT_SEED),
            .init_seed = DEFAULT_SEED,
            .training = true,
            .kernels = k,
        };
    }

    /// 全ランダム性 (init_params, dropout) を seed 固定する。
    /// init_params() の前に呼ぶと重みが決定論的になる。
    pub fn set_seed(self: *DiffCudaRuntime, seed: u64) void {
        self.init_seed = seed;
        self.prng = std.Random.DefaultPrng.init(seed);
    }

    pub fn deinit(self: *DiffCudaRuntime) void {
        self.free_arena_gpu_buffers();
        self.arena_gpu_bufs.deinit(self.allocator);
        self.arena_gpu_sizes.deinit(self.allocator);
        self.gpu_pool.deinit();
        self.arena.deinit();
        for (self.param_nodes) |node| {
            self.cuda_ctx.free_buffer(node.data);
        }
        if (self.total_grad_floats > 0) self.cuda_ctx.free_buffer(self.grad_base_dptr);
        self.allocator.free(self.param_nodes);
        self.allocator.free(self.param_grad_dptrs);
        self.topo_buf.deinit(self.allocator);
    }

    fn free_arena_gpu_buffers(self: *DiffCudaRuntime) void {
        for (self.arena_gpu_bufs.items, self.arena_gpu_sizes.items) |dptr, size_bytes| {
            self.gpu_pool.release(dptr, size_bytes);
        }
        self.arena_gpu_bufs.clearRetainingCapacity();
        self.arena_gpu_sizes.clearRetainingCapacity();
    }

    pub fn reset_arena(self: *DiffCudaRuntime) void {
        self.free_arena_gpu_buffers();
        _ = self.arena.reset(.retain_capacity);
        for (self.param_nodes) |*node| {
            node.visited = false;
        }
    }

    pub fn zero_grad(self: *DiffCudaRuntime) void {
        // Single async memset for the entire contiguous gradient buffer
        if (self.total_grad_floats > 0)
            self.cuda_ctx.memset_zero_async(
                self.grad_base_dptr,
                self.total_grad_floats,
            ) catch unreachable;
        for (self.param_nodes, 0..) |*node, i| {
            node.grad = self.param_grad_dptrs[i];
        }
    }

    fn arena_alloc(self: *DiffCudaRuntime) Allocator {
        return self.arena.allocator();
    }

    /// GPU メモリ確保 (arena tracked, pool 優先)
    fn alloc_gpu_buf(self: *DiffCudaRuntime, num_floats: usize) CUdeviceptr {
        const size_bytes = num_floats * @sizeOf(f32);
        const dptr = self.gpu_pool.acquire(size_bytes) orelse (self.cuda_ctx.alloc_buffer(
            cuda.GpuMemPool.bucket_size(cuda.GpuMemPool.bucket_index(size_bytes)),
        ) catch unreachable);
        self.arena_gpu_bufs.append(self.allocator, dptr) catch unreachable;
        self.arena_gpu_sizes.append(self.allocator, size_bytes) catch unreachable;
        return dptr;
    }

    /// GPU メモリ確保 + ゼロ初期化
    fn alloc_gpu_buf_zeroed(self: *DiffCudaRuntime, num_floats: usize) CUdeviceptr {
        const dptr = self.alloc_gpu_buf(num_floats);
        self.cuda_ctx.memset_zero(dptr, num_floats) catch unreachable;
        return dptr;
    }

    fn alloc_context(self: *DiffCudaRuntime, comptime T: type) *T {
        return self.arena_alloc().create(T) catch unreachable;
    }

    // ── Node creation ──

    pub fn make_node(
        self: *DiffCudaRuntime,
        dptr: CUdeviceptr,
        shape_slice: []const usize,
        requires_grad: bool,
    ) *DiffCudaNode {
        const node = self.arena_alloc().create(DiffCudaNode) catch unreachable;
        node.* = .{
            .data = dptr,
            .shape = ops.init_shape_array(shape_slice),
            .ndim = shape_slice.len,
            .grad = null,
            .backward_fn = null,
            .parents = .{ null, null, null },
            .context = null,
            .requires_grad = requires_grad,
            .visited = false,
            .is_param = false,
            .param_index = null,
        };
        return node;
    }

    pub fn make_tensor(self: *DiffCudaRuntime, data: []f32, shape: []const usize) DiffCudaTensor {
        var total: usize = 1;
        for (shape) |s| total *= s;
        const dptr = self.alloc_gpu_buf(total);
        self.cuda_ctx.copy_host_to_device(
            dptr,
            @ptrCast(data.ptr),
            total * @sizeOf(f32),
        ) catch unreachable;
        return self.make_node(dptr, shape, false);
    }

    /// allocData returns CPU arena memory (for host-side metadata/indices)
    pub fn alloc_data(self: *DiffCudaRuntime, size: usize) []f32 {
        return self.arena_alloc().alloc(f32, size) catch unreachable;
    }

    // ── Param access ──

    pub fn param(self: *DiffCudaRuntime, handle: ParamHandle) DiffCudaTensor {
        return &self.param_nodes[handle.index];
    }

    /// D2H copy: パラメータデータを CPU スライスとして返す (デバッグ用)
    pub fn param_data(self: *DiffCudaRuntime, index: usize) []f32 {
        const size = self.module.param_size(.{ .index = index });
        const buf = self.alloc_data(size);
        self.cuda_ctx.copy_device_to_host(
            @ptrCast(buf.ptr),
            self.param_nodes[index].data,
            size * @sizeOf(f32),
        ) catch unreachable;
        return buf;
    }

    /// D2H copy: パラメータ勾配を CPU スライスとして返す (デバッグ用)
    pub fn param_grad(self: *DiffCudaRuntime, index: usize) []f32 {
        const size = self.module.param_size(.{ .index = index });
        const buf = self.alloc_data(size);
        self.cuda_ctx.copy_device_to_host(
            @ptrCast(buf.ptr),
            self.param_grad_dptrs[index],
            size * @sizeOf(f32),
        ) catch unreachable;
        return buf;
    }

    /// GPU スカラー値を CPU にコピー
    pub fn copy_scalar_to_host(self: *DiffCudaRuntime, t: DiffCudaTensor) f32 {
        var val: f32 = 0;
        self.cuda_ctx.copy_device_to_host(@ptrCast(&val), t.data, @sizeOf(f32)) catch unreachable;
        return val;
    }

    /// GPU テンソルを CPU バッファにコピー
    pub fn copy_to_host(self: *DiffCudaRuntime, t: DiffCudaTensor, dst: []f32) void {
        const total = t.total_elements();
        self.cuda_ctx.copy_device_to_host(
            @ptrCast(dst.ptr),
            t.data,
            total * @sizeOf(f32),
        ) catch unreachable;
    }

    // ── Leaf ops ──

    pub fn constant_scalar(self: *DiffCudaRuntime, val: f64, dtype: u32) DiffCudaTensor {
        _ = dtype;
        const dptr = self.alloc_gpu_buf(1);
        var v: f32 = @floatCast(val);
        self.cuda_ctx.copy_host_to_device(dptr, @ptrCast(&v), @sizeOf(f32)) catch unreachable;
        return self.make_node(dptr, &.{1}, false);
    }

    pub fn constant_data(
        self: *DiffCudaRuntime,
        data: [*]const u8,
        len: usize,
        new_shape: []const usize,
        dtype: u32,
    ) DiffCudaTensor {
        _ = dtype;
        const n_floats = len / @sizeOf(f32);
        const dptr = self.alloc_gpu_buf(n_floats);
        self.cuda_ctx.copy_host_to_device(
            dptr,
            @ptrCast(data),
            n_floats * @sizeOf(f32),
        ) catch unreachable;
        return self.make_node(dptr, new_shape, false);
    }

    pub fn stop_gradient(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        return self.make_node(x.data, x.shape[0..x.ndim], false);
    }

    // ── Training mode ──

    pub fn eval(self: *DiffCudaRuntime) void {
        self.training = false;
    }

    pub fn train(self: *DiffCudaRuntime) void {
        self.training = true;
    }

    // ════════════════════════════════════════════════════════════════
    // Unary ops
    // ════════════════════════════════════════════════════════════════

    pub fn negative(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_negative, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_negative;
        }
        return node;
    }

    const RtContext = struct {
        rt: *DiffCudaRuntime,
    };

    fn backward_negative(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            const total = self_node.total_elements();
            // ga -= go → accum_scaled(ga, go, -1.0)
            const go = self_node.grad.?;
            // Use scale kernel: tmp = -go, then accum
            const tmp = rt.alloc_gpu_buf(total);
            ops.dispatch_elementwise(rt.cuda_ctx, rt.kernels.fn_negative, tmp, go, total);
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, total);
        }
    }

    pub fn gelu(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_gelu, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.data }; // cache input x
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_gelu;
        }
        return node;
    }

    const UnaryBwContext = struct {
        rt: *DiffCudaRuntime,
        cache_dptr: CUdeviceptr,
    };

    fn backward_gelu(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_gelu_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn silu(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        const sig_cache = self.alloc_gpu_buf(total);
        ops.dispatch_silu_fwd_cache(
            self.cuda_ctx,
            self.kernels.fn_silu_fwd_cache,
            out_dptr,
            sig_cache,
            x.data,
            total,
        );
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(SiluContext);
            ctx.* = .{ .rt = self, .sig_cache_dptr = sig_cache };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_silu;
        }
        return node;
    }

    const SiluContext = struct {
        rt: *DiffCudaRuntime,
        sig_cache_dptr: CUdeviceptr,
    };

    fn backward_silu(self_node: *DiffCudaNode) void {
        const ctx: *SiluContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_silu_backward(
                rt.cuda_ctx,
                rt.kernels.fn_silu_bw,
                ga,
                self_node.grad.?,
                pa.data,
                ctx.sig_cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    // ── Fused Add + SiLU ──

    pub fn add_silu(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const a_total = a.total_elements();
        const b_total = b.total_elements();
        const rg = a.requires_grad or b.requires_grad;
        const out_dptr = self.alloc_gpu_buf(a_total);
        const sig_cache = self.alloc_gpu_buf(a_total);
        ops.dispatch_add_silu_fwd_cache(
            self.cuda_ctx,
            self.kernels.fn_add_silu_fwd_cache,
            out_dptr,
            sig_cache,
            a.data,
            b.data,
            a_total,
            b_total,
        );
        const node = self.make_node(out_dptr, a.shape[0..a.ndim], rg);
        if (rg) {
            node.parents[0] = a;
            node.parents[1] = b;
            const ctx = self.alloc_context(AddSiluContext);
            ctx.* = .{ .rt = self, .sig_cache_dptr = sig_cache };
            node.context = @ptrCast(ctx);
            if (a_total == b_total) {
                node.backward_fn = &backward_add_silu_same;
            } else {
                node.backward_fn = &backward_add_silu_bcast;
            }
        }
        return node;
    }

    const AddSiluContext = struct {
        rt: *DiffCudaRuntime,
        sig_cache_dptr: CUdeviceptr,
    };

    fn backward_add_silu_same(self_node: *DiffCudaNode) void {
        const ctx: *AddSiluContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const n = self_node.total_elements();
        const ga = pa.grad orelse rt.alloc_gpu_buf_zeroed(n);
        const gb = pb.grad orelse rt.alloc_gpu_buf_zeroed(n);
        ops.dispatch_add_silu_backward_same(
            rt.cuda_ctx,
            rt.kernels.fn_add_silu_bw_same,
            ga,
            gb,
            go,
            ctx.sig_cache_dptr,
            pa.data,
            pb.data,
            n,
        );
    }

    fn backward_add_silu_bcast(self_node: *DiffCudaNode) void {
        const ctx: *AddSiluContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.total_elements();
        const b_total = pb.total_elements();
        const ga = pa.grad orelse rt.alloc_gpu_buf_zeroed(a_total);
        const gb = pb.grad orelse rt.alloc_gpu_buf_zeroed(b_total);
        ops.dispatch_add_silu_backward_bcast(
            rt.cuda_ctx,
            rt.kernels.fn_add_silu_bw_bcast,
            ga,
            gb,
            go,
            ctx.sig_cache_dptr,
            pa.data,
            pb.data,
            a_total,
            b_total,
        );
    }

    pub fn relu(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_relu, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.data };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_relu;
        }
        return node;
    }

    fn backward_relu(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_relu_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn sigmoid(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_sigmoid, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = out_dptr }; // cache output
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_sigmoid;
        }
        return node;
    }

    fn backward_sigmoid(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_sigmoid_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn tanh(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_tanh, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = out_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_tanh;
        }
        return node;
    }

    fn backward_tanh(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_tanh_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn exp(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_exp, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = out_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_exp;
        }
        return node;
    }

    fn backward_exp(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_exp_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn log(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_log, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.data };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_log;
        }
        return node;
    }

    fn backward_log(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_log_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn square(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_square, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.data };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_square;
        }
        return node;
    }

    fn backward_square(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_square_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn abs(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_abs, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.data };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_abs;
        }
        return node;
    }

    fn backward_abs(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_abs_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn sqrt(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_elementwise(self.cuda_ctx, self.kernels.fn_sqrt, out_dptr, x.data, total);
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = out_dptr }; // cache output
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_sqrt;
        }
        return node;
    }

    fn backward_sqrt(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_backward_elementwise3(
                rt.cuda_ctx,
                rt.kernels.fn_sqrt_bw,
                ga,
                self_node.grad.?,
                ctx.cache_dptr,
                self_node.total_elements(),
            );
        }
    }

    pub fn clamp(
        self: *DiffCudaRuntime,
        x: DiffCudaTensor,
        min_val: f32,
        max_val: f32,
    ) DiffCudaTensor {
        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        // clamp_kernel(out, x, min, max, n)
        var n_i: c_int = @intCast(total);
        var mn: f32 = min_val;
        var mx: f32 = max_val;
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&x.data)),
            @ptrCast(&mn),
            @ptrCast(&mx),
            @ptrCast(&n_i),
        };
        self.cuda_ctx.launch_kernel(
            self.kernels.fn_clamp,
            .{ ops.grid_for(total), 1, 1 },
            .{ ops.BLOCK_SIZE, 1, 1 },
            0,
            &params,
        ) catch unreachable;
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(ClampContext);
            ctx.* = .{ .rt = self, .min_val = min_val, .max_val = max_val };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_clamp;
        }
        return node;
    }

    const ClampContext = struct {
        rt: *DiffCudaRuntime,
        min_val: f32,
        max_val: f32,
    };

    fn backward_clamp(self_node: *DiffCudaNode) void {
        const ctx: *ClampContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_clamp_backward(
                rt.cuda_ctx,
                rt.kernels.fn_clamp_bw,
                ga,
                self_node.grad.?,
                pa.data,
                ctx.min_val,
                ctx.max_val,
                self_node.total_elements(),
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Binary ops
    // ════════════════════════════════════════════════════════════════

    /// 放送対応の二項演算ノードを構築する共通ヘルパー。
    /// out_dptr はすでに計算済みのバッファ、shape_src は出力形状を提供するテンソル、
    /// backward_fn は rg=true の場合に設定される backward 関数。
    fn make_binop_node(
        self: *DiffCudaRuntime,
        out_dptr: CUdeviceptr,
        shape_src: DiffCudaTensor,
        a: DiffCudaTensor,
        b: DiffCudaTensor,
        rg: bool,
        backward_fn: BackwardFn,
    ) DiffCudaTensor {
        const node = self.make_node(out_dptr, shape_src.shape[0..shape_src.ndim], rg);
        if (rg) {
            node.parents[0] = a;
            node.parents[1] = b;
            const ctx = self.alloc_context(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = backward_fn;
        }
        return node;
    }

    pub fn add(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const a_total = a.total_elements();
        const b_total = b.total_elements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out_dptr = self.alloc_gpu_buf(a_total);
            ops.dispatch_broadcast_binop(
                self.cuda_ctx,
                self.kernels.fn_add,
                out_dptr,
                a.data,
                b.data,
                a_total,
                b_total,
            );
            return self.make_binop_node(out_dptr, a, a, b, rg, &backward_add_same);
        }

        if (b_total < a_total and a_total % b_total == 0) {
            const out_dptr = self.alloc_gpu_buf(a_total);
            ops.dispatch_broadcast_binop(
                self.cuda_ctx,
                self.kernels.fn_add,
                out_dptr,
                a.data,
                b.data,
                a_total,
                b_total,
            );
            return self.make_binop_node(out_dptr, a, a, b, rg, &backward_add_broadcast_b);
        }

        if (a_total < b_total and b_total % a_total == 0) {
            const out_dptr = self.alloc_gpu_buf(b_total);
            ops.dispatch_broadcast_binop(
                self.cuda_ctx,
                self.kernels.fn_add,
                out_dptr,
                b.data,
                a.data,
                b_total,
                a_total,
            );
            return self.make_binop_node(out_dptr, b, a, b, rg, &backward_add_broadcast_a);
        }

        @panic("add: incompatible shapes for broadcast");
    }

    fn backward_add_same(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const total = self_node.total_elements();
        if (pa.grad) |ga| {
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, total);
        }
        if (pb.grad) |gb| {
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, gb, go, total);
        }
    }

    fn backward_add_broadcast_b(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.total_elements();
        const b_total = pb.total_elements();
        if (pa.grad) |ga| {
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, a_total);
        }
        if (pb.grad) |gb| ops.dispatch_reduce_to_broadcast(
            rt.cuda_ctx,
            rt.kernels.fn_reduce_add_bcast,
            gb,
            go,
            a_total,
            b_total,
        );
    }

    fn backward_add_broadcast_a(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.total_elements();
        const b_total = pb.total_elements();
        if (pa.grad) |ga| ops.dispatch_reduce_to_broadcast(
            rt.cuda_ctx,
            rt.kernels.fn_reduce_add_bcast,
            ga,
            go,
            b_total,
            a_total,
        );
        if (pb.grad) |gb| {
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, gb, go, b_total);
        }
    }

    pub fn sub(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const a_total = a.total_elements();
        const b_total = b.total_elements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out_dptr = self.alloc_gpu_buf(a_total);
            ops.dispatch_broadcast_binop(
                self.cuda_ctx,
                self.kernels.fn_sub,
                out_dptr,
                a.data,
                b.data,
                a_total,
                b_total,
            );
            const node = self.make_node(out_dptr, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.alloc_context(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backward_sub_same;
            }
            return node;
        }

        if (b_total < a_total and a_total % b_total == 0) {
            const out_dptr = self.alloc_gpu_buf(a_total);
            ops.dispatch_broadcast_binop(
                self.cuda_ctx,
                self.kernels.fn_sub,
                out_dptr,
                a.data,
                b.data,
                a_total,
                b_total,
            );
            const node = self.make_node(out_dptr, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.alloc_context(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backward_sub_broadcast_b;
            }
            return node;
        }

        @panic("sub: incompatible shapes for broadcast");
    }

    fn backward_sub_same(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const total = self_node.total_elements();
        if (pa.grad) |ga| {
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, total);
        }
        if (pb.grad) |gb| {
            // gb -= go
            const tmp = rt.alloc_gpu_buf(total);
            ops.dispatch_elementwise(rt.cuda_ctx, rt.kernels.fn_negative, tmp, go, total);
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, gb, tmp, total);
        }
    }

    fn backward_sub_broadcast_b(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.total_elements();
        const b_total = pb.total_elements();
        if (pa.grad) |ga| {
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, a_total);
        }
        if (pb.grad) |gb| ops.dispatch_reduce_to_broadcast(
            rt.cuda_ctx,
            rt.kernels.fn_reduce_sub_bcast,
            gb,
            go,
            a_total,
            b_total,
        );
    }

    pub fn mul(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const a_total = a.total_elements();
        const b_total = b.total_elements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out_dptr = self.alloc_gpu_buf(a_total);
            ops.dispatch_broadcast_binop(
                self.cuda_ctx,
                self.kernels.fn_mul,
                out_dptr,
                a.data,
                b.data,
                a_total,
                b_total,
            );
            return self.make_binop_node(out_dptr, a, a, b, rg, &backward_mul_same);
        }

        if (b_total <= a_total and a_total % b_total == 0) {
            const out_dptr = self.alloc_gpu_buf(a_total);
            ops.dispatch_broadcast_binop(
                self.cuda_ctx,
                self.kernels.fn_mul,
                out_dptr,
                a.data,
                b.data,
                a_total,
                b_total,
            );
            return self.make_binop_node(out_dptr, a, a, b, rg, &backward_mul_broadcast_b);
        }

        if (a_total < b_total and b_total % a_total == 0) {
            const out_dptr = self.alloc_gpu_buf(b_total);
            ops.dispatch_broadcast_binop(
                self.cuda_ctx,
                self.kernels.fn_mul,
                out_dptr,
                b.data,
                a.data,
                b_total,
                a_total,
            );
            return self.make_binop_node(out_dptr, b, a, b, rg, &backward_mul_broadcast_a);
        }

        @panic("mul: incompatible shapes for broadcast");
    }

    fn backward_mul_same(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const n = self_node.total_elements();
        // Use a zero buffer for the grad we don't need, or dispatch individually
        if (pa.grad != null and pb.grad != null) {
            ops.dispatch_mul_backward_same(
                rt.cuda_ctx,
                rt.kernels.fn_mul_bw_same,
                pa.grad.?,
                pb.grad.?,
                go,
                pa.data,
                pb.data,
                n,
            );
        } else if (pa.grad) |ga| {
            // ga += go * b
            const tmp = rt.alloc_gpu_buf(n);
            ops.dispatch_broadcast_binop(rt.cuda_ctx, rt.kernels.fn_mul, tmp, go, pb.data, n, n);
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, n);
        } else if (pb.grad) |gb| {
            const tmp = rt.alloc_gpu_buf(n);
            ops.dispatch_broadcast_binop(rt.cuda_ctx, rt.kernels.fn_mul, tmp, go, pa.data, n, n);
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, gb, tmp, n);
        }
    }

    fn backward_mul_broadcast_b(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.total_elements();
        const b_total = pb.total_elements();
        if (pa.grad) |ga| ops.dispatch_mul_backward_broadcast(
            rt.cuda_ctx,
            rt.kernels.fn_mul_bw_bcast_ga,
            ga,
            go,
            pb.data,
            a_total,
            b_total,
        );
        if (pb.grad) |gb| ops.dispatch_mul_backward_broadcast(
            rt.cuda_ctx,
            rt.kernels.fn_mul_bw_bcast_gb,
            gb,
            go,
            pa.data,
            a_total,
            b_total,
        );
    }

    fn backward_mul_broadcast_a(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.total_elements();
        const b_total = pb.total_elements();
        // a is smaller, b is larger. Swap roles.
        if (pa.grad) |ga| ops.dispatch_mul_backward_broadcast(
            rt.cuda_ctx,
            rt.kernels.fn_mul_bw_bcast_gb,
            ga,
            go,
            pb.data,
            b_total,
            a_total,
        );
        if (pb.grad) |gb| ops.dispatch_mul_backward_broadcast(
            rt.cuda_ctx,
            rt.kernels.fn_mul_bw_bcast_ga,
            gb,
            go,
            pa.data,
            b_total,
            a_total,
        );
    }

    pub fn div(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const total = a.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        // div = a * (1/b), but we can compute directly on CPU-side or use a kernel
        // For now, implement as: out[i] = a[i] / b[i] using a two-step approach
        // Actually, there's no div forward kernel. Use scale approach: compute b_inv, then mul.
        // Simpler: just do it as a/b via the broadcast kernels.
        // div_backward_kernel exists. For forward, just compute via CPU upload or use mul(a, 1/b).
        // Actually the simplest approach: download, compute, upload. But that's slow.
        // Better: just compute a * (1/b) using existing kernels.
        // But we don't have a reciprocal kernel.
        // For now, fall back to host-side computation for div forward.
        const a_data = self.alloc_data(total);
        const b_data = self.alloc_data(total);
        self.cuda_ctx.copy_device_to_host(
            @ptrCast(a_data.ptr),
            a.data,
            total * @sizeOf(f32),
        ) catch unreachable;
        self.cuda_ctx.copy_device_to_host(
            @ptrCast(b_data.ptr),
            b.data,
            total * @sizeOf(f32),
        ) catch unreachable;
        const out_data = self.alloc_data(total);
        for (0..total) |i| out_data[i] = a_data[i] / b_data[i];
        self.cuda_ctx.copy_host_to_device(
            out_dptr,
            @ptrCast(out_data.ptr),
            total * @sizeOf(f32),
        ) catch unreachable;

        const rg = a.requires_grad or b.requires_grad;
        const node = self.make_node(out_dptr, a.shape[0..a.ndim], rg);
        if (rg) {
            node.parents[0] = a;
            node.parents[1] = b;
            const ctx = self.alloc_context(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_div;
        }
        return node;
    }

    fn backward_div(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const n = self_node.total_elements();
        const ga = pa.grad orelse @as(CUdeviceptr, 0);
        const gb = pb.grad orelse @as(CUdeviceptr, 0);
        if (pa.grad != null or pb.grad != null) {
            // Use temp buffers for grads we don't track
            const real_ga = if (pa.grad != null) ga else rt.alloc_gpu_buf_zeroed(n);
            const real_gb = if (pb.grad != null) gb else rt.alloc_gpu_buf_zeroed(n);
            ops.dispatch_div_backward(
                rt.cuda_ctx,
                rt.kernels.fn_div_bw,
                real_ga,
                real_gb,
                self_node.grad.?,
                pa.data,
                pb.data,
                n,
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Matmul
    // ════════════════════════════════════════════════════════════════

    /// matmul の出力ノードを構築する共通ヘルパー。
    fn make_matmul_node(
        self: *DiffCudaRuntime,
        out_dptr: CUdeviceptr,
        out_shape: []const usize,
        a: DiffCudaTensor,
        b: DiffCudaTensor,
        rg: bool,
        backward_fn: BackwardFn,
    ) DiffCudaTensor {
        const node = self.make_node(out_dptr, out_shape, rg);
        if (rg) {
            node.parents[0] = a;
            node.parents[1] = b;
            const ctx = self.alloc_context(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = backward_fn;
        }
        return node;
    }

    fn matmul2d(
        self: *DiffCudaRuntime,
        a: DiffCudaTensor,
        b: DiffCudaTensor,
        rg: bool,
    ) DiffCudaTensor {
        const M = a.shape[0];
        const K = a.shape[1];
        const N = b.shape[1];
        const out_dptr = self.alloc_gpu_buf(M * N);
        self.cuda_ctx.sgemm(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            @intCast(N),
            @intCast(M),
            @intCast(K),
            1.0,
            b.data,
            @intCast(N),
            a.data,
            @intCast(K),
            0.0,
            out_dptr,
            @intCast(N),
        ) catch unreachable;
        return self.make_matmul_node(out_dptr, &.{ M, N }, a, b, rg, &backward_matmul2d);
    }

    fn matmul3d(
        self: *DiffCudaRuntime,
        a: DiffCudaTensor,
        b: DiffCudaTensor,
        rg: bool,
    ) DiffCudaTensor {
        const B = a.shape[0];
        const M = a.shape[1];
        const K = a.shape[2];
        const N = b.shape[2];
        const out_dptr = self.alloc_gpu_buf(B * M * N);
        self.cuda_ctx.sgemm_strided_batched(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            @intCast(N),
            @intCast(M),
            @intCast(K),
            1.0,
            b.data,
            @intCast(N),
            @intCast(K * N),
            a.data,
            @intCast(K),
            @intCast(M * K),
            0.0,
            out_dptr,
            @intCast(N),
            @intCast(M * N),
            @intCast(B),
        ) catch unreachable;
        return self.make_matmul_node(out_dptr, &.{ B, M, N }, a, b, rg, &backward_matmul3d);
    }

    fn matmul2_d3d(
        self: *DiffCudaRuntime,
        a: DiffCudaTensor,
        b: DiffCudaTensor,
        rg: bool,
    ) DiffCudaTensor {
        const B = b.shape[0];
        const M = a.shape[0];
        const K = a.shape[1];
        const N = b.shape[2];
        const out_dptr = self.alloc_gpu_buf(B * M * N);
        for (0..B) |batch| {
            const b_off = batch * K * N;
            const o_off = batch * M * N;
            self.cuda_ctx.sgemm(
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                @intCast(N),
                @intCast(M),
                @intCast(K),
                1.0,
                b.data + b_off * @sizeOf(f32),
                @intCast(N),
                a.data,
                @intCast(K),
                0.0,
                out_dptr + o_off * @sizeOf(f32),
                @intCast(N),
            ) catch unreachable;
        }
        return self.make_matmul_node(out_dptr, &.{ B, M, N }, a, b, rg, &backward_matmul2_d3d);
    }

    pub fn matmul(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const rg = a.requires_grad or b.requires_grad;

        if (a.ndim == 2 and b.ndim == 2) return self.matmul2d(a, b, rg);
        if (a.ndim == 3 and b.ndim == 3) return self.matmul3d(a, b, rg);
        if (a.ndim == 2 and b.ndim == 3) return self.matmul2_d3d(a, b, rg);

        @panic("matmul: unsupported shape combination (expected 2D or 3D)");
    }

    fn backward_matmul2d(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const M = pa.shape[0];
        const K = pa.shape[1];
        const N = pb.shape[1];
        // dA += go @ B^T: sgemm(N=K, M=M, K=N, B^T, go)
        if (pa.grad) |ga| rt.cuda_ctx.sgemm_accum(
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            @intCast(K),
            @intCast(M),
            @intCast(N),
            1.0,
            pb.data,
            @intCast(N),
            go,
            @intCast(N),
            ga,
            @intCast(K),
        ) catch unreachable;
        // dB += A^T @ go: sgemm(N=N, M=K, K=M, go, A^T)
        if (pb.grad) |gb| rt.cuda_ctx.sgemm_accum(
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            @intCast(N),
            @intCast(K),
            @intCast(M),
            1.0,
            go,
            @intCast(N),
            pa.data,
            @intCast(K),
            gb,
            @intCast(N),
        ) catch unreachable;
    }

    fn backward_matmul3d(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const B = pa.shape[0];
        const M = pa.shape[1];
        const K = pa.shape[2];
        const N = pb.shape[2];
        // dA += go @ B^T: batched sgemm
        if (pa.grad) |ga| rt.cuda_ctx.sgemm_strided_batched(
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            @intCast(K),
            @intCast(M),
            @intCast(N),
            1.0,
            pb.data,
            @intCast(N),
            @intCast(K * N),
            go,
            @intCast(N),
            @intCast(M * N),
            1.0,
            ga,
            @intCast(K),
            @intCast(M * K),
            @intCast(B),
        ) catch unreachable;
        // dB += A^T @ go: batched sgemm
        if (pb.grad) |gb| rt.cuda_ctx.sgemm_strided_batched(
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            @intCast(N),
            @intCast(K),
            @intCast(M),
            1.0,
            go,
            @intCast(N),
            @intCast(M * N),
            pa.data,
            @intCast(K),
            @intCast(M * K),
            1.0,
            gb,
            @intCast(N),
            @intCast(K * N),
            @intCast(B),
        ) catch unreachable;
    }

    fn backward_matmul2_d3d(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const B = pb.shape[0];
        const M = pa.shape[0];
        const K = pa.shape[1];
        const N = pb.shape[2];
        // dA (2D) += sum_b(go_b @ B_b^T): compute batched into temp 3D, then reduce
        if (pa.grad) |ga| {
            // Use batched sgemm into a temp [B, M, K] buffer, then reduce-sum over B
            const tmp = rt.alloc_gpu_buf_zeroed(B * M * K);
            rt.cuda_ctx.sgemm_strided_batched(
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                @intCast(K),
                @intCast(M),
                @intCast(N),
                1.0,
                pb.data,
                @intCast(N),
                @intCast(K * N),
                go,
                @intCast(N),
                @intCast(M * N),
                0.0,
                tmp,
                @intCast(K),
                @intCast(M * K),
                @intCast(B),
            ) catch unreachable;
            // Reduce: ga += sum over batch dimension
            for (0..B) |batch| {
                ops.dispatch_accum_grad(
                    rt.cuda_ctx,
                    rt.kernels.fn_accum_grad,
                    ga,
                    tmp + batch * M * K * @sizeOf(f32),
                    M * K,
                );
            }
        }
        // dB (3D) += A^T @ go_b: batched sgemm (A is shared across batches, stride_a=0)
        if (pb.grad) |gb| rt.cuda_ctx.sgemm_strided_batched(
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            @intCast(N),
            @intCast(K),
            @intCast(M),
            1.0,
            go,
            @intCast(N),
            @intCast(M * N),
            pa.data,
            @intCast(K),
            0, // stride_a=0: same A for all batches
            1.0,
            gb,
            @intCast(N),
            @intCast(K * N),
            @intCast(B),
        ) catch unreachable;
    }

    // ════════════════════════════════════════════════════════════════
    // Shape ops
    // ════════════════════════════════════════════════════════════════

    pub fn reshape(
        self: *DiffCudaRuntime,
        x: DiffCudaTensor,
        new_shape: []const usize,
    ) DiffCudaTensor {
        const node = self.make_node(x.data, new_shape, x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_reshape;
        }
        return node;
    }

    fn backward_reshape(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            if (ga == go) return; // same buffer, no-op
            ops.dispatch_accum_grad(
                rt.cuda_ctx,
                rt.kernels.fn_accum_grad,
                ga,
                go,
                self_node.total_elements(),
            );
        }
    }

    pub fn transpose(self: *DiffCudaRuntime, x: DiffCudaTensor, d1: u64, d2: u64) DiffCudaTensor {
        const dim1: usize = @intCast(d1);
        const dim2: usize = @intCast(d2);

        if (x.ndim == 3 and dim1 == 1 and dim2 == 2) {
            const B = x.shape[0];
            const R = x.shape[1];
            const C = x.shape[2];
            const total = B * R * C;
            const out_dptr = self.alloc_gpu_buf(total);
            for (0..B) |b| {
                const in_off = b * R * C;
                const out_off = b * C * R;
                ops.dispatch_transpose2d(
                    self.cuda_ctx,
                    self.kernels.fn_transpose,
                    out_dptr + out_off * @sizeOf(f32),
                    x.data + in_off * @sizeOf(f32),
                    R,
                    C,
                );
            }
            const node = self.make_node(out_dptr, &.{ B, C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                const ctx = self.alloc_context(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backward_transpose3d;
            }
            return node;
        }

        if (x.ndim == 2 and dim1 == 0 and dim2 == 1) {
            const R = x.shape[0];
            const C = x.shape[1];
            const out_dptr = self.alloc_gpu_buf(R * C);
            ops.dispatch_transpose2d(
                self.cuda_ctx,
                self.kernels.fn_transpose,
                out_dptr,
                x.data,
                R,
                C,
            );
            const node = self.make_node(out_dptr, &.{ C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                const ctx = self.alloc_context(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backward_transpose2d;
            }
            return node;
        }

        @panic("transpose: unsupported ndim (expected 2D or 3D)");
    }

    fn backward_transpose3d(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const B = pa.shape[0];
            const R = pa.shape[1];
            const C = pa.shape[2];
            // go is [B, C, R], transpose back to [B, R, C]
            const tmp = rt.alloc_gpu_buf(B * R * C);
            for (0..B) |b| {
                ops.dispatch_transpose2d(
                    rt.cuda_ctx,
                    rt.kernels.fn_transpose,
                    tmp + b * R * C * @sizeOf(f32),
                    go + b * C * R * @sizeOf(f32),
                    C,
                    R,
                );
            }
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, B * R * C);
        }
    }

    fn backward_transpose2d(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const R = pa.shape[0];
            const C = pa.shape[1];
            const tmp = rt.alloc_gpu_buf(R * C);
            ops.dispatch_transpose2d(rt.cuda_ctx, rt.kernels.fn_transpose, tmp, go, C, R);
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, R * C);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Softmax
    // ════════════════════════════════════════════════════════════════

    pub fn softmax(self: *DiffCudaRuntime, x: DiffCudaTensor, axis: i64) DiffCudaTensor {
        _ = axis;
        const total = x.total_elements();
        const cols = x.last_dim();
        const rows = total / cols;
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_softmax_out(
            self.cuda_ctx,
            self.kernels.fn_softmax_out,
            out_dptr,
            x.data,
            rows,
            cols,
        );
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_softmax;
        }
        return node;
    }

    fn backward_softmax(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            const total = self_node.total_elements();
            const cols = self_node.last_dim();
            const rows = total / cols;
            ops.dispatch_softmax_backward(
                rt.cuda_ctx,
                rt.kernels.fn_softmax_bw,
                ga,
                self_node.grad.?,
                self_node.data,
                rows,
                cols,
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Reduction
    // ════════════════════════════════════════════════════════════════

    /// reductionSum の出力ノードを構築する共通ヘルパー。
    fn make_reduction_sum_node(
        self: *DiffCudaRuntime,
        out_dptr: CUdeviceptr,
        out_shape: []const usize,
        x: DiffCudaTensor,
        backward_fn: BackwardFn,
    ) DiffCudaTensor {
        const node = self.make_node(out_dptr, out_shape, x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = backward_fn;
        }
        return node;
    }

    fn reduction_sum2d(
        self: *DiffCudaRuntime,
        x: DiffCudaTensor,
        actual_axis: usize,
    ) DiffCudaTensor {
        const rows = x.shape[0];
        const cols = x.shape[1];
        if (actual_axis == 1) {
            const out_dptr = self.alloc_gpu_buf(rows);
            ops.dispatch_reduction_sum_rows(
                self.cuda_ctx,
                self.kernels.fn_reduce_sum_rows,
                out_dptr,
                x.data,
                rows,
                cols,
            );
            return self.make_reduction_sum_node(
                out_dptr,
                &.{ rows, 1 },
                x,
                &backward_reduction_sum_axis1,
            );
        } else {
            const out_dptr = self.alloc_gpu_buf(cols);
            ops.dispatch_reduction_sum_cols(
                self.cuda_ctx,
                self.kernels.fn_reduce_sum_cols,
                out_dptr,
                x.data,
                rows,
                cols,
            );
            return self.make_reduction_sum_node(
                out_dptr,
                &.{ 1, cols },
                x,
                &backward_reduction_sum_axis0,
            );
        }
    }

    fn reduction_sum1d(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const out_dptr = self.alloc_gpu_buf(1);
        ops.dispatch_reduction_sum1d(
            self.cuda_ctx,
            self.kernels.fn_reduce_sum_1d,
            out_dptr,
            x.data,
            x.total_elements(),
        );
        return self.make_reduction_sum_node(out_dptr, &.{1}, x, &backward_reduction_sum1d);
    }

    // ndim >= 3: flatten around the reduction axis (same as CPU)
    fn reduction_sum_nd(
        self: *DiffCudaRuntime,
        x: DiffCudaTensor,
        actual_axis: usize,
    ) DiffCudaTensor {
        const total = x.total_elements();
        var before: usize = 1;
        for (0..actual_axis) |d| before *= x.shape[d];
        const axis_dim = x.shape[actual_axis];
        if (actual_axis == x.ndim - 1) {
            const flat = self.reshape(x, &.{ total / axis_dim, axis_dim });
            const reduced = self.reduction_sum(flat, 1);
            var new_shape: [8]usize = undefined;
            for (0..x.ndim - 1) |d| new_shape[d] = x.shape[d];
            new_shape[x.ndim - 1] = 1;
            return self.reshape(reduced, new_shape[0..x.ndim]);
        } else if (actual_axis == 0) {
            const flat = self.reshape(x, &.{ axis_dim, total / axis_dim });
            const reduced = self.reduction_sum(flat, 0);
            var new_shape: [8]usize = undefined;
            new_shape[0] = 1;
            for (1..x.ndim) |d| new_shape[d] = x.shape[d];
            return self.reshape(reduced, new_shape[0..x.ndim]);
        } else {
            var after: usize = 1;
            for (actual_axis + 1..x.ndim) |d| after *= x.shape[d];
            const r3 = self.reshape(x, &.{ before, axis_dim, after });
            const t3 = self.transpose(r3, 1, 2);
            const flat = self.reshape(t3, &.{ before * after, axis_dim });
            const reduced = self.reduction_sum(flat, 1);
            var new_shape: [8]usize = undefined;
            for (0..x.ndim) |d| new_shape[d] = x.shape[d];
            new_shape[actual_axis] = 1;
            return self.reshape(reduced, new_shape[0..x.ndim]);
        }
    }

    pub fn reduction_sum(self: *DiffCudaRuntime, x: DiffCudaTensor, axis: i64) DiffCudaTensor {
        const actual_axis: usize = if (axis < 0)
            @intCast(@as(i64, @intCast(x.ndim)) + axis)
        else
            @intCast(axis);

        if (x.ndim == 2) return self.reduction_sum2d(x, actual_axis);
        if (x.ndim == 1) return self.reduction_sum1d(x);
        if (x.ndim >= 3) return self.reduction_sum_nd(x, actual_axis);

        @panic("reductionSum: unsupported ndim/axis combination");
    }

    fn backward_reduction_sum_axis1(self_node: *DiffCudaNode) void {
        // [rows, cols] → [rows, 1]: go has rows elements
        // Need: ga[r*cols + c] += go[r] for all r, c
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            const go = self_node.grad.?;
            // Per-row broadcast: ga_row[c] += go[r] (scalar broadcast to cols elements)
            for (0..rows) |r| {
                const ga_row = ga + r * cols * @sizeOf(f32);
                const go_r = go + r * @sizeOf(f32);
                ops.dispatch_broadcast_binop(
                    rt.cuda_ctx,
                    rt.kernels.fn_add,
                    ga_row,
                    ga_row,
                    go_r,
                    cols,
                    1,
                );
            }
        }
    }

    fn backward_reduction_sum_axis0(self_node: *DiffCudaNode) void {
        // [rows, cols] → [1, cols]: go has cols elements
        // Need: ga[r*cols + c] += go[c] for all r, c
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            // go[i % cols] = go[c], so broadcast-add works directly
            ops.dispatch_broadcast_binop(
                rt.cuda_ctx,
                rt.kernels.fn_add,
                ga,
                ga,
                self_node.grad.?,
                rows * cols,
                cols,
            );
        }
    }

    fn backward_reduction_sum1d(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            const total = pa.total_elements();
            // go is scalar [1], broadcast-add to all elements: ga[i] += go[0]
            ops.dispatch_broadcast_binop(
                rt.cuda_ctx,
                rt.kernels.fn_add,
                ga,
                ga,
                self_node.grad.?,
                total,
                1,
            );
        }
    }

    pub fn reduction_mean(self: *DiffCudaRuntime, x: DiffCudaTensor, axis: i64) DiffCudaTensor {
        // mean = sum / count
        const s = self.reduction_sum(x, axis);
        const actual_axis: usize = if (axis < 0)
            @intCast(@as(i64, @intCast(x.ndim)) + axis)
        else
            @intCast(axis);
        const count = x.shape[actual_axis];
        const scale = 1.0 / @as(f32, @floatFromInt(count));
        const total = s.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        ops.dispatch_scale(self.cuda_ctx, self.kernels.fn_scale, out_dptr, s.data, scale, total);
        const node = self.make_node(out_dptr, s.shape[0..s.ndim], s.requires_grad);
        if (s.requires_grad) {
            node.parents[0] = s;
            const ctx = self.alloc_context(ScaleCtx);
            ctx.* = .{ .rt = self, .scale = scale };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_scale;
        }
        return node;
    }

    const ScaleCtx = struct {
        rt: *DiffCudaRuntime,
        scale: f32,
    };

    fn backward_scale(self_node: *DiffCudaNode) void {
        const ctx: *ScaleCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            const total = self_node.total_elements();
            const go = self_node.grad.?;
            // ga += go * scale
            const tmp = rt.alloc_gpu_buf(total);
            ops.dispatch_scale(rt.cuda_ctx, rt.kernels.fn_scale, tmp, go, ctx.scale, total);
            ops.dispatch_accum_grad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, total);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // LayerNorm
    // ════════════════════════════════════════════════════════════════

    pub fn layer_norm(
        self: *DiffCudaRuntime,
        x: DiffCudaTensor,
        gamma: DiffCudaTensor,
        beta: DiffCudaTensor,
        eps: f32,
        axis: i64,
    ) DiffCudaTensor {
        _ = axis;
        const total = x.total_elements();
        const dim = x.last_dim();
        const rows = total / dim;
        const out_dptr = self.alloc_gpu_buf(total);
        const x_norm_dptr = self.alloc_gpu_buf(total);
        const inv_stds_dptr = self.alloc_gpu_buf(rows);
        ops.dispatch_layer_norm_fwd(
            self.cuda_ctx,
            self.kernels.fn_layernorm_fwd,
            out_dptr,
            x_norm_dptr,
            inv_stds_dptr,
            x.data,
            gamma.data,
            beta.data,
            rows,
            dim,
            eps,
        );
        const rg = x.requires_grad or gamma.requires_grad or beta.requires_grad;
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], rg);
        if (rg) {
            node.parents[0] = x;
            node.parents[1] = gamma;
            node.parents[2] = beta;
            const ctx = self.alloc_context(LayerNormContext);
            ctx.* = .{ .rt = self, .x_norm_dptr = x_norm_dptr, .inv_stds_dptr = inv_stds_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_layer_norm;
        }
        return node;
    }

    const LayerNormContext = struct {
        rt: *DiffCudaRuntime,
        x_norm_dptr: CUdeviceptr,
        inv_stds_dptr: CUdeviceptr,
    };

    fn backward_layer_norm(self_node: *DiffCudaNode) void {
        const ctx: *LayerNormContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const px = self_node.parents[0].?;
        const pgamma = self_node.parents[1].?;
        const pbeta = self_node.parents[2].?;
        const go = self_node.grad.?;
        const total = self_node.total_elements();
        const dim = self_node.last_dim();
        const rows = total / dim;

        if (pbeta.grad != null or pgamma.grad != null) {
            const gg = pgamma.grad orelse rt.alloc_gpu_buf_zeroed(dim);
            const gb = pbeta.grad orelse rt.alloc_gpu_buf_zeroed(dim);
            ops.dispatch_layer_norm_backward_dg_db(
                rt.cuda_ctx,
                rt.kernels.fn_ln_bw_dg_db,
                gg,
                gb,
                go,
                ctx.x_norm_dptr,
                rows,
                dim,
            );
        }

        if (px.grad) |gx| {
            ops.dispatch_layer_norm_backward_dx(
                rt.cuda_ctx,
                rt.kernels.fn_ln_bw_dx,
                gx,
                go,
                pgamma.data,
                ctx.x_norm_dptr,
                ctx.inv_stds_dptr,
                rows,
                dim,
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Dropout
    // ════════════════════════════════════════════════════════════════

    pub fn dropout(self: *DiffCudaRuntime, x: DiffCudaTensor, rate: f32) DiffCudaTensor {
        if (!self.training) return x;

        const total = x.total_elements();
        const out_dptr = self.alloc_gpu_buf(total);
        const mask_dptr = self.alloc_gpu_buf(total);
        const seed = self.prng.random().int(u64);
        const inv_keep = 1.0 / (1.0 - rate);
        ops.dispatch_dropout(
            self.cuda_ctx,
            self.kernels.fn_dropout,
            out_dptr,
            mask_dptr,
            x.data,
            seed,
            rate,
            inv_keep,
            total,
        );
        const node = self.make_node(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(DropoutCtx);
            ctx.* = .{ .rt = self, .mask_dptr = mask_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_dropout;
        }
        return node;
    }

    const DropoutCtx = struct {
        rt: *DiffCudaRuntime,
        mask_dptr: CUdeviceptr,
    };

    fn backward_dropout(self_node: *DiffCudaNode) void {
        const ctx: *DropoutCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_dropout_backward(
                rt.cuda_ctx,
                rt.kernels.fn_dropout_bw,
                ga,
                self_node.grad.?,
                ctx.mask_dptr,
                self_node.total_elements(),
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Gather (Embedding lookup)
    // ════════════════════════════════════════════════════════════════

    pub fn gather(
        self: *DiffCudaRuntime,
        table: DiffCudaTensor,
        indices: []const u32,
    ) DiffCudaTensor {
        const embed_dim = table.shape[1];
        const num_indices = indices.len;
        const out_dptr = self.alloc_gpu_buf(num_indices * embed_dim);
        // Upload indices to GPU
        const idx_dptr = self.alloc_gpu_buf(num_indices); // reuse float buf for u32 (same size)
        self.cuda_ctx.copy_host_to_device(
            idx_dptr,
            @ptrCast(indices.ptr),
            num_indices * @sizeOf(u32),
        ) catch unreachable;
        ops.dispatch_gather(
            self.cuda_ctx,
            self.kernels.fn_gather,
            out_dptr,
            table.data,
            idx_dptr,
            num_indices,
            embed_dim,
        );
        const node = self.make_node(out_dptr, &.{ num_indices, embed_dim }, table.requires_grad);
        if (table.requires_grad) {
            node.parents[0] = table;
            const ctx = self.alloc_context(GatherCtx);
            ctx.* = .{ .rt = self, .idx_dptr = idx_dptr, .num_indices = num_indices };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_gather;
        }
        return node;
    }

    const GatherCtx = struct {
        rt: *DiffCudaRuntime,
        idx_dptr: CUdeviceptr,
        num_indices: usize,
    };

    fn backward_gather(self_node: *DiffCudaNode) void {
        const ctx: *GatherCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            const embed_dim = pa.shape[1];
            ops.dispatch_scatter_add(
                rt.cuda_ctx,
                rt.kernels.fn_scatter_add,
                ga,
                self_node.grad.?,
                ctx.idx_dptr,
                ctx.num_indices,
                embed_dim,
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Loss functions
    // ════════════════════════════════════════════════════════════════

    pub fn mse_loss(
        self: *DiffCudaRuntime,
        pred: DiffCudaTensor,
        target: []const f32,
    ) DiffCudaTensor {
        const total = pred.total_elements();
        const target_dptr = self.alloc_gpu_buf(total);
        self.cuda_ctx.copy_host_to_device(
            target_dptr,
            @ptrCast(target.ptr),
            total * @sizeOf(f32),
        ) catch unreachable;
        const out_dptr = self.alloc_gpu_buf_zeroed(1);
        ops.dispatch_loss_forward(
            self.cuda_ctx,
            self.kernels.fn_mse_fwd,
            out_dptr,
            pred.data,
            target_dptr,
            total,
        );
        const node = self.make_node(out_dptr, &.{1}, pred.requires_grad);
        if (pred.requires_grad) {
            node.parents[0] = pred;
            const ctx = self.alloc_context(LossCtx);
            ctx.* = .{ .rt = self, .target_dptr = target_dptr, .is_mse = true };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_mse_loss;
        }
        return node;
    }

    const LossCtx = struct {
        rt: *DiffCudaRuntime,
        target_dptr: CUdeviceptr,
        is_mse: bool,
    };

    fn backward_mse_loss(self_node: *DiffCudaNode) void {
        const ctx: *LossCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_loss_backward(
                rt.cuda_ctx,
                rt.kernels.fn_mse_bw,
                ga,
                self_node.grad.?,
                pa.data,
                ctx.target_dptr,
                pa.total_elements(),
            );
        }
    }

    pub fn cross_entropy_loss_with_indices(
        self: *DiffCudaRuntime,
        logits: DiffCudaTensor,
        indices: []const u32,
    ) DiffCudaTensor {
        const batch = logits.shape[0];
        const num_classes = logits.shape[1];
        const idx_dptr = self.alloc_gpu_buf(batch); // u32 same size as f32
        self.cuda_ctx.copy_host_to_device(
            idx_dptr,
            @ptrCast(indices.ptr),
            batch * @sizeOf(u32),
        ) catch unreachable;
        const softmax_cache = self.alloc_gpu_buf(batch * num_classes);
        const out_dptr = self.alloc_gpu_buf_zeroed(1);
        ops.dispatch_cross_entropy_forward(
            self.cuda_ctx,
            self.kernels.fn_ce_fwd,
            out_dptr,
            softmax_cache,
            logits.data,
            idx_dptr,
            batch,
            num_classes,
        );
        const node = self.make_node(out_dptr, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.alloc_context(CECtx);
            ctx.* = .{ .rt = self, .softmax_cache_dptr = softmax_cache, .idx_dptr = idx_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_cross_entropy;
        }
        return node;
    }

    const CECtx = struct {
        rt: *DiffCudaRuntime,
        softmax_cache_dptr: CUdeviceptr,
        idx_dptr: CUdeviceptr,
    };

    fn backward_cross_entropy(self_node: *DiffCudaNode) void {
        const ctx: *CECtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_cross_entropy_backward(
                rt.cuda_ctx,
                rt.kernels.fn_ce_bw,
                ga,
                self_node.grad.?,
                ctx.softmax_cache_dptr,
                ctx.idx_dptr,
                pa.shape[0],
                pa.shape[1],
            );
        }
    }

    pub fn bce_loss_with_logits(
        self: *DiffCudaRuntime,
        logits: DiffCudaTensor,
        target: []const f32,
    ) DiffCudaTensor {
        const total = logits.total_elements();
        const target_dptr = self.alloc_gpu_buf(total);
        self.cuda_ctx.copy_host_to_device(
            target_dptr,
            @ptrCast(target.ptr),
            total * @sizeOf(f32),
        ) catch unreachable;
        const out_dptr = self.alloc_gpu_buf_zeroed(1);
        ops.dispatch_loss_forward(
            self.cuda_ctx,
            self.kernels.fn_bce_fwd,
            out_dptr,
            logits.data,
            target_dptr,
            total,
        );
        const node = self.make_node(out_dptr, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.alloc_context(LossCtx);
            ctx.* = .{ .rt = self, .target_dptr = target_dptr, .is_mse = false };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_bce_loss;
        }
        return node;
    }

    fn backward_bce_loss(self_node: *DiffCudaNode) void {
        const ctx: *LossCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga| {
            ops.dispatch_loss_backward(
                rt.cuda_ctx,
                rt.kernels.fn_bce_bw,
                ga,
                self_node.grad.?,
                pa.data,
                ctx.target_dptr,
                pa.total_elements(),
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Tensor factories
    // ════════════════════════════════════════════════════════════════

    pub fn zeros(self: *DiffCudaRuntime, new_shape: []const usize) DiffCudaTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const dptr = self.alloc_gpu_buf_zeroed(size);
        return self.make_node(dptr, new_shape, false);
    }

    pub fn ones(self: *DiffCudaRuntime, new_shape: []const usize) DiffCudaTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const dptr = self.alloc_gpu_buf(size);
        ops.dispatch_fill(self.cuda_ctx, self.kernels.fn_fill, dptr, 1.0, size);
        return self.make_node(dptr, new_shape, false);
    }

    pub fn randn(self: *DiffCudaRuntime, new_shape: []const usize) DiffCudaTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        // Generate on CPU, then upload
        const buf = self.alloc_data(size);
        const rng = self.prng.random();
        var i: usize = 0;
        while (i + 1 < size) : (i += 2) {
            const r1 = rng.float(f32) *
                (1.0 - std.math.floatEps(f32)) + std.math.floatEps(f32);
            const r2 = rng.float(f32);
            const r = @sqrt(-2.0 * @log(r1));
            buf[i] = r * @cos(2.0 * std.math.pi * r2);
            buf[i + 1] = r * @sin(2.0 * std.math.pi * r2);
        }
        if (size % 2 == 1) {
            const r1 = rng.float(f32) *
                (1.0 - std.math.floatEps(f32)) + std.math.floatEps(f32);
            const r2 = rng.float(f32);
            buf[size - 1] = @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
        }
        const dptr = self.alloc_gpu_buf(size);
        self.cuda_ctx.copy_host_to_device(
            dptr,
            @ptrCast(buf.ptr),
            size * @sizeOf(f32),
        ) catch unreachable;
        return self.make_node(dptr, new_shape, false);
    }

    pub fn rand(self: *DiffCudaRuntime, new_shape: []const usize) DiffCudaTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const buf = self.alloc_data(size);
        const rng = self.prng.random();
        for (buf) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
        const dptr = self.alloc_gpu_buf(size);
        self.cuda_ctx.copy_host_to_device(
            dptr,
            @ptrCast(buf.ptr),
            size * @sizeOf(f32),
        ) catch unreachable;
        return self.make_node(dptr, new_shape, false);
    }

    // ════════════════════════════════════════════════════════════════
    // Backward (topological sort + reverse traversal)
    // ════════════════════════════════════════════════════════════════

    pub fn backward(self: *DiffCudaRuntime, loss: DiffCudaTensor) void {
        // 1. Set loss gradient to 1.0 (CUDA: GPU fill kernel)
        if (loss.grad == null) {
            loss.grad = self.alloc_gpu_buf(loss.total_elements());
        }
        ops.dispatch_fill(
            self.cuda_ctx,
            self.kernels.fn_fill,
            loss.grad.?,
            1.0,
            loss.total_elements(),
        );

        // 2. Topological sort (DFS)
        self.topo_buf.clearRetainingCapacity();
        diff_node.topo_sort(DiffCudaNode, loss, &self.topo_buf, self.allocator);

        // 3. Allocate grad buffers for intermediate nodes (CUDA: GPU zeroed alloc)
        for (self.topo_buf.items) |node| {
            if (node.grad == null and node.requires_grad) {
                node.grad = self.alloc_gpu_buf_zeroed(node.total_elements());
            }
        }

        // 4-5. Reverse traversal + reset visited
        diff_node.backward_pass(DiffCudaNode, &self.topo_buf, self.param_nodes);
    }

    // ════════════════════════════════════════════════════════════════
    // Adam optimizer (GPU)
    // ════════════════════════════════════════════════════════════════

    pub fn apply_adam(
        self: *DiffCudaRuntime,
        adam: *GpuAdamState,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) void {
        adam.step += 1;
        const bc1 = 1.0 - std.math.pow(f32, beta1, @floatFromInt(adam.step));
        const bc2 = 1.0 - std.math.pow(f32, beta2, @floatFromInt(adam.step));
        const count = self.module.param_count();
        for (0..count) |i| {
            const size = self.module.param_size(.{ .index = i });
            ops.dispatch_adam_step(
                self.cuda_ctx,
                self.kernels.fn_adam_step,
                self.param_nodes[i].data,
                self.param_grad_dptrs[i],
                adam.m_dptrs[i],
                adam.v_dptrs[i],
                lr,
                beta1,
                beta2,
                eps,
                wd,
                bc1,
                bc2,
                size,
            );
        }
    }

    pub fn apply_adam_clipped(
        self: *DiffCudaRuntime,
        adam: *GpuAdamState,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        max_grad_norm: f32,
    ) void {
        const count = self.module.param_count();

        // 1. Compute total gradient norm on GPU
        var total_norm_sq: f64 = 0;
        for (0..count) |i| {
            const size = self.module.param_size(.{ .index = i });
            const num_blocks = (size + ops.BLOCK_SIZE - 1) / ops.BLOCK_SIZE;
            const partial_dptr = self.alloc_gpu_buf(num_blocks);
            ops.dispatch_norm_sq(
                self.cuda_ctx,
                self.kernels.fn_norm_sq,
                partial_dptr,
                self.param_grad_dptrs[i],
                size,
            );
            // D2H partial sums
            const partial = self.alloc_data(num_blocks);
            self.cuda_ctx.copy_device_to_host(
                @ptrCast(partial.ptr),
                partial_dptr,
                num_blocks * @sizeOf(f32),
            ) catch unreachable;
            for (partial) |v| total_norm_sq += @as(f64, v);
        }

        // 2. Clip gradients
        const total_norm: f32 = @floatCast(@sqrt(total_norm_sq));
        if (total_norm > max_grad_norm) {
            const clip_coef = max_grad_norm / (total_norm + 1e-6);
            for (0..count) |i| {
                const size = self.module.param_size(.{ .index = i });
                ops.dispatch_scale_grad(
                    self.cuda_ctx,
                    self.kernels.fn_scale_grad,
                    self.param_grad_dptrs[i],
                    clip_coef,
                    size,
                );
            }
        }

        // 3. Apply Adam
        self.apply_adam(adam, lr, beta1, beta2, eps, wd);
    }

    // ════════════════════════════════════════════════════════════════
    // Parameter initialization
    // ════════════════════════════════════════════════════════════════

    pub fn init_params(self: *DiffCudaRuntime) void {
        // dropout 用 prng とは独立に、init_seed からローカル PRNG を作る。
        var rng_state = std.Random.DefaultPrng.init(self.init_seed);
        const rng = rng_state.random();

        for (self.module.params.items, 0..) |meta, i| {
            const size = self.module.param_size(.{ .index = i });
            const buf = self.allocator.alloc(f32, size) catch unreachable;
            defer self.allocator.free(buf);

            switch (meta.init_kind) {
                .ones => @memset(buf, 1.0),
                .zeros => @memset(buf, 0.0),
                .xavier => {
                    const fan_in: f32 = @floatFromInt(meta.shape[0]);
                    const scale = @sqrt(1.0 / fan_in);
                    for (buf) |*val| val.* = (rng.float(f32) * 2.0 - 1.0) * scale;
                },
                .kaiming => {
                    const fan_in: f32 = @floatFromInt(meta.shape[0]);
                    const scale = @sqrt(2.0 / fan_in);
                    for (buf) |*val| val.* = rng.floatNorm(f32) * scale;
                },
                .kaiming_fan => |fi| {
                    const fan_in: f32 = @floatFromInt(fi);
                    const scale = @sqrt(2.0 / fan_in);
                    for (buf) |*val| val.* = rng.floatNorm(f32) * scale;
                },
                .normal => |cfg| {
                    for (buf) |*val| val.* = rng.floatNorm(f32) * cfg.std_dev + cfg.mean;
                },
            }

            self.cuda_ctx.copy_host_to_device(
                self.param_nodes[i].data,
                @ptrCast(buf.ptr),
                size * @sizeOf(f32),
            ) catch unreachable;
        }
    }

    /// CpuRuntime / DiffCpuRuntime からパラメータをロード
    pub fn load_from_cpu(self: *DiffCudaRuntime, cpu: anytype) void {
        for (0..self.module.param_count()) |i| {
            const size = self.module.param_size(.{ .index = i });
            const cpu_data = cpu.param_data(i);
            self.cuda_ctx.copy_host_to_device(
                self.param_nodes[i].data,
                @ptrCast(cpu_data.ptr),
                size * @sizeOf(f32),
            ) catch unreachable;
        }
    }
};
