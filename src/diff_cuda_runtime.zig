/// diff_cuda_runtime.zig: 微分可能 CUDA ランタイム
///
/// DiffCpuRuntime と同じ duck-typed ops インターフェースを GPU (CUDA) 上で提供し、
/// forward 時に計算グラフを構築し backward() で GPU 上で自動微分する。
/// 統一モジュールの forward(ctx: anytype, ...) が DiffCudaRuntime を ctx として
/// 受け取れば、同じ forward コードで GPU training が可能。
const std = @import("std");
const Allocator = std.mem.Allocator;
const compute = @import("compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const cuda = @import("backend/cuda.zig");
const CudaContext = cuda.CudaContext;
const CUdeviceptr = cuda.CUdeviceptr;
const CUfunction = cuda.CUfunction;
const CUBLAS_OP_N = cuda.CUBLAS_OP_N;
const CUBLAS_OP_T = cuda.CUBLAS_OP_T;
const ops = @import("cuda_ops.zig");

pub const MAX_NDIM = ops.MAX_NDIM;

// ── Backward function type ──
const BackwardFn = *const fn (*DiffCudaNode) void;

pub const DiffCudaNode = struct {
    dptr: CUdeviceptr, // GPU 上の forward データ
    shape: [MAX_NDIM]usize,
    ndim: usize,
    grad_dptr: ?CUdeviceptr, // GPU 上の勾配バッファ
    backward_fn: ?BackwardFn,
    parents: [3]?*DiffCudaNode, // 最大3入力 (layerNorm: x, gamma, beta)
    context: ?*anyopaque, // backward 用キャッシュ (CPU メタデータ)
    requires_grad: bool,
    visited: bool,
    is_param: bool,
    param_index: ?usize,

    pub fn totalElements(self: *const DiffCudaNode) usize {
        var total: usize = 1;
        for (0..self.ndim) |i| total *= self.shape[i];
        return total;
    }

    pub fn lastDim(self: *const DiffCudaNode) usize {
        return self.shape[self.ndim - 1];
    }

    pub fn numRows(self: *const DiffCudaNode) usize {
        return self.totalElements() / self.lastDim();
    }
};

pub const DiffCudaTensor = *DiffCudaNode;

// ── GPU Adam state ──

pub const GpuAdamState = struct {
    m_dptrs: []CUdeviceptr, // 1次モーメント (GPU)
    v_dptrs: []CUdeviceptr, // 2次モーメント (GPU)
    step: u32,
    allocator: Allocator,
    cuda_ctx: *CudaContext,

    pub fn init(allocator: Allocator, cuda_ctx: *CudaContext, param_sizes: []const usize) !GpuAdamState {
        const count = param_sizes.len;
        const m_dptrs = try allocator.alloc(CUdeviceptr, count);
        const v_dptrs = try allocator.alloc(CUdeviceptr, count);
        for (param_sizes, 0..) |size, i| {
            m_dptrs[i] = try cuda_ctx.allocBuffer(size * @sizeOf(f32));
            try cuda_ctx.memsetZero(m_dptrs[i], size);
            v_dptrs[i] = try cuda_ctx.allocBuffer(size * @sizeOf(f32));
            try cuda_ctx.memsetZero(v_dptrs[i], size);
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
        for (self.m_dptrs) |d| self.cuda_ctx.freeBuffer(d);
        for (self.v_dptrs) |d| self.cuda_ctx.freeBuffer(d);
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
    prng: std.Random.DefaultPrng,
    training: bool,
    kernels: KernelHandles,

    pub fn init(module: *const Module, cuda_ctx: *CudaContext, allocator: Allocator) !DiffCudaRuntime {
        // Load PTX module
        const ptx_data = @embedFile("backend/cuda_kernels.ptx");
        try cuda_ctx.loadModule(@ptrCast(ptx_data.ptr));

        // Get all kernel function handles
        const k = KernelHandles{
            // Forward elementwise
            .fn_add = try cuda_ctx.getFunction("add_broadcast"),
            .fn_mul = try cuda_ctx.getFunction("mul_broadcast"),
            .fn_sub = try cuda_ctx.getFunction("sub_broadcast"),
            .fn_gelu = try cuda_ctx.getFunction("gelu_kernel"),
            .fn_silu_fwd_cache = try cuda_ctx.getFunction("silu_fwd_cache_kernel"),
            .fn_add_silu_fwd_cache = try cuda_ctx.getFunction("add_silu_fwd_cache_kernel"),
            .fn_relu = try cuda_ctx.getFunction("relu_kernel"),
            .fn_sigmoid = try cuda_ctx.getFunction("sigmoid_kernel"),
            .fn_tanh = try cuda_ctx.getFunction("tanh_kernel"),
            .fn_exp = try cuda_ctx.getFunction("exp_kernel"),
            .fn_log = try cuda_ctx.getFunction("log_kernel"),
            .fn_square = try cuda_ctx.getFunction("square_kernel"),
            .fn_sqrt = try cuda_ctx.getFunction("sqrt_kernel"),
            .fn_abs = try cuda_ctx.getFunction("abs_kernel"),
            .fn_clamp = try cuda_ctx.getFunction("clamp_kernel"),
            .fn_negative = try cuda_ctx.getFunction("negative_kernel"),
            .fn_scale = try cuda_ctx.getFunction("scale_kernel"),
            // Forward structured
            .fn_softmax_out = try cuda_ctx.getFunction("softmax_out_kernel"),
            .fn_layernorm_fwd = try cuda_ctx.getFunction("layernorm_fwd_kernel"),
            .fn_transpose = try cuda_ctx.getFunction("transpose_2d_kernel"),
            .fn_gather = try cuda_ctx.getFunction("gather_kernel"),
            // Backward elementwise
            .fn_gelu_bw = try cuda_ctx.getFunction("gelu_backward_kernel"),
            .fn_silu_bw = try cuda_ctx.getFunction("silu_backward_kernel"),
            .fn_relu_bw = try cuda_ctx.getFunction("relu_backward_kernel"),
            .fn_sigmoid_bw = try cuda_ctx.getFunction("sigmoid_backward_kernel"),
            .fn_tanh_bw = try cuda_ctx.getFunction("tanh_backward_kernel"),
            .fn_exp_bw = try cuda_ctx.getFunction("exp_backward_kernel"),
            .fn_log_bw = try cuda_ctx.getFunction("log_backward_kernel"),
            .fn_square_bw = try cuda_ctx.getFunction("square_backward_kernel"),
            .fn_sqrt_bw = try cuda_ctx.getFunction("sqrt_backward_kernel"),
            .fn_abs_bw = try cuda_ctx.getFunction("abs_backward_kernel"),
            .fn_clamp_bw = try cuda_ctx.getFunction("clamp_backward_kernel"),
            .fn_dropout_bw = try cuda_ctx.getFunction("dropout_backward_kernel"),
            .fn_add_silu_bw_same = try cuda_ctx.getFunction("add_silu_backward_same_kernel"),
            .fn_add_silu_bw_bcast = try cuda_ctx.getFunction("add_silu_backward_bcast_kernel"),
            // Backward structured
            .fn_mul_bw_same = try cuda_ctx.getFunction("mul_backward_same_kernel"),
            .fn_mul_bw_bcast_ga = try cuda_ctx.getFunction("mul_backward_broadcast_b_ga_kernel"),
            .fn_mul_bw_bcast_gb = try cuda_ctx.getFunction("mul_backward_broadcast_b_gb_kernel"),
            .fn_reduce_add_bcast = try cuda_ctx.getFunction("reduce_add_to_broadcast_kernel"),
            .fn_reduce_sub_bcast = try cuda_ctx.getFunction("reduce_sub_to_broadcast_kernel"),
            .fn_div_bw = try cuda_ctx.getFunction("div_backward_kernel"),
            .fn_softmax_bw = try cuda_ctx.getFunction("softmax_backward_kernel"),
            .fn_ln_bw_dx = try cuda_ctx.getFunction("layernorm_backward_dx_kernel"),
            .fn_ln_bw_dg_db = try cuda_ctx.getFunction("layernorm_backward_dgamma_dbeta_kernel"),
            .fn_scatter_add = try cuda_ctx.getFunction("scatter_add_kernel"),
            // Loss
            .fn_ce_fwd = try cuda_ctx.getFunction("cross_entropy_forward_kernel"),
            .fn_ce_bw = try cuda_ctx.getFunction("cross_entropy_backward_kernel"),
            .fn_mse_fwd = try cuda_ctx.getFunction("mse_forward_kernel"),
            .fn_mse_bw = try cuda_ctx.getFunction("mse_backward_kernel"),
            .fn_bce_fwd = try cuda_ctx.getFunction("bce_forward_kernel"),
            .fn_bce_bw = try cuda_ctx.getFunction("bce_backward_kernel"),
            // Utility
            .fn_fill = try cuda_ctx.getFunction("fill_kernel"),
            .fn_accum_grad = try cuda_ctx.getFunction("accum_grad_kernel"),
            .fn_dropout = try cuda_ctx.getFunction("dropout_kernel"),
            // Reduction
            .fn_reduce_sum_rows = try cuda_ctx.getFunction("reduction_sum_rows_kernel"),
            .fn_reduce_sum_cols = try cuda_ctx.getFunction("reduction_sum_cols_kernel"),
            .fn_reduce_sum_1d = try cuda_ctx.getFunction("reduction_sum_1d_kernel"),
            // Optimizer
            .fn_adam_step = try cuda_ctx.getFunction("adam_step_kernel"),
            .fn_norm_sq = try cuda_ctx.getFunction("norm_sq_kernel"),
            .fn_scale_grad = try cuda_ctx.getFunction("scale_grad_kernel"),
        };

        // Allocate parameter nodes and GPU buffers
        const count = module.paramCount();
        const param_nodes = try allocator.alloc(DiffCudaNode, count);
        const param_grad_dptrs = try allocator.alloc(CUdeviceptr, count);

        // Compute total gradient size and allocate one contiguous buffer
        var total_grad_floats: usize = 0;
        for (0..count) |i| total_grad_floats += module.paramSize(.{ .index = i });
        const grad_base_dptr = try cuda_ctx.allocBuffer(total_grad_floats * @sizeOf(f32));
        try cuda_ctx.memsetZero(grad_base_dptr, total_grad_floats);

        var grad_offset: usize = 0;
        for (module.params.items, 0..) |meta, i| {
            const size = module.paramSize(.{ .index = i });
            const dptr = try cuda_ctx.allocBuffer(size * @sizeOf(f32));
            const grad_dptr = grad_base_dptr + grad_offset * @sizeOf(f32);

            param_nodes[i] = .{
                .dptr = dptr,
                .shape = ops.initShapeArray(meta.shape),
                .ndim = meta.shape.len,
                .grad_dptr = grad_dptr,
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
        }

        return .{
            .allocator = allocator,
            .cuda_ctx = cuda_ctx,
            .module = module,
            .param_nodes = param_nodes,
            .param_grad_dptrs = param_grad_dptrs,
            .grad_base_dptr = grad_base_dptr,
            .total_grad_floats = total_grad_floats,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .arena_gpu_bufs = .{},
            .arena_gpu_sizes = .{},
            .gpu_pool = cuda.GpuMemPool.init(allocator),
            .topo_buf = .empty,
            .prng = std.Random.DefaultPrng.init(42),
            .training = true,
            .kernels = k,
        };
    }

    pub fn deinit(self: *DiffCudaRuntime) void {
        self.freeArenaGpuBuffers();
        self.arena_gpu_bufs.deinit(self.allocator);
        self.arena_gpu_sizes.deinit(self.allocator);
        self.gpu_pool.deinit();
        self.arena.deinit();
        for (self.param_nodes) |node| {
            self.cuda_ctx.freeBuffer(node.dptr);
        }
        self.cuda_ctx.freeBuffer(self.grad_base_dptr);
        self.allocator.free(self.param_nodes);
        self.allocator.free(self.param_grad_dptrs);
        self.topo_buf.deinit(self.allocator);
    }

    fn freeArenaGpuBuffers(self: *DiffCudaRuntime) void {
        for (self.arena_gpu_bufs.items, self.arena_gpu_sizes.items) |dptr, size_bytes| {
            self.gpu_pool.release(dptr, size_bytes);
        }
        self.arena_gpu_bufs.clearRetainingCapacity();
        self.arena_gpu_sizes.clearRetainingCapacity();
    }

    pub fn resetArena(self: *DiffCudaRuntime) void {
        self.freeArenaGpuBuffers();
        _ = self.arena.reset(.retain_capacity);
        for (self.param_nodes) |*node| {
            node.visited = false;
        }
    }

    pub fn zeroGrad(self: *DiffCudaRuntime) void {
        // Single async memset for the entire contiguous gradient buffer
        self.cuda_ctx.memsetZeroAsync(self.grad_base_dptr, self.total_grad_floats) catch unreachable;
        for (self.param_nodes, 0..) |*node, i| {
            node.grad_dptr = self.param_grad_dptrs[i];
        }
    }

    fn arenaAlloc(self: *DiffCudaRuntime) Allocator {
        return self.arena.allocator();
    }

    /// GPU メモリ確保 (arena tracked, pool 優先)
    fn allocGpuBuf(self: *DiffCudaRuntime, num_floats: usize) CUdeviceptr {
        const size_bytes = num_floats * @sizeOf(f32);
        const dptr = self.gpu_pool.acquire(size_bytes) orelse
            (self.cuda_ctx.allocBuffer(cuda.GpuMemPool.bucketSize(cuda.GpuMemPool.bucketIndex(size_bytes))) catch unreachable);
        self.arena_gpu_bufs.append(self.allocator, dptr) catch unreachable;
        self.arena_gpu_sizes.append(self.allocator, size_bytes) catch unreachable;
        return dptr;
    }

    /// GPU メモリ確保 + ゼロ初期化
    fn allocGpuBufZeroed(self: *DiffCudaRuntime, num_floats: usize) CUdeviceptr {
        const dptr = self.allocGpuBuf(num_floats);
        self.cuda_ctx.memsetZero(dptr, num_floats) catch unreachable;
        return dptr;
    }

    fn allocContext(self: *DiffCudaRuntime, comptime T: type) *T {
        return self.arenaAlloc().create(T) catch unreachable;
    }

    // ── Node creation ──

    pub fn makeNode(self: *DiffCudaRuntime, dptr: CUdeviceptr, shape_slice: []const usize, requires_grad: bool) *DiffCudaNode {
        const node = self.arenaAlloc().create(DiffCudaNode) catch unreachable;
        node.* = .{
            .dptr = dptr,
            .shape = ops.initShapeArray(shape_slice),
            .ndim = shape_slice.len,
            .grad_dptr = null,
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

    pub fn makeTensor(self: *DiffCudaRuntime, data: []f32, shape: []const usize) DiffCudaTensor {
        var total: usize = 1;
        for (shape) |s| total *= s;
        const dptr = self.allocGpuBuf(total);
        self.cuda_ctx.copyHostToDevice(dptr, @ptrCast(data.ptr), total * @sizeOf(f32)) catch unreachable;
        return self.makeNode(dptr, shape, false);
    }

    /// allocData returns CPU arena memory (for host-side metadata/indices)
    pub fn allocData(self: *DiffCudaRuntime, size: usize) []f32 {
        return self.arenaAlloc().alloc(f32, size) catch unreachable;
    }

    // ── Param access ──

    pub fn param(self: *DiffCudaRuntime, handle: ParamHandle) DiffCudaTensor {
        return &self.param_nodes[handle.index];
    }

    /// D2H copy: パラメータデータを CPU スライスとして返す (デバッグ用)
    pub fn paramData(self: *DiffCudaRuntime, index: usize) []f32 {
        const size = self.module.paramSize(.{ .index = index });
        const buf = self.allocData(size);
        self.cuda_ctx.copyDeviceToHost(@ptrCast(buf.ptr), self.param_nodes[index].dptr, size * @sizeOf(f32)) catch unreachable;
        return buf;
    }

    /// D2H copy: パラメータ勾配を CPU スライスとして返す (デバッグ用)
    pub fn paramGrad(self: *DiffCudaRuntime, index: usize) []f32 {
        const size = self.module.paramSize(.{ .index = index });
        const buf = self.allocData(size);
        self.cuda_ctx.copyDeviceToHost(@ptrCast(buf.ptr), self.param_grad_dptrs[index], size * @sizeOf(f32)) catch unreachable;
        return buf;
    }

    /// GPU スカラー値を CPU にコピー
    pub fn copyScalarToHost(self: *DiffCudaRuntime, t: DiffCudaTensor) f32 {
        var val: f32 = 0;
        self.cuda_ctx.copyDeviceToHost(@ptrCast(&val), t.dptr, @sizeOf(f32)) catch unreachable;
        return val;
    }

    /// GPU テンソルを CPU バッファにコピー
    pub fn copyToHost(self: *DiffCudaRuntime, t: DiffCudaTensor, dst: []f32) void {
        const total = t.totalElements();
        self.cuda_ctx.copyDeviceToHost(@ptrCast(dst.ptr), t.dptr, total * @sizeOf(f32)) catch unreachable;
    }

    // ── Leaf ops ──

    pub fn constantScalar(self: *DiffCudaRuntime, val: f64, dtype: u32) DiffCudaTensor {
        _ = dtype;
        const dptr = self.allocGpuBuf(1);
        var v: f32 = @floatCast(val);
        self.cuda_ctx.copyHostToDevice(dptr, @ptrCast(&v), @sizeOf(f32)) catch unreachable;
        return self.makeNode(dptr, &.{1}, false);
    }

    pub fn constantData(self: *DiffCudaRuntime, data: [*]const u8, len: usize, new_shape: []const usize, dtype: u32) DiffCudaTensor {
        _ = dtype;
        const n_floats = len / @sizeOf(f32);
        const dptr = self.allocGpuBuf(n_floats);
        self.cuda_ctx.copyHostToDevice(dptr, @ptrCast(data), n_floats * @sizeOf(f32)) catch unreachable;
        return self.makeNode(dptr, new_shape, false);
    }

    pub fn stopGradient(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        return self.makeNode(x.dptr, x.shape[0..x.ndim], false);
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
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_negative, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardNegative;
        }
        return node;
    }

    const RtContext = struct {
        rt: *DiffCudaRuntime,
    };

    fn backwardNegative(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            const total = self_node.totalElements();
            // ga -= go → accum_scaled(ga, go, -1.0)
            const go = self_node.grad_dptr.?;
            // Use scale kernel: tmp = -go, then accum
            const tmp = rt.allocGpuBuf(total);
            ops.dispatchElementwise(rt.cuda_ctx, rt.kernels.fn_negative, tmp, go, total);
            ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, total);
        }
    }

    pub fn gelu(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_gelu, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.dptr }; // cache input x
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardGelu;
        }
        return node;
    }

    const UnaryBwContext = struct {
        rt: *DiffCudaRuntime,
        cache_dptr: CUdeviceptr,
    };

    fn backwardGelu(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_gelu_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn silu(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        const sig_cache = self.allocGpuBuf(total);
        ops.dispatchSiluFwdCache(self.cuda_ctx, self.kernels.fn_silu_fwd_cache, out_dptr, sig_cache, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(SiluContext);
            ctx.* = .{ .rt = self, .sig_cache_dptr = sig_cache };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardSilu;
        }
        return node;
    }

    const SiluContext = struct {
        rt: *DiffCudaRuntime,
        sig_cache_dptr: CUdeviceptr,
    };

    fn backwardSilu(self_node: *DiffCudaNode) void {
        const ctx: *SiluContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchSiluBackward(rt.cuda_ctx, rt.kernels.fn_silu_bw, ga, self_node.grad_dptr.?, pa.dptr, ctx.sig_cache_dptr, self_node.totalElements());
        }
    }

    // ── Fused Add + SiLU ──

    pub fn addSilu(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;
        const out_dptr = self.allocGpuBuf(a_total);
        const sig_cache = self.allocGpuBuf(a_total);
        ops.dispatchAddSiluFwdCache(self.cuda_ctx, self.kernels.fn_add_silu_fwd_cache, out_dptr, sig_cache, a.dptr, b.dptr, a_total, b_total);
        const node = self.makeNode(out_dptr, a.shape[0..a.ndim], rg);
        if (rg) {
            node.parents[0] = a;
            node.parents[1] = b;
            const ctx = self.allocContext(AddSiluContext);
            ctx.* = .{ .rt = self, .sig_cache_dptr = sig_cache };
            node.context = @ptrCast(ctx);
            if (a_total == b_total) {
                node.backward_fn = &backwardAddSiluSame;
            } else {
                node.backward_fn = &backwardAddSiluBcast;
            }
        }
        return node;
    }

    const AddSiluContext = struct {
        rt: *DiffCudaRuntime,
        sig_cache_dptr: CUdeviceptr,
    };

    fn backwardAddSiluSame(self_node: *DiffCudaNode) void {
        const ctx: *AddSiluContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const n = self_node.totalElements();
        const ga = pa.grad_dptr orelse rt.allocGpuBufZeroed(n);
        const gb = pb.grad_dptr orelse rt.allocGpuBufZeroed(n);
        ops.dispatchAddSiluBackwardSame(rt.cuda_ctx, rt.kernels.fn_add_silu_bw_same, ga, gb, go, ctx.sig_cache_dptr, pa.dptr, pb.dptr, n);
    }

    fn backwardAddSiluBcast(self_node: *DiffCudaNode) void {
        const ctx: *AddSiluContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        const ga = pa.grad_dptr orelse rt.allocGpuBufZeroed(a_total);
        const gb = pb.grad_dptr orelse rt.allocGpuBufZeroed(b_total);
        ops.dispatchAddSiluBackwardBcast(rt.cuda_ctx, rt.kernels.fn_add_silu_bw_bcast, ga, gb, go, ctx.sig_cache_dptr, pa.dptr, pb.dptr, a_total, b_total);
    }

    pub fn relu(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_relu, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardRelu;
        }
        return node;
    }

    fn backwardRelu(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_relu_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn sigmoid(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_sigmoid, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = out_dptr }; // cache output
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardSigmoid;
        }
        return node;
    }

    fn backwardSigmoid(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_sigmoid_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn tanh_(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_tanh, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = out_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardTanh;
        }
        return node;
    }

    fn backwardTanh(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_tanh_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn exp(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_exp, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = out_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardExp;
        }
        return node;
    }

    fn backwardExp(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_exp_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn log(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_log, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardLog;
        }
        return node;
    }

    fn backwardLog(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_log_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn square(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_square, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardSquare;
        }
        return node;
    }

    fn backwardSquare(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_square_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn abs(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_abs, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = x.dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardAbs;
        }
        return node;
    }

    fn backwardAbs(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_abs_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn sqrt(self: *DiffCudaRuntime, x: DiffCudaTensor) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchElementwise(self.cuda_ctx, self.kernels.fn_sqrt, out_dptr, x.dptr, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(UnaryBwContext);
            ctx.* = .{ .rt = self, .cache_dptr = out_dptr }; // cache output
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardSqrt;
        }
        return node;
    }

    fn backwardSqrt(self_node: *DiffCudaNode) void {
        const ctx: *UnaryBwContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchBackwardElementwise3(rt.cuda_ctx, rt.kernels.fn_sqrt_bw, ga, self_node.grad_dptr.?, ctx.cache_dptr, self_node.totalElements());
        }
    }

    pub fn clamp(self: *DiffCudaRuntime, x: DiffCudaTensor, min_val: f32, max_val: f32) DiffCudaTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        // clamp_kernel(out, x, min, max, n)
        var n_i: c_int = @intCast(total);
        var mn: f32 = min_val;
        var mx: f32 = max_val;
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&x.dptr)),
            @ptrCast(&mn),
            @ptrCast(&mx),
            @ptrCast(&n_i),
        };
        self.cuda_ctx.launchKernel(
            self.kernels.fn_clamp,
            .{ ops.gridFor(total), 1, 1 },
            .{ ops.BLOCK_SIZE, 1, 1 },
            0,
            &params,
        ) catch unreachable;
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(ClampContext);
            ctx.* = .{ .rt = self, .min_val = min_val, .max_val = max_val };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardClamp;
        }
        return node;
    }

    const ClampContext = struct {
        rt: *DiffCudaRuntime,
        min_val: f32,
        max_val: f32,
    };

    fn backwardClamp(self_node: *DiffCudaNode) void {
        const ctx: *ClampContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchClampBackward(rt.cuda_ctx, rt.kernels.fn_clamp_bw, ga, self_node.grad_dptr.?, pa.dptr, ctx.min_val, ctx.max_val, self_node.totalElements());
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Binary ops
    // ════════════════════════════════════════════════════════════════

    pub fn add(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out_dptr = self.allocGpuBuf(a_total);
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.kernels.fn_add, out_dptr, a.dptr, b.dptr, a_total, b_total);
            const node = self.makeNode(out_dptr, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardAddSame;
            }
            return node;
        }

        if (b_total < a_total and a_total % b_total == 0) {
            const out_dptr = self.allocGpuBuf(a_total);
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.kernels.fn_add, out_dptr, a.dptr, b.dptr, a_total, b_total);
            const node = self.makeNode(out_dptr, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardAddBroadcastB;
            }
            return node;
        }

        if (a_total < b_total and b_total % a_total == 0) {
            const out_dptr = self.allocGpuBuf(b_total);
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.kernels.fn_add, out_dptr, b.dptr, a.dptr, b_total, a_total);
            const node = self.makeNode(out_dptr, b.shape[0..b.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardAddBroadcastA;
            }
            return node;
        }

        @panic("add: incompatible shapes for broadcast");
    }

    fn backwardAddSame(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const total = self_node.totalElements();
        if (pa.grad_dptr) |ga| ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, total);
        if (pb.grad_dptr) |gb| ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, gb, go, total);
    }

    fn backwardAddBroadcastB(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad_dptr) |ga| ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, a_total);
        if (pb.grad_dptr) |gb| ops.dispatchReduceToBroadcast(rt.cuda_ctx, rt.kernels.fn_reduce_add_bcast, gb, go, a_total, b_total);
    }

    fn backwardAddBroadcastA(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad_dptr) |ga| ops.dispatchReduceToBroadcast(rt.cuda_ctx, rt.kernels.fn_reduce_add_bcast, ga, go, b_total, a_total);
        if (pb.grad_dptr) |gb| ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, gb, go, b_total);
    }

    pub fn sub(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out_dptr = self.allocGpuBuf(a_total);
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.kernels.fn_sub, out_dptr, a.dptr, b.dptr, a_total, b_total);
            const node = self.makeNode(out_dptr, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardSubSame;
            }
            return node;
        }

        if (b_total < a_total and a_total % b_total == 0) {
            const out_dptr = self.allocGpuBuf(a_total);
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.kernels.fn_sub, out_dptr, a.dptr, b.dptr, a_total, b_total);
            const node = self.makeNode(out_dptr, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardSubBroadcastB;
            }
            return node;
        }

        @panic("sub: incompatible shapes for broadcast");
    }

    fn backwardSubSame(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const total = self_node.totalElements();
        if (pa.grad_dptr) |ga| ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, total);
        if (pb.grad_dptr) |gb| {
            // gb -= go
            const tmp = rt.allocGpuBuf(total);
            ops.dispatchElementwise(rt.cuda_ctx, rt.kernels.fn_negative, tmp, go, total);
            ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, gb, tmp, total);
        }
    }

    fn backwardSubBroadcastB(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad_dptr) |ga| ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, a_total);
        if (pb.grad_dptr) |gb| ops.dispatchReduceToBroadcast(rt.cuda_ctx, rt.kernels.fn_reduce_sub_bcast, gb, go, a_total, b_total);
    }

    pub fn mul(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out_dptr = self.allocGpuBuf(a_total);
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.kernels.fn_mul, out_dptr, a.dptr, b.dptr, a_total, b_total);
            const node = self.makeNode(out_dptr, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardMulSame;
            }
            return node;
        }

        if (b_total <= a_total and a_total % b_total == 0) {
            const out_dptr = self.allocGpuBuf(a_total);
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.kernels.fn_mul, out_dptr, a.dptr, b.dptr, a_total, b_total);
            const node = self.makeNode(out_dptr, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardMulBroadcastB;
            }
            return node;
        }

        if (a_total < b_total and b_total % a_total == 0) {
            const out_dptr = self.allocGpuBuf(b_total);
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.kernels.fn_mul, out_dptr, b.dptr, a.dptr, b_total, a_total);
            const node = self.makeNode(out_dptr, b.shape[0..b.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardMulBroadcastA;
            }
            return node;
        }

        @panic("mul: incompatible shapes for broadcast");
    }

    fn backwardMulSame(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const n = self_node.totalElements();
        // Use a zero buffer for the grad we don't need, or dispatch individually
        if (pa.grad_dptr != null and pb.grad_dptr != null) {
            ops.dispatchMulBackwardSame(rt.cuda_ctx, rt.kernels.fn_mul_bw_same, pa.grad_dptr.?, pb.grad_dptr.?, go, pa.dptr, pb.dptr, n);
        } else if (pa.grad_dptr) |ga| {
            // ga += go * b
            const tmp = rt.allocGpuBuf(n);
            ops.dispatchBroadcastBinop(rt.cuda_ctx, rt.kernels.fn_mul, tmp, go, pb.dptr, n, n);
            ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, n);
        } else if (pb.grad_dptr) |gb| {
            const tmp = rt.allocGpuBuf(n);
            ops.dispatchBroadcastBinop(rt.cuda_ctx, rt.kernels.fn_mul, tmp, go, pa.dptr, n, n);
            ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, gb, tmp, n);
        }
    }

    fn backwardMulBroadcastB(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad_dptr) |ga| ops.dispatchMulBackwardBroadcast(rt.cuda_ctx, rt.kernels.fn_mul_bw_bcast_ga, ga, go, pb.dptr, a_total, b_total);
        if (pb.grad_dptr) |gb| ops.dispatchMulBackwardBroadcast(rt.cuda_ctx, rt.kernels.fn_mul_bw_bcast_gb, gb, go, pa.dptr, a_total, b_total);
    }

    fn backwardMulBroadcastA(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        // a is smaller, b is larger. Swap roles.
        if (pa.grad_dptr) |ga| ops.dispatchMulBackwardBroadcast(rt.cuda_ctx, rt.kernels.fn_mul_bw_bcast_gb, ga, go, pb.dptr, b_total, a_total);
        if (pb.grad_dptr) |gb| ops.dispatchMulBackwardBroadcast(rt.cuda_ctx, rt.kernels.fn_mul_bw_bcast_ga, gb, go, pa.dptr, b_total, a_total);
    }

    pub fn div(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const total = a.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        // div = a * (1/b), but we can compute directly on CPU-side or use a kernel
        // For now, implement as: out[i] = a[i] / b[i] using a two-step approach
        // Actually, there's no div forward kernel. Use scale approach: compute b_inv, then mul.
        // Simpler: just do it as a/b via the broadcast kernels.
        // div_backward_kernel exists. For forward, just compute via CPU upload or use mul(a, 1/b).
        // Actually the simplest approach: download, compute, upload. But that's slow.
        // Better: just compute a * (1/b) using existing kernels. But we don't have a reciprocal kernel.
        // For now, fall back to host-side computation for div forward.
        const a_data = self.allocData(total);
        const b_data = self.allocData(total);
        self.cuda_ctx.copyDeviceToHost(@ptrCast(a_data.ptr), a.dptr, total * @sizeOf(f32)) catch unreachable;
        self.cuda_ctx.copyDeviceToHost(@ptrCast(b_data.ptr), b.dptr, total * @sizeOf(f32)) catch unreachable;
        const out_data = self.allocData(total);
        for (0..total) |i| out_data[i] = a_data[i] / b_data[i];
        self.cuda_ctx.copyHostToDevice(out_dptr, @ptrCast(out_data.ptr), total * @sizeOf(f32)) catch unreachable;

        const rg = a.requires_grad or b.requires_grad;
        const node = self.makeNode(out_dptr, a.shape[0..a.ndim], rg);
        if (rg) {
            node.parents[0] = a;
            node.parents[1] = b;
            const ctx = self.allocContext(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardDiv;
        }
        return node;
    }

    fn backwardDiv(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const n = self_node.totalElements();
        const ga = pa.grad_dptr orelse @as(CUdeviceptr, 0);
        const gb = pb.grad_dptr orelse @as(CUdeviceptr, 0);
        if (pa.grad_dptr != null or pb.grad_dptr != null) {
            // Use temp buffers for grads we don't track
            const real_ga = if (pa.grad_dptr != null) ga else rt.allocGpuBufZeroed(n);
            const real_gb = if (pb.grad_dptr != null) gb else rt.allocGpuBufZeroed(n);
            ops.dispatchDivBackward(rt.cuda_ctx, rt.kernels.fn_div_bw, real_ga, real_gb, self_node.grad_dptr.?, pa.dptr, pb.dptr, n);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Matmul
    // ════════════════════════════════════════════════════════════════

    pub fn matmul(self: *DiffCudaRuntime, a: DiffCudaTensor, b: DiffCudaTensor) DiffCudaTensor {
        const rg = a.requires_grad or b.requires_grad;

        if (a.ndim == 2 and b.ndim == 2) {
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[1];
            const out_dptr = self.allocGpuBuf(M * N);
            self.cuda_ctx.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, @intCast(N), @intCast(M), @intCast(K), 1.0, b.dptr, @intCast(N), a.dptr, @intCast(K), 0.0, out_dptr, @intCast(N)) catch unreachable;
            const node = self.makeNode(out_dptr, &.{ M, N }, rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardMatmul2D;
            }
            return node;
        }

        if (a.ndim == 3 and b.ndim == 3) {
            const B = a.shape[0];
            const M = a.shape[1];
            const K = a.shape[2];
            const N = b.shape[2];
            const out_dptr = self.allocGpuBuf(B * M * N);
            self.cuda_ctx.sgemmStridedBatched(CUBLAS_OP_N, CUBLAS_OP_N, @intCast(N), @intCast(M), @intCast(K), 1.0, b.dptr, @intCast(N), @intCast(K * N), a.dptr, @intCast(K), @intCast(M * K), 0.0, out_dptr, @intCast(N), @intCast(M * N), @intCast(B)) catch unreachable;
            const node = self.makeNode(out_dptr, &.{ B, M, N }, rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardMatmul3D;
            }
            return node;
        }

        if (a.ndim == 2 and b.ndim == 3) {
            const B = b.shape[0];
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[2];
            const out_dptr = self.allocGpuBuf(B * M * N);
            for (0..B) |batch| {
                const b_off = batch * K * N;
                const o_off = batch * M * N;
                self.cuda_ctx.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, @intCast(N), @intCast(M), @intCast(K), 1.0, b.dptr + b_off * @sizeOf(f32), @intCast(N), a.dptr, @intCast(K), 0.0, out_dptr + o_off * @sizeOf(f32), @intCast(N)) catch unreachable;
            }
            const node = self.makeNode(out_dptr, &.{ B, M, N }, rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardMatmul2D3D;
            }
            return node;
        }

        @panic("matmul: unsupported shape combination (expected 2D or 3D)");
    }

    fn backwardMatmul2D(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const M = pa.shape[0];
        const K = pa.shape[1];
        const N = pb.shape[1];
        // dA += go @ B^T: sgemm(N=K, M=M, K=N, B^T, go)
        if (pa.grad_dptr) |ga| rt.cuda_ctx.sgemmAccum(CUBLAS_OP_T, CUBLAS_OP_N, @intCast(K), @intCast(M), @intCast(N), 1.0, pb.dptr, @intCast(N), go, @intCast(N), ga, @intCast(K)) catch unreachable;
        // dB += A^T @ go: sgemm(N=N, M=K, K=M, go, A^T)
        if (pb.grad_dptr) |gb| rt.cuda_ctx.sgemmAccum(CUBLAS_OP_N, CUBLAS_OP_T, @intCast(N), @intCast(K), @intCast(M), 1.0, go, @intCast(N), pa.dptr, @intCast(K), gb, @intCast(N)) catch unreachable;
    }

    fn backwardMatmul3D(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const B = pa.shape[0];
        const M = pa.shape[1];
        const K = pa.shape[2];
        const N = pb.shape[2];
        // dA += go @ B^T: batched sgemm
        if (pa.grad_dptr) |ga| rt.cuda_ctx.sgemmStridedBatched(
            CUBLAS_OP_T, CUBLAS_OP_N,
            @intCast(K), @intCast(M), @intCast(N), 1.0,
            pb.dptr, @intCast(N), @intCast(K * N),
            go, @intCast(N), @intCast(M * N),
            1.0, ga, @intCast(K), @intCast(M * K),
            @intCast(B),
        ) catch unreachable;
        // dB += A^T @ go: batched sgemm
        if (pb.grad_dptr) |gb| rt.cuda_ctx.sgemmStridedBatched(
            CUBLAS_OP_N, CUBLAS_OP_T,
            @intCast(N), @intCast(K), @intCast(M), 1.0,
            go, @intCast(N), @intCast(M * N),
            pa.dptr, @intCast(K), @intCast(M * K),
            1.0, gb, @intCast(N), @intCast(K * N),
            @intCast(B),
        ) catch unreachable;
    }

    fn backwardMatmul2D3D(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad_dptr.?;
        const B = pb.shape[0];
        const M = pa.shape[0];
        const K = pa.shape[1];
        const N = pb.shape[2];
        // dA (2D) += sum_b(go_b @ B_b^T): compute batched into temp 3D, then reduce
        if (pa.grad_dptr) |ga| {
            // Use batched sgemm into a temp [B, M, K] buffer, then reduce-sum over B
            const tmp = rt.allocGpuBufZeroed(B * M * K);
            rt.cuda_ctx.sgemmStridedBatched(
                CUBLAS_OP_T, CUBLAS_OP_N,
                @intCast(K), @intCast(M), @intCast(N), 1.0,
                pb.dptr, @intCast(N), @intCast(K * N),
                go, @intCast(N), @intCast(M * N),
                0.0, tmp, @intCast(K), @intCast(M * K),
                @intCast(B),
            ) catch unreachable;
            // Reduce: ga += sum over batch dimension
            for (0..B) |batch| {
                ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp + batch * M * K * @sizeOf(f32), M * K);
            }
        }
        // dB (3D) += A^T @ go_b: batched sgemm (A is shared across batches, stride_a=0)
        if (pb.grad_dptr) |gb| rt.cuda_ctx.sgemmStridedBatched(
            CUBLAS_OP_N, CUBLAS_OP_T,
            @intCast(N), @intCast(K), @intCast(M), 1.0,
            go, @intCast(N), @intCast(M * N),
            pa.dptr, @intCast(K), 0, // stride_a=0: same A for all batches
            1.0, gb, @intCast(N), @intCast(K * N),
            @intCast(B),
        ) catch unreachable;
    }

    // ════════════════════════════════════════════════════════════════
    // Shape ops
    // ════════════════════════════════════════════════════════════════

    pub fn reshape(self: *DiffCudaRuntime, x: DiffCudaTensor, new_shape: []const usize) DiffCudaTensor {
        const node = self.makeNode(x.dptr, new_shape, x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardReshape;
        }
        return node;
    }

    fn backwardReshape(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const go = self_node.grad_dptr.?;
        if (pa.grad_dptr) |ga| {
            if (ga == go) return; // same buffer, no-op
            ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, go, self_node.totalElements());
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
            const out_dptr = self.allocGpuBuf(total);
            for (0..B) |b| {
                const in_off = b * R * C;
                const out_off = b * C * R;
                ops.dispatchTranspose2d(self.cuda_ctx, self.kernels.fn_transpose, out_dptr + out_off * @sizeOf(f32), x.dptr + in_off * @sizeOf(f32), R, C);
            }
            const node = self.makeNode(out_dptr, &.{ B, C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardTranspose3D;
            }
            return node;
        }

        if (x.ndim == 2 and dim1 == 0 and dim2 == 1) {
            const R = x.shape[0];
            const C = x.shape[1];
            const out_dptr = self.allocGpuBuf(R * C);
            ops.dispatchTranspose2d(self.cuda_ctx, self.kernels.fn_transpose, out_dptr, x.dptr, R, C);
            const node = self.makeNode(out_dptr, &.{ C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardTranspose2D;
            }
            return node;
        }

        @panic("transpose: unsupported ndim (expected 2D or 3D)");
    }

    fn backwardTranspose3D(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const go = self_node.grad_dptr.?;
        if (pa.grad_dptr) |ga| {
            const B = pa.shape[0];
            const R = pa.shape[1];
            const C = pa.shape[2];
            // go is [B, C, R], transpose back to [B, R, C]
            const tmp = rt.allocGpuBuf(B * R * C);
            for (0..B) |b| {
                ops.dispatchTranspose2d(rt.cuda_ctx, rt.kernels.fn_transpose, tmp + b * R * C * @sizeOf(f32), go + b * C * R * @sizeOf(f32), C, R);
            }
            ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, B * R * C);
        }
    }

    fn backwardTranspose2D(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        const go = self_node.grad_dptr.?;
        if (pa.grad_dptr) |ga| {
            const R = pa.shape[0];
            const C = pa.shape[1];
            const tmp = rt.allocGpuBuf(R * C);
            ops.dispatchTranspose2d(rt.cuda_ctx, rt.kernels.fn_transpose, tmp, go, C, R);
            ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, tmp, R * C);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Softmax
    // ════════════════════════════════════════════════════════════════

    pub fn softmax(self: *DiffCudaRuntime, x: DiffCudaTensor, axis: i64) DiffCudaTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchSoftmaxOut(self.cuda_ctx, self.kernels.fn_softmax_out, out_dptr, x.dptr, rows, cols);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardSoftmax;
        }
        return node;
    }

    fn backwardSoftmax(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            const total = self_node.totalElements();
            const cols = self_node.lastDim();
            const rows = total / cols;
            ops.dispatchSoftmaxBackward(rt.cuda_ctx, rt.kernels.fn_softmax_bw, ga, self_node.grad_dptr.?, self_node.dptr, rows, cols);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Reduction
    // ════════════════════════════════════════════════════════════════

    pub fn reductionSum(self: *DiffCudaRuntime, x: DiffCudaTensor, axis: i64) DiffCudaTensor {
        const actual_axis: usize = if (axis < 0) @intCast(@as(i64, @intCast(x.ndim)) + axis) else @intCast(axis);

        if (x.ndim == 2) {
            const rows = x.shape[0];
            const cols = x.shape[1];
            if (actual_axis == 1) {
                const out_dptr = self.allocGpuBuf(rows);
                ops.dispatchReductionSumRows(self.cuda_ctx, self.kernels.fn_reduce_sum_rows, out_dptr, x.dptr, rows, cols);
                const node = self.makeNode(out_dptr, &.{ rows, 1 }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    const ctx = self.allocContext(RtContext);
                    ctx.* = .{ .rt = self };
                    node.context = @ptrCast(ctx);
                    node.backward_fn = &backwardReductionSumAxis1;
                }
                return node;
            } else {
                const out_dptr = self.allocGpuBuf(cols);
                ops.dispatchReductionSumCols(self.cuda_ctx, self.kernels.fn_reduce_sum_cols, out_dptr, x.dptr, rows, cols);
                const node = self.makeNode(out_dptr, &.{ 1, cols }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    const ctx = self.allocContext(RtContext);
                    ctx.* = .{ .rt = self };
                    node.context = @ptrCast(ctx);
                    node.backward_fn = &backwardReductionSumAxis0;
                }
                return node;
            }
        }

        if (x.ndim == 1) {
            const out_dptr = self.allocGpuBuf(1);
            ops.dispatchReductionSum1d(self.cuda_ctx, self.kernels.fn_reduce_sum_1d, out_dptr, x.dptr, x.totalElements());
            const node = self.makeNode(out_dptr, &.{1}, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                const ctx = self.allocContext(RtContext);
                ctx.* = .{ .rt = self };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardReductionSum1D;
            }
            return node;
        }

        // ndim >= 3: flatten around the reduction axis (same as CPU)
        if (x.ndim >= 3) {
            const total = x.totalElements();
            var before: usize = 1;
            for (0..actual_axis) |d| before *= x.shape[d];
            const axis_dim = x.shape[actual_axis];
            if (actual_axis == x.ndim - 1) {
                const flat = self.reshape(x, &.{ total / axis_dim, axis_dim });
                const reduced = self.reductionSum(flat, 1);
                var new_shape: [8]usize = undefined;
                for (0..x.ndim - 1) |d| new_shape[d] = x.shape[d];
                new_shape[x.ndim - 1] = 1;
                return self.reshape(reduced, new_shape[0..x.ndim]);
            } else if (actual_axis == 0) {
                const flat = self.reshape(x, &.{ axis_dim, total / axis_dim });
                const reduced = self.reductionSum(flat, 0);
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
                const reduced = self.reductionSum(flat, 1);
                var new_shape: [8]usize = undefined;
                for (0..x.ndim) |d| new_shape[d] = x.shape[d];
                new_shape[actual_axis] = 1;
                return self.reshape(reduced, new_shape[0..x.ndim]);
            }
        }

        @panic("reductionSum: unsupported ndim/axis combination");
    }

    fn backwardReductionSumAxis1(self_node: *DiffCudaNode) void {
        // [rows, cols] → [rows, 1]: broadcast go back
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            // go is [rows, 1], need to broadcast-add to ga [rows, cols]
            ops.dispatchReduceToBroadcast(rt.cuda_ctx, rt.kernels.fn_reduce_add_bcast, ga, self_node.grad_dptr.?, rows * cols, rows);
        }
    }

    fn backwardReductionSumAxis0(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            ops.dispatchReduceToBroadcast(rt.cuda_ctx, rt.kernels.fn_reduce_add_bcast, ga, self_node.grad_dptr.?, rows * cols, cols);
        }
    }

    fn backwardReductionSum1D(self_node: *DiffCudaNode) void {
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            const total = pa.totalElements();
            // go is scalar [1], broadcast to all elements
            ops.dispatchReduceToBroadcast(rt.cuda_ctx, rt.kernels.fn_reduce_add_bcast, ga, self_node.grad_dptr.?, total, 1);
        }
    }

    pub fn reductionMean(self: *DiffCudaRuntime, x: DiffCudaTensor, axis: i64) DiffCudaTensor {
        // mean = sum / count
        const s = self.reductionSum(x, axis);
        const actual_axis: usize = if (axis < 0) @intCast(@as(i64, @intCast(x.ndim)) + axis) else @intCast(axis);
        const count = x.shape[actual_axis];
        const total = s.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        ops.dispatchScale(self.cuda_ctx, self.kernels.fn_scale, out_dptr, s.dptr, 1.0 / @as(f32, @floatFromInt(count)), total);
        const node = self.makeNode(out_dptr, s.shape[0..s.ndim], s.requires_grad);
        if (s.requires_grad) {
            node.parents[0] = s;
            const ctx = self.allocContext(RtContext);
            ctx.* = .{ .rt = self };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardScale;
        }
        return node;
    }

    fn backwardScale(self_node: *DiffCudaNode) void {
        // This is backward for a simple scale operation: inherits from parent
        const ctx: *RtContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchAccumGrad(rt.cuda_ctx, rt.kernels.fn_accum_grad, ga, self_node.grad_dptr.?, self_node.totalElements());
        }
    }

    // ════════════════════════════════════════════════════════════════
    // LayerNorm
    // ════════════════════════════════════════════════════════════════

    pub fn layerNorm(self: *DiffCudaRuntime, x: DiffCudaTensor, gamma: DiffCudaTensor, beta: DiffCudaTensor, eps: f32, axis: i64) DiffCudaTensor {
        _ = axis;
        const total = x.totalElements();
        const dim = x.lastDim();
        const rows = total / dim;
        const out_dptr = self.allocGpuBuf(total);
        const x_norm_dptr = self.allocGpuBuf(total);
        const inv_stds_dptr = self.allocGpuBuf(rows);
        ops.dispatchLayerNormFwd(self.cuda_ctx, self.kernels.fn_layernorm_fwd, out_dptr, x_norm_dptr, inv_stds_dptr, x.dptr, gamma.dptr, beta.dptr, rows, dim, eps);
        const rg = x.requires_grad or gamma.requires_grad or beta.requires_grad;
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], rg);
        if (rg) {
            node.parents[0] = x;
            node.parents[1] = gamma;
            node.parents[2] = beta;
            const ctx = self.allocContext(LayerNormContext);
            ctx.* = .{ .rt = self, .x_norm_dptr = x_norm_dptr, .inv_stds_dptr = inv_stds_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardLayerNorm;
        }
        return node;
    }

    const LayerNormContext = struct {
        rt: *DiffCudaRuntime,
        x_norm_dptr: CUdeviceptr,
        inv_stds_dptr: CUdeviceptr,
    };

    fn backwardLayerNorm(self_node: *DiffCudaNode) void {
        const ctx: *LayerNormContext = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const px = self_node.parents[0].?;
        const pgamma = self_node.parents[1].?;
        const pbeta = self_node.parents[2].?;
        const go = self_node.grad_dptr.?;
        const total = self_node.totalElements();
        const dim = self_node.lastDim();
        const rows = total / dim;

        if (pbeta.grad_dptr != null or pgamma.grad_dptr != null) {
            const gg = pgamma.grad_dptr orelse rt.allocGpuBufZeroed(dim);
            const gb = pbeta.grad_dptr orelse rt.allocGpuBufZeroed(dim);
            ops.dispatchLayerNormBackwardDgDb(rt.cuda_ctx, rt.kernels.fn_ln_bw_dg_db, gg, gb, go, ctx.x_norm_dptr, rows, dim);
        }

        if (px.grad_dptr) |gx| {
            ops.dispatchLayerNormBackwardDx(rt.cuda_ctx, rt.kernels.fn_ln_bw_dx, gx, go, pgamma.dptr, ctx.x_norm_dptr, ctx.inv_stds_dptr, rows, dim);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Dropout
    // ════════════════════════════════════════════════════════════════

    pub fn dropout(self: *DiffCudaRuntime, x: DiffCudaTensor, rate: f32) DiffCudaTensor {
        if (!self.training) return x;

        const total = x.totalElements();
        const out_dptr = self.allocGpuBuf(total);
        const mask_dptr = self.allocGpuBuf(total);
        const seed = self.prng.random().int(u64);
        const inv_keep = 1.0 / (1.0 - rate);
        ops.dispatchDropout(self.cuda_ctx, self.kernels.fn_dropout, out_dptr, mask_dptr, x.dptr, seed, rate, inv_keep, total);
        const node = self.makeNode(out_dptr, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(DropoutCtx);
            ctx.* = .{ .rt = self, .mask_dptr = mask_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardDropout;
        }
        return node;
    }

    const DropoutCtx = struct {
        rt: *DiffCudaRuntime,
        mask_dptr: CUdeviceptr,
    };

    fn backwardDropout(self_node: *DiffCudaNode) void {
        const ctx: *DropoutCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchDropoutBackward(rt.cuda_ctx, rt.kernels.fn_dropout_bw, ga, self_node.grad_dptr.?, ctx.mask_dptr, self_node.totalElements());
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Gather (Embedding lookup)
    // ════════════════════════════════════════════════════════════════

    pub fn gather(self: *DiffCudaRuntime, table: DiffCudaTensor, indices: []const u32) DiffCudaTensor {
        const embed_dim = table.shape[1];
        const num_indices = indices.len;
        const out_dptr = self.allocGpuBuf(num_indices * embed_dim);
        // Upload indices to GPU
        const idx_dptr = self.allocGpuBuf(num_indices); // reuse float buf for u32 (same size)
        self.cuda_ctx.copyHostToDevice(idx_dptr, @ptrCast(indices.ptr), num_indices * @sizeOf(u32)) catch unreachable;
        ops.dispatchGather(self.cuda_ctx, self.kernels.fn_gather, out_dptr, table.dptr, idx_dptr, num_indices, embed_dim);
        const node = self.makeNode(out_dptr, &.{ num_indices, embed_dim }, table.requires_grad);
        if (table.requires_grad) {
            node.parents[0] = table;
            const ctx = self.allocContext(GatherCtx);
            ctx.* = .{ .rt = self, .idx_dptr = idx_dptr, .num_indices = num_indices };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardGather;
        }
        return node;
    }

    const GatherCtx = struct {
        rt: *DiffCudaRuntime,
        idx_dptr: CUdeviceptr,
        num_indices: usize,
    };

    fn backwardGather(self_node: *DiffCudaNode) void {
        const ctx: *GatherCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            const embed_dim = pa.shape[1];
            ops.dispatchScatterAdd(rt.cuda_ctx, rt.kernels.fn_scatter_add, ga, self_node.grad_dptr.?, ctx.idx_dptr, ctx.num_indices, embed_dim);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Loss functions
    // ════════════════════════════════════════════════════════════════

    pub fn mseLoss(self: *DiffCudaRuntime, pred: DiffCudaTensor, target: []const f32) DiffCudaTensor {
        const total = pred.totalElements();
        const target_dptr = self.allocGpuBuf(total);
        self.cuda_ctx.copyHostToDevice(target_dptr, @ptrCast(target.ptr), total * @sizeOf(f32)) catch unreachable;
        const out_dptr = self.allocGpuBufZeroed(1);
        ops.dispatchLossForward(self.cuda_ctx, self.kernels.fn_mse_fwd, out_dptr, pred.dptr, target_dptr, total);
        const node = self.makeNode(out_dptr, &.{1}, pred.requires_grad);
        if (pred.requires_grad) {
            node.parents[0] = pred;
            const ctx = self.allocContext(LossCtx);
            ctx.* = .{ .rt = self, .target_dptr = target_dptr, .is_mse = true };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardMseLoss;
        }
        return node;
    }

    const LossCtx = struct {
        rt: *DiffCudaRuntime,
        target_dptr: CUdeviceptr,
        is_mse: bool,
    };

    fn backwardMseLoss(self_node: *DiffCudaNode) void {
        const ctx: *LossCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchLossBackward(rt.cuda_ctx, rt.kernels.fn_mse_bw, ga, self_node.grad_dptr.?, pa.dptr, ctx.target_dptr, pa.totalElements());
        }
    }

    pub fn crossEntropyLossWithIndices(self: *DiffCudaRuntime, logits: DiffCudaTensor, indices: []const u32) DiffCudaTensor {
        const batch = logits.shape[0];
        const num_classes = logits.shape[1];
        const idx_dptr = self.allocGpuBuf(batch); // u32 same size as f32
        self.cuda_ctx.copyHostToDevice(idx_dptr, @ptrCast(indices.ptr), batch * @sizeOf(u32)) catch unreachable;
        const softmax_cache = self.allocGpuBuf(batch * num_classes);
        const out_dptr = self.allocGpuBufZeroed(1);
        ops.dispatchCrossEntropyForward(self.cuda_ctx, self.kernels.fn_ce_fwd, out_dptr, softmax_cache, logits.dptr, idx_dptr, batch, num_classes);
        const node = self.makeNode(out_dptr, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.allocContext(CECtx);
            ctx.* = .{ .rt = self, .softmax_cache_dptr = softmax_cache, .idx_dptr = idx_dptr };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardCrossEntropy;
        }
        return node;
    }

    const CECtx = struct {
        rt: *DiffCudaRuntime,
        softmax_cache_dptr: CUdeviceptr,
        idx_dptr: CUdeviceptr,
    };

    fn backwardCrossEntropy(self_node: *DiffCudaNode) void {
        const ctx: *CECtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchCrossEntropyBackward(rt.cuda_ctx, rt.kernels.fn_ce_bw, ga, self_node.grad_dptr.?, ctx.softmax_cache_dptr, ctx.idx_dptr, pa.shape[0], pa.shape[1]);
        }
    }

    pub fn bceLossWithLogits(self: *DiffCudaRuntime, logits: DiffCudaTensor, target: []const f32) DiffCudaTensor {
        const total = logits.totalElements();
        const target_dptr = self.allocGpuBuf(total);
        self.cuda_ctx.copyHostToDevice(target_dptr, @ptrCast(target.ptr), total * @sizeOf(f32)) catch unreachable;
        const out_dptr = self.allocGpuBufZeroed(1);
        ops.dispatchLossForward(self.cuda_ctx, self.kernels.fn_bce_fwd, out_dptr, logits.dptr, target_dptr, total);
        const node = self.makeNode(out_dptr, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.allocContext(LossCtx);
            ctx.* = .{ .rt = self, .target_dptr = target_dptr, .is_mse = false };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardBceLoss;
        }
        return node;
    }

    fn backwardBceLoss(self_node: *DiffCudaNode) void {
        const ctx: *LossCtx = @ptrCast(@alignCast(self_node.context.?));
        const rt = ctx.rt;
        const pa = self_node.parents[0].?;
        if (pa.grad_dptr) |ga| {
            ops.dispatchLossBackward(rt.cuda_ctx, rt.kernels.fn_bce_bw, ga, self_node.grad_dptr.?, pa.dptr, ctx.target_dptr, pa.totalElements());
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Tensor factories
    // ════════════════════════════════════════════════════════════════

    pub fn zeros(self: *DiffCudaRuntime, new_shape: []const usize) DiffCudaTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const dptr = self.allocGpuBufZeroed(size);
        return self.makeNode(dptr, new_shape, false);
    }

    pub fn ones(self: *DiffCudaRuntime, new_shape: []const usize) DiffCudaTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const dptr = self.allocGpuBuf(size);
        ops.dispatchFill(self.cuda_ctx, self.kernels.fn_fill, dptr, 1.0, size);
        return self.makeNode(dptr, new_shape, false);
    }

    pub fn randn(self: *DiffCudaRuntime, new_shape: []const usize) DiffCudaTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        // Generate on CPU, then upload
        const buf = self.allocData(size);
        const rng = self.prng.random();
        var i: usize = 0;
        while (i + 1 < size) : (i += 2) {
            const r1 = rng.float(f32) * (1.0 - std.math.floatEps(f32)) + std.math.floatEps(f32);
            const r2 = rng.float(f32);
            const r = @sqrt(-2.0 * @log(r1));
            buf[i] = r * @cos(2.0 * std.math.pi * r2);
            buf[i + 1] = r * @sin(2.0 * std.math.pi * r2);
        }
        if (size % 2 == 1) {
            const r1 = rng.float(f32) * (1.0 - std.math.floatEps(f32)) + std.math.floatEps(f32);
            const r2 = rng.float(f32);
            buf[size - 1] = @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
        }
        const dptr = self.allocGpuBuf(size);
        self.cuda_ctx.copyHostToDevice(dptr, @ptrCast(buf.ptr), size * @sizeOf(f32)) catch unreachable;
        return self.makeNode(dptr, new_shape, false);
    }

    pub fn rand(self: *DiffCudaRuntime, new_shape: []const usize) DiffCudaTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const buf = self.allocData(size);
        const rng = self.prng.random();
        for (buf) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
        const dptr = self.allocGpuBuf(size);
        self.cuda_ctx.copyHostToDevice(dptr, @ptrCast(buf.ptr), size * @sizeOf(f32)) catch unreachable;
        return self.makeNode(dptr, new_shape, false);
    }

    // ════════════════════════════════════════════════════════════════
    // Backward (topological sort + reverse traversal)
    // ════════════════════════════════════════════════════════════════

    pub fn backward(self: *DiffCudaRuntime, loss: DiffCudaTensor) void {
        // 1. Set loss gradient to 1.0
        if (loss.grad_dptr == null) {
            loss.grad_dptr = self.allocGpuBuf(loss.totalElements());
        }
        ops.dispatchFill(self.cuda_ctx, self.kernels.fn_fill, loss.grad_dptr.?, 1.0, loss.totalElements());

        // 2. Topological sort (DFS)
        self.topo_buf.clearRetainingCapacity();
        self.topoSort(loss);

        // 3. Allocate grad buffers for intermediate nodes
        for (self.topo_buf.items) |node| {
            if (node.grad_dptr == null and node.requires_grad) {
                node.grad_dptr = self.allocGpuBufZeroed(node.totalElements());
            }
        }

        // 4. Reverse traversal: call backward_fn
        var idx = self.topo_buf.items.len;
        while (idx > 0) {
            idx -= 1;
            const node = self.topo_buf.items[idx];
            if (node.backward_fn) |bfn| {
                bfn(node);
            }
        }

        // 5. Reset visited flags
        for (self.topo_buf.items) |node| {
            node.visited = false;
        }
        for (self.param_nodes) |*node| {
            node.visited = false;
        }
    }

    fn topoSort(self: *DiffCudaRuntime, node: *DiffCudaNode) void {
        if (node.visited) return;
        node.visited = true;

        for (&node.parents) |maybe_parent| {
            if (maybe_parent) |parent| {
                self.topoSort(parent);
            }
        }

        self.topo_buf.append(self.allocator, node) catch unreachable;
    }

    // ════════════════════════════════════════════════════════════════
    // Adam optimizer (GPU)
    // ════════════════════════════════════════════════════════════════

    pub fn applyAdam(self: *DiffCudaRuntime, adam: *GpuAdamState, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32) void {
        adam.step += 1;
        const bc1 = 1.0 - std.math.pow(f32, beta1, @floatFromInt(adam.step));
        const bc2 = 1.0 - std.math.pow(f32, beta2, @floatFromInt(adam.step));
        const count = self.module.paramCount();
        for (0..count) |i| {
            const size = self.module.paramSize(.{ .index = i });
            ops.dispatchAdamStep(self.cuda_ctx, self.kernels.fn_adam_step, self.param_nodes[i].dptr, self.param_grad_dptrs[i], adam.m_dptrs[i], adam.v_dptrs[i], lr, beta1, beta2, eps, wd, bc1, bc2, size);
        }
    }

    pub fn applyAdamClipped(self: *DiffCudaRuntime, adam: *GpuAdamState, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, max_grad_norm: f32) void {
        const count = self.module.paramCount();

        // 1. Compute total gradient norm on GPU
        var total_norm_sq: f64 = 0;
        for (0..count) |i| {
            const size = self.module.paramSize(.{ .index = i });
            const num_blocks = (size + ops.BLOCK_SIZE - 1) / ops.BLOCK_SIZE;
            const partial_dptr = self.allocGpuBuf(num_blocks);
            ops.dispatchNormSq(self.cuda_ctx, self.kernels.fn_norm_sq, partial_dptr, self.param_grad_dptrs[i], size);
            // D2H partial sums
            const partial = self.allocData(num_blocks);
            self.cuda_ctx.copyDeviceToHost(@ptrCast(partial.ptr), partial_dptr, num_blocks * @sizeOf(f32)) catch unreachable;
            for (partial) |v| total_norm_sq += @as(f64, v);
        }

        // 2. Clip gradients
        const total_norm: f32 = @floatCast(@sqrt(total_norm_sq));
        if (total_norm > max_grad_norm) {
            const clip_coef = max_grad_norm / (total_norm + 1e-6);
            for (0..count) |i| {
                const size = self.module.paramSize(.{ .index = i });
                ops.dispatchScaleGrad(self.cuda_ctx, self.kernels.fn_scale_grad, self.param_grad_dptrs[i], clip_coef, size);
            }
        }

        // 3. Apply Adam
        self.applyAdam(adam, lr, beta1, beta2, eps, wd);
    }

    // ════════════════════════════════════════════════════════════════
    // Parameter initialization
    // ════════════════════════════════════════════════════════════════

    pub fn initParams(self: *DiffCudaRuntime) void {
        var rng_state = std.Random.DefaultPrng.init(42);
        const rng = rng_state.random();

        for (self.module.params.items, 0..) |meta, i| {
            const size = self.module.paramSize(.{ .index = i });
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

            self.cuda_ctx.copyHostToDevice(self.param_nodes[i].dptr, @ptrCast(buf.ptr), size * @sizeOf(f32)) catch unreachable;
        }
    }

    /// CpuRuntime / DiffCpuRuntime からパラメータをロード
    pub fn loadFromCpu(self: *DiffCudaRuntime, cpu: anytype) void {
        for (0..self.module.paramCount()) |i| {
            const size = self.module.paramSize(.{ .index = i });
            const cpu_data = cpu.paramData(i);
            self.cuda_ctx.copyHostToDevice(self.param_nodes[i].dptr, @ptrCast(cpu_data.ptr), size * @sizeOf(f32)) catch unreachable;
        }
    }
};
