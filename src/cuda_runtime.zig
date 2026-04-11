/// cuda_runtime.zig: CUDA 推論ランタイム
///
/// CpuRuntime と同じ duck-typing インターフェースを GPU (CUDA) 上で提供する。
/// mdlm_forward.zig の共有 forward ロジックをそのまま使えるようにする。
const std = @import("std");
const Allocator = std.mem.Allocator;
const compute = @import("compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const AdamState = compute.AdamState;
const cuda = @import("backend/cuda.zig");
const CudaContext = cuda.CudaContext;
const CUdeviceptr = cuda.CUdeviceptr;
const CUfunction = cuda.CUfunction;
const CUBLAS_OP_N = cuda.CUBLAS_OP_N;
const CUBLAS_OP_T = cuda.CUBLAS_OP_T;
const ops = @import("cuda_ops.zig");

pub const MAX_NDIM = ops.MAX_NDIM;

pub const GpuTensorInfo = struct {
    dptr: CUdeviceptr,
    shape: [MAX_NDIM]usize,
    ndim: usize,

    pub fn totalElements(self: *const GpuTensorInfo) usize {
        var total: usize = 1;
        for (0..self.ndim) |i| total *= self.shape[i];
        return total;
    }

    pub fn lastDim(self: *const GpuTensorInfo) usize {
        return self.shape[self.ndim - 1];
    }

    pub fn numRows(self: *const GpuTensorInfo) usize {
        return self.totalElements() / self.lastDim();
    }
};

pub const GpuTensor = *GpuTensorInfo;

pub const CudaRuntime = struct {
    allocator: Allocator,
    cuda_ctx: *CudaContext,
    module: *const Module,
    param_bufs: []GpuTensorInfo,
    arena_bufs: std.ArrayList(CUdeviceptr),
    arena_infos: std.heap.ArenaAllocator,

    // PTX kernel function handles
    fn_add: CUfunction,
    fn_mul: CUfunction,
    fn_gelu: CUfunction,
    fn_silu: CUfunction,
    fn_softmax: CUfunction,
    fn_layernorm: CUfunction,
    fn_transpose: CUfunction,
    fn_adaln: CUfunction,
    fn_gather: CUfunction,
    fn_scale: CUfunction,

    pub fn init(mod: *const Module, cuda_ctx: *CudaContext, allocator: Allocator) !CudaRuntime {
        const ptx_data = @embedFile("backend/cuda_kernels.ptx");
        try cuda_ctx.loadModule(@ptrCast(ptx_data.ptr));

        const fn_add = try cuda_ctx.getFunction("add_broadcast");
        const fn_mul = try cuda_ctx.getFunction("mul_broadcast");
        const fn_gelu = try cuda_ctx.getFunction("gelu_kernel");
        const fn_silu = try cuda_ctx.getFunction("silu_kernel");
        const fn_softmax = try cuda_ctx.getFunction("softmax_kernel");
        const fn_layernorm = try cuda_ctx.getFunction("layernorm_kernel");
        const fn_transpose = try cuda_ctx.getFunction("transpose_2d_kernel");
        const fn_adaln = try cuda_ctx.getFunction("modulate_adaln_kernel");
        const fn_gather = try cuda_ctx.getFunction("gather_kernel");
        const fn_scale = try cuda_ctx.getFunction("scale_kernel");

        const count = mod.paramCount();
        const param_bufs = try allocator.alloc(GpuTensorInfo, count);
        for (mod.params.items, 0..) |meta, i| {
            const size = mod.paramSize(.{ .index = i });
            const dptr = try cuda_ctx.allocBuffer(size * @sizeOf(f32));
            param_bufs[i] = .{
                .dptr = dptr,
                .shape = ops.initShapeArray(meta.shape),
                .ndim = meta.shape.len,
            };
        }

        return .{
            .allocator = allocator,
            .cuda_ctx = cuda_ctx,
            .module = mod,
            .param_bufs = param_bufs,
            .arena_bufs = .{},
            .arena_infos = std.heap.ArenaAllocator.init(allocator),
            .fn_add = fn_add,
            .fn_mul = fn_mul,
            .fn_gelu = fn_gelu,
            .fn_silu = fn_silu,
            .fn_softmax = fn_softmax,
            .fn_layernorm = fn_layernorm,
            .fn_transpose = fn_transpose,
            .fn_adaln = fn_adaln,
            .fn_gather = fn_gather,
            .fn_scale = fn_scale,
        };
    }

    pub fn deinit(self: *CudaRuntime) void {
        self.freeArenaBuffers();
        self.arena_bufs.deinit(self.allocator);
        self.arena_infos.deinit();
        for (self.param_bufs) |t| self.cuda_ctx.freeBuffer(t.dptr);
        self.allocator.free(self.param_bufs);
    }

    fn freeArenaBuffers(self: *CudaRuntime) void {
        for (self.arena_bufs.items) |dptr| {
            self.cuda_ctx.freeBuffer(dptr);
        }
        self.arena_bufs.clearRetainingCapacity();
    }

    /// Forward 間で中間テンソルをクリア
    pub fn resetArena(self: *CudaRuntime) void {
        self.freeArenaBuffers();
        _ = self.arena_infos.reset(.retain_capacity);
    }

    fn arenaAlloc(self: *CudaRuntime) Allocator {
        return self.arena_infos.allocator();
    }

    /// GPU メモリ確保 (arena tracked)
    fn allocGpuData(self: *CudaRuntime, num_floats: usize) !CUdeviceptr {
        const dptr = try self.cuda_ctx.allocBuffer(num_floats * @sizeOf(f32));
        self.arena_bufs.append(self.allocator, dptr) catch unreachable;
        return dptr;
    }

    pub fn makeTensorGpu(self: *CudaRuntime, dptr: CUdeviceptr, shape_slice: []const usize) GpuTensor {
        const t = self.arenaAlloc().create(GpuTensorInfo) catch unreachable;
        t.* = .{
            .dptr = dptr,
            .shape = ops.initShapeArray(shape_slice),
            .ndim = shape_slice.len,
        };
        return t;
    }

    // ── Param access ──

    pub fn param(self: *CudaRuntime, handle: ParamHandle) GpuTensor {
        return &self.param_bufs[handle.index];
    }

    // ── Tensor creation helpers ──

    pub fn makeTensor(self: *CudaRuntime, data: []f32, shape_slice: []const usize) GpuTensor {
        var total: usize = 1;
        for (shape_slice) |s| total *= s;
        const dptr = self.allocGpuData(total) catch unreachable;
        self.cuda_ctx.copyHostToDevice(dptr, @ptrCast(data.ptr), total * @sizeOf(f32)) catch unreachable;
        return self.makeTensorGpu(dptr, shape_slice);
    }

    pub fn allocData(self: *CudaRuntime, size: usize) []f32 {
        return self.arenaAlloc().alloc(f32, size) catch unreachable;
    }

    // ── Ops ──

    pub fn add(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        if (a_total >= b_total) {
            const out_dptr = self.allocGpuData(a_total) catch unreachable;
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.fn_add, out_dptr, a.dptr, b.dptr, a_total, b_total);
            return self.makeTensorGpu(out_dptr, a.shape[0..a.ndim]);
        } else {
            const out_dptr = self.allocGpuData(b_total) catch unreachable;
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.fn_add, out_dptr, b.dptr, a.dptr, b_total, a_total);
            return self.makeTensorGpu(out_dptr, b.shape[0..b.ndim]);
        }
    }

    pub fn mul(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        if (a_total >= b_total) {
            const out_dptr = self.allocGpuData(a_total) catch unreachable;
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.fn_mul, out_dptr, a.dptr, b.dptr, a_total, b_total);
            return self.makeTensorGpu(out_dptr, a.shape[0..a.ndim]);
        } else {
            const out_dptr = self.allocGpuData(b_total) catch unreachable;
            ops.dispatchBroadcastBinop(self.cuda_ctx, self.fn_mul, out_dptr, b.dptr, a.dptr, b_total, a_total);
            return self.makeTensorGpu(out_dptr, b.shape[0..b.ndim]);
        }
    }

    pub fn sub(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        const neg_b = self.negative(b);
        return self.add(a, neg_b);
    }

    pub fn gelu(self: *CudaRuntime, x: GpuTensor) GpuTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuData(total) catch unreachable;
        ops.dispatchElementwise(self.cuda_ctx, self.fn_gelu, out_dptr, x.dptr, total);
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    pub fn silu(self: *CudaRuntime, x: GpuTensor) GpuTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuData(total) catch unreachable;
        ops.dispatchElementwise(self.cuda_ctx, self.fn_silu, out_dptr, x.dptr, total);
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    pub fn softmax(self: *CudaRuntime, x: GpuTensor, axis: i64) GpuTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        const out_dptr = self.allocGpuData(total) catch unreachable;
        self.cuda_ctx.copyDeviceToDevice(out_dptr, x.dptr, total * @sizeOf(f32)) catch unreachable;
        ops.dispatchSoftmax(self.cuda_ctx, self.fn_softmax, out_dptr, rows, cols);
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    pub fn layerNorm(self: *CudaRuntime, x: GpuTensor, gamma: GpuTensor, beta: GpuTensor, eps: f32, axis: i64) GpuTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        const out_dptr = self.allocGpuData(total) catch unreachable;
        ops.dispatchLayerNorm(self.cuda_ctx, self.fn_layernorm, out_dptr, x.dptr, gamma.dptr, beta.dptr, rows, cols, eps);
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    pub fn modulateAdaLN(self: *CudaRuntime, norm: GpuTensor, scale_t: GpuTensor, beta: GpuTensor) GpuTensor {
        const B = norm.shape[0];
        const S = norm.shape[1];
        const D = norm.shape[2];
        const out_dptr = self.allocGpuData(B * S * D) catch unreachable;
        ops.dispatchAdaLN(self.cuda_ctx, self.fn_adaln, out_dptr, norm.dptr, scale_t.dptr, beta.dptr, B, S, D);
        return self.makeTensorGpu(out_dptr, norm.shape[0..norm.ndim]);
    }

    /// Matrix multiply via cuBLAS sgemm
    pub fn matmul(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        if (a.ndim == 2 and b.ndim == 2) {
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[1];
            const out_dptr = self.allocGpuData(M * N) catch unreachable;
            self.cuda_ctx.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, @intCast(N), @intCast(M), @intCast(K), 1.0, b.dptr, @intCast(N), a.dptr, @intCast(K), 0.0, out_dptr, @intCast(N)) catch unreachable;
            return self.makeTensorGpu(out_dptr, &.{ M, N });
        }

        if (a.ndim == 3 and b.ndim == 3) {
            const B = a.shape[0];
            const M = a.shape[1];
            const K = a.shape[2];
            const N = b.shape[2];
            const out_dptr = self.allocGpuData(B * M * N) catch unreachable;
            for (0..B) |batch| {
                const a_off = a.dptr + batch * M * K * @sizeOf(f32);
                const b_off = b.dptr + batch * K * N * @sizeOf(f32);
                const c_off = out_dptr + batch * M * N * @sizeOf(f32);
                self.cuda_ctx.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, @intCast(N), @intCast(M), @intCast(K), 1.0, b_off, @intCast(N), a_off, @intCast(K), 0.0, c_off, @intCast(N)) catch unreachable;
            }
            return self.makeTensorGpu(out_dptr, &.{ B, M, N });
        }

        if (a.ndim == 2 and b.ndim == 3) {
            const B = b.shape[0];
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[2];
            const out_dptr = self.allocGpuData(B * M * N) catch unreachable;
            for (0..B) |batch| {
                const b_off = b.dptr + batch * K * N * @sizeOf(f32);
                const c_off = out_dptr + batch * M * N * @sizeOf(f32);
                self.cuda_ctx.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, @intCast(N), @intCast(M), @intCast(K), 1.0, b_off, @intCast(N), a.dptr, @intCast(K), 0.0, c_off, @intCast(N)) catch unreachable;
            }
            return self.makeTensorGpu(out_dptr, &.{ B, M, N });
        }

        unreachable;
    }

    /// Batched matmul with transposed B: [B, M, K] @ [B, N, K]^T → [B, M, N]
    pub fn matmulBatchedTransB(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        const B = a.shape[0];
        const M = a.shape[1];
        const K = a.shape[2];
        const N = b.shape[1];
        const out_dptr = self.allocGpuData(B * M * N) catch unreachable;
        for (0..B) |batch| {
            const a_off = a.dptr + batch * M * K * @sizeOf(f32);
            const b_off = b.dptr + batch * N * K * @sizeOf(f32);
            const c_off = out_dptr + batch * M * N * @sizeOf(f32);
            self.cuda_ctx.sgemm(CUBLAS_OP_T, CUBLAS_OP_N, @intCast(N), @intCast(M), @intCast(K), 1.0, b_off, @intCast(K), a_off, @intCast(K), 0.0, c_off, @intCast(N)) catch unreachable;
        }
        return self.makeTensorGpu(out_dptr, &.{ B, M, N });
    }

    /// Reshape (GPU pointer reuse, no copy)
    pub fn reshape(self: *CudaRuntime, x: GpuTensor, new_shape: []const usize) GpuTensor {
        return self.makeTensorGpu(x.dptr, new_shape);
    }

    /// Transpose dimensions
    pub fn transpose(self: *CudaRuntime, x: GpuTensor, d1: u64, d2: u64) GpuTensor {
        const dim1: usize = @intCast(d1);
        const dim2: usize = @intCast(d2);

        if (x.ndim == 3 and dim1 == 1 and dim2 == 2) {
            const B = x.shape[0];
            const R = x.shape[1];
            const C = x.shape[2];
            const out_dptr = self.allocGpuData(B * R * C) catch unreachable;
            for (0..B) |batch| {
                const in_off = x.dptr + batch * R * C * @sizeOf(f32);
                const out_off = out_dptr + batch * R * C * @sizeOf(f32);
                ops.dispatchTranspose2d(self.cuda_ctx, self.fn_transpose, out_off, in_off, R, C);
            }
            return self.makeTensorGpu(out_dptr, &.{ B, C, R });
        }

        if (x.ndim == 2 and dim1 == 0 and dim2 == 1) {
            const R = x.shape[0];
            const C = x.shape[1];
            const out_dptr = self.allocGpuData(R * C) catch unreachable;
            ops.dispatchTranspose2d(self.cuda_ctx, self.fn_transpose, out_dptr, x.dptr, R, C);
            return self.makeTensorGpu(out_dptr, &.{ C, R });
        }

        unreachable;
    }

    /// Constant scalar
    pub fn constantScalar(self: *CudaRuntime, val: f64, dtype: u32) GpuTensor {
        _ = dtype;
        const dptr = self.allocGpuData(1) catch unreachable;
        var v: f32 = @floatCast(val);
        self.cuda_ctx.copyHostToDevice(dptr, @ptrCast(&v), @sizeOf(f32)) catch unreachable;
        return self.makeTensorGpu(dptr, &.{1});
    }

    /// Constant data (Host → Device copy)
    pub fn constantData(self: *CudaRuntime, data: [*]const u8, len: usize, new_shape: []const usize, dtype: u32) GpuTensor {
        _ = dtype;
        const n_floats = len / @sizeOf(f32);
        const dptr = self.allocGpuData(n_floats) catch unreachable;
        self.cuda_ctx.copyHostToDevice(dptr, @ptrCast(data), len) catch unreachable;
        return self.makeTensorGpu(dptr, new_shape);
    }

    /// Gather (embedding lookup): table[indices[i]] → out[i]
    pub fn gather(self: *CudaRuntime, table: GpuTensor, indices: []const u32) GpuTensor {
        const embed_dim = table.shape[1];
        const num_indices = indices.len;
        const idx_dptr = self.allocGpuData(num_indices) catch unreachable;
        self.cuda_ctx.copyHostToDevice(idx_dptr, @ptrCast(indices.ptr), num_indices * @sizeOf(u32)) catch unreachable;
        const out_dptr = self.allocGpuData(num_indices * embed_dim) catch unreachable;
        ops.dispatchGather(self.cuda_ctx, self.fn_gather, out_dptr, table.dptr, idx_dptr, num_indices, embed_dim);
        return self.makeTensorGpu(out_dptr, &.{ num_indices, embed_dim });
    }

    /// Negate: -x
    pub fn negative(self: *CudaRuntime, x: GpuTensor) GpuTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuData(total) catch unreachable;
        ops.dispatchScale(self.cuda_ctx, self.fn_scale, out_dptr, x.dptr, -1.0, total);
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    /// Stop gradient (no-op for inference)
    pub fn stopGradient(self: *CudaRuntime, x: GpuTensor) GpuTensor {
        _ = self;
        return x;
    }

    // ── Param data access ──

    /// CPU→GPU パラメータ転送
    pub fn loadFromCpu(self: *CudaRuntime, other: anytype) void {
        for (0..self.module.paramCount()) |i| {
            const size = self.module.paramSize(.{ .index = i });
            const cpu_data = other.paramData(i);
            self.cuda_ctx.copyHostToDevice(self.param_bufs[i].dptr, @ptrCast(cpu_data.ptr), size * @sizeOf(f32)) catch unreachable;
        }
    }

    /// Device → Host コピー (logits 取得用)
    pub fn copyToHost(self: *CudaRuntime, tensor: GpuTensor, dst: []f32) void {
        const total = tensor.totalElements();
        self.cuda_ctx.copyDeviceToHost(@ptrCast(dst.ptr), tensor.dptr, total * @sizeOf(f32)) catch unreachable;
    }

    /// 同期
    pub fn synchronize(self: *CudaRuntime) void {
        self.cuda_ctx.synchronize();
    }
};
