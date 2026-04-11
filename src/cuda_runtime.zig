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

pub const MAX_NDIM = 8;

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
        // Load PTX module (embedded at compile time)
        const ptx_data = @embedFile("backend/cuda_kernels.ptx");
        try cuda_ctx.loadModule(@ptrCast(ptx_data.ptr));

        // Get kernel functions
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

        // Allocate GPU param buffers
        const count = mod.paramCount();
        const param_bufs = try allocator.alloc(GpuTensorInfo, count);
        for (mod.params.items, 0..) |meta, i| {
            const size = mod.paramSize(.{ .index = i });
            const dptr = try cuda_ctx.allocBuffer(size * @sizeOf(f32));
            param_bufs[i] = .{
                .dptr = dptr,
                .shape = initShapeArray(meta.shape),
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
            .shape = initShapeArray(shape_slice),
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
        // For CudaRuntime, allocData returns CPU memory from arena (used for host-side prep)
        return self.arenaAlloc().alloc(f32, size) catch unreachable;
    }

    // ── Kernel launch helpers ──

    const BLOCK_SIZE: c_uint = 256;

    fn gridFor(n: usize) c_uint {
        return @intCast((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }

    fn launchElementwise(self: *CudaRuntime, func: CUfunction, out_dptr: CUdeviceptr, x_dptr: CUdeviceptr, n: usize) void {
        var n_i: c_int = @intCast(n);
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&x_dptr)),
            @ptrCast(&n_i),
        };
        self.cuda_ctx.launchKernel(
            func,
            .{ gridFor(n), 1, 1 },
            .{ BLOCK_SIZE, 1, 1 },
            0,
            &params,
        ) catch unreachable;
    }

    // ── Ops ──

    pub fn add(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();

        if (a_total >= b_total) {
            const out_dptr = self.allocGpuData(a_total) catch unreachable;
            var a_i: c_int = @intCast(a_total);
            var b_i: c_int = @intCast(b_total);
            var params = [_]?*anyopaque{
                @ptrCast(@constCast(&out_dptr)),
                @ptrCast(@constCast(&a.dptr)),
                @ptrCast(@constCast(&b.dptr)),
                @ptrCast(&a_i),
                @ptrCast(&b_i),
            };
            self.cuda_ctx.launchKernel(
                self.fn_add,
                .{ gridFor(a_total), 1, 1 },
                .{ BLOCK_SIZE, 1, 1 },
                0,
                &params,
            ) catch unreachable;
            return self.makeTensorGpu(out_dptr, a.shape[0..a.ndim]);
        } else {
            const out_dptr = self.allocGpuData(b_total) catch unreachable;
            var a_i: c_int = @intCast(b_total);
            var b_i: c_int = @intCast(a_total);
            var params = [_]?*anyopaque{
                @ptrCast(@constCast(&out_dptr)),
                @ptrCast(@constCast(&b.dptr)),
                @ptrCast(@constCast(&a.dptr)),
                @ptrCast(&a_i),
                @ptrCast(&b_i),
            };
            self.cuda_ctx.launchKernel(
                self.fn_add,
                .{ gridFor(b_total), 1, 1 },
                .{ BLOCK_SIZE, 1, 1 },
                0,
                &params,
            ) catch unreachable;
            return self.makeTensorGpu(out_dptr, b.shape[0..b.ndim]);
        }
    }

    pub fn mul(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();

        if (a_total >= b_total) {
            const out_dptr = self.allocGpuData(a_total) catch unreachable;
            var a_i: c_int = @intCast(a_total);
            var b_i: c_int = @intCast(b_total);
            var params = [_]?*anyopaque{
                @ptrCast(@constCast(&out_dptr)),
                @ptrCast(@constCast(&a.dptr)),
                @ptrCast(@constCast(&b.dptr)),
                @ptrCast(&a_i),
                @ptrCast(&b_i),
            };
            self.cuda_ctx.launchKernel(
                self.fn_mul,
                .{ gridFor(a_total), 1, 1 },
                .{ BLOCK_SIZE, 1, 1 },
                0,
                &params,
            ) catch unreachable;
            return self.makeTensorGpu(out_dptr, a.shape[0..a.ndim]);
        } else {
            const out_dptr = self.allocGpuData(b_total) catch unreachable;
            var a_i: c_int = @intCast(b_total);
            var b_i: c_int = @intCast(a_total);
            var params = [_]?*anyopaque{
                @ptrCast(@constCast(&out_dptr)),
                @ptrCast(@constCast(&b.dptr)),
                @ptrCast(@constCast(&a.dptr)),
                @ptrCast(&a_i),
                @ptrCast(&b_i),
            };
            self.cuda_ctx.launchKernel(
                self.fn_mul,
                .{ gridFor(b_total), 1, 1 },
                .{ BLOCK_SIZE, 1, 1 },
                0,
                &params,
            ) catch unreachable;
            return self.makeTensorGpu(out_dptr, b.shape[0..b.ndim]);
        }
    }

    pub fn sub(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        // a - b = a + (-1 * b)
        const neg_b = self.negative(b);
        return self.add(a, neg_b);
    }

    pub fn gelu(self: *CudaRuntime, x: GpuTensor) GpuTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuData(total) catch unreachable;
        var n_i: c_int = @intCast(total);
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&x.dptr)),
            @ptrCast(&n_i),
        };
        self.cuda_ctx.launchKernel(
            self.fn_gelu,
            .{ gridFor(total), 1, 1 },
            .{ BLOCK_SIZE, 1, 1 },
            0,
            &params,
        ) catch unreachable;
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    pub fn silu(self: *CudaRuntime, x: GpuTensor) GpuTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuData(total) catch unreachable;
        var n_i: c_int = @intCast(total);
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&x.dptr)),
            @ptrCast(&n_i),
        };
        self.cuda_ctx.launchKernel(
            self.fn_silu,
            .{ gridFor(total), 1, 1 },
            .{ BLOCK_SIZE, 1, 1 },
            0,
            &params,
        ) catch unreachable;
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    pub fn softmax(self: *CudaRuntime, x: GpuTensor, axis: i64) GpuTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        // softmax kernel works in-place, so copy first
        const out_dptr = self.allocGpuData(total) catch unreachable;
        self.cuda_ctx.copyDeviceToDevice(out_dptr, x.dptr, total * @sizeOf(f32)) catch unreachable;
        const block_dim: c_uint = @intCast(@min(cols, 1024));
        // Ensure block_dim is power of 2 for reduction
        const block_pow2 = nextPow2(block_dim);
        var rows_i: c_int = @intCast(rows);
        var cols_i: c_int = @intCast(cols);
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(&rows_i),
            @ptrCast(&cols_i),
        };
        self.cuda_ctx.launchKernel(
            self.fn_softmax,
            .{ @intCast(rows), 1, 1 },
            .{ block_pow2, 1, 1 },
            block_pow2 * @sizeOf(f32),
            &params,
        ) catch unreachable;
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    pub fn layerNorm(self: *CudaRuntime, x: GpuTensor, gamma: GpuTensor, beta: GpuTensor, eps: f32, axis: i64) GpuTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        const out_dptr = self.allocGpuData(total) catch unreachable;
        const block_dim = nextPow2(@intCast(@min(cols, 1024)));
        var rows_i: c_int = @intCast(rows);
        var cols_i: c_int = @intCast(cols);
        var eps_v: f32 = eps;
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&x.dptr)),
            @ptrCast(@constCast(&gamma.dptr)),
            @ptrCast(@constCast(&beta.dptr)),
            @ptrCast(&rows_i),
            @ptrCast(&cols_i),
            @ptrCast(&eps_v),
        };
        self.cuda_ctx.launchKernel(
            self.fn_layernorm,
            .{ @intCast(rows), 1, 1 },
            .{ block_dim, 1, 1 },
            block_dim * @sizeOf(f32),
            &params,
        ) catch unreachable;
        return self.makeTensorGpu(out_dptr, x.shape[0..x.ndim]);
    }

    pub fn modulateAdaLN(self: *CudaRuntime, norm: GpuTensor, scale_t: GpuTensor, beta: GpuTensor) GpuTensor {
        const B = norm.shape[0];
        const S = norm.shape[1];
        const D = norm.shape[2];
        const total = B * S * D;
        const out_dptr = self.allocGpuData(total) catch unreachable;
        var b_i: c_int = @intCast(B);
        var s_i: c_int = @intCast(S);
        var d_i: c_int = @intCast(D);
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&norm.dptr)),
            @ptrCast(@constCast(&scale_t.dptr)),
            @ptrCast(@constCast(&beta.dptr)),
            @ptrCast(&b_i),
            @ptrCast(&s_i),
            @ptrCast(&d_i),
        };
        self.cuda_ctx.launchKernel(
            self.fn_adaln,
            .{ gridFor(total), 1, 1 },
            .{ BLOCK_SIZE, 1, 1 },
            0,
            &params,
        ) catch unreachable;
        return self.makeTensorGpu(out_dptr, norm.shape[0..norm.ndim]);
    }

    /// Matrix multiply via cuBLAS sgemm
    pub fn matmul(self: *CudaRuntime, a: GpuTensor, b: GpuTensor) GpuTensor {
        if (a.ndim == 2 and b.ndim == 2) {
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[1];
            const out_dptr = self.allocGpuData(M * N) catch unreachable;
            // cuBLAS is column-major: C = A @ B in row-major = B^T @ A^T in col-major
            self.cuda_ctx.sgemm(
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                @intCast(N),
                @intCast(M),
                @intCast(K),
                1.0,
                b.dptr,
                @intCast(N),
                a.dptr,
                @intCast(K),
                0.0,
                out_dptr,
                @intCast(N),
            ) catch unreachable;
            return self.makeTensorGpu(out_dptr, &.{ M, N });
        }

        // Batched: 3D @ 3D
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
                self.cuda_ctx.sgemm(
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    @intCast(N),
                    @intCast(M),
                    @intCast(K),
                    1.0,
                    b_off,
                    @intCast(N),
                    a_off,
                    @intCast(K),
                    0.0,
                    c_off,
                    @intCast(N),
                ) catch unreachable;
            }
            return self.makeTensorGpu(out_dptr, &.{ B, M, N });
        }

        // 2D @ 3D: treat as batch with shared 2D
        if (a.ndim == 2 and b.ndim == 3) {
            const B = b.shape[0];
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[2];
            const out_dptr = self.allocGpuData(B * M * N) catch unreachable;
            for (0..B) |batch| {
                const b_off = b.dptr + batch * K * N * @sizeOf(f32);
                const c_off = out_dptr + batch * M * N * @sizeOf(f32);
                self.cuda_ctx.sgemm(
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    @intCast(N),
                    @intCast(M),
                    @intCast(K),
                    1.0,
                    b_off,
                    @intCast(N),
                    a.dptr,
                    @intCast(K),
                    0.0,
                    c_off,
                    @intCast(N),
                ) catch unreachable;
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
        const N = b.shape[1]; // b is [B, N, K] (NOT transposed in memory)
        const out_dptr = self.allocGpuData(B * M * N) catch unreachable;
        for (0..B) |batch| {
            const a_off = a.dptr + batch * M * K * @sizeOf(f32);
            const b_off = b.dptr + batch * N * K * @sizeOf(f32);
            const c_off = out_dptr + batch * M * N * @sizeOf(f32);
            // row-major A @ B^T: cuBLAS col-major B^T^T @ A^T = B @ A^T
            // Actually: C_rowmajor = A @ B^T  ↔  C_colmajor^T = A @ B^T
            // cuBLAS: C_col = B @ A^T → sgemm(N=N, M=M, K=K, B=B_ptr(N×K colmaj), A=A_ptr, C=C_ptr)
            // B is [N, K] in row-major = [K, N] in col-major, need transpose = OP_T
            self.cuda_ctx.sgemm(
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                @intCast(N),
                @intCast(M),
                @intCast(K),
                1.0,
                b_off,
                @intCast(K),
                a_off,
                @intCast(K),
                0.0,
                c_off,
                @intCast(N),
            ) catch unreachable;
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
            const total = B * R * C;
            const out_dptr = self.allocGpuData(total) catch unreachable;
            // Transpose each batch's R×C matrix
            for (0..B) |batch| {
                const in_off = x.dptr + batch * R * C * @sizeOf(f32);
                const out_off = out_dptr + batch * R * C * @sizeOf(f32);
                var rows_i: c_int = @intCast(R);
                var cols_i: c_int = @intCast(C);
                var params = [_]?*anyopaque{
                    @ptrCast(@constCast(&out_off)),
                    @ptrCast(@constCast(&in_off)),
                    @ptrCast(&rows_i),
                    @ptrCast(&cols_i),
                };
                self.cuda_ctx.launchKernel(
                    self.fn_transpose,
                    .{ gridFor(R * C), 1, 1 },
                    .{ BLOCK_SIZE, 1, 1 },
                    0,
                    &params,
                ) catch unreachable;
            }
            return self.makeTensorGpu(out_dptr, &.{ B, C, R });
        }

        if (x.ndim == 2 and dim1 == 0 and dim2 == 1) {
            const R = x.shape[0];
            const C = x.shape[1];
            const out_dptr = self.allocGpuData(R * C) catch unreachable;
            var rows_i: c_int = @intCast(R);
            var cols_i: c_int = @intCast(C);
            var params = [_]?*anyopaque{
                @ptrCast(@constCast(&out_dptr)),
                @ptrCast(@constCast(&x.dptr)),
                @ptrCast(&rows_i),
                @ptrCast(&cols_i),
            };
            self.cuda_ctx.launchKernel(
                self.fn_transpose,
                .{ gridFor(R * C), 1, 1 },
                .{ BLOCK_SIZE, 1, 1 },
                0,
                &params,
            ) catch unreachable;
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
        const total = num_indices * embed_dim;

        // Upload indices to GPU as int32
        const idx_dptr = self.allocGpuData(num_indices) catch unreachable; // reuse f32 alloc (same size as i32)
        self.cuda_ctx.copyHostToDevice(idx_dptr, @ptrCast(indices.ptr), num_indices * @sizeOf(u32)) catch unreachable;

        const out_dptr = self.allocGpuData(total) catch unreachable;
        var n_i: c_int = @intCast(num_indices);
        var ed_i: c_int = @intCast(embed_dim);
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&table.dptr)),
            @ptrCast(@constCast(&idx_dptr)),
            @ptrCast(&n_i),
            @ptrCast(&ed_i),
        };
        self.cuda_ctx.launchKernel(
            self.fn_gather,
            .{ gridFor(total), 1, 1 },
            .{ BLOCK_SIZE, 1, 1 },
            0,
            &params,
        ) catch unreachable;
        return self.makeTensorGpu(out_dptr, &.{ num_indices, embed_dim });
    }

    /// Negate: -x
    pub fn negative(self: *CudaRuntime, x: GpuTensor) GpuTensor {
        const total = x.totalElements();
        const out_dptr = self.allocGpuData(total) catch unreachable;
        var n_i: c_int = @intCast(total);
        var s: f32 = -1.0;
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&out_dptr)),
            @ptrCast(@constCast(&x.dptr)),
            @ptrCast(&s),
            @ptrCast(&n_i),
        };
        self.cuda_ctx.launchKernel(
            self.fn_scale,
            .{ gridFor(total), 1, 1 },
            .{ BLOCK_SIZE, 1, 1 },
            0,
            &params,
        ) catch unreachable;
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

    // ── Helpers ──

    fn nextPow2(v: c_uint) c_uint {
        if (v == 0) return 1;
        var x = v - 1;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    }

    fn initShapeArray(shape_slice: []const usize) [MAX_NDIM]usize {
        var arr: [MAX_NDIM]usize = .{0} ** MAX_NDIM;
        for (shape_slice, 0..) |s, i| arr[i] = s;
        return arr;
    }
};
