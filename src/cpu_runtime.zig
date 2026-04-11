/// cpu_runtime.zig: CPU 推論ランタイム
///
/// compute.Module のパラメータを CPU 上で管理し、eager execution で forward を実行する。
/// MpsRuntime と同じ op インターフェースを提供するため、統一モデルの forward() を
/// GPU (MpsRuntime) と CPU (CpuRuntime) の両方で実行できる。
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const compute = @import("compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const AdamState = compute.AdamState;
const cpu_backend = @import("backend/cpu.zig");
const kernels = @import("runtime_kernels.zig");

pub const MAX_NDIM = kernels.MAX_NDIM;

pub const TensorInfo = struct {
    data: []f32,
    shape: [MAX_NDIM]usize,
    ndim: usize,

    pub fn totalElements(self: *const TensorInfo) usize {
        return kernels.totalElements(self.shape, self.ndim);
    }

    /// 最後の次元のサイズ
    pub fn lastDim(self: *const TensorInfo) usize {
        return kernels.lastDim(self.shape, self.ndim);
    }

    /// 最後の次元を除いた行数
    pub fn numRows(self: *const TensorInfo) usize {
        return kernels.numRows(self.shape, self.ndim);
    }
};

pub const CpuTensor = *TensorInfo;

pub const CpuRuntime = struct {
    allocator: Allocator,
    module: *const Module,
    /// Weight data per param
    param_tensors: []TensorInfo,
    arena: std.heap.ArenaAllocator,

    pub fn init(module: *const Module, allocator: Allocator) !CpuRuntime {
        const count = module.paramCount();
        const param_tensors = try allocator.alloc(TensorInfo, count);
        for (module.params.items, 0..) |meta, i| {
            const size = module.paramSize(.{ .index = i });
            const data = try allocator.alloc(f32, size);
            param_tensors[i] = .{
                .data = data,
                .shape = kernels.initShapeArray(meta.shape),
                .ndim = meta.shape.len,
            };
        }

        return .{
            .allocator = allocator,
            .module = module,
            .param_tensors = param_tensors,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *CpuRuntime) void {
        for (self.param_tensors) |t| self.allocator.free(t.data);
        self.allocator.free(self.param_tensors);
        self.arena.deinit();
    }

    /// Forward 間で中間テンソルをクリア
    pub fn resetArena(self: *CpuRuntime) void {
        _ = self.arena.reset(.retain_capacity);
    }

    fn arenaAlloc(self: *CpuRuntime) Allocator {
        return self.arena.allocator();
    }

    // ── Param access ──

    pub fn param(self: *CpuRuntime, handle: ParamHandle) CpuTensor {
        return &self.param_tensors[handle.index];
    }

    // ── Tensor creation helpers ──

    pub fn makeTensor(self: *CpuRuntime, data: []f32, shape: []const usize) CpuTensor {
        const t = self.arenaAlloc().create(TensorInfo) catch unreachable;
        t.* = .{
            .data = data,
            .shape = kernels.initShapeArray(shape),
            .ndim = shape.len,
        };
        return t;
    }

    pub fn allocData(self: *CpuRuntime, size: usize) []f32 {
        return self.arenaAlloc().alloc(f32, size) catch unreachable;
    }

    // ── Ops (immediate execution) ──

    /// Element-wise add (with broadcast: supports scalar, bias [N]+[M,N], and 3D [B,S,D]+[B,1,D])
    pub fn add(self: *CpuRuntime, a: CpuTensor, b: CpuTensor) CpuTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();

        if (a_total == b_total) {
            const out = self.allocData(a_total);
            cpu_backend.add(f32, a.data.ptr, b.data.ptr, out.ptr, a_total);
            return self.makeTensor(out, a.shape[0..a.ndim]);
        }

        // Broadcast: b is scalar
        if (b_total == 1) {
            const out = self.allocData(a_total);
            cpu_backend.addScalar(f32, a.data.ptr, b.data[0], out.ptr, a_total);
            return self.makeTensor(out, a.shape[0..a.ndim]);
        }

        // Broadcast: a is scalar
        if (a_total == 1) {
            const out = self.allocData(b_total);
            cpu_backend.addScalar(f32, b.data.ptr, a.data[0], out.ptr, b_total);
            return self.makeTensor(out, b.shape[0..b.ndim]);
        }

        // 3D broadcast: [B, S, D] + [B, 1, D] (b has dim1=1)
        if (a.ndim == 3 and b.ndim == 3 and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[2]) {
            if (b.shape[1] == 1 and a.shape[1] > 1) {
                const B = a.shape[0];
                const S = a.shape[1];
                const D = a.shape[2];
                const out = self.allocData(a_total);
                cpu_backend.addBias(f32, a.data.ptr, b.data.ptr, out.ptr, B, S, D);
                return self.makeTensor(out, a.shape[0..a.ndim]);
            }
            if (a.shape[1] == 1 and b.shape[1] > 1) {
                const B = b.shape[0];
                const S = b.shape[1];
                const D = b.shape[2];
                const out = self.allocData(b_total);
                cpu_backend.addBias(f32, b.data.ptr, a.data.ptr, out.ptr, B, S, D);
                return self.makeTensor(out, b.shape[0..b.ndim]);
            }
        }

        // Broadcast: b is smaller (e.g., bias [N] added to [M, N])
        // batches=1 so bias pointer stays at b[0..b_total] for all rows
        if (b_total < a_total and a_total % b_total == 0) {
            const out = self.allocData(a_total);
            const rows = a_total / b_total;
            cpu_backend.addBias(f32, a.data.ptr, b.data.ptr, out.ptr, 1, rows, b_total);
            return self.makeTensor(out, a.shape[0..a.ndim]);
        }

        // Broadcast: a is smaller
        if (a_total < b_total and b_total % a_total == 0) {
            const out = self.allocData(b_total);
            const rows = b_total / a_total;
            cpu_backend.addBias(f32, b.data.ptr, a.data.ptr, out.ptr, 1, rows, a_total);
            return self.makeTensor(out, b.shape[0..b.ndim]);
        }

        unreachable; // unsupported broadcast
    }

    /// Element-wise multiply (with broadcast: supports scalar, bias, and 3D [B,S,D]*[B,1,D])
    pub fn mul(self: *CpuRuntime, a: CpuTensor, b: CpuTensor) CpuTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();

        if (a_total == b_total) {
            const out = self.allocData(a_total);
            cpu_backend.mul(f32, a.data.ptr, b.data.ptr, out.ptr, a_total);
            return self.makeTensor(out, a.shape[0..a.ndim]);
        }

        // Broadcast: b is scalar (1 element) → vDSP_vsmul
        if (b_total == 1) {
            const out = self.allocData(a_total);
            cpu_backend.scale(f32, a.data.ptr, b.data[0], out.ptr, a_total);
            return self.makeTensor(out, a.shape[0..a.ndim]);
        }
        if (a_total == 1) {
            const out = self.allocData(b_total);
            cpu_backend.scale(f32, b.data.ptr, a.data[0], out.ptr, b_total);
            return self.makeTensor(out, b.shape[0..b.ndim]);
        }

        // 3D broadcast: [B, S, D] * [B, 1, D] (b has dim1=1)
        if (a.ndim == 3 and b.ndim == 3 and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[2]) {
            if (b.shape[1] == 1 and a.shape[1] > 1) {
                const B = a.shape[0];
                const S = a.shape[1];
                const D = a.shape[2];
                const out = self.allocData(a_total);
                cpu_backend.mulBroadcast(f32, a.data.ptr, b.data.ptr, out.ptr, B, S, D);
                return self.makeTensor(out, a.shape[0..a.ndim]);
            }
            if (a.shape[1] == 1 and b.shape[1] > 1) {
                const B = b.shape[0];
                const S = b.shape[1];
                const D = b.shape[2];
                const out = self.allocData(b_total);
                cpu_backend.mulBroadcast(f32, b.data.ptr, a.data.ptr, out.ptr, B, S, D);
                return self.makeTensor(out, b.shape[0..b.ndim]);
            }
        }

        // Broadcast: b is smaller (non-scalar)
        // batches=1 so scale pointer stays at b[0..b_total] for all rows
        if (b_total < a_total and a_total % b_total == 0) {
            const out = self.allocData(a_total);
            const rows = a_total / b_total;
            cpu_backend.mulBroadcast(f32, a.data.ptr, b.data.ptr, out.ptr, 1, rows, b_total);
            return self.makeTensor(out, a.shape[0..a.ndim]);
        }

        // Broadcast: a is smaller (non-scalar)
        if (a_total < b_total and b_total % a_total == 0) {
            const out = self.allocData(b_total);
            const rows = b_total / a_total;
            cpu_backend.mulBroadcast(f32, b.data.ptr, a.data.ptr, out.ptr, 1, rows, a_total);
            return self.makeTensor(out, b.shape[0..b.ndim]);
        }

        unreachable;
    }

    /// Element-wise subtract
    pub fn sub(self: *CpuRuntime, a: CpuTensor, b: CpuTensor) CpuTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();

        if (a_total == b_total) {
            const out = self.allocData(a_total);
            cpu_backend.sub(f32, a.data.ptr, b.data.ptr, out.ptr, a_total);
            return self.makeTensor(out, a.shape[0..a.ndim]);
        }

        if (b_total < a_total and a_total % b_total == 0) {
            const out = self.allocData(a_total);
            @memcpy(out, a.data[0..a_total]);
            const rows = a_total / b_total;
            for (0..rows) |r| {
                // out[row] -= b  (vDSP_vsub: C = B - A, so we negate)
                for (0..b_total) |j| {
                    out[r * b_total + j] -= b.data[j];
                }
            }
            return self.makeTensor(out, a.shape[0..a.ndim]);
        }

        unreachable;
    }

    /// Matrix multiply: [M, K] @ [K, N] → [M, N], batched: [B, M, K] @ [B, K, N] → [B, M, N]
    pub fn matmul(self: *CpuRuntime, a: CpuTensor, b: CpuTensor) CpuTensor {
        if (a.ndim == 2 and b.ndim == 2) {
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[1];
            const out = self.allocData(M * N);
            computeMatmul2D(M, K, N, a.data, b.data, out);
            return self.makeTensor(out, &.{ M, N });
        }

        // Batched: 3D @ 3D
        if (a.ndim == 3 and b.ndim == 3) {
            const B = a.shape[0];
            const M = a.shape[1];
            const K = a.shape[2];
            const N = b.shape[2];
            const out = self.allocData(B * M * N);
            for (0..B) |batch| {
                computeMatmul2D(
                    M,
                    K,
                    N,
                    a.data[batch * M * K ..][0 .. M * K],
                    b.data[batch * K * N ..][0 .. K * N],
                    out[batch * M * N ..][0 .. M * N],
                );
            }
            return self.makeTensor(out, &.{ B, M, N });
        }

        // 2D @ 3D: treat as batch with shared 2D
        if (a.ndim == 2 and b.ndim == 3) {
            const B = b.shape[0];
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[2];
            const out = self.allocData(B * M * N);
            for (0..B) |batch| {
                computeMatmul2D(
                    M,
                    K,
                    N,
                    a.data[0 .. M * K],
                    b.data[batch * K * N ..][0 .. K * N],
                    out[batch * M * N ..][0 .. M * N],
                );
            }
            return self.makeTensor(out, &.{ B, M, N });
        }

        unreachable;
    }

    fn computeMatmul2D(M: usize, K: usize, N: usize, a_data: []const f32, b_data: []const f32, out: []f32) void {
        cpu_backend.matmul(f32, a_data.ptr, b_data.ptr, out.ptr, M, K, N);
    }

    /// Tanh GELU activation (SIMD vectorized)
    pub fn gelu(self: *CpuRuntime, x: CpuTensor) CpuTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        const vec_len = std.simd.suggestVectorLength(f32) orelse 4;
        const vec_count = total / vec_len;
        const remainder = total % vec_len;
        const sqrt_2_over_pi: @Vector(vec_len, f32) = @splat(0.7978845608028654);
        const coeff: @Vector(vec_len, f32) = @splat(0.044715);
        const half: @Vector(vec_len, f32) = @splat(0.5);
        const ones: @Vector(vec_len, f32) = @splat(1.0);
        for (0..vec_count) |vi| {
            const off = vi * vec_len;
            const v: @Vector(vec_len, f32) = x.data[off..][0..vec_len].*;
            const inner = sqrt_2_over_pi * (v + coeff * v * v * v);
            // tanh via (exp(2x)-1)/(exp(2x)+1)
            const e2 = @exp(inner + inner);
            const th = (e2 - ones) / (e2 + ones);
            out[off..][0..vec_len].* = half * v * (ones + th);
        }
        for (0..remainder) |i| {
            const idx = vec_count * vec_len + i;
            const v = x.data[idx];
            const inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
            out[idx] = 0.5 * v * (1.0 + std.math.tanh(inner));
        }
        return self.makeTensor(out, x.shape[0..x.ndim]);
    }

    /// SiLU activation (SIMD vectorized)
    pub fn silu(self: *CpuRuntime, x: CpuTensor) CpuTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        const vec_len = std.simd.suggestVectorLength(f32) orelse 4;
        const vec_count = total / vec_len;
        const remainder = total % vec_len;
        const ones: @Vector(vec_len, f32) = @splat(1.0);
        for (0..vec_count) |vi| {
            const off = vi * vec_len;
            const v: @Vector(vec_len, f32) = x.data[off..][0..vec_len].*;
            const sig = ones / (ones + @exp(-v));
            out[off..][0..vec_len].* = v * sig;
        }
        for (0..remainder) |i| {
            const idx = vec_count * vec_len + i;
            const v = x.data[idx];
            out[idx] = v / (1.0 + @exp(-v));
        }
        return self.makeTensor(out, x.shape[0..x.ndim]);
    }

    /// Fused AdaLN: result[B,S,D] = norm[B,S,D] * scale[B,1,D] + beta[B,1,D]
    pub fn modulateAdaLN(self: *CpuRuntime, norm: CpuTensor, scale_t: CpuTensor, beta: CpuTensor) CpuTensor {
        const B = norm.shape[0];
        const S = norm.shape[1];
        const D = norm.shape[2];
        const total = B * S * D;
        const out = self.allocData(total);
        cpu_backend.modulateAdaLN(f32, norm.data.ptr, scale_t.data.ptr, beta.data.ptr, out.ptr, B, S, D);
        return self.makeTensor(out, norm.shape[0..norm.ndim]);
    }

    /// Batched matmul with transposed B: [B, M, K] @ [B, N, K]^T → [B, M, N]
    /// transpose メモリコピー不要。CblasTrans を直接使用。
    pub fn matmulBatchedTransB(self: *CpuRuntime, a: CpuTensor, b: CpuTensor) CpuTensor {
        const B = a.shape[0];
        const M = a.shape[1];
        const K = a.shape[2];
        const N = b.shape[1]; // b is [B, N, K] (NOT transposed in memory)
        const out = self.allocData(B * M * N);
        for (0..B) |batch| {
            cpu_backend.matmulTransB(f32, a.data.ptr + batch * M * K, b.data.ptr + batch * N * K, out.ptr + batch * M * N, M, K, N);
        }
        return self.makeTensor(out, &.{ B, M, N });
    }

    /// Reshape (data is shared, no copy)
    pub fn reshape(self: *CpuRuntime, x: CpuTensor, new_shape: []const usize) CpuTensor {
        return self.makeTensor(x.data, new_shape);
    }

    /// Transpose dimensions d1 and d2 (creates a copy, uses vDSP_mtrans on macOS)
    pub fn transpose(self: *CpuRuntime, x: CpuTensor, d1: u64, d2: u64) CpuTensor {
        const dim1: usize = @intCast(d1);
        const dim2: usize = @intCast(d2);

        // Only support 3D transpose (batch, rows, cols) for attention patterns
        if (x.ndim == 3 and dim1 == 1 and dim2 == 2) {
            const B = x.shape[0];
            const R = x.shape[1];
            const C = x.shape[2];
            const total = B * R * C;
            const out = self.allocData(total);
            cpu_backend.transposeMatrices(f32, x.data.ptr, out.ptr, B, R, C);
            return self.makeTensor(out, &.{ B, C, R });
        }

        // 2D transpose
        if (x.ndim == 2 and dim1 == 0 and dim2 == 1) {
            const R = x.shape[0];
            const C = x.shape[1];
            const out = self.allocData(R * C);
            cpu_backend.transposeMatrices(f32, x.data.ptr, out.ptr, 1, R, C);
            return self.makeTensor(out, &.{ C, R });
        }

        unreachable;
    }

    /// Softmax along specified axis (supports -1 for last axis)
    pub fn softmax(self: *CpuRuntime, x: CpuTensor, axis: i64) CpuTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        const out = self.allocData(total);
        @memcpy(out, x.data[0..total]);
        kernels.softmaxForward(out, rows, cols);
        return self.makeTensor(out, x.shape[0..x.ndim]);
    }

    /// Log-softmax along specified axis
    pub fn logSoftmax(self: *CpuRuntime, x: CpuTensor, axis: i64) CpuTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        const out = self.allocData(total);
        kernels.logSoftmaxForward(x.data[0..total], out, rows, cols, null);
        return self.makeTensor(out, x.shape[0..x.ndim]);
    }

    /// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn layerNorm(self: *CpuRuntime, x: CpuTensor, gamma: CpuTensor, beta: CpuTensor, eps: f32, axis: i64) CpuTensor {
        _ = axis;
        const total = x.totalElements();
        const dim = x.lastDim();
        const rows = total / dim;
        const out = self.allocData(total);
        kernels.layerNormForward(x.data[0..total], out, gamma.data[0..dim], beta.data[0..dim], rows, dim, eps, null, null);
        return self.makeTensor(out, x.shape[0..x.ndim]);
    }

    /// Constant scalar
    pub fn constantScalar(self: *CpuRuntime, val: f64, dtype: u32) CpuTensor {
        _ = dtype;
        const out = self.allocData(1);
        out[0] = @floatCast(val);
        return self.makeTensor(out, &.{1});
    }

    /// Constant data
    pub fn constantData(self: *CpuRuntime, data: [*]const u8, len: usize, new_shape: []const usize, dtype: u32) CpuTensor {
        _ = dtype;
        const n_floats = len / @sizeOf(f32);
        const out = self.allocData(n_floats);
        const src: [*]const f32 = @ptrCast(@alignCast(data));
        @memcpy(out, src[0..n_floats]);
        return self.makeTensor(out, new_shape);
    }

    /// Negate
    pub fn negative(self: *CpuRuntime, x: CpuTensor) CpuTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        cpu_backend.scale(f32, x.data.ptr, -1.0, out.ptr, total);
        return self.makeTensor(out, x.shape[0..x.ndim]);
    }

    /// Sum reduction along axis
    pub fn reductionSum(self: *CpuRuntime, x: CpuTensor, axis: i64) CpuTensor {
        const actual_axis: usize = if (axis < 0) @intCast(@as(i64, @intCast(x.ndim)) + axis) else @intCast(axis);

        if (x.ndim == 2) {
            const rows = x.shape[0];
            const cols = x.shape[1];
            if (actual_axis == 1) {
                const out = self.allocData(rows);
                kernels.reductionSumRows(x.data[0 .. rows * cols], out, rows, cols);
                return self.makeTensor(out, &.{ rows, 1 });
            } else {
                const out = self.allocData(cols);
                kernels.reductionSumCols(x.data[0 .. rows * cols], out, rows, cols);
                return self.makeTensor(out, &.{ 1, cols });
            }
        }

        if (x.ndim == 1) {
            const out = self.allocData(1);
            out[0] = kernels.reductionSum1D(x.data[0..x.totalElements()]);
            return self.makeTensor(out, &.{1});
        }

        unreachable;
    }

    /// Stop gradient (no-op for CPU inference)
    pub fn stopGradient(self: *CpuRuntime, x: CpuTensor) CpuTensor {
        _ = self;
        return x;
    }

    // ── ReLU ──

    pub fn relu(self: *CpuRuntime, x: CpuTensor) CpuTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        cpu_backend.relu(f32, x.data.ptr, out.ptr, total);
        return self.makeTensor(out, x.shape[0..x.ndim]);
    }

    // ── Gather (Embedding lookup) ──

    pub fn gather(self: *CpuRuntime, table: CpuTensor, indices: []const u32) CpuTensor {
        const embed_dim = table.shape[1];
        const num_indices = indices.len;
        const out = self.allocData(num_indices * embed_dim);
        for (0..num_indices) |i| {
            const idx = indices[i];
            @memcpy(out[i * embed_dim .. (i + 1) * embed_dim], table.data[idx * embed_dim .. (idx + 1) * embed_dim]);
        }
        return self.makeTensor(out, &.{ num_indices, embed_dim });
    }

    // ── Param initialization ──

    pub fn initParams(self: *CpuRuntime) void {
        var rng_state = std.Random.DefaultPrng.init(42);
        const rng = rng_state.random();

        for (self.module.params.items, 0..) |meta, i| {
            const data = self.param_tensors[i].data;
            switch (meta.init_kind) {
                .ones => @memset(data, 1.0),
                .zeros => @memset(data, 0.0),
                .xavier => {
                    const fan_in: f32 = @floatFromInt(meta.shape[0]);
                    const scale = @sqrt(1.0 / fan_in);
                    for (data) |*val| {
                        val.* = (rng.float(f32) * 2.0 - 1.0) * scale;
                    }
                },
                .kaiming => {
                    const fan_in: f32 = @floatFromInt(meta.shape[0]);
                    const scale = @sqrt(2.0 / fan_in);
                    for (data) |*val| {
                        val.* = rng.floatNorm(f32) * scale;
                    }
                },
                .kaiming_fan => |fi| {
                    const fan_in: f32 = @floatFromInt(fi);
                    const scale = @sqrt(2.0 / fan_in);
                    for (data) |*val| {
                        val.* = rng.floatNorm(f32) * scale;
                    }
                },
                .normal => |cfg| {
                    for (data) |*val| {
                        val.* = rng.floatNorm(f32) * cfg.std_dev + cfg.mean;
                    }
                },
            }
        }
    }

    // ── Param data access ──

    /// i 番目のパラメータの f32 データスライス
    pub fn paramData(self: *const CpuRuntime, index: usize) []f32 {
        return self.param_tensors[index].data;
    }

    /// 他の Runtime のパラメータデータをロード (duck-typed: paramData(i) を呼べる型)
    pub fn loadFromCpu(self: *CpuRuntime, other: anytype) void {
        for (0..self.module.paramCount()) |i| {
            const size = self.module.paramSize(.{ .index = i });
            @memcpy(self.param_tensors[i].data[0..size], other.paramData(i)[0..size]);
        }
    }

    /// MpsRuntime のパラメータデータを CPU にロード
    pub fn loadFromMps(self: *CpuRuntime, mps: anytype) void {
        const metal_mod = @import("backend/metal.zig");
        const MetalContext = metal_mod.MetalContext;
        for (0..self.module.paramCount()) |i| {
            const size = self.module.paramSize(.{ .index = i });
            const ptr = MetalContext.bufferContents(f32, mps.param_bufs[i]);
            @memcpy(self.param_tensors[i].data[0..size], ptr[0..size]);
        }
    }

    /// Checkpoint 保存
    pub fn saveCheckpoint(self: *const CpuRuntime, adam: *const AdamState, path: []const u8) !void {
        const count = self.module.paramCount();
        const slices = try self.allocator.alloc([]const f32, count);
        defer self.allocator.free(slices);
        for (0..count) |i| {
            slices[i] = self.param_tensors[i].data;
        }
        try compute.saveCheckpoint(self.module, slices, adam, path);
    }

    /// Checkpoint 読み込み
    pub fn loadCheckpoint(self: *CpuRuntime, adam: *AdamState, path: []const u8) !void {
        const count = self.module.paramCount();
        const slices = try self.allocator.alloc([]f32, count);
        defer self.allocator.free(slices);
        for (0..count) |i| {
            slices[i] = self.param_tensors[i].data;
        }
        try compute.loadCheckpoint(self.module, slices, adam, path);
    }
};
