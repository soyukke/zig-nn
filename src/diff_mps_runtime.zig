/// diff_mps_runtime.zig: 微分可能 Metal (MPS/UMA) ランタイム
///
/// DiffCpuRuntime / DiffCudaRuntime と同じ duck-typed ops インターフェースを
/// Metal GPU 上で提供する。Apple Silicon の UMA (Unified Memory Architecture) を活用し、
/// forward は Metal compute カーネルで実行、backward は UMA 経由 CPU で実行する。
/// 数値勾配チェックによる Metal カーネルの正当性検証が主な用途。
///
/// ビルド: zig build test-diff-mps (macOS only)
const std = @import("std");
const Allocator = std.mem.Allocator;
const compute = @import("compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const metal = @import("backend/metal.zig");
const MetalContext = metal.MetalContext;
const id = metal.id;
const kernels = @import("runtime_kernels.zig");
const diff_node = @import("diff_node.zig");

pub const MAX_NDIM = kernels.MAX_NDIM;

pub const DiffMpsNode = diff_node.DiffNodeGeneric(id, MAX_NDIM);
pub const DiffMpsTensor = *DiffMpsNode;

// ── UMA ヘルパー ──

/// MTLBuffer から CPU アクセス可能な f32 ポインタを取得
fn bufPtr(b: id) [*]f32 {
    return MetalContext.bufferContents(f32, b);
}

// ── Metal 同期実行ヘルパー ──

const GpuExec = struct { cmd: id, enc: id };

fn beginGpu(ctx: *MetalContext) GpuExec {
    const cmd = ctx.newCommandBuffer();
    const enc = MetalContext.newComputeEncoder(cmd);
    return .{ .cmd = cmd, .enc = enc };
}

fn endGpu(gpu: GpuExec) void {
    MetalContext.memoryBarrier(gpu.enc);
    MetalContext.endEncoding(gpu.enc);
    MetalContext.commit(gpu.cmd);
    MetalContext.waitUntilCompleted(gpu.cmd);
}

// ════════════════════════════════════════════════════════════════
// DiffMpsRuntime
// ════════════════════════════════════════════════════════════════

pub const DiffMpsRuntime = struct {
    allocator: Allocator,
    metal_ctx: *MetalContext,
    module: *const Module,
    param_nodes: []DiffMpsNode,
    param_grad_bufs: []id, // パラメータ勾配用 MTLBuffer
    arena: std.heap.ArenaAllocator,
    arena_bufs: std.ArrayListUnmanaged(id), // 中間 MTLBuffer (resetArena で解放)
    topo_buf: std.ArrayListUnmanaged(*DiffMpsNode),
    training: bool,

    pub fn init(module: *const Module, metal_ctx: *MetalContext, allocator: Allocator) !DiffMpsRuntime {
        try metal_ctx.initTrainingPipelines();

        const count = module.paramCount();
        const param_nodes = try allocator.alloc(DiffMpsNode, count);
        const param_grad_bufs = try allocator.alloc(id, count);

        for (module.params.items, 0..) |meta, i| {
            const size = module.paramSize(.{ .index = i });
            const data_buf = try metal_ctx.createBuffer(size * @sizeOf(f32));
            const grad_buf = try metal_ctx.createBuffer(size * @sizeOf(f32));
            @memset(bufPtr(grad_buf)[0..size], 0);

            param_nodes[i] = .{
                .data = data_buf,
                .shape = kernels.initShapeArray(meta.shape),
                .ndim = meta.shape.len,
                .grad = grad_buf,
                .backward_fn = null,
                .parents = .{ null, null, null },
                .context = null,
                .requires_grad = true,
                .visited = false,
                .is_param = true,
                .param_index = i,
            };
            param_grad_bufs[i] = grad_buf;
        }

        return .{
            .allocator = allocator,
            .metal_ctx = metal_ctx,
            .module = module,
            .param_nodes = param_nodes,
            .param_grad_bufs = param_grad_bufs,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .arena_bufs = .{},
            .topo_buf = .empty,
            .training = true,
        };
    }

    pub fn deinit(self: *DiffMpsRuntime) void {
        self.freeArenaBufs();
        self.arena_bufs.deinit(self.allocator);
        self.arena.deinit();
        for (self.param_nodes) |node| {
            metal.objRelease(node.data);
        }
        for (self.param_grad_bufs) |b| metal.objRelease(b);
        self.allocator.free(self.param_nodes);
        self.allocator.free(self.param_grad_bufs);
        self.topo_buf.deinit(self.allocator);
    }

    fn freeArenaBufs(self: *DiffMpsRuntime) void {
        for (self.arena_bufs.items) |b| metal.objRelease(b);
        self.arena_bufs.clearRetainingCapacity();
    }

    pub fn resetArena(self: *DiffMpsRuntime) void {
        self.freeArenaBufs();
        _ = self.arena.reset(.retain_capacity);
        for (self.param_nodes) |*node| {
            node.visited = false;
        }
    }

    pub fn zeroGrad(self: *DiffMpsRuntime) void {
        for (self.param_nodes, 0..) |*node, i| {
            const size = self.module.paramSize(.{ .index = i });
            @memset(bufPtr(self.param_grad_bufs[i])[0..size], 0);
            node.grad = self.param_grad_bufs[i];
        }
    }

    fn arenaAlloc(self: *DiffMpsRuntime) Allocator {
        return self.arena.allocator();
    }

    /// MTLBuffer 確保 (arena tracked)
    fn allocBuf(self: *DiffMpsRuntime, num_floats: usize) id {
        const b = self.metal_ctx.createBuffer(num_floats * @sizeOf(f32)) catch unreachable;
        self.arena_bufs.append(self.allocator, b) catch unreachable;
        return b;
    }

    /// MTLBuffer 確保 + ゼロ初期化
    fn allocBufZeroed(self: *DiffMpsRuntime, num_floats: usize) id {
        const b = self.allocBuf(num_floats);
        @memset(bufPtr(b)[0..num_floats], 0);
        return b;
    }

    fn allocContext(self: *DiffMpsRuntime, comptime T: type) *T {
        return self.arenaAlloc().create(T) catch unreachable;
    }

    /// allocData: CPU arena メモリ (indices 等のホストデータ用)
    pub fn allocData(self: *DiffMpsRuntime, size: usize) []f32 {
        return self.arenaAlloc().alloc(f32, size) catch unreachable;
    }

    // ── Node creation ──

    pub fn makeNode(self: *DiffMpsRuntime, data_buf: id, shape_slice: []const usize, requires_grad: bool) *DiffMpsNode {
        const node = self.arenaAlloc().create(DiffMpsNode) catch unreachable;
        node.* = .{
            .data = data_buf,
            .shape = kernels.initShapeArray(shape_slice),
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

    /// ホストデータを MTLBuffer にコピーしてノードを作成
    pub fn makeTensor(self: *DiffMpsRuntime, data: []f32, shape: []const usize) DiffMpsTensor {
        var total: usize = 1;
        for (shape) |s| total *= s;
        const b = self.allocBuf(total);
        @memcpy(bufPtr(b)[0..total], data[0..total]);
        return self.makeNode(b, shape, false);
    }

    // ── Param access ──

    pub fn param(self: *DiffMpsRuntime, handle: ParamHandle) DiffMpsTensor {
        return &self.param_nodes[handle.index];
    }

    pub fn paramGrad(self: *DiffMpsRuntime, index: usize) []f32 {
        const size = self.module.paramSize(.{ .index = index });
        const dst = self.allocData(size);
        @memcpy(dst, bufPtr(self.param_grad_bufs[index])[0..size]);
        return dst;
    }

    pub fn initParams(self: *DiffMpsRuntime) void {
        var rng_state = std.Random.DefaultPrng.init(42);
        const rng = rng_state.random();
        for (self.module.params.items, 0..) |meta_item, i| {
            const size = self.module.paramSize(.{ .index = i });
            const ptr = bufPtr(self.param_nodes[i].data);
            switch (meta_item.init_kind) {
                .ones => @memset(ptr[0..size], 1.0),
                .zeros => @memset(ptr[0..size], 0.0),
                .xavier => {
                    const fan_in: f32 = @floatFromInt(meta_item.shape[0]);
                    const scale = @sqrt(1.0 / fan_in);
                    for (ptr[0..size]) |*val| val.* = (rng.float(f32) * 2.0 - 1.0) * scale;
                },
                .kaiming => {
                    const fan_in: f32 = @floatFromInt(meta_item.shape[0]);
                    const scale = @sqrt(2.0 / fan_in);
                    for (ptr[0..size]) |*val| val.* = rng.floatNorm(f32) * scale;
                },
                .kaiming_fan => |fi| {
                    const fan_in: f32 = @floatFromInt(fi);
                    const scale = @sqrt(2.0 / fan_in);
                    for (ptr[0..size]) |*val| val.* = rng.floatNorm(f32) * scale;
                },
                .normal => |cfg| {
                    for (ptr[0..size]) |*val| val.* = rng.floatNorm(f32) * cfg.std_dev + cfg.mean;
                },
            }
        }
    }

    /// GPU テンソルを CPU バッファにコピー (UMA: 実質 memcpy)
    pub fn copyToHost(self: *DiffMpsRuntime, t: DiffMpsTensor, dst: []f32) void {
        _ = self;
        const total = t.totalElements();
        @memcpy(dst, bufPtr(t.data)[0..total]);
    }

    // ════════════════════════════════════════════════════════════════
    // Unary ops (Metal forward + CPU backward via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn gelu(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n: u32 = @intCast(x.totalElements());
        const out_buf = self.allocBuf(n);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchGeluForward(gpu.enc, x.data, out_buf, n);
        endGpu(gpu);
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardGelu;
        }
        return node;
    }

    fn backwardGelu(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const total = self_node.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const x_data = bufPtr(pa.data);
            const sqrt_2_over_pi: f32 = 0.7978845608028654;
            for (0..total) |i| {
                const v = x_data[i];
                const inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                const tanh_val = std.math.tanh(inner);
                const sech2 = 1.0 - tanh_val * tanh_val;
                const inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * v * v);
                ga[i] += go[i] * (0.5 * (1.0 + tanh_val) + 0.5 * v * sech2 * inner_deriv);
            }
        }
    }

    pub fn silu(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n: u32 = @intCast(x.totalElements());
        const out_buf = self.allocBuf(n);
        const sig_buf = self.allocBuf(n);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchSiluForward(gpu.enc, x.data, out_buf, sig_buf, n);
        endGpu(gpu);
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(SiluCtx);
            ctx.* = .{ .sig_buf = sig_buf };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardSilu;
        }
        return node;
    }

    const SiluCtx = struct { sig_buf: id };

    fn backwardSilu(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *SiluCtx = @ptrCast(@alignCast(self_node.context.?));
            const total = self_node.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const x_data = bufPtr(pa.data);
            const sig = bufPtr(ctx.sig_buf);
            for (0..total) |i| {
                ga[i] += go[i] * (sig[i] + x_data[i] * sig[i] * (1.0 - sig[i]));
            }
        }
    }

    pub fn relu(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n: u32 = @intCast(x.totalElements());
        const out_buf = self.allocBuf(n);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchReluForward(gpu.enc, x.data, out_buf, n);
        endGpu(gpu);
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardRelu;
        }
        return node;
    }

    fn backwardRelu(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const total = self_node.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const x_data = bufPtr(pa.data);
            for (0..total) |i| {
                if (x_data[i] > 0) ga[i] += go[i];
            }
        }
    }

    pub fn tanh_(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n: u32 = @intCast(x.totalElements());
        const out_buf = self.allocBuf(n);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchTanhForward(gpu.enc, x.data, out_buf, n);
        endGpu(gpu);
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardTanh;
        }
        return node;
    }

    fn backwardTanh(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const total = self_node.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const out = bufPtr(self_node.data); // tanh(x)
            for (0..total) |i| {
                ga[i] += go[i] * (1.0 - out[i] * out[i]);
            }
        }
    }

    pub fn softmax(self: *DiffMpsRuntime, x: DiffMpsTensor, axis: i64) DiffMpsTensor {
        _ = axis; // assume last axis (2D)
        const rows: u32 = @intCast(x.numRows());
        const cols: u32 = @intCast(x.lastDim());
        const n = x.totalElements();
        const out_buf = self.allocBuf(n);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchSoftmaxF32(gpu.enc, x.data, out_buf, rows, cols);
        endGpu(gpu);
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardSoftmax;
        }
        return node;
    }

    fn backwardSoftmax(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const rows = pa.numRows();
            const cols = pa.lastDim();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const sm = bufPtr(self_node.data); // softmax output
            for (0..rows) |r| {
                var dot: f32 = 0;
                for (0..cols) |c| {
                    dot += go[r * cols + c] * sm[r * cols + c];
                }
                for (0..cols) |c| {
                    const idx = r * cols + c;
                    ga[idx] += sm[idx] * (go[idx] - dot);
                }
            }
        }
    }

    // ── CPU fallback unary ops ──

    pub fn negative(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n = x.totalElements();
        const out_buf = self.allocBuf(n);
        const src = bufPtr(x.data);
        const dst = bufPtr(out_buf);
        for (0..n) |i| dst[i] = -src[i];
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardNegative;
        }
        return node;
    }

    fn backwardNegative(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const total = self_node.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            for (0..total) |i| ga[i] -= go[i];
        }
    }

    pub fn square(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n = x.totalElements();
        const out_buf = self.allocBuf(n);
        const src = bufPtr(x.data);
        const dst = bufPtr(out_buf);
        for (0..n) |i| dst[i] = src[i] * src[i];
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardSquare;
        }
        return node;
    }

    fn backwardSquare(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const total = self_node.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const x_data = bufPtr(pa.data);
            for (0..total) |i| ga[i] += go[i] * 2.0 * x_data[i];
        }
    }

    pub fn sigmoid(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n = x.totalElements();
        const out_buf = self.allocBuf(n);
        const src = bufPtr(x.data);
        const dst = bufPtr(out_buf);
        for (0..n) |i| dst[i] = 1.0 / (1.0 + @exp(-src[i]));
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardSigmoid;
        }
        return node;
    }

    fn backwardSigmoid(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const total = self_node.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const sig = bufPtr(self_node.data);
            for (0..total) |i| ga[i] += go[i] * sig[i] * (1.0 - sig[i]);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Binary ops
    // ════════════════════════════════════════════════════════════════

    pub fn add(self: *DiffMpsRuntime, a: DiffMpsTensor, b: DiffMpsTensor) DiffMpsTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out_buf = self.allocBuf(a_total);
            const gpu = beginGpu(self.metal_ctx);
            self.metal_ctx.dispatchAddF32(gpu.enc, a.data, b.data, out_buf, @intCast(a_total));
            endGpu(gpu);
            const node = self.makeNode(out_buf, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardAddSame;
            }
            return node;
        }

        // Broadcast: b smaller
        if (b_total < a_total and a_total % b_total == 0) {
            const out_buf = self.allocBuf(a_total);
            const ap = bufPtr(a.data);
            const bp = bufPtr(b.data);
            const op = bufPtr(out_buf);
            for (0..a_total) |i| op[i] = ap[i] + bp[i % b_total];
            const node = self.makeNode(out_buf, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardAddBroadcastB;
            }
            return node;
        }

        // Broadcast: a smaller
        if (a_total < b_total and b_total % a_total == 0) {
            const out_buf = self.allocBuf(b_total);
            const ap = bufPtr(a.data);
            const bp = bufPtr(b.data);
            const op = bufPtr(out_buf);
            for (0..b_total) |i| op[i] = ap[i % a_total] + bp[i];
            const node = self.makeNode(out_buf, b.shape[0..b.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardAddBroadcastA;
            }
            return node;
        }

        @panic("add: incompatible shapes for broadcast");
    }

    fn backwardAddSame(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = bufPtr(self_node.grad.?);
        const total = self_node.totalElements();
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            for (0..total) |i| ga[i] += go[i];
        }
        if (pb.grad) |g| {
            const gb = bufPtr(g);
            for (0..total) |i| gb[i] += go[i];
        }
    }

    fn backwardAddBroadcastB(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = bufPtr(self_node.grad.?);
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            for (0..a_total) |i| ga[i] += go[i];
        }
        if (pb.grad) |g| {
            const gb = bufPtr(g);
            for (0..a_total) |i| gb[i % b_total] += go[i];
        }
    }

    fn backwardAddBroadcastA(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = bufPtr(self_node.grad.?);
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            for (0..b_total) |i| ga[i % a_total] += go[i];
        }
        if (pb.grad) |g| {
            const gb = bufPtr(g);
            for (0..b_total) |i| gb[i] += go[i];
        }
    }

    pub fn mul(self: *DiffMpsRuntime, a: DiffMpsTensor, b: DiffMpsTensor) DiffMpsTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;
        const ap = bufPtr(a.data);
        const bp = bufPtr(b.data);

        if (a_total == b_total) {
            const out_buf = self.allocBuf(a_total);
            const op = bufPtr(out_buf);
            for (0..a_total) |i| op[i] = ap[i] * bp[i];
            const node = self.makeNode(out_buf, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMulSame;
            }
            return node;
        }
        if (b_total <= a_total and a_total % b_total == 0) {
            const out_buf = self.allocBuf(a_total);
            const op = bufPtr(out_buf);
            for (0..a_total) |i| op[i] = ap[i] * bp[i % b_total];
            const node = self.makeNode(out_buf, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMulBroadcastB;
            }
            return node;
        }
        if (a_total < b_total and b_total % a_total == 0) {
            const out_buf = self.allocBuf(b_total);
            const op = bufPtr(out_buf);
            for (0..b_total) |i| op[i] = ap[i % a_total] * bp[i];
            const node = self.makeNode(out_buf, b.shape[0..b.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMulBroadcastA;
            }
            return node;
        }
        @panic("mul: incompatible shapes for broadcast");
    }

    fn backwardMulSame(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = bufPtr(self_node.grad.?);
        const total = self_node.totalElements();
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            const bd = bufPtr(pb.data);
            for (0..total) |i| ga[i] += go[i] * bd[i];
        }
        if (pb.grad) |g| {
            const gb = bufPtr(g);
            const ad = bufPtr(pa.data);
            for (0..total) |i| gb[i] += go[i] * ad[i];
        }
    }

    fn backwardMulBroadcastB(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = bufPtr(self_node.grad.?);
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            const bd = bufPtr(pb.data);
            for (0..a_total) |i| ga[i] += go[i] * bd[i % b_total];
        }
        if (pb.grad) |g| {
            const gb = bufPtr(g);
            const ad = bufPtr(pa.data);
            for (0..a_total) |i| gb[i % b_total] += go[i] * ad[i];
        }
    }

    fn backwardMulBroadcastA(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = bufPtr(self_node.grad.?);
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            const bd = bufPtr(pb.data);
            for (0..b_total) |i| ga[i % a_total] += go[i] * bd[i];
        }
        if (pb.grad) |g| {
            const gb = bufPtr(g);
            const ad = bufPtr(pa.data);
            for (0..b_total) |i| gb[i] += go[i] * ad[i % a_total];
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Matmul (Metal forward, CPU backward via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn matmul(self: *DiffMpsRuntime, a: DiffMpsTensor, b: DiffMpsTensor) DiffMpsTensor {
        const rg = a.requires_grad or b.requires_grad;
        if (a.ndim == 2 and b.ndim == 2) {
            const M: u32 = @intCast(a.shape[0]);
            const K: u32 = @intCast(a.shape[1]);
            const N: u32 = @intCast(b.shape[1]);
            const out_buf = self.allocBuf(M * N);
            const gpu = beginGpu(self.metal_ctx);
            self.metal_ctx.dispatchMatmulF32(gpu.enc, a.data, b.data, out_buf, M, K, N);
            endGpu(gpu);
            const node = self.makeNode(out_buf, &.{ M, N }, rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMatmul2D;
            }
            return node;
        }
        @panic("matmul: only 2D supported in MPS runtime");
    }

    fn backwardMatmul2D(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = bufPtr(self_node.grad.?);
        const M = pa.shape[0];
        const K = pa.shape[1];
        const N = pb.shape[1];
        // dA += go @ B^T
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            const bd = bufPtr(pb.data);
            for (0..M) |i| {
                for (0..K) |j| {
                    var s: f32 = 0;
                    for (0..N) |k| s += go[i * N + k] * bd[j * N + k];
                    ga[i * K + j] += s;
                }
            }
        }
        // dB += A^T @ go
        if (pb.grad) |g| {
            const gb = bufPtr(g);
            const ad = bufPtr(pa.data);
            for (0..K) |i| {
                for (0..N) |j| {
                    var s: f32 = 0;
                    for (0..M) |k| s += ad[k * K + i] * go[k * N + j];
                    gb[i * N + j] += s;
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Shape ops (CPU via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn reshape(self: *DiffMpsRuntime, x: DiffMpsTensor, new_shape: []const usize) DiffMpsTensor {
        const node = self.makeNode(x.data, new_shape, x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardReshape;
        }
        return node;
    }

    fn backwardReshape(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const total = self_node.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            for (0..total) |i| ga[i] += go[i];
        }
    }

    pub fn transpose(self: *DiffMpsRuntime, x: DiffMpsTensor, d1: u64, d2: u64) DiffMpsTensor {
        const dim1: usize = @intCast(d1);
        const dim2: usize = @intCast(d2);

        if (x.ndim == 3 and dim1 == 1 and dim2 == 2) {
            const B = x.shape[0];
            const R = x.shape[1];
            const C = x.shape[2];
            const total = B * R * C;
            const out_buf = self.allocBuf(total);
            const src = bufPtr(x.data);
            const dst = bufPtr(out_buf);
            for (0..B) |b| {
                for (0..R) |i| {
                    for (0..C) |j| {
                        dst[b * C * R + j * R + i] = src[b * R * C + i * C + j];
                    }
                }
            }
            const node = self.makeNode(out_buf, &.{ B, C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backwardTranspose3D;
            }
            return node;
        }
        if (x.ndim == 2 and dim1 == 0 and dim2 == 1) {
            const R = x.shape[0];
            const C = x.shape[1];
            const out_buf = self.allocBuf(R * C);
            const src = bufPtr(x.data);
            const dst = bufPtr(out_buf);
            for (0..R) |i| {
                for (0..C) |j| dst[j * R + i] = src[i * C + j];
            }
            const node = self.makeNode(out_buf, &.{ C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backwardTranspose2D;
            }
            return node;
        }
        @panic("transpose: unsupported ndim");
    }

    fn backwardTranspose3D(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const B = pa.shape[0];
            const R = pa.shape[1];
            const C = pa.shape[2];
            for (0..B) |b| {
                for (0..R) |i| {
                    for (0..C) |j| ga[b * R * C + i * C + j] += go[b * C * R + j * R + i];
                }
            }
        }
    }

    fn backwardTranspose2D(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const R = pa.shape[0];
            const C = pa.shape[1];
            for (0..R) |i| {
                for (0..C) |j| ga[i * C + j] += go[j * R + i];
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Reduction ops (CPU via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn reductionSum(self: *DiffMpsRuntime, x: DiffMpsTensor, axis: i64) DiffMpsTensor {
        const actual_axis: usize = if (axis < 0) @intCast(@as(i64, @intCast(x.ndim)) + axis) else @intCast(axis);

        if (x.ndim == 2) {
            const rows = x.shape[0];
            const cols = x.shape[1];
            const src = bufPtr(x.data);
            if (actual_axis == 1) {
                const out_buf = self.allocBuf(rows);
                const dst = bufPtr(out_buf);
                for (0..rows) |i| {
                    var s: f32 = 0;
                    for (0..cols) |j| s += src[i * cols + j];
                    dst[i] = s;
                }
                const node = self.makeNode(out_buf, &.{ rows, 1 }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    node.backward_fn = &backwardReductionSumAxis1;
                }
                return node;
            } else {
                const out_buf = self.allocBuf(cols);
                const dst = bufPtr(out_buf);
                @memset(dst[0..cols], 0);
                for (0..rows) |i| {
                    for (0..cols) |j| dst[j] += src[i * cols + j];
                }
                const node = self.makeNode(out_buf, &.{ 1, cols }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    node.backward_fn = &backwardReductionSumAxis0;
                }
                return node;
            }
        }

        if (x.ndim == 1) {
            const n = x.totalElements();
            const src = bufPtr(x.data);
            const out_buf = self.allocBuf(1);
            var s: f32 = 0;
            for (0..n) |i| s += src[i];
            bufPtr(out_buf)[0] = s;
            const node = self.makeNode(out_buf, &.{1}, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backwardReductionSum1D;
            }
            return node;
        }

        // ndim >= 3: flatten around the reduction axis
        if (x.ndim >= 3) {
            const total = x.totalElements();
            var before: usize = 1;
            for (0..actual_axis) |d| before *= x.shape[d];
            const axis_dim = x.shape[actual_axis];
            if (actual_axis == x.ndim - 1) {
                const flat = self.reshape(x, &.{ total / axis_dim, axis_dim });
                const reduced = self.reductionSum(flat, 1);
                var new_shape: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
                for (0..x.ndim - 1) |d| new_shape[d] = x.shape[d];
                return self.reshape(reduced, new_shape[0..x.ndim]);
            } else if (actual_axis == 0) {
                const flat = self.reshape(x, &.{ axis_dim, total / axis_dim });
                const reduced = self.reductionSum(flat, 0);
                var new_shape: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
                for (1..x.ndim) |d| new_shape[d] = x.shape[d];
                return self.reshape(reduced, new_shape[0..x.ndim]);
            } else {
                var after: usize = 1;
                for (actual_axis + 1..x.ndim) |d| after *= x.shape[d];
                const r3 = self.reshape(x, &.{ before, axis_dim, after });
                const t3 = self.transpose(r3, 1, 2);
                const flat = self.reshape(t3, &.{ before * after, axis_dim });
                const reduced = self.reductionSum(flat, 1);
                var new_shape: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
                for (0..x.ndim) |d| new_shape[d] = x.shape[d];
                new_shape[actual_axis] = 1;
                return self.reshape(reduced, new_shape[0..x.ndim]);
            }
        }

        @panic("reductionSum: unsupported ndim/axis");
    }

    fn backwardReductionSumAxis1(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            for (0..rows) |i| {
                for (0..cols) |j| ga[i * cols + j] += go[i];
            }
        }
    }

    fn backwardReductionSumAxis0(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            for (0..rows) |i| {
                for (0..cols) |j| ga[i * cols + j] += go[j];
            }
        }
    }

    fn backwardReductionSum1D(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const total = pa.totalElements();
            for (0..total) |i| ga[i] += go[0];
        }
    }

    pub fn reductionMean(self: *DiffMpsRuntime, x: DiffMpsTensor, axis: i64) DiffMpsTensor {
        const actual_axis: usize = if (axis < 0) @intCast(@as(i64, @intCast(x.ndim)) + axis) else @intCast(axis);

        if (x.ndim == 2) {
            const rows = x.shape[0];
            const cols = x.shape[1];
            const src = bufPtr(x.data);
            if (actual_axis == 1) {
                const out_buf = self.allocBuf(rows);
                const dst = bufPtr(out_buf);
                const cols_f: f32 = @floatFromInt(cols);
                for (0..rows) |i| {
                    var s: f32 = 0;
                    for (0..cols) |j| s += src[i * cols + j];
                    dst[i] = s / cols_f;
                }
                const node = self.makeNode(out_buf, &.{ rows, 1 }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    node.backward_fn = &backwardReductionMeanAxis1;
                }
                return node;
            } else {
                const out_buf = self.allocBuf(cols);
                const dst = bufPtr(out_buf);
                const rows_f: f32 = @floatFromInt(rows);
                @memset(dst[0..cols], 0);
                for (0..rows) |i| {
                    for (0..cols) |j| dst[j] += src[i * cols + j];
                }
                for (0..cols) |j| dst[j] /= rows_f;
                const node = self.makeNode(out_buf, &.{ 1, cols }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    node.backward_fn = &backwardReductionMeanAxis0;
                }
                return node;
            }
        }
        @panic("reductionMean: only 2D supported");
    }

    fn backwardReductionMeanAxis1(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            const cols_f: f32 = @floatFromInt(cols);
            for (0..rows) |i| {
                const g = go[i] / cols_f;
                for (0..cols) |j| ga[i * cols + j] += g;
            }
        }
    }

    fn backwardReductionMeanAxis0(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            const rows_f: f32 = @floatFromInt(rows);
            for (0..rows) |i| {
                for (0..cols) |j| ga[i * cols + j] += go[j] / rows_f;
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // LayerNorm (Metal forward, CPU backward via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn layerNorm(self: *DiffMpsRuntime, x: DiffMpsTensor, gamma: DiffMpsTensor, beta: DiffMpsTensor, eps: f32, axis: i64) DiffMpsTensor {
        _ = axis;
        const rows: u32 = @intCast(x.numRows());
        const cols: u32 = @intCast(x.lastDim());
        const out_buf = self.allocBuf(rows * cols);
        const mean_buf = self.allocBuf(rows);
        const inv_std_buf = self.allocBuf(rows);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchLayerNormForward(gpu.enc, x.data, gamma.data, beta.data, out_buf, mean_buf, inv_std_buf, rows, cols, eps);
        endGpu(gpu);
        const rg = x.requires_grad or gamma.requires_grad or beta.requires_grad;
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], rg);
        if (rg) {
            node.parents[0] = x;
            node.parents[1] = gamma;
            node.parents[2] = beta;
            const ctx = self.allocContext(LayerNormCtx);
            ctx.* = .{ .mean_buf = mean_buf, .inv_std_buf = inv_std_buf };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardLayerNorm;
        }
        return node;
    }

    const LayerNormCtx = struct {
        mean_buf: id,
        inv_std_buf: id,
    };

    fn backwardLayerNorm(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?; // x
        const pg = self_node.parents[1].?; // gamma
        const pb = self_node.parents[2].?; // beta
        const ctx: *LayerNormCtx = @ptrCast(@alignCast(self_node.context.?));
        const rows = pa.numRows();
        const cols = pa.lastDim();
        const go = bufPtr(self_node.grad.?);
        const x_data = bufPtr(pa.data);
        const gamma_data = bufPtr(pg.data);
        const mean = bufPtr(ctx.mean_buf);
        const inv_std = bufPtr(ctx.inv_std_buf);

        // dBeta, dGamma
        if (pb.grad) |g| {
            const g_beta = bufPtr(g);
            for (0..rows) |r| {
                for (0..cols) |c| g_beta[c] += go[r * cols + c];
            }
        }
        if (pg.grad) |g| {
            const g_gamma = bufPtr(g);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    const xhat = (x_data[r * cols + c] - mean[r]) * inv_std[r];
                    g_gamma[c] += go[r * cols + c] * xhat;
                }
            }
        }
        // dX
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            const cols_f: f32 = @floatFromInt(cols);
            for (0..rows) |r| {
                var sum_go_gamma: f32 = 0;
                var sum_go_gamma_xhat: f32 = 0;
                for (0..cols) |c| {
                    const gg = go[r * cols + c] * gamma_data[c];
                    sum_go_gamma += gg;
                    sum_go_gamma_xhat += gg * (x_data[r * cols + c] - mean[r]) * inv_std[r];
                }
                for (0..cols) |c| {
                    const xhat = (x_data[r * cols + c] - mean[r]) * inv_std[r];
                    const dx = inv_std[r] * (go[r * cols + c] * gamma_data[c] - sum_go_gamma / cols_f - xhat * sum_go_gamma_xhat / cols_f);
                    ga[r * cols + c] += dx;
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // RMSNorm: y = x * inv_rms * weight, inv_rms = 1/sqrt(mean(x²) + eps)
    // ════════════════════════════════════════════════════════════════

    pub fn rmsNorm(self: *DiffMpsRuntime, x: DiffMpsTensor, weight: DiffMpsTensor, eps: f32) DiffMpsTensor {
        const rows: u32 = @intCast(x.numRows());
        const cols: u32 = @intCast(x.lastDim());
        const n = x.totalElements();
        const out_buf = self.allocBuf(n);
        const inv_rms_buf = self.allocBuf(rows);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchRMSNormForwardTraining(gpu.enc, x.data, weight.data, out_buf, inv_rms_buf, rows, cols, eps);
        endGpu(gpu);
        const rg = x.requires_grad or weight.requires_grad;
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], rg);
        if (rg) {
            node.parents[0] = x;
            node.parents[1] = weight;
            const ctx = self.allocContext(RmsNormCtx);
            ctx.* = .{ .inv_rms_buf = inv_rms_buf };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardRmsNorm;
        }
        return node;
    }

    const RmsNormCtx = struct { inv_rms_buf: id };

    fn backwardRmsNorm(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?; // x
        const pw = self_node.parents[1].?; // weight
        const ctx: *RmsNormCtx = @ptrCast(@alignCast(self_node.context.?));
        const rows = pa.numRows();
        const cols = pa.lastDim();
        const go = bufPtr(self_node.grad.?);
        const x_data = bufPtr(pa.data);
        const w_data = bufPtr(pw.data);
        const inv_rms = bufPtr(ctx.inv_rms_buf);
        const cols_f: f32 = @floatFromInt(cols);

        // dWeight[c] = Σ_r go[r,c] * x[r,c] * inv_rms[r]
        if (pw.grad) |g| {
            const g_w = bufPtr(g);
            for (0..rows) |r| {
                const s = inv_rms[r];
                for (0..cols) |c| {
                    g_w[c] += go[r * cols + c] * x_data[r * cols + c] * s;
                }
            }
        }
        // dX[r,c] = inv_rms[r] * (w[c] * go[r,c] - x[r,c] * inv_rms[r]² · (Σⱼ x[r,j]·w[j]·go[r,j]) / D)
        if (pa.grad) |g| {
            const ga = bufPtr(g);
            for (0..rows) |r| {
                const s = inv_rms[r];
                var dot: f32 = 0;
                for (0..cols) |c| {
                    dot += x_data[r * cols + c] * w_data[c] * go[r * cols + c];
                }
                const coef = dot * s * s / cols_f;
                for (0..cols) |c| {
                    const idx = r * cols + c;
                    ga[idx] += s * (w_data[c] * go[idx] - x_data[idx] * coef);
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Causal softmax: softmax with upper-triangular mask (attention scores)
    // x shape: [batch*num_heads, seq_q, seq_k] flattened to rows×cols
    // where rows = batch*num_heads*seq_q, cols = seq_k = seq_len
    // ════════════════════════════════════════════════════════════════

    pub fn causalSoftmax(self: *DiffMpsRuntime, x: DiffMpsTensor, num_heads: u32, seq_len: u32) DiffMpsTensor {
        const rows: u32 = @intCast(x.numRows());
        const cols: u32 = @intCast(x.lastDim());
        const n = x.totalElements();
        const out_buf = self.allocBuf(n);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchCausalSoftmaxF32(gpu.enc, x.data, out_buf, rows, cols, num_heads, seq_len);
        endGpu(gpu);
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            // Masked positions have softmax output = 0, so standard softmax backward
            // naturally zeroes the gradient at those positions. Reuse existing kernel.
            node.backward_fn = &backwardSoftmax;
        }
        return node;
    }

    // ════════════════════════════════════════════════════════════════
    // Rotary Position Embedding (RoPE)
    // x shape: [seq_len, n_heads, 2*half_dim] (last dim is head_dim, must be even)
    // freqs shape: [half_dim] precomputed frequencies θᵢ = base^(-2i/d)
    // ════════════════════════════════════════════════════════════════

    pub fn rope(self: *DiffMpsRuntime, x: DiffMpsTensor, freqs: DiffMpsTensor, n_heads: u32, seq_len: u32, half_dim: u32) DiffMpsTensor {
        const n = x.totalElements();
        const out_buf = self.allocBuf(n);
        // Metal RoPE kernel works in-place; copy x into out_buf first.
        @memcpy(bufPtr(out_buf)[0..n], bufPtr(x.data)[0..n]);
        const sin_cache = self.allocBuf(seq_len * half_dim);
        const cos_cache = self.allocBuf(seq_len * half_dim);
        const gpu = beginGpu(self.metal_ctx);
        self.metal_ctx.dispatchRoPEForwardTraining(gpu.enc, out_buf, freqs.data, sin_cache, cos_cache, seq_len, n_heads, half_dim);
        endGpu(gpu);
        const node = self.makeNode(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(RopeCtx);
            ctx.* = .{
                .sin_cache = sin_cache,
                .cos_cache = cos_cache,
                .seq_len = seq_len,
                .n_heads = n_heads,
                .half_dim = half_dim,
            };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardRope;
        }
        return node;
    }

    const RopeCtx = struct {
        sin_cache: id,
        cos_cache: id,
        seq_len: u32,
        n_heads: u32,
        half_dim: u32,
    };

    fn backwardRope(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *RopeCtx = @ptrCast(@alignCast(self_node.context.?));
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const sin_c = bufPtr(ctx.sin_cache);
            const cos_c = bufPtr(ctx.cos_cache);
            const seq_len: usize = ctx.seq_len;
            const n_heads: usize = ctx.n_heads;
            const half_dim: usize = ctx.half_dim;
            const head_dim = half_dim * 2;
            // Metal kernel stores rotated pairs as interleaved [i*2, i*2+1]
            // Forward: (x0, x1) → (x0·cos - x1·sin, x0·sin + x1·cos)
            // Backward (transpose):
            //   g_x0 =  g_out0·cos + g_out1·sin
            //   g_x1 = -g_out0·sin + g_out1·cos
            for (0..seq_len) |t| {
                for (0..n_heads) |h| {
                    const base = (t * n_heads + h) * head_dim;
                    const trig_base = t * half_dim;
                    for (0..half_dim) |i| {
                        const c = cos_c[trig_base + i];
                        const s = sin_c[trig_base + i];
                        const g0 = go[base + i * 2];
                        const g1 = go[base + i * 2 + 1];
                        ga[base + i * 2] += g0 * c + g1 * s;
                        ga[base + i * 2 + 1] += -g0 * s + g1 * c;
                    }
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Gather (Embedding lookup — CPU via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn gather(self: *DiffMpsRuntime, table: DiffMpsTensor, indices: []const u32) DiffMpsTensor {
        const embed_dim = table.shape[1];
        const num_indices = indices.len;
        const out_buf = self.allocBuf(num_indices * embed_dim);
        const t_data = bufPtr(table.data);
        const o_data = bufPtr(out_buf);
        for (0..num_indices) |i| {
            const row = indices[i];
            @memcpy(o_data[i * embed_dim ..][0..embed_dim], t_data[row * embed_dim ..][0..embed_dim]);
        }
        const node = self.makeNode(out_buf, &.{ num_indices, embed_dim }, table.requires_grad);
        if (table.requires_grad) {
            node.parents[0] = table;
            const ctx = self.allocContext(GatherCtx);
            ctx.* = .{ .indices = indices, .num_indices = num_indices };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardGather;
        }
        return node;
    }

    const GatherCtx = struct {
        indices: []const u32,
        num_indices: usize,
    };

    fn backwardGather(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *GatherCtx = @ptrCast(@alignCast(self_node.context.?));
            const embed_dim = pa.shape[1];
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            for (0..ctx.num_indices) |i| {
                const row = ctx.indices[i];
                for (0..embed_dim) |j| ga[row * embed_dim + j] += go[i * embed_dim + j];
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Loss functions (CPU via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn mseLoss(self: *DiffMpsRuntime, pred: DiffMpsTensor, target: []const f32) DiffMpsTensor {
        const total = pred.totalElements();
        const pd = bufPtr(pred.data);
        var sum_sq: f32 = 0;
        for (0..total) |i| {
            const diff = pd[i] - target[i];
            sum_sq += diff * diff;
        }
        const n_f: f32 = @floatFromInt(total);
        const out_buf = self.allocBuf(1);
        bufPtr(out_buf)[0] = sum_sq / n_f;
        const node = self.makeNode(out_buf, &.{1}, pred.requires_grad);
        if (pred.requires_grad) {
            node.parents[0] = pred;
            const ctx = self.allocContext(MseLossCtx);
            ctx.* = .{ .target = target };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardMseLoss;
        }
        return node;
    }

    const MseLossCtx = struct { target: []const f32 };

    fn backwardMseLoss(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *MseLossCtx = @ptrCast(@alignCast(self_node.context.?));
            const total = pa.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const pd = bufPtr(pa.data);
            const n_f: f32 = @floatFromInt(total);
            const scale = go[0] * 2.0 / n_f;
            for (0..total) |i| ga[i] += scale * (pd[i] - ctx.target[i]);
        }
    }

    pub fn crossEntropyLossWithIndices(self: *DiffMpsRuntime, logits: DiffMpsTensor, indices: []const u32) DiffMpsTensor {
        const batch = logits.shape[0];
        const num_classes = logits.shape[1];
        const ld = bufPtr(logits.data);

        const softmax_cache = self.allocData(batch * num_classes);
        var total_loss: f32 = 0;
        for (0..batch) |i| {
            var max_val: f32 = -std.math.inf(f32);
            for (0..num_classes) |j| {
                const v = ld[i * num_classes + j];
                if (v > max_val) max_val = v;
            }
            var sum_exp: f32 = 0;
            for (0..num_classes) |j| {
                const e = @exp(ld[i * num_classes + j] - max_val);
                softmax_cache[i * num_classes + j] = e;
                sum_exp += e;
            }
            for (0..num_classes) |j| softmax_cache[i * num_classes + j] /= sum_exp;
            total_loss -= @log(softmax_cache[i * num_classes + indices[i]] + 1e-10);
        }

        const batch_f: f32 = @floatFromInt(batch);
        const out_buf = self.allocBuf(1);
        bufPtr(out_buf)[0] = total_loss / batch_f;
        const node = self.makeNode(out_buf, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.allocContext(CECtx);
            ctx.* = .{ .softmax_cache = softmax_cache, .indices = indices };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardCrossEntropy;
        }
        return node;
    }

    const CECtx = struct { softmax_cache: []f32, indices: []const u32 };

    fn backwardCrossEntropy(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *CECtx = @ptrCast(@alignCast(self_node.context.?));
            const batch = pa.shape[0];
            const num_classes = pa.shape[1];
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const batch_f: f32 = @floatFromInt(batch);
            const scale = go[0] / batch_f;
            for (0..batch) |i| {
                for (0..num_classes) |j| {
                    var g = ctx.softmax_cache[i * num_classes + j];
                    if (j == ctx.indices[i]) g -= 1.0;
                    ga[i * num_classes + j] += scale * g;
                }
            }
        }
    }

    pub fn bceLossWithLogits(self: *DiffMpsRuntime, logits: DiffMpsTensor, target: []const f32) DiffMpsTensor {
        const total = logits.totalElements();
        const ld = bufPtr(logits.data);
        var loss_sum: f32 = 0;
        for (0..total) |i| {
            const x = ld[i];
            const t = target[i];
            const pos_part: f32 = if (x > 0) x else 0;
            loss_sum += pos_part - x * t + @log(1.0 + @exp(-@abs(x)));
        }
        const n_f: f32 = @floatFromInt(total);
        const out_buf = self.allocBuf(1);
        bufPtr(out_buf)[0] = loss_sum / n_f;
        const node = self.makeNode(out_buf, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.allocContext(BceLossCtx);
            ctx.* = .{ .target = target };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardBceLoss;
        }
        return node;
    }

    const BceLossCtx = struct { target: []const f32 };

    fn backwardBceLoss(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *BceLossCtx = @ptrCast(@alignCast(self_node.context.?));
            const total = pa.totalElements();
            const go = bufPtr(self_node.grad.?);
            const ga = bufPtr(ga_buf);
            const ld = bufPtr(pa.data);
            const n_f: f32 = @floatFromInt(total);
            const scale = go[0] / n_f;
            for (0..total) |i| {
                const sig = 1.0 / (1.0 + @exp(-ld[i]));
                ga[i] += scale * (sig - ctx.target[i]);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Backward (topological sort + reverse traversal)
    // ════════════════════════════════════════════════════════════════

    pub fn backward(self: *DiffMpsRuntime, loss: DiffMpsTensor) void {
        // 1. Set loss gradient to 1.0 (UMA: direct write)
        if (loss.grad == null) {
            loss.grad = self.allocBuf(loss.totalElements());
        }
        const loss_total = loss.totalElements();
        const lg = bufPtr(loss.grad.?);
        for (0..loss_total) |i| lg[i] = 1.0;

        // 2. Topological sort
        self.topo_buf.clearRetainingCapacity();
        diff_node.topoSort(DiffMpsNode, loss, &self.topo_buf, self.allocator);

        // 3. Allocate grad buffers for intermediate nodes (UMA: zeroed MTLBuffer)
        for (self.topo_buf.items) |node| {
            if (node.grad == null and node.requires_grad) {
                node.grad = self.allocBufZeroed(node.totalElements());
            }
        }

        // 4-5. Reverse traversal + reset visited
        diff_node.backwardPass(DiffMpsNode, &self.topo_buf, self.param_nodes);
    }
};
