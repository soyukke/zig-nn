/// diff/mps_runtime.zig: 微分可能 Metal (MPS/UMA) ランタイム
///
/// DiffCpuRuntime / DiffCudaRuntime と同じ duck-typed ops インターフェースを
/// Metal GPU 上で提供する。Apple Silicon の UMA (Unified Memory Architecture) を活用し、
/// forward は Metal compute カーネルで実行、backward は UMA 経由 CPU で実行する。
/// 数値勾配チェックによる Metal カーネルの正当性検証が主な用途。
///
/// ビルド: zig build test-diff-mps (macOS only)
const std = @import("std");
const Allocator = std.mem.Allocator;
const compute = @import("../compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const metal = @import("../backend/metal.zig");
const MetalContext = metal.MetalContext;
const id = metal.id;
const kernels = @import("../runtime_kernels.zig");
const diff_node = @import("node.zig");
const unary = @import("common/unary.zig");
const binary = @import("common/binary.zig");
const reduce = @import("common/reduce.zig");
const softmax_common = @import("common/softmax.zig");
const matmul_common = @import("common/matmul.zig");

pub const MAX_NDIM = kernels.MAX_NDIM;

pub const DiffMpsNode = diff_node.diff_node_generic(id, MAX_NDIM);
pub const DiffMpsTensor = *DiffMpsNode;

// ── UMA ヘルパー ──

/// MTLBuffer から CPU アクセス可能な f32 ポインタを取得
fn buf_ptr(b: id) [*]f32 {
    return MetalContext.buffer_contents(f32, b);
}

// ── Metal 同期実行ヘルパー ──

const GpuExec = struct { cmd: id, enc: id };

fn begin_gpu(ctx: *MetalContext) GpuExec {
    const cmd = ctx.new_command_buffer();
    const enc = MetalContext.new_compute_encoder(cmd);
    return .{ .cmd = cmd, .enc = enc };
}

fn end_gpu(gpu: GpuExec) void {
    MetalContext.memory_barrier(gpu.enc);
    MetalContext.end_encoding(gpu.enc);
    MetalContext.commit(gpu.cmd);
    MetalContext.wait_until_completed(gpu.cmd);
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
    prng: std.Random.DefaultPrng, // stochastic op 用 (op で状態更新)
    init_seed: u64, // init_params 用の固定 seed
    training: bool,

    pub const DEFAULT_SEED: u64 = 42;

    pub fn init(
        module: *const Module,
        metal_ctx: *MetalContext,
        allocator: Allocator,
    ) !DiffMpsRuntime {
        try metal_ctx.init_training_pipelines();

        const count = module.param_count();
        const param_nodes = try allocator.alloc(DiffMpsNode, count);
        const param_grad_bufs = try allocator.alloc(id, count);

        for (module.params.items, 0..) |meta, i| {
            const size = module.param_size(.{ .index = i });
            const data_buf = try metal_ctx.create_buffer(size * @sizeOf(f32));
            const grad_buf = try metal_ctx.create_buffer(size * @sizeOf(f32));
            @memset(buf_ptr(grad_buf)[0..size], 0);

            param_nodes[i] = .{
                .data = data_buf,
                .shape = kernels.init_shape_array(meta.shape),
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
            .arena_bufs = .empty,
            .topo_buf = .empty,
            .prng = std.Random.DefaultPrng.init(DEFAULT_SEED),
            .init_seed = DEFAULT_SEED,
            .training = true,
        };
    }

    /// 全ランダム性 (init_params 等) を seed 固定する。
    /// init_params() の前に呼ぶと重みが決定論的になる。
    pub fn set_seed(self: *DiffMpsRuntime, seed: u64) void {
        self.init_seed = seed;
        self.prng = std.Random.DefaultPrng.init(seed);
    }

    pub fn deinit(self: *DiffMpsRuntime) void {
        self.free_arena_bufs();
        self.arena_bufs.deinit(self.allocator);
        self.arena.deinit();
        for (self.param_nodes) |node| {
            metal.obj_release(node.data);
        }
        for (self.param_grad_bufs) |b| metal.obj_release(b);
        self.allocator.free(self.param_nodes);
        self.allocator.free(self.param_grad_bufs);
        self.topo_buf.deinit(self.allocator);
    }

    fn free_arena_bufs(self: *DiffMpsRuntime) void {
        for (self.arena_bufs.items) |b| metal.obj_release(b);
        self.arena_bufs.clearRetainingCapacity();
    }

    pub fn reset_arena(self: *DiffMpsRuntime) void {
        self.free_arena_bufs();
        _ = self.arena.reset(.retain_capacity);
        for (self.param_nodes) |*node| {
            node.visited = false;
        }
    }

    pub fn zero_grad(self: *DiffMpsRuntime) void {
        for (self.param_nodes, 0..) |*node, i| {
            const size = self.module.param_size(.{ .index = i });
            @memset(buf_ptr(self.param_grad_bufs[i])[0..size], 0);
            node.grad = self.param_grad_bufs[i];
        }
    }

    fn arena_alloc(self: *DiffMpsRuntime) Allocator {
        return self.arena.allocator();
    }

    /// MTLBuffer 確保 (arena tracked, resetArena で解放される)。
    /// num_floats × 4 バイトを確保する。u32 等 4 バイト型にも流用可。
    pub fn alloc_buf(self: *DiffMpsRuntime, num_floats: usize) id {
        const b = self.metal_ctx.create_buffer(num_floats * @sizeOf(f32)) catch unreachable;
        self.arena_bufs.append(self.allocator, b) catch unreachable;
        return b;
    }

    /// MTLBuffer 確保 + ゼロ初期化
    fn alloc_buf_zeroed(self: *DiffMpsRuntime, num_floats: usize) id {
        const b = self.alloc_buf(num_floats);
        @memset(buf_ptr(b)[0..num_floats], 0);
        return b;
    }

    fn alloc_context(self: *DiffMpsRuntime, comptime T: type) *T {
        return self.arena_alloc().create(T) catch unreachable;
    }

    /// allocData: CPU arena メモリ (indices 等のホストデータ用)
    pub fn alloc_data(self: *DiffMpsRuntime, size: usize) []f32 {
        return self.arena_alloc().alloc(f32, size) catch unreachable;
    }

    // ── Node creation ──

    pub fn make_node(
        self: *DiffMpsRuntime,
        data_buf: id,
        shape_slice: []const usize,
        requires_grad: bool,
    ) *DiffMpsNode {
        const node = self.arena_alloc().create(DiffMpsNode) catch unreachable;
        node.* = .{
            .data = data_buf,
            .shape = kernels.init_shape_array(shape_slice),
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
    pub fn make_tensor(self: *DiffMpsRuntime, data: []f32, shape: []const usize) DiffMpsTensor {
        var total: usize = 1;
        for (shape) |s| total *= s;
        const b = self.alloc_buf(total);
        @memcpy(buf_ptr(b)[0..total], data[0..total]);
        return self.make_node(b, shape, false);
    }

    // ── Param access ──

    pub fn param(self: *DiffMpsRuntime, handle: ParamHandle) DiffMpsTensor {
        return &self.param_nodes[handle.index];
    }

    pub fn param_grad(self: *DiffMpsRuntime, index: usize) []f32 {
        const size = self.module.param_size(.{ .index = index });
        const dst = self.alloc_data(size);
        @memcpy(dst, buf_ptr(self.param_grad_bufs[index])[0..size]);
        return dst;
    }

    pub fn init_params(self: *DiffMpsRuntime) void {
        var rng_state = std.Random.DefaultPrng.init(self.init_seed);
        const rng = rng_state.random();
        for (self.module.params.items, 0..) |meta_item, i| {
            const size = self.module.param_size(.{ .index = i });
            const ptr = buf_ptr(self.param_nodes[i].data);
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
    pub fn copy_to_host(self: *DiffMpsRuntime, t: DiffMpsTensor, dst: []f32) void {
        _ = self;
        const total = t.total_elements();
        @memcpy(dst, buf_ptr(t.data)[0..total]);
    }

    // ════════════════════════════════════════════════════════════════
    // Unary ops (Metal forward + CPU backward via UMA)
    // ════════════════════════════════════════════════════════════════
    //
    // Pointwise unary の数式は diff/common/unary.zig に集約。
    // - CPU fallback forward + CPU backward: unaryKindCpu を使う
    // - GPU forward + CPU backward:         forward は dispatchXxx、
    //                                        backward_fn = &UnaryBackward(Kind).apply

    /// CPU fwd (UMA) + CPU bwd (UMA) の pointwise unary op (Metal kernel なし)。
    fn unary_kind_cpu(
        self: *DiffMpsRuntime,
        comptime kind: unary.Kind,
        x: DiffMpsTensor,
    ) DiffMpsTensor {
        const n = x.total_elements();
        const out_buf = self.alloc_buf(n);
        const src = buf_ptr(x.data);
        const dst = buf_ptr(out_buf);
        for (0..n) |i| dst[i] = kind.fwd(src[i]);
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &unary_backward(kind).apply;
        }
        return node;
    }

    /// 共通 backward: ga[i] += go[i] * kind.deriv(x[i], y[i])
    /// GPU forward を使う op (gelu/relu/tanh) からも直接参照して共有する。
    fn unary_backward(comptime kind: unary.Kind) type {
        return struct {
            fn apply(self_node: *DiffMpsNode) void {
                const pa = self_node.parents[0].?;
                if (pa.grad) |ga_buf| {
                    const total = self_node.total_elements();
                    const go = buf_ptr(self_node.grad.?);
                    const ga = buf_ptr(ga_buf);
                    const x_data = buf_ptr(pa.data);
                    const y_data = buf_ptr(self_node.data);
                    for (0..total) |i| {
                        ga[i] += go[i] * kind.deriv(x_data[i], y_data[i]);
                    }
                }
            }
        };
    }

    // ── Binary ops driver (数式は diff/common/binary.zig) ──

    /// CPU fwd (UMA) + CPU bwd (UMA) の pointwise binary op。broadcast 対応。
    /// GPU kernel を持つ op (add の same-shape) は別パスで先に処理し、broadcast のみ
    /// ここに fall-through することもできる。
    fn binary_kind_cpu(
        self: *DiffMpsRuntime,
        comptime kind: binary.Kind,
        a: DiffMpsTensor,
        b: DiffMpsTensor,
    ) DiffMpsTensor {
        const a_total = a.total_elements();
        const b_total = b.total_elements();
        const out_total = @max(a_total, b_total);
        const smaller = @min(a_total, b_total);
        if (!(a_total == b_total or out_total % smaller == 0)) {
            @panic("binary: incompatible shapes for broadcast");
        }
        const out_shape = if (a_total >= b_total) a.shape[0..a.ndim] else b.shape[0..b.ndim];
        const out_buf = self.alloc_buf(out_total);
        const ap = buf_ptr(a.data);
        const bp = buf_ptr(b.data);
        const op = buf_ptr(out_buf);
        for (0..out_total) |i| {
            op[i] = kind.fwd(ap[i % a_total], bp[i % b_total]);
        }
        const rg = a.requires_grad or b.requires_grad;
        const node = self.make_node(out_buf, out_shape, rg);
        if (rg) {
            node.parents[0] = a;
            node.parents[1] = b;
            node.backward_fn = &binary_backward(kind).apply;
        }
        return node;
    }

    fn binary_backward(comptime kind: binary.Kind) type {
        return struct {
            fn apply(self_node: *DiffMpsNode) void {
                const pa = self_node.parents[0].?;
                const pb = self_node.parents[1].?;
                const out_total = self_node.total_elements();
                const a_total = pa.total_elements();
                const b_total = pb.total_elements();
                const go = buf_ptr(self_node.grad.?);
                const y = buf_ptr(self_node.data);
                const ad = buf_ptr(pa.data);
                const bd = buf_ptr(pb.data);
                if (pa.grad) |g| {
                    const ga = buf_ptr(g);
                    for (0..out_total) |i| {
                        const ai = i % a_total;
                        const bi = i % b_total;
                        ga[ai] += go[i] * kind.deriv_a(ad[ai], bd[bi], y[i]);
                    }
                }
                if (pb.grad) |g| {
                    const gb = buf_ptr(g);
                    for (0..out_total) |i| {
                        const ai = i % a_total;
                        const bi = i % b_total;
                        gb[bi] += go[i] * kind.deriv_b(ad[ai], bd[bi], y[i]);
                    }
                }
            }
        };
    }

    pub fn gelu(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n: u32 = @intCast(x.total_elements());
        const out_buf = self.alloc_buf(n);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_gelu_forward(gpu.enc, x.data, out_buf, n);
        end_gpu(gpu);
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &unary_backward(unary.Gelu).apply;
        }
        return node;
    }

    pub fn silu(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n: u32 = @intCast(x.total_elements());
        const out_buf = self.alloc_buf(n);
        // dispatchSiluForward は sig_buf にも書き込むが、共通 backward (unary.Silu) は
        // 内部で sig を再計算するため、ここで確保した sig_buf は forward 後は未参照。
        // (arena が zeroGrad() で回収する。将来 sig 不要な Metal kernel を追加すれば
        // この無駄な GPU write を除去できる。)
        const sig_buf = self.alloc_buf(n);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_silu_forward(gpu.enc, x.data, out_buf, sig_buf, n);
        end_gpu(gpu);
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &unary_backward(unary.Silu).apply;
        }
        return node;
    }

    pub fn relu(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n: u32 = @intCast(x.total_elements());
        const out_buf = self.alloc_buf(n);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_relu_forward(gpu.enc, x.data, out_buf, n);
        end_gpu(gpu);
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &unary_backward(unary.Relu).apply;
        }
        return node;
    }

    pub fn tanh(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n: u32 = @intCast(x.total_elements());
        const out_buf = self.alloc_buf(n);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_tanh_forward(gpu.enc, x.data, out_buf, n);
        end_gpu(gpu);
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &unary_backward(unary.Tanh).apply;
        }
        return node;
    }

    pub fn softmax(self: *DiffMpsRuntime, x: DiffMpsTensor, axis: i64) DiffMpsTensor {
        _ = axis; // assume last axis (2D)
        const rows: u32 = @intCast(x.num_rows());
        const cols: u32 = @intCast(x.last_dim());
        const n = x.total_elements();
        const out_buf = self.alloc_buf(n);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_softmax_f32(gpu.enc, x.data, out_buf, rows, cols);
        end_gpu(gpu);
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backward_softmax;
        }
        return node;
    }

    fn backward_softmax(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const rows = pa.num_rows();
            const cols = pa.last_dim();
            softmax_common.softmax_backward(
                buf_ptr(ga_buf),
                buf_ptr(self_node.grad.?),
                buf_ptr(self_node.data),
                rows,
                cols,
            );
        }
    }

    // ── CPU fallback unary ops (数式は diff/common/unary.zig) ──

    pub fn negative(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        return self.unary_kind_cpu(unary.Negative, x);
    }

    pub fn square(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        return self.unary_kind_cpu(unary.Square, x);
    }

    pub fn sigmoid(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        return self.unary_kind_cpu(unary.Sigmoid, x);
    }

    // ════════════════════════════════════════════════════════════════
    // Binary ops
    // ════════════════════════════════════════════════════════════════

    pub fn add(self: *DiffMpsRuntime, a: DiffMpsTensor, b: DiffMpsTensor) DiffMpsTensor {
        const a_total = a.total_elements();
        const b_total = b.total_elements();

        // Same-shape: GPU forward (dispatchAddF32) + 共通 backward
        if (a_total == b_total) {
            const out_buf = self.alloc_buf(a_total);
            const gpu = begin_gpu(self.metal_ctx);
            self.metal_ctx.dispatch_add_f32(gpu.enc, a.data, b.data, out_buf, @intCast(a_total));
            end_gpu(gpu);
            const rg = a.requires_grad or b.requires_grad;
            const node = self.make_node(out_buf, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &binary_backward(binary.Add).apply;
            }
            return node;
        }

        // Broadcast: CPU fallback via共通 driver
        return self.binary_kind_cpu(binary.Add, a, b);
    }

    pub fn mul(self: *DiffMpsRuntime, a: DiffMpsTensor, b: DiffMpsTensor) DiffMpsTensor {
        return self.binary_kind_cpu(binary.Mul, a, b);
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
            const out_buf = self.alloc_buf(M * N);
            const gpu = begin_gpu(self.metal_ctx);
            self.metal_ctx.dispatch_matmul_f32(gpu.enc, a.data, b.data, out_buf, M, K, N);
            end_gpu(gpu);
            const node = self.make_node(out_buf, &.{ M, N }, rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backward_matmul2d;
            }
            return node;
        }
        @panic("matmul: only 2D supported in MPS runtime");
    }

    fn backward_matmul2d(self_node: *DiffMpsNode) void {
        // 数式は diff/common/matmul.zig:
        //   ga = go @ b^T,  gb = a^T @ go
        // UMA buf を [*]f32 に変換し、CPU 側 BLAS (Accelerate) で計算。
        // 従来の手書き三重ループより高速。
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const M = pa.shape[0];
        const K = pa.shape[1];
        const N = pb.shape[1];
        matmul_common.backward2d(
            buf_ptr(self_node.grad.?),
            buf_ptr(pa.data),
            buf_ptr(pb.data),
            if (pa.grad) |g| buf_ptr(g) else null,
            if (pb.grad) |g| buf_ptr(g) else null,
            M,
            K,
            N,
        );
    }

    // ════════════════════════════════════════════════════════════════
    // Shape ops (CPU via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn reshape(
        self: *DiffMpsRuntime,
        x: DiffMpsTensor,
        new_shape: []const usize,
    ) DiffMpsTensor {
        const node = self.make_node(x.data, new_shape, x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backward_reshape;
        }
        return node;
    }

    fn backward_reshape(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const total = self_node.total_elements();
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(ga_buf);
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
            const out_buf = self.alloc_buf(total);
            const src = buf_ptr(x.data);
            const dst = buf_ptr(out_buf);
            for (0..B) |b| {
                for (0..R) |i| {
                    for (0..C) |j| {
                        dst[b * C * R + j * R + i] = src[b * R * C + i * C + j];
                    }
                }
            }
            const node = self.make_node(out_buf, &.{ B, C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backward_transpose3d;
            }
            return node;
        }
        if (x.ndim == 2 and dim1 == 0 and dim2 == 1) {
            const R = x.shape[0];
            const C = x.shape[1];
            const out_buf = self.alloc_buf(R * C);
            const src = buf_ptr(x.data);
            const dst = buf_ptr(out_buf);
            for (0..R) |i| {
                for (0..C) |j| dst[j * R + i] = src[i * C + j];
            }
            const node = self.make_node(out_buf, &.{ C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backward_transpose2d;
            }
            return node;
        }
        @panic("transpose: unsupported ndim");
    }

    fn backward_transpose3d(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(ga_buf);
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

    fn backward_transpose2d(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(ga_buf);
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

    pub fn reduction_sum(self: *DiffMpsRuntime, x: DiffMpsTensor, axis: i64) DiffMpsTensor {
        const actual_axis: usize = if (axis < 0)
            @intCast(@as(i64, @intCast(x.ndim)) + axis)
        else
            @intCast(axis);

        if (x.ndim == 2) return self.reduction_sum2d(x, actual_axis);
        if (x.ndim == 1) return self.reduction_sum1d(x);
        if (x.ndim >= 3) return self.reduction_sum_nd(x, actual_axis);

        @panic("reductionSum: unsupported ndim/axis");
    }

    fn reduction_sum2d(self: *DiffMpsRuntime, x: DiffMpsTensor, actual_axis: usize) DiffMpsTensor {
        const rows = x.shape[0];
        const cols = x.shape[1];
        const src = buf_ptr(x.data);
        if (actual_axis == 1) {
            const out_buf = self.alloc_buf(rows);
            const dst = buf_ptr(out_buf);
            for (0..rows) |i| {
                var s: f32 = 0;
                for (0..cols) |j| s += src[i * cols + j];
                dst[i] = s;
            }
            const node = self.make_node(out_buf, &.{ rows, 1 }, x.requires_grad);
            if (x.requires_grad) self.set_reduce_backward(node, x, .axis1_2d, reduce.scale_sum());
            return node;
        } else {
            const out_buf = self.alloc_buf(cols);
            const dst = buf_ptr(out_buf);
            @memset(dst[0..cols], 0);
            for (0..rows) |i| {
                for (0..cols) |j| dst[j] += src[i * cols + j];
            }
            const node = self.make_node(out_buf, &.{ 1, cols }, x.requires_grad);
            if (x.requires_grad) self.set_reduce_backward(node, x, .axis0_2d, reduce.scale_sum());
            return node;
        }
    }

    fn reduction_sum1d(self: *DiffMpsRuntime, x: DiffMpsTensor) DiffMpsTensor {
        const n = x.total_elements();
        const src = buf_ptr(x.data);
        const out_buf = self.alloc_buf(1);
        var s: f32 = 0;
        for (0..n) |i| s += src[i];
        buf_ptr(out_buf)[0] = s;
        const node = self.make_node(out_buf, &.{1}, x.requires_grad);
        if (x.requires_grad) self.set_reduce_backward(node, x, .all, reduce.scale_sum());
        return node;
    }

    // ndim >= 3: flatten around the reduction axis
    fn reduction_sum_nd(self: *DiffMpsRuntime, x: DiffMpsTensor, actual_axis: usize) DiffMpsTensor {
        const total = x.total_elements();
        var before: usize = 1;
        for (0..actual_axis) |d| before *= x.shape[d];
        const axis_dim = x.shape[actual_axis];
        if (actual_axis == x.ndim - 1) {
            const flat = self.reshape(x, &.{ total / axis_dim, axis_dim });
            const reduced = self.reduction_sum(flat, 1);
            var new_shape: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
            for (0..x.ndim - 1) |d| new_shape[d] = x.shape[d];
            return self.reshape(reduced, new_shape[0..x.ndim]);
        } else if (actual_axis == 0) {
            const flat = self.reshape(x, &.{ axis_dim, total / axis_dim });
            const reduced = self.reduction_sum(flat, 0);
            var new_shape: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
            for (1..x.ndim) |d| new_shape[d] = x.shape[d];
            return self.reshape(reduced, new_shape[0..x.ndim]);
        } else {
            var after: usize = 1;
            for (actual_axis + 1..x.ndim) |d| after *= x.shape[d];
            const r3 = self.reshape(x, &.{ before, axis_dim, after });
            const t3 = self.transpose(r3, 1, 2);
            const flat = self.reshape(t3, &.{ before * after, axis_dim });
            const reduced = self.reduction_sum(flat, 1);
            var new_shape: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
            for (0..x.ndim) |d| new_shape[d] = x.shape[d];
            new_shape[actual_axis] = 1;
            return self.reshape(reduced, new_shape[0..x.ndim]);
        }
    }

    // ── Reduction backward 共通基盤 (数式は diff/common/reduce.zig) ──

    const ReduceContext = struct {
        case: reduce.Case,
        scale: f32,
    };

    fn set_reduce_backward(
        self: *DiffMpsRuntime,
        node: *DiffMpsNode,
        parent: DiffMpsTensor,
        case: reduce.Case,
        scale: f32,
    ) void {
        node.parents[0] = parent;
        const ctx = self.alloc_context(ReduceContext);
        ctx.* = .{ .case = case, .scale = scale };
        node.context = @ptrCast(ctx);
        node.backward_fn = &backward_reduction;
    }

    fn backward_reduction(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *const ReduceContext = @ptrCast(@alignCast(self_node.context.?));
            reduce.scatter(
                buf_ptr(ga_buf),
                buf_ptr(self_node.grad.?),
                pa.shape[0..pa.ndim],
                ctx.case,
                ctx.scale,
            );
        }
    }

    pub fn reduction_mean(self: *DiffMpsRuntime, x: DiffMpsTensor, axis: i64) DiffMpsTensor {
        const actual_axis: usize = if (axis < 0)
            @intCast(@as(i64, @intCast(x.ndim)) + axis)
        else
            @intCast(axis);

        if (x.ndim == 2) {
            const rows = x.shape[0];
            const cols = x.shape[1];
            const src = buf_ptr(x.data);
            if (actual_axis == 1) {
                const out_buf = self.alloc_buf(rows);
                const dst = buf_ptr(out_buf);
                const cols_f: f32 = @floatFromInt(cols);
                for (0..rows) |i| {
                    var s: f32 = 0;
                    for (0..cols) |j| s += src[i * cols + j];
                    dst[i] = s / cols_f;
                }
                const node = self.make_node(out_buf, &.{ rows, 1 }, x.requires_grad);
                if (x.requires_grad)
                    self.set_reduce_backward(node, x, .axis1_2d, reduce.scale_mean(cols));
                return node;
            } else {
                const out_buf = self.alloc_buf(cols);
                const dst = buf_ptr(out_buf);
                const rows_f: f32 = @floatFromInt(rows);
                @memset(dst[0..cols], 0);
                for (0..rows) |i| {
                    for (0..cols) |j| dst[j] += src[i * cols + j];
                }
                for (0..cols) |j| dst[j] /= rows_f;
                const node = self.make_node(out_buf, &.{ 1, cols }, x.requires_grad);
                if (x.requires_grad)
                    self.set_reduce_backward(node, x, .axis0_2d, reduce.scale_mean(rows));
                return node;
            }
        }
        @panic("reductionMean: only 2D supported");
    }

    // ════════════════════════════════════════════════════════════════
    // LayerNorm (Metal forward, CPU backward via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn layer_norm(
        self: *DiffMpsRuntime,
        x: DiffMpsTensor,
        gamma: DiffMpsTensor,
        beta: DiffMpsTensor,
        eps: f32,
        axis: i64,
    ) DiffMpsTensor {
        _ = axis;
        const rows: u32 = @intCast(x.num_rows());
        const cols: u32 = @intCast(x.last_dim());
        const out_buf = self.alloc_buf(rows * cols);
        const mean_buf = self.alloc_buf(rows);
        const inv_std_buf = self.alloc_buf(rows);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_layer_norm_forward(
            gpu.enc,
            x.data,
            gamma.data,
            beta.data,
            out_buf,
            mean_buf,
            inv_std_buf,
            rows,
            cols,
            eps,
        );
        end_gpu(gpu);
        const rg = x.requires_grad or gamma.requires_grad or beta.requires_grad;
        const node = self.make_node(out_buf, x.shape[0..x.ndim], rg);
        if (rg) {
            node.parents[0] = x;
            node.parents[1] = gamma;
            node.parents[2] = beta;
            const ctx = self.alloc_context(LayerNormCtx);
            ctx.* = .{ .mean_buf = mean_buf, .inv_std_buf = inv_std_buf };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_layer_norm;
        }
        return node;
    }

    const LayerNormCtx = struct {
        mean_buf: id,
        inv_std_buf: id,
    };

    fn backward_layer_norm(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?; // x
        const pg = self_node.parents[1].?; // gamma
        const pb = self_node.parents[2].?; // beta
        const ctx: *LayerNormCtx = @ptrCast(@alignCast(self_node.context.?));
        const rows = pa.num_rows();
        const cols = pa.last_dim();
        const go = buf_ptr(self_node.grad.?);
        const x_data = buf_ptr(pa.data);
        const gamma_data = buf_ptr(pg.data);
        const mean = buf_ptr(ctx.mean_buf);
        const inv_std = buf_ptr(ctx.inv_std_buf);

        // dBeta, dGamma
        if (pb.grad) |g| {
            const g_beta = buf_ptr(g);
            for (0..rows) |r| {
                for (0..cols) |c| g_beta[c] += go[r * cols + c];
            }
        }
        if (pg.grad) |g| {
            const g_gamma = buf_ptr(g);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    const xhat = (x_data[r * cols + c] - mean[r]) * inv_std[r];
                    g_gamma[c] += go[r * cols + c] * xhat;
                }
            }
        }
        // dX
        if (pa.grad) |g| {
            const ga = buf_ptr(g);
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
                    const dx = inv_std[r] *
                        (go[r * cols + c] * gamma_data[c] -
                            sum_go_gamma / cols_f -
                            xhat * sum_go_gamma_xhat / cols_f);
                    ga[r * cols + c] += dx;
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // RMSNorm: y = x * inv_rms * weight, inv_rms = 1/sqrt(mean(x²) + eps)
    // ════════════════════════════════════════════════════════════════

    pub fn rms_norm(
        self: *DiffMpsRuntime,
        x: DiffMpsTensor,
        weight: DiffMpsTensor,
        eps: f32,
    ) DiffMpsTensor {
        const rows: u32 = @intCast(x.num_rows());
        const cols: u32 = @intCast(x.last_dim());
        const n = x.total_elements();
        const out_buf = self.alloc_buf(n);
        const inv_rms_buf = self.alloc_buf(rows);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_rms_norm_forward_training(
            gpu.enc,
            x.data,
            weight.data,
            out_buf,
            inv_rms_buf,
            rows,
            cols,
            eps,
        );
        end_gpu(gpu);
        const rg = x.requires_grad or weight.requires_grad;
        const node = self.make_node(out_buf, x.shape[0..x.ndim], rg);
        if (rg) {
            node.parents[0] = x;
            node.parents[1] = weight;
            const ctx = self.alloc_context(RmsNormCtx);
            ctx.* = .{ .inv_rms_buf = inv_rms_buf };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_rms_norm;
        }
        return node;
    }

    const RmsNormCtx = struct { inv_rms_buf: id };

    fn backward_rms_norm(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?; // x
        const pw = self_node.parents[1].?; // weight
        const ctx: *RmsNormCtx = @ptrCast(@alignCast(self_node.context.?));
        const rows = pa.num_rows();
        const cols = pa.last_dim();
        const go = buf_ptr(self_node.grad.?);
        const x_data = buf_ptr(pa.data);
        const w_data = buf_ptr(pw.data);
        const inv_rms = buf_ptr(ctx.inv_rms_buf);
        const cols_f: f32 = @floatFromInt(cols);

        // dWeight[c] = Σ_r go[r,c] * x[r,c] * inv_rms[r]
        if (pw.grad) |g| {
            const g_w = buf_ptr(g);
            for (0..rows) |r| {
                const s = inv_rms[r];
                for (0..cols) |c| {
                    g_w[c] += go[r * cols + c] * x_data[r * cols + c] * s;
                }
            }
        }
        // dX[r,c] = inv_rms[r] *
        //           (w[c] * go[r,c] - x[r,c] * inv_rms[r]² · (Σⱼ x[r,j]·w[j]·go[r,j]) / D)
        if (pa.grad) |g| {
            const ga = buf_ptr(g);
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

    pub fn causal_softmax(
        self: *DiffMpsRuntime,
        x: DiffMpsTensor,
        num_heads: u32,
        seq_len: u32,
    ) DiffMpsTensor {
        const rows: u32 = @intCast(x.num_rows());
        const cols: u32 = @intCast(x.last_dim());
        const n = x.total_elements();
        const out_buf = self.alloc_buf(n);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_causal_softmax_f32(
            gpu.enc,
            x.data,
            out_buf,
            rows,
            cols,
            num_heads,
            seq_len,
        );
        end_gpu(gpu);
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            // Masked positions have softmax output = 0, so standard softmax backward
            // naturally zeroes the gradient at those positions. Reuse existing kernel.
            node.backward_fn = &backward_softmax;
        }
        return node;
    }

    // ════════════════════════════════════════════════════════════════
    // Rotary Position Embedding (RoPE)
    // x shape: [seq_len, n_heads, 2*half_dim] (last dim is head_dim, must be even)
    // freqs shape: [half_dim] precomputed frequencies θᵢ = base^(-2i/d)
    // ════════════════════════════════════════════════════════════════

    pub fn rope(
        self: *DiffMpsRuntime,
        x: DiffMpsTensor,
        freqs: DiffMpsTensor,
        n_heads: u32,
        seq_len: u32,
        half_dim: u32,
    ) DiffMpsTensor {
        const n = x.total_elements();
        const out_buf = self.alloc_buf(n);
        // Metal RoPE kernel works in-place; copy x into out_buf first.
        @memcpy(buf_ptr(out_buf)[0..n], buf_ptr(x.data)[0..n]);
        const sin_cache = self.alloc_buf(seq_len * half_dim);
        const cos_cache = self.alloc_buf(seq_len * half_dim);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_rope_forward_training(
            gpu.enc,
            out_buf,
            freqs.data,
            sin_cache,
            cos_cache,
            seq_len,
            n_heads,
            half_dim,
        );
        end_gpu(gpu);
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(RopeCtx);
            ctx.* = .{
                .sin_cache = sin_cache,
                .cos_cache = cos_cache,
                .seq_len = seq_len,
                .n_heads = n_heads,
                .half_dim = half_dim,
            };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_rope;
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

    // ════════════════════════════════════════════════════════════════
    // mulScalar: y = x * c (compile-time constant multiplier)
    // ════════════════════════════════════════════════════════════════

    pub fn mul_scalar(self: *DiffMpsRuntime, x: DiffMpsTensor, c: f32) DiffMpsTensor {
        const n = x.total_elements();
        const out_buf = self.alloc_buf(n);
        const src = buf_ptr(x.data);
        const dst = buf_ptr(out_buf);
        for (0..n) |i| dst[i] = src[i] * c;
        const node = self.make_node(out_buf, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(MulScalarCtx);
            ctx.* = .{ .c = c };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_mul_scalar;
        }
        return node;
    }

    const MulScalarCtx = struct { c: f32 };

    fn backward_mul_scalar(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |g| {
            const ctx: *MulScalarCtx = @ptrCast(@alignCast(self_node.context.?));
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(g);
            const total = self_node.total_elements();
            for (0..total) |i| ga[i] += go[i] * ctx.c;
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Quantized matmul (no gradient on weights: frozen base for QLoRA)
    // x: [..., in_dim], quant_weight: [out_dim, in_dim] (Q4_0/Q4_1/Q8_0)
    // out: [..., out_dim]
    // ════════════════════════════════════════════════════════════════

    pub const QuantWeight = struct {
        buf: id, // GPU-side quantized bytes (MTLBuffer)
        quant_type: metal.QuantType,
        out_dim: u32,
        in_dim: u32,
    };

    pub fn quant_matmul_no_grad(
        self: *DiffMpsRuntime,
        x: DiffMpsTensor,
        qw: *const QuantWeight,
    ) DiffMpsTensor {
        const m: u32 = @intCast(x.num_rows());
        std.debug.assert(@as(u32, @intCast(x.last_dim())) == qw.in_dim);
        const total_out: usize = @as(usize, m) * @as(usize, qw.out_dim);
        const out_buf = self.alloc_buf(total_out);
        const gpu = begin_gpu(self.metal_ctx);
        self.metal_ctx.dispatch_matmul_batched(
            gpu.enc,
            qw.buf,
            x.data,
            out_buf,
            qw.out_dim,
            qw.in_dim,
            m,
            qw.quant_type,
        );
        end_gpu(gpu);

        var new_shape: [MAX_NDIM]usize = .{ 1, 1, 1, 1 };
        for (0..x.ndim - 1) |i| new_shape[i] = x.shape[i];
        new_shape[x.ndim - 1] = @as(usize, qw.out_dim);
        const node = self.make_node(out_buf, new_shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.alloc_context(QuantMatmulCtx);
            ctx.* = .{ .metal_ctx = self.metal_ctx, .qw = qw };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_quant_matmul;
        }
        return node;
    }

    const QuantMatmulCtx = struct {
        metal_ctx: *MetalContext,
        qw: *const QuantWeight,
    };

    fn backward_quant_matmul(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *QuantMatmulCtx = @ptrCast(@alignCast(self_node.context.?));
            const m: u32 = @intCast(pa.num_rows());
            const gpu = begin_gpu(ctx.metal_ctx);
            ctx.metal_ctx.dispatch_quant_trans_batched(
                gpu.enc,
                ctx.qw.buf,
                self_node.grad.?,
                ga_buf,
                ctx.qw.out_dim,
                ctx.qw.in_dim,
                m,
                ctx.qw.quant_type,
            );
            end_gpu(gpu);
        }
    }

    fn backward_rope(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *RopeCtx = @ptrCast(@alignCast(self_node.context.?));
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(ga_buf);
            const sin_c = buf_ptr(ctx.sin_cache);
            const cos_c = buf_ptr(ctx.cos_cache);
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
        const out_buf = self.alloc_buf(num_indices * embed_dim);
        const t_data = buf_ptr(table.data);
        const o_data = buf_ptr(out_buf);
        for (0..num_indices) |i| {
            const row = indices[i];
            @memcpy(
                o_data[i * embed_dim ..][0..embed_dim],
                t_data[row * embed_dim ..][0..embed_dim],
            );
        }
        const node = self.make_node(out_buf, &.{ num_indices, embed_dim }, table.requires_grad);
        if (table.requires_grad) {
            node.parents[0] = table;
            const ctx = self.alloc_context(GatherCtx);
            ctx.* = .{ .indices = indices, .num_indices = num_indices };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_gather;
        }
        return node;
    }

    const GatherCtx = struct {
        indices: []const u32,
        num_indices: usize,
    };

    fn backward_gather(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *GatherCtx = @ptrCast(@alignCast(self_node.context.?));
            const embed_dim = pa.shape[1];
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(ga_buf);
            for (0..ctx.num_indices) |i| {
                const row = ctx.indices[i];
                for (0..embed_dim) |j| ga[row * embed_dim + j] += go[i * embed_dim + j];
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Loss functions (CPU via UMA)
    // ════════════════════════════════════════════════════════════════

    pub fn mse_loss(self: *DiffMpsRuntime, pred: DiffMpsTensor, target: []const f32) DiffMpsTensor {
        const total = pred.total_elements();
        const pd = buf_ptr(pred.data);
        var sum_sq: f32 = 0;
        for (0..total) |i| {
            const diff = pd[i] - target[i];
            sum_sq += diff * diff;
        }
        const n_f: f32 = @floatFromInt(total);
        const out_buf = self.alloc_buf(1);
        buf_ptr(out_buf)[0] = sum_sq / n_f;
        const node = self.make_node(out_buf, &.{1}, pred.requires_grad);
        if (pred.requires_grad) {
            node.parents[0] = pred;
            const ctx = self.alloc_context(MseLossCtx);
            ctx.* = .{ .target = target };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_mse_loss;
        }
        return node;
    }

    const MseLossCtx = struct { target: []const f32 };

    fn backward_mse_loss(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *MseLossCtx = @ptrCast(@alignCast(self_node.context.?));
            const total = pa.total_elements();
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(ga_buf);
            const pd = buf_ptr(pa.data);
            const n_f: f32 = @floatFromInt(total);
            const scale = go[0] * 2.0 / n_f;
            for (0..total) |i| ga[i] += scale * (pd[i] - ctx.target[i]);
        }
    }

    pub fn cross_entropy_loss_with_indices(
        self: *DiffMpsRuntime,
        logits: DiffMpsTensor,
        indices: []const u32,
    ) DiffMpsTensor {
        const batch = logits.shape[0];
        const num_classes = logits.shape[1];
        const ld = buf_ptr(logits.data);

        const softmax_cache = self.alloc_data(batch * num_classes);
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
        const out_buf = self.alloc_buf(1);
        buf_ptr(out_buf)[0] = total_loss / batch_f;
        const node = self.make_node(out_buf, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.alloc_context(CECtx);
            ctx.* = .{ .softmax_cache = softmax_cache, .indices = indices };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_cross_entropy;
        }
        return node;
    }

    const CECtx = struct { softmax_cache: []f32, indices: []const u32 };

    fn backward_cross_entropy(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *CECtx = @ptrCast(@alignCast(self_node.context.?));
            const batch = pa.shape[0];
            const num_classes = pa.shape[1];
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(ga_buf);
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

    pub fn bce_loss_with_logits(
        self: *DiffMpsRuntime,
        logits: DiffMpsTensor,
        target: []const f32,
    ) DiffMpsTensor {
        const total = logits.total_elements();
        const ld = buf_ptr(logits.data);
        var loss_sum: f32 = 0;
        for (0..total) |i| {
            const x = ld[i];
            const t = target[i];
            const pos_part: f32 = if (x > 0) x else 0;
            loss_sum += pos_part - x * t + @log(1.0 + @exp(-@abs(x)));
        }
        const n_f: f32 = @floatFromInt(total);
        const out_buf = self.alloc_buf(1);
        buf_ptr(out_buf)[0] = loss_sum / n_f;
        const node = self.make_node(out_buf, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.alloc_context(BceLossCtx);
            ctx.* = .{ .target = target };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backward_bce_loss;
        }
        return node;
    }

    const BceLossCtx = struct { target: []const f32 };

    fn backward_bce_loss(self_node: *DiffMpsNode) void {
        const pa = self_node.parents[0].?;
        if (pa.grad) |ga_buf| {
            const ctx: *BceLossCtx = @ptrCast(@alignCast(self_node.context.?));
            const total = pa.total_elements();
            const go = buf_ptr(self_node.grad.?);
            const ga = buf_ptr(ga_buf);
            const ld = buf_ptr(pa.data);
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
            loss.grad = self.alloc_buf(loss.total_elements());
        }
        const loss_total = loss.total_elements();
        const lg = buf_ptr(loss.grad.?);
        for (0..loss_total) |i| lg[i] = 1.0;

        // 2. Topological sort
        self.topo_buf.clearRetainingCapacity();
        diff_node.topo_sort(DiffMpsNode, loss, &self.topo_buf, self.allocator);

        // 3. Allocate grad buffers for intermediate nodes (UMA: zeroed MTLBuffer)
        for (self.topo_buf.items) |node| {
            if (node.grad == null and node.requires_grad) {
                node.grad = self.alloc_buf_zeroed(node.total_elements());
            }
        }

        // 4-5. Reverse traversal + reset visited
        diff_node.backward_pass(DiffMpsNode, &self.topo_buf, self.param_nodes);
    }

    // ════════════════════════════════════════════════════════════════
    // Optimizer step (UMA: compute.adamStep directly on MTLBuffer contents)
    // ════════════════════════════════════════════════════════════════

    pub fn apply_adam(
        self: *DiffMpsRuntime,
        adam: *compute.AdamState,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) void {
        adam.step += 1;
        for (self.param_nodes, 0..) |*node, i| {
            const size = self.module.param_size(.{ .index = i });
            const p = buf_ptr(node.data)[0..size];
            const g = buf_ptr(self.param_grad_bufs[i])[0..size];
            compute.adam_step(p, g, adam.m[i], adam.v[i], lr, beta1, beta2, eps, wd, adam.step);
        }
    }
};
