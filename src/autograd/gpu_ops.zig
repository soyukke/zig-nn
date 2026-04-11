/// GPU 版の微分可能な演算
///
/// 各演算は Metal GPU カーネルで forward 計算を行い、
/// backward_fn 内でも GPU カーネルを使って勾配を計算する。
/// context 構造体に MetalContext ポインタを埋め込むことで、
/// backward_fn (fn(*GraphNode) void) から GPU にアクセスする。
///
/// Phase 1: dispatch ごとに commit/wait (正しさ優先)
const std = @import("std");
const Allocator = std.mem.Allocator;
const GraphNodeMod = @import("../core/graph.zig");
const metal = @import("../backend/metal.zig");
const gpu_var = @import("gpu_variable.zig");

const MetalContext = metal.MetalContext;
const id = metal.id;

/// GPU 上で計算した中間結果を持つ一時変数
/// forward で生成、backward で勾配伝播
pub fn GpuResult(comptime T: type, comptime n: usize) type {
    const Node = GraphNodeMod.GraphNode(T);

    return struct {
        const Self = @This();
        pub const num_elements = n;

        data: [*]T,
        data_buf: id,
        grad_buf: ?id,
        node: *Node,
        metal_ctx: *MetalContext,
    };
}

pub const GpuProfileCat = enum {
    batched_matmul,
    layernorm,
    softmax,
    gelu_tanh,
    embedding,
    add_bias,
    scale,
    concat,
    loss,
    other,
};

/// プロファイルモード: 現在のカテゴリの dispatches を flush して GPU 完了待ち
fn profileFlushCategory(mtl: *MetalContext) void {
    mtl.profileFlush();
}

/// GPU コマンド実行ヘルパー
/// バッチモード時は共有エンコーダを使い、commit/wait をスキップ
fn gpuExec(mtl: *MetalContext, dispatch_fn: anytype, args: anytype) void {
    gpuExecCat(mtl, dispatch_fn, args, .other);
}

fn gpuExecCat(mtl: *MetalContext, dispatch_fn: anytype, args: anytype, cat: GpuProfileCat) void {
    if (mtl.profile_mode and mtl.batch_encoder != null) {
        const cat_id = @intFromEnum(cat);
        // カテゴリが変わったら前のカテゴリを flush して計測
        if (mtl.profile_current_cat != cat_id) {
            profileFlushCategory(mtl);
            mtl.profile_current_cat = cat_id;
            mtl.profile_timer = std.time.Timer.start() catch null;
        }
        // Batch mode と同様に共有エンコーダに dispatch
        @call(.auto, dispatch_fn, .{mtl, mtl.batch_encoder.?} ++ args);
        MetalContext.memoryBarrier(mtl.batch_encoder.?);
        mtl.profile_stats.addCount(cat, 1);
    } else if (mtl.batch_encoder) |encoder| {
        // Batch mode: 共有エンコーダに dispatch + barrier のみ
        @call(.auto, dispatch_fn, .{mtl, encoder} ++ args);
        MetalContext.memoryBarrier(encoder);
    } else {
        // Non-batch mode: per-dispatch commit/wait
        const cmd_buf = mtl.newCommandBuffer();
        const encoder = MetalContext.newComputeEncoder(cmd_buf);
        @call(.auto, dispatch_fn, .{mtl, encoder} ++ args);
        MetalContext.memoryBarrier(encoder);
        MetalContext.endEncoding(encoder);
        MetalContext.commit(cmd_buf);
        MetalContext.waitUntilCompleted(cmd_buf);
    }
}

// ============================================================
// matmul: z = a @ b
// A: (M, K), B: (K, N) => Z: (M, N)
// ============================================================

pub fn matmul(
    comptime T: type,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    _: [*]T,
    a_buf: id,
    a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    b_buf: id,
    b_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, M * N) {
    const Node = GraphNodeMod.GraphNode(T);
    const result_n = M * N;

    // Forward: C = A @ B on GPU
    const c_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    const c_data = MetalContext.bufferContents(T, c_buf);

    // C(M,N) = A(M,K) × B(K,N)
    mtl.dispatchMPSMatmul(a_buf, b_buf, c_buf, @as(u32, M), @as(u32, N), @as(u32, K), false, false, 1.0, 0.0);

    // Context for backward
    const Context = struct {
        a_buf: id,
        b_buf: id,
        mtl: *MetalContext,
        a_node: *Node,
        b_node: *Node,
        grad_out_buf: id,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{
        .a_buf = a_buf,
        .b_buf = b_buf,
        .mtl = mtl,
        .a_node = a_node,
        .b_node = b_node,
        .grad_out_buf = undefined,
    };

    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);
    node.parents = .{ a_node, b_node, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), result_n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), M * K * @sizeOf(T));
                        // dA += dC × B^T
                        m.dispatchMPSMatmul(go_buf, context.b_buf, ga_buf, @as(u32, M), @as(u32, K), @as(u32, N), false, true, 1.0, 1.0);
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        const gb_buf = m.getOrAllocGradBuf(@ptrCast(pb), K * N * @sizeOf(T));
                        // dB(K,N) += A^T(K,M) × dC(M,N)
                        m.dispatchMPSMatmul(context.a_buf, go_buf, gb_buf, @as(u32, K), @as(u32, N), @as(u32, M), true, false, 1.0, 1.0);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;

            const grad_out_buf = m.createBuffer(result_n * @sizeOf(T)) catch return;
            const grad_out_ptr = MetalContext.bufferContents(T, grad_out_buf);
            @memcpy(grad_out_ptr[0..result_n], grad_out);

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const grad_a_buf = m.createBuffer(M * K * @sizeOf(T)) catch return;
                        const grad_a_ptr = MetalContext.bufferContents(T, grad_a_buf);
                        @memcpy(grad_a_ptr[0 .. M * K], g);

                        // dA += dC × B^T
                        m.dispatchMPSMatmul(grad_out_buf, context.b_buf, grad_a_buf, @as(u32, M), @as(u32, K), @as(u32, N), false, true, 1.0, 1.0);

                        @memcpy(g, grad_a_ptr[0 .. M * K]);
                        metal.objRelease(grad_a_buf);
                    }
                }
            }

            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |g| {
                        const grad_b_buf = m.createBuffer(K * N * @sizeOf(T)) catch return;
                        const grad_b_ptr = MetalContext.bufferContents(T, grad_b_buf);
                        @memcpy(grad_b_ptr[0 .. K * N], g);

                        // dB(K,N) += A^T(K,M) × dC(M,N)
                        m.dispatchMPSMatmul(context.a_buf, grad_out_buf, grad_b_buf, @as(u32, K), @as(u32, N), @as(u32, M), true, false, 1.0, 1.0);

                        @memcpy(g, grad_b_ptr[0 .. K * N]);
                        metal.objRelease(grad_b_buf);
                    }
                }
            }

            metal.objRelease(grad_out_buf);
        }
    }.backward;

    return .{
        .data = c_data,
        .data_buf = c_buf,
        .grad_buf = null,
        .node = node,
        .metal_ctx = mtl,
    };
}

// ============================================================
// addBias: z[i,j] = a[i,j] + bias[j]
// a: (rows, cols), bias: (cols) => z: (rows, cols)
// ============================================================

pub fn addBias(
    comptime T: type,
    comptime rows: usize,
    comptime cols: usize,
    _: [*]T,
    a_buf: id,
    a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    bias_buf: id,
    bias_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, rows * cols) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = rows * cols;

    // Forward
    const z_buf = try mtl.createBuffer(n * @sizeOf(T));
    const z_data = MetalContext.bufferContents(T, z_buf);

    gpuExecCat(mtl, MetalContext.dispatchAddBiasF32, .{
        a_buf, bias_buf, z_buf,
        @as(u32, rows), @as(u32, cols),
    }, .add_bias);

    const Context = struct {
        mtl: *MetalContext,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents = .{ a_node, bias_node, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        // addBias の入力勾配は恒等 (da = dz): バッファエイリアスで dispatch 不要
                        if (!m.tryAliasGradBuf(@ptrCast(pa), go_buf)) {
                            const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                            gpuExecCat(m, MetalContext.dispatchAddBackwardAccum, .{ go_buf, ga_buf, @as(u32, n) }, .add_bias);
                        }
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        const gb_buf = m.getOrAllocGradBuf(@ptrCast(pb), cols * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchAddBiasBackward, .{ go_buf, gb_buf, @as(u32, rows), @as(u32, cols) }, .add_bias);
                    }
                }
                return;
            }

            const grad_out = self.grad orelse return;
            const grad_out_buf = m.createBuffer(n * @sizeOf(T)) catch return;
            const grad_out_ptr = MetalContext.bufferContents(T, grad_out_buf);
            @memcpy(grad_out_ptr[0..n], grad_out);

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const ga_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const ga_ptr = MetalContext.bufferContents(T, ga_buf);
                        @memcpy(ga_ptr[0..n], g);
                        gpuExecCat(m, MetalContext.dispatchAddBackwardAccum, .{ grad_out_buf, ga_buf, @as(u32, n) }, .add_bias);
                        @memcpy(g, ga_ptr[0..n]);
                        metal.objRelease(ga_buf);
                    }
                }
            }
            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |g| {
                        const gb_buf = m.createBuffer(cols * @sizeOf(T)) catch return;
                        const gb_ptr = MetalContext.bufferContents(T, gb_buf);
                        @memcpy(gb_ptr[0..cols], g);
                        gpuExecCat(m, MetalContext.dispatchAddBiasBackward, .{ grad_out_buf, gb_buf, @as(u32, rows), @as(u32, cols) }, .add_bias);
                        @memcpy(g, gb_ptr[0..cols]);
                        metal.objRelease(gb_buf);
                    }
                }
            }
            metal.objRelease(grad_out_buf);
        }
    }.backward;

    return .{
        .data = z_data,
        .data_buf = z_buf,
        .grad_buf = null,
        .node = node,
        .metal_ctx = mtl,
    };
}

// ============================================================
// add: z = a + b (element-wise)
// ============================================================

pub fn add(
    comptime T: type,
    comptime n: usize,
    _: [*]T,
    a_buf: id,
    a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    b_buf: id,
    b_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, n) {
    const Node = GraphNodeMod.GraphNode(T);

    const c_buf = try mtl.createBuffer(n * @sizeOf(T));
    const c_data = MetalContext.bufferContents(T, c_buf);

    gpuExecCat(mtl, MetalContext.dispatchAddF32, .{
        a_buf, b_buf, c_buf, @as(u32, n),
    }, .add_bias);

    const Context = struct {
        mtl: *MetalContext,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents = .{ a_node, b_node, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                // add の勾配は恒等 (da = dz, db = dz): 1つ目の親にはバッファエイリアス
                var aliased = false;
                inline for (0..2) |pi| {
                    if (self.parents[pi]) |pa| {
                        if (pa.requires_grad) {
                            if (!aliased and m.tryAliasGradBuf(@ptrCast(pa), go_buf)) {
                                aliased = true;
                            } else {
                                const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                                gpuExecCat(m, MetalContext.dispatchAddBackwardAccum, .{ go_buf, ga_buf, @as(u32, n) }, .add_bias);
                            }
                        }
                    }
                }
                return;
            }

            const grad_out = self.grad orelse return;
            const grad_out_buf = m.createBuffer(n * @sizeOf(T)) catch return;
            const grad_out_ptr = MetalContext.bufferContents(T, grad_out_buf);
            @memcpy(grad_out_ptr[0..n], grad_out);

            inline for (0..2) |pi| {
                if (self.parents[pi]) |pa| {
                    if (pa.requires_grad) {
                        if (pa.grad) |g| {
                            const ga_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                            const ga_ptr = MetalContext.bufferContents(T, ga_buf);
                            @memcpy(ga_ptr[0..n], g);
                            gpuExecCat(m, MetalContext.dispatchAddBackwardAccum, .{ grad_out_buf, ga_buf, @as(u32, n) }, .add_bias);
                            @memcpy(g, ga_ptr[0..n]);
                            metal.objRelease(ga_buf);
                        }
                    }
                }
            }
            metal.objRelease(grad_out_buf);
        }
    }.backward;

    return .{
        .data = c_data,
        .data_buf = c_buf,
        .grad_buf = null,
        .node = node,
        .metal_ctx = mtl,
    };
}

// ============================================================
// silu: z = x * sigmoid(x)
// ============================================================

pub fn silu(
    comptime T: type,
    comptime n: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, n) {
    const Node = GraphNodeMod.GraphNode(T);

    // Forward: out = x * sigmoid(x), sig_out = sigmoid(x)
    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const sig_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExec(mtl, MetalContext.dispatchSiluForward, .{
        x_buf, out_buf, sig_buf, @as(u32, n),
    });

    const Context = struct {
        x_buf: id,
        sig_buf: id,
        mtl: *MetalContext,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .x_buf = x_buf, .sig_buf = sig_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExec(m, MetalContext.dispatchSiluBackward, .{ go_buf, context.x_buf, context.sig_buf, gx_buf, @as(u32, n) });
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const grad_out_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const grad_out_ptr = MetalContext.bufferContents(T, grad_out_buf);
                        @memcpy(grad_out_ptr[0..n], grad_out);

                        const grad_x_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const grad_x_ptr = MetalContext.bufferContents(T, grad_x_buf);
                        @memcpy(grad_x_ptr[0..n], g);

                        gpuExec(m, MetalContext.dispatchSiluBackward, .{
                            grad_out_buf, context.x_buf, context.sig_buf, grad_x_buf,
                            @as(u32, n),
                        });

                        @memcpy(g, grad_x_ptr[0..n]);
                        metal.objRelease(grad_out_buf);
                        metal.objRelease(grad_x_buf);
                    }
                }
            }
        }
    }.backward;

    return .{
        .data = out_data,
        .data_buf = out_buf,
        .grad_buf = null,
        .node = node,
        .metal_ctx = mtl,
    };
}

// ============================================================
// mseLoss: L = mean((pred - target)^2)
// ============================================================

pub fn mseLoss(
    comptime T: type,
    comptime n: usize,
    _: [*]T,
    pred_buf: id,
    pred_node: *GraphNodeMod.GraphNode(T),
    target_slice: []const T,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, 1) {
    const Node = GraphNodeMod.GraphNode(T);

    // target を GPU バッファにコピー
    const target_buf = try mtl.createBuffer(n * @sizeOf(T));
    const target_ptr = MetalContext.bufferContents(T, target_buf);
    @memcpy(target_ptr[0..n], target_slice);

    // diff = pred - target
    const diff_buf = try mtl.createBuffer(n * @sizeOf(T));
    gpuExecCat(mtl, MetalContext.dispatchMseLossDiff, .{
        pred_buf, target_buf, diff_buf, @as(u32, n),
    }, .loss);

    // loss = mean(diff^2)
    const loss_buf = try mtl.createBuffer(1 * @sizeOf(T));
    gpuExecCat(mtl, MetalContext.dispatchMseLossReduce, .{
        diff_buf, loss_buf, @as(u32, n),
    }, .loss);

    const loss_data = MetalContext.bufferContents(T, loss_buf);

    const Context = struct {
        diff_buf: id,
        target_buf: id, // batch mode で GPU 実行前に解放されるのを防ぐ
        mtl: *MetalContext,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .diff_buf = diff_buf, .target_buf = target_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(1, true);
    node.parents[0] = pred_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), 1 * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gp_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchMseLossBackward, .{ go_buf, context.diff_buf, gp_buf, @as(u32, n) }, .loss);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        // grad_out を GPU バッファに
                        const grad_out_buf = m.createBuffer(1 * @sizeOf(T)) catch return;
                        const grad_out_ptr = MetalContext.bufferContents(T, grad_out_buf);
                        grad_out_ptr[0] = grad_out[0];

                        // grad_pred の現在値を GPU バッファにコピー
                        const grad_pred_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const grad_pred_ptr = MetalContext.bufferContents(T, grad_pred_buf);
                        @memcpy(grad_pred_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchMseLossBackward, .{
                            grad_out_buf, context.diff_buf, grad_pred_buf,
                            @as(u32, n),
                        }, .loss);

                        @memcpy(g, grad_pred_ptr[0..n]);
                        metal.objRelease(grad_out_buf);
                        metal.objRelease(grad_pred_buf);
                    }
                }
            }

            // forward 中間バッファを backward 完了後に解放
            metal.objRelease(context.diff_buf);
            metal.objRelease(context.target_buf);
        }
    }.backward;

    return .{
        .data = loss_data,
        .data_buf = loss_buf,
        .grad_buf = null,
        .node = node,
        .metal_ctx = mtl,
    };
}

// ============================================================
// Phase 2: Transformer ops
// ============================================================

// ============================================================
// relu: z = max(0, x)
// ============================================================

pub fn relu(
    comptime T: type,
    comptime n: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, n) {
    const Node = GraphNodeMod.GraphNode(T);

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExec(mtl, MetalContext.dispatchReluForward, .{
        x_buf, out_buf, @as(u32, n),
    });

    const Context = struct { x_buf: id, mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .x_buf = x_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExec(m, MetalContext.dispatchReluBackward, .{ context.x_buf, go_buf, gx_buf, @as(u32, n) });
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const go_ptr = MetalContext.bufferContents(T, go_buf);
                        @memcpy(go_ptr[0..n], grad_out);

                        const gx_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gx_ptr = MetalContext.bufferContents(T, gx_buf);
                        @memcpy(gx_ptr[0..n], g);

                        gpuExec(m, MetalContext.dispatchReluBackward, .{
                            context.x_buf, go_buf, gx_buf, @as(u32, n),
                        });

                        @memcpy(g, gx_ptr[0..n]);
                        metal.objRelease(go_buf);
                        metal.objRelease(gx_buf);
                    }
                }
            }
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// gelu: z = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// ============================================================

pub fn gelu(
    comptime T: type,
    comptime n: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, n) {
    const Node = GraphNodeMod.GraphNode(T);

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExecCat(mtl, MetalContext.dispatchGeluForward, .{
        x_buf, out_buf, @as(u32, n),
    }, .gelu_tanh);

    const Context = struct { x_buf: id, mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .x_buf = x_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchGeluBackward, .{ context.x_buf, go_buf, gx_buf, @as(u32, n) }, .gelu_tanh);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const go_ptr = MetalContext.bufferContents(T, go_buf);
                        @memcpy(go_ptr[0..n], grad_out);

                        const gx_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gx_ptr = MetalContext.bufferContents(T, gx_buf);
                        @memcpy(gx_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchGeluBackward, .{
                            context.x_buf, go_buf, gx_buf, @as(u32, n),
                        }, .gelu_tanh);

                        @memcpy(g, gx_ptr[0..n]);
                        metal.objRelease(go_buf);
                        metal.objRelease(gx_buf);
                    }
                }
            }
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// softmax: per-row softmax (no causal mask)
// input: (rows, cols), output: (rows, cols)
// ============================================================

pub fn softmaxOp(
    comptime T: type,
    comptime rows: usize,
    comptime cols: usize,
    _: [*]T,
    input_buf: id,
    input_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, rows * cols) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = rows * cols;

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExecCat(mtl, MetalContext.dispatchSoftmaxF32, .{
        input_buf, out_buf, @as(u32, rows), @as(u32, cols),
    }, .softmax);

    const Context = struct { out_buf: id, mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .out_buf = out_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = input_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gi_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchSoftmaxBackward, .{ context.out_buf, go_buf, gi_buf, @as(u32, rows), @as(u32, cols) }, .softmax);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const go_ptr = MetalContext.bufferContents(T, go_buf);
                        @memcpy(go_ptr[0..n], grad_out);

                        const gi_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gi_ptr = MetalContext.bufferContents(T, gi_buf);
                        @memcpy(gi_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchSoftmaxBackward, .{
                            context.out_buf, go_buf, gi_buf,
                            @as(u32, rows), @as(u32, cols),
                        }, .softmax);

                        @memcpy(g, gi_ptr[0..n]);
                        metal.objRelease(go_buf);
                        metal.objRelease(gi_buf);
                    }
                }
            }
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// causalSoftmax: per-row softmax with causal mask
// Used for decoder self-attention
// ============================================================

pub fn causalSoftmax(
    comptime T: type,
    comptime rows: usize,
    comptime cols: usize,
    comptime num_heads: usize,
    comptime seq_len: usize,
    _: [*]T,
    input_buf: id,
    input_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, rows * cols) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = rows * cols;

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExecCat(mtl, MetalContext.dispatchCausalSoftmaxF32, .{
        input_buf, out_buf, @as(u32, rows), @as(u32, cols), @as(u32, num_heads), @as(u32, seq_len),
    }, .softmax);

    // Backward is same as regular softmax (masked positions have out=0, grad=0)
    const Context = struct { out_buf: id, mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .out_buf = out_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = input_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gi_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchSoftmaxBackward, .{ context.out_buf, go_buf, gi_buf, @as(u32, rows), @as(u32, cols) }, .softmax);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const go_ptr = MetalContext.bufferContents(T, go_buf);
                        @memcpy(go_ptr[0..n], grad_out);

                        const gi_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gi_ptr = MetalContext.bufferContents(T, gi_buf);
                        @memcpy(gi_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchSoftmaxBackward, .{
                            context.out_buf, go_buf, gi_buf,
                            @as(u32, rows), @as(u32, cols),
                        }, .softmax);

                        @memcpy(g, gi_ptr[0..n]);
                        metal.objRelease(go_buf);
                        metal.objRelease(gi_buf);
                    }
                }
            }
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// layerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
// x: (rows, cols), gamma: (cols), beta: (cols) => y: (rows, cols)
// ============================================================

pub fn layerNorm(
    comptime T: type,
    comptime rows: usize,
    comptime cols: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    gamma_buf: id,
    gamma_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    beta_buf: id,
    beta_node: *GraphNodeMod.GraphNode(T),
    epsilon: T,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, rows * cols) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = rows * cols;

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    // Save mean and inv_std for backward
    const mean_buf = try mtl.createBuffer(rows * @sizeOf(T));
    const inv_std_buf = try mtl.createBuffer(rows * @sizeOf(T));

    gpuExecCat(mtl, MetalContext.dispatchLayerNormForward, .{
        x_buf, gamma_buf, beta_buf, out_buf, mean_buf, inv_std_buf,
        @as(u32, rows), @as(u32, cols), @as(f32, epsilon),
    }, .layernorm);

    const Context = struct {
        x_buf: id,
        gamma_buf: id,
        mean_buf: id,
        inv_std_buf: id,
        mtl: *MetalContext,
        gamma_node: *Node,
        beta_node: *Node,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{
        .x_buf = x_buf,
        .gamma_buf = gamma_buf,
        .mean_buf = mean_buf,
        .inv_std_buf = inv_std_buf,
        .mtl = mtl,
        .gamma_node = gamma_node,
        .beta_node = beta_node,
    };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents = .{ x_node, null, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchLayerNormBackwardX, .{
                            context.x_buf, context.gamma_buf, go_buf,
                            context.mean_buf, context.inv_std_buf, gx_buf,
                            @as(u32, rows), @as(u32, cols),
                        }, .layernorm);
                    }
                }
                const gamma_n = context.gamma_node;
                const beta_n = context.beta_node;
                if (gamma_n.requires_grad or beta_n.requires_grad) {
                    const gg_buf = m.getOrAllocGradBuf(@ptrCast(gamma_n), cols * @sizeOf(T));
                    const gb_buf = m.getOrAllocGradBuf(@ptrCast(beta_n), cols * @sizeOf(T));
                    gpuExecCat(m, MetalContext.dispatchLayerNormBackwardParams, .{
                        context.x_buf, go_buf, context.mean_buf, context.inv_std_buf,
                        gg_buf, gb_buf, @as(u32, rows), @as(u32, cols),
                    }, .layernorm);
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;

            const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
            const go_ptr = MetalContext.bufferContents(T, go_buf);
            @memcpy(go_ptr[0..n], grad_out);

            // grad_x
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const gx_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gx_ptr = MetalContext.bufferContents(T, gx_buf);
                        @memcpy(gx_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchLayerNormBackwardX, .{
                            context.x_buf, context.gamma_buf, go_buf,
                            context.mean_buf, context.inv_std_buf, gx_buf,
                            @as(u32, rows), @as(u32, cols),
                        }, .layernorm);

                        @memcpy(g, gx_ptr[0..n]);
                        metal.objRelease(gx_buf);
                    }
                }
            }

            // grad_gamma and grad_beta
            const gamma_n = context.gamma_node;
            const beta_n = context.beta_node;

            if (gamma_n.requires_grad or beta_n.requires_grad) {
                const gg_buf = m.createBuffer(cols * @sizeOf(T)) catch return;
                const gg_ptr = MetalContext.bufferContents(T, gg_buf);
                const gb_buf = m.createBuffer(cols * @sizeOf(T)) catch return;
                const gb_ptr = MetalContext.bufferContents(T, gb_buf);

                // Copy existing grads
                if (gamma_n.grad) |g| @memcpy(gg_ptr[0..cols], g) else @memset(gg_ptr[0..cols], 0);
                if (beta_n.grad) |g| @memcpy(gb_ptr[0..cols], g) else @memset(gb_ptr[0..cols], 0);

                gpuExecCat(m, MetalContext.dispatchLayerNormBackwardParams, .{
                    context.x_buf, go_buf, context.mean_buf, context.inv_std_buf,
                    gg_buf, gb_buf, @as(u32, rows), @as(u32, cols),
                }, .layernorm);

                if (gamma_n.grad) |g| @memcpy(g, gg_ptr[0..cols]);
                if (beta_n.grad) |g| @memcpy(g, gb_ptr[0..cols]);
                metal.objRelease(gg_buf);
                metal.objRelease(gb_buf);
            }

            metal.objRelease(go_buf);
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// crossEntropyLoss: L = -mean(log(softmax(logits)[target]))
// logits: (batch, num_classes), targets: (batch,) u32 indices
// ============================================================

pub fn crossEntropyLoss(
    comptime T: type,
    comptime batch_size: usize,
    comptime num_classes: usize,
    _: [*]T,
    logits_buf: id,
    logits_node: *GraphNodeMod.GraphNode(T),
    targets: []const u32,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, 1) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = batch_size * num_classes;

    // Upload targets to GPU
    const targets_buf = try mtl.createBuffer(batch_size * @sizeOf(u32));
    const targets_ptr = MetalContext.bufferContents(u32, targets_buf);
    @memcpy(targets_ptr[0..batch_size], targets);

    // Compute valid count (targets < num_classes) for padding support
    var valid_count: u32 = 0;
    for (targets[0..batch_size]) |t| {
        if (t < num_classes) valid_count += 1;
    }

    // Softmax output (saved for backward)
    const softmax_buf = try mtl.createBuffer(n * @sizeOf(T));

    // Per-sample loss
    const loss_per_sample_buf = try mtl.createBuffer(batch_size * @sizeOf(T));

    gpuExecCat(mtl, MetalContext.dispatchCrossEntropyForward, .{
        logits_buf, targets_buf, softmax_buf, loss_per_sample_buf,
        @as(u32, batch_size), @as(u32, num_classes),
    }, .loss);

    // Reduce to scalar loss (padding-aware)
    const loss_buf = try mtl.createBuffer(1 * @sizeOf(T));
    gpuExecCat(mtl, MetalContext.dispatchCrossEntropyReduce, .{
        loss_per_sample_buf, loss_buf, @as(u32, batch_size),
        targets_buf, @as(u32, num_classes),
    }, .loss);

    const loss_data = MetalContext.bufferContents(T, loss_buf);

    const Context = struct {
        softmax_buf: id,
        targets_buf: id,
        valid_count: u32,
        loss_per_sample_buf: id, // batch mode で GPU 実行前に解放されるのを防ぐ
        mtl: *MetalContext,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .softmax_buf = softmax_buf, .targets_buf = targets_buf, .valid_count = valid_count, .loss_per_sample_buf = loss_per_sample_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(1, true);
    node.parents[0] = logits_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), 1 * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gl_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchCrossEntropyBackward, .{
                            context.softmax_buf, context.targets_buf,
                            gl_buf, go_buf,
                            @as(u32, batch_size), @as(u32, num_classes),
                            context.valid_count,
                        }, .loss);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const go_buf = m.createBuffer(1 * @sizeOf(T)) catch return;
                        const go_ptr = MetalContext.bufferContents(T, go_buf);
                        go_ptr[0] = grad_out[0];

                        const gl_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gl_ptr = MetalContext.bufferContents(T, gl_buf);
                        @memcpy(gl_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchCrossEntropyBackward, .{
                            context.softmax_buf, context.targets_buf,
                            gl_buf, go_buf,
                            @as(u32, batch_size), @as(u32, num_classes),
                            context.valid_count,
                        }, .loss);

                        @memcpy(g, gl_ptr[0..n]);
                        metal.objRelease(go_buf);
                        metal.objRelease(gl_buf);
                    }
                }
            }

            // forward 中間バッファを backward 完了後に解放
            metal.objRelease(context.softmax_buf);
            metal.objRelease(context.targets_buf);
            metal.objRelease(context.loss_per_sample_buf);
        }
    }.backward;

    return .{ .data = loss_data, .data_buf = loss_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// embedding: out[i] = weight[indices[i]]
// weight: (vocab_size, embed_dim), indices: (num_tokens,) u32
// ============================================================

pub fn embedding(
    comptime T: type,
    comptime num_tokens: usize,
    comptime embed_dim: usize,
    comptime vocab_size: usize,
    _: [*]T,
    weight_buf: id,
    weight_node: *GraphNodeMod.GraphNode(T),
    indices: []const u32,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, num_tokens * embed_dim) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = num_tokens * embed_dim;

    // Upload indices to GPU
    const indices_buf = try mtl.createBuffer(num_tokens * @sizeOf(u32));
    const indices_ptr = MetalContext.bufferContents(u32, indices_buf);
    @memcpy(indices_ptr[0..num_tokens], indices);

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExecCat(mtl, MetalContext.dispatchEmbeddingForward, .{
        weight_buf, indices_buf, out_buf,
        @as(u32, num_tokens), @as(u32, embed_dim),
    }, .embedding);

    const Context = struct {
        indices_buf: id,
        mtl: *MetalContext,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .indices_buf = indices_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = weight_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gw_buf = m.getOrAllocGradBuf(@ptrCast(pa), vocab_size * embed_dim * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchEmbeddingBackward, .{
                            context.indices_buf, go_buf, gw_buf,
                            @as(u32, num_tokens), @as(u32, embed_dim),
                        }, .embedding);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const go_ptr = MetalContext.bufferContents(T, go_buf);
                        @memcpy(go_ptr[0..n], grad_out);

                        // grad_weight uses atomic add, so copy existing grad first
                        const gw_buf = m.createBuffer(vocab_size * embed_dim * @sizeOf(T)) catch return;
                        const gw_ptr = MetalContext.bufferContents(T, gw_buf);
                        @memcpy(gw_ptr[0 .. vocab_size * embed_dim], g);

                        // Use zero_buffer first since atomic_uint has different interpretation
                        // Actually we need to copy existing float grad into atomic buffer
                        // For CAS-based atomic, the buffer is just uint reinterpreted as float, so memcpy works
                        gpuExecCat(m, MetalContext.dispatchEmbeddingBackward, .{
                            context.indices_buf, go_buf, gw_buf,
                            @as(u32, num_tokens), @as(u32, embed_dim),
                        }, .embedding);

                        @memcpy(g, gw_ptr[0 .. vocab_size * embed_dim]);
                        metal.objRelease(go_buf);
                        metal.objRelease(gw_buf);
                    }
                }
            }
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// scale: z = x * scalar
// ============================================================

pub fn scale(
    comptime T: type,
    comptime n: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    scale_val: T,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, n) {
    const Node = GraphNodeMod.GraphNode(T);

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExecCat(mtl, MetalContext.dispatchScaleF32, .{
        x_buf, out_buf, @as(f32, scale_val), @as(u32, n),
    }, .scale);

    const Context = struct { scale_val: T, mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .scale_val = scale_val, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchScaleBackward, .{ go_buf, gx_buf, @as(f32, context.scale_val), @as(u32, n) }, .scale);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const go_ptr = MetalContext.bufferContents(T, go_buf);
                        @memcpy(go_ptr[0..n], grad_out);

                        const gx_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gx_ptr = MetalContext.bufferContents(T, gx_buf);
                        @memcpy(gx_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchScaleBackward, .{
                            go_buf, gx_buf, @as(f32, context.scale_val), @as(u32, n),
                        }, .scale);

                        @memcpy(g, gx_ptr[0..n]);
                        metal.objRelease(go_buf);
                        metal.objRelease(gx_buf);
                    }
                }
            }
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// matmulTransB: C = A @ B^T
// A: (M, K), B: (N, K) => C: (M, N)
// Used for attention: scores = Q @ K^T
// ============================================================

pub fn matmulTransB(
    comptime T: type,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    _: [*]T,
    a_buf: id,
    a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    b_buf: id,
    b_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, M * N) {
    const Node = GraphNodeMod.GraphNode(T);
    const result_n = M * N;

    const c_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    const c_data = MetalContext.bufferContents(T, c_buf);

    // C(M,N) = A(M,K) × B^T(N,K)
    mtl.dispatchMPSMatmul(a_buf, b_buf, c_buf, @as(u32, M), @as(u32, N), @as(u32, K), false, true, 1.0, 0.0);

    const Context = struct {
        a_buf: id,
        b_buf: id,
        mtl: *MetalContext,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .a_buf = a_buf, .b_buf = b_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);
    node.parents = .{ a_node, b_node, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode: MTLBuffer ベースで CPU round-trip なし ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), result_n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), M * K * @sizeOf(T));
                        // ga(M,K) += go(M,N) × B(N,K)
                        m.dispatchMPSMatmul(go_buf, context.b_buf, ga_buf, @as(u32, M), @as(u32, K), @as(u32, N), false, false, 1.0, 1.0);
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        const gb_buf = m.getOrAllocGradBuf(@ptrCast(pb), N * K * @sizeOf(T));
                        // gb(N,K) += go^T(N,M) × A(M,K)
                        m.dispatchMPSMatmul(go_buf, context.a_buf, gb_buf, @as(u32, N), @as(u32, K), @as(u32, M), true, false, 1.0, 1.0);
                    }
                }
                return;
            }

            // --- Non-batch mode: 従来の CPU copy 方式 ---
            const grad_out = self.grad orelse return;

            const go_buf = m.createBuffer(result_n * @sizeOf(T)) catch return;
            const go_ptr = MetalContext.bufferContents(T, go_buf);
            @memcpy(go_ptr[0..result_n], grad_out);

            // C = A @ B^T => grad_A = grad_C @ B (accum)
            // grad_C: (M,N), B: (N,K) => grad_A: (M,K)
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const ga_buf = m.createBuffer(M * K * @sizeOf(T)) catch return;
                        const ga_ptr = MetalContext.bufferContents(T, ga_buf);
                        @memcpy(ga_ptr[0 .. M * K], g);

                        // ga(M,K) += go(M,N) × B(N,K)
                        m.dispatchMPSMatmul(go_buf, context.b_buf, ga_buf, @as(u32, M), @as(u32, K), @as(u32, N), false, false, 1.0, 1.0);

                        @memcpy(g, ga_ptr[0 .. M * K]);
                        metal.objRelease(ga_buf);
                    }
                }
            }

            // C = A @ B^T => grad_B = grad_C^T @ A (accum)
            // gb(N,K) += go^T(N,M) × A(M,K)
            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |g| {
                        const gb_buf = m.createBuffer(N * K * @sizeOf(T)) catch return;
                        const gb_ptr = MetalContext.bufferContents(T, gb_buf);
                        @memcpy(gb_ptr[0 .. N * K], g);

                        // gb(N,K) += go^T(N,M) × A(M,K)
                        m.dispatchMPSMatmul(go_buf, context.a_buf, gb_buf, @as(u32, N), @as(u32, K), @as(u32, M), true, false, 1.0, 1.0);

                        @memcpy(g, gb_ptr[0 .. N * K]);
                        metal.objRelease(gb_buf);
                    }
                }
            }

            metal.objRelease(go_buf);
        }
    }.backward;

    return .{ .data = c_data, .data_buf = c_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// Phase 3: QLoRA Training Ops
// ============================================================

// ============================================================
// quantizedMatmul: y = x @ W^T (frozen quantized, forward only from batched kernel)
// x: (M, in_dim) f32, W: (out_dim, in_dim) quantized => y: (M, out_dim)
// Backward: grad_x = grad_y @ W (transposed quantized matmul), no grad_w
// ============================================================

pub fn quantizedMatmul(
    comptime T: type,
    comptime M: usize,
    comptime in_dim: usize,
    comptime out_dim: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    weight_buf: id,
    quant_type: metal.QuantType,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, M * out_dim) {
    return quantizedMatmulImpl(T, M, in_dim, out_dim, true, undefined, x_buf, x_node, weight_buf, quant_type, mtl, allocator);
}

/// quantizedMatmul without backward gradient propagation to input
/// Used for frozen layers where we don't need to backprop through the quantized weights
pub fn quantizedMatmulNoGrad(
    comptime T: type,
    comptime M: usize,
    comptime in_dim: usize,
    comptime out_dim: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    weight_buf: id,
    quant_type: metal.QuantType,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, M * out_dim) {
    return quantizedMatmulImpl(T, M, in_dim, out_dim, false, undefined, x_buf, x_node, weight_buf, quant_type, mtl, allocator);
}

fn quantizedMatmulImpl(
    comptime T: type,
    comptime M: usize,
    comptime in_dim: usize,
    comptime out_dim: usize,
    comptime propagate_grad: bool,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    weight_buf: id,
    quant_type: metal.QuantType,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, M * out_dim) {
    const Node = GraphNodeMod.GraphNode(T);
    const result_n = M * out_dim;

    // Forward: y = x @ W^T using batched quantized matmul
    const y_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    const y_data = MetalContext.bufferContents(T, y_buf);

    gpuExecCat(mtl, MetalContext.dispatchMatmulBatched, .{
        weight_buf, x_buf, y_buf,
        @as(u32, out_dim), @as(u32, in_dim), @as(u32, M), quant_type,
    }, .batched_matmul);

    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);

    if (propagate_grad) {
        const Context = struct {
            x_buf: id,
            weight_buf: id,
            quant_type: metal.QuantType,
            mtl: *MetalContext,
        };
        const ctx = try allocator.create(Context);
        ctx.* = .{ .x_buf = x_buf, .weight_buf = weight_buf, .quant_type = quant_type, .mtl = mtl };

        node.parents[0] = x_node;
        node.context = @ptrCast(ctx);

        node.backward_fn = struct {
            fn backward(self: *Node) void {
                const context: *Context = @ptrCast(@alignCast(self.context.?));
                const m = context.mtl;

                if (m.backward_grad_state != null) {
                    const go_buf = m.getOrAllocGradBuf(@ptrCast(self), result_n * @sizeOf(T));
                    if (self.parents[0]) |pa| {
                        if (pa.requires_grad) {
                            const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), M * in_dim * @sizeOf(T));
                            gpuExecCat(m, MetalContext.dispatchQuantTransBatched, .{
                                context.weight_buf, go_buf, gx_buf,
                                @as(u32, out_dim), @as(u32, in_dim), @as(u32, M),
                                context.quant_type,
                            }, .batched_matmul);
                        }
                    }
                    return;
                }

                const grad_out = self.grad orelse return;
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        if (pa.grad) |g| {
                            const go_buf = m.createBuffer(result_n * @sizeOf(T)) catch return;
                            const go_ptr = MetalContext.bufferContents(T, go_buf);
                            @memcpy(go_ptr[0..result_n], grad_out);

                            const gx_buf = m.createBuffer(M * in_dim * @sizeOf(T)) catch return;
                            const gx_ptr = MetalContext.bufferContents(T, gx_buf);
                            @memcpy(gx_ptr[0 .. M * in_dim], g);

                            gpuExecCat(m, MetalContext.dispatchQuantTransBatched, .{
                                context.weight_buf, go_buf, gx_buf,
                                @as(u32, out_dim), @as(u32, in_dim), @as(u32, M),
                                context.quant_type,
                            }, .batched_matmul);

                            @memcpy(g, gx_ptr[0 .. M * in_dim]);
                            metal.objRelease(go_buf);
                            metal.objRelease(gx_buf);
                        }
                    }
                }
            }
        }.backward;
    }
    // If !propagate_grad: no backward_fn, no parents — gradient stops here

    return .{ .data = y_data, .data_buf = y_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// rmsNorm: y = x * inv_rms * weight (training version with backward)
// x: (rows, dim), weight: (dim) => y: (rows, dim)
// ============================================================

pub fn rmsNorm(
    comptime T: type,
    comptime rows: usize,
    comptime dim: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    weight_buf: id,
    weight_node: *GraphNodeMod.GraphNode(T),
    eps: T,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, rows * dim) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = rows * dim;

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);
    const inv_rms_buf = try mtl.createBuffer(rows * @sizeOf(T));

    gpuExecCat(mtl, MetalContext.dispatchRMSNormForwardTraining, .{
        x_buf, weight_buf, out_buf, inv_rms_buf,
        @as(u32, rows), @as(u32, dim), @as(f32, eps),
    }, .layernorm);

    const Context = struct {
        x_buf: id,
        weight_buf: id,
        inv_rms_buf: id,
        mtl: *MetalContext,
        weight_node: *Node,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{
        .x_buf = x_buf,
        .weight_buf = weight_buf,
        .inv_rms_buf = inv_rms_buf,
        .mtl = mtl,
        .weight_node = weight_node,
    };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                // grad_x
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchRMSNormBackwardX, .{
                            context.x_buf, context.weight_buf, go_buf,
                            context.inv_rms_buf, gx_buf,
                            @as(u32, rows), @as(u32, dim),
                        }, .layernorm);
                    }
                }
                // grad_weight
                const wn = context.weight_node;
                if (wn.requires_grad) {
                    const gw_buf = m.getOrAllocGradBuf(@ptrCast(wn), dim * @sizeOf(T));
                    gpuExecCat(m, MetalContext.dispatchRMSNormBackwardWeight, .{
                        context.x_buf, go_buf, context.inv_rms_buf,
                        gw_buf, @as(u32, rows), @as(u32, dim),
                    }, .layernorm);
                }
                return;
            }

            // Non-batch mode
            const grad_out = self.grad orelse return;
            const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
            const go_ptr = MetalContext.bufferContents(T, go_buf);
            @memcpy(go_ptr[0..n], grad_out);

            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const gx_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gx_ptr = MetalContext.bufferContents(T, gx_buf);
                        @memcpy(gx_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchRMSNormBackwardX, .{
                            context.x_buf, context.weight_buf, go_buf,
                            context.inv_rms_buf, gx_buf,
                            @as(u32, rows), @as(u32, dim),
                        }, .layernorm);

                        @memcpy(g, gx_ptr[0..n]);
                        metal.objRelease(gx_buf);
                    }
                }
            }

            const wn = context.weight_node;
            if (wn.requires_grad) {
                if (wn.grad) |g| {
                    const gw_buf = m.createBuffer(dim * @sizeOf(T)) catch return;
                    const gw_ptr = MetalContext.bufferContents(T, gw_buf);
                    @memcpy(gw_ptr[0..dim], g);

                    gpuExecCat(m, MetalContext.dispatchRMSNormBackwardWeight, .{
                        context.x_buf, go_buf, context.inv_rms_buf,
                        gw_buf, @as(u32, rows), @as(u32, dim),
                    }, .layernorm);

                    @memcpy(g, gw_ptr[0..dim]);
                    metal.objRelease(gw_buf);
                }
            }

            metal.objRelease(go_buf);
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// rope: in-place RoPE with training backward support
// x: (seq_len, n_heads * head_dim) => x (rotated in-place)
// ============================================================

pub fn rope(
    comptime T: type,
    comptime seq_len: usize,
    comptime n_heads: usize,
    comptime half_dim: usize,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    freqs_buf: id,
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, seq_len * n_heads * half_dim * 2) {
    const Node = GraphNodeMod.GraphNode(T);
    const n = seq_len * n_heads * half_dim * 2;

    // Forward: in-place rotation, save sin/cos
    const sin_buf = try mtl.createBuffer(seq_len * half_dim * @sizeOf(T));
    const cos_buf = try mtl.createBuffer(seq_len * half_dim * @sizeOf(T));

    gpuExec(mtl, MetalContext.dispatchRoPEForwardTraining, .{
        x_buf, freqs_buf, sin_buf, cos_buf,
        @as(u32, seq_len), @as(u32, n_heads), @as(u32, half_dim),
    });

    const x_data = MetalContext.bufferContents(T, x_buf);

    const Context = struct {
        sin_buf: id,
        cos_buf: id,
        mtl: *MetalContext,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .sin_buf = sin_buf, .cos_buf = cos_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            if (m.backward_grad_state != null) {
                // grad is in-place on parent's grad (RoPE backward = inverse rotation)
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        // First copy grad_out into grad_x if different
                        const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                        if (@intFromPtr(go_buf) != @intFromPtr(gx_buf)) {
                            gpuExec(m, MetalContext.dispatchAddBackwardAccum, .{ go_buf, gx_buf, @as(u32, n) });
                        }
                        // Apply inverse rotation in-place on grad
                        gpuExec(m, MetalContext.dispatchRoPEBackward, .{
                            gx_buf, context.sin_buf, context.cos_buf,
                            @as(u32, seq_len), @as(u32, n_heads), @as(u32, half_dim),
                        });
                    }
                }
                return;
            }

            // Non-batch mode
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const gx_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gx_ptr = MetalContext.bufferContents(T, gx_buf);
                        // Copy grad_out, then apply inverse rotation
                        for (0..n) |idx| {
                            gx_ptr[idx] = g[idx] + grad_out[idx];
                        }

                        gpuExec(m, MetalContext.dispatchRoPEBackward, .{
                            gx_buf, context.sin_buf, context.cos_buf,
                            @as(u32, seq_len), @as(u32, n_heads), @as(u32, half_dim),
                        });

                        @memcpy(g, gx_ptr[0..n]);
                        metal.objRelease(gx_buf);
                    }
                }
            }
        }
    }.backward;

    // RoPE is in-place on x_buf, return same buffer
    return .{ .data = x_data, .data_buf = x_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// Phase 4: Sequence ops
// ============================================================

/// tanh activation: out = tanh(x)
/// backward: grad_in += grad_out * (1 - out^2)
/// Saves output for backward (like silu pattern with sigmoid).
pub fn tanhOp(
    comptime T: type,
    comptime n: usize,
    _: [*]T,
    x_buf: id,
    x_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, n) {
    const Node = GraphNodeMod.GraphNode(T);

    const out_buf = try mtl.createBuffer(n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExecCat(mtl, MetalContext.dispatchTanhForward, .{
        x_buf, out_buf, @as(u32, n),
    }, .gelu_tanh);

    const Context = struct { out_buf: id, mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .out_buf = out_buf, .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n, true);
    node.parents[0] = x_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const gx_buf = m.getOrAllocGradBuf(@ptrCast(pa), n * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchTanhBackward, .{ context.out_buf, go_buf, gx_buf, @as(u32, n) }, .gelu_tanh);
                    }
                }
                return;
            }

            // --- Non-batch mode ---
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |g| {
                        const go_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const go_ptr = MetalContext.bufferContents(T, go_buf);
                        @memcpy(go_ptr[0..n], grad_out);

                        const gx_buf = m.createBuffer(n * @sizeOf(T)) catch return;
                        const gx_ptr = MetalContext.bufferContents(T, gx_buf);
                        @memcpy(gx_ptr[0..n], g);

                        gpuExecCat(m, MetalContext.dispatchTanhBackward, .{
                            context.out_buf, go_buf, gx_buf, @as(u32, n),
                        }, .gelu_tanh);

                        @memcpy(g, gx_ptr[0..n]);
                        metal.objRelease(go_buf);
                        metal.objRelease(gx_buf);
                    }
                }
            }
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

/// Concatenate two 2D tensors along the last dimension.
/// a: (rows, cols_a), b: (rows, cols_b) → out: (rows, cols_a + cols_b)
pub fn concatLastDim(
    comptime T: type,
    comptime rows: usize,
    comptime cols_a: usize,
    comptime cols_b: usize,
    _: [*]T,
    a_buf: id,
    a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    b_buf: id,
    b_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, rows * (cols_a + cols_b)) {
    const Node = GraphNodeMod.GraphNode(T);
    const n_out = rows * (cols_a + cols_b);

    const out_buf = try mtl.createBuffer(n_out * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);

    gpuExecCat(mtl, MetalContext.dispatchConcatLastDim, .{
        a_buf, b_buf, out_buf, @as(u32, rows), @as(u32, cols_a), @as(u32, cols_b),
    }, .concat);

    const Context = struct { mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .mtl = mtl };

    const node = try allocator.create(Node);
    node.* = Node.init(n_out, true);
    node.parents[0] = a_node;
    node.parents[1] = b_node;
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            // --- Backward batch mode ---
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), n_out * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), rows * cols_a * @sizeOf(T));
                        if (self.parents[1]) |pb| {
                            if (pb.requires_grad) {
                                const gb_buf = m.getOrAllocGradBuf(@ptrCast(pb), rows * cols_b * @sizeOf(T));
                                gpuExecCat(m, MetalContext.dispatchConcatLastDimBackward, .{ go_buf, ga_buf, gb_buf, @as(u32, rows), @as(u32, cols_a), @as(u32, cols_b) }, .concat);
                            } else {
                                // Only a needs grad - still need a temp buffer for b
                                const gb_buf = m.createBuffer(rows * cols_b * @sizeOf(T)) catch return;
                                gpuExecCat(m, MetalContext.dispatchConcatLastDimBackward, .{ go_buf, ga_buf, gb_buf, @as(u32, rows), @as(u32, cols_a), @as(u32, cols_b) }, .concat);
                            }
                        } else {
                            const gb_buf = m.createBuffer(rows * cols_b * @sizeOf(T)) catch return;
                            gpuExecCat(m, MetalContext.dispatchConcatLastDimBackward, .{ go_buf, ga_buf, gb_buf, @as(u32, rows), @as(u32, cols_a), @as(u32, cols_b) }, .concat);
                        }
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        if (self.parents[0]) |pa| {
                            if (!pa.requires_grad) {
                                // Only b needs grad
                                const go_buf2 = m.getOrAllocGradBuf(@ptrCast(self), n_out * @sizeOf(T));
                                const ga_buf = m.createBuffer(rows * cols_a * @sizeOf(T)) catch return;
                                const gb_buf = m.getOrAllocGradBuf(@ptrCast(pb), rows * cols_b * @sizeOf(T));
                                gpuExecCat(m, MetalContext.dispatchConcatLastDimBackward, .{ go_buf2, ga_buf, gb_buf, @as(u32, rows), @as(u32, cols_a), @as(u32, cols_b) }, .concat);
                            }
                        }
                    }
                }
                return;
            }

            // --- Non-batch mode ---
            const grad_out = self.grad orelse return;
            if (self.parents[0]) |pa| {
                if (pa.requires_grad) {
                    if (pa.grad) |ga| {
                        for (0..rows) |r| {
                            for (0..cols_a) |c| {
                                ga[r * cols_a + c] += grad_out[r * (cols_a + cols_b) + c];
                            }
                        }
                    }
                }
            }
            if (self.parents[1]) |pb| {
                if (pb.requires_grad) {
                    if (pb.grad) |gb| {
                        for (0..rows) |r| {
                            for (0..cols_b) |c| {
                                gb[r * cols_b + c] += grad_out[r * (cols_a + cols_b) + cols_a + c];
                            }
                        }
                    }
                }
            }
        }
    }.backward;

    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// batchedMatmul: C[b] = A[b] @ B[b]
// A: (batch, M, K), B: (batch, K, N) => C: (batch, M, N)
// ============================================================

pub fn batchedMatmul(
    comptime T: type,
    comptime batch: usize,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    _: [*]T,
    a_buf: id,
    a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    b_buf: id,
    b_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, batch * M * N) {
    const Node = GraphNodeMod.GraphNode(T);
    const result_n = batch * M * N;

    const c_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    const c_data = MetalContext.bufferContents(T, c_buf);

    gpuExecCat(mtl, MetalContext.dispatchBatchedMatmulF32, .{
        a_buf, b_buf, c_buf,
        @as(u32, batch), @as(u32, M), @as(u32, K), @as(u32, N),
    }, .batched_matmul);

    const Context = struct {
        a_buf: id,
        b_buf: id,
        mtl: *MetalContext,
        a_node: *Node,
        b_node: *Node,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .a_buf = a_buf, .b_buf = b_buf, .mtl = mtl, .a_node = a_node, .b_node = b_node };

    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);
    node.parents = .{ a_node, b_node, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), result_n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), batch * M * K * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchBatchedMatmulBackwardA, .{
                            go_buf, context.b_buf, ga_buf,
                            @as(u32, batch), @as(u32, M), @as(u32, K), @as(u32, N),
                        }, .batched_matmul);
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        const gb_buf = m.getOrAllocGradBuf(@ptrCast(pb), batch * K * N * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchBatchedMatmulBackwardB, .{
                            context.a_buf, go_buf, gb_buf,
                            @as(u32, batch), @as(u32, M), @as(u32, K), @as(u32, N),
                        }, .batched_matmul);
                    }
                }
                return;
            }

            // Non-batch mode: not supported for batched ops
        }
    }.backward;

    return .{ .data = c_data, .data_buf = c_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// batchedMatmulTransB: C[b] = A[b] @ B[b]^T
// A: (batch, M, K), B: (batch, N, K) => C: (batch, M, N)
// ============================================================

pub fn batchedMatmulTransB(
    comptime T: type,
    comptime batch: usize,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    _: [*]T,
    a_buf: id,
    a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T,
    b_buf: id,
    b_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext,
    allocator: Allocator,
) !GpuResult(T, batch * M * N) {
    const Node = GraphNodeMod.GraphNode(T);
    const result_n = batch * M * N;

    const c_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    const c_data = MetalContext.bufferContents(T, c_buf);

    gpuExecCat(mtl, MetalContext.dispatchBatchedMatmulTransBF32, .{
        a_buf, b_buf, c_buf,
        @as(u32, batch), @as(u32, M), @as(u32, K), @as(u32, N),
    }, .batched_matmul);

    const Context = struct {
        a_buf: id,
        b_buf: id,
        mtl: *MetalContext,
        a_node: *Node,
        b_node: *Node,
    };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .a_buf = a_buf, .b_buf = b_buf, .mtl = mtl, .a_node = a_node, .b_node = b_node };

    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);
    node.parents = .{ a_node, b_node, null };
    node.context = @ptrCast(ctx);

    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;

            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), result_n * @sizeOf(T));
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), batch * M * K * @sizeOf(T));
                        // dA[b] = dC[b] @ B[b]
                        gpuExecCat(m, MetalContext.dispatchBatchedMatmulTransBBackwardA, .{
                            go_buf, context.b_buf, ga_buf,
                            @as(u32, batch), @as(u32, M), @as(u32, K), @as(u32, N),
                        }, .batched_matmul);
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        const gb_buf = m.getOrAllocGradBuf(@ptrCast(pb), batch * N * K * @sizeOf(T));
                        // dB[b] = dC[b]^T @ A[b]
                        gpuExecCat(m, MetalContext.dispatchBatchedMatmulTransBBackwardB, .{
                            go_buf, context.a_buf, gb_buf,
                            @as(u32, batch), @as(u32, M), @as(u32, K), @as(u32, N),
                        }, .batched_matmul);
                    }
                }
                return;
            }
        }
    }.backward;

    return .{ .data = c_data, .data_buf = c_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

// ============================================================
// Fused ops: matmul + addBias + activation
// ============================================================

/// Fused matmul + addBias + gelu: out = gelu(A @ B + bias)
pub fn matmulAddbiasGelu(
    comptime T: type,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    _: [*]T, a_buf: id, a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T, b_buf: id, b_node: *GraphNodeMod.GraphNode(T),
    _: [*]T, bias_buf: id, bias_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext, allocator: Allocator,
) !GpuResult(T, M * N) {
    const Node = GraphNodeMod.GraphNode(T);
    const result_n = M * N;
    const out_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);
    const pre_act_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    gpuExecCat(mtl, MetalContext.dispatchMatmulAddbiasGeluF32, .{
        a_buf, b_buf, bias_buf, out_buf, pre_act_buf,
        @as(u32, M), @as(u32, K), @as(u32, N),
    }, .gelu_tanh);
    const Context = struct { a_buf: id, b_buf: id, bias_node: *Node, pre_act_buf: id, mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .a_buf = a_buf, .b_buf = b_buf, .bias_node = bias_node, .pre_act_buf = pre_act_buf, .mtl = mtl };
    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);
    node.parents = .{ a_node, b_node, null };
    node.context = @ptrCast(ctx);
    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), result_n * @sizeOf(T));
                const gpa_buf = m.createBuffer(result_n * @sizeOf(T)) catch return;
                const gpa_ptr = MetalContext.bufferContents(T, gpa_buf);
                @memset(gpa_ptr[0..result_n], 0);
                m.addTempBuf(gpa_buf);
                const gb_buf = if (context.bias_node.requires_grad)
                    m.getOrAllocGradBuf(@ptrCast(context.bias_node), N * @sizeOf(T))
                else blk: {
                    const d = m.createBuffer(N * @sizeOf(T)) catch return;
                    m.addTempBuf(d);
                    break :blk d;
                };
                gpuExecCat(m, MetalContext.dispatchGeluBiasBackward, .{ go_buf, context.pre_act_buf, gpa_buf, gb_buf, @as(u32, M), @as(u32, N) }, .gelu_tanh);
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), M * K * @sizeOf(T));
                        // dA(M,K) += dC(M,N) × B^T(N,K)
                        m.dispatchMPSMatmul(gpa_buf, context.b_buf, ga_buf, @as(u32, M), @as(u32, K), @as(u32, N), false, true, 1.0, 1.0);
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        const gb2_buf = m.getOrAllocGradBuf(@ptrCast(pb), K * N * @sizeOf(T));
                        // dB(K,N) += A^T(K,M) × dC(M,N)
                        m.dispatchMPSMatmul(context.a_buf, gpa_buf, gb2_buf, @as(u32, K), @as(u32, N), @as(u32, M), true, false, 1.0, 1.0);
                    }
                }
                return;
            }
        }
    }.backward;
    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

/// Fused matmul + addBias + tanh: out = tanh(A @ B + bias)
pub fn matmulAddbiasTanh(
    comptime T: type,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    _: [*]T, a_buf: id, a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T, b_buf: id, b_node: *GraphNodeMod.GraphNode(T),
    _: [*]T, bias_buf: id, bias_node: *GraphNodeMod.GraphNode(T),
    mtl: *MetalContext, allocator: Allocator,
) !GpuResult(T, M * N) {
    const Node = GraphNodeMod.GraphNode(T);
    const result_n = M * N;
    const out_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    const out_data = MetalContext.bufferContents(T, out_buf);
    gpuExecCat(mtl, MetalContext.dispatchMatmulAddbiasTanhF32, .{
        a_buf, b_buf, bias_buf, out_buf,
        @as(u32, M), @as(u32, K), @as(u32, N),
    }, .gelu_tanh);
    const Context = struct { a_buf: id, b_buf: id, out_buf: id, bias_node: *Node, mtl: *MetalContext };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .a_buf = a_buf, .b_buf = b_buf, .out_buf = out_buf, .bias_node = bias_node, .mtl = mtl };
    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);
    node.parents = .{ a_node, b_node, null };
    node.context = @ptrCast(ctx);
    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), result_n * @sizeOf(T));
                const gpa_buf = m.createBuffer(result_n * @sizeOf(T)) catch return;
                const gpa_ptr = MetalContext.bufferContents(T, gpa_buf);
                @memset(gpa_ptr[0..result_n], 0);
                m.addTempBuf(gpa_buf);
                const gb_buf = if (context.bias_node.requires_grad)
                    m.getOrAllocGradBuf(@ptrCast(context.bias_node), N * @sizeOf(T))
                else blk: {
                    const d = m.createBuffer(N * @sizeOf(T)) catch return;
                    m.addTempBuf(d);
                    break :blk d;
                };
                gpuExecCat(m, MetalContext.dispatchTanhBiasBackward, .{ go_buf, context.out_buf, gpa_buf, gb_buf, @as(u32, M), @as(u32, N) }, .gelu_tanh);
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), M * K * @sizeOf(T));
                        // dA(M,K) += dC(M,N) × B^T(N,K)
                        m.dispatchMPSMatmul(gpa_buf, context.b_buf, ga_buf, @as(u32, M), @as(u32, K), @as(u32, N), false, true, 1.0, 1.0);
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        const gb2_buf = m.getOrAllocGradBuf(@ptrCast(pb), K * N * @sizeOf(T));
                        // dB(K,N) += A^T(K,M) × dC(M,N)
                        m.dispatchMPSMatmul(context.a_buf, gpa_buf, gb2_buf, @as(u32, K), @as(u32, N), @as(u32, M), true, false, 1.0, 1.0);
                    }
                }
                return;
            }
        }
    }.backward;
    return .{ .data = out_data, .data_buf = out_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}

/// Fused batchedMatmulTransB + scale: C[b] = scale * (A[b] @ B[b]^T)
pub fn batchedMatmulTransBScale(
    comptime T: type,
    comptime batch_count: usize,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    _: [*]T, a_buf: id, a_node: *GraphNodeMod.GraphNode(T),
    _: [*]T, b_buf: id, b_node: *GraphNodeMod.GraphNode(T),
    scale_val: T,
    mtl: *MetalContext, allocator: Allocator,
) !GpuResult(T, batch_count * M * N) {
    const Node = GraphNodeMod.GraphNode(T);
    const result_n = batch_count * M * N;
    const result_buf = try mtl.createBuffer(result_n * @sizeOf(T));
    const result_data = MetalContext.bufferContents(T, result_buf);
    gpuExecCat(mtl, MetalContext.dispatchBatchedMatmulTransBScaleF32, .{
        a_buf, b_buf, result_buf,
        @as(u32, batch_count), @as(u32, M), @as(u32, K), @as(u32, N), scale_val,
    }, .batched_matmul);
    const Context = struct { a_buf: id, b_buf: id, mtl: *MetalContext, sv: T };
    const ctx = try allocator.create(Context);
    ctx.* = .{ .a_buf = a_buf, .b_buf = b_buf, .mtl = mtl, .sv = scale_val };
    const node = try allocator.create(Node);
    node.* = Node.init(result_n, true);
    node.parents = .{ a_node, b_node, null };
    node.context = @ptrCast(ctx);
    node.backward_fn = struct {
        fn backward(self: *Node) void {
            const context: *Context = @ptrCast(@alignCast(self.context.?));
            const m = context.mtl;
            if (m.backward_grad_state != null) {
                const go_buf = m.getOrAllocGradBuf(@ptrCast(self), result_n * @sizeOf(T));
                const gs_buf = m.createBuffer(result_n * @sizeOf(T)) catch return;
                m.addTempBuf(gs_buf);
                gpuExecCat(m, MetalContext.dispatchScaleF32, .{ go_buf, gs_buf, context.sv, @as(u32, result_n) }, .scale);
                if (self.parents[0]) |pa| {
                    if (pa.requires_grad) {
                        const ga_buf = m.getOrAllocGradBuf(@ptrCast(pa), batch_count * M * K * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchBatchedMatmulTransBBackwardA, .{ gs_buf, context.b_buf, ga_buf, @as(u32, batch_count), @as(u32, M), @as(u32, K), @as(u32, N) }, .batched_matmul);
                    }
                }
                if (self.parents[1]) |pb| {
                    if (pb.requires_grad) {
                        const gb_buf = m.getOrAllocGradBuf(@ptrCast(pb), batch_count * N * K * @sizeOf(T));
                        gpuExecCat(m, MetalContext.dispatchBatchedMatmulTransBBackwardB, .{ gs_buf, context.a_buf, gb_buf, @as(u32, batch_count), @as(u32, M), @as(u32, K), @as(u32, N) }, .batched_matmul);
                    }
                }
                return;
            }
        }
    }.backward;
    return .{ .data = result_data, .data_buf = result_buf, .grad_buf = null, .node = node, .metal_ctx = mtl };
}
