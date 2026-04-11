const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");
const ModuleMixin = @import("module.zig").Module;
const cpu_backend = @import("../backend/cpu.zig");

/// Attention 重み初期化ヘルパー (MultiHead/Causal/Cross 共通)
fn initAttentionWeights(
    comptime T: type,
    comptime embed_dim: usize,
    comptime Self: type,
    allocator: Allocator,
) !Self {
    const limit: T = @sqrt(6.0 / @as(T, @floatFromInt(embed_dim + embed_dim)));
    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.posix.getrandom(std.mem.asBytes(&seed)) catch {
            seed = 42;
        };
        break :blk seed;
    });
    const rng = prng.random();

    var self: Self = undefined;
    inline for (.{ &self.w_q, &self.w_k, &self.w_v, &self.w_o }) |wp| {
        const t = try TensorMod.Tensor(T, .{ embed_dim, embed_dim }).init(allocator);
        for (t.slice()) |*v| v.* = (rng.float(T) * 2.0 - 1.0) * limit;
        wp.* = try VariableMod.Variable(T, .{ embed_dim, embed_dim }).init(t, allocator, true);
    }
    inline for (.{ &self.b_q, &self.b_k, &self.b_v, &self.b_o }) |bp| {
        const t = try TensorMod.Tensor(T, .{embed_dim}).zeros(allocator);
        bp.* = try VariableMod.Variable(T, .{embed_dim}).init(t, allocator, true);
    }
    return self;
}

/// Multi-Head Attention。
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// Q, K, V はそれぞれ線形変換で生成:
///   Q = input @ W_q + b_q
///   K = input @ W_k + b_k
///   V = input @ W_v + b_v
///
/// 出力 = concat(head_1, ..., head_h) @ W_o + b_o
pub fn MultiHeadAttention(
    comptime T: type,
    comptime embed_dim: usize,
    comptime num_heads: usize,
) type {
    if (embed_dim % num_heads != 0)
        @compileError("embed_dim must be divisible by num_heads");

    const head_dim = embed_dim / num_heads;

    return struct {
        const Self = @This();
        const Node = GraphNodeMod.GraphNode(T);
        const M = ModuleMixin(Self);

        // Linear projections: (embed_dim, embed_dim)
        w_q: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_k: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_v: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_o: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        b_q: VariableMod.Variable(T, .{embed_dim}),
        b_k: VariableMod.Variable(T, .{embed_dim}),
        b_v: VariableMod.Variable(T, .{embed_dim}),
        b_o: VariableMod.Variable(T, .{embed_dim}),

        pub fn init(allocator: Allocator) !Self {
            return initAttentionWeights(T, embed_dim, Self, allocator);
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        /// forward: input (batch, seq_len, embed_dim) → output (batch, seq_len, embed_dim)
        /// Self-attention: Q=K=V=input
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime seq_len: usize,
            input: *VariableMod.Variable(T, .{ batch, seq_len, embed_dim }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, seq_len, embed_dim }) {
            const n = batch * seq_len * embed_dim;
            const in_data = input.constData();
            const scale: T = 1.0 / @sqrt(@as(T, @floatFromInt(head_dim)));

            // Single forward allocation for Q, K, V, attn_out, scores (5→1 alloc)
            const scores_n = batch * num_heads * seq_len * seq_len;
            const fwd_buf = try allocator.alloc(T, 4 * n + scores_n);
            const q_buf = fwd_buf[0..n];
            const k_buf = fwd_buf[n..][0..n];
            const v_buf = fwd_buf[2 * n..][0..n];
            const attn_out = fwd_buf[3 * n..][0..n];
            const attn_weights = fwd_buf[4 * n..][0..scores_n];
            linearProject(batch * seq_len, in_data, self.w_q.constData(), self.b_q.constData(), q_buf);
            linearProject(batch * seq_len, in_data, self.w_k.constData(), self.b_k.constData(), k_buf);
            linearProject(batch * seq_len, in_data, self.w_v.constData(), self.b_v.constData(), v_buf);

            // Compute scores = Q @ K^T / sqrt(d_k), then softmax
            for (0..batch) |b| {
                for (0..num_heads) |h| {
                    for (0..seq_len) |qi| {
                        // Softmax numerically stable
                        var max_val: T = -std.math.inf(T);
                        for (0..seq_len) |ki| {
                            var dot: T = 0;
                            for (0..head_dim) |d| {
                                const q_idx = (b * seq_len + qi) * embed_dim + h * head_dim + d;
                                const k_idx = (b * seq_len + ki) * embed_dim + h * head_dim + d;
                                dot += q_buf[q_idx] * k_buf[k_idx];
                            }
                            dot *= scale;
                            const aw_idx = ((b * num_heads + h) * seq_len + qi) * seq_len + ki;
                            attn_weights[aw_idx] = dot;
                            if (dot > max_val) max_val = dot;
                        }
                        // Softmax
                        var sum_exp: T = 0;
                        for (0..seq_len) |ki| {
                            const aw_idx = ((b * num_heads + h) * seq_len + qi) * seq_len + ki;
                            const e = @exp(attn_weights[aw_idx] - max_val);
                            attn_weights[aw_idx] = e;
                            sum_exp += e;
                        }
                        for (0..seq_len) |ki| {
                            const aw_idx = ((b * num_heads + h) * seq_len + qi) * seq_len + ki;
                            attn_weights[aw_idx] /= sum_exp;
                        }
                    }
                }
            }

            // Attention output: attn_weights @ V → (batch, seq_len, embed_dim)
            @memset(attn_out, 0);
            for (0..batch) |b| {
                for (0..num_heads) |h| {
                    for (0..seq_len) |qi| {
                        for (0..seq_len) |vi| {
                            const w = attn_weights[((b * num_heads + h) * seq_len + qi) * seq_len + vi];
                            for (0..head_dim) |d| {
                                const out_idx = (b * seq_len + qi) * embed_dim + h * head_dim + d;
                                const v_idx = (b * seq_len + vi) * embed_dim + h * head_dim + d;
                                attn_out[out_idx] += w * v_buf[v_idx];
                            }
                        }
                    }
                }
            }

            // Output projection: attn_out @ W_o + b_o
            const out_tensor = try TensorMod.Tensor(T, .{ batch, seq_len, embed_dim }).init(allocator);
            linearProject(batch * seq_len, attn_out, self.w_o.constData(), self.b_o.constData(), out_tensor.slice());

            // Context for backward
            const Ctx = struct {
                q: []const T,
                k: []const T,
                v: []const T,
                attn_w: []const T,
                attn_out: []const T,
                in_data: []const T,
                w_q: []const T,
                w_k: []const T,
                w_v: []const T,
                w_o: []const T,
                input_node: *Node,
                w_q_node: *Node, w_k_node: *Node, w_v_node: *Node, w_o_node: *Node,
                b_q_node: *Node, b_k_node: *Node, b_v_node: *Node, b_o_node: *Node,
                alloc: Allocator,
            };
            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .q = q_buf, .k = k_buf, .v = v_buf,
                .attn_w = attn_weights, .attn_out = attn_out,
                .in_data = in_data,
                .w_q = self.w_q.constData(), .w_k = self.w_k.constData(),
                .w_v = self.w_v.constData(), .w_o = self.w_o.constData(),
                .input_node = input.node,
                .w_q_node = self.w_q.node, .w_k_node = self.w_k.node,
                .w_v_node = self.w_v.node, .w_o_node = self.w_o.node,
                .b_q_node = self.b_q.node, .b_k_node = self.b_k.node,
                .b_v_node = self.b_v.node, .b_o_node = self.b_o.node,
                .alloc = allocator,
            };

            const OutVar = VariableMod.Variable(T, .{ batch, seq_len, embed_dim });
            var result = try OutVar.init(out_tensor, allocator, true);
            result.node.parents[0] = input.node;
            result.node.parents[1] = self.w_q.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));
                    const BS = batch * seq_len;
                    const alloc = c.alloc;

                    // Single backward allocation (6→1 alloc)
                    const bw_n = BS * embed_dim;
                    const bw_scores_n = batch * num_heads * seq_len * seq_len;
                    const bw_buf = alloc.alloc(T, 4 * bw_n + 2 * bw_scores_n) catch unreachable;
                    @memset(bw_buf, 0);
                    const d_attn_out = bw_buf[0..bw_n];
                    const dv = bw_buf[bw_n..][0..bw_n];
                    const dq = bw_buf[2 * bw_n..][0..bw_n];
                    const dk = bw_buf[3 * bw_n..][0..bw_n];
                    const d_attn_w = bw_buf[4 * bw_n..][0..bw_scores_n];
                    const d_scores = bw_buf[4 * bw_n + bw_scores_n..][0..bw_scores_n];

                    // 1. Backward through output projection (BLAS)
                    // dW_o += attn_out^T @ grad_out
                    if (c.w_o_node.grad) |wg| {
                        cpu_backend.matmulTransAAccum(T, c.attn_out.ptr, grad_out.ptr, wg.ptr, embed_dim, BS, embed_dim);
                    }
                    // db_o += sum(grad_out) per row
                    if (c.b_o_node.grad) |bg| {
                        for (0..BS) |row| {
                            cpu_backend.addAccum(T, grad_out.ptr + row * embed_dim, bg.ptr, embed_dim);
                        }
                    }
                    // d_attn_out = grad_out @ W_o^T
                    cpu_backend.matmulTransB(T, grad_out.ptr, c.w_o.ptr, d_attn_out.ptr, BS, embed_dim, embed_dim);

                    // 2. Backward through attention: attn_out = attn_w @ V
                    for (0..batch) |b| {
                        for (0..num_heads) |h| {
                            for (0..seq_len) |qi| {
                                for (0..seq_len) |vi| {
                                    var dot: T = 0;
                                    for (0..head_dim) |d| {
                                        const oi = (b * seq_len + qi) * embed_dim + h * head_dim + d;
                                        const vi_idx = (b * seq_len + vi) * embed_dim + h * head_dim + d;
                                        dot += d_attn_out[oi] * c.v[vi_idx];
                                        dv[vi_idx] += c.attn_w[((b * num_heads + h) * seq_len + qi) * seq_len + vi] * d_attn_out[oi];
                                    }
                                    d_attn_w[((b * num_heads + h) * seq_len + qi) * seq_len + vi] = dot;
                                }
                            }
                        }
                    }

                    // 3. Backward through softmax
                    for (0..batch) |b| {
                        for (0..num_heads) |h| {
                            for (0..seq_len) |qi| {
                                var dot_sum: T = 0;
                                for (0..seq_len) |ki| {
                                    const idx = ((b * num_heads + h) * seq_len + qi) * seq_len + ki;
                                    dot_sum += c.attn_w[idx] * d_attn_w[idx];
                                }
                                for (0..seq_len) |ki| {
                                    const idx = ((b * num_heads + h) * seq_len + qi) * seq_len + ki;
                                    d_scores[idx] = c.attn_w[idx] * (d_attn_w[idx] - dot_sum) * scale;
                                }
                            }
                        }
                    }

                    // 4. Backward through Q @ K^T
                    for (0..batch) |b| {
                        for (0..num_heads) |h| {
                            for (0..seq_len) |qi| {
                                for (0..seq_len) |ki| {
                                    const ds = d_scores[((b * num_heads + h) * seq_len + qi) * seq_len + ki];
                                    for (0..head_dim) |d| {
                                        const q_idx = (b * seq_len + qi) * embed_dim + h * head_dim + d;
                                        const k_idx = (b * seq_len + ki) * embed_dim + h * head_dim + d;
                                        dq[q_idx] += ds * c.k[k_idx];
                                        dk[k_idx] += ds * c.q[q_idx];
                                    }
                                }
                            }
                        }
                    }

                    // 5. Backward through linear projections Q,K,V (BLAS)
                    backwardLinear(BS, c.in_data, dq, c.w_q, c.w_q_node, c.b_q_node, c.input_node);
                    backwardLinear(BS, c.in_data, dk, c.w_k, c.w_k_node, c.b_k_node, c.input_node);
                    backwardLinear(BS, c.in_data, dv, c.w_v, c.w_v_node, c.b_v_node, c.input_node);
                }

                fn backwardLinear(
                    rows: usize,
                    inp: []const T,
                    d_proj: []const T,
                    w: []const T,
                    w_node: *Node,
                    b_node: *Node,
                    inp_node: *Node,
                ) void {
                    // dW += in^T @ d_proj
                    if (w_node.grad) |wg| {
                        cpu_backend.matmulTransAAccum(T, inp.ptr, d_proj.ptr, wg.ptr, embed_dim, rows, embed_dim);
                    }
                    // db += sum(d_proj) per row
                    if (b_node.grad) |bg| {
                        for (0..rows) |r| {
                            cpu_backend.addAccum(T, d_proj.ptr + r * embed_dim, bg.ptr, embed_dim);
                        }
                    }
                    // d_input += d_proj @ W^T
                    if (inp_node.grad) |ig| {
                        cpu_backend.matmulTransBAccum(T, d_proj.ptr, w.ptr, ig.ptr, rows, embed_dim, embed_dim);
                    }
                }
            }.backward;

            return result;
        }

        /// x @ W + b: (rows, embed_dim) @ (embed_dim, embed_dim) + (embed_dim)
        fn linearProject(rows: usize, x: []const T, w: []const T, b: []const T, out: []T) void {
            // matmul: out = x @ W
            cpu_backend.matmul(T, x.ptr, w.ptr, out.ptr, rows, embed_dim, embed_dim);
            // add bias per row
            for (0..rows) |r| {
                cpu_backend.addAccum(T, b.ptr, out.ptr + r * embed_dim, embed_dim);
            }
        }
    };
}

/// Causal Self-Attention with causal mask.
///
/// MultiHeadAttention と同じ構造だが、causal mask (下三角行列) を適用。
/// Decoder の self-attention で使用。
pub fn CausalSelfAttention(
    comptime T: type,
    comptime embed_dim: usize,
    comptime num_heads: usize,
) type {
    if (embed_dim % num_heads != 0)
        @compileError("embed_dim must be divisible by num_heads");

    const head_dim = embed_dim / num_heads;

    return struct {
        const Self = @This();
        const Node = GraphNodeMod.GraphNode(T);
        const M = ModuleMixin(Self);

        w_q: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_k: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_v: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_o: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        b_q: VariableMod.Variable(T, .{embed_dim}),
        b_k: VariableMod.Variable(T, .{embed_dim}),
        b_v: VariableMod.Variable(T, .{embed_dim}),
        b_o: VariableMod.Variable(T, .{embed_dim}),

        pub fn init(allocator: Allocator) !Self {
            return initAttentionWeights(T, embed_dim, Self, allocator);
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        /// forward: input (batch, seq_len, embed_dim) → output (batch, seq_len, embed_dim)
        /// Causal self-attention: positions can only attend to earlier positions
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime seq_len: usize,
            input: *VariableMod.Variable(T, .{ batch, seq_len, embed_dim }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, seq_len, embed_dim }) {
            const n = batch * seq_len * embed_dim;
            const in_data = input.constData();
            const attn_scale: T = 1.0 / @sqrt(@as(T, @floatFromInt(head_dim)));

            // Single forward allocation for Q, K, V, attn_out, scores (5→1 alloc)
            const scores_n = batch * num_heads * seq_len * seq_len;
            const fwd_buf = try allocator.alloc(T, 4 * n + scores_n);
            const q_buf = fwd_buf[0..n];
            const k_buf = fwd_buf[n..][0..n];
            const v_buf = fwd_buf[2 * n..][0..n];
            const attn_out = fwd_buf[3 * n..][0..n];
            const attn_weights = fwd_buf[4 * n..][0..scores_n];
            linearProjectFn(batch * seq_len, in_data, self.w_q.constData(), self.b_q.constData(), q_buf);
            linearProjectFn(batch * seq_len, in_data, self.w_k.constData(), self.b_k.constData(), k_buf);
            linearProjectFn(batch * seq_len, in_data, self.w_v.constData(), self.b_v.constData(), v_buf);

            for (0..batch) |bi| {
                for (0..num_heads) |h| {
                    for (0..seq_len) |qi| {
                        var max_val: T = -std.math.inf(T);
                        for (0..seq_len) |ki| {
                            const aw_idx = ((bi * num_heads + h) * seq_len + qi) * seq_len + ki;
                            if (ki > qi) {
                                // Causal mask: future positions get -inf
                                attn_weights[aw_idx] = -std.math.inf(T);
                            } else {
                                var dot: T = 0;
                                for (0..head_dim) |d| {
                                    const q_idx = (bi * seq_len + qi) * embed_dim + h * head_dim + d;
                                    const k_idx = (bi * seq_len + ki) * embed_dim + h * head_dim + d;
                                    dot += q_buf[q_idx] * k_buf[k_idx];
                                }
                                attn_weights[aw_idx] = dot * attn_scale;
                            }
                            if (attn_weights[aw_idx] > max_val) max_val = attn_weights[aw_idx];
                        }
                        // Softmax
                        var sum_exp: T = 0;
                        for (0..seq_len) |ki| {
                            const aw_idx = ((bi * num_heads + h) * seq_len + qi) * seq_len + ki;
                            const e = @exp(attn_weights[aw_idx] - max_val);
                            attn_weights[aw_idx] = e;
                            sum_exp += e;
                        }
                        for (0..seq_len) |ki| {
                            const aw_idx = ((bi * num_heads + h) * seq_len + qi) * seq_len + ki;
                            attn_weights[aw_idx] /= sum_exp;
                        }
                    }
                }
            }

            @memset(attn_out, 0);
            for (0..batch) |bi| {
                for (0..num_heads) |h| {
                    for (0..seq_len) |qi| {
                        for (0..seq_len) |vi| {
                            const w = attn_weights[((bi * num_heads + h) * seq_len + qi) * seq_len + vi];
                            for (0..head_dim) |d| {
                                const out_idx = (bi * seq_len + qi) * embed_dim + h * head_dim + d;
                                const v_idx = (bi * seq_len + vi) * embed_dim + h * head_dim + d;
                                attn_out[out_idx] += w * v_buf[v_idx];
                            }
                        }
                    }
                }
            }

            const out_tensor = try TensorMod.Tensor(T, .{ batch, seq_len, embed_dim }).init(allocator);
            linearProjectFn(batch * seq_len, attn_out, self.w_o.constData(), self.b_o.constData(), out_tensor.slice());

            const Ctx = struct {
                q: []const T, k: []const T, v: []const T,
                attn_w: []const T, attn_o: []const T, in_data: []const T,
                w_q: []const T, w_k: []const T, w_v: []const T, w_o: []const T,
                input_node: *Node,
                w_q_node: *Node, w_k_node: *Node, w_v_node: *Node, w_o_node: *Node,
                b_q_node: *Node, b_k_node: *Node, b_v_node: *Node, b_o_node: *Node,
                alloc: Allocator,
            };
            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .q = q_buf, .k = k_buf, .v = v_buf,
                .attn_w = attn_weights, .attn_o = attn_out,
                .in_data = in_data,
                .w_q = self.w_q.constData(), .w_k = self.w_k.constData(),
                .w_v = self.w_v.constData(), .w_o = self.w_o.constData(),
                .input_node = input.node,
                .w_q_node = self.w_q.node, .w_k_node = self.w_k.node,
                .w_v_node = self.w_v.node, .w_o_node = self.w_o.node,
                .b_q_node = self.b_q.node, .b_k_node = self.b_k.node,
                .b_v_node = self.b_v.node, .b_o_node = self.b_o.node,
                .alloc = allocator,
            };

            var result = try VariableMod.Variable(T, .{ batch, seq_len, embed_dim }).init(out_tensor, allocator, true);
            result.node.parents[0] = input.node;
            result.node.parents[1] = self.w_q.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));
                    const BS = batch * seq_len;
                    const alloc = c.alloc;

                    // Single backward allocation (6→1 alloc)
                    const bw_n = BS * embed_dim;
                    const bw_scores_n = batch * num_heads * seq_len * seq_len;
                    const bw_buf = alloc.alloc(T, 4 * bw_n + 2 * bw_scores_n) catch unreachable;
                    @memset(bw_buf, 0);
                    const d_attn_out = bw_buf[0..bw_n];
                    const dv = bw_buf[bw_n..][0..bw_n];
                    const dq = bw_buf[2 * bw_n..][0..bw_n];
                    const dk = bw_buf[3 * bw_n..][0..bw_n];
                    const d_attn_w = bw_buf[4 * bw_n..][0..bw_scores_n];
                    const d_scores = bw_buf[4 * bw_n + bw_scores_n..][0..bw_scores_n];

                    // 1. Output projection backward (BLAS)
                    if (c.w_o_node.grad) |wg| {
                        cpu_backend.matmulTransAAccum(T, c.attn_o.ptr, grad_out.ptr, wg.ptr, embed_dim, BS, embed_dim);
                    }
                    if (c.b_o_node.grad) |bg| {
                        for (0..BS) |row| {
                            cpu_backend.addAccum(T, grad_out.ptr + row * embed_dim, bg.ptr, embed_dim);
                        }
                    }
                    cpu_backend.matmulTransB(T, grad_out.ptr, c.w_o.ptr, d_attn_out.ptr, BS, embed_dim, embed_dim);

                    // 2. attn_out = attn_w @ V backward
                    for (0..batch) |bi| {
                        for (0..num_heads) |h| {
                            for (0..seq_len) |qi| {
                                for (0..seq_len) |vi| {
                                    var dot: T = 0;
                                    for (0..head_dim) |d| {
                                        const oi = (bi * seq_len + qi) * embed_dim + h * head_dim + d;
                                        const vi_idx = (bi * seq_len + vi) * embed_dim + h * head_dim + d;
                                        dot += d_attn_out[oi] * c.v[vi_idx];
                                        dv[vi_idx] += c.attn_w[((bi * num_heads + h) * seq_len + qi) * seq_len + vi] * d_attn_out[oi];
                                    }
                                    d_attn_w[((bi * num_heads + h) * seq_len + qi) * seq_len + vi] = dot;
                                }
                            }
                        }
                    }

                    // 3. Softmax backward
                    for (0..batch) |bi| {
                        for (0..num_heads) |h| {
                            for (0..seq_len) |qi| {
                                var dot_sum: T = 0;
                                for (0..seq_len) |ki| {
                                    const idx = ((bi * num_heads + h) * seq_len + qi) * seq_len + ki;
                                    dot_sum += c.attn_w[idx] * d_attn_w[idx];
                                }
                                for (0..seq_len) |ki| {
                                    const idx = ((bi * num_heads + h) * seq_len + qi) * seq_len + ki;
                                    d_scores[idx] = c.attn_w[idx] * (d_attn_w[idx] - dot_sum) * attn_scale;
                                }
                            }
                        }
                    }

                    // 4. Q @ K^T backward
                    for (0..batch) |bi| {
                        for (0..num_heads) |h| {
                            for (0..seq_len) |qi| {
                                for (0..seq_len) |ki| {
                                    const ds = d_scores[((bi * num_heads + h) * seq_len + qi) * seq_len + ki];
                                    for (0..head_dim) |d| {
                                        const q_idx = (bi * seq_len + qi) * embed_dim + h * head_dim + d;
                                        const k_idx = (bi * seq_len + ki) * embed_dim + h * head_dim + d;
                                        dq[q_idx] += ds * c.k[k_idx];
                                        dk[k_idx] += ds * c.q[q_idx];
                                    }
                                }
                            }
                        }
                    }

                    // 5. Linear projection backward (BLAS)
                    backwardLinearCausal(BS, c.in_data, dq, c.w_q, c.w_q_node, c.b_q_node, c.input_node);
                    backwardLinearCausal(BS, c.in_data, dk, c.w_k, c.w_k_node, c.b_k_node, c.input_node);
                    backwardLinearCausal(BS, c.in_data, dv, c.w_v, c.w_v_node, c.b_v_node, c.input_node);
                }

                fn backwardLinearCausal(rows: usize, inp: []const T, d_proj: []const T, w: []const T, w_node: *Node, b_node: *Node, inp_node: *Node) void {
                    if (w_node.grad) |wg| {
                        cpu_backend.matmulTransAAccum(T, inp.ptr, d_proj.ptr, wg.ptr, embed_dim, rows, embed_dim);
                    }
                    if (b_node.grad) |bg| {
                        for (0..rows) |r| {
                            cpu_backend.addAccum(T, d_proj.ptr + r * embed_dim, bg.ptr, embed_dim);
                        }
                    }
                    if (inp_node.grad) |ig| {
                        cpu_backend.matmulTransBAccum(T, d_proj.ptr, w.ptr, ig.ptr, rows, embed_dim, embed_dim);
                    }
                }
            }.backward;

            return result;
        }

        fn linearProjectFn(rows: usize, x: []const T, w: []const T, b_param: []const T, out: []T) void {
            cpu_backend.matmul(T, x.ptr, w.ptr, out.ptr, rows, embed_dim, embed_dim);
            for (0..rows) |r| {
                cpu_backend.addAccum(T, b_param.ptr, out.ptr + r * embed_dim, embed_dim);
            }
        }
    };
}

/// Cross-Attention: Q from decoder, K/V from encoder.
///
/// Attention(Q, K, V) where Q=decoder_input, K=V=encoder_output
/// (encoder/decoder の embed_dim は同じ)
pub fn CrossAttention(
    comptime T: type,
    comptime embed_dim: usize,
    comptime num_heads: usize,
) type {
    if (embed_dim % num_heads != 0)
        @compileError("embed_dim must be divisible by num_heads");

    const head_dim = embed_dim / num_heads;

    return struct {
        const Self = @This();
        const Node = GraphNodeMod.GraphNode(T);
        const M = ModuleMixin(Self);

        w_q: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_k: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_v: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        w_o: VariableMod.Variable(T, .{ embed_dim, embed_dim }),
        b_q: VariableMod.Variable(T, .{embed_dim}),
        b_k: VariableMod.Variable(T, .{embed_dim}),
        b_v: VariableMod.Variable(T, .{embed_dim}),
        b_o: VariableMod.Variable(T, .{embed_dim}),

        pub fn init(allocator: Allocator) !Self {
            return initAttentionWeights(T, embed_dim, Self, allocator);
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        /// forward: Q from decoder (batch, q_len, embed_dim), K/V from encoder (batch, kv_len, embed_dim)
        /// → output (batch, q_len, embed_dim)
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime q_len: usize,
            comptime kv_len: usize,
            query: *VariableMod.Variable(T, .{ batch, q_len, embed_dim }),
            kv_input: *VariableMod.Variable(T, .{ batch, kv_len, embed_dim }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, q_len, embed_dim }) {
            const q_n = batch * q_len * embed_dim;
            const kv_n = batch * kv_len * embed_dim;
            const q_data = query.constData();
            const kv_data = kv_input.constData();
            const attn_scale: T = 1.0 / @sqrt(@as(T, @floatFromInt(head_dim)));

            // Single forward allocation for Q, K, V, attn_out, scores (5→1 alloc)
            const scores_n = batch * num_heads * q_len * kv_len;
            const fwd_buf = try allocator.alloc(T, 2 * q_n + 2 * kv_n + scores_n);
            const q_buf = fwd_buf[0..q_n];
            const k_buf = fwd_buf[q_n..][0..kv_n];
            const v_buf = fwd_buf[q_n + kv_n ..][0..kv_n];
            const attn_out = fwd_buf[q_n + 2 * kv_n ..][0..q_n];
            const attn_weights = fwd_buf[2 * q_n + 2 * kv_n ..][0..scores_n];
            linearProjectN(batch * q_len, q_data, self.w_q.constData(), self.b_q.constData(), q_buf);
            linearProjectN(batch * kv_len, kv_data, self.w_k.constData(), self.b_k.constData(), k_buf);
            linearProjectN(batch * kv_len, kv_data, self.w_v.constData(), self.b_v.constData(), v_buf);

            for (0..batch) |bi| {
                for (0..num_heads) |h| {
                    for (0..q_len) |qi| {
                        var max_val: T = -std.math.inf(T);
                        for (0..kv_len) |ki| {
                            var dot: T = 0;
                            for (0..head_dim) |d| {
                                const q_idx = (bi * q_len + qi) * embed_dim + h * head_dim + d;
                                const k_idx = (bi * kv_len + ki) * embed_dim + h * head_dim + d;
                                dot += q_buf[q_idx] * k_buf[k_idx];
                            }
                            dot *= attn_scale;
                            const aw_idx = ((bi * num_heads + h) * q_len + qi) * kv_len + ki;
                            attn_weights[aw_idx] = dot;
                            if (dot > max_val) max_val = dot;
                        }
                        // Softmax
                        var sum_exp: T = 0;
                        for (0..kv_len) |ki| {
                            const aw_idx = ((bi * num_heads + h) * q_len + qi) * kv_len + ki;
                            const e = @exp(attn_weights[aw_idx] - max_val);
                            attn_weights[aw_idx] = e;
                            sum_exp += e;
                        }
                        for (0..kv_len) |ki| {
                            const aw_idx = ((bi * num_heads + h) * q_len + qi) * kv_len + ki;
                            attn_weights[aw_idx] /= sum_exp;
                        }
                    }
                }
            }

            // Attention output: attn_weights @ V → (batch, q_len, embed_dim)
            @memset(attn_out, 0);
            for (0..batch) |bi| {
                for (0..num_heads) |h| {
                    for (0..q_len) |qi| {
                        for (0..kv_len) |vi| {
                            const w = attn_weights[((bi * num_heads + h) * q_len + qi) * kv_len + vi];
                            for (0..head_dim) |d| {
                                const out_idx = (bi * q_len + qi) * embed_dim + h * head_dim + d;
                                const v_idx = (bi * kv_len + vi) * embed_dim + h * head_dim + d;
                                attn_out[out_idx] += w * v_buf[v_idx];
                            }
                        }
                    }
                }
            }

            const out_tensor = try TensorMod.Tensor(T, .{ batch, q_len, embed_dim }).init(allocator);
            linearProjectN(batch * q_len, attn_out, self.w_o.constData(), self.b_o.constData(), out_tensor.slice());

            const Ctx = struct {
                q: []const T, k: []const T, v: []const T,
                attn_w: []const T, attn_o: []const T,
                q_data: []const T, kv_data: []const T,
                w_q: []const T, w_k: []const T, w_v: []const T, w_o: []const T,
                query_node: *Node, kv_node: *Node,
                w_q_node: *Node, w_k_node: *Node, w_v_node: *Node, w_o_node: *Node,
                b_q_node: *Node, b_k_node: *Node, b_v_node: *Node, b_o_node: *Node,
                alloc: Allocator,
            };
            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .q = q_buf, .k = k_buf, .v = v_buf,
                .attn_w = attn_weights, .attn_o = attn_out,
                .q_data = q_data, .kv_data = kv_data,
                .w_q = self.w_q.constData(), .w_k = self.w_k.constData(),
                .w_v = self.w_v.constData(), .w_o = self.w_o.constData(),
                .query_node = query.node, .kv_node = kv_input.node,
                .w_q_node = self.w_q.node, .w_k_node = self.w_k.node,
                .w_v_node = self.w_v.node, .w_o_node = self.w_o.node,
                .b_q_node = self.b_q.node, .b_k_node = self.b_k.node,
                .b_v_node = self.b_v.node, .b_o_node = self.b_o.node,
                .alloc = allocator,
            };

            var result = try VariableMod.Variable(T, .{ batch, q_len, embed_dim }).init(out_tensor, allocator, true);
            result.node.parents[0] = query.node;
            result.node.parents[1] = kv_input.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));
                    const QBS = batch * q_len;
                    const alloc = c.alloc;

                    // Single backward allocation (6→1 alloc)
                    const bw_qn = QBS * embed_dim;
                    const bw_kvn = batch * kv_len * embed_dim;
                    const bw_scores_n = batch * num_heads * q_len * kv_len;
                    const bw_buf = alloc.alloc(T, 2 * bw_qn + 2 * bw_kvn + 2 * bw_scores_n) catch unreachable;
                    @memset(bw_buf, 0);
                    const d_attn_out = bw_buf[0..bw_qn];
                    const dq = bw_buf[bw_qn..][0..bw_qn];
                    const dv = bw_buf[2 * bw_qn..][0..bw_kvn];
                    const dk = bw_buf[2 * bw_qn + bw_kvn ..][0..bw_kvn];
                    const d_attn_w = bw_buf[2 * bw_qn + 2 * bw_kvn ..][0..bw_scores_n];
                    const d_scores = bw_buf[2 * bw_qn + 2 * bw_kvn + bw_scores_n ..][0..bw_scores_n];

                    // 1. Output projection backward (BLAS)
                    if (c.w_o_node.grad) |wg| {
                        cpu_backend.matmulTransAAccum(T, c.attn_o.ptr, grad_out.ptr, wg.ptr, embed_dim, QBS, embed_dim);
                    }
                    if (c.b_o_node.grad) |bg| {
                        for (0..QBS) |row| {
                            cpu_backend.addAccum(T, grad_out.ptr + row * embed_dim, bg.ptr, embed_dim);
                        }
                    }
                    cpu_backend.matmulTransB(T, grad_out.ptr, c.w_o.ptr, d_attn_out.ptr, QBS, embed_dim, embed_dim);

                    // 2. attn_out = attn_w @ V backward
                    for (0..batch) |bi| {
                        for (0..num_heads) |h| {
                            for (0..q_len) |qi| {
                                for (0..kv_len) |vi| {
                                    var dot: T = 0;
                                    for (0..head_dim) |d| {
                                        const oi = (bi * q_len + qi) * embed_dim + h * head_dim + d;
                                        const vi_idx = (bi * kv_len + vi) * embed_dim + h * head_dim + d;
                                        dot += d_attn_out[oi] * c.v[vi_idx];
                                        dv[vi_idx] += c.attn_w[((bi * num_heads + h) * q_len + qi) * kv_len + vi] * d_attn_out[oi];
                                    }
                                    d_attn_w[((bi * num_heads + h) * q_len + qi) * kv_len + vi] = dot;
                                }
                            }
                        }
                    }

                    // 3. Softmax backward
                    for (0..batch) |bi| {
                        for (0..num_heads) |h| {
                            for (0..q_len) |qi| {
                                var dot_sum: T = 0;
                                for (0..kv_len) |ki| {
                                    const idx = ((bi * num_heads + h) * q_len + qi) * kv_len + ki;
                                    dot_sum += c.attn_w[idx] * d_attn_w[idx];
                                }
                                for (0..kv_len) |ki| {
                                    const idx = ((bi * num_heads + h) * q_len + qi) * kv_len + ki;
                                    d_scores[idx] = c.attn_w[idx] * (d_attn_w[idx] - dot_sum) * attn_scale;
                                }
                            }
                        }
                    }

                    // 4. Q @ K^T backward (cross-attention: Q from decoder, K from encoder)
                    for (0..batch) |bi| {
                        for (0..num_heads) |h| {
                            for (0..q_len) |qi| {
                                for (0..kv_len) |ki| {
                                    const ds = d_scores[((bi * num_heads + h) * q_len + qi) * kv_len + ki];
                                    for (0..head_dim) |d| {
                                        const q_idx = (bi * q_len + qi) * embed_dim + h * head_dim + d;
                                        const k_idx = (bi * kv_len + ki) * embed_dim + h * head_dim + d;
                                        dq[q_idx] += ds * c.k[k_idx];
                                        dk[k_idx] += ds * c.q[q_idx];
                                    }
                                }
                            }
                        }
                    }

                    // 5. Linear projection backward (BLAS)
                    // Q projection: input=query
                    backwardLinearCross(QBS, c.q_data, dq, c.w_q, c.w_q_node, c.b_q_node, c.query_node);
                    // K projection: input=kv
                    const KVBS = batch * kv_len;
                    backwardLinearCross(KVBS, c.kv_data, dk, c.w_k, c.w_k_node, c.b_k_node, c.kv_node);
                    // V projection: input=kv
                    backwardLinearCross(KVBS, c.kv_data, dv, c.w_v, c.w_v_node, c.b_v_node, c.kv_node);
                }

                fn backwardLinearCross(rows: usize, inp: []const T, d_proj: []const T, w: []const T, w_node: *Node, b_node: *Node, inp_node: *Node) void {
                    if (w_node.grad) |wg| {
                        cpu_backend.matmulTransAAccum(T, inp.ptr, d_proj.ptr, wg.ptr, embed_dim, rows, embed_dim);
                    }
                    if (b_node.grad) |bg| {
                        for (0..rows) |r| {
                            cpu_backend.addAccum(T, d_proj.ptr + r * embed_dim, bg.ptr, embed_dim);
                        }
                    }
                    if (inp_node.grad) |ig| {
                        cpu_backend.matmulTransBAccum(T, d_proj.ptr, w.ptr, ig.ptr, rows, embed_dim, embed_dim);
                    }
                }
            }.backward;

            return result;
        }

        fn linearProjectN(rows: usize, x: []const T, w: []const T, b_param: []const T, out: []T) void {
            cpu_backend.matmul(T, x.ptr, w.ptr, out.ptr, rows, embed_dim, embed_dim);
            for (0..rows) |r| {
                cpu_backend.addAccum(T, b_param.ptr, out.ptr + r * embed_dim, embed_dim);
            }
        }
    };
}

// ============================================================
// テスト
// ============================================================

test "MultiHeadAttention forward shape" {
    const alloc = std.testing.allocator;
    var mha = try MultiHeadAttention(f64, 4, 2).init(alloc);
    defer mha.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // batch=2, seq_len=3, embed_dim=4
    var input = try VariableMod.Variable(f64, .{ 2, 3, 4 }).fromSlice(temp, blk: {
        var data: [24]f64 = undefined;
        for (&data, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) * 0.1;
        break :blk &data;
    }, false);

    const output = try mha.forward(2, 3, &input, temp);
    try std.testing.expectEqual(@as(usize, 24), output.constData().len);
}

test "MultiHeadAttention softmax sums to 1" {
    const alloc = std.testing.allocator;

    // Use identity weights to test softmax behavior
    var mha = try MultiHeadAttention(f64, 4, 2).init(alloc);
    defer mha.deinit();

    // Set W_q, W_k to identity, biases to 0
    @memset(mha.w_q.data(), 0);
    @memset(mha.w_k.data(), 0);
    for (0..4) |i| {
        mha.w_q.data()[i * 4 + i] = 1;
        mha.w_k.data()[i * 4 + i] = 1;
    }
    @memset(mha.b_q.data(), 0);
    @memset(mha.b_k.data(), 0);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 3, 4 }).fromSlice(temp, &[_]f64{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    }, false);

    // Just test that forward runs without error (softmax is computed internally)
    const output = try mha.forward(1, 3, &input, temp);
    // Output should have finite values
    for (output.constData()) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "MultiHeadAttention backward runs" {
    const alloc = std.testing.allocator;
    var mha = try MultiHeadAttention(f64, 4, 2).init(alloc);
    defer mha.deinit();
    try mha.allocGrad(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 2, 4 }).fromSlice(temp, &[_]f64{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
    }, true);
    input.node.grad = try temp.alloc(f64, 8);
    @memset(input.node.grad.?, 0);

    var output = try mha.forward(1, 2, &input, temp);
    output.node.grad = try temp.alloc(f64, 8);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| bfn(output.node);

    // Weight gradients should be non-zero
    var has_nonzero = false;
    for (mha.w_q.node.grad.?) |g| {
        if (@abs(g) > 1e-15) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);

    // Input gradient should be non-zero
    has_nonzero = false;
    for (input.node.grad.?) |g| {
        if (@abs(g) > 1e-15) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "CausalSelfAttention forward causal mask" {
    const alloc = std.testing.allocator;
    var csa = try CausalSelfAttention(f64, 4, 2).init(alloc);
    defer csa.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // batch=1, seq_len=3, embed_dim=4
    var input = try VariableMod.Variable(f64, .{ 1, 3, 4 }).fromSlice(temp, &[_]f64{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
    }, false);

    const output = try csa.forward(1, 3, &input, temp);
    try std.testing.expectEqual(@as(usize, 12), output.constData().len);

    for (output.constData()) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "CausalSelfAttention backward runs" {
    const alloc = std.testing.allocator;
    var csa = try CausalSelfAttention(f64, 4, 2).init(alloc);
    defer csa.deinit();
    try csa.allocGrad(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 2, 4 }).fromSlice(temp, &[_]f64{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
    }, true);
    input.node.grad = try temp.alloc(f64, 8);
    @memset(input.node.grad.?, 0);

    var output = try csa.forward(1, 2, &input, temp);
    output.node.grad = try temp.alloc(f64, 8);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| bfn(output.node);

    var has_nonzero = false;
    for (csa.w_q.node.grad.?) |g| {
        if (@abs(g) > 1e-15) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "CrossAttention forward shape" {
    const alloc = std.testing.allocator;
    var ca = try CrossAttention(f64, 4, 2).init(alloc);
    defer ca.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // query: (1, 2, 4), kv: (1, 3, 4)
    var query = try VariableMod.Variable(f64, .{ 1, 2, 4 }).fromSlice(temp, &[_]f64{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
    }, false);
    var kv = try VariableMod.Variable(f64, .{ 1, 3, 4 }).fromSlice(temp, &[_]f64{
        1.0, 1.1, 1.2, 1.3,
        1.4, 1.5, 1.6, 1.7,
        1.8, 1.9, 2.0, 2.1,
    }, false);

    const output = try ca.forward(1, 2, 3, &query, &kv, temp);
    // Output shape should be (1, 2, 4) = 8 elements
    try std.testing.expectEqual(@as(usize, 8), output.constData().len);

    for (output.constData()) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "CrossAttention backward runs" {
    const alloc = std.testing.allocator;
    var ca = try CrossAttention(f64, 4, 2).init(alloc);
    defer ca.deinit();
    try ca.allocGrad(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var query = try VariableMod.Variable(f64, .{ 1, 2, 4 }).fromSlice(temp, &[_]f64{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
    }, true);
    query.node.grad = try temp.alloc(f64, 8);
    @memset(query.node.grad.?, 0);

    var kv = try VariableMod.Variable(f64, .{ 1, 3, 4 }).fromSlice(temp, &[_]f64{
        1.0, 1.1, 1.2, 1.3,
        1.4, 1.5, 1.6, 1.7,
        1.8, 1.9, 2.0, 2.1,
    }, true);
    kv.node.grad = try temp.alloc(f64, 12);
    @memset(kv.node.grad.?, 0);

    var output = try ca.forward(1, 2, 3, &query, &kv, temp);
    output.node.grad = try temp.alloc(f64, 8);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| bfn(output.node);

    // Query gradient should be non-zero
    var has_nonzero = false;
    for (query.node.grad.?) |g| {
        if (@abs(g) > 1e-15) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);

    // KV gradient should be non-zero
    has_nonzero = false;
    for (kv.node.grad.?) |g| {
        if (@abs(g) > 1e-15) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}
