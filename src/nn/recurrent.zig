const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");
const ModuleMixin = @import("module.zig").Module;

// ============================================================
// LSTM
// ============================================================

/// LSTM (Long Short-Term Memory)。
///
/// ゲート:
///   i = σ(W_ih * x + W_hh * h + b_ih + b_hh)
///   f = σ(...)
///   g = tanh(...)
///   o = σ(...)
///   c' = f * c + i * g
///   h' = o * tanh(c')
pub fn LSTM(comptime T: type, comptime input_size: usize, comptime hidden_size: usize) type {
    const gate_size = 4 * hidden_size;

    return struct {
        const Self = @This();
        const Node = GraphNodeMod.GraphNode(T);
        const M = ModuleMixin(Self);

        w_ih: VariableMod.Variable(T, .{ gate_size, input_size }),
        w_hh: VariableMod.Variable(T, .{ gate_size, hidden_size }),
        b_ih: VariableMod.Variable(T, .{gate_size}),
        b_hh: VariableMod.Variable(T, .{gate_size}),

        pub fn init(allocator: Allocator) !Self {
            const k: T = 1.0 / @sqrt(@as(T, @floatFromInt(hidden_size)));
            var prng = std.Random.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                std.posix.getrandom(std.mem.asBytes(&seed)) catch { seed = 42; };
                break :blk seed;
            });
            const rng = prng.random();

            const w_ih_t = try TensorMod.Tensor(T, .{ gate_size, input_size }).init(allocator);
            for (w_ih_t.slice()) |*v| v.* = (rng.float(T) * 2.0 - 1.0) * k;
            const w_ih = try VariableMod.Variable(T, .{ gate_size, input_size }).init(w_ih_t, allocator, true);

            const w_hh_t = try TensorMod.Tensor(T, .{ gate_size, hidden_size }).init(allocator);
            for (w_hh_t.slice()) |*v| v.* = (rng.float(T) * 2.0 - 1.0) * k;
            const w_hh = try VariableMod.Variable(T, .{ gate_size, hidden_size }).init(w_hh_t, allocator, true);

            const b_ih_t = try TensorMod.Tensor(T, .{gate_size}).zeros(allocator);
            const b_ih = try VariableMod.Variable(T, .{gate_size}).init(b_ih_t, allocator, true);

            const b_hh_t = try TensorMod.Tensor(T, .{gate_size}).zeros(allocator);
            const b_hh = try VariableMod.Variable(T, .{gate_size}).init(b_hh_t, allocator, true);

            return .{ .w_ih = w_ih, .w_hh = w_hh, .b_ih = b_ih, .b_hh = b_hh };
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        /// forward: input (batch, seq_len, input_size) → output (batch, seq_len, hidden_size)
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime seq_len: usize,
            input: *VariableMod.Variable(T, .{ batch, seq_len, input_size }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, seq_len, hidden_size }) {
            const out_tensor = try TensorMod.Tensor(T, .{ batch, seq_len, hidden_size }).init(allocator);
            const out_data = out_tensor.slice();
            const in_data = input.constData();
            const w_ih_d = self.w_ih.constData();
            const w_hh_d = self.w_hh.constData();
            const b_ih_d = self.b_ih.constData();
            const b_hh_d = self.b_hh.constData();

            // Saved for backward: activated gates(i,f,g,o), c_states, h_states
            const total = batch * seq_len;
            const gates_buf = try allocator.alloc(T, total * gate_size);
            // c_states: index 0..batch*H = c0 (zeros), then total*H for each step
            const c_all = try allocator.alloc(T, (batch + total) * hidden_size);
            @memset(c_all[0 .. batch * hidden_size], 0); // c0 = 0
            // h_states: same layout, h0 = 0
            const h_all = try allocator.alloc(T, (batch + total) * hidden_size);
            @memset(h_all[0 .. batch * hidden_size], 0); // h0 = 0
            const tanh_c = try allocator.alloc(T, total * hidden_size);

            for (0..seq_len) |t| {
                for (0..batch) |b| {
                    const si = b * seq_len + t; // step index
                    const x_off = si * input_size;
                    const h_prev_off = (if (t == 0) b else batch + b * seq_len + (t - 1)) * hidden_size;
                    const c_prev_off = h_prev_off; // same layout
                    const c_curr_off = (batch + si) * hidden_size;
                    const h_curr_off = c_curr_off;

                    // Compute raw gates: W_ih @ x + W_hh @ h_prev + b
                    for (0..gate_size) |g| {
                        var val: T = b_ih_d[g] + b_hh_d[g];
                        for (0..input_size) |kk| {
                            val += w_ih_d[g * input_size + kk] * in_data[x_off + kk];
                        }
                        for (0..hidden_size) |kk| {
                            val += w_hh_d[g * hidden_size + kk] * h_all[h_prev_off + kk];
                        }
                        gates_buf[si * gate_size + g] = val;
                    }

                    // Apply activations and compute c', h'
                    for (0..hidden_size) |h| {
                        const go = si * gate_size;
                        const ig = sigm(gates_buf[go + h]);
                        const fg = sigm(gates_buf[go + hidden_size + h]);
                        const gg = tanhFn(gates_buf[go + 2 * hidden_size + h]);
                        const og = sigm(gates_buf[go + 3 * hidden_size + h]);

                        // Store activated gates
                        gates_buf[go + h] = ig;
                        gates_buf[go + hidden_size + h] = fg;
                        gates_buf[go + 2 * hidden_size + h] = gg;
                        gates_buf[go + 3 * hidden_size + h] = og;

                        const cp = c_all[c_prev_off + h];
                        const cn = fg * cp + ig * gg;
                        c_all[c_curr_off + h] = cn;

                        const tc = tanhFn(cn);
                        tanh_c[si * hidden_size + h] = tc;
                        const hn = og * tc;
                        h_all[h_curr_off + h] = hn;
                        out_data[si * hidden_size + h] = hn;
                    }
                }
            }

            const Ctx = struct {
                gates: []const T,
                c_all: []const T,
                h_all: []const T,
                tanh_c: []const T,
                in_data: []const T,
                w_ih: []const T,
                w_hh: []const T,
                input_node: *Node,
                w_ih_node: *Node,
                w_hh_node: *Node,
                b_ih_node: *Node,
                b_hh_node: *Node,
            };
            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .gates = gates_buf, .c_all = c_all, .h_all = h_all,
                .tanh_c = tanh_c, .in_data = in_data,
                .w_ih = w_ih_d, .w_hh = w_hh_d,
                .input_node = input.node, .w_ih_node = self.w_ih.node,
                .w_hh_node = self.w_hh.node, .b_ih_node = self.b_ih.node,
                .b_hh_node = self.b_hh.node,
            };

            const OutVar = VariableMod.Variable(T, .{ batch, seq_len, hidden_size });
            var result = try OutVar.init(out_tensor, allocator, true);
            result.node.parents[0] = input.node;
            result.node.parents[1] = self.w_ih.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));

                    var dh_next: [batch * hidden_size]T = @splat(0);
                    var dc_next: [batch * hidden_size]T = @splat(0);

                    var ti: usize = seq_len;
                    while (ti > 0) {
                        ti -= 1;
                        for (0..batch) |bb| {
                            const si = bb * seq_len + ti;
                            const go = si * gate_size;
                            const x_off = si * input_size;
                            const h_off = bb * hidden_size;
                            const c_prev_off = (if (ti == 0) bb else batch + bb * seq_len + (ti - 1)) * hidden_size;
                            const h_prev_off = c_prev_off;

                            // Gate derivatives
                            var dg_buf: [gate_size]T = undefined;
                            for (0..hidden_size) |h| {
                                const dh = grad_out[si * hidden_size + h] + dh_next[h_off + h];
                                const ig = c.gates[go + h];
                                const fg = c.gates[go + hidden_size + h];
                                const gg = c.gates[go + 2 * hidden_size + h];
                                const og = c.gates[go + 3 * hidden_size + h];
                                const tc = c.tanh_c[si * hidden_size + h];
                                const cp = c.c_all[c_prev_off + h];

                                const do_raw = dh * tc * og * (1.0 - og);
                                const dc = dh * og * (1.0 - tc * tc) + dc_next[h_off + h];
                                const di_raw = dc * gg * ig * (1.0 - ig);
                                const df_raw = dc * cp * fg * (1.0 - fg);
                                const dg_raw = dc * ig * (1.0 - gg * gg);

                                dg_buf[h] = di_raw;
                                dg_buf[hidden_size + h] = df_raw;
                                dg_buf[2 * hidden_size + h] = dg_raw;
                                dg_buf[3 * hidden_size + h] = do_raw;

                                dc_next[h_off + h] = dc * fg;
                            }

                            // dW_ih += dg @ x^T
                            if (c.w_ih_node.grad) |wg| {
                                for (0..gate_size) |g| {
                                    for (0..input_size) |kk| {
                                        wg[g * input_size + kk] += dg_buf[g] * c.in_data[x_off + kk];
                                    }
                                }
                            }
                            // dW_hh += dg @ h_prev^T
                            if (c.w_hh_node.grad) |wg| {
                                for (0..gate_size) |g| {
                                    for (0..hidden_size) |kk| {
                                        wg[g * hidden_size + kk] += dg_buf[g] * c.h_all[h_prev_off + kk];
                                    }
                                }
                            }
                            // dBias
                            if (c.b_ih_node.grad) |bg| {
                                for (0..gate_size) |g| bg[g] += dg_buf[g];
                            }
                            if (c.b_hh_node.grad) |bg| {
                                for (0..gate_size) |g| bg[g] += dg_buf[g];
                            }
                            // dInput
                            if (c.input_node.grad) |ig| {
                                for (0..gate_size) |g| {
                                    for (0..input_size) |kk| {
                                        ig[x_off + kk] += dg_buf[g] * c.w_ih[g * input_size + kk];
                                    }
                                }
                            }
                            // dh_next = W_hh^T @ dg
                            @memset(dh_next[h_off .. h_off + hidden_size], 0);
                            for (0..gate_size) |g| {
                                for (0..hidden_size) |kk| {
                                    dh_next[h_off + kk] += dg_buf[g] * c.w_hh[g * hidden_size + kk];
                                }
                            }
                        }
                    }
                }
            }.backward;

            return result;
        }
    };
}

// ============================================================
// GRU
// ============================================================

/// GRU (Gated Recurrent Unit)。
///
///   r = σ(W_ir*x + b_ir + W_hr*h + b_hr)
///   z = σ(W_iz*x + b_iz + W_hz*h + b_hz)
///   n = tanh(W_in*x + b_in + r*(W_hn*h + b_hn))
///   h' = (1-z)*n + z*h
pub fn GRU(comptime T: type, comptime input_size: usize, comptime hidden_size: usize) type {
    const gate_size = 3 * hidden_size;

    return struct {
        const Self = @This();
        const Node = GraphNodeMod.GraphNode(T);
        const M = ModuleMixin(Self);

        w_ih: VariableMod.Variable(T, .{ gate_size, input_size }),
        w_hh: VariableMod.Variable(T, .{ gate_size, hidden_size }),
        b_ih: VariableMod.Variable(T, .{gate_size}),
        b_hh: VariableMod.Variable(T, .{gate_size}),

        pub fn init(allocator: Allocator) !Self {
            const k: T = 1.0 / @sqrt(@as(T, @floatFromInt(hidden_size)));
            var prng = std.Random.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                std.posix.getrandom(std.mem.asBytes(&seed)) catch { seed = 42; };
                break :blk seed;
            });
            const rng = prng.random();

            const w_ih_t = try TensorMod.Tensor(T, .{ gate_size, input_size }).init(allocator);
            for (w_ih_t.slice()) |*v| v.* = (rng.float(T) * 2.0 - 1.0) * k;
            const w_ih = try VariableMod.Variable(T, .{ gate_size, input_size }).init(w_ih_t, allocator, true);

            const w_hh_t = try TensorMod.Tensor(T, .{ gate_size, hidden_size }).init(allocator);
            for (w_hh_t.slice()) |*v| v.* = (rng.float(T) * 2.0 - 1.0) * k;
            const w_hh = try VariableMod.Variable(T, .{ gate_size, hidden_size }).init(w_hh_t, allocator, true);

            const b_ih_t = try TensorMod.Tensor(T, .{gate_size}).zeros(allocator);
            const b_ih = try VariableMod.Variable(T, .{gate_size}).init(b_ih_t, allocator, true);

            const b_hh_t = try TensorMod.Tensor(T, .{gate_size}).zeros(allocator);
            const b_hh = try VariableMod.Variable(T, .{gate_size}).init(b_hh_t, allocator, true);

            return .{ .w_ih = w_ih, .w_hh = w_hh, .b_ih = b_ih, .b_hh = b_hh };
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        /// forward: input (batch, seq_len, input_size) → output (batch, seq_len, hidden_size)
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime seq_len: usize,
            input: *VariableMod.Variable(T, .{ batch, seq_len, input_size }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, seq_len, hidden_size }) {
            const out_tensor = try TensorMod.Tensor(T, .{ batch, seq_len, hidden_size }).init(allocator);
            const out_data = out_tensor.slice();
            const in_data = input.constData();
            const w_ih_d = self.w_ih.constData();
            const w_hh_d = self.w_hh.constData();
            const b_ih_d = self.b_ih.constData();
            const b_hh_d = self.b_hh.constData();

            const total = batch * seq_len;
            const r_buf = try allocator.alloc(T, total * hidden_size);
            const z_buf = try allocator.alloc(T, total * hidden_size);
            const n_buf = try allocator.alloc(T, total * hidden_size);
            const n_h_buf = try allocator.alloc(T, total * hidden_size); // W_hn@h+b_hn before r
            // h states: h0(batch*H) + steps(total*H)
            const h_all = try allocator.alloc(T, (batch + total) * hidden_size);
            @memset(h_all[0 .. batch * hidden_size], 0);

            for (0..seq_len) |t| {
                for (0..batch) |b| {
                    const si = b * seq_len + t;
                    const x_off = si * input_size;
                    const h_prev_off = (if (t == 0) b else batch + b * seq_len + (t - 1)) * hidden_size;
                    const h_curr_off = (batch + si) * hidden_size;

                    for (0..hidden_size) |h| {
                        var r_pre: T = b_ih_d[h] + b_hh_d[h];
                        var z_pre: T = b_ih_d[hidden_size + h] + b_hh_d[hidden_size + h];
                        for (0..input_size) |kk| {
                            r_pre += w_ih_d[h * input_size + kk] * in_data[x_off + kk];
                            z_pre += w_ih_d[(hidden_size + h) * input_size + kk] * in_data[x_off + kk];
                        }
                        for (0..hidden_size) |kk| {
                            r_pre += w_hh_d[h * hidden_size + kk] * h_all[h_prev_off + kk];
                            z_pre += w_hh_d[(hidden_size + h) * hidden_size + kk] * h_all[h_prev_off + kk];
                        }
                        const r = sigm(r_pre);
                        const z = sigm(z_pre);

                        // n = tanh(W_in@x + b_in + r * (W_hn@h + b_hn))
                        var n_x: T = b_ih_d[2 * hidden_size + h];
                        for (0..input_size) |kk| {
                            n_x += w_ih_d[(2 * hidden_size + h) * input_size + kk] * in_data[x_off + kk];
                        }
                        var nh: T = b_hh_d[2 * hidden_size + h];
                        for (0..hidden_size) |kk| {
                            nh += w_hh_d[(2 * hidden_size + h) * hidden_size + kk] * h_all[h_prev_off + kk];
                        }
                        n_h_buf[si * hidden_size + h] = nh;
                        const n = tanhFn(n_x + r * nh);

                        r_buf[si * hidden_size + h] = r;
                        z_buf[si * hidden_size + h] = z;
                        n_buf[si * hidden_size + h] = n;

                        const hp = h_all[h_prev_off + h];
                        const hn = (1.0 - z) * n + z * hp;
                        h_all[h_curr_off + h] = hn;
                        out_data[si * hidden_size + h] = hn;
                    }
                }
            }

            const Ctx = struct {
                r: []const T, z: []const T, n: []const T, n_h: []const T,
                h_all: []const T, in_data: []const T,
                w_ih: []const T, w_hh: []const T,
                input_node: *Node,
                w_ih_node: *Node, w_hh_node: *Node,
                b_ih_node: *Node, b_hh_node: *Node,
            };
            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .r = r_buf, .z = z_buf, .n = n_buf, .n_h = n_h_buf,
                .h_all = h_all, .in_data = in_data,
                .w_ih = w_ih_d, .w_hh = w_hh_d,
                .input_node = input.node, .w_ih_node = self.w_ih.node,
                .w_hh_node = self.w_hh.node, .b_ih_node = self.b_ih.node,
                .b_hh_node = self.b_hh.node,
            };

            const OutVar = VariableMod.Variable(T, .{ batch, seq_len, hidden_size });
            var result = try OutVar.init(out_tensor, allocator, true);
            result.node.parents[0] = input.node;
            result.node.parents[1] = self.w_ih.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));

                    var dh_next: [batch * hidden_size]T = @splat(0);

                    var ti: usize = seq_len;
                    while (ti > 0) {
                        ti -= 1;
                        for (0..batch) |bb| {
                            const si = bb * seq_len + ti;
                            const x_off = si * input_size;
                            const h_off = bb * hidden_size;
                            const h_prev_off = (if (ti == 0) bb else batch + bb * seq_len + (ti - 1)) * hidden_size;

                            // Per-hidden derivatives
                            var dr_buf: [hidden_size]T = undefined;
                            var dz_buf: [hidden_size]T = undefined;
                            var dn_buf: [hidden_size]T = undefined;

                            for (0..hidden_size) |h| {
                                const dh = grad_out[si * hidden_size + h] + dh_next[h_off + h];
                                const r = c.r[si * hidden_size + h];
                                const z = c.z[si * hidden_size + h];
                                const n = c.n[si * hidden_size + h];
                                const nh = c.n_h[si * hidden_size + h];
                                const hp = c.h_all[h_prev_off + h];

                                // dz = dh * (h_prev - n) * σ'(z)
                                dz_buf[h] = dh * (hp - n) * z * (1.0 - z);
                                // dn = dh * (1-z) * tanh'(n)
                                dn_buf[h] = dh * (1.0 - z) * (1.0 - n * n);
                                // dr = dn * n_h * σ'(r)
                                dr_buf[h] = dn_buf[h] * nh * r * (1.0 - r);
                            }

                            // dW_ih, db_ih
                            if (c.w_ih_node.grad) |wg| {
                                for (0..hidden_size) |h| {
                                    for (0..input_size) |kk| {
                                        wg[h * input_size + kk] += dr_buf[h] * c.in_data[x_off + kk];
                                        wg[(hidden_size + h) * input_size + kk] += dz_buf[h] * c.in_data[x_off + kk];
                                        wg[(2 * hidden_size + h) * input_size + kk] += dn_buf[h] * c.in_data[x_off + kk];
                                    }
                                }
                            }
                            if (c.b_ih_node.grad) |bg| {
                                for (0..hidden_size) |h| {
                                    bg[h] += dr_buf[h];
                                    bg[hidden_size + h] += dz_buf[h];
                                    bg[2 * hidden_size + h] += dn_buf[h];
                                }
                            }

                            // dW_hh, db_hh
                            if (c.w_hh_node.grad) |wg| {
                                for (0..hidden_size) |h| {
                                    const dn_r = dn_buf[h] * c.r[si * hidden_size + h]; // dn * r
                                    for (0..hidden_size) |kk| {
                                        wg[h * hidden_size + kk] += dr_buf[h] * c.h_all[h_prev_off + kk];
                                        wg[(hidden_size + h) * hidden_size + kk] += dz_buf[h] * c.h_all[h_prev_off + kk];
                                        wg[(2 * hidden_size + h) * hidden_size + kk] += dn_r * c.h_all[h_prev_off + kk];
                                    }
                                }
                            }
                            if (c.b_hh_node.grad) |bg| {
                                for (0..hidden_size) |h| {
                                    bg[h] += dr_buf[h];
                                    bg[hidden_size + h] += dz_buf[h];
                                    bg[2 * hidden_size + h] += dn_buf[h] * c.r[si * hidden_size + h];
                                }
                            }

                            // dInput
                            if (c.input_node.grad) |ig| {
                                for (0..hidden_size) |h| {
                                    for (0..input_size) |kk| {
                                        ig[x_off + kk] += dr_buf[h] * c.w_ih[h * input_size + kk];
                                        ig[x_off + kk] += dz_buf[h] * c.w_ih[(hidden_size + h) * input_size + kk];
                                        ig[x_off + kk] += dn_buf[h] * c.w_ih[(2 * hidden_size + h) * input_size + kk];
                                    }
                                }
                            }

                            // dh_next = z * dh + W_hh^T @ [dr, dz, dn*r]
                            @memset(dh_next[h_off .. h_off + hidden_size], 0);
                            for (0..hidden_size) |h| {
                                const dh = grad_out[si * hidden_size + h] + dh_next[h_off + h];
                                _ = dh;
                                const z = c.z[si * hidden_size + h];
                                dh_next[h_off + h] = (grad_out[si * hidden_size + h]) * z;
                            }
                            // Add W_hh^T contributions
                            for (0..hidden_size) |h| {
                                const dn_r = dn_buf[h] * c.r[si * hidden_size + h];
                                for (0..hidden_size) |kk| {
                                    dh_next[h_off + kk] += dr_buf[h] * c.w_hh[h * hidden_size + kk];
                                    dh_next[h_off + kk] += dz_buf[h] * c.w_hh[(hidden_size + h) * hidden_size + kk];
                                    dh_next[h_off + kk] += dn_r * c.w_hh[(2 * hidden_size + h) * hidden_size + kk];
                                }
                            }
                        }
                    }
                }
            }.backward;

            return result;
        }
    };
}

// ============================================================
// ヘルパー
// ============================================================

fn sigm(x: anytype) @TypeOf(x) {
    return 1.0 / (1.0 + @exp(-x));
}

fn tanhFn(x: anytype) @TypeOf(x) {
    const ex = @exp(x);
    const enx = @exp(-x);
    return (ex - enx) / (ex + enx);
}

// ============================================================
// テスト
// ============================================================

test "LSTM forward shape" {
    const alloc = std.testing.allocator;
    var lstm = try LSTM(f64, 4, 3).init(alloc);
    defer lstm.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 2, 5, 4 }).fromSlice(temp, blk: {
        var data: [40]f64 = undefined;
        for (&data, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) * 0.01;
        break :blk &data;
    }, false);

    const output = try lstm.forward(2, 5, &input, temp);
    try std.testing.expectEqual(@as(usize, 30), output.constData().len);
    for (output.constData()) |v| {
        try std.testing.expect(@abs(v) < 1.0);
    }
}

test "LSTM forward zero input" {
    const alloc = std.testing.allocator;
    var lstm = try LSTM(f64, 2, 2).init(alloc);
    defer lstm.deinit();
    @memset(lstm.b_ih.data(), 0);
    @memset(lstm.b_hh.data(), 0);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // σ(0)=0.5, tanh(0)=0 → c=0.5*0+0.5*0=0, h=0.5*tanh(0)=0
    var input = try VariableMod.Variable(f64, .{ 1, 1, 2 }).fromSlice(temp, &[_]f64{ 0, 0 }, false);
    const output = try lstm.forward(1, 1, &input, temp);
    for (output.constData()) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 0), v, 1e-10);
    }
}

test "LSTM backward runs" {
    const alloc = std.testing.allocator;
    var lstm = try LSTM(f64, 3, 2).init(alloc);
    defer lstm.deinit();
    try lstm.allocGrad(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 2, 3 }).fromSlice(
        temp, &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, true,
    );
    input.node.grad = try temp.alloc(f64, 6);
    @memset(input.node.grad.?, 0);

    var output = try lstm.forward(1, 2, &input, temp);
    output.node.grad = try temp.alloc(f64, 4);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| bfn(output.node);

    var has_nonzero = false;
    for (lstm.w_ih.node.grad.?) |g| {
        if (@abs(g) > 1e-15) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "GRU forward shape" {
    const alloc = std.testing.allocator;
    var gru = try GRU(f64, 4, 3).init(alloc);
    defer gru.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 2, 5, 4 }).fromSlice(temp, blk: {
        var data: [40]f64 = undefined;
        for (&data, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) * 0.01;
        break :blk &data;
    }, false);

    const output = try gru.forward(2, 5, &input, temp);
    try std.testing.expectEqual(@as(usize, 30), output.constData().len);
}

test "GRU forward zero input" {
    const alloc = std.testing.allocator;
    var gru = try GRU(f64, 2, 2).init(alloc);
    defer gru.deinit();
    @memset(gru.b_ih.data(), 0);
    @memset(gru.b_hh.data(), 0);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 1, 2 }).fromSlice(temp, &[_]f64{ 0, 0 }, false);
    const output = try gru.forward(1, 1, &input, temp);
    for (output.constData()) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 0), v, 1e-10);
    }
}

test "GRU backward runs" {
    const alloc = std.testing.allocator;
    var gru = try GRU(f64, 3, 2).init(alloc);
    defer gru.deinit();
    try gru.allocGrad(alloc);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f64, .{ 1, 2, 3 }).fromSlice(
        temp, &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, true,
    );
    input.node.grad = try temp.alloc(f64, 6);
    @memset(input.node.grad.?, 0);

    var output = try gru.forward(1, 2, &input, temp);
    output.node.grad = try temp.alloc(f64, 4);
    @memset(output.node.grad.?, 1.0);

    if (output.node.backward_fn) |bfn| bfn(output.node);

    var has_nonzero = false;
    for (gru.w_ih.node.grad.?) |g| {
        if (@abs(g) > 1e-15) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}
