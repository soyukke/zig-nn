const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");
const ModuleMixin = @import("module.zig").Module;

/// Embedding: 整数インデックスからベクトルへの埋め込みテーブル。
/// weight: (vocab_size, embed_dim)
/// forward: indices → weight[indices] → output (batch, seq_len, embed_dim)
pub fn Embedding(comptime T: type, comptime vocab_size: usize, comptime embed_dim: usize) type {
    return struct {
        const Self = @This();
        const M = ModuleMixin(Self);

        weight: VariableMod.Variable(T, .{ vocab_size, embed_dim }),

        pub fn init(allocator: Allocator) !Self {
            const tensor = try TensorMod.Tensor(T, .{ vocab_size, embed_dim }).init(allocator);
            // Normal initialization: N(0, 1)
            var prng = std.Random.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                std.posix.getrandom(std.mem.asBytes(&seed)) catch {
                    seed = 42;
                };
                break :blk seed;
            });
            const rng = prng.random();
            for (tensor.slice()) |*v| {
                v.* = rng.float(T) * 2.0 - 1.0;
            }

            const weight = try VariableMod.Variable(T, .{ vocab_size, embed_dim }).init(
                tensor,
                allocator,
                true,
            );
            return .{ .weight = weight };
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }

        /// forward: indices (batch * seq_len) → output (batch, seq_len, embed_dim)
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime seq_len: usize,
            indices: *const [batch * seq_len]usize,
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, seq_len, embed_dim }) {
            const total = batch * seq_len;
            const Node = GraphNodeMod.GraphNode(T);

            const out_tensor = try TensorMod.Tensor(T, .{ batch, seq_len, embed_dim }).init(allocator);
            const out_data = out_tensor.slice();
            const w_data = self.weight.constData();

            for (0..total) |i| {
                const idx = indices[i];
                @memcpy(out_data[i * embed_dim ..][0..embed_dim], w_data[idx * embed_dim ..][0..embed_dim]);
            }

            // Context: save indices for backward
            const Ctx = struct {
                indices: []const usize,
                weight_parent: *Node,
            };

            const idx_copy = try allocator.alloc(usize, total);
            @memcpy(idx_copy, indices);

            const ctx = try allocator.create(Ctx);
            ctx.* = .{
                .indices = idx_copy,
                .weight_parent = self.weight.node,
            };

            const OutVar = VariableMod.Variable(T, .{ batch, seq_len, embed_dim });
            var result = try OutVar.init(out_tensor, allocator, true);
            result.node.parents[0] = self.weight.node;
            result.node.context = @ptrCast(ctx);

            result.node.backward_fn = struct {
                fn backward(node: *Node) void {
                    const grad_out = node.grad orelse return;
                    const c: *const Ctx = @ptrCast(@alignCast(node.context.?));

                    if (c.weight_parent.grad) |w_grad| {
                        for (0..total) |i| {
                            const idx = c.indices[i];
                            for (0..embed_dim) |j| {
                                w_grad[idx * embed_dim + j] += grad_out[i * embed_dim + j];
                            }
                        }
                    }
                }
            }.backward;

            return result;
        }

        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }
    };
}

// ============================================================
// テスト
// ============================================================

test "Embedding forward" {
    const alloc = std.testing.allocator;

    var emb = try Embedding(f64, 5, 3).init(alloc);
    defer emb.deinit();

    // 手動でweightを設定
    const w = emb.weight.data();
    // word 0: [1, 0, 0]
    // word 1: [0, 1, 0]
    // word 2: [0, 0, 1]
    // word 3: [1, 1, 0]
    // word 4: [1, 1, 1]
    @memset(w, 0);
    w[0 * 3 + 0] = 1; // word 0
    w[1 * 3 + 1] = 1; // word 1
    w[2 * 3 + 2] = 1; // word 2
    w[3 * 3 + 0] = 1;
    w[3 * 3 + 1] = 1; // word 3
    w[4 * 3 + 0] = 1;
    w[4 * 3 + 1] = 1;
    w[4 * 3 + 2] = 1; // word 4

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // batch=1, seq_len=3, indices=[2, 0, 4]
    const indices = [_]usize{ 2, 0, 4 };
    const output = try emb.forward(1, 3, &indices, temp);
    const out = output.constData();

    // word 2: [0, 0, 1]
    try std.testing.expectApproxEqAbs(@as(f64, 0), out[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), out[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1), out[2], 1e-10);

    // word 0: [1, 0, 0]
    try std.testing.expectApproxEqAbs(@as(f64, 1), out[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), out[4], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), out[5], 1e-10);

    // word 4: [1, 1, 1]
    try std.testing.expectApproxEqAbs(@as(f64, 1), out[6], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1), out[7], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1), out[8], 1e-10);
}

test "Embedding backward" {
    const alloc = std.testing.allocator;

    var emb = try Embedding(f64, 4, 2).init(alloc);
    defer emb.deinit();
    try emb.allocGrad(alloc);

    @memset(emb.weight.data(), 0); // weight values don't matter for backward

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // indices = [1, 3, 1] (word 1 appears twice)
    const indices = [_]usize{ 1, 3, 1 };
    var output = try emb.forward(1, 3, &indices, temp);

    output.node.grad = try temp.alloc(f64, 6);
    // grad_out for token 0: [1, 2], token 1: [3, 4], token 2: [5, 6]
    output.node.grad.?[0] = 1;
    output.node.grad.?[1] = 2;
    output.node.grad.?[2] = 3;
    output.node.grad.?[3] = 4;
    output.node.grad.?[4] = 5;
    output.node.grad.?[5] = 6;

    if (output.node.backward_fn) |bfn| bfn(output.node);

    const wg = emb.weight.node.grad.?;
    // word 0: no gradient (not in indices)
    try std.testing.expectApproxEqAbs(@as(f64, 0), wg[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), wg[1], 1e-10);

    // word 1: appears at position 0 and 2 → grad = [1+5, 2+6] = [6, 8]
    try std.testing.expectApproxEqAbs(@as(f64, 6), wg[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8), wg[3], 1e-10);

    // word 2: no gradient
    try std.testing.expectApproxEqAbs(@as(f64, 0), wg[4], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), wg[5], 1e-10);

    // word 3: appears at position 1 → grad = [3, 4]
    try std.testing.expectApproxEqAbs(@as(f64, 3), wg[6], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4), wg[7], 1e-10);
}
