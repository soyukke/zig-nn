/// Unified Attention layers (CPU/GPU backend-agnostic)
///
/// SelfAttention, MultiHeadSelfAttention, CausalSelfAttention,
/// MultiHeadCausalSelfAttention, CrossAttention, MultiHeadCrossAttention を提供。
/// anytype ctx で Q/K/V/O 投影 + scaled dot-product attention を実行。
const std = @import("std");
const compute = @import("../compute.zig");
const Module = compute.Module;
const Linear = @import("graph_linear.zig").Linear;
const DTypeFloat32 = compute.MPSDataTypeFloat32;

/// Non-causal self-attention (single-head, backward compatible)
pub fn SelfAttention(comptime d_model: usize) type {
    return MultiHeadSelfAttention(d_model, 1);
}

/// Multi-head non-causal self-attention
pub fn MultiHeadSelfAttention(comptime d_model: usize, comptime n_heads: usize) type {
    const d_head = d_model / n_heads;
    return struct {
        q_proj: Linear(d_model, d_model),
        k_proj: Linear(d_model, d_model),
        v_proj: Linear(d_model, d_model),
        o_proj: Linear(d_model, d_model),

        pub fn init(module: anytype) @This() {
            return .{
                .q_proj = Linear(d_model, d_model).init(module),
                .k_proj = Linear(d_model, d_model).init(module),
                .v_proj = Linear(d_model, d_model).init(module),
                .o_proj = Linear(d_model, d_model).init(module),
            };
        }

        pub fn forward(
            self: @This(),
            ctx: anytype,
            input: anytype,
            batch_size: usize,
            seq_len: usize,
        ) @TypeOf(input) {
            const total = batch_size * seq_len;
            const bh = batch_size * n_heads;

            const q = ctx.reshape(self.q_proj.forward(ctx, input), &.{ bh, seq_len, d_head });
            const k = ctx.reshape(self.k_proj.forward(ctx, input), &.{ bh, seq_len, d_head });
            const v = ctx.reshape(self.v_proj.forward(ctx, input), &.{ bh, seq_len, d_head });

            const scale_val: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
            const scale_c = ctx.constantScalar(scale_val, DTypeFloat32);
            const scores = if (@typeInfo(@TypeOf(ctx)) == .pointer and
                @hasDecl(std.meta.Child(@TypeOf(ctx)), "matmulBatchedTransB"))
                ctx.mul(ctx.matmulBatchedTransB(q, k), scale_c)
            else blk: {
                const k_t = ctx.transpose(k, 1, 2);
                break :blk ctx.mul(ctx.matmul(q, k_t), scale_c);
            };
            const weights = ctx.softmax(scores, -1);
            const attn = ctx.matmul(weights, v);
            const flat = ctx.reshape(attn, &.{ total, d_model });
            return self.o_proj.forward(ctx, flat);
        }
    };
}

/// Causal self-attention (single-head, backward compatible)
pub fn CausalSelfAttention(comptime d_model: usize, comptime tgt_len: usize) type {
    return MultiHeadCausalSelfAttention(d_model, 1, tgt_len);
}

/// Multi-head causal self-attention (decoder用)
pub fn MultiHeadCausalSelfAttention(
    comptime d_model: usize,
    comptime n_heads: usize,
    comptime tgt_len: usize,
) type {
    const d_head = d_model / n_heads;
    return struct {
        q_proj: Linear(d_model, d_model),
        k_proj: Linear(d_model, d_model),
        v_proj: Linear(d_model, d_model),
        o_proj: Linear(d_model, d_model),

        pub fn init(module: anytype) @This() {
            return .{
                .q_proj = Linear(d_model, d_model).init(module),
                .k_proj = Linear(d_model, d_model).init(module),
                .v_proj = Linear(d_model, d_model).init(module),
                .o_proj = Linear(d_model, d_model).init(module),
            };
        }

        pub fn forward(
            self: @This(),
            ctx: anytype,
            input: anytype,
            batch_size: usize,
        ) @TypeOf(input) {
            const total = batch_size * tgt_len;
            const bh = batch_size * n_heads;

            const q = ctx.reshape(self.q_proj.forward(ctx, input), &.{ bh, tgt_len, d_head });
            const k = ctx.reshape(self.k_proj.forward(ctx, input), &.{ bh, tgt_len, d_head });
            const v = ctx.reshape(self.v_proj.forward(ctx, input), &.{ bh, tgt_len, d_head });

            const scale_val: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
            const scale_c = ctx.constantScalar(scale_val, DTypeFloat32);

            // Causal mask: (1, tgt_len, tgt_len), 0 = attend, -1e9 = mask
            const neg_inf: f32 = -1.0e9;
            var mask_data: [tgt_len * tgt_len]f32 = undefined;
            for (0..tgt_len) |i| {
                for (0..tgt_len) |j| {
                    mask_data[i * tgt_len + j] = if (j <= i) 0.0 else neg_inf;
                }
            }
            const causal_mask = ctx.constantData(
                @ptrCast(&mask_data),
                tgt_len * tgt_len * @sizeOf(f32),
                &.{ 1, tgt_len, tgt_len },
                DTypeFloat32,
            );

            const scores = if (@typeInfo(@TypeOf(ctx)) == .pointer and
                @hasDecl(std.meta.Child(@TypeOf(ctx)), "matmulBatchedTransB"))
                ctx.add(ctx.mul(ctx.matmulBatchedTransB(q, k), scale_c), causal_mask)
            else blk: {
                const k_t = ctx.transpose(k, 1, 2);
                break :blk ctx.add(ctx.mul(ctx.matmul(q, k_t), scale_c), causal_mask);
            };
            const weights = ctx.softmax(scores, -1);
            const attn = ctx.matmul(weights, v);
            const flat = ctx.reshape(attn, &.{ total, d_model });
            return self.o_proj.forward(ctx, flat);
        }
    };
}

/// Cross-attention (single-head, backward compatible)
pub fn CrossAttention(comptime d_model: usize) type {
    return MultiHeadCrossAttention(d_model, 1);
}

/// Multi-head cross-attention (encoder-decoder attention)
pub fn MultiHeadCrossAttention(comptime d_model: usize, comptime n_heads: usize) type {
    const d_head = d_model / n_heads;
    return struct {
        q_proj: Linear(d_model, d_model),
        k_proj: Linear(d_model, d_model),
        v_proj: Linear(d_model, d_model),
        o_proj: Linear(d_model, d_model),

        pub fn init(module: anytype) @This() {
            return .{
                .q_proj = Linear(d_model, d_model).init(module),
                .k_proj = Linear(d_model, d_model).init(module),
                .v_proj = Linear(d_model, d_model).init(module),
                .o_proj = Linear(d_model, d_model).init(module),
            };
        }

        /// query_input: (batch*tgt_len, d_model), kv_input: (batch*src_len, d_model)
        pub fn forward(
            self: @This(),
            ctx: anytype,
            query_input: anytype,
            kv_input: anytype,
            batch_size: usize,
            tgt_len: usize,
            src_len: usize,
        ) @TypeOf(query_input) {
            const total_tgt = batch_size * tgt_len;
            const bh = batch_size * n_heads;

            const q = ctx.reshape(self.q_proj.forward(ctx, query_input), &.{ bh, tgt_len, d_head });
            const k = ctx.reshape(self.k_proj.forward(ctx, kv_input), &.{ bh, src_len, d_head });
            const v = ctx.reshape(self.v_proj.forward(ctx, kv_input), &.{ bh, src_len, d_head });

            const scale_val: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
            const scale_c = ctx.constantScalar(scale_val, DTypeFloat32);

            const scores = if (@typeInfo(@TypeOf(ctx)) == .pointer and
                @hasDecl(std.meta.Child(@TypeOf(ctx)), "matmulBatchedTransB"))
                ctx.mul(ctx.matmulBatchedTransB(q, k), scale_c)
            else blk: {
                const k_t = ctx.transpose(k, 1, 2);
                break :blk ctx.mul(ctx.matmul(q, k_t), scale_c);
            };
            const weights = ctx.softmax(scores, -1);
            const attn = ctx.matmul(weights, v);
            const flat = ctx.reshape(attn, &.{ total_tgt, d_model });
            return self.o_proj.forward(ctx, flat);
        }
    };
}

/// Dynamic causal self-attention (runtime seq_len)
/// NOTE: DiffCpuRuntime 専用。allocData/makeNode を使うため GPU backend では使用不可。
/// GPU で runtime seq_len が必要な場合は MPSGraph の constantData で mask を生成すること。
pub fn DynamicCausalSelfAttention(comptime d_model: usize, comptime n_heads: usize) type {
    const d_head = d_model / n_heads;
    return struct {
        q_proj: Linear(d_model, d_model),
        k_proj: Linear(d_model, d_model),
        v_proj: Linear(d_model, d_model),
        o_proj: Linear(d_model, d_model),

        pub fn init(module: anytype) @This() {
            return .{
                .q_proj = Linear(d_model, d_model).init(module),
                .k_proj = Linear(d_model, d_model).init(module),
                .v_proj = Linear(d_model, d_model).init(module),
                .o_proj = Linear(d_model, d_model).init(module),
            };
        }

        pub fn forward(
            self: @This(),
            ctx: anytype,
            input: anytype,
            batch_size: usize,
            seq_len: usize,
        ) @TypeOf(input) {
            const total = batch_size * seq_len;
            const bh = batch_size * n_heads;

            const q = ctx.reshape(self.q_proj.forward(ctx, input), &.{ bh, seq_len, d_head });
            const k = ctx.reshape(self.k_proj.forward(ctx, input), &.{ bh, seq_len, d_head });
            const v = ctx.reshape(self.v_proj.forward(ctx, input), &.{ bh, seq_len, d_head });

            const scale_val: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
            const scale_c = ctx.constantScalar(scale_val, DTypeFloat32);

            // Runtime causal mask
            const mask_size = seq_len * seq_len;
            const neg_inf: f32 = -1.0e9;
            const mask_buf = ctx.allocData(mask_size);
            for (0..seq_len) |i| {
                for (0..seq_len) |j| {
                    mask_buf[i * seq_len + j] = if (j <= i) 0.0 else neg_inf;
                }
            }
            const causal_mask = ctx.makeNode(mask_buf, &.{ 1, seq_len, seq_len }, false);

            const scores = if (@typeInfo(@TypeOf(ctx)) == .pointer and
                @hasDecl(std.meta.Child(@TypeOf(ctx)), "matmulBatchedTransB"))
                ctx.add(ctx.mul(ctx.matmulBatchedTransB(q, k), scale_c), causal_mask)
            else blk: {
                const k_t = ctx.transpose(k, 1, 2);
                break :blk ctx.add(ctx.mul(ctx.matmul(q, k_t), scale_c), causal_mask);
            };
            const weights = ctx.softmax(scores, -1);
            const attn = ctx.matmul(weights, v);
            const flat = ctx.reshape(attn, &.{ total, d_model });
            return self.o_proj.forward(ctx, flat);
        }
    };
}

/// Backward-compatible aliases
pub const GraphSelfAttention = SelfAttention;
pub const GraphCausalSelfAttention = CausalSelfAttention;
pub const GraphCrossAttention = CrossAttention;
