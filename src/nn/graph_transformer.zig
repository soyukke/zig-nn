/// Unified Transformer layers (CPU/GPU backend-agnostic)
///
/// Pre-norm Transformer の EncoderLayer と DecoderLayer を提供。
/// anytype ctx で LayerNorm + Attention + FF (GELU) を実行。
const compute = @import("../compute.zig");
const Module = compute.Module;
const linear = @import("graph_linear.zig").linear;
const layer_norm = @import("graph_norm.zig").layer_norm;
const graph_attention = @import("graph_attention.zig");

/// Pre-norm Transformer Encoder Layer
/// LayerNorm → SelfAttention → Residual → LayerNorm → FF (GELU) → Residual
pub fn transformer_encoder_layer(comptime d_model: usize, comptime ff_dim: usize) type {
    return struct {
        ln1: layer_norm(d_model),
        self_attn: graph_attention.self_attention(d_model),
        ln2: layer_norm(d_model),
        ff1: linear(d_model, ff_dim),
        ff2: linear(ff_dim, d_model),

        pub fn init(module: anytype) @This() {
            return .{
                .ln1 = layer_norm(d_model).init(module),
                .self_attn = graph_attention.self_attention(d_model).init(module),
                .ln2 = layer_norm(d_model).init(module),
                .ff1 = linear(d_model, ff_dim).init(module),
                .ff2 = linear(ff_dim, d_model).init(module),
            };
        }

        pub fn forward(
            self: @This(),
            ctx: anytype,
            input: anytype,
            batch_size: usize,
            seq_len: usize,
        ) @TypeOf(input) {
            // Pre-norm self-attention + residual
            const ln1 = self.ln1.forward(ctx, input);
            const attn = self.self_attn.forward(ctx, ln1, batch_size, seq_len);
            const res1 = ctx.add(input, attn);

            // Pre-norm FF + residual
            const ln2 = self.ln2.forward(ctx, res1);
            const ff = ctx.gelu(self.ff1.forward(ctx, ln2));
            return ctx.add(res1, self.ff2.forward(ctx, ff));
        }
    };
}

/// Pre-norm Transformer Decoder Layer
/// LayerNorm → CausalSelfAttention → Residual → LayerNorm → CrossAttention → Residual
///   → LayerNorm → FF (GELU) → Residual
pub fn transformer_decoder_layer(
    comptime d_model: usize,
    comptime ff_dim: usize,
    comptime tgt_len: usize,
) type {
    return struct {
        ln1: layer_norm(d_model),
        self_attn: graph_attention.causal_self_attention(d_model, tgt_len),
        ln2: layer_norm(d_model),
        cross_attn: graph_attention.cross_attention(d_model),
        ln3: layer_norm(d_model),
        ff1: linear(d_model, ff_dim),
        ff2: linear(ff_dim, d_model),

        pub fn init(module: anytype) @This() {
            return .{
                .ln1 = layer_norm(d_model).init(module),
                .self_attn = graph_attention.causal_self_attention(d_model, tgt_len).init(module),
                .ln2 = layer_norm(d_model).init(module),
                .cross_attn = graph_attention.cross_attention(d_model).init(module),
                .ln3 = layer_norm(d_model).init(module),
                .ff1 = linear(d_model, ff_dim).init(module),
                .ff2 = linear(ff_dim, d_model).init(module),
            };
        }

        pub fn forward(
            self: @This(),
            ctx: anytype,
            input: anytype,
            enc_out: anytype,
            batch_size: usize,
            src_len: usize,
        ) @TypeOf(input) {
            // Pre-norm causal self-attention + residual
            const ln1 = self.ln1.forward(ctx, input);
            const sa = self.self_attn.forward(ctx, ln1, batch_size);
            const res1 = ctx.add(input, sa);

            // Pre-norm cross-attention + residual
            const ln2 = self.ln2.forward(ctx, res1);
            const ca = self.cross_attn.forward(ctx, ln2, enc_out, batch_size, tgt_len, src_len);
            const res2 = ctx.add(res1, ca);

            // Pre-norm FF + residual
            const ln3 = self.ln3.forward(ctx, res2);
            const ff = ctx.gelu(self.ff1.forward(ctx, ln3));
            return ctx.add(res2, self.ff2.forward(ctx, ff));
        }
    };
}

/// Backward-compatible aliases
pub const GraphTransformerEncoderLayer = transformer_encoder_layer;
pub const GraphTransformerDecoderLayer = transformer_decoder_layer;
