const std = @import("std");
const Allocator = std.mem.Allocator;
const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");
const ops = @import("../autograd/ops.zig");
const LinearMod = @import("linear.zig");
const NormMod = @import("normalization.zig");
const AttentionMod = @import("attention.zig");
const ModuleMixin = @import("module.zig").Module;

/// Transformer Encoder Block: SelfAttn + FF + LayerNorm + Residual
///
/// Pre-norm 構造 (GPT-2 スタイル):
///   x = x + SelfAttn(LayerNorm(x))
///   x = x + FF(LayerNorm(x))
pub fn TransformerEncoderBlock(
    comptime T: type,
    comptime d_model: usize,
    comptime num_heads: usize,
    comptime ff_dim: usize,
) type {
    return struct {
        const Self = @This();
        const M = ModuleMixin(Self);

        self_attn: AttentionMod.MultiHeadAttention(T, d_model, num_heads),
        norm1: NormMod.LayerNorm(T, d_model),
        norm2: NormMod.LayerNorm(T, d_model),
        ff1: LinearMod.Linear(T, d_model, ff_dim),
        ff2: LinearMod.Linear(T, ff_dim, d_model),

        pub fn init(allocator: Allocator) !Self {
            return .{
                .self_attn = try AttentionMod.MultiHeadAttention(T, d_model, num_heads).init(allocator),
                .norm1 = try NormMod.LayerNorm(T, d_model).init(allocator),
                .norm2 = try NormMod.LayerNorm(T, d_model).init(allocator),
                .ff1 = try LinearMod.Linear(T, d_model, ff_dim).init(allocator),
                .ff2 = try LinearMod.Linear(T, ff_dim, d_model).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        /// forward: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime seq_len: usize,
            input: *VariableMod.Variable(T, .{ batch, seq_len, d_model }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, seq_len, d_model }) {
            const flat = batch * seq_len;

            // Pre-norm self-attention
            var flat_in = try ops.flatten3dTo2d(T, batch, seq_len, d_model, input, allocator);
            var normed1 = try self.norm1.forward(flat, &flat_in, allocator);
            var normed1_3d = try ops.unflatten2dTo3d(T, batch, seq_len, d_model, &normed1, allocator);
            var attn_out = try self.self_attn.forward(batch, seq_len, &normed1_3d, allocator);

            // Residual: x = input + attn_out
            var attn_flat = try ops.flatten3dTo2d(T, batch, seq_len, d_model, &attn_out, allocator);
            var residual1 = try ops.add(T, .{ flat, d_model }, &flat_in, &attn_flat, allocator);

            // Pre-norm FF
            var normed2 = try self.norm2.forward(flat, &residual1, allocator);
            var ff_h = try self.ff1.forward(flat, &normed2, allocator);
            var ff_act = try ops.relu(T, .{ flat, ff_dim }, &ff_h, allocator);
            var ff_out = try self.ff2.forward(flat, &ff_act, allocator);

            // Residual: x = residual1 + ff_out
            var residual2 = try ops.add(T, .{ flat, d_model }, &residual1, &ff_out, allocator);

            return ops.unflatten2dTo3d(T, batch, seq_len, d_model, &residual2, allocator);
        }
    };
}

/// Transformer Decoder Block: CausalSelfAttn + CrossAttn + FF + LayerNorm + Residual
///
/// Pre-norm 構造:
///   x = x + CausalSelfAttn(LayerNorm(x))
///   x = x + CrossAttn(LayerNorm(x), enc_out)
///   x = x + FF(LayerNorm(x))
pub fn TransformerDecoderBlock(
    comptime T: type,
    comptime d_model: usize,
    comptime num_heads: usize,
    comptime ff_dim: usize,
) type {
    return struct {
        const Self = @This();
        const M = ModuleMixin(Self);

        causal_attn: AttentionMod.CausalSelfAttention(T, d_model, num_heads),
        cross_attn: AttentionMod.CrossAttention(T, d_model, num_heads),
        norm1: NormMod.LayerNorm(T, d_model),
        norm2: NormMod.LayerNorm(T, d_model),
        norm3: NormMod.LayerNorm(T, d_model),
        ff1: LinearMod.Linear(T, d_model, ff_dim),
        ff2: LinearMod.Linear(T, ff_dim, d_model),

        pub fn init(allocator: Allocator) !Self {
            return .{
                .causal_attn = try AttentionMod.CausalSelfAttention(T, d_model, num_heads).init(allocator),
                .cross_attn = try AttentionMod.CrossAttention(T, d_model, num_heads).init(allocator),
                .norm1 = try NormMod.LayerNorm(T, d_model).init(allocator),
                .norm2 = try NormMod.LayerNorm(T, d_model).init(allocator),
                .norm3 = try NormMod.LayerNorm(T, d_model).init(allocator),
                .ff1 = try LinearMod.Linear(T, d_model, ff_dim).init(allocator),
                .ff2 = try LinearMod.Linear(T, ff_dim, d_model).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        /// forward: decoder_input (batch, tgt_len, d_model), enc_out (batch, src_len, d_model)
        /// → output (batch, tgt_len, d_model)
        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime tgt_len: usize,
            comptime src_len: usize,
            dec_input: *VariableMod.Variable(T, .{ batch, tgt_len, d_model }),
            enc_out: *VariableMod.Variable(T, .{ batch, src_len, d_model }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, tgt_len, d_model }) {
            const tgt_flat = batch * tgt_len;

            // 1. Pre-norm causal self-attention
            var flat_in = try ops.flatten3dTo2d(T, batch, tgt_len, d_model, dec_input, allocator);
            var normed1 = try self.norm1.forward(tgt_flat, &flat_in, allocator);
            var normed1_3d = try ops.unflatten2dTo3d(T, batch, tgt_len, d_model, &normed1, allocator);
            var sa_out = try self.causal_attn.forward(batch, tgt_len, &normed1_3d, allocator);

            // Residual
            var sa_flat = try ops.flatten3dTo2d(T, batch, tgt_len, d_model, &sa_out, allocator);
            var residual1 = try ops.add(T, .{ tgt_flat, d_model }, &flat_in, &sa_flat, allocator);

            // 2. Pre-norm cross-attention
            var normed2 = try self.norm2.forward(tgt_flat, &residual1, allocator);
            var normed2_3d = try ops.unflatten2dTo3d(T, batch, tgt_len, d_model, &normed2, allocator);
            var ca_out = try self.cross_attn.forward(batch, tgt_len, src_len, &normed2_3d, enc_out, allocator);

            // Residual
            var ca_flat = try ops.flatten3dTo2d(T, batch, tgt_len, d_model, &ca_out, allocator);
            var residual2 = try ops.add(T, .{ tgt_flat, d_model }, &residual1, &ca_flat, allocator);

            // 3. Pre-norm FF
            var normed3 = try self.norm3.forward(tgt_flat, &residual2, allocator);
            var ff_h = try self.ff1.forward(tgt_flat, &normed3, allocator);
            var ff_act = try ops.relu(T, .{ tgt_flat, ff_dim }, &ff_h, allocator);
            var ff_out = try self.ff2.forward(tgt_flat, &ff_act, allocator);

            // Residual
            var residual3 = try ops.add(T, .{ tgt_flat, d_model }, &residual2, &ff_out, allocator);

            return ops.unflatten2dTo3d(T, batch, tgt_len, d_model, &residual3, allocator);
        }
    };
}

/// N-layer Transformer Encoder
pub fn TransformerEncoder(
    comptime T: type,
    comptime d_model: usize,
    comptime num_heads: usize,
    comptime ff_dim: usize,
    comptime num_layers: usize,
) type {
    return struct {
        const Self = @This();
        const Block = TransformerEncoderBlock(T, d_model, num_heads, ff_dim);
        const M = ModuleMixin(Self);

        layers: [num_layers]Block,

        pub fn init(allocator: Allocator) !Self {
            var self: Self = undefined;
            for (0..num_layers) |i| {
                self.layers[i] = try Block.init(allocator);
            }
            return self;
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime seq_len: usize,
            input: *VariableMod.Variable(T, .{ batch, seq_len, d_model }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, seq_len, d_model }) {
            var h = try self.layers[0].forward(batch, seq_len, input, allocator);
            for (1..num_layers) |i| {
                h = try self.layers[i].forward(batch, seq_len, &h, allocator);
            }
            return h;
        }
    };
}

/// N-layer Transformer Decoder
pub fn TransformerDecoder(
    comptime T: type,
    comptime d_model: usize,
    comptime num_heads: usize,
    comptime ff_dim: usize,
    comptime num_layers: usize,
) type {
    return struct {
        const Self = @This();
        const Block = TransformerDecoderBlock(T, d_model, num_heads, ff_dim);
        const M = ModuleMixin(Self);

        layers: [num_layers]Block,

        pub fn init(allocator: Allocator) !Self {
            var self: Self = undefined;
            for (0..num_layers) |i| {
                self.layers[i] = try Block.init(allocator);
            }
            return self;
        }

        pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
        pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
        pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }

        pub fn forward(
            self: *Self,
            comptime batch: usize,
            comptime tgt_len: usize,
            comptime src_len: usize,
            dec_input: *VariableMod.Variable(T, .{ batch, tgt_len, d_model }),
            enc_out: *VariableMod.Variable(T, .{ batch, src_len, d_model }),
            allocator: Allocator,
        ) !VariableMod.Variable(T, .{ batch, tgt_len, d_model }) {
            var h = try self.layers[0].forward(batch, tgt_len, src_len, dec_input, enc_out, allocator);
            for (1..num_layers) |i| {
                h = try self.layers[i].forward(batch, tgt_len, src_len, &h, enc_out, allocator);
            }
            return h;
        }
    };
}

// ============================================================
// テスト
// ============================================================

test "TransformerEncoderBlock forward" {
    const alloc = std.testing.allocator;
    var block = try TransformerEncoderBlock(f32, 8, 2, 16).init(alloc);
    defer block.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // batch=1, seq_len=3, d_model=8
    var input = try VariableMod.Variable(f32, .{ 1, 3, 8 }).zeros(temp, false);
    // Fill with some values
    for (input.data(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;

    const output = try block.forward(1, 3, &input, temp);
    try std.testing.expectEqual(@as(usize, 24), output.constData().len);

    for (output.constData()) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "TransformerDecoderBlock forward" {
    const alloc = std.testing.allocator;
    var block = try TransformerDecoderBlock(f32, 8, 2, 16).init(alloc);
    defer block.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    // dec: (1, 2, 8), enc: (1, 3, 8)
    var dec_input = try VariableMod.Variable(f32, .{ 1, 2, 8 }).zeros(temp, false);
    var enc_out = try VariableMod.Variable(f32, .{ 1, 3, 8 }).zeros(temp, false);
    for (dec_input.data(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;
    for (enc_out.data(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.02;

    const output = try block.forward(1, 2, 3, &dec_input, &enc_out, temp);
    try std.testing.expectEqual(@as(usize, 16), output.constData().len);

    for (output.constData()) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "TransformerEncoder 2-layer forward" {
    const alloc = std.testing.allocator;
    var encoder = try TransformerEncoder(f32, 8, 2, 16, 2).init(alloc);
    defer encoder.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var input = try VariableMod.Variable(f32, .{ 1, 3, 8 }).zeros(temp, false);
    for (input.data(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;

    const output = try encoder.forward(1, 3, &input, temp);
    try std.testing.expectEqual(@as(usize, 24), output.constData().len);
}

test "TransformerDecoder 2-layer forward" {
    const alloc = std.testing.allocator;
    var decoder = try TransformerDecoder(f32, 8, 2, 16, 2).init(alloc);
    defer decoder.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp = arena.allocator();

    var dec_input = try VariableMod.Variable(f32, .{ 1, 2, 8 }).zeros(temp, false);
    var enc_out = try VariableMod.Variable(f32, .{ 1, 3, 8 }).zeros(temp, false);
    for (dec_input.data(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;
    for (enc_out.data(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.02;

    const output = try decoder.forward(1, 2, 3, &dec_input, &enc_out, temp);
    try std.testing.expectEqual(@as(usize, 16), output.constData().len);
}
