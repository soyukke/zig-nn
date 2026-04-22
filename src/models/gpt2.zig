const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf_mod = @import("../gguf/gguf.zig");
const dequant_mod = @import("../gguf/dequant.zig");

// ============================================================
// GPT-2 Config
// ============================================================

pub const GPT2Small = struct {
    pub const VOCAB: usize = 50257;
    pub const EMBED: usize = 768;
    pub const HEAD: usize = 12;
    pub const LAYER: usize = 12;
    pub const CTX: usize = 1024;
    pub const HEAD_DIM: usize = EMBED / HEAD; // 64
    pub const FFN_DIM: usize = EMBED * 4; // 3072
};

pub const GPT2Medium = struct {
    pub const VOCAB: usize = 50257;
    pub const EMBED: usize = 1024;
    pub const HEAD: usize = 16;
    pub const LAYER: usize = 24;
    pub const CTX: usize = 1024;
    pub const HEAD_DIM: usize = EMBED / HEAD;
    pub const FFN_DIM: usize = EMBED * 4;
};

// ============================================================
// Weight 構造体
// ============================================================

/// 量子化重み: GGUF file_buf への直接参照（コピーなし）
pub const QuantizedWeight = struct {
    data: []const u8,
    type_: gguf_mod.GGMLType,
    out_dim: usize,
    in_dim: usize,
};

pub fn TransformerBlockWeights(comptime _: type) type {
    return struct {
        // Attention
        ln1_weight: []f32, // (EMBED,)
        ln1_bias: []f32, // (EMBED,)
        attn_qkv_weight: QuantizedWeight, // (3*EMBED, EMBED) Q4_0
        attn_qkv_bias: []f32, // (3*EMBED,)
        attn_proj_weight: QuantizedWeight, // (EMBED, EMBED) Q4_0
        attn_proj_bias: []f32, // (EMBED,)
        // MLP
        ln2_weight: []f32, // (EMBED,)
        ln2_bias: []f32, // (EMBED,)
        mlp_fc_weight: QuantizedWeight, // (FFN_DIM, EMBED) Q4_0
        mlp_fc_bias: []f32, // (FFN_DIM,)
        mlp_proj_weight: QuantizedWeight, // (EMBED, FFN_DIM) Q4_0
        mlp_proj_bias: []f32, // (EMBED,)
    };
}

pub fn GPT2Weights(comptime C: type) type {
    return struct {
        const Self = @This();

        wte: []f32, // (VOCAB, EMBED)
        wpe: []f32, // (CTX, EMBED)
        blocks: [C.LAYER]TransformerBlockWeights(C),
        ln_f_weight: []f32, // (EMBED,)
        ln_f_bias: []f32, // (EMBED,)
        allocator: Allocator,

        pub fn loadFromGGUF(gguf_file: *const gguf_mod.GGUFFile, allocator: Allocator) !Self {
            var self: Self = undefined;
            self.allocator = allocator;

            // Embeddings (1D配列はそのまま、2D配列は転置)
            self.wte = try gguf_file.loadTensorF32("token_embd.weight", allocator);
            self.wpe = try gguf_file.loadTensorF32("position_embd.weight", allocator);

            // Final LayerNorm
            self.ln_f_weight = try gguf_file.loadTensorF32("output_norm.weight", allocator);
            self.ln_f_bias = try gguf_file.loadTensorF32("output_norm.bias", allocator);

            // Transformer blocks
            for (0..C.LAYER) |i| {
                self.blocks[i] = try loadBlockWeights(C, gguf_file, allocator, i);
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.wte);
            self.allocator.free(self.wpe);
            self.allocator.free(self.ln_f_weight);
            self.allocator.free(self.ln_f_bias);
            for (&self.blocks) |*blk| {
                self.allocator.free(blk.ln1_weight);
                self.allocator.free(blk.ln1_bias);
                // attn_qkv_weight は QuantizedWeight (file_buf 参照) → free 不要
                self.allocator.free(blk.attn_qkv_bias);
                // attn_proj_weight は QuantizedWeight → free 不要
                self.allocator.free(blk.attn_proj_bias);
                self.allocator.free(blk.ln2_weight);
                self.allocator.free(blk.ln2_bias);
                // mlp_fc_weight は QuantizedWeight → free 不要
                self.allocator.free(blk.mlp_fc_bias);
                // mlp_proj_weight は QuantizedWeight → free 不要
                self.allocator.free(blk.mlp_proj_bias);
            }
        }
    };
}

fn loadQuantizedWeight(gguf_file: *const gguf_mod.GGUFFile, name: []const u8) !QuantizedWeight {
    const ref = try gguf_file.getTensorRawBytes(name);
    return .{
        .data = ref.data,
        .type_ = ref.type_,
        .out_dim = ref.out_dim,
        .in_dim = ref.in_dim,
    };
}

fn loadBlockWeights(
    comptime C: type,
    gguf_file: *const gguf_mod.GGUFFile,
    allocator: Allocator,
    layer_idx: usize,
) !TransformerBlockWeights(C) {
    var name_buf: [64]u8 = undefined;

    return .{
        .ln1_weight = try gguf_file.loadTensorF32(
            try fmtBlockName(&name_buf, layer_idx, "attn_norm.weight"),
            allocator,
        ),
        .ln1_bias = try gguf_file.loadTensorF32(
            try fmtBlockName(&name_buf, layer_idx, "attn_norm.bias"),
            allocator,
        ),
        .attn_qkv_weight = try loadQuantizedWeight(
            gguf_file,
            try fmtBlockName(&name_buf, layer_idx, "attn_qkv.weight"),
        ),
        .attn_qkv_bias = try gguf_file.loadTensorF32(
            try fmtBlockName(&name_buf, layer_idx, "attn_qkv.bias"),
            allocator,
        ),
        .attn_proj_weight = try loadQuantizedWeight(
            gguf_file,
            try fmtBlockName(&name_buf, layer_idx, "attn_output.weight"),
        ),
        .attn_proj_bias = try gguf_file.loadTensorF32(
            try fmtBlockName(&name_buf, layer_idx, "attn_output.bias"),
            allocator,
        ),
        .ln2_weight = try gguf_file.loadTensorF32(
            try fmtBlockName(&name_buf, layer_idx, "ffn_norm.weight"),
            allocator,
        ),
        .ln2_bias = try gguf_file.loadTensorF32(
            try fmtBlockName(&name_buf, layer_idx, "ffn_norm.bias"),
            allocator,
        ),
        .mlp_fc_weight = try loadQuantizedWeight(
            gguf_file,
            try fmtBlockName(&name_buf, layer_idx, "ffn_up.weight"),
        ),
        .mlp_fc_bias = try gguf_file.loadTensorF32(
            try fmtBlockName(&name_buf, layer_idx, "ffn_up.bias"),
            allocator,
        ),
        .mlp_proj_weight = try loadQuantizedWeight(
            gguf_file,
            try fmtBlockName(&name_buf, layer_idx, "ffn_down.weight"),
        ),
        .mlp_proj_bias = try gguf_file.loadTensorF32(
            try fmtBlockName(&name_buf, layer_idx, "ffn_down.bias"),
            allocator,
        ),
    };
}

fn fmtBlockName(buf: []u8, layer: usize, suffix: []const u8) ![]const u8 {
    const result = std.fmt.bufPrint(
        buf,
        "blk.{d}.{s}",
        .{ layer, suffix },
    ) catch return error.NameTooLong;
    return result;
}

// ============================================================
// GPT-2 推論エンジン
// ============================================================

pub fn GPT2(comptime C: type) type {
    return struct {
        const Self = @This();

        weights: GPT2Weights(C),
        kv_cache: KVCache,
        allocator: Allocator,

        /// KV キャッシュ: 各レイヤーの K, V を CTX 分プリアロケート
        const KVCache = struct {
            k: [C.LAYER][]f32, // 各 (CTX, EMBED)
            v: [C.LAYER][]f32, // 各 (CTX, EMBED)
            seq_len: usize, // キャッシュ済みトークン数

            fn init(allocator: Allocator) !KVCache {
                var cache: KVCache = undefined;
                cache.seq_len = 0;
                for (0..C.LAYER) |i| {
                    cache.k[i] = try allocator.alloc(f32, C.CTX * C.EMBED);
                    cache.v[i] = try allocator.alloc(f32, C.CTX * C.EMBED);
                    @memset(cache.k[i], 0);
                    @memset(cache.v[i], 0);
                }
                return cache;
            }

            fn deinit(self: *KVCache, allocator: Allocator) void {
                for (0..C.LAYER) |i| {
                    allocator.free(self.k[i]);
                    allocator.free(self.v[i]);
                }
            }

            fn reset(self: *KVCache) void {
                self.seq_len = 0;
            }
        };

        pub fn init(gguf_file: *const gguf_mod.GGUFFile, allocator: Allocator) !Self {
            return .{
                .weights = try GPT2Weights(C).loadFromGGUF(gguf_file, allocator),
                .kv_cache = try KVCache.init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.kv_cache.deinit(self.allocator);
            self.weights.deinit();
        }

        pub fn resetCache(self: *Self) void {
            self.kv_cache.reset();
        }

        /// Forward pass (キャッシュ不使用版、互換性のため残す)
        pub fn forward(self: *const Self, tokens: []const u32, arena: Allocator) ![]f32 {
            const seq_len = tokens.len;
            if (seq_len == 0) return error.EmptySequence;
            if (seq_len > C.CTX) return error.SequenceTooLong;

            const w = &self.weights;

            // 1. Token embedding + Position embedding
            const x = try arena.alloc(f32, seq_len * C.EMBED);
            for (0..seq_len) |t| {
                const tok: usize = tokens[t];
                for (0..C.EMBED) |i| {
                    x[t * C.EMBED + i] = w.wte[tok * C.EMBED + i] + w.wpe[t * C.EMBED + i];
                }
            }

            // 中間バッファ
            const ln_out = try arena.alloc(f32, seq_len * C.EMBED);
            const qkv = try arena.alloc(f32, seq_len * 3 * C.EMBED);
            const attn_out = try arena.alloc(f32, seq_len * C.EMBED);
            const proj_out = try arena.alloc(f32, seq_len * C.EMBED);
            const mlp_hidden = try arena.alloc(f32, seq_len * C.FFN_DIM);
            const mlp_out = try arena.alloc(f32, seq_len * C.EMBED);
            const attn_scores = try arena.alloc(f32, seq_len * seq_len);

            for (0..C.LAYER) |layer| {
                const blk = &w.blocks[layer];
                layerNormRows(x, blk.ln1_weight, blk.ln1_bias, ln_out, seq_len, C.EMBED);
                linearForwardQ(ln_out, blk.attn_qkv_weight, blk.attn_qkv_bias, qkv, seq_len);
                causalSelfAttention(
                    qkv,
                    attn_out,
                    attn_scores,
                    seq_len,
                    C.HEAD,
                    C.HEAD_DIM,
                    C.EMBED,
                );
                linearForwardQ(
                    attn_out,
                    blk.attn_proj_weight,
                    blk.attn_proj_bias,
                    proj_out,
                    seq_len,
                );
                addInPlace(x, proj_out, seq_len * C.EMBED);
                layerNormRows(x, blk.ln2_weight, blk.ln2_bias, ln_out, seq_len, C.EMBED);
                linearForwardQ(ln_out, blk.mlp_fc_weight, blk.mlp_fc_bias, mlp_hidden, seq_len);
                geluInPlace(mlp_hidden, seq_len * C.FFN_DIM);
                linearForwardQ(
                    mlp_hidden,
                    blk.mlp_proj_weight,
                    blk.mlp_proj_bias,
                    mlp_out,
                    seq_len,
                );
                addInPlace(x, mlp_out, seq_len * C.EMBED);
            }

            layerNormRows(x, w.ln_f_weight, w.ln_f_bias, ln_out, seq_len, C.EMBED);
            return computeLogits(w, ln_out[(seq_len - 1) * C.EMBED ..][0..C.EMBED], arena);
        }

        const PrefillBufs = struct {
            x: []f32,
            ln_out: []f32,
            qkv: []f32,
            attn_out: []f32,
            proj_out: []f32,
            mlp_hidden: []f32,
            mlp_out: []f32,
            attn_scores: []f32,
        };

        fn prefillAllocBufs(arena: Allocator, seq_len: usize) !PrefillBufs {
            return .{
                .x = try arena.alloc(f32, seq_len * C.EMBED),
                .ln_out = try arena.alloc(f32, seq_len * C.EMBED),
                .qkv = try arena.alloc(f32, seq_len * 3 * C.EMBED),
                .attn_out = try arena.alloc(f32, seq_len * C.EMBED),
                .proj_out = try arena.alloc(f32, seq_len * C.EMBED),
                .mlp_hidden = try arena.alloc(f32, seq_len * C.FFN_DIM),
                .mlp_out = try arena.alloc(f32, seq_len * C.EMBED),
                .attn_scores = try arena.alloc(f32, seq_len * seq_len),
            };
        }

        /// Token + position embedding を合成して x に書き込む
        fn prefillEmbed(
            w: *const GPT2Weights(C),
            tokens: []const u32,
            x: []f32,
            seq_len: usize,
        ) void {
            for (0..seq_len) |t| {
                const tok: usize = tokens[t];
                for (0..C.EMBED) |i| {
                    x[t * C.EMBED + i] = w.wte[tok * C.EMBED + i] + w.wpe[t * C.EMBED + i];
                }
            }
        }

        /// prefill 1 レイヤ分 (LN+QKV+attention / LN+MLP + KV キャッシュ書き込み)
        fn prefillLayer(
            self: *Self,
            bufs: PrefillBufs,
            layer: usize,
            seq_len: usize,
        ) void {
            const w = &self.weights;
            const blk = &w.blocks[layer];
            layerNormRows(bufs.x, blk.ln1_weight, blk.ln1_bias, bufs.ln_out, seq_len, C.EMBED);
            linearForwardQ(bufs.ln_out, blk.attn_qkv_weight, blk.attn_qkv_bias, bufs.qkv, seq_len);

            // K, V をキャッシュに格納
            for (0..seq_len) |t| {
                @memcpy(
                    self.kv_cache.k[layer][t * C.EMBED ..][0..C.EMBED],
                    bufs.qkv[t * 3 * C.EMBED + C.EMBED ..][0..C.EMBED],
                );
                @memcpy(
                    self.kv_cache.v[layer][t * C.EMBED ..][0..C.EMBED],
                    bufs.qkv[t * 3 * C.EMBED + 2 * C.EMBED ..][0..C.EMBED],
                );
            }

            causalSelfAttention(
                bufs.qkv,
                bufs.attn_out,
                bufs.attn_scores,
                seq_len,
                C.HEAD,
                C.HEAD_DIM,
                C.EMBED,
            );
            linearForwardQ(
                bufs.attn_out,
                blk.attn_proj_weight,
                blk.attn_proj_bias,
                bufs.proj_out,
                seq_len,
            );
            addInPlace(bufs.x, bufs.proj_out, seq_len * C.EMBED);
            layerNormRows(bufs.x, blk.ln2_weight, blk.ln2_bias, bufs.ln_out, seq_len, C.EMBED);
            linearForwardQ(
                bufs.ln_out,
                blk.mlp_fc_weight,
                blk.mlp_fc_bias,
                bufs.mlp_hidden,
                seq_len,
            );
            geluInPlace(bufs.mlp_hidden, seq_len * C.FFN_DIM);
            linearForwardQ(
                bufs.mlp_hidden,
                blk.mlp_proj_weight,
                blk.mlp_proj_bias,
                bufs.mlp_out,
                seq_len,
            );
            addInPlace(bufs.x, bufs.mlp_out, seq_len * C.EMBED);
        }

        /// Prefill: プロンプト全体を処理して KV キャッシュを埋める
        pub fn prefill(self: *Self, tokens: []const u32, arena: Allocator) ![]f32 {
            const seq_len = tokens.len;
            if (seq_len == 0) return error.EmptySequence;
            if (seq_len > C.CTX) return error.SequenceTooLong;

            self.kv_cache.reset();
            const w = &self.weights;
            const bufs = try prefillAllocBufs(arena, seq_len);
            prefillEmbed(w, tokens, bufs.x, seq_len);

            for (0..C.LAYER) |layer| {
                self.prefillLayer(bufs, layer, seq_len);
            }

            self.kv_cache.seq_len = seq_len;
            layerNormRows(bufs.x, w.ln_f_weight, w.ln_f_bias, bufs.ln_out, seq_len, C.EMBED);
            return computeLogits(w, bufs.ln_out[(seq_len - 1) * C.EMBED ..][0..C.EMBED], arena);
        }

        /// DecodeNext: 1トークンのみ処理、KV キャッシュに追記
        pub fn decodeNext(self: *Self, token: u32, arena: Allocator) ![]f32 {
            const pos = self.kv_cache.seq_len;
            if (pos >= C.CTX) return error.ContextFull;
            const w = &self.weights;

            // 1行分の embedding
            const x = try arena.alloc(f32, C.EMBED);
            const tok: usize = token;
            for (0..C.EMBED) |i| {
                x[i] = w.wte[tok * C.EMBED + i] + w.wpe[pos * C.EMBED + i];
            }

            // 1行分のバッファ
            const ln_out = try arena.alloc(f32, C.EMBED);
            const qkv_buf = try arena.alloc(f32, 3 * C.EMBED);
            const attn_out = try arena.alloc(f32, C.EMBED);
            const proj_out = try arena.alloc(f32, C.EMBED);
            const mlp_hidden = try arena.alloc(f32, C.FFN_DIM);
            const mlp_out = try arena.alloc(f32, C.EMBED);
            const attn_scores = try arena.alloc(f32, pos + 1);

            for (0..C.LAYER) |layer| {
                const blk = &w.blocks[layer];

                // LayerNorm + QKV (1行)
                layerNormRows(x, blk.ln1_weight, blk.ln1_bias, ln_out, 1, C.EMBED);
                linearForwardQ(ln_out, blk.attn_qkv_weight, blk.attn_qkv_bias, qkv_buf, 1);

                // 新しい K, V をキャッシュに格納
                @memcpy(
                    self.kv_cache.k[layer][pos * C.EMBED ..][0..C.EMBED],
                    qkv_buf[C.EMBED..][0..C.EMBED],
                );
                @memcpy(
                    self.kv_cache.v[layer][pos * C.EMBED ..][0..C.EMBED],
                    qkv_buf[2 * C.EMBED ..][0..C.EMBED],
                );

                // Cached attention: 新 Q が全キャッシュに attend
                cachedAttention(
                    qkv_buf[0..C.EMBED],
                    self.kv_cache.k[layer],
                    self.kv_cache.v[layer],
                    attn_out,
                    attn_scores,
                    pos + 1,
                    C.HEAD,
                    C.HEAD_DIM,
                    C.EMBED,
                );

                // Output projection + Residual
                linearForwardQ(attn_out, blk.attn_proj_weight, blk.attn_proj_bias, proj_out, 1);
                addInPlace(x, proj_out, C.EMBED);

                // MLP
                layerNormRows(x, blk.ln2_weight, blk.ln2_bias, ln_out, 1, C.EMBED);
                linearForwardQ(ln_out, blk.mlp_fc_weight, blk.mlp_fc_bias, mlp_hidden, 1);
                geluInPlace(mlp_hidden, C.FFN_DIM);
                linearForwardQ(mlp_hidden, blk.mlp_proj_weight, blk.mlp_proj_bias, mlp_out, 1);
                addInPlace(x, mlp_out, C.EMBED);
            }

            self.kv_cache.seq_len = pos + 1;
            layerNormRows(x, w.ln_f_weight, w.ln_f_bias, ln_out, 1, C.EMBED);
            return computeLogits(w, ln_out, arena);
        }

        /// hidden → logits (共通ヘルパー, SIMD)
        fn computeLogits(w: *const GPT2Weights(C), hidden: []const f32, arena: Allocator) ![]f32 {
            const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
            const logits = try arena.alloc(f32, C.VOCAB);
            for (0..C.VOCAB) |v| {
                const wte_row = w.wte[v * C.EMBED ..][0..C.EMBED];
                var acc: @Vector(vl, f32) = @splat(0);
                var i: usize = 0;
                while (i + vl <= C.EMBED) : (i += vl) {
                    const h: @Vector(vl, f32) = hidden[i..][0..vl].*;
                    const wv: @Vector(vl, f32) = wte_row[i..][0..vl].*;
                    acc += h * wv;
                }
                var sum: f32 = @reduce(.Add, acc);
                while (i < C.EMBED) : (i += 1) {
                    sum += hidden[i] * wte_row[i];
                }
                logits[v] = sum;
            }
            return logits;
        }
    };
}

// ============================================================
// ヘルパー関数
// ============================================================

/// LayerNorm: 各行を独立に正規化 (SIMD)
/// input/output: (rows, dim), weight/bias: (dim,)
fn layerNormRows(
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    output: []f32,
    rows: usize,
    dim: usize,
) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    const eps: f32 = 1e-5;

    for (0..rows) |r| {
        const row = input[r * dim ..][0..dim];
        const out = output[r * dim ..][0..dim];

        // Mean (SIMD)
        var mean_acc: @Vector(vl, f32) = @splat(0);
        var i: usize = 0;
        while (i + vl <= dim) : (i += vl) {
            mean_acc += @as(@Vector(vl, f32), row[i..][0..vl].*);
        }
        var mean: f32 = @reduce(.Add, mean_acc);
        while (i < dim) : (i += 1) mean += row[i];
        mean /= @floatFromInt(dim);

        // Variance (SIMD)
        const mean_v: @Vector(vl, f32) = @splat(mean);
        var var_acc: @Vector(vl, f32) = @splat(0);
        i = 0;
        while (i + vl <= dim) : (i += vl) {
            const d = @as(@Vector(vl, f32), row[i..][0..vl].*) - mean_v;
            var_acc += d * d;
        }
        var variance: f32 = @reduce(.Add, var_acc);
        while (i < dim) : (i += 1) {
            const d = row[i] - mean;
            variance += d * d;
        }
        variance /= @floatFromInt(dim);

        const inv_std = 1.0 / @sqrt(variance + eps);
        const inv_std_v: @Vector(vl, f32) = @splat(inv_std);

        // Normalize (SIMD)
        i = 0;
        while (i + vl <= dim) : (i += vl) {
            const v = (@as(@Vector(vl, f32), row[i..][0..vl].*) - mean_v) * inv_std_v;
            const w: @Vector(vl, f32) = weight[i..][0..vl].*;
            const b: @Vector(vl, f32) = bias[i..][0..vl].*;
            out[i..][0..vl].* = v * w + b;
        }
        while (i < dim) : (i += 1) {
            out[i] = (row[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

/// Linear forward: output = input @ weight^T + bias (SIMD)
/// input: (rows, in_dim), weight: (out_dim, in_dim) row-major (GGUF layout),
/// output: (rows, out_dim)
fn linearForward(
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    output: []f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    for (0..rows) |r| {
        const in_row = input[r * in_dim ..][0..in_dim];
        for (0..out_dim) |j| {
            const w_row = weight[j * in_dim ..][0..in_dim];
            var acc: @Vector(vl, f32) = @splat(0);
            var k: usize = 0;
            while (k + vl <= in_dim) : (k += vl) {
                const iv: @Vector(vl, f32) = in_row[k..][0..vl].*;
                const wv: @Vector(vl, f32) = w_row[k..][0..vl].*;
                acc += iv * wv;
            }
            var sum: f32 = bias[j] + @reduce(.Add, acc);
            while (k < in_dim) : (k += 1) {
                sum += in_row[k] * w_row[k];
            }
            output[r * out_dim + j] = sum;
        }
    }
}

/// Linear forward with quantized weights: output = input @ weight^T + bias (SIMD)
fn linearForwardQ(
    input: []const f32,
    weight: QuantizedWeight,
    bias: []const f32,
    output: []f32,
    rows: usize,
) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    for (0..rows) |r| {
        const in_row = input[r * weight.in_dim ..][0..weight.in_dim];
        const out_row = output[r * weight.out_dim ..][0..weight.out_dim];

        switch (weight.type_) {
            .q4_0 => dequant_mod.matmulQ4_0_f32(
                weight.data,
                in_row,
                out_row,
                weight.out_dim,
                weight.in_dim,
            ),
            .f32 => {
                // f32 重み: バイト列を f32 として再解釈 (SIMD)
                const w: [*]const f32 = @ptrCast(@alignCast(weight.data.ptr));
                for (0..weight.out_dim) |j| {
                    var acc: @Vector(vl, f32) = @splat(0);
                    var k: usize = 0;
                    while (k + vl <= weight.in_dim) : (k += vl) {
                        const iv: @Vector(vl, f32) = in_row[k..][0..vl].*;
                        const wv_ptr: [*]const f32 = @ptrCast(@alignCast(weight.data.ptr));
                        const wv: @Vector(vl, f32) =
                            wv_ptr[j * weight.in_dim + k ..][0..vl].*;
                        acc += iv * wv;
                    }
                    var sum: f32 = @reduce(.Add, acc);
                    while (k < weight.in_dim) : (k += 1) {
                        sum += in_row[k] * w[j * weight.in_dim + k];
                    }
                    out_row[j] = sum;
                }
            },
            else => unreachable,
        }

        // Add bias (SIMD)
        var i: usize = 0;
        while (i + vl <= weight.out_dim) : (i += vl) {
            const ov: @Vector(vl, f32) = out_row[i..][0..vl].*;
            const bv: @Vector(vl, f32) = bias[i..][0..vl].*;
            out_row[i..][0..vl].* = ov + bv;
        }
        while (i < weight.out_dim) : (i += 1) {
            out_row[i] += bias[i];
        }
    }
}

/// GELU activation (tanh approximation, in-place)
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn geluInPlace(x: []f32, n: usize) void {
    const sqrt_2_over_pi: f32 = 0.7978845608; // sqrt(2/pi)
    for (0..n) |i| {
        const v = x[i];
        const inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
        x[i] = 0.5 * v * (1.0 + std.math.tanh(inner));
    }
}

/// Softmax (in-place, numerically stable, SIMD)
fn softmaxInPlace(x: []f32, n: usize) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;

    // Max (SIMD)
    var max_v: @Vector(vl, f32) = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + vl <= n) : (i += vl) {
        max_v = @max(max_v, @as(@Vector(vl, f32), x[i..][0..vl].*));
    }
    var max_val: f32 = @reduce(.Max, max_v);
    while (i < n) : (i += 1) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Exp + Sum (scalar, @exp doesn't work on vectors in Zig 0.15)
    var sum: f32 = 0;
    for (0..n) |j| {
        x[j] = @exp(x[j] - max_val);
        sum += x[j];
    }

    // Divide (SIMD)
    const inv_sum: @Vector(vl, f32) = @splat(1.0 / sum);
    i = 0;
    while (i + vl <= n) : (i += vl) {
        const v: @Vector(vl, f32) = x[i..][0..vl].*;
        x[i..][0..vl].* = v * inv_sum;
    }
    while (i < n) : (i += 1) {
        x[i] /= sum;
    }
}

/// Causal multi-head self-attention (SIMD)
/// qkv: (seq_len, 3*embed) - interleaved Q, K, V
/// output: (seq_len, embed)
fn causalSelfAttention(
    qkv: []const f32,
    output: []f32,
    scores_buf: []f32,
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
    embed: usize,
) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output[0 .. seq_len * embed], 0);

    for (0..n_head) |h| {
        const q_offset = h * head_dim;
        const k_offset = embed + h * head_dim;
        const v_offset = 2 * embed + h * head_dim;

        for (0..seq_len) |qi| {
            // Compute attention scores (SIMD)
            for (0..qi + 1) |ki| {
                var acc: @Vector(vl, f32) = @splat(0);
                var d: usize = 0;
                while (d + vl <= head_dim) : (d += vl) {
                    const qv: @Vector(vl, f32) = qkv[qi * 3 * embed + q_offset + d ..][0..vl].*;
                    const kv: @Vector(vl, f32) = qkv[ki * 3 * embed + k_offset + d ..][0..vl].*;
                    acc += qv * kv;
                }
                var dot: f32 = @reduce(.Add, acc);
                while (d < head_dim) : (d += 1) {
                    dot += qkv[qi * 3 * embed + q_offset + d] * qkv[ki * 3 * embed + k_offset + d];
                }
                scores_buf[ki] = dot * scale;
            }

            softmaxInPlace(scores_buf[0 .. qi + 1], qi + 1);

            // Weighted sum of values (SIMD)
            for (0..qi + 1) |ki| {
                const w_scalar = scores_buf[ki];
                const wv: @Vector(vl, f32) = @splat(w_scalar);
                var d: usize = 0;
                while (d + vl <= head_dim) : (d += vl) {
                    var ov: @Vector(vl, f32) = output[qi * embed + h * head_dim + d ..][0..vl].*;
                    const vv: @Vector(vl, f32) = qkv[ki * 3 * embed + v_offset + d ..][0..vl].*;
                    ov += wv * vv;
                    output[qi * embed + h * head_dim + d ..][0..vl].* = ov;
                }
                while (d < head_dim) : (d += 1) {
                    output[qi * embed + h * head_dim + d] +=
                        w_scalar * qkv[ki * 3 * embed + v_offset + d];
                }
            }
        }
    }
}

/// Cached attention: 1つの Q が全キャッシュ K/V に attend (SIMD)
/// q: (EMBED,), k_cache/v_cache: (kv_len * EMBED), output: (EMBED,)
fn cachedAttention(
    q: []const f32,
    k_cache: []const f32,
    v_cache: []const f32,
    output: []f32,
    scores_buf: []f32,
    kv_len: usize,
    n_head: usize,
    head_dim: usize,
    embed: usize,
) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output[0..embed], 0);

    for (0..n_head) |h| {
        const h_off = h * head_dim;

        // dot(Q_h, K_h[ki]) for all cached positions (SIMD)
        for (0..kv_len) |ki| {
            var acc: @Vector(vl, f32) = @splat(0);
            var d: usize = 0;
            while (d + vl <= head_dim) : (d += vl) {
                const qv: @Vector(vl, f32) = q[h_off + d ..][0..vl].*;
                const kv: @Vector(vl, f32) = k_cache[ki * embed + h_off + d ..][0..vl].*;
                acc += qv * kv;
            }
            var dot: f32 = @reduce(.Add, acc);
            while (d < head_dim) : (d += 1) {
                dot += q[h_off + d] * k_cache[ki * embed + h_off + d];
            }
            scores_buf[ki] = dot * scale;
        }

        softmaxInPlace(scores_buf[0..kv_len], kv_len);

        // Weighted sum of V (SIMD)
        for (0..kv_len) |ki| {
            const w_scalar = scores_buf[ki];
            const wv: @Vector(vl, f32) = @splat(w_scalar);
            var d: usize = 0;
            while (d + vl <= head_dim) : (d += vl) {
                var ov: @Vector(vl, f32) = output[h_off + d ..][0..vl].*;
                const vv: @Vector(vl, f32) = v_cache[ki * embed + h_off + d ..][0..vl].*;
                ov += wv * vv;
                output[h_off + d ..][0..vl].* = ov;
            }
            while (d < head_dim) : (d += 1) {
                output[h_off + d] += w_scalar * v_cache[ki * embed + h_off + d];
            }
        }
    }
}

/// Element-wise add in-place: a += b (SIMD)
fn addInPlace(a: []f32, b: []const f32, n: usize) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    var i: usize = 0;
    while (i + vl <= n) : (i += vl) {
        const va: @Vector(vl, f32) = a[i..][0..vl].*;
        const vb: @Vector(vl, f32) = b[i..][0..vl].*;
        a[i..][0..vl].* = va + vb;
    }
    while (i < n) : (i += 1) {
        a[i] += b[i];
    }
}

const SAMPLE_MAX_K: usize = 256;

/// 全要素 softmax → 累積分布からサンプリング (top_k 無効時のフォールバック)
fn sampleFullSoftmax(logits: []f32, temperature: f32, rng: std.Random) u32 {
    const n = logits.len;
    if (temperature > 0 and temperature != 1.0) {
        const inv_temp = 1.0 / temperature;
        for (0..n) |i| logits[i] *= inv_temp;
    }
    softmaxInPlace(logits, n);
    const rand_val = rng.float(f32);
    var cumsum: f32 = 0;
    for (0..n) |i| {
        cumsum += logits[i];
        if (cumsum >= rand_val) return @intCast(i);
    }
    return @intCast(n - 1);
}

/// min-buffer による O(n) top-k 収集。
/// top_vals / top_idxs の先頭 actual_k 要素を埋める。
fn sampleTopKCollect(
    logits: []const f32,
    actual_k: usize,
    inv_temp: f32,
    top_vals: *[SAMPLE_MAX_K]f32,
    top_idxs: *[SAMPLE_MAX_K]u32,
) void {
    const n = logits.len;
    var top_count: usize = 0;
    var min_val: f32 = -std.math.inf(f32);
    var min_pos: usize = 0;

    for (0..n) |i| {
        const v = logits[i] * inv_temp;
        if (top_count < actual_k) {
            top_vals[top_count] = v;
            top_idxs[top_count] = @intCast(i);
            top_count += 1;
            if (top_count == actual_k) {
                min_val = top_vals[0];
                min_pos = 0;
                for (1..actual_k) |j| {
                    if (top_vals[j] < min_val) {
                        min_val = top_vals[j];
                        min_pos = j;
                    }
                }
            }
        } else if (v > min_val) {
            top_vals[min_pos] = v;
            top_idxs[min_pos] = @intCast(i);
            min_val = top_vals[0];
            min_pos = 0;
            for (1..actual_k) |j| {
                if (top_vals[j] < min_val) {
                    min_val = top_vals[j];
                    min_pos = j;
                }
            }
        }
    }
}

/// top_vals 先頭 actual_k 要素を softmax した後、累積分布からサンプリング。
fn sampleTopKChoose(
    top_vals: *[SAMPLE_MAX_K]f32,
    top_idxs: *const [SAMPLE_MAX_K]u32,
    actual_k: usize,
    rng: std.Random,
) u32 {
    var max_val: f32 = top_vals[0];
    for (1..actual_k) |j| {
        if (top_vals[j] > max_val) max_val = top_vals[j];
    }
    var sum_exp: f32 = 0;
    for (0..actual_k) |j| {
        const e = @exp(top_vals[j] - max_val);
        top_vals[j] = e;
        sum_exp += e;
    }
    const inv_sum = 1.0 / sum_exp;
    for (0..actual_k) |j| {
        top_vals[j] *= inv_sum;
    }

    const rand_val = rng.float(f32);
    var cumsum: f32 = 0;
    for (0..actual_k) |j| {
        cumsum += top_vals[j];
        if (cumsum >= rand_val) return top_idxs[j];
    }
    return top_idxs[actual_k - 1];
}

/// Temperature + top-k サンプリング
pub fn sampleTopK(logits: []f32, top_k: usize, temperature: f32, rng: std.Random) u32 {
    const n = logits.len;
    const k = if (top_k > 0 and top_k < n) top_k else n;

    // top_k が無効なら全 logits でサンプリング (フォールバック)
    if (k >= n) return sampleFullSoftmax(logits, temperature, rng);

    // O(n) 1パスで top-k を収集 (min-buffer)
    // その後 k 要素だけで softmax + sampling → O(n + k) total
    const actual_k = if (k <= SAMPLE_MAX_K) k else SAMPLE_MAX_K;
    var top_vals: [SAMPLE_MAX_K]f32 = undefined;
    var top_idxs: [SAMPLE_MAX_K]u32 = undefined;

    const inv_temp: f32 = if (temperature > 0 and temperature != 1.0) 1.0 / temperature else 1.0;
    sampleTopKCollect(logits, actual_k, inv_temp, &top_vals, &top_idxs);
    return sampleTopKChoose(&top_vals, &top_idxs, actual_k, rng);
}

/// Argmax over logits
pub fn argmax(logits: []const f32) u32 {
    var best: u32 = 0;
    var best_val: f32 = logits[0];
    for (1..logits.len) |i| {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = @intCast(i);
        }
    }
    return best;
}

// ============================================================
// テスト
// ============================================================

test "layerNorm basic" {
    // input = [1, 2, 3, 4], weight = [1,1,1,1], bias = [0,0,0,0]
    // mean = 2.5, var = 1.25, inv_std = 1/sqrt(1.25+1e-5)
    const input = [_]f32{ 1, 2, 3, 4 };
    const weight = [_]f32{ 1, 1, 1, 1 };
    const bias = [_]f32{ 0, 0, 0, 0 };
    var output: [4]f32 = undefined;
    layerNormRows(&input, &weight, &bias, &output, 1, 4);

    // output should have mean ≈ 0, sum ≈ 0
    var sum: f32 = 0;
    for (&output) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0), sum, 1e-5);
    // First element should be negative (below mean)
    try std.testing.expect(output[0] < 0);
    // Last element should be positive (above mean)
    try std.testing.expect(output[3] > 0);
}

test "gelu basic" {
    var x = [_]f32{ 0, 1, -1, 2, -2 };
    geluInPlace(&x, 5);
    // gelu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0), x[0], 1e-5);
    // gelu(1) ≈ 0.8412
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), x[1], 0.01);
    // gelu(-1) ≈ -0.1588
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), x[2], 0.01);
    // gelu(x) > 0 for x > 0
    try std.testing.expect(x[3] > 0);
    // gelu(x) ≈ 0 for x << 0
    try std.testing.expect(@abs(x[4]) < 0.1);
}

test "softmax basic" {
    var x = [_]f32{ 1, 2, 3 };
    softmaxInPlace(&x, 3);
    // Should sum to 1
    const sum = x[0] + x[1] + x[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Should be monotonically increasing
    try std.testing.expect(x[0] < x[1]);
    try std.testing.expect(x[1] < x[2]);
}

test "linearForward basic" {
    // input (1, 2), weight (3, 2) row-major (out_dim, in_dim), bias (3)
    // input = [1, 2]
    // weight = [[1, 0],   <- out 0: 1*1 + 0*2 = 1
    //           [0, 1],   <- out 1: 0*1 + 1*2 = 2
    //           [0, 0]]   <- out 2: 0*1 + 0*2 = 0
    // bias = [10, 20, 30]
    // output = [11, 22, 30]
    const input = [_]f32{ 1, 2 };
    const weight = [_]f32{ 1, 0, 0, 1, 0, 0 }; // (3, 2) row-major
    const bias = [_]f32{ 10, 20, 30 };
    var output: [3]f32 = undefined;
    linearForward(&input, &weight, &bias, &output, 1, 2, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 11), output[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22), output[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 30), output[2], 1e-5);
}

test "causalSelfAttention single head" {
    // seq=2, heads=1, head_dim=2, embed=2
    // qkv: (2, 6) - [q0, k0, v0, q1, k1, v1]
    // q0=[1,0], k0=[1,0], v0=[1,2]
    // q1=[0,1], k1=[0,1], v1=[3,4]
    var qkv = [_]f32{
        1, 0, 1, 0, 1, 2, // pos 0
        0, 1, 0, 1, 3, 4, // pos 1
    };
    var output: [4]f32 = undefined;
    var scores: [4]f32 = undefined;
    causalSelfAttention(&qkv, &output, &scores, 2, 1, 2, 2);

    // pos 0: can only attend to pos 0 → softmax([dot(q0,k0)/sqrt(2)]) = [1.0]
    // output[0] = 1.0 * v0 = [1, 2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 1e-4);

    // pos 1: attends to pos 0 and 1
    // Verify output is a valid weighted average of v0 and v1
}

test "argmax" {
    const logits = [_]f32{ -1, 3, 2, 5, 0 };
    try std.testing.expectEqual(@as(u32, 3), argmax(&logits));
}
