/// Gemma3 Metal GPU 推論エンジン
///
/// Metal GPU で全演算を実行する Gemma3 推論パイプライン。
/// Apple Silicon の UMA を活用し、ゼロコピーバッファ共有を実現。
const std = @import("std");
const Allocator = std.mem.Allocator;
const Timer = @import("../util/timer.zig").Timer;
const gguf_mod = @import("../gguf/gguf.zig");
const dequant_mod = @import("../gguf/dequant.zig");
const gemma3_mod = @import("gemma3.zig");
const metal_mod = @import("../backend/metal.zig");
const gpt2_mod = @import("gpt2.zig");

const MetalContext = metal_mod.MetalContext;
const id = metal_mod.id;

pub const Gemma3_1B = gemma3_mod.Gemma3_1B;
pub const ProfileStats = gemma3_mod.ProfileStats;
pub const QuantizedWeight = gpt2_mod.QuantizedWeight;
pub const sampleTopK = gemma3_mod.sampleTopK;
pub const argmax = gemma3_mod.argmax;

// ============================================================
// Metal バッファプール
// ============================================================

fn LayerWeightBuffers() type {
    return struct {
        attn_norm_weight: id,
        post_attention_norm_weight: id,
        attn_q_weight: id,
        attn_k_weight: id,
        attn_v_weight: id,
        attn_output_weight: id,
        attn_q_norm_weight: id,
        attn_k_norm_weight: id,
        ffn_norm_weight: id,
        post_ffw_norm_weight: id,
        ffn_gate_weight: id,
        ffn_up_weight: id,
        ffn_down_weight: id,
    };
}

// ============================================================
// Gemma3 Metal 推論エンジン
// ============================================================

pub fn Gemma3Metal(comptime C: type) type {
    return struct {
        const Self = @This();

        metal: MetalContext,
        weights: gemma3_mod.Gemma3Weights(C),

        // Metal バッファ: 重み
        token_embd_buf: id,
        output_norm_weight_buf: id,
        layer_bufs: [C.LAYER]LayerWeightBuffers(),

        // Metal バッファ: 中間テンソル
        x_buf: id, // (EMBED,) f32
        h_buf: id, // (EMBED,) f32
        q_buf: id, // (Q_DIM,) f32
        k_buf: id, // (KV_DIM,) f32
        v_buf: id, // (KV_DIM,) f32
        attn_out_buf: id, // (Q_DIM,) f32
        proj_out_buf: id, // (EMBED,) f32
        gate_buf: id, // (FFN_DIM,) f32
        up_buf: id, // (FFN_DIM,) f32
        ffn_out_buf: id, // (EMBED,) f32
        norm_buf: id, // (EMBED,) f32
        logits_buf: id, // (VOCAB,) f32
        scores_buf: id, // (CTX * HEAD,) f32 — attention scores scratch
        rope_freqs_buf: id, // (HEAD_DIM/2,) f32

        // KV キャッシュ (Metal バッファ)
        kv_k_bufs: [C.LAYER]id,
        kv_v_bufs: [C.LAYER]id,
        kv_seq_len: usize,

        // MTLFence: エンコーダ内同期
        fence: id,

        profile: ProfileStats,
        allocator: Allocator,

        pub fn init(gguf_file: *const gguf_mod.GGUFFile, allocator: Allocator) !Self {
            var metal = MetalContext.init() catch |err| {
                std.debug.print("Metal init failed: {}\n", .{err});
                return err;
            };

            // CPU 側の重みロード (既存コード再利用)
            const weights = try gemma3_mod.Gemma3Weights(C).loadFromGGUF(gguf_file, allocator);

            var self: Self = undefined;
            self.metal = metal;
            self.weights = weights;
            self.kv_seq_len = 0;
            self.profile = .{};
            self.allocator = allocator;

            // 重みバッファを Metal に転送
            self.token_embd_buf = try metal.createBufferWithData(weights.token_embd.data);
            self.output_norm_weight_buf = try createF32Buffer(&metal, weights.output_norm_weight);

            for (0..C.LAYER) |i| {
                const blk = &weights.blocks[i];
                self.layer_bufs[i] = .{
                    .attn_norm_weight = try createF32Buffer(&metal, blk.attn_norm_weight),
                    .post_attention_norm_weight = try createF32Buffer(&metal, blk.post_attention_norm_weight),
                    .attn_q_weight = try metal.createBufferWithData(blk.attn_q_weight.data),
                    .attn_k_weight = try metal.createBufferWithData(blk.attn_k_weight.data),
                    .attn_v_weight = try metal.createBufferWithData(blk.attn_v_weight.data),
                    .attn_output_weight = try metal.createBufferWithData(blk.attn_output_weight.data),
                    .attn_q_norm_weight = try createF32Buffer(&metal, blk.attn_q_norm_weight),
                    .attn_k_norm_weight = try createF32Buffer(&metal, blk.attn_k_norm_weight),
                    .ffn_norm_weight = try createF32Buffer(&metal, blk.ffn_norm_weight),
                    .post_ffw_norm_weight = try createF32Buffer(&metal, blk.post_ffw_norm_weight),
                    .ffn_gate_weight = try metal.createBufferWithData(blk.ffn_gate_weight.data),
                    .ffn_up_weight = try metal.createBufferWithData(blk.ffn_up_weight.data),
                    .ffn_down_weight = try metal.createBufferWithData(blk.ffn_down_weight.data),
                };
            }

            // 中間バッファ作成
            self.x_buf = try metal.createBuffer(C.EMBED * 4);
            self.h_buf = try metal.createBuffer(C.EMBED * 4);
            self.q_buf = try metal.createBuffer(C.Q_DIM * 4);
            self.k_buf = try metal.createBuffer(C.KV_DIM * 4);
            self.v_buf = try metal.createBuffer(C.KV_DIM * 4);
            self.attn_out_buf = try metal.createBuffer(C.Q_DIM * 4);
            self.proj_out_buf = try metal.createBuffer(C.EMBED * 4);
            self.gate_buf = try metal.createBuffer(C.FFN_DIM * 4);
            self.up_buf = try metal.createBuffer(C.FFN_DIM * 4);
            self.ffn_out_buf = try metal.createBuffer(C.EMBED * 4);
            self.norm_buf = try metal.createBuffer(C.EMBED * 4);
            self.logits_buf = try metal.createBuffer(C.VOCAB * 4);
            self.scores_buf = try metal.createBuffer(C.CTX * C.HEAD * 4);

            // RoPE 周波数テーブル
            var rope_freqs = gemma3_mod.computeRoPEFreqs(C.HEAD_DIM, C.ROPE_BASE);
            self.rope_freqs_buf = try metal.createBufferWithData(std.mem.sliceAsBytes(&rope_freqs));

            // KV キャッシュバッファ
            for (0..C.LAYER) |i| {
                self.kv_k_bufs[i] = try metal.createBuffer(C.CTX * C.KV_DIM * 4);
                self.kv_v_bufs[i] = try metal.createBuffer(C.CTX * C.KV_DIM * 4);
            }

            // MTLFence for intra-encoder synchronization
            self.fence = metal.newFence();

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.metal.deinit();
            self.weights.deinit();
        }

        pub fn resetCache(self: *Self) void {
            self.kv_seq_len = 0;
        }

        /// decodeNext: 1トークンをデコード (全GPU実行)
        pub fn decodeNext(self: *Self, token: u32, arena: Allocator) ![]f32 {
            _ = arena;
            const pos = self.kv_seq_len;
            if (pos >= C.CTX) return error.ContextFull;

            var timer = Timer.start() catch unreachable;

            const cmd_buf = self.metal.newCommandBuffer();
            const fence = self.fence;

            // 全演算を1エンコーダ内で実行、MTLFence で依存関係を同期
            const enc = MetalContext.newComputeEncoder(cmd_buf);

            // Fused Embedding: dequant + scale
            {
                const embed_scale = @sqrt(@as(f32, @floatFromInt(C.EMBED)));
                self.metal.dispatchDequantQ8RowScaled(enc, self.token_embd_buf, self.x_buf, token, @intCast(C.EMBED), embed_scale);
                MetalContext.updateFence(enc, fence);
            }

            // 26 Transformer layers
            for (0..C.LAYER) |layer| {
                const lb = &self.layer_bufs[layer];

                // Pre-attention RMSNorm: x → h
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchRMSNorm(enc, self.x_buf, lb.attn_norm_weight, self.h_buf, @intCast(C.EMBED), 1, C.RMS_EPS);
                MetalContext.updateFence(enc, fence);

                // Q/K/V projections (独立: 全て h_buf 読み、別バッファ書き)
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchMatmul(enc, lb.attn_q_weight, self.h_buf, self.q_buf, @intCast(C.Q_DIM), @intCast(C.EMBED), gemma3_mod.quantTypeOfWeight(self.weights.blocks[layer].attn_q_weight));
                self.metal.dispatchMatmul(enc, lb.attn_k_weight, self.h_buf, self.k_buf, @intCast(C.KV_DIM), @intCast(C.EMBED), gemma3_mod.quantTypeOfWeight(self.weights.blocks[layer].attn_k_weight));
                self.metal.dispatchMatmul(enc, lb.attn_v_weight, self.h_buf, self.v_buf, @intCast(C.KV_DIM), @intCast(C.EMBED), gemma3_mod.quantTypeOfWeight(self.weights.blocks[layer].attn_v_weight));
                MetalContext.updateFence(enc, fence);

                // Per-head Q norm + K norm
                MetalContext.waitForFence(enc, fence);
                for (0..C.HEAD) |hd| {
                    self.metal.dispatchRMSNormInPlace(enc, self.q_buf, @as(u64, hd * C.HEAD_DIM * 4), lb.attn_q_norm_weight, @intCast(C.HEAD_DIM), C.RMS_EPS);
                }
                self.metal.dispatchRMSNormInPlace(enc, self.k_buf, 0, lb.attn_k_norm_weight, @intCast(C.HEAD_DIM), C.RMS_EPS);
                MetalContext.updateFence(enc, fence);

                // Q/K RoPE
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchRoPE(enc, self.q_buf, self.rope_freqs_buf, @intCast(C.HEAD_DIM / 2), @intCast(C.HEAD), @floatFromInt(pos));
                self.metal.dispatchRoPE(enc, self.k_buf, self.rope_freqs_buf, @intCast(C.HEAD_DIM / 2), 1, @floatFromInt(pos));
                MetalContext.updateFence(enc, fence);

                // KV cache 書き込み
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchWriteKVCache(enc, self.k_buf, self.v_buf, self.kv_k_bufs[layer], self.kv_v_bufs[layer], @intCast(pos), @intCast(C.KV_DIM));
                MetalContext.updateFence(enc, fence);

                // GQA Cached Attention
                const is_global = gemma3_mod.isGlobalLayer(layer);
                const kv_len = pos + 1;
                const window = if (is_global) kv_len else @min(kv_len, C.SLIDING_WINDOW);
                const kv_start: u32 = if (kv_len > window) @intCast(kv_len - window) else 0;
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchAttentionDecode(enc, self.q_buf, self.kv_k_bufs[layer], self.kv_v_bufs[layer], self.attn_out_buf, self.scores_buf, @intCast(C.HEAD), @intCast(C.HEAD_DIM), @intCast(C.Q_DIM), @intCast(C.KV_DIM), kv_start, @intCast(kv_len));
                MetalContext.updateFence(enc, fence);

                // Output projection
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchMatmul(enc, lb.attn_output_weight, self.attn_out_buf, self.proj_out_buf, @intCast(C.EMBED), @intCast(C.Q_DIM), gemma3_mod.quantTypeOfWeight(self.weights.blocks[layer].attn_output_weight));
                MetalContext.updateFence(enc, fence);

                // Fused: x += RMSNorm(proj_out) (post-attn norm + residual)
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchRMSNormResidual(enc, self.proj_out_buf, lb.post_attention_norm_weight, self.x_buf, @intCast(C.EMBED), 1, C.RMS_EPS);
                MetalContext.updateFence(enc, fence);

                // Pre-FFN RMSNorm: x → h
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchRMSNorm(enc, self.x_buf, lb.ffn_norm_weight, self.h_buf, @intCast(C.EMBED), 1, C.RMS_EPS);
                MetalContext.updateFence(enc, fence);

                // Gate + Up matmul (独立: h_buf 読み、別バッファ書き)
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchMatmul(enc, lb.ffn_gate_weight, self.h_buf, self.gate_buf, @intCast(C.FFN_DIM), @intCast(C.EMBED), gemma3_mod.quantTypeOfWeight(self.weights.blocks[layer].ffn_gate_weight));
                self.metal.dispatchMatmul(enc, lb.ffn_up_weight, self.h_buf, self.up_buf, @intCast(C.FFN_DIM), @intCast(C.EMBED), gemma3_mod.quantTypeOfWeight(self.weights.blocks[layer].ffn_up_weight));
                MetalContext.updateFence(enc, fence);

                // Fused: gate = GELU(gate) * up
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchGELUMul(enc, self.gate_buf, self.up_buf, @intCast(C.FFN_DIM));
                MetalContext.updateFence(enc, fence);

                // Down matmul
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchMatmul(enc, lb.ffn_down_weight, self.gate_buf, self.ffn_out_buf, @intCast(C.EMBED), @intCast(C.FFN_DIM), gemma3_mod.quantTypeOfWeight(self.weights.blocks[layer].ffn_down_weight));
                MetalContext.updateFence(enc, fence);

                // Fused: x += RMSNorm(ffn_out) (post-FFN norm + residual)
                MetalContext.waitForFence(enc, fence);
                self.metal.dispatchRMSNormResidual(enc, self.ffn_out_buf, lb.post_ffw_norm_weight, self.x_buf, @intCast(C.EMBED), 1, C.RMS_EPS);
                MetalContext.updateFence(enc, fence);
            }

            // Final RMSNorm
            MetalContext.waitForFence(enc, fence);
            self.metal.dispatchRMSNorm(enc, self.x_buf, self.output_norm_weight_buf, self.h_buf, @intCast(C.EMBED), 1, C.RMS_EPS);
            MetalContext.updateFence(enc, fence);

            // Logits
            MetalContext.waitForFence(enc, fence);
            self.metal.dispatchMatmul(enc, self.token_embd_buf, self.h_buf, self.logits_buf, @intCast(C.VOCAB), @intCast(C.EMBED), .q8_0);

            MetalContext.endEncoding(enc);

            // コミット & 完了待ち
            MetalContext.commit(cmd_buf);
            MetalContext.waitUntilCompleted(cmd_buf);

            const elapsed_ns = timer.read();
            self.profile.logits_ns += elapsed_ns;
            self.kv_seq_len = pos + 1;
            self.profile.call_count += 1;

            // Shared バッファから logits を直接返す
            const logits_ptr = MetalContext.bufferContents(f32, self.logits_buf);
            return logits_ptr[0..C.VOCAB];
        }

        /// prefill: プロンプト全体を処理 (token-by-token on GPU)
        pub fn prefill(self: *Self, tokens: []const u32, arena: Allocator) ![]f32 {
            if (tokens.len == 0) return error.EmptySequence;
            if (tokens.len > C.CTX) return error.SequenceTooLong;

            self.resetCache();

            var logits: []f32 = undefined;
            for (tokens) |token| {
                logits = try self.decodeNext(token, arena);
            }
            return logits;
        }
    };
}


fn createF32Buffer(metal: *MetalContext, data: []const f32) !id {
    return metal.createBufferWithData(std.mem.sliceAsBytes(data));
}

