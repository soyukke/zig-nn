/// Gemma3 Metal GPU 推論エンジン
///
/// Metal GPU で全演算を実行する Gemma3 推論パイプライン。
/// Apple Silicon の UMA を活用し、ゼロコピーバッファ共有を実現。
const std = @import("std");
const Allocator = std.mem.Allocator;
const Timer = @import("../util/timer.zig").Timer;
const log = @import("../log.zig").gemma3;
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
pub const sample_top_k = gemma3_mod.sample_top_k;
pub const argmax = gemma3_mod.argmax;

// ============================================================
// Metal バッファプール
// ============================================================

fn layer_weight_buffers() type {
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

pub fn gemma3_metal(comptime C: type) type {
    return struct {
        const Self = @This();

        metal: MetalContext,
        weights: gemma3_mod.gemma3_weights(C),

        // Metal バッファ: 重み
        token_embd_buf: id,
        output_norm_weight_buf: id,
        layer_bufs: [C.LAYER]layer_weight_buffers(),

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
                log.err("Metal init failed: {}", .{err});
                return err;
            };

            // CPU 側の重みロード (既存コード再利用)
            const weights = try gemma3_mod.gemma3_weights(C).load_from_gguf(gguf_file, allocator);

            var self: Self = undefined;
            self.metal = metal;
            self.weights = weights;
            self.kv_seq_len = 0;
            self.profile = .{};
            self.allocator = allocator;

            // 重みバッファを Metal に転送
            self.token_embd_buf = try metal.create_buffer_with_data(weights.token_embd.data);
            self.output_norm_weight_buf = try create_f32_buffer(&metal, weights.output_norm_weight);
            try self.init_layer_weight_buffers(&metal, &weights);

            // 中間バッファ作成
            try self.init_intermediate_buffers(&metal);

            // RoPE 周波数テーブル
            var rope_freqs = gemma3_mod.compute_rope_freqs(C.HEAD_DIM, C.ROPE_BASE);
            self.rope_freqs_buf = try metal.create_buffer_with_data(
                std.mem.sliceAsBytes(&rope_freqs),
            );

            // KV キャッシュバッファ
            for (0..C.LAYER) |i| {
                self.kv_k_bufs[i] = try metal.create_buffer(C.CTX * C.KV_DIM * 4);
                self.kv_v_bufs[i] = try metal.create_buffer(C.CTX * C.KV_DIM * 4);
            }

            // MTLFence for intra-encoder synchronization
            self.fence = metal.new_fence();

            return self;
        }

        /// 全 Transformer ブロックの重みバッファを作成
        fn init_layer_weight_buffers(
            self: *Self,
            metal: *MetalContext,
            weights: *const gemma3_mod.gemma3_weights(C),
        ) !void {
            for (0..C.LAYER) |i| {
                const blk = &weights.blocks[i];
                self.layer_bufs[i] = .{
                    .attn_norm_weight = try create_f32_buffer(metal, blk.attn_norm_weight),
                    .post_attention_norm_weight = try create_f32_buffer(
                        metal,
                        blk.post_attention_norm_weight,
                    ),
                    .attn_q_weight = try metal.create_buffer_with_data(blk.attn_q_weight.data),
                    .attn_k_weight = try metal.create_buffer_with_data(blk.attn_k_weight.data),
                    .attn_v_weight = try metal.create_buffer_with_data(blk.attn_v_weight.data),
                    .attn_output_weight = try metal.create_buffer_with_data(
                        blk.attn_output_weight.data,
                    ),
                    .attn_q_norm_weight = try create_f32_buffer(metal, blk.attn_q_norm_weight),
                    .attn_k_norm_weight = try create_f32_buffer(metal, blk.attn_k_norm_weight),
                    .ffn_norm_weight = try create_f32_buffer(metal, blk.ffn_norm_weight),
                    .post_ffw_norm_weight = try create_f32_buffer(metal, blk.post_ffw_norm_weight),
                    .ffn_gate_weight = try metal.create_buffer_with_data(blk.ffn_gate_weight.data),
                    .ffn_up_weight = try metal.create_buffer_with_data(blk.ffn_up_weight.data),
                    .ffn_down_weight = try metal.create_buffer_with_data(blk.ffn_down_weight.data),
                };
            }
        }

        /// 各種中間テンソル用バッファ (x/h/q/k/v/ffn/logits/scores) を確保
        fn init_intermediate_buffers(self: *Self, metal: *MetalContext) !void {
            self.x_buf = try metal.create_buffer(C.EMBED * 4);
            self.h_buf = try metal.create_buffer(C.EMBED * 4);
            self.q_buf = try metal.create_buffer(C.Q_DIM * 4);
            self.k_buf = try metal.create_buffer(C.KV_DIM * 4);
            self.v_buf = try metal.create_buffer(C.KV_DIM * 4);
            self.attn_out_buf = try metal.create_buffer(C.Q_DIM * 4);
            self.proj_out_buf = try metal.create_buffer(C.EMBED * 4);
            self.gate_buf = try metal.create_buffer(C.FFN_DIM * 4);
            self.up_buf = try metal.create_buffer(C.FFN_DIM * 4);
            self.ffn_out_buf = try metal.create_buffer(C.EMBED * 4);
            self.norm_buf = try metal.create_buffer(C.EMBED * 4);
            self.logits_buf = try metal.create_buffer(C.VOCAB * 4);
            self.scores_buf = try metal.create_buffer(C.CTX * C.HEAD * 4);
        }

        pub fn deinit(self: *Self) void {
            self.metal.deinit();
            self.weights.deinit();
        }

        pub fn reset_cache(self: *Self) void {
            self.kv_seq_len = 0;
        }

        /// decodeNext: 1トークンをデコード (全GPU実行)
        pub fn decode_next(self: *Self, token: u32, arena: Allocator) ![]f32 {
            _ = arena;
            const pos = self.kv_seq_len;
            if (pos >= C.CTX) return error.ContextFull;

            var timer = Timer.start() catch unreachable;

            const cmd_buf = self.metal.new_command_buffer();
            const fence = self.fence;

            // 全演算を1エンコーダ内で実行、MTLFence で依存関係を同期
            const enc = MetalContext.new_compute_encoder(cmd_buf);

            // Fused Embedding: dequant + scale
            self.decode_next_embed(enc, fence, token);

            // 26 Transformer layers
            for (0..C.LAYER) |layer| {
                self.decode_next_layer(enc, fence, layer, pos);
            }

            // Final norm + logits matmul
            self.decode_next_final_logits(enc, fence);

            MetalContext.end_encoding(enc);

            // コミット & 完了待ち
            MetalContext.commit(cmd_buf);
            MetalContext.wait_until_completed(cmd_buf);

            const elapsed_ns = timer.read();
            self.profile.logits_ns += elapsed_ns;
            self.kv_seq_len = pos + 1;
            self.profile.call_count += 1;

            // Shared バッファから logits を直接返す
            const logits_ptr = MetalContext.buffer_contents(f32, self.logits_buf);
            return logits_ptr[0..C.VOCAB];
        }

        /// Fused Embedding: dequant + scale で x_buf を初期化
        fn decode_next_embed(self: *Self, enc: id, fence: id, token: u32) void {
            const embed_scale = @sqrt(@as(f32, @floatFromInt(C.EMBED)));
            self.metal.dispatch_dequant_q8_row_scaled(
                enc,
                self.token_embd_buf,
                self.x_buf,
                token,
                @intCast(C.EMBED),
                embed_scale,
            );
            MetalContext.update_fence(enc, fence);
        }

        /// 1 Transformer ブロック分の演算 (attention + FFN) をエンコード
        fn decode_next_layer(self: *Self, enc: id, fence: id, layer: usize, pos: usize) void {
            const lb = &self.layer_bufs[layer];

            self.decode_next_pre_attn_norm(enc, fence, lb);
            self.decode_next_qkv_projections(enc, fence, lb, layer);
            self.decode_next_qk_norm(enc, fence, lb);
            self.decode_next_rope(enc, fence, pos);
            self.decode_next_write_kv_cache(enc, fence, layer, pos);
            self.decode_next_attention(enc, fence, layer, pos);
            self.decode_next_output_proj(enc, fence, lb, layer);
            self.decode_next_post_attn_residual(enc, fence, lb);
            self.decode_next_pre_ffn_norm(enc, fence, lb);
            self.decode_next_ffn_gate_up(enc, fence, lb, layer);
            self.decode_next_ffn_gelu_mul(enc, fence);
            self.decode_next_ffn_down(enc, fence, lb, layer);
            self.decode_next_post_ffn_residual(enc, fence, lb);
        }

        /// Pre-attention RMSNorm: x → h
        fn decode_next_pre_attn_norm(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_rms_norm(
                enc,
                self.x_buf,
                lb.attn_norm_weight,
                self.h_buf,
                @intCast(C.EMBED),
                1,
                C.RMS_EPS,
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Q/K/V projections (独立: 全て h_buf 読み、別バッファ書き)
        fn decode_next_qkv_projections(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
            layer: usize,
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_matmul(
                enc,
                lb.attn_q_weight,
                self.h_buf,
                self.q_buf,
                @intCast(C.Q_DIM),
                @intCast(C.EMBED),
                gemma3_mod.quant_type_of_weight(self.weights.blocks[layer].attn_q_weight),
            );
            self.metal.dispatch_matmul(
                enc,
                lb.attn_k_weight,
                self.h_buf,
                self.k_buf,
                @intCast(C.KV_DIM),
                @intCast(C.EMBED),
                gemma3_mod.quant_type_of_weight(self.weights.blocks[layer].attn_k_weight),
            );
            self.metal.dispatch_matmul(
                enc,
                lb.attn_v_weight,
                self.h_buf,
                self.v_buf,
                @intCast(C.KV_DIM),
                @intCast(C.EMBED),
                gemma3_mod.quant_type_of_weight(self.weights.blocks[layer].attn_v_weight),
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Per-head Q norm + K norm
        fn decode_next_qk_norm(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            for (0..C.HEAD) |hd| {
                self.metal.dispatch_rms_norm_in_place(
                    enc,
                    self.q_buf,
                    @as(u64, hd * C.HEAD_DIM * 4),
                    lb.attn_q_norm_weight,
                    @intCast(C.HEAD_DIM),
                    C.RMS_EPS,
                );
            }
            self.metal.dispatch_rms_norm_in_place(
                enc,
                self.k_buf,
                0,
                lb.attn_k_norm_weight,
                @intCast(C.HEAD_DIM),
                C.RMS_EPS,
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Q/K RoPE
        fn decode_next_rope(self: *Self, enc: id, fence: id, pos: usize) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_rope(
                enc,
                self.q_buf,
                self.rope_freqs_buf,
                @intCast(C.HEAD_DIM / 2),
                @intCast(C.HEAD),
                @floatFromInt(pos),
            );
            self.metal.dispatch_rope(
                enc,
                self.k_buf,
                self.rope_freqs_buf,
                @intCast(C.HEAD_DIM / 2),
                1,
                @floatFromInt(pos),
            );
            MetalContext.update_fence(enc, fence);
        }

        /// KV cache 書き込み
        fn decode_next_write_kv_cache(
            self: *Self,
            enc: id,
            fence: id,
            layer: usize,
            pos: usize,
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_write_kv_cache(
                enc,
                self.k_buf,
                self.v_buf,
                self.kv_k_bufs[layer],
                self.kv_v_bufs[layer],
                @intCast(pos),
                @intCast(C.KV_DIM),
            );
            MetalContext.update_fence(enc, fence);
        }

        /// GQA Cached Attention (sliding window 対応)
        fn decode_next_attention(
            self: *Self,
            enc: id,
            fence: id,
            layer: usize,
            pos: usize,
        ) void {
            const is_global = gemma3_mod.is_global_layer(layer);
            const kv_len = pos + 1;
            const window = if (is_global) kv_len else @min(kv_len, C.SLIDING_WINDOW);
            const kv_start: u32 = if (kv_len > window) @intCast(kv_len - window) else 0;
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_attention_decode(
                enc,
                self.q_buf,
                self.kv_k_bufs[layer],
                self.kv_v_bufs[layer],
                self.attn_out_buf,
                self.scores_buf,
                @intCast(C.HEAD),
                @intCast(C.HEAD_DIM),
                @intCast(C.Q_DIM),
                @intCast(C.KV_DIM),
                kv_start,
                @intCast(kv_len),
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Output projection
        fn decode_next_output_proj(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
            layer: usize,
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_matmul(
                enc,
                lb.attn_output_weight,
                self.attn_out_buf,
                self.proj_out_buf,
                @intCast(C.EMBED),
                @intCast(C.Q_DIM),
                gemma3_mod.quant_type_of_weight(self.weights.blocks[layer].attn_output_weight),
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Fused: x += RMSNorm(proj_out) (post-attn norm + residual)
        fn decode_next_post_attn_residual(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_rms_norm_residual(
                enc,
                self.proj_out_buf,
                lb.post_attention_norm_weight,
                self.x_buf,
                @intCast(C.EMBED),
                1,
                C.RMS_EPS,
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Pre-FFN RMSNorm: x → h
        fn decode_next_pre_ffn_norm(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_rms_norm(
                enc,
                self.x_buf,
                lb.ffn_norm_weight,
                self.h_buf,
                @intCast(C.EMBED),
                1,
                C.RMS_EPS,
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Gate + Up matmul (独立: h_buf 読み、別バッファ書き)
        fn decode_next_ffn_gate_up(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
            layer: usize,
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_matmul(
                enc,
                lb.ffn_gate_weight,
                self.h_buf,
                self.gate_buf,
                @intCast(C.FFN_DIM),
                @intCast(C.EMBED),
                gemma3_mod.quant_type_of_weight(self.weights.blocks[layer].ffn_gate_weight),
            );
            self.metal.dispatch_matmul(
                enc,
                lb.ffn_up_weight,
                self.h_buf,
                self.up_buf,
                @intCast(C.FFN_DIM),
                @intCast(C.EMBED),
                gemma3_mod.quant_type_of_weight(self.weights.blocks[layer].ffn_up_weight),
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Fused: gate = GELU(gate) * up
        fn decode_next_ffn_gelu_mul(self: *Self, enc: id, fence: id) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_gelu_mul(enc, self.gate_buf, self.up_buf, @intCast(C.FFN_DIM));
            MetalContext.update_fence(enc, fence);
        }

        /// Down matmul: gate_buf → ffn_out_buf
        fn decode_next_ffn_down(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
            layer: usize,
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_matmul(
                enc,
                lb.ffn_down_weight,
                self.gate_buf,
                self.ffn_out_buf,
                @intCast(C.EMBED),
                @intCast(C.FFN_DIM),
                gemma3_mod.quant_type_of_weight(self.weights.blocks[layer].ffn_down_weight),
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Fused: x += RMSNorm(ffn_out) (post-FFN norm + residual)
        fn decode_next_post_ffn_residual(
            self: *Self,
            enc: id,
            fence: id,
            lb: *const layer_weight_buffers(),
        ) void {
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_rms_norm_residual(
                enc,
                self.ffn_out_buf,
                lb.post_ffw_norm_weight,
                self.x_buf,
                @intCast(C.EMBED),
                1,
                C.RMS_EPS,
            );
            MetalContext.update_fence(enc, fence);
        }

        /// Final RMSNorm + logits 投影
        fn decode_next_final_logits(self: *Self, enc: id, fence: id) void {
            // Final RMSNorm
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_rms_norm(
                enc,
                self.x_buf,
                self.output_norm_weight_buf,
                self.h_buf,
                @intCast(C.EMBED),
                1,
                C.RMS_EPS,
            );
            MetalContext.update_fence(enc, fence);

            // Logits
            MetalContext.wait_for_fence(enc, fence);
            self.metal.dispatch_matmul(
                enc,
                self.token_embd_buf,
                self.h_buf,
                self.logits_buf,
                @intCast(C.VOCAB),
                @intCast(C.EMBED),
                .q8_0,
            );
        }

        /// prefill: プロンプト全体を処理 (token-by-token on GPU)
        pub fn prefill(self: *Self, tokens: []const u32, arena: Allocator) ![]f32 {
            if (tokens.len == 0) return error.EmptySequence;
            if (tokens.len > C.CTX) return error.SequenceTooLong;

            self.reset_cache();

            var logits: []f32 = undefined;
            for (tokens) |token| {
                logits = try self.decode_next(token, arena);
            }
            return logits;
        }
    };
}

fn create_f32_buffer(metal: *MetalContext, data: []const f32) !id {
    return metal.create_buffer_with_data(std.mem.sliceAsBytes(data));
}
