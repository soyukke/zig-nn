/// Gemma3 QLoRA fine-tuning (unified API).
///
/// 凍結量子化ベース重み (Q4_0/Q4_1/Q8_0) + 学習可能な LoRA A/B adapter を
/// Q/V projection に付与し、DiffMpsRuntime 上で forward/backward を行う。
///
/// 設計:
///   - base 重み (token_embd, attn_q/k/v/o, ffn_gate/up/down) は `Module` に
///     登録せず、GPU-side QuantWeight として保持 (勾配なし)。
///   - LoRA A/B と 各 RMSNorm weight は `ParamHandle` として `Module` に登録。
///   - 使い方:
///       var module = compute.Module.init(alloc);
///       const model = Gemma3QLoRA(C, RANK).initParams(&module);
///       var rt = try DiffMpsRuntime.init(&module, mtl, alloc);
///       rt.initParams();                           // xavier/zeros/ones で初期化
///       try model.loadFromGguf(&rt, &gguf_file);   // GGUF から RMSNorm + 量子化重みをロード
///       ...
///       const logits = model.forward(&rt, input_ids);
const std = @import("std");
const Allocator = std.mem.Allocator;

const compute = @import("../compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const gguf_mod = @import("../gguf/gguf.zig");
const gemma3_mod = @import("gemma3.zig");
const metal = @import("../backend/metal.zig");
const MetalContext = metal.MetalContext;
const id = metal.id;
const diff_mps = @import("../diff/mps_runtime.zig");
const DiffMpsRuntime = diff_mps.DiffMpsRuntime;
const DiffMpsTensor = diff_mps.DiffMpsTensor;

pub const Gemma3_1B = gemma3_mod.Gemma3_1B;

fn param_buf(rt: *DiffMpsRuntime, handle: ParamHandle) [*]f32 {
    return MetalContext.buffer_contents(f32, rt.param_nodes[handle.index].data);
}

pub fn gemma3_qlora(comptime C: type, comptime RANK: usize) type {
    const EMBED = C.EMBED;
    const Q_DIM = C.Q_DIM;
    const KV_DIM = C.KV_DIM;
    const HEAD = C.HEAD;
    const HEAD_KV = C.HEAD_KV;
    const HEAD_DIM = C.HEAD_DIM;
    const FFN_DIM = C.FFN_DIM;
    const LAYER = C.LAYER;
    const VOCAB = C.VOCAB;
    const HALF_DIM = HEAD_DIM / 2;
    const SCALING: f32 = 16.0 / @as(f32, @floatFromInt(RANK)); // α=16
    const INV_SQRT_HEAD_DIM: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HEAD_DIM)));
    const QW = DiffMpsRuntime.QuantWeight;

    return struct {
        const Self = @This();

        // Frozen GPU-side quantized weights (not in Module).
        token_embd: QW,
        layers: [LAYER]LayerQuant,

        // RoPE precomputed frequency table (GPU buffer).
        rope_freqs_buf: id,

        // Trainable parameter handles.
        lora_q_a: [LAYER]ParamHandle,
        lora_q_b: [LAYER]ParamHandle,
        lora_v_a: [LAYER]ParamHandle,
        lora_v_b: [LAYER]ParamHandle,
        attn_norm: [LAYER]ParamHandle,
        ffn_norm: [LAYER]ParamHandle,
        q_norm: [LAYER]ParamHandle,
        k_norm: [LAYER]ParamHandle,
        post_attn_norm: [LAYER]ParamHandle,
        post_ffw_norm: [LAYER]ParamHandle,
        output_norm: ParamHandle,

        const LayerQuant = struct {
            q: QW,
            k: QW,
            v: QW,
            o: QW,
            gate: QW,
            up: QW,
            down: QW,
        };

        /// Step 1: register all trainable parameters in the `Module`.
        /// Called before `DiffMpsRuntime.init`.
        pub fn init_params(module: *Module) Self {
            var self: Self = undefined;
            for (0..LAYER) |i| {
                self.lora_q_a[i] = module.add_param(&.{ EMBED, RANK }, .xavier);
                self.lora_q_b[i] = module.add_param(&.{ RANK, Q_DIM }, .zeros);
                self.lora_v_a[i] = module.add_param(&.{ EMBED, RANK }, .xavier);
                self.lora_v_b[i] = module.add_param(&.{ RANK, KV_DIM }, .zeros);
                self.attn_norm[i] = module.add_param(&.{EMBED}, .ones);
                self.ffn_norm[i] = module.add_param(&.{EMBED}, .ones);
                self.q_norm[i] = module.add_param(&.{HEAD_DIM}, .ones);
                self.k_norm[i] = module.add_param(&.{HEAD_DIM}, .ones);
                self.post_attn_norm[i] = module.add_param(&.{EMBED}, .ones);
                self.post_ffw_norm[i] = module.add_param(&.{EMBED}, .ones);
            }
            self.output_norm = module.add_param(&.{EMBED}, .ones);
            return self;
        }

        /// 1 レイヤ分の凍結量子化重みを GPU バッファに載せる
        fn build_layer_quant(
            rt: *DiffMpsRuntime,
            blk: *const gemma3_mod.GemmaBlockWeights,
        ) !LayerQuant {
            const qt = gemma3_mod.quant_type_of_weight;
            return .{
                .q = .{
                    .buf = try rt.metal_ctx.create_buffer_with_data(blk.attn_q_weight.data),
                    .quant_type = qt(blk.attn_q_weight),
                    .out_dim = @intCast(Q_DIM),
                    .in_dim = @intCast(EMBED),
                },
                .k = .{
                    .buf = try rt.metal_ctx.create_buffer_with_data(blk.attn_k_weight.data),
                    .quant_type = qt(blk.attn_k_weight),
                    .out_dim = @intCast(KV_DIM),
                    .in_dim = @intCast(EMBED),
                },
                .v = .{
                    .buf = try rt.metal_ctx.create_buffer_with_data(blk.attn_v_weight.data),
                    .quant_type = qt(blk.attn_v_weight),
                    .out_dim = @intCast(KV_DIM),
                    .in_dim = @intCast(EMBED),
                },
                .o = .{
                    .buf = try rt.metal_ctx.create_buffer_with_data(blk.attn_output_weight.data),
                    .quant_type = qt(blk.attn_output_weight),
                    .out_dim = @intCast(EMBED),
                    .in_dim = @intCast(Q_DIM),
                },
                .gate = .{
                    .buf = try rt.metal_ctx.create_buffer_with_data(blk.ffn_gate_weight.data),
                    .quant_type = qt(blk.ffn_gate_weight),
                    .out_dim = @intCast(FFN_DIM),
                    .in_dim = @intCast(EMBED),
                },
                .up = .{
                    .buf = try rt.metal_ctx.create_buffer_with_data(blk.ffn_up_weight.data),
                    .quant_type = qt(blk.ffn_up_weight),
                    .out_dim = @intCast(FFN_DIM),
                    .in_dim = @intCast(EMBED),
                },
                .down = .{
                    .buf = try rt.metal_ctx.create_buffer_with_data(blk.ffn_down_weight.data),
                    .quant_type = qt(blk.ffn_down_weight),
                    .out_dim = @intCast(EMBED),
                    .in_dim = @intCast(FFN_DIM),
                },
            };
        }

        /// RMSNorm パラメータを GGUF から上書きする (そのままだと .ones のまま)
        fn copy_layer_norms(
            self: *Self,
            rt: *DiffMpsRuntime,
            blk: *const gemma3_mod.GemmaBlockWeights,
            i: usize,
        ) void {
            @memcpy(param_buf(rt, self.attn_norm[i])[0..EMBED], blk.attn_norm_weight);
            @memcpy(param_buf(rt, self.ffn_norm[i])[0..EMBED], blk.ffn_norm_weight);
            @memcpy(param_buf(rt, self.q_norm[i])[0..HEAD_DIM], blk.attn_q_norm_weight);
            @memcpy(param_buf(rt, self.k_norm[i])[0..HEAD_DIM], blk.attn_k_norm_weight);
            @memcpy(
                param_buf(rt, self.post_attn_norm[i])[0..EMBED],
                blk.post_attention_norm_weight,
            );
            @memcpy(param_buf(rt, self.post_ffw_norm[i])[0..EMBED], blk.post_ffw_norm_weight);
        }

        /// Step 2: load frozen quantized weights into GPU buffers and overwrite
        /// RMSNorm params with values from the GGUF file. Call after `rt.initParams()`.
        pub fn load_from_gguf(
            self: *Self,
            rt: *DiffMpsRuntime,
            gguf_file: *const gguf_mod.GGUFFile,
        ) !void {
            var weights = try gemma3_mod.gemma3_weights(C).load_from_gguf(gguf_file, rt.allocator);
            defer weights.deinit();

            // Token embedding (Q8_0) — also re-used for the final logits projection (weight-tied).
            self.token_embd = .{
                .buf = try rt.metal_ctx.create_buffer_with_data(weights.token_embd.data),
                .quant_type = .q8_0,
                .out_dim = @intCast(VOCAB),
                .in_dim = @intCast(EMBED),
            };

            // RoPE precomputed frequencies (f32).
            var rope_freqs = gemma3_mod.compute_rope_freqs(HEAD_DIM, C.ROPE_BASE);
            self.rope_freqs_buf = try rt.metal_ctx.create_buffer_with_data(
                std.mem.sliceAsBytes(&rope_freqs),
            );

            // Final RMSNorm weight from GGUF.
            @memcpy(param_buf(rt, self.output_norm)[0..EMBED], weights.output_norm_weight);

            for (0..LAYER) |i| {
                const blk = &weights.blocks[i];
                self.layers[i] = try build_layer_quant(rt, blk);
                self.copy_layer_norms(rt, blk, i);
            }
        }

        pub fn deinit(self: *Self) void {
            metal.obj_release(self.token_embd.buf);
            metal.obj_release(self.rope_freqs_buf);
            for (&self.layers) |*lq| {
                metal.obj_release(lq.q.buf);
                metal.obj_release(lq.k.buf);
                metal.obj_release(lq.v.buf);
                metal.obj_release(lq.o.buf);
                metal.obj_release(lq.gate.buf);
                metal.obj_release(lq.up.buf);
                metal.obj_release(lq.down.buf);
            }
        }

        /// Dequant Q8_0 embedding table for the given token ids and scale by √EMBED.
        fn embed(self: *const Self, rt: *DiffMpsRuntime, input_ids: []const u32) DiffMpsTensor {
            const seq_len = input_ids.len;
            const embed_scale: f32 = @sqrt(@as(f32, @floatFromInt(EMBED)));

            // Upload token ids (u32 → MTLBuffer, 4B per element = same count as f32)
            const tok_buf = rt.alloc_buf(seq_len);
            @memcpy(MetalContext.buffer_contents(u32, tok_buf)[0..seq_len], input_ids);

            const emb_buf = rt.alloc_buf(seq_len * EMBED);
            const cmd = rt.metal_ctx.new_command_buffer();
            const enc = MetalContext.new_compute_encoder(cmd);
            rt.metal_ctx.dispatch_dequant_q8_batch_scaled(
                enc,
                self.token_embd.buf,
                tok_buf,
                emb_buf,
                @intCast(seq_len),
                @intCast(EMBED),
                embed_scale,
            );
            MetalContext.memory_barrier(enc);
            MetalContext.end_encoding(enc);
            MetalContext.commit(cmd);
            MetalContext.wait_until_completed(cmd);

            return rt.make_node(emb_buf, &.{ seq_len, EMBED }, false);
        }

        /// Full forward: input_ids → logits [seq_len, VOCAB].
        pub fn forward(
            self: *const Self,
            rt: *DiffMpsRuntime,
            input_ids: []const u32,
        ) DiffMpsTensor {
            const seq_len = input_ids.len;

            var x = self.embed(rt, input_ids);

            for (0..LAYER) |i| {
                x = self.block(rt, x, i, seq_len);
            }

            // Final norm + tied logits head.
            const final = rt.rms_norm(x, rt.param(self.output_norm), C.RMS_EPS);
            return rt.quant_matmul_no_grad(final, &self.token_embd);
        }

        /// QKV projection with optional LoRA (base + scaling * (h @ A) @ B).
        fn qkv_with_lora(
            self: *const Self,
            rt: *DiffMpsRuntime,
            h: DiffMpsTensor,
            base_w: *const QW,
            lora_a: ParamHandle,
            lora_b: ParamHandle,
        ) DiffMpsTensor {
            _ = self;
            const base = rt.quant_matmul_no_grad(h, base_w);
            const lora = rt.mul_scalar(
                rt.matmul(rt.matmul(h, rt.param(lora_a)), rt.param(lora_b)),
                SCALING,
            );
            return rt.add(base, lora);
        }

        /// Per-head RMSNorm + RoPE for Q and K.
        fn qk_norm_rope(
            self: *const Self,
            rt: *DiffMpsRuntime,
            q: DiffMpsTensor,
            k: DiffMpsTensor,
            i: usize,
            seq_len: usize,
            rope_freqs: DiffMpsTensor,
        ) struct { q: DiffMpsTensor, k: DiffMpsTensor } {
            const q_flat = rt.reshape(q, &.{ seq_len * HEAD, HEAD_DIM });
            const q_normed = rt.rms_norm(q_flat, rt.param(self.q_norm[i]), C.RMS_EPS);
            const k_flat = rt.reshape(k, &.{ seq_len * HEAD_KV, HEAD_DIM });
            const k_normed = rt.rms_norm(k_flat, rt.param(self.k_norm[i]), C.RMS_EPS);

            const q_rope_in = rt.reshape(q_normed, &.{ seq_len, HEAD, HEAD_DIM });
            const q_roped = rt.rope(q_rope_in, rope_freqs, HEAD, @intCast(seq_len), HALF_DIM);
            const k_rope_in = rt.reshape(k_normed, &.{ seq_len, HEAD_KV, HEAD_DIM });
            const k_roped = rt.rope(k_rope_in, rope_freqs, HEAD_KV, @intCast(seq_len), HALF_DIM);
            return .{ .q = q_roped, .k = k_roped };
        }

        /// Causal attention: softmax(Q @ K^T / √d) @ V → (seq, Q_DIM)
        fn attention_qkv(
            self: *const Self,
            rt: *DiffMpsRuntime,
            q: DiffMpsTensor,
            k: DiffMpsTensor,
            v: DiffMpsTensor,
            seq_len: usize,
        ) DiffMpsTensor {
            _ = self;
            const q_2d = rt.reshape(q, &.{ seq_len * HEAD, HEAD_DIM });
            const k_2d = rt.reshape(k, &.{ seq_len * HEAD_KV, HEAD_DIM });
            const k_t = rt.transpose(k_2d, 0, 1);
            const scores = rt.matmul(q_2d, k_t);
            const scaled = rt.mul_scalar(scores, INV_SQRT_HEAD_DIM);
            const probs = rt.causal_softmax(scaled, HEAD, @intCast(seq_len));

            const v_2d = rt.reshape(v, &.{ seq_len * HEAD_KV, HEAD_DIM });
            const attn_out = rt.matmul(probs, v_2d);
            return rt.reshape(attn_out, &.{ seq_len, Q_DIM });
        }

        /// SwiGLU FFN: gate(x) * up(x) → down; with post-norm + residual outside.
        fn ffn_swi_glu(
            self: *const Self,
            rt: *DiffMpsRuntime,
            ff_h: DiffMpsTensor,
            i: usize,
        ) DiffMpsTensor {
            const gate = rt.quant_matmul_no_grad(ff_h, &self.layers[i].gate);
            const gate_act = rt.gelu(gate);
            const up = rt.quant_matmul_no_grad(ff_h, &self.layers[i].up);
            const ffn_inner = rt.mul(gate_act, up);
            return rt.quant_matmul_no_grad(ffn_inner, &self.layers[i].down);
        }

        /// Single transformer block (Gemma3 style: pre-norm attention + pre-norm FFN,
        /// with additional post-attention / post-ffw RMSNorm before the residual).
        fn block(
            self: *const Self,
            rt: *DiffMpsRuntime,
            x_in: DiffMpsTensor,
            i: usize,
            seq_len: usize,
        ) DiffMpsTensor {
            const rope_freqs = rt.make_node(self.rope_freqs_buf, &.{HALF_DIM}, false);

            // ── Pre-attention RMSNorm ──
            const h = rt.rms_norm(x_in, rt.param(self.attn_norm[i]), C.RMS_EPS);

            // ── QKV projections (Q/V have LoRA; K is frozen base only) ──
            const q = self.qkv_with_lora(
                rt,
                h,
                &self.layers[i].q,
                self.lora_q_a[i],
                self.lora_q_b[i],
            );
            const k = rt.quant_matmul_no_grad(h, &self.layers[i].k);
            const v = self.qkv_with_lora(
                rt,
                h,
                &self.layers[i].v,
                self.lora_v_a[i],
                self.lora_v_b[i],
            );

            // ── Per-head QK RMSNorm + RoPE ──
            const qk = self.qk_norm_rope(rt, q, k, i, seq_len, rope_freqs);

            // ── Attention output: softmax(Q @ K^T) @ V ──
            const attn_reshape = self.attention_qkv(rt, qk.q, qk.k, v, seq_len);

            // Output projection + post-attention RMSNorm + residual.
            const proj = rt.quant_matmul_no_grad(attn_reshape, &self.layers[i].o);
            const proj_n = rt.rms_norm(proj, rt.param(self.post_attn_norm[i]), C.RMS_EPS);
            const post_attn = rt.add(x_in, proj_n);

            // ── FFN: pre-norm → SwiGLU ──
            const ff_h = rt.rms_norm(post_attn, rt.param(self.ffn_norm[i]), C.RMS_EPS);
            const down = self.ffn_swi_glu(rt, ff_h, i);

            // Post-FFN RMSNorm + residual.
            const down_n = rt.rms_norm(down, rt.param(self.post_ffw_norm[i]), C.RMS_EPS);
            return rt.add(post_attn, down_n);
        }

        /// Return indices of the LoRA-only parameter handles
        /// (useful when you want to train only LoRA adapters).
        pub fn lora_param_indices(self: *const Self, allocator: Allocator) ![]usize {
            const n = LAYER * 4;
            const out = try allocator.alloc(usize, n);
            var k: usize = 0;
            for (0..LAYER) |i| {
                out[k] = self.lora_q_a[i].index;
                k += 1;
                out[k] = self.lora_q_b[i].index;
                k += 1;
                out[k] = self.lora_v_a[i].index;
                k += 1;
                out[k] = self.lora_v_b[i].index;
                k += 1;
            }
            return out;
        }
    };
}
