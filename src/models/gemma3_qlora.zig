/// Gemma3 QLoRA Fine-tuning
///
/// 凍結量子化ベースモデル + f32 LoRA adapters (Q/V projection) で fine-tuning。
/// Metal GPU で forward/backward を実行。
const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf_mod = @import("../gguf/gguf.zig");
const gemma3_mod = @import("gemma3.zig");
const gemma3_metal_mod = @import("gemma3_metal.zig");
const metal_mod = @import("../backend/metal.zig");
const gpt2_mod = @import("gpt2.zig");
const gpu_var_mod = @import("../autograd/gpu_variable.zig");
const gpu_ops = @import("../autograd/gpu_ops.zig");
const gpu_opt = @import("../autograd/gpu_optimizer.zig");
const graph_mod = @import("../core/graph.zig");
const engine_mod = @import("../autograd/engine.zig");

const MetalContext = metal_mod.MetalContext;
const id = metal_mod.id;
const QuantType = metal_mod.QuantType;
pub const Gemma3_1B = gemma3_mod.Gemma3_1B;
const QuantizedWeight = gpt2_mod.QuantizedWeight;

fn GpuVariable(comptime T: type, comptime shape: anytype) type {
    return gpu_var_mod.GpuVariable(T, shape);
}

fn GpuResult(comptime T: type, comptime n: usize) type {
    return gpu_ops.GpuResult(T, n);
}

const GpuAdam = gpu_opt.GpuAdam;

// ============================================================
// QLoRA Model
// ============================================================

pub fn Gemma3QLoRA(comptime C: type, comptime RANK: usize) type {
    const EMBED = C.EMBED;
    const Q_DIM = C.Q_DIM;
    const KV_DIM = C.KV_DIM;
    const LAYER = C.LAYER;
    const HEAD = C.HEAD;
    const HEAD_DIM = C.HEAD_DIM;
    const FFN_DIM = C.FFN_DIM;
    const SCALING: f32 = 16.0 / @as(f32, @floatFromInt(RANK)); // alpha=16, rank=RANK

    return struct {
        const Self = @This();

        mtl: *MetalContext,

        // 凍結重みバッファ (GGUF quantized → Metal buffer)
        token_embd_buf: id,
        layer_bufs: [LAYER]LayerWeightBufs,

        // 量子化タイプ (レイヤーごと)
        q_quant_types: [LAYER]QuantType,
        k_quant_types: [LAYER]QuantType,
        v_quant_types: [LAYER]QuantType,
        o_quant_types: [LAYER]QuantType,
        gate_quant_types: [LAYER]QuantType,
        up_quant_types: [LAYER]QuantType,
        down_quant_types: [LAYER]QuantType,

        // LoRA adapters (trainable, f32)
        lora_q_a: [LAYER]GpuVariable(f32, .{ EMBED, RANK }),
        lora_q_b: [LAYER]GpuVariable(f32, .{ RANK, Q_DIM }),
        lora_v_a: [LAYER]GpuVariable(f32, .{ EMBED, RANK }),
        lora_v_b: [LAYER]GpuVariable(f32, .{ RANK, KV_DIM }),

        // RMSNorm weights (trainable)
        attn_norm_vars: [LAYER]GpuVariable(f32, .{EMBED}),
        ffn_norm_vars: [LAYER]GpuVariable(f32, .{EMBED}),
        output_norm_var: GpuVariable(f32, .{EMBED}),

        // Per-head Q/K norm weights (frozen, but autograd nodes for gradient flow)
        q_norm_vars: [LAYER]GpuVariable(f32, .{HEAD_DIM}),
        k_norm_vars: [LAYER]GpuVariable(f32, .{HEAD_DIM}),

        // Post-attention/FFN norm weights (frozen)
        post_attn_norm_vars: [LAYER]GpuVariable(f32, .{EMBED}),
        post_ffw_norm_vars: [LAYER]GpuVariable(f32, .{EMBED}),

        // RoPE freqs
        rope_freqs_buf: id,

        allocator: Allocator,

        const LayerWeightBufs = struct {
            attn_q_weight: id,
            attn_k_weight: id,
            attn_v_weight: id,
            attn_output_weight: id,
            ffn_gate_weight: id,
            ffn_up_weight: id,
            ffn_down_weight: id,
        };

        pub fn init(gguf_file: *const gguf_mod.GGUFFile, mtl: *MetalContext, allocator: Allocator) !Self {
            var self: Self = undefined;
            self.mtl = mtl;
            self.allocator = allocator;

            // Load CPU weights
            const weights = try gemma3_mod.Gemma3Weights(C).loadFromGGUF(gguf_file, allocator);

            // Token embedding (Q8_0, frozen)
            self.token_embd_buf = try mtl.createBufferWithData(weights.token_embd.data);

            // Output norm weight (trainable)
            self.output_norm_var = try GpuVariable(f32, .{EMBED}).fromSlice(mtl, allocator, weights.output_norm_weight, true);
            try self.output_norm_var.allocGrad();

            // RoPE freqs
            var rope_freqs = gemma3_mod.computeRoPEFreqs(C.HEAD_DIM, C.ROPE_BASE);
            self.rope_freqs_buf = try mtl.createBufferWithData(std.mem.sliceAsBytes(&rope_freqs));

            var rng = std.Random.DefaultPrng.init(42);
            const random = rng.random();

            for (0..LAYER) |layer| {
                const blk = &weights.blocks[layer];

                // Frozen quantized weights → Metal buffers
                self.layer_bufs[layer] = .{
                    .attn_q_weight = try mtl.createBufferWithData(blk.attn_q_weight.data),
                    .attn_k_weight = try mtl.createBufferWithData(blk.attn_k_weight.data),
                    .attn_v_weight = try mtl.createBufferWithData(blk.attn_v_weight.data),
                    .attn_output_weight = try mtl.createBufferWithData(blk.attn_output_weight.data),
                    .ffn_gate_weight = try mtl.createBufferWithData(blk.ffn_gate_weight.data),
                    .ffn_up_weight = try mtl.createBufferWithData(blk.ffn_up_weight.data),
                    .ffn_down_weight = try mtl.createBufferWithData(blk.ffn_down_weight.data),
                };

                self.q_quant_types[layer] = gemma3_mod.quantTypeOfWeight(blk.attn_q_weight);
                self.k_quant_types[layer] = gemma3_mod.quantTypeOfWeight(blk.attn_k_weight);
                self.v_quant_types[layer] = gemma3_mod.quantTypeOfWeight(blk.attn_v_weight);
                self.o_quant_types[layer] = gemma3_mod.quantTypeOfWeight(blk.attn_output_weight);
                self.gate_quant_types[layer] = gemma3_mod.quantTypeOfWeight(blk.ffn_gate_weight);
                self.up_quant_types[layer] = gemma3_mod.quantTypeOfWeight(blk.ffn_up_weight);
                self.down_quant_types[layer] = gemma3_mod.quantTypeOfWeight(blk.ffn_down_weight);

                // LoRA adapters: A = small random, B = zero (standard init)
                self.lora_q_a[layer] = try GpuVariable(f32, .{ EMBED, RANK }).xavierInit(mtl, allocator, EMBED, RANK, random);
                try self.lora_q_a[layer].allocGrad();
                self.lora_q_b[layer] = try GpuVariable(f32, .{ RANK, Q_DIM }).init(mtl, allocator, true);
                try self.lora_q_b[layer].allocGrad();

                self.lora_v_a[layer] = try GpuVariable(f32, .{ EMBED, RANK }).xavierInit(mtl, allocator, EMBED, RANK, random);
                try self.lora_v_a[layer].allocGrad();
                self.lora_v_b[layer] = try GpuVariable(f32, .{ RANK, KV_DIM }).init(mtl, allocator, true);
                try self.lora_v_b[layer].allocGrad();

                // RMSNorm weights (trainable)
                self.attn_norm_vars[layer] = try GpuVariable(f32, .{EMBED}).fromSlice(mtl, allocator, blk.attn_norm_weight, true);
                try self.attn_norm_vars[layer].allocGrad();
                self.ffn_norm_vars[layer] = try GpuVariable(f32, .{EMBED}).fromSlice(mtl, allocator, blk.ffn_norm_weight, true);
                try self.ffn_norm_vars[layer].allocGrad();

                // Per-head Q/K norm weights (frozen, needs autograd nodes)
                self.q_norm_vars[layer] = try GpuVariable(f32, .{HEAD_DIM}).fromSlice(mtl, allocator, blk.attn_q_norm_weight, true);
                try self.q_norm_vars[layer].allocGrad();
                self.k_norm_vars[layer] = try GpuVariable(f32, .{HEAD_DIM}).fromSlice(mtl, allocator, blk.attn_k_norm_weight, true);
                try self.k_norm_vars[layer].allocGrad();

                // Post-attention/FFN norm weights (frozen)
                self.post_attn_norm_vars[layer] = try GpuVariable(f32, .{EMBED}).fromSlice(mtl, allocator, blk.post_attention_norm_weight, true);
                try self.post_attn_norm_vars[layer].allocGrad();
                self.post_ffw_norm_vars[layer] = try GpuVariable(f32, .{EMBED}).fromSlice(mtl, allocator, blk.post_ffw_norm_weight, true);
                try self.post_ffw_norm_vars[layer].allocGrad();
            }

            // Free CPU weights
            var w = weights;
            w.deinit();

            return self;
        }

        pub fn deinit(self: *Self) void {
            for (0..LAYER) |i| {
                self.lora_q_a[i].deinit();
                self.lora_q_b[i].deinit();
                self.lora_v_a[i].deinit();
                self.lora_v_b[i].deinit();
                self.attn_norm_vars[i].deinit();
                self.ffn_norm_vars[i].deinit();
                self.q_norm_vars[i].deinit();
                self.k_norm_vars[i].deinit();
                self.post_attn_norm_vars[i].deinit();
                self.post_ffw_norm_vars[i].deinit();
            }
            self.output_norm_var.deinit();
        }

        /// Forward pass for training: returns logits GpuResult
        pub fn forward(
            self: *Self,
            comptime seq_len: usize,
            input_ids: []const u32,
            arena: Allocator,
        ) !GpuResult(f32, seq_len * C.VOCAB) {
            const mtl = self.mtl;
            const Node = graph_mod.GraphNode(f32);

            // 1. Embedding: dequant Q8_0 + scale
            const embed_scale = @sqrt(@as(f32, @floatFromInt(EMBED)));
            const token_ids_buf = try mtl.createBuffer(seq_len * @sizeOf(u32));
            const token_ids_ptr = MetalContext.bufferContents(u32, token_ids_buf);
            @memcpy(token_ids_ptr[0..seq_len], input_ids[0..seq_len]);

            const emb_buf = try mtl.createBuffer(seq_len * EMBED * @sizeOf(f32));
            gpuExec(mtl, MetalContext.dispatchDequantQ8BatchScaled, .{
                self.token_embd_buf, token_ids_buf, emb_buf,
                @as(u32, seq_len), @as(u32, EMBED), embed_scale,
            });

            const emb_node = try arena.create(Node);
            emb_node.* = Node.init(seq_len * EMBED, true);

            var x_buf = emb_buf;
            var x_node: *Node = emb_node;

            // 2. Transformer layers
            for (0..LAYER) |layer| {
                const lb = &self.layer_bufs[layer];

                // Pre-attention RMSNorm
                const norm_result = try gpu_ops.rmsNorm(
                    f32, seq_len, EMBED,
                    undefined, x_buf, x_node,
                    undefined, self.attn_norm_vars[layer].data_buf, self.attn_norm_vars[layer].node,
                    C.RMS_EPS, mtl, arena,
                );

                // Q projection: base + LoRA (no grad for base: LoRA gets grad via residual)
                const q_base = try gpu_ops.quantizedMatmulNoGrad(
                    f32, seq_len, EMBED, Q_DIM,
                    undefined, norm_result.data_buf, norm_result.node,
                    lb.attn_q_weight, self.q_quant_types[layer],
                    mtl, arena,
                );

                // LoRA Q: h @ A @ B * scaling
                const lora_q_h = try gpu_ops.matmul(
                    f32, seq_len, EMBED, RANK,
                    undefined, norm_result.data_buf, norm_result.node,
                    undefined, self.lora_q_a[layer].data_buf, self.lora_q_a[layer].node,
                    mtl, arena,
                );
                const lora_q_out = try gpu_ops.matmul(
                    f32, seq_len, RANK, Q_DIM,
                    undefined, lora_q_h.data_buf, lora_q_h.node,
                    undefined, self.lora_q_b[layer].data_buf, self.lora_q_b[layer].node,
                    mtl, arena,
                );
                const lora_q_scaled = try gpu_ops.scale(
                    f32, seq_len * Q_DIM,
                    undefined, lora_q_out.data_buf, lora_q_out.node,
                    SCALING, mtl, arena,
                );
                const q_combined = try gpu_ops.add(
                    f32, seq_len * Q_DIM,
                    undefined, q_base.data_buf, q_base.node,
                    undefined, lora_q_scaled.data_buf, lora_q_scaled.node,
                    mtl, arena,
                );

                // Per-head Q norm: treat (seq_len, Q_DIM) as (seq_len*HEAD, HEAD_DIM)
                const q_normed = try gpu_ops.rmsNorm(
                    f32, seq_len * HEAD, HEAD_DIM,
                    undefined, q_combined.data_buf, q_combined.node,
                    undefined, self.q_norm_vars[layer].data_buf, self.q_norm_vars[layer].node,
                    C.RMS_EPS, mtl, arena,
                );

                // K projection: base only (no LoRA, no grad propagation)
                const k_result = try gpu_ops.quantizedMatmulNoGrad(
                    f32, seq_len, EMBED, KV_DIM,
                    undefined, norm_result.data_buf, norm_result.node,
                    lb.attn_k_weight, self.k_quant_types[layer],
                    mtl, arena,
                );

                // Per-head K norm: (seq_len, KV_DIM=HEAD_DIM) → rmsNorm(seq_len, HEAD_DIM)
                const k_normed = try gpu_ops.rmsNorm(
                    f32, seq_len, HEAD_DIM,
                    undefined, k_result.data_buf, k_result.node,
                    undefined, self.k_norm_vars[layer].data_buf, self.k_norm_vars[layer].node,
                    C.RMS_EPS, mtl, arena,
                );

                // V projection: base + LoRA (no grad for base)
                const v_base = try gpu_ops.quantizedMatmulNoGrad(
                    f32, seq_len, EMBED, KV_DIM,
                    undefined, norm_result.data_buf, norm_result.node,
                    lb.attn_v_weight, self.v_quant_types[layer],
                    mtl, arena,
                );

                const lora_v_h = try gpu_ops.matmul(
                    f32, seq_len, EMBED, RANK,
                    undefined, norm_result.data_buf, norm_result.node,
                    undefined, self.lora_v_a[layer].data_buf, self.lora_v_a[layer].node,
                    mtl, arena,
                );
                const lora_v_out = try gpu_ops.matmul(
                    f32, seq_len, RANK, KV_DIM,
                    undefined, lora_v_h.data_buf, lora_v_h.node,
                    undefined, self.lora_v_b[layer].data_buf, self.lora_v_b[layer].node,
                    mtl, arena,
                );
                const lora_v_scaled = try gpu_ops.scale(
                    f32, seq_len * KV_DIM,
                    undefined, lora_v_out.data_buf, lora_v_out.node,
                    SCALING, mtl, arena,
                );
                const v_result = try gpu_ops.add(
                    f32, seq_len * KV_DIM,
                    undefined, v_base.data_buf, v_base.node,
                    undefined, lora_v_scaled.data_buf, lora_v_scaled.node,
                    mtl, arena,
                );

                // RoPE on Q and K (after per-head norms)
                const q_roped = try gpu_ops.rope(
                    f32, seq_len, HEAD, HEAD_DIM / 2,
                    q_normed.data_buf, q_normed.node,
                    self.rope_freqs_buf, mtl, arena,
                );

                const k_roped = try gpu_ops.rope(
                    f32, seq_len, 1, HEAD_DIM / 2,
                    k_normed.data_buf, k_normed.node,
                    self.rope_freqs_buf, mtl, arena,
                );

                // GQA Attention: Q(seq*HEAD, HEAD_DIM) @ K^T(HEAD_DIM, seq) → (seq*HEAD, seq)
                const attn_scores = try gpu_ops.matmulTransB(
                    f32, seq_len * HEAD, HEAD_DIM, seq_len,
                    undefined, q_roped.data_buf, q_roped.node,
                    undefined, k_roped.data_buf, k_roped.node,
                    mtl, arena,
                );

                // Scale + causal softmax
                const attn_scaled = try gpu_ops.scale(
                    f32, seq_len * HEAD * seq_len,
                    undefined, attn_scores.data_buf, attn_scores.node,
                    1.0 / @sqrt(@as(f32, @floatFromInt(HEAD_DIM))),
                    mtl, arena,
                );

                const attn_probs = try gpu_ops.causalSoftmax(
                    f32, seq_len * HEAD, seq_len, HEAD, seq_len,
                    undefined, attn_scaled.data_buf, attn_scaled.node,
                    mtl, arena,
                );

                // attn_out = probs @ V: (seq*HEAD, seq) @ (seq, HEAD_DIM) → (seq*HEAD, HEAD_DIM)
                const attn_out = try gpu_ops.matmul(
                    f32, seq_len * HEAD, seq_len, HEAD_DIM,
                    undefined, attn_probs.data_buf, attn_probs.node,
                    undefined, v_result.data_buf, v_result.node,
                    mtl, arena,
                );

                // Output projection (frozen)
                const proj_out = try gpu_ops.quantizedMatmul(
                    f32, seq_len, Q_DIM, EMBED,
                    undefined, attn_out.data_buf, attn_out.node,
                    lb.attn_output_weight, self.o_quant_types[layer],
                    mtl, arena,
                );

                // Post-attention norm + residual: x = x + rmsNorm(proj_out)
                const proj_normed = try gpu_ops.rmsNorm(
                    f32, seq_len, EMBED,
                    undefined, proj_out.data_buf, proj_out.node,
                    undefined, self.post_attn_norm_vars[layer].data_buf, self.post_attn_norm_vars[layer].node,
                    C.RMS_EPS, mtl, arena,
                );
                const post_attn = try gpu_ops.add(
                    f32, seq_len * EMBED,
                    undefined, x_buf, x_node,
                    undefined, proj_normed.data_buf, proj_normed.node,
                    mtl, arena,
                );

                // Pre-FFN RMSNorm
                const ffn_norm = try gpu_ops.rmsNorm(
                    f32, seq_len, EMBED,
                    undefined, post_attn.data_buf, post_attn.node,
                    undefined, self.ffn_norm_vars[layer].data_buf, self.ffn_norm_vars[layer].node,
                    C.RMS_EPS, mtl, arena,
                );

                // MLP: gate=GELU(h@W_gate), up=h@W_up, ffn_h=gate*up, out=ffn_h@W_down
                // No grad propagation through MLP (no LoRA in MLP, prevents gradient explosion)
                const gate_out = try gpu_ops.quantizedMatmulNoGrad(
                    f32, seq_len, EMBED, FFN_DIM,
                    undefined, ffn_norm.data_buf, ffn_norm.node,
                    lb.ffn_gate_weight, self.gate_quant_types[layer],
                    mtl, arena,
                );
                const gate_act = try gpu_ops.gelu(
                    f32, seq_len * FFN_DIM,
                    undefined, gate_out.data_buf, gate_out.node,
                    mtl, arena,
                );

                const up_out = try gpu_ops.quantizedMatmulNoGrad(
                    f32, seq_len, EMBED, FFN_DIM,
                    undefined, ffn_norm.data_buf, ffn_norm.node,
                    lb.ffn_up_weight, self.up_quant_types[layer],
                    mtl, arena,
                );

                // Element-wise multiply: gate * up (in-place)
                gpuExec(mtl, MetalContext.dispatchMul, .{
                    gate_act.data_buf, up_out.data_buf, @as(u32, seq_len * FFN_DIM),
                });

                const down_out = try gpu_ops.quantizedMatmulNoGrad(
                    f32, seq_len, FFN_DIM, EMBED,
                    undefined, gate_act.data_buf, gate_act.node,
                    lb.ffn_down_weight, self.down_quant_types[layer],
                    mtl, arena,
                );

                // Post-FFN norm + residual: x = post_attn + rmsNorm(down_out)
                const ffn_normed = try gpu_ops.rmsNorm(
                    f32, seq_len, EMBED,
                    undefined, down_out.data_buf, down_out.node,
                    undefined, self.post_ffw_norm_vars[layer].data_buf, self.post_ffw_norm_vars[layer].node,
                    C.RMS_EPS, mtl, arena,
                );
                const post_ffn = try gpu_ops.add(
                    f32, seq_len * EMBED,
                    undefined, post_attn.data_buf, post_attn.node,
                    undefined, ffn_normed.data_buf, ffn_normed.node,
                    mtl, arena,
                );

                x_buf = post_ffn.data_buf;
                x_node = post_ffn.node;
            }

            // Final RMSNorm
            const final_norm = try gpu_ops.rmsNorm(
                f32, seq_len, EMBED,
                undefined, x_buf, x_node,
                undefined, self.output_norm_var.data_buf, self.output_norm_var.node,
                C.RMS_EPS, mtl, arena,
            );

            // Logits: h @ embd^T (weight-tied, Q8_0)
            const logits = try gpu_ops.quantizedMatmul(
                f32, seq_len, EMBED, C.VOCAB,
                undefined, final_norm.data_buf, final_norm.node,
                self.token_embd_buf, .q8_0,
                mtl, arena,
            );

            return logits;
        }

        /// Get all trainable LoRA parameter descriptors for optimizer
        pub fn getLoRAParams(self: *Self, allocator: Allocator) ![]GpuAdam(f32).GpuParam {
            const n_params = LAYER * 4; // q_a, q_b, v_a, v_b per layer
            var params = try allocator.alloc(GpuAdam(f32).GpuParam, n_params);

            for (0..LAYER) |i| {
                params[i * 4 + 0] = .{
                    .data = self.lora_q_a[i].dataSlice(),
                    .data_buf = self.lora_q_a[i].data_buf,
                    .grad = &self.lora_q_a[i].node.grad,
                    .grad_buf = self.lora_q_a[i].grad_buf,
                    .count = EMBED * RANK,
                };
                params[i * 4 + 1] = .{
                    .data = self.lora_q_b[i].dataSlice(),
                    .data_buf = self.lora_q_b[i].data_buf,
                    .grad = &self.lora_q_b[i].node.grad,
                    .grad_buf = self.lora_q_b[i].grad_buf,
                    .count = RANK * Q_DIM,
                };
                params[i * 4 + 2] = .{
                    .data = self.lora_v_a[i].dataSlice(),
                    .data_buf = self.lora_v_a[i].data_buf,
                    .grad = &self.lora_v_a[i].node.grad,
                    .grad_buf = self.lora_v_a[i].grad_buf,
                    .count = EMBED * RANK,
                };
                params[i * 4 + 3] = .{
                    .data = self.lora_v_b[i].dataSlice(),
                    .data_buf = self.lora_v_b[i].data_buf,
                    .grad = &self.lora_v_b[i].node.grad,
                    .grad_buf = self.lora_v_b[i].grad_buf,
                    .count = RANK * KV_DIM,
                };
            }

            return params;
        }

        pub fn zeroGrad(self: *Self) void {
            for (0..LAYER) |i| {
                self.lora_q_a[i].zeroGrad();
                self.lora_q_b[i].zeroGrad();
                self.lora_v_a[i].zeroGrad();
                self.lora_v_b[i].zeroGrad();
                self.attn_norm_vars[i].zeroGrad();
                self.ffn_norm_vars[i].zeroGrad();
                self.q_norm_vars[i].zeroGrad();
                self.k_norm_vars[i].zeroGrad();
                self.post_attn_norm_vars[i].zeroGrad();
                self.post_ffw_norm_vars[i].zeroGrad();
            }
            self.output_norm_var.zeroGrad();
        }
    };
}

// ============================================================
// Helper functions
// ============================================================

fn gpuExec(mtl: *MetalContext, dispatch_fn: anytype, args: anytype) void {
    if (mtl.batch_encoder) |encoder| {
        @call(.auto, dispatch_fn, .{mtl, encoder} ++ args);
        MetalContext.memoryBarrier(encoder);
    } else {
        const cmd_buf = mtl.newCommandBuffer();
        const encoder = MetalContext.newComputeEncoder(cmd_buf);
        @call(.auto, dispatch_fn, .{mtl, encoder} ++ args);
        MetalContext.memoryBarrier(encoder);
        MetalContext.endEncoding(encoder);
        MetalContext.commit(cmd_buf);
        MetalContext.waitUntilCompleted(cmd_buf);
    }
}
