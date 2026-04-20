const std = @import("std");
const Allocator = std.mem.Allocator;
const Timer = @import("../util/timer.zig").Timer;
const gguf_mod = @import("../gguf/gguf.zig");
const dequant_mod = @import("../gguf/dequant.zig");
const thread_pool_mod = @import("../gguf/thread_pool.zig");
const gpt2_mod = @import("gpt2.zig");
const log_mod = @import("../log.zig");
const log = log_mod.gemma3;

// ============================================================
// Gemma 3 1B Config
// ============================================================

pub const Gemma3_1B = struct {
    pub const VOCAB: usize = 262144;
    pub const EMBED: usize = 1152;
    pub const HEAD: usize = 4;
    pub const HEAD_KV: usize = 1;
    pub const HEAD_DIM: usize = 256;
    pub const LAYER: usize = 26;
    pub const CTX: usize = 2048;
    pub const FFN_DIM: usize = 6912;
    pub const ROPE_BASE: f32 = 1000000.0;
    pub const SLIDING_WINDOW: usize = 512;
    pub const RMS_EPS: f32 = 1e-6;
    pub const Q_DIM: usize = HEAD * HEAD_DIM; // 1024
    pub const KV_DIM: usize = HEAD_KV * HEAD_DIM; // 256
};

// ============================================================
// プロファイリング
// ============================================================

pub const ProfileStats = struct {
    embedding_ns: u64 = 0,
    rms_norm_ns: u64 = 0,
    linear_ns: u64 = 0,
    rope_ns: u64 = 0,
    attention_ns: u64 = 0,
    elementwise_ns: u64 = 0,
    logits_ns: u64 = 0,
    call_count: u64 = 0,

    pub fn reset(self: *ProfileStats) void {
        self.* = .{};
    }

    pub fn print(self: *const ProfileStats) void {
        const Entry = struct { name: []const u8, ns: u64 };
        var entries = [_]Entry{
            .{ .name = "linear", .ns = self.linear_ns },
            .{ .name = "logits", .ns = self.logits_ns },
            .{ .name = "attention", .ns = self.attention_ns },
            .{ .name = "rms_norm", .ns = self.rms_norm_ns },
            .{ .name = "rope", .ns = self.rope_ns },
            .{ .name = "elementwise", .ns = self.elementwise_ns },
            .{ .name = "embedding", .ns = self.embedding_ns },
        };

        // バブルソート (降順)
        for (0..entries.len) |i| {
            for (i + 1..entries.len) |j| {
                if (entries[j].ns > entries[i].ns) {
                    const tmp = entries[i];
                    entries[i] = entries[j];
                    entries[j] = tmp;
                }
            }
        }

        const total_ns = self.embedding_ns + self.rms_norm_ns + self.linear_ns +
            self.rope_ns + self.attention_ns + self.elementwise_ns + self.logits_ns;
        const total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;

        var maybe_artifact = log_mod.openProfileArtifact("gemma3") catch null;
        if (maybe_artifact) |*artifact| {
            defer artifact.close();
            const w = artifact.writer();
            w.print("=== Gemma3 Profile ({d} calls) ===\n", .{self.call_count}) catch return;
            for (&entries) |*e| {
                const ms = @as(f64, @floatFromInt(e.ns)) / 1_000_000.0;
                const pct = if (total_ns > 0) ms / total_ms * 100.0 else 0.0;
                w.print("  {s:<12}: {d:>8.1}ms  {d:>5.1}%\n", .{ e.name, ms, pct }) catch return;
            }
            w.print("  {s:->36}\n", .{""}) catch return;
            w.print("  {s:<12}: {d:>8.1}ms  100.0%\n", .{ "total", total_ms }) catch return;
            log.info("profile written: {s}", .{artifact.path});
        }
    }
};

// ============================================================
// Weight 構造体
// ============================================================

pub const QuantizedWeight = gpt2_mod.QuantizedWeight;

pub const GemmaBlockWeights = struct {
    attn_norm_weight: []f32, // (EMBED,)
    post_attention_norm_weight: []f32, // (EMBED,)
    attn_q_weight: QuantizedWeight, // (Q_DIM, EMBED) Q4_0
    attn_k_weight: QuantizedWeight, // (KV_DIM, EMBED) Q4_0
    attn_v_weight: QuantizedWeight, // (KV_DIM, EMBED) Q4_0
    attn_output_weight: QuantizedWeight, // (EMBED, Q_DIM) Q4_0
    attn_q_norm_weight: []f32, // (HEAD_DIM,)
    attn_k_norm_weight: []f32, // (HEAD_DIM,)
    ffn_norm_weight: []f32, // (EMBED,)
    post_ffw_norm_weight: []f32, // (EMBED,)
    ffn_gate_weight: QuantizedWeight, // (FFN_DIM, EMBED) Q4_0
    ffn_up_weight: QuantizedWeight, // (FFN_DIM, EMBED) Q4_0
    ffn_down_weight: QuantizedWeight, // (EMBED, FFN_DIM) Q4_1
};

pub fn Gemma3Weights(comptime C: type) type {
    return struct {
        const Self = @This();

        token_embd: QuantizedWeight, // (VOCAB, EMBED) Q8_0
        output_norm_weight: []f32, // (EMBED,)
        blocks: [C.LAYER]GemmaBlockWeights,
        allocator: Allocator,

        pub fn loadFromGGUF(gguf_file: *const gguf_mod.GGUFFile, allocator: Allocator) !Self {
            var self: Self = undefined;
            self.allocator = allocator;

            self.token_embd = try loadQuantizedWeight(gguf_file, "token_embd.weight");
            self.output_norm_weight = try gguf_file.loadTensorF32("output_norm.weight", allocator);

            for (0..C.LAYER) |i| {
                self.blocks[i] = try loadBlockWeights(C, gguf_file, allocator, i);
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.output_norm_weight);
            for (&self.blocks) |*blk| {
                self.allocator.free(blk.attn_norm_weight);
                self.allocator.free(blk.post_attention_norm_weight);
                self.allocator.free(blk.attn_q_norm_weight);
                self.allocator.free(blk.attn_k_norm_weight);
                self.allocator.free(blk.ffn_norm_weight);
                self.allocator.free(blk.post_ffw_norm_weight);
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
    comptime _: type,
    gguf_file: *const gguf_mod.GGUFFile,
    allocator: Allocator,
    layer_idx: usize,
) !GemmaBlockWeights {
    var name_buf: [64]u8 = undefined;

    return .{
        .attn_norm_weight = try gguf_file.loadTensorF32(try fmtBlockName(&name_buf, layer_idx, "attn_norm.weight"), allocator),
        .post_attention_norm_weight = try gguf_file.loadTensorF32(try fmtBlockName(&name_buf, layer_idx, "post_attention_norm.weight"), allocator),
        .attn_q_weight = try loadQuantizedWeight(gguf_file, try fmtBlockName(&name_buf, layer_idx, "attn_q.weight")),
        .attn_k_weight = try loadQuantizedWeight(gguf_file, try fmtBlockName(&name_buf, layer_idx, "attn_k.weight")),
        .attn_v_weight = try loadQuantizedWeight(gguf_file, try fmtBlockName(&name_buf, layer_idx, "attn_v.weight")),
        .attn_output_weight = try loadQuantizedWeight(gguf_file, try fmtBlockName(&name_buf, layer_idx, "attn_output.weight")),
        .attn_q_norm_weight = try gguf_file.loadTensorF32(try fmtBlockName(&name_buf, layer_idx, "attn_q_norm.weight"), allocator),
        .attn_k_norm_weight = try gguf_file.loadTensorF32(try fmtBlockName(&name_buf, layer_idx, "attn_k_norm.weight"), allocator),
        .ffn_norm_weight = try gguf_file.loadTensorF32(try fmtBlockName(&name_buf, layer_idx, "ffn_norm.weight"), allocator),
        .post_ffw_norm_weight = try gguf_file.loadTensorF32(try fmtBlockName(&name_buf, layer_idx, "post_ffw_norm.weight"), allocator),
        .ffn_gate_weight = try loadQuantizedWeight(gguf_file, try fmtBlockName(&name_buf, layer_idx, "ffn_gate.weight")),
        .ffn_up_weight = try loadQuantizedWeight(gguf_file, try fmtBlockName(&name_buf, layer_idx, "ffn_up.weight")),
        .ffn_down_weight = try loadQuantizedWeight(gguf_file, try fmtBlockName(&name_buf, layer_idx, "ffn_down.weight")),
    };
}

fn fmtBlockName(buf: []u8, layer: usize, suffix: []const u8) ![]const u8 {
    const result = std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer, suffix }) catch return error.NameTooLong;
    return result;
}

// ============================================================
// Gemma 3 推論エンジン
// ============================================================

pub fn Gemma3(comptime C: type) type {
    return struct {
        const Self = @This();

        weights: Gemma3Weights(C),
        kv_cache: KVCache,
        pool: *thread_pool_mod.ThreadPool,
        rope_freqs: [C.HEAD_DIM / 2]f32, // 事前計算した RoPE 周波数テーブル
        profile: ProfileStats,
        allocator: Allocator,

        const KVCache = struct {
            k: [C.LAYER][]f32, // 各 (CTX, KV_DIM)
            v: [C.LAYER][]f32, // 各 (CTX, KV_DIM)
            seq_len: usize,

            fn init(allocator: Allocator) !KVCache {
                var cache: KVCache = undefined;
                cache.seq_len = 0;
                for (0..C.LAYER) |i| {
                    cache.k[i] = try allocator.alloc(f32, C.CTX * C.KV_DIM);
                    cache.v[i] = try allocator.alloc(f32, C.CTX * C.KV_DIM);
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
            // CPU コア数を検出してスレッド数を決定
            const cpu_count = std.Thread.getCpuCount() catch 1;
            const n_threads = @max(1, @min(cpu_count, 8));

            const self: Self = .{
                .weights = try Gemma3Weights(C).loadFromGGUF(gguf_file, allocator),
                .kv_cache = try KVCache.init(allocator),
                .pool = try thread_pool_mod.ThreadPool.init(n_threads, allocator),
                .rope_freqs = computeRoPEFreqs(C.HEAD_DIM, C.ROPE_BASE),
                .profile = .{},
                .allocator = allocator,
            };
            log.info("ready: layers={d} embed={d} ctx={d} threads={d}", .{
                C.LAYERS, C.EMBED, C.CTX, n_threads,
            });
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.pool.deinit(self.allocator);
            self.kv_cache.deinit(self.allocator);
            self.weights.deinit();
        }

        pub fn resetCache(self: *Self) void {
            self.kv_cache.reset();
        }

        /// Prefill: プロンプト全体を処理して KV キャッシュを埋める
        pub fn prefill(self: *Self, tokens: []const u32, arena: Allocator) ![]f32 {
            const seq_len = tokens.len;
            if (seq_len == 0) return error.EmptySequence;
            if (seq_len > C.CTX) return error.SequenceTooLong;
            log.debug("prefill: tokens={d}", .{seq_len});

            self.kv_cache.reset();
            const w = &self.weights;
            const embed_scale = @sqrt(@as(f32, @floatFromInt(C.EMBED)));
            const p = &self.profile;

            var timer = Timer.start() catch unreachable;
            var t0 = timer.read();

            // Embedding: 各トークンを Q8_0 から逆量子化
            const x = try arena.alloc(f32, seq_len * C.EMBED);
            for (0..seq_len) |t| {
                dequantRow(w.token_embd, tokens[t], x[t * C.EMBED ..][0..C.EMBED]);
                for (0..C.EMBED) |i| {
                    x[t * C.EMBED + i] *= embed_scale;
                }
            }

            var t1 = timer.read();
            p.embedding_ns += t1 - t0;
            t0 = t1;

            // 中間バッファ
            const h = try arena.alloc(f32, seq_len * C.EMBED);
            const q_buf = try arena.alloc(f32, seq_len * C.Q_DIM);
            const k_buf = try arena.alloc(f32, seq_len * C.KV_DIM);
            const v_buf = try arena.alloc(f32, seq_len * C.KV_DIM);
            const attn_out = try arena.alloc(f32, seq_len * C.Q_DIM);
            const proj_out = try arena.alloc(f32, seq_len * C.EMBED);
            const gate_buf = try arena.alloc(f32, seq_len * C.FFN_DIM);
            const up_buf = try arena.alloc(f32, seq_len * C.FFN_DIM);
            const ffn_out = try arena.alloc(f32, seq_len * C.EMBED);
            const norm_buf = try arena.alloc(f32, seq_len * C.EMBED);

            for (0..C.LAYER) |layer| {
                const blk = &w.blocks[layer];

                // Pre-attention RMSNorm
                t0 = timer.read();
                rmsNormRows(x, blk.attn_norm_weight, h, seq_len, C.EMBED, C.RMS_EPS);
                t1 = timer.read();
                p.rms_norm_ns += t1 - t0;
                t0 = t1;

                // Q/K/V projections
                linearForwardQ(h, blk.attn_q_weight, q_buf, seq_len, self.pool);
                linearForwardQ(h, blk.attn_k_weight, k_buf, seq_len, self.pool);
                linearForwardQ(h, blk.attn_v_weight, v_buf, seq_len, self.pool);
                t1 = timer.read();
                p.linear_ns += t1 - t0;
                t0 = t1;

                // Per-head Q/K norms + RoPE
                for (0..seq_len) |t| {
                    for (0..C.HEAD) |hd| {
                        rmsNormInPlace(
                            q_buf[t * C.Q_DIM + hd * C.HEAD_DIM ..][0..C.HEAD_DIM],
                            blk.attn_q_norm_weight,
                            C.RMS_EPS,
                        );
                    }
                    rmsNormInPlace(
                        k_buf[t * C.KV_DIM ..][0..C.HEAD_DIM],
                        blk.attn_k_norm_weight,
                        C.RMS_EPS,
                    );

                    for (0..C.HEAD) |hd| {
                        applyRoPEPrecomputed(
                            q_buf[t * C.Q_DIM + hd * C.HEAD_DIM ..][0..C.HEAD_DIM],
                            t,
                            &self.rope_freqs,
                        );
                    }
                    applyRoPEPrecomputed(
                        k_buf[t * C.KV_DIM ..][0..C.HEAD_DIM],
                        t,
                        &self.rope_freqs,
                    );

                    @memcpy(self.kv_cache.k[layer][t * C.KV_DIM ..][0..C.KV_DIM], k_buf[t * C.KV_DIM ..][0..C.KV_DIM]);
                    @memcpy(self.kv_cache.v[layer][t * C.KV_DIM ..][0..C.KV_DIM], v_buf[t * C.KV_DIM ..][0..C.KV_DIM]);
                }
                t1 = timer.read();
                p.rope_ns += t1 - t0;
                t0 = t1;

                // GQA causal self-attention (prefill)
                const is_global = isGlobalLayer(layer);
                gqaCausalAttentionPrefill(
                    q_buf,
                    self.kv_cache.k[layer],
                    self.kv_cache.v[layer],
                    attn_out,
                    seq_len,
                    C.HEAD,
                    C.HEAD_DIM,
                    C.Q_DIM,
                    C.KV_DIM,
                    if (is_global) seq_len else C.SLIDING_WINDOW,
                    arena,
                );
                t1 = timer.read();
                p.attention_ns += t1 - t0;
                t0 = t1;

                // Output projection
                linearForwardQ(attn_out, blk.attn_output_weight, proj_out, seq_len, self.pool);
                t1 = timer.read();
                p.linear_ns += t1 - t0;
                t0 = t1;

                // Post-attention norm
                rmsNormRows(proj_out, blk.post_attention_norm_weight, norm_buf, seq_len, C.EMBED, C.RMS_EPS);
                t1 = timer.read();
                p.rms_norm_ns += t1 - t0;
                t0 = t1;

                // Residual add
                addInPlace(x, norm_buf, seq_len * C.EMBED);
                t1 = timer.read();
                p.elementwise_ns += t1 - t0;
                t0 = t1;

                // Pre-FFN RMSNorm
                rmsNormRows(x, blk.ffn_norm_weight, h, seq_len, C.EMBED, C.RMS_EPS);
                t1 = timer.read();
                p.rms_norm_ns += t1 - t0;
                t0 = t1;

                // GeGLU MLP
                linearForwardQ(h, blk.ffn_gate_weight, gate_buf, seq_len, self.pool);
                geluInPlace(gate_buf, seq_len * C.FFN_DIM);
                linearForwardQ(h, blk.ffn_up_weight, up_buf, seq_len, self.pool);
                mulInPlace(gate_buf, up_buf, seq_len * C.FFN_DIM);
                linearForwardQ(gate_buf, blk.ffn_down_weight, ffn_out, seq_len, self.pool);
                t1 = timer.read();
                p.linear_ns += t1 - t0;
                t0 = t1;

                // Post-FFN norm
                rmsNormRows(ffn_out, blk.post_ffw_norm_weight, norm_buf, seq_len, C.EMBED, C.RMS_EPS);
                t1 = timer.read();
                p.rms_norm_ns += t1 - t0;
                t0 = t1;

                // Residual add
                addInPlace(x, norm_buf, seq_len * C.EMBED);
                t1 = timer.read();
                p.elementwise_ns += t1 - t0;
                t0 = t1;
            }

            self.kv_cache.seq_len = seq_len;

            // Final RMSNorm
            rmsNormRows(x, w.output_norm_weight, h, seq_len, C.EMBED, C.RMS_EPS);
            t1 = timer.read();
            p.rms_norm_ns += t1 - t0;
            t0 = t1;

            // Logits: hidden @ token_embd^T (weight-tied, Q8_0)
            const result = try computeLogits(w, h[(seq_len - 1) * C.EMBED ..][0..C.EMBED], arena, self.pool);
            t1 = timer.read();
            p.logits_ns += t1 - t0;

            p.call_count += 1;
            return result;
        }

        /// DecodeNext: 1トークンのみ処理、KV キャッシュに追記
        pub fn decodeNext(self: *Self, token: u32, arena: Allocator) ![]f32 {
            const pos = self.kv_cache.seq_len;
            if (pos >= C.CTX) return error.ContextFull;
            const w = &self.weights;
            const embed_scale = @sqrt(@as(f32, @floatFromInt(C.EMBED)));
            const p = &self.profile;

            var timer = Timer.start() catch unreachable;
            var t0 = timer.read();

            // Embedding
            const x = try arena.alloc(f32, C.EMBED);
            dequantRow(w.token_embd, token, x);
            for (0..C.EMBED) |i| {
                x[i] *= embed_scale;
            }

            var t1 = timer.read();
            p.embedding_ns += t1 - t0;
            t0 = t1;

            // 1行分のバッファ
            const h = try arena.alloc(f32, C.EMBED);
            const q_buf = try arena.alloc(f32, C.Q_DIM);
            const k_buf = try arena.alloc(f32, C.KV_DIM);
            const v_buf = try arena.alloc(f32, C.KV_DIM);
            const attn_out = try arena.alloc(f32, C.Q_DIM);
            const proj_out = try arena.alloc(f32, C.EMBED);
            const gate_buf = try arena.alloc(f32, C.FFN_DIM);
            const up_buf = try arena.alloc(f32, C.FFN_DIM);
            const ffn_out = try arena.alloc(f32, C.EMBED);
            const norm_buf = try arena.alloc(f32, C.EMBED);

            for (0..C.LAYER) |layer| {
                const blk = &w.blocks[layer];

                // Pre-attention RMSNorm
                t0 = timer.read();
                rmsNormRows(x, blk.attn_norm_weight, h, 1, C.EMBED, C.RMS_EPS);
                t1 = timer.read();
                p.rms_norm_ns += t1 - t0;
                t0 = t1;

                // Q/K/V projections
                linearForwardQ(h, blk.attn_q_weight, q_buf, 1, self.pool);
                linearForwardQ(h, blk.attn_k_weight, k_buf, 1, self.pool);
                linearForwardQ(h, blk.attn_v_weight, v_buf, 1, self.pool);
                t1 = timer.read();
                p.linear_ns += t1 - t0;
                t0 = t1;

                // Per-head Q norm + RoPE
                for (0..C.HEAD) |hd| {
                    rmsNormInPlace(
                        q_buf[hd * C.HEAD_DIM ..][0..C.HEAD_DIM],
                        blk.attn_q_norm_weight,
                        C.RMS_EPS,
                    );
                    applyRoPEPrecomputed(
                        q_buf[hd * C.HEAD_DIM ..][0..C.HEAD_DIM],
                        pos,
                        &self.rope_freqs,
                    );
                }
                // K norm + RoPE
                rmsNormInPlace(k_buf[0..C.HEAD_DIM], blk.attn_k_norm_weight, C.RMS_EPS);
                applyRoPEPrecomputed(k_buf[0..C.HEAD_DIM], pos, &self.rope_freqs);

                // KV cache
                @memcpy(self.kv_cache.k[layer][pos * C.KV_DIM ..][0..C.KV_DIM], k_buf[0..C.KV_DIM]);
                @memcpy(self.kv_cache.v[layer][pos * C.KV_DIM ..][0..C.KV_DIM], v_buf[0..C.KV_DIM]);
                t1 = timer.read();
                p.rope_ns += t1 - t0;
                t0 = t1;

                // GQA cached attention
                const is_global = isGlobalLayer(layer);
                const kv_len = pos + 1;
                const window = if (is_global) kv_len else @min(kv_len, C.SLIDING_WINDOW);
                const kv_start = if (kv_len > window) kv_len - window else 0;

                const scores_buf = try arena.alloc(f32, window);

                gqaCachedAttention(
                    q_buf,
                    self.kv_cache.k[layer],
                    self.kv_cache.v[layer],
                    attn_out,
                    scores_buf,
                    kv_start,
                    kv_len,
                    C.HEAD,
                    C.HEAD_DIM,
                    C.Q_DIM,
                    C.KV_DIM,
                );
                t1 = timer.read();
                p.attention_ns += t1 - t0;
                t0 = t1;

                // Output projection
                linearForwardQ(attn_out, blk.attn_output_weight, proj_out, 1, self.pool);
                t1 = timer.read();
                p.linear_ns += t1 - t0;
                t0 = t1;

                // Post-attention norm
                rmsNormRows(proj_out, blk.post_attention_norm_weight, norm_buf, 1, C.EMBED, C.RMS_EPS);
                t1 = timer.read();
                p.rms_norm_ns += t1 - t0;
                t0 = t1;

                // Residual add
                addInPlace(x, norm_buf, C.EMBED);
                t1 = timer.read();
                p.elementwise_ns += t1 - t0;
                t0 = t1;

                // Pre-FFN RMSNorm
                rmsNormRows(x, blk.ffn_norm_weight, h, 1, C.EMBED, C.RMS_EPS);
                t1 = timer.read();
                p.rms_norm_ns += t1 - t0;
                t0 = t1;

                // GeGLU MLP
                linearForwardQ(h, blk.ffn_gate_weight, gate_buf, 1, self.pool);
                geluInPlace(gate_buf, C.FFN_DIM);
                linearForwardQ(h, blk.ffn_up_weight, up_buf, 1, self.pool);
                mulInPlace(gate_buf, up_buf, C.FFN_DIM);
                linearForwardQ(gate_buf, blk.ffn_down_weight, ffn_out, 1, self.pool);
                t1 = timer.read();
                p.linear_ns += t1 - t0;
                t0 = t1;

                // Post-FFN norm
                rmsNormRows(ffn_out, blk.post_ffw_norm_weight, norm_buf, 1, C.EMBED, C.RMS_EPS);
                t1 = timer.read();
                p.rms_norm_ns += t1 - t0;
                t0 = t1;

                // Residual add
                addInPlace(x, norm_buf, C.EMBED);
                t1 = timer.read();
                p.elementwise_ns += t1 - t0;
                t0 = t1;
            }

            // Final RMSNorm
            rmsNormRows(x, w.output_norm_weight, h, 1, C.EMBED, C.RMS_EPS);
            t1 = timer.read();
            p.rms_norm_ns += t1 - t0;
            t0 = t1;

            self.kv_cache.seq_len = pos + 1;

            const result = try computeLogits(w, h, arena, self.pool);
            t1 = timer.read();
            p.logits_ns += t1 - t0;

            p.call_count += 1;
            return result;
        }

        /// hidden → logits (weight-tied with token_embd, Q8_0 matmul)
        fn computeLogits(w: *const Gemma3Weights(C), hidden: []const f32, arena: Allocator, pool: *thread_pool_mod.ThreadPool) ![]f32 {
            const logits = try arena.alloc(f32, C.VOCAB);
            pool.matmul(
                w.token_embd.data,
                hidden,
                logits,
                C.VOCAB,
                C.EMBED,
                .q8_0,
            );
            return logits;
        }
    };
}

// ============================================================
// ヘルパー関数
// ============================================================

/// 量子化行から 1 行分を f32 に逆量子化
pub fn dequantRow(w: QuantizedWeight, row_idx: u32, dst: []f32) void {
    const in_dim = w.in_dim;
    const idx: usize = row_idx;
    switch (w.type_) {
        .q8_0 => {
            const row_bytes = dequant_mod.tensorBytes(.q8_0, in_dim);
            const src = w.data[idx * row_bytes ..][0..row_bytes];
            dequant_mod.dequantizeQ8_0(src, dst, in_dim);
        },
        .q4_0 => {
            const row_bytes = dequant_mod.tensorBytes(.q4_0, in_dim);
            const src = w.data[idx * row_bytes ..][0..row_bytes];
            dequant_mod.dequantizeQ4_0(src, dst, in_dim);
        },
        else => unreachable,
    }
}

/// Linear forward with quantized weights (no bias)
/// ThreadPool を使用したマルチスレッド matmul
fn linearForwardQ(
    input: []const f32,
    weight: QuantizedWeight,
    output: []f32,
    rows: usize,
    pool: *thread_pool_mod.ThreadPool,
) void {
    const quant_type: thread_pool_mod.QuantType = switch (weight.type_) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q8_0 => .q8_0,
        else => unreachable,
    };

    for (0..rows) |r| {
        const in_row = input[r * weight.in_dim ..][0..weight.in_dim];
        const out_row = output[r * weight.out_dim ..][0..weight.out_dim];

        pool.matmul(
            weight.data,
            in_row,
            out_row,
            weight.out_dim,
            weight.in_dim,
            quant_type,
        );
    }
}

/// RMSNorm: 各行を独立に正規化 (SIMD)
/// output = x / rms * weight, rms = sqrt(mean(x^2) + eps)
pub fn rmsNormRows(
    input: []const f32,
    weight: []const f32,
    output: []f32,
    rows: usize,
    dim: usize,
    eps: f32,
) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;

    for (0..rows) |r| {
        const row = input[r * dim ..][0..dim];
        const out = output[r * dim ..][0..dim];

        // Sum of squares (SIMD)
        var ss_acc: @Vector(vl, f32) = @splat(0);
        var i: usize = 0;
        while (i + vl <= dim) : (i += vl) {
            const v: @Vector(vl, f32) = row[i..][0..vl].*;
            ss_acc += v * v;
        }
        var ss: f32 = @reduce(.Add, ss_acc);
        while (i < dim) : (i += 1) {
            ss += row[i] * row[i];
        }

        const rms_inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dim)) + eps);
        const rms_v: @Vector(vl, f32) = @splat(rms_inv);

        // Normalize (SIMD)
        i = 0;
        while (i + vl <= dim) : (i += vl) {
            const v: @Vector(vl, f32) = row[i..][0..vl].*;
            const w: @Vector(vl, f32) = weight[i..][0..vl].*;
            out[i..][0..vl].* = v * rms_v * w;
        }
        while (i < dim) : (i += 1) {
            out[i] = row[i] * rms_inv * weight[i];
        }
    }
}

/// RMSNorm in-place (for per-head normalization)
fn rmsNormInPlace(x: []f32, weight: []const f32, eps: f32) void {
    const dim = x.len;
    var ss: f32 = 0;
    for (0..dim) |i| {
        ss += x[i] * x[i];
    }
    const rms_inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dim)) + eps);
    for (0..dim) |i| {
        x[i] = x[i] * rms_inv * weight[i];
    }
}

/// RoPE (Rotary Position Embedding) — 事前計算周波数テーブル使用版
fn applyRoPEPrecomputed(x: []f32, pos: usize, freqs: []const f32) void {
    const pos_f: f32 = @floatFromInt(pos);
    for (0..freqs.len) |i| {
        const theta = pos_f * freqs[i];
        const cos_t = @cos(theta);
        const sin_t = @sin(theta);
        const x0 = x[i * 2];
        const x1 = x[i * 2 + 1];
        x[i * 2] = x0 * cos_t - x1 * sin_t;
        x[i * 2 + 1] = x0 * sin_t + x1 * cos_t;
    }
}

/// RoPE (Rotary Position Embedding) — フォールバック版 (テスト用)
fn applyRoPE(x: []f32, pos: usize, head_dim: usize, rope_base: f32) void {
    const pos_f: f32 = @floatFromInt(pos);
    var i: usize = 0;
    while (i < head_dim) : (i += 2) {
        const freq_exp = -@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim));
        const theta = pos_f * std.math.pow(f32, rope_base, freq_exp);
        const cos_t = @cos(theta);
        const sin_t = @sin(theta);
        const x0 = x[i];
        const x1 = x[i + 1];
        x[i] = x0 * cos_t - x1 * sin_t;
        x[i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

/// GQA causal self-attention (prefill, multi-token)
fn gqaCausalAttentionPrefill(
    q: []const f32, // (seq_len, Q_DIM)
    k_cache: []const f32, // (CTX, KV_DIM)
    v_cache: []const f32, // (CTX, KV_DIM)
    output: []f32, // (seq_len, Q_DIM)
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    window: usize,
    arena: Allocator,
) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output[0 .. seq_len * q_dim], 0);

    // Allocate scores buffer
    const scores_buf = arena.alloc(f32, seq_len) catch unreachable;

    for (0..n_head) |hd| {
        const q_off = hd * head_dim;
        // GQA: all Q heads share the same KV head (HEAD_KV=1)
        const kv_off: usize = 0;

        for (0..seq_len) |qi| {
            // Determine attention range
            const kv_start = if (qi + 1 > window) qi + 1 - window else 0;
            const kv_end = qi + 1;
            const kv_len = kv_end - kv_start;

            // Compute attention scores
            for (kv_start..kv_end) |ki| {
                var acc: @Vector(vl, f32) = @splat(0);
                var d: usize = 0;
                while (d + vl <= head_dim) : (d += vl) {
                    const qv: @Vector(vl, f32) = q[qi * q_dim + q_off + d ..][0..vl].*;
                    const kv: @Vector(vl, f32) = k_cache[ki * kv_dim + kv_off + d ..][0..vl].*;
                    acc += qv * kv;
                }
                var dot: f32 = @reduce(.Add, acc);
                while (d < head_dim) : (d += 1) {
                    dot += q[qi * q_dim + q_off + d] * k_cache[ki * kv_dim + kv_off + d];
                }
                scores_buf[ki - kv_start] = dot * scale;
            }

            softmaxInPlace(scores_buf[0..kv_len], kv_len);

            // Weighted sum of values
            for (kv_start..kv_end) |ki| {
                const w_scalar = scores_buf[ki - kv_start];
                const wv: @Vector(vl, f32) = @splat(w_scalar);
                var d: usize = 0;
                while (d + vl <= head_dim) : (d += vl) {
                    var ov: @Vector(vl, f32) = output[qi * q_dim + q_off + d ..][0..vl].*;
                    const vv: @Vector(vl, f32) = v_cache[ki * kv_dim + kv_off + d ..][0..vl].*;
                    ov += wv * vv;
                    output[qi * q_dim + q_off + d ..][0..vl].* = ov;
                }
                while (d < head_dim) : (d += 1) {
                    output[qi * q_dim + q_off + d] += w_scalar * v_cache[ki * kv_dim + kv_off + d];
                }
            }
        }
    }
}

/// GQA cached attention (decode, single token)
fn gqaCachedAttention(
    q: []const f32, // (Q_DIM,)
    k_cache: []const f32, // (CTX, KV_DIM)
    v_cache: []const f32, // (CTX, KV_DIM)
    output: []f32, // (Q_DIM,)
    scores_buf: []f32,
    kv_start: usize,
    kv_end: usize,
    n_head: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const kv_len = kv_end - kv_start;

    @memset(output[0..q_dim], 0);

    for (0..n_head) |hd| {
        const q_off = hd * head_dim;
        // GQA: all Q heads share the same KV head
        const kv_off: usize = 0;

        // Compute scores
        for (kv_start..kv_end) |ki| {
            var acc: @Vector(vl, f32) = @splat(0);
            var d: usize = 0;
            while (d + vl <= head_dim) : (d += vl) {
                const qv: @Vector(vl, f32) = q[q_off + d ..][0..vl].*;
                const kv: @Vector(vl, f32) = k_cache[ki * kv_dim + kv_off + d ..][0..vl].*;
                acc += qv * kv;
            }
            var dot: f32 = @reduce(.Add, acc);
            while (d < head_dim) : (d += 1) {
                dot += q[q_off + d] * k_cache[ki * kv_dim + kv_off + d];
            }
            scores_buf[ki - kv_start] = dot * scale;
        }

        softmaxInPlace(scores_buf[0..kv_len], kv_len);

        // Weighted sum
        for (kv_start..kv_end) |ki| {
            const w_scalar = scores_buf[ki - kv_start];
            const wv: @Vector(vl, f32) = @splat(w_scalar);
            var d: usize = 0;
            while (d + vl <= head_dim) : (d += vl) {
                var ov: @Vector(vl, f32) = output[q_off + d ..][0..vl].*;
                const vv: @Vector(vl, f32) = v_cache[ki * kv_dim + kv_off + d ..][0..vl].*;
                ov += wv * vv;
                output[q_off + d ..][0..vl].* = ov;
            }
            while (d < head_dim) : (d += 1) {
                output[q_off + d] += w_scalar * v_cache[ki * kv_dim + kv_off + d];
            }
        }
    }
}

/// Sliding window: layer 5,11,17,23 = global
pub fn isGlobalLayer(layer: usize) bool {
    return (layer % 6 == 5);
}

/// RoPE 周波数テーブルを事前計算
pub fn computeRoPEFreqs(comptime head_dim: usize, rope_base: f32) [head_dim / 2]f32 {
    var freqs: [head_dim / 2]f32 = undefined;
    for (0..head_dim / 2) |i| {
        const freq_exp = -@as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(head_dim));
        freqs[i] = std.math.pow(f32, rope_base, freq_exp);
    }
    return freqs;
}

/// QuantizedWeight → Metal QuantType 変換
pub fn quantTypeOfWeight(w: QuantizedWeight) @import("../backend/metal.zig").QuantType {
    return switch (w.type_) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q8_0 => .q8_0,
        else => unreachable,
    };
}

/// GELU activation (tanh approximation, in-place)
fn geluInPlace(x: []f32, n: usize) void {
    const sqrt_2_over_pi: f32 = 0.7978845608;
    for (0..n) |i| {
        const v = x[i];
        const inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
        x[i] = 0.5 * v * (1.0 + std.math.tanh(inner));
    }
}

/// Softmax (in-place, numerically stable)
fn softmaxInPlace(x: []f32, n: usize) void {
    var max_val: f32 = -std.math.inf(f32);
    for (0..n) |i| {
        if (x[i] > max_val) max_val = x[i];
    }
    var sum: f32 = 0;
    for (0..n) |j| {
        x[j] = @exp(x[j] - max_val);
        sum += x[j];
    }
    const inv_sum = 1.0 / sum;
    for (0..n) |j| {
        x[j] *= inv_sum;
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

/// Element-wise multiply in-place: a *= b (SIMD)
fn mulInPlace(a: []f32, b: []const f32, n: usize) void {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    var i: usize = 0;
    while (i + vl <= n) : (i += vl) {
        const va: @Vector(vl, f32) = a[i..][0..vl].*;
        const vb: @Vector(vl, f32) = b[i..][0..vl].*;
        a[i..][0..vl].* = va * vb;
    }
    while (i < n) : (i += 1) {
        a[i] *= b[i];
    }
}

// Re-export sampling utilities from gpt2
pub const sampleTopK = gpt2_mod.sampleTopK;
pub const argmax = gpt2_mod.argmax;

// ============================================================
// テスト
// ============================================================

test "rmsNorm basic" {
    const input = [_]f32{ 1, 2, 3, 4 };
    const weight = [_]f32{ 1, 1, 1, 1 };
    var output: [4]f32 = undefined;
    rmsNormRows(&input, &weight, &output, 1, 4, 1e-6);

    // rms = sqrt((1+4+9+16)/4 + 1e-6) = sqrt(7.5)
    // output[i] = input[i] / rms
    const rms = @sqrt(7.5 + 1e-6);
    for (0..4) |i| {
        const expected = input[i] / rms;
        try std.testing.expectApproxEqAbs(expected, output[i], 1e-4);
    }
}

test "applyRoPE basic" {
    // pos=0 → theta=0 for all freqs → cos=1, sin=0 → no change
    var x = [_]f32{ 1, 2, 3, 4 };
    applyRoPE(&x, 0, 4, 10000.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), x[3], 1e-5);
}

test "isGlobalLayer" {
    try std.testing.expect(isGlobalLayer(5));
    try std.testing.expect(isGlobalLayer(11));
    try std.testing.expect(isGlobalLayer(17));
    try std.testing.expect(isGlobalLayer(23));
    try std.testing.expect(!isGlobalLayer(0));
    try std.testing.expect(!isGlobalLayer(1));
    try std.testing.expect(!isGlobalLayer(6));
    try std.testing.expect(!isGlobalLayer(25));
}

test "geluInPlace basic" {
    var x = [_]f32{ 0, 1, -1 };
    geluInPlace(&x, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), x[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), x[2], 0.01);
}
