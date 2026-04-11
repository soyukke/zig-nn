const std = @import("std");
const nn = @import("nn");
const metal = nn.metal;

pub fn main() !void {
    var args = std.process.args();
    _ = args.skip();
    const mode = args.next() orelse "cpu";

    if (std.mem.eql(u8, mode, "cpu")) {
        try gemma3Demo();
    } else if (std.mem.eql(u8, mode, "metal")) {
        try gemma3MetalDemo();
    } else if (std.mem.eql(u8, mode, "qlora")) {
        try gemma3QLoRADemo();
    } else {
        std.debug.print("Usage: gemma3 [cpu|metal|qlora]\n", .{});
    }
}

// ============================================================
// Gemma 3 1B CPU Text Generation
// ============================================================

fn gemma3Demo() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== Demo 5: Gemma 3 1B Text Generation (GGUF Loader) ===\n\n", .{});

    const model_path = "models/gemma3-1b.gguf";

    // 1. Parse GGUF file
    std.debug.print("  Loading {s}...\n", .{model_path});
    var gguf_file = nn.gguf.parse(allocator, model_path) catch |err| {
        std.debug.print("  Error: could not load {s}: {}\n", .{ model_path, err });
        std.debug.print("  (Download Gemma 3 1B GGUF model to models/ directory)\n", .{});
        return;
    };
    defer gguf_file.deinit();

    // Print model info
    if (gguf_file.getMetadataString("general.architecture")) |arch| {
        std.debug.print("  Architecture: {s}\n", .{arch});
    }
    if (gguf_file.getMetadataString("general.name")) |name| {
        std.debug.print("  Model: {s}\n", .{name});
    }
    std.debug.print("  Tensors: {d}, Metadata: {d}\n", .{ gguf_file.tensors.len, gguf_file.metadata.len });

    // 2. Load tokenizer (SentencePiece for Gemma)
    std.debug.print("  Loading tokenizer...\n", .{});
    var tokenizer = nn.sentencepiece.SentencePieceTokenizer.initFromGGUF(&gguf_file, allocator) catch |err| {
        std.debug.print("  Error loading tokenizer: {}\n", .{err});
        return;
    };
    defer tokenizer.deinit();
    std.debug.print("  Vocab size: {d}\n", .{tokenizer.vocab_count});

    // 3. Load model weights
    std.debug.print("  Loading weights...\n", .{});
    var model = nn.gemma3.Gemma3(nn.gemma3.Gemma3_1B).init(&gguf_file, allocator) catch |err| {
        std.debug.print("  Error loading weights: {}\n", .{err});
        return;
    };
    defer model.deinit();
    std.debug.print("  Model loaded!\n\n", .{});

    // 4. Generate text
    const prompt = "The meaning of life is";
    const gen_tokens = 50;

    std.debug.print("  Prompt: \"{s}\"\n", .{prompt});
    std.debug.print("  Generating {d} tokens...\n\n  Output: \"", .{gen_tokens});

    // Encode prompt
    var tokens: std.ArrayListAligned(u32, null) = .empty;
    defer tokens.deinit(allocator);
    const prompt_ids = tokenizer.encode(prompt, allocator) catch |err| {
        std.debug.print("  Error encoding: {}\n", .{err});
        return;
    };
    defer allocator.free(prompt_ids);
    for (prompt_ids) |id| try tokens.append(allocator, id);

    // Print prompt
    std.debug.print("{s}", .{prompt});

    // Generate tokens with KV cache
    var gen_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gen_arena.deinit();

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = prng.random();

    model.resetCache();

    var timer = std.time.Timer.start() catch unreachable;

    // Prefill
    {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const logits = model.prefill(prompt_ids, temp) catch |err| {
            std.debug.print("\n  Error in prefill: {}\n", .{err});
            return;
        };

        // Repetition penalty
        const penalty_window = @min(tokens.items.len, 32);
        const penalty_start = tokens.items.len - penalty_window;
        for (tokens.items[penalty_start..]) |prev_tok| {
            if (prev_tok < logits.len) {
                if (logits[prev_tok] > 0) {
                    logits[prev_tok] /= 1.2;
                } else {
                    logits[prev_tok] *= 1.2;
                }
            }
        }

        const next_token = nn.gemma3.sampleTopK(logits, 40, 0.8, rng);
        try tokens.append(allocator, next_token);

        if (next_token != tokenizer.eos_id) {
            const tok_slice = [_]u32{next_token};
            if (tokenizer.decode(&tok_slice, temp)) |decoded| {
                std.debug.print("{s}", .{decoded});
            } else |_| {}
        }
    }

    // Decode
    var generated: usize = 1;
    const first_gen_token = tokens.items[tokens.items.len - 1];
    if (first_gen_token == tokenizer.eos_id) {
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        std.debug.print("[EOS]\"\n\n", .{});
        std.debug.print("  Generated 1 tokens in {d:.0}ms\n", .{elapsed_ms});
        return;
    }
    for (1..gen_tokens) |_| {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const last_token = tokens.items[tokens.items.len - 1];
        const logits_d = model.decodeNext(last_token, temp) catch |err| {
            std.debug.print("\n  Error in decodeNext: {}\n", .{err});
            return;
        };

        // Repetition penalty
        const penalty_window = @min(tokens.items.len, 32);
        const penalty_start = tokens.items.len - penalty_window;
        for (tokens.items[penalty_start..]) |prev_tok| {
            if (prev_tok < logits_d.len) {
                if (logits_d[prev_tok] > 0) {
                    logits_d[prev_tok] /= 1.2;
                } else {
                    logits_d[prev_tok] *= 1.2;
                }
            }
        }

        const next_token = nn.gemma3.sampleTopK(logits_d, 40, 0.8, rng);
        if (next_token == tokenizer.eos_id) break;
        try tokens.append(allocator, next_token);
        generated += 1;

        const tok_slice = [_]u32{next_token};
        const decoded = tokenizer.decode(&tok_slice, temp) catch continue;
        std.debug.print("{s}", .{decoded});
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(generated + 1)) / (elapsed_ms / 1000.0);

    std.debug.print("\"\n\n", .{});
    std.debug.print("  Generated {d} tokens in {d:.0}ms ({d:.1} tokens/sec)\n", .{ generated + 1, elapsed_ms, tokens_per_sec });
    model.profile.print();
}

// ============================================================
// Gemma 3 1B Metal GPU Text Generation
// ============================================================

fn gemma3MetalDemo() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== Demo 5b: Gemma 3 1B Metal GPU Text Generation ===\n\n", .{});

    const model_path = "models/gemma3-1b.gguf";

    // 1. Parse GGUF file
    std.debug.print("  Loading {s}...\n", .{model_path});
    var gguf_file = nn.gguf.parse(allocator, model_path) catch |err| {
        std.debug.print("  Error: could not load {s}: {}\n", .{ model_path, err });
        return;
    };
    defer gguf_file.deinit();

    if (gguf_file.getMetadataString("general.name")) |name| {
        std.debug.print("  Model: {s}\n", .{name});
    }

    // 2. Load tokenizer
    std.debug.print("  Loading tokenizer...\n", .{});
    var tokenizer = nn.sentencepiece.SentencePieceTokenizer.initFromGGUF(&gguf_file, allocator) catch |err| {
        std.debug.print("  Error loading tokenizer: {}\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    // 3. Load model with Metal
    std.debug.print("  Loading weights (Metal GPU)...\n", .{});
    var model = nn.gemma3_metal.Gemma3Metal(nn.gemma3_metal.Gemma3_1B).init(&gguf_file, allocator) catch |err| {
        std.debug.print("  Error loading model: {}\n", .{err});
        return;
    };
    defer model.deinit();
    std.debug.print("  Model loaded!\n\n", .{});

    // 4. Generate text
    const prompt = "The meaning of life is";
    const gen_tokens = 20;

    std.debug.print("  Prompt: \"{s}\"\n", .{prompt});
    std.debug.print("  Generating {d} tokens...\n\n  Output: \"", .{gen_tokens});

    var tokens: std.ArrayListAligned(u32, null) = .empty;
    defer tokens.deinit(allocator);
    const prompt_ids = tokenizer.encode(prompt, allocator) catch |err| {
        std.debug.print("  Error encoding: {}\n", .{err});
        return;
    };
    defer allocator.free(prompt_ids);
    for (prompt_ids) |tok_id| try tokens.append(allocator, tok_id);

    std.debug.print("{s}", .{prompt});

    var gen_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gen_arena.deinit();

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = prng.random();

    model.resetCache();

    var timer_gen = std.time.Timer.start() catch unreachable;
    var prefill_ns: u64 = 0;

    // Prefill
    {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const logits = model.prefill(prompt_ids, temp) catch |err| {
            std.debug.print("\n  Error in prefill: {}\n", .{err});
            return;
        };

        // Repetition penalty (copy needed — sampleTopK modifies logits in-place)
        const logits_mut = try temp.alloc(f32, logits.len);
        @memcpy(logits_mut, logits);
        const penalty_window = @min(tokens.items.len, 32);
        const penalty_start = tokens.items.len - penalty_window;
        for (tokens.items[penalty_start..]) |prev_tok| {
            if (prev_tok < logits_mut.len) {
                if (logits_mut[prev_tok] > 0) {
                    logits_mut[prev_tok] /= 1.2;
                } else {
                    logits_mut[prev_tok] *= 1.2;
                }
            }
        }

        const next_token = nn.gemma3_metal.sampleTopK(logits_mut, 40, 0.8, rng);
        try tokens.append(allocator, next_token);

        if (next_token != tokenizer.eos_id) {
            const tok_slice = [_]u32{next_token};
            if (tokenizer.decode(&tok_slice, temp)) |decoded| {
                std.debug.print("{s}", .{decoded});
            } else |_| {}
        }
    }
    prefill_ns = timer_gen.read();

    // Decode
    var generated: usize = 1;
    const first_gen_token = tokens.items[tokens.items.len - 1];
    if (first_gen_token == tokenizer.eos_id) {
        const elapsed_ns = timer_gen.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        std.debug.print("[EOS]\"\n\n", .{});
        std.debug.print("  Generated 1 tokens in {d:.0}ms\n", .{elapsed_ms});
        return;
    }
    var timer_decode = std.time.Timer.start() catch unreachable;
    for (1..gen_tokens) |_| {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const last_token = tokens.items[tokens.items.len - 1];
        const logits_d = model.decodeNext(last_token, temp) catch |err| {
            std.debug.print("\n  Error in decodeNext: {}\n", .{err});
            return;
        };

        // Repetition penalty (in-place on shared buffer, restore after sampling)
        const penalty_window = @min(tokens.items.len, 32);
        const penalty_start = tokens.items.len - penalty_window;
        var saved_vals: [32]f32 = undefined;
        var saved_idxs: [32]u32 = undefined;
        var saved_count: usize = 0;
        for (tokens.items[penalty_start..]) |prev_tok| {
            if (prev_tok < logits_d.len) {
                saved_vals[saved_count] = logits_d[prev_tok];
                saved_idxs[saved_count] = prev_tok;
                saved_count += 1;
                if (logits_d[prev_tok] > 0) {
                    logits_d[prev_tok] /= 1.2;
                } else {
                    logits_d[prev_tok] *= 1.2;
                }
            }
        }

        const next_token = nn.gemma3_metal.sampleTopK(logits_d, 40, 0.8, rng);

        // Restore modified values
        for (0..saved_count) |si| {
            logits_d[saved_idxs[si]] = saved_vals[si];
        }
        if (next_token == tokenizer.eos_id) break;
        try tokens.append(allocator, next_token);
        generated += 1;

        const tok_slice = [_]u32{next_token};
        const decoded = tokenizer.decode(&tok_slice, temp) catch continue;
        std.debug.print("{s}", .{decoded});
    }

    const decode_ns = timer_decode.read();
    const elapsed_ns = timer_gen.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const prefill_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0;
    const decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(generated + 1)) / (elapsed_ms / 1000.0);
    const decode_tps = @as(f64, @floatFromInt(generated)) / (decode_ms / 1000.0);

    std.debug.print("\"\n\n", .{});
    std.debug.print("  Generated {d} tokens in {d:.0}ms ({d:.1} tokens/sec)\n", .{ generated + 1, elapsed_ms, tokens_per_sec });
    std.debug.print("  Prefill: {d:.1}ms | Decode: {d} tokens in {d:.1}ms ({d:.1} tok/s)\n", .{ prefill_ms, generated, decode_ms, decode_tps });
    model.profile.print();
}

// ============================================================
// Gemma 3 1B QLoRA Fine-tuning (Metal GPU)
// ============================================================

fn gemma3QLoRADemo() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== Demo 9: Gemma 3 1B QLoRA Fine-tuning (Metal GPU) ===\n\n", .{});

    const C = nn.gemma3_qlora.Gemma3_1B;
    const RANK = 8;
    const seq_len = 32;
    const T = f32;
    const MetalCtx = metal.MetalContext;
    const gpu_ops = nn.gpu_ops;

    // --- Metal init ---
    var mtl = try MetalCtx.init();
    defer mtl.deinit();
    try mtl.initTrainingPipelines();
    std.debug.print("  Metal training pipelines initialized.\n", .{});

    // --- Load GGUF ---
    const model_path = "models/gemma3-1b.gguf";
    std.debug.print("  Loading model: {s}\n", .{model_path});
    var gguf_file = nn.gguf.parse(allocator, model_path) catch |err| {
        std.debug.print("  Error: {}\n  Place gemma3-1b.gguf in models/\n", .{err});
        return;
    };
    defer gguf_file.deinit();

    // --- Load tokenizer ---
    std.debug.print("  Loading tokenizer...\n", .{});
    var tokenizer = nn.sentencepiece.SentencePieceTokenizer.initFromGGUF(&gguf_file, allocator) catch |err| {
        std.debug.print("  Error loading tokenizer: {}\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    // --- Init QLoRA model ---
    std.debug.print("  Initializing QLoRA model (rank={d})...\n", .{RANK});
    var model = nn.gemma3_qlora.Gemma3QLoRA(C, RANK).init(&gguf_file, &mtl, allocator) catch |err| {
        std.debug.print("  Error init model: {}\n", .{err});
        return;
    };
    defer model.deinit();

    const total_lora_params = C.LAYER * 4 * (C.EMBED * RANK + RANK * C.Q_DIM + C.EMBED * RANK + RANK * C.KV_DIM) / 4;
    _ = total_lora_params;
    const n_params = C.LAYER * 4; // q_a, q_b, v_a, v_b per layer
    std.debug.print("  LoRA params: {d} adapters ({d} per layer × {d} layers)\n", .{ n_params, 4, C.LAYER });

    // --- Training data ---
    // Longer sentences to fill seq_len and reduce padding noise
    const train_texts = [_][]const u8{
        "The capital of Japan is Tokyo, which is one of the largest cities in the world with over thirteen million people.",
        "The capital of France is Paris, which is known for the Eiffel Tower and its beautiful art museums and cafes.",
        "The capital of Japan is Tokyo, a modern city that blends traditional culture with cutting edge technology.",
        "The capital of France is Paris, the city of light that attracts millions of tourists from around the world.",
    };

    // Tokenize training data
    var train_input_ids: [train_texts.len][seq_len]u32 = undefined;
    var train_target_ids: [train_texts.len][seq_len]u32 = undefined;
    var actual_train_seqs: usize = 0;

    for (train_texts, 0..) |text, i| {
        const tokens = tokenizer.encode(text, allocator) catch continue;
        defer allocator.free(tokens);

        if (tokens.len < 2) continue;

        // Pad/truncate to seq_len
        const n = @min(tokens.len - 1, seq_len);
        for (0..seq_len) |t| {
            if (t < n) {
                train_input_ids[i][t] = tokens[t];
                train_target_ids[i][t] = tokens[t + 1];
            } else {
                // Pad with 0 (pad token), target = UINT_MAX (ignore in loss)
                train_input_ids[i][t] = 0;
                train_target_ids[i][t] = std.math.maxInt(u32);
            }
        }
        actual_train_seqs += 1;
    }

    if (actual_train_seqs == 0) {
        std.debug.print("  Error: no training sequences produced\n", .{});
        return;
    }
    // Count valid tokens per sequence
    for (0..actual_train_seqs) |seq_i| {
        var valid: usize = 0;
        for (0..seq_len) |t| {
            if (train_target_ids[seq_i][t] < C.VOCAB) valid += 1;
        }
        std.debug.print("  Seq {d}: {d}/{d} valid tokens\n", .{ seq_i, valid, seq_len });
    }
    std.debug.print("  Training sequences: {d} (seq_len={d})\n\n", .{ actual_train_seqs, seq_len });

    // --- Optimizer ---
    const GpuAdamT = nn.GpuAdam(T);
    const lora_params = model.getLoRAParams(allocator) catch |err| {
        std.debug.print("  Error getting LoRA params: {}\n", .{err});
        return;
    };
    defer allocator.free(lora_params);

    var optimizer = try GpuAdamT.init(lora_params, &mtl, allocator, 3e-5, 0.9, 0.999, 1e-8, 0.01);
    defer optimizer.deinit();

    // --- Pre-training verification: check base model output ---
    {
        var verify_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer verify_arena.deinit();
        const temp = verify_arena.allocator();

        mtl.beginBatch();
        const logits = model.forward(seq_len, &train_input_ids[0], temp) catch |err| {
            std.debug.print("  Verify forward error: {}\n", .{err});
            mtl.endBatch();
            return;
        };
        mtl.endBatch();

        // Print top-5 predictions at last non-padded position
        std.debug.print("  [Verify] Base model output for first training sequence:\n", .{});
        // Find last non-padded position
        var last_real_pos: usize = 0;
        for (0..seq_len) |t| {
            if (train_input_ids[0][t] != 0) last_real_pos = t;
        }
        const verify_logits = logits.data[last_real_pos * C.VOCAB .. (last_real_pos + 1) * C.VOCAB];

        // Find top-5
        var top5_idx: [5]u32 = .{ 0, 0, 0, 0, 0 };
        var top5_val: [5]T = .{ -std.math.inf(T), -std.math.inf(T), -std.math.inf(T), -std.math.inf(T), -std.math.inf(T) };
        for (verify_logits, 0..) |v, idx| {
            if (v > top5_val[4]) {
                top5_val[4] = v;
                top5_idx[4] = @intCast(idx);
                // Bubble up
                var k: usize = 4;
                while (k > 0 and top5_val[k] > top5_val[k - 1]) : (k -= 1) {
                    const tmp_v = top5_val[k];
                    const tmp_i = top5_idx[k];
                    top5_val[k] = top5_val[k - 1];
                    top5_idx[k] = top5_idx[k - 1];
                    top5_val[k - 1] = tmp_v;
                    top5_idx[k - 1] = tmp_i;
                }
            }
        }
        std.debug.print("    Position {d}, target token={d}\n", .{ last_real_pos, train_target_ids[0][last_real_pos] });
        for (0..5) |k| {
            const tok_str = tokenizer.decode(&[_]u32{top5_idx[k]}, allocator) catch "?";
            defer allocator.free(tok_str);
            std.debug.print("    Top-{d}: token={d} logit={d:.4} \"{s}\"\n", .{ k + 1, top5_idx[k], top5_val[k], tok_str });
        }
        // Also print loss at this position for reference
        const target_tok = train_target_ids[0][last_real_pos];
        std.debug.print("    Target logit={d:.4}\n\n", .{verify_logits[target_tok]});

        metal.objRelease(logits.data_buf);
    }

    // --- Training loop ---
    const num_steps = 100;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var timer = try std.time.Timer.start();
    var fwd_ns: u64 = 0;
    var bwd_ns: u64 = 0;
    var opt_ns: u64 = 0;

    std.debug.print("  Training ({d} steps, lr=3e-5, wd=0.01)...\n", .{num_steps});

    for (0..num_steps) |step| {
        var step_loss: T = 0;

        for (0..actual_train_seqs) |seq_idx| {
            _ = arena.reset(.retain_capacity);
            const temp = arena.allocator();

            // Forward (batched)
            var phase_timer = try std.time.Timer.start();
            mtl.beginBatch();

            const logits = model.forward(seq_len, &train_input_ids[seq_idx], temp) catch |err| {
                std.debug.print("  Forward error at step {d}: {}\n", .{ step, err });
                mtl.endBatch();
                return;
            };

            // Cross-entropy loss
            const loss = try gpu_ops.crossEntropyLoss(
                T, seq_len, C.VOCAB,
                logits.data, logits.data_buf, logits.node,
                &train_target_ids[seq_idx], &mtl, temp,
            );

            mtl.endBatch();
            fwd_ns += phase_timer.read();

            // Backward (batched)
            phase_timer.reset();
            mtl.beginBackwardBatch(temp);

            // Register weight grad bufs (LoRA + all norm vars)
            for (0..C.LAYER) |layer| {
                mtl.registerGradBuf(@ptrCast(model.lora_q_a[layer].node), model.lora_q_a[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.lora_q_b[layer].node), model.lora_q_b[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.lora_v_a[layer].node), model.lora_v_a[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.lora_v_b[layer].node), model.lora_v_b[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.attn_norm_vars[layer].node), model.attn_norm_vars[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.ffn_norm_vars[layer].node), model.ffn_norm_vars[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.q_norm_vars[layer].node), model.q_norm_vars[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.k_norm_vars[layer].node), model.k_norm_vars[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.post_attn_norm_vars[layer].node), model.post_attn_norm_vars[layer].grad_buf.?);
                mtl.registerGradBuf(@ptrCast(model.post_ffw_norm_vars[layer].node), model.post_ffw_norm_vars[layer].grad_buf.?);
            }
            mtl.registerGradBuf(@ptrCast(model.output_norm_var.node), model.output_norm_var.grad_buf.?);

            // Loss gradient = 1.0
            const loss_grad_buf = mtl.getOrAllocGradBuf(@ptrCast(loss.node), 1 * @sizeOf(T));
            const loss_grad_uma = MetalCtx.bufferContents(T, loss_grad_buf);
            loss_grad_uma[0] = 1.0;
            loss.node.grad = loss_grad_uma[0..1];

            var engine = nn.GradEngine(T).init(temp);
            try engine.backward(loss.node);
            mtl.endBackwardBatch();
            bwd_ns += phase_timer.read();

            step_loss += loss.data[0];

            // Release temp GPU buffers
            metal.objRelease(logits.data_buf);
            metal.objRelease(loss.data_buf);
        }


        // Gradient clipping (max_norm=1.0)
        {
            const max_norm: T = 100.0;
            var total_norm_sq: f64 = 0;
            for (lora_params) |p| {
                const grad_ptr = MetalCtx.bufferContents(T, p.grad_buf.?);
                for (0..p.count) |gi| {
                    total_norm_sq += @as(f64, grad_ptr[gi]) * @as(f64, grad_ptr[gi]);
                }
            }
            const total_norm: T = @floatCast(@sqrt(total_norm_sq));
            if (total_norm > max_norm) {
                const clip_coef = max_norm / total_norm;
                for (lora_params) |p| {
                    const grad_ptr = MetalCtx.bufferContents(T, p.grad_buf.?);
                    for (0..p.count) |gi| {
                        grad_ptr[gi] *= clip_coef;
                    }
                }
            }
        }

        // Optimizer step (gradient accumulated over sequences)
        var opt_timer = try std.time.Timer.start();
        optimizer.step();
        optimizer.zeroGrad();
        model.zeroGrad();
        opt_ns += opt_timer.read();

        const avg_loss = step_loss / @as(T, @floatFromInt(actual_train_seqs));
        if (step < 5 or step % 10 == 0 or step == num_steps - 1) {
            std.debug.print("  Step {d:>3}: loss = {d:.4}\n", .{ step, avg_loss });
        }
    }

    const elapsed_ms = timer.read() / 1_000_000;
    std.debug.print("\n  Training time: {d}ms ({d}ms/step)\n", .{ elapsed_ms, elapsed_ms / num_steps });
    std.debug.print("  Profile: forward={d}ms, backward={d}ms, optimizer={d}ms\n", .{
        fwd_ns / 1_000_000, bwd_ns / 1_000_000, opt_ns / 1_000_000,
    });

    // --- Text generation after fine-tuning ---
    std.debug.print("\n  Generating text after fine-tuning:\n", .{});

    const prompts = [_][]const u8{
        "The capital of Japan",
        "The capital of France",
    };

    for (prompts) |prompt| {
        const prompt_tokens = tokenizer.encode(prompt, allocator) catch continue;
        defer allocator.free(prompt_tokens);

        std.debug.print("    \"{s}", .{prompt});

        // Fill context with prompt tokens
        var gen_ctx: [seq_len]u32 = undefined;
        @memset(&gen_ctx, 0);
        const start = seq_len - @min(prompt_tokens.len, seq_len);
        for (prompt_tokens[0..@min(prompt_tokens.len, seq_len)], 0..) |tok, i| {
            gen_ctx[start + i] = tok;
        }

        var gen_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer gen_arena.deinit();

        // Generate 20 tokens
        for (0..20) |_| {
            _ = gen_arena.reset(.retain_capacity);
            const temp = gen_arena.allocator();

            mtl.beginBatch();
            const logits = model.forward(seq_len, &gen_ctx, temp) catch break;
            mtl.endBatch();

            // Last position logits → argmax
            const last_logits = logits.data[(seq_len - 1) * C.VOCAB .. seq_len * C.VOCAB];
            var max_val: T = -std.math.inf(T);
            var max_idx: u32 = 0;
            for (last_logits, 0..) |v, idx| {
                if (v > max_val) {
                    max_val = v;
                    max_idx = @intCast(idx);
                }
            }

            // Decode and print
            const tok_str = tokenizer.decode(&[_]u32{max_idx}, allocator) catch break;
            defer allocator.free(tok_str);
            std.debug.print("{s}", .{tok_str});

            // Shift context
            for (0..seq_len - 1) |i| {
                gen_ctx[i] = gen_ctx[i + 1];
            }
            gen_ctx[seq_len - 1] = max_idx;

            metal.objRelease(logits.data_buf);

            // Stop on EOS (token 1 for Gemma)
            if (max_idx == 1) break;
        }
        std.debug.print("\"\n", .{});
    }
}
