const std = @import("std");
const nn = @import("nn");
const metal = nn.metal;

pub const std_options = nn.log.std_options;
const log = nn.log.example;

pub fn main(init: std.process.Init) !void {
    var args = init.minimal.args.iterate();
    _ = args.skip();
    const mode = args.next() orelse "cpu";

    if (std.mem.eql(u8, mode, "cpu")) {
        try gemma3Demo(init.io);
    } else if (std.mem.eql(u8, mode, "metal")) {
        try gemma3MetalDemo(init.io);
    } else if (std.mem.eql(u8, mode, "qlora")) {
        try gemma3QLoRADemo(init.io);
    } else {
        log.err("usage: gemma3 [cpu|metal|qlora]", .{});
    }
}

// ============================================================
// Gemma 3 1B CPU Text Generation
// ============================================================

fn gemma3Demo(io: std.Io) !void {
    const allocator = std.heap.page_allocator;
    log.info("=== Demo 5: Gemma 3 1B Text Generation (GGUF Loader) ===", .{});

    const model_path = "models/gemma3-1b.gguf";

    // 1. Parse GGUF file
    log.info("loading {s}...", .{model_path});
    var gguf_file = nn.gguf.parse(allocator, io, model_path) catch |err| {
        log.err("could not load {s}: {} (download Gemma 3 1B GGUF model to models/)", .{ model_path, err });
        return;
    };
    defer gguf_file.deinit();

    // Print model info
    if (gguf_file.get_metadata_string("general.architecture")) |arch| {
        log.info("architecture: {s}", .{arch});
    }
    if (gguf_file.get_metadata_string("general.name")) |name| {
        log.info("model: {s}", .{name});
    }
    log.info("tensors: {d}, metadata: {d}", .{ gguf_file.tensors.len, gguf_file.metadata.len });

    // 2. Load tokenizer (SentencePiece for Gemma)
    log.info("loading tokenizer...", .{});
    var tokenizer = nn.sentencepiece.SentencePieceTokenizer.init_from_gguf(&gguf_file, allocator) catch |err| {
        log.err("loading tokenizer: {}", .{err});
        return;
    };
    defer tokenizer.deinit();
    log.info("vocab size: {d}", .{tokenizer.vocab_count});

    // 3. Load model weights
    log.info("loading weights...", .{});
    var model = nn.gemma3.gemma3(nn.gemma3.Gemma3_1B).init(&gguf_file, allocator) catch |err| {
        log.err("loading weights: {}", .{err});
        return;
    };
    defer model.deinit();
    log.info("model loaded", .{});

    // 4. Generate text
    const prompt = "The meaning of life is";
    const gen_tokens = 50;

    log.info("prompt: \"{s}\"", .{prompt});
    log.info("generating {d} tokens...", .{gen_tokens});

    const stdout = std.fs.File.stdout().deprecatedWriter();

    // Encode prompt
    var tokens: std.ArrayListAligned(u32, null) = .empty;
    defer tokens.deinit(allocator);
    const prompt_ids = tokenizer.encode(prompt, allocator) catch |err| {
        log.err("encoding: {}", .{err});
        return;
    };
    defer allocator.free(prompt_ids);
    for (prompt_ids) |id| try tokens.append(allocator, id);

    // Print prompt to stdout
    stdout.print("{s}", .{prompt}) catch {};

    // Generate tokens with KV cache
    var gen_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gen_arena.deinit();

    var prng = std.Random.DefaultPrng.init(nn.now_nanos());
    const rng = prng.random();

    model.reset_cache();

    var timer = nn.Timer.start() catch unreachable;

    // Prefill
    {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const logits = model.prefill(prompt_ids, temp) catch |err| {
            log.err("in prefill: {}", .{err});
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

        const next_token = nn.gemma3.sample_top_k(logits, 40, 0.8, rng);
        try tokens.append(allocator, next_token);

        if (next_token != tokenizer.eos_id) {
            const tok_slice = [_]u32{next_token};
            if (tokenizer.decode(&tok_slice, temp)) |decoded| {
                stdout.print("{s}", .{decoded}) catch {};
            } else |_| {}
        }
    }

    // Decode
    var generated: usize = 1;
    const first_gen_token = tokens.items[tokens.items.len - 1];
    if (first_gen_token == tokenizer.eos_id) {
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        stdout.writeAll("\n") catch {};
        log.info("generated 1 tokens in {d:.0}ms (EOS)", .{elapsed_ms});
        return;
    }
    for (1..gen_tokens) |_| {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const last_token = tokens.items[tokens.items.len - 1];
        const logits_d = model.decode_next(last_token, temp) catch |err| {
            log.err("in decodeNext: {}", .{err});
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

        const next_token = nn.gemma3.sample_top_k(logits_d, 40, 0.8, rng);
        if (next_token == tokenizer.eos_id) break;
        try tokens.append(allocator, next_token);
        generated += 1;

        const tok_slice = [_]u32{next_token};
        const decoded = tokenizer.decode(&tok_slice, temp) catch continue;
        stdout.print("{s}", .{decoded}) catch {};
    }
    stdout.writeAll("\n") catch {};

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(generated + 1)) / (elapsed_ms / 1000.0);
    log.info("generated {d} tokens in {d:.0}ms ({d:.1} tokens/sec)", .{ generated + 1, elapsed_ms, tokens_per_sec });
    model.profile.print();
}

// ============================================================
// Gemma 3 1B Metal GPU Text Generation
// ============================================================

fn gemma3MetalDemo(io: std.Io) !void {
    const allocator = std.heap.page_allocator;
    log.info("=== Demo 5b: Gemma 3 1B Metal GPU Text Generation ===", .{});

    const model_path = "models/gemma3-1b.gguf";

    // 1. Parse GGUF file
    log.info("loading {s}...", .{model_path});
    var gguf_file = nn.gguf.parse(allocator, io, model_path) catch |err| {
        log.err("could not load {s}: {}", .{ model_path, err });
        return;
    };
    defer gguf_file.deinit();

    if (gguf_file.get_metadata_string("general.name")) |name| {
        log.info("model: {s}", .{name});
    }

    // 2. Load tokenizer
    log.info("loading tokenizer...", .{});
    var tokenizer = nn.sentencepiece.SentencePieceTokenizer.init_from_gguf(&gguf_file, allocator) catch |err| {
        log.err("loading tokenizer: {}", .{err});
        return;
    };
    defer tokenizer.deinit();

    // 3. Load model with Metal
    log.info("loading weights (Metal GPU)...", .{});
    var model = nn.gemma3_metal.gemma3_metal(nn.gemma3_metal.Gemma3_1B).init(&gguf_file, allocator) catch |err| {
        log.err("loading model: {}", .{err});
        return;
    };
    defer model.deinit();
    log.info("model loaded", .{});

    // 4. Generate text
    const prompt = "The meaning of life is";
    const gen_tokens = 20;

    log.info("prompt: \"{s}\"", .{prompt});
    log.info("generating {d} tokens...", .{gen_tokens});

    const stdout = std.fs.File.stdout().deprecatedWriter();

    var tokens: std.ArrayListAligned(u32, null) = .empty;
    defer tokens.deinit(allocator);
    const prompt_ids = tokenizer.encode(prompt, allocator) catch |err| {
        log.err("encoding: {}", .{err});
        return;
    };
    defer allocator.free(prompt_ids);
    for (prompt_ids) |tok_id| try tokens.append(allocator, tok_id);

    stdout.print("{s}", .{prompt}) catch {};

    var gen_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gen_arena.deinit();

    var prng = std.Random.DefaultPrng.init(nn.now_nanos());
    const rng = prng.random();

    model.reset_cache();

    var timer_gen = nn.Timer.start() catch unreachable;
    var prefill_ns: u64 = 0;

    // Prefill
    {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const logits = model.prefill(prompt_ids, temp) catch |err| {
            log.err("in prefill: {}", .{err});
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

        const next_token = nn.gemma3_metal.sample_top_k(logits_mut, 40, 0.8, rng);
        try tokens.append(allocator, next_token);

        if (next_token != tokenizer.eos_id) {
            const tok_slice = [_]u32{next_token};
            if (tokenizer.decode(&tok_slice, temp)) |decoded| {
                stdout.print("{s}", .{decoded}) catch {};
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
        stdout.writeAll("\n") catch {};
        log.info("generated 1 tokens in {d:.0}ms (EOS)", .{elapsed_ms});
        return;
    }
    var timer_decode = nn.Timer.start() catch unreachable;
    for (1..gen_tokens) |_| {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const last_token = tokens.items[tokens.items.len - 1];
        const logits_d = model.decode_next(last_token, temp) catch |err| {
            log.err("in decodeNext: {}", .{err});
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

        const next_token = nn.gemma3_metal.sample_top_k(logits_d, 40, 0.8, rng);

        // Restore modified values
        for (0..saved_count) |si| {
            logits_d[saved_idxs[si]] = saved_vals[si];
        }
        if (next_token == tokenizer.eos_id) break;
        try tokens.append(allocator, next_token);
        generated += 1;

        const tok_slice = [_]u32{next_token};
        const decoded = tokenizer.decode(&tok_slice, temp) catch continue;
        stdout.print("{s}", .{decoded}) catch {};
    }
    stdout.writeAll("\n") catch {};

    const decode_ns = timer_decode.read();
    const elapsed_ns = timer_gen.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const prefill_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0;
    const decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(generated + 1)) / (elapsed_ms / 1000.0);
    const decode_tps = @as(f64, @floatFromInt(generated)) / (decode_ms / 1000.0);

    log.info("generated {d} tokens in {d:.0}ms ({d:.1} tokens/sec)", .{ generated + 1, elapsed_ms, tokens_per_sec });
    log.info("prefill: {d:.1}ms | decode: {d} tokens in {d:.1}ms ({d:.1} tok/s)", .{ prefill_ms, generated, decode_ms, decode_tps });
    model.profile.print();
}

// ============================================================
// Gemma 3 1B QLoRA Fine-tuning (Metal GPU, unified DiffMpsRuntime)
// ============================================================

fn gemma3QLoRADemo(io: std.Io) !void {
    const allocator = std.heap.page_allocator;
    log.info("=== Demo 9: Gemma 3 1B QLoRA Fine-tuning (Metal GPU) ===", .{});

    const C = nn.gemma3_qlora.Gemma3_1B;
    const RANK = 8;
    const SEQ_LEN = 32;

    // --- Metal context ---
    var mtl = try metal.MetalContext.init();
    defer mtl.deinit();
    log.info("metal device initialized", .{});

    // --- GGUF ---
    const model_path = "models/gemma3-1b.gguf";
    log.info("loading {s}...", .{model_path});
    var gguf_file = nn.gguf.parse(allocator, io, model_path) catch |err| {
        log.err("could not load model: {} (place gemma3-1b.gguf in models/)", .{err});
        return;
    };
    defer gguf_file.deinit();

    var tokenizer = nn.sentencepiece.SentencePieceTokenizer.init_from_gguf(&gguf_file, allocator) catch |err| {
        log.err("loading tokenizer: {}", .{err});
        return;
    };
    defer tokenizer.deinit();

    // --- Module + model + runtime ---
    log.info("initializing QLoRA model (rank={d})...", .{RANK});
    const QLoRAModel = nn.gemma3_qlora.gemma3_qlora(C, RANK);

    var module = nn.unified.Module.init(allocator);
    defer module.deinit();

    var model = QLoRAModel.init_params(&module);

    var rt = try nn.unified.DiffMpsRuntime.init(&module, &mtl, allocator);
    defer rt.deinit();
    rt.init_params();

    model.load_from_gguf(&rt, &gguf_file) catch |err| {
        log.err("loading weights: {}", .{err});
        return;
    };
    defer model.deinit();

    const total_params = module.total_param_elements();
    log.info("trainable params: {d} ({d} handles)", .{ total_params, module.param_count() });

    // --- Adam (GPU-resident grad buffers already allocated by rt) ---
    const sizes = try module.param_sizes(allocator);
    defer allocator.free(sizes);
    var adam = try nn.unified.AdamState.init(allocator, sizes);
    defer adam.deinit();

    // --- Training data (tiny demo) ---
    const train_texts = [_][]const u8{
        "The capital of Japan is Tokyo, which is one of the largest cities in the world with over thirteen million people.",
        "The capital of France is Paris, which is known for the Eiffel Tower and its beautiful art museums and cafes.",
    };

    var train_input_ids: [train_texts.len][SEQ_LEN]u32 = undefined;
    var train_target_ids: [train_texts.len][SEQ_LEN]u32 = undefined;
    for (train_texts, 0..) |text, ti| {
        const ids = tokenizer.encode(text, allocator) catch continue;
        defer allocator.free(ids);
        const n = @min(SEQ_LEN + 1, ids.len);
        for (0..SEQ_LEN) |t| {
            train_input_ids[ti][t] = if (t < n - 1) ids[t] else 0;
            train_target_ids[ti][t] = if (t + 1 < n) ids[t + 1] else 0;
        }
    }

    // --- Training loop ---
    const steps: usize = 5;
    const lr: f32 = 1e-4;
    log.info("training {d} steps (lr={d}, α={d}, rank={d})...", .{ steps, lr, 16, RANK });

    var timer = try nn.Timer.start();
    for (0..steps) |step| {
        var step_loss: f32 = 0;
        for (train_input_ids, 0..) |_, seq_idx| {
            rt.zero_grad();
            rt.reset_arena();

            const logits = model.forward(&rt, &train_input_ids[seq_idx]);
            const loss = rt.cross_entropy_loss_with_indices(logits, &train_target_ids[seq_idx]);
            const loss_val = MetalContextReadScalar(&rt, loss);

            rt.backward(loss);
            rt.apply_adam(&adam, lr, 0.9, 0.999, 1e-8, 0.0);

            step_loss += loss_val;
        }
        step_loss /= @floatFromInt(train_texts.len);
        const elapsed = timer.read();
        log.info("step {d:>2}: loss = {d:.4}  ({d} ms)", .{ step, step_loss, elapsed / std.time.ns_per_ms });
        timer.reset();
    }

    log.info("QLoRA demo complete", .{});
}

fn MetalContextReadScalar(rt: *nn.unified.DiffMpsRuntime, t: nn.unified.DiffMpsTensor) f32 {
    var buf: [1]f32 = undefined;
    rt.copy_to_host(t, &buf);
    return buf[0];
}
