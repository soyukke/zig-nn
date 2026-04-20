const std = @import("std");
const nn = @import("nn");

pub const std_options = nn.log.std_options;
const log = nn.log.example;

pub fn main(init: std.process.Init) !void {
    try gpt2Demo(init.io);
}

fn gpt2Demo(io: std.Io) !void {
    const allocator = std.heap.page_allocator;
    log.info("=== Demo 4: GPT-2 Text Generation (GGUF Loader) ===", .{});

    const model_path = "models/gpt2.gguf";

    // 1. Parse GGUF file
    log.info("loading {s}...", .{model_path});
    var gguf_file = nn.gguf.parse(allocator, io, model_path) catch |err| {
        log.err("could not load {s}: {} (download GPT-2 GGUF model to models/)", .{ model_path, err });
        return;
    };
    defer gguf_file.deinit();

    // Print model info
    if (gguf_file.getMetadataString("general.architecture")) |arch| {
        log.info("architecture: {s}", .{arch});
    }
    if (gguf_file.getMetadataString("general.name")) |name| {
        log.info("model: {s}", .{name});
    }
    log.info("tensors: {d}, metadata: {d}", .{ gguf_file.tensors.len, gguf_file.metadata.len });

    // 2. Load tokenizer
    log.info("loading tokenizer...", .{});
    var tokenizer = nn.bpe.BPETokenizer.initFromGGUF(&gguf_file, allocator) catch |err| {
        log.err("loading tokenizer: {}", .{err});
        return;
    };
    defer tokenizer.deinit();
    log.info("vocab size: {d}", .{tokenizer.vocab_count});

    // 3. Load model weights
    log.info("loading weights...", .{});
    var model = nn.gpt2.GPT2(nn.gpt2.GPT2Small).init(&gguf_file, allocator) catch |err| {
        log.err("loading weights: {}", .{err});
        return;
    };
    defer model.deinit();
    log.info("model loaded", .{});

    // 4. Generate text
    const prompt = "Hello, I am";
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

    var prng = std.Random.DefaultPrng.init(nn.nowNanos());
    const rng = prng.random();

    model.resetCache();

    var timer = nn.Timer.start() catch unreachable;

    // Prefill: プロンプト全体を一括処理
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

        const next_token = nn.gpt2.sampleTopK(logits, 40, 0.9, rng);
        try tokens.append(allocator, next_token);

        const tok_slice = [_]u32{next_token};
        if (tokenizer.decode(&tok_slice, temp)) |decoded| {
            stdout.print("{s}", .{decoded}) catch {};
        } else |_| {}
    }

    // Decode: 1トークンずつ生成
    for (1..gen_tokens) |_| {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const last_token = tokens.items[tokens.items.len - 1];
        const logits = model.decodeNext(last_token, temp) catch |err| {
            log.err("in decodeNext: {}", .{err});
            return;
        };

        // Repetition penalty (直近32トークンのみ)
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

        const next_token = nn.gpt2.sampleTopK(logits, 40, 0.9, rng);
        try tokens.append(allocator, next_token);

        const tok_slice = [_]u32{next_token};
        const decoded = tokenizer.decode(&tok_slice, temp) catch continue;
        stdout.print("{s}", .{decoded}) catch {};
    }
    stdout.writeAll("\n") catch {};

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(gen_tokens)) / (elapsed_ms / 1000.0);

    log.info("generation: {d:.0}ms ({d:.1} tokens/sec)", .{ elapsed_ms, tokens_per_sec });
}
