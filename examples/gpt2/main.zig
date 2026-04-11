const std = @import("std");
const nn = @import("nn");

pub fn main() !void {
    try gpt2Demo();
}

fn gpt2Demo() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== Demo 4: GPT-2 Text Generation (GGUF Loader) ===\n\n", .{});

    const model_path = "models/gpt2.gguf";

    // 1. Parse GGUF file
    std.debug.print("  Loading {s}...\n", .{model_path});
    var gguf_file = nn.gguf.parse(allocator, model_path) catch |err| {
        std.debug.print("  Error: could not load {s}: {}\n", .{ model_path, err });
        std.debug.print("  (Download GPT-2 GGUF model to models/ directory)\n", .{});
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

    // 2. Load tokenizer
    std.debug.print("  Loading tokenizer...\n", .{});
    var tokenizer = nn.bpe.BPETokenizer.initFromGGUF(&gguf_file, allocator) catch |err| {
        std.debug.print("  Error loading tokenizer: {}\n", .{err});
        return;
    };
    defer tokenizer.deinit();
    std.debug.print("  Vocab size: {d}\n", .{tokenizer.vocab_count});

    // 3. Load model weights
    std.debug.print("  Loading weights...\n", .{});
    var model = nn.gpt2.GPT2(nn.gpt2.GPT2Small).init(&gguf_file, allocator) catch |err| {
        std.debug.print("  Error loading weights: {}\n", .{err});
        return;
    };
    defer model.deinit();
    std.debug.print("  Model loaded!\n\n", .{});

    // 4. Generate text
    const prompt = "Hello, I am";
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

    // Prefill: プロンプト全体を一括処理
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

        const next_token = nn.gpt2.sampleTopK(logits, 40, 0.9, rng);
        try tokens.append(allocator, next_token);

        const tok_slice = [_]u32{next_token};
        if (tokenizer.decode(&tok_slice, temp)) |decoded| {
            std.debug.print("{s}", .{decoded});
        } else |_| {}
    }

    // Decode: 1トークンずつ生成
    for (1..gen_tokens) |_| {
        _ = gen_arena.reset(.retain_capacity);
        const temp = gen_arena.allocator();

        const last_token = tokens.items[tokens.items.len - 1];
        const logits = model.decodeNext(last_token, temp) catch |err| {
            std.debug.print("\n  Error in decodeNext: {}\n", .{err});
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
        std.debug.print("{s}", .{decoded});
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(gen_tokens)) / (elapsed_ms / 1000.0);

    std.debug.print("\"\n\n", .{});
    std.debug.print("  Generation: {d:.0}ms ({d:.1} tokens/sec)\n", .{ elapsed_ms, tokens_per_sec });
}
