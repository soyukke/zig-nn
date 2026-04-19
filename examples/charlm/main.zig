const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");
const compute = nn.compute;
const Module = compute.Module;
const AdamState = compute.AdamState;
const DiffCpuRuntime = nn.unified.DiffCpuRuntime;
const DiffTensor = nn.unified.DiffTensor;
const DiffCudaRuntime = nn.unified.DiffCudaRuntime;
const GpuAdamState = nn.unified.GpuAdamState;
const Linear = nn.unified.Linear;
const LayerNorm = nn.unified.LayerNorm;
const Embedding = nn.unified.Embedding;
const CausalSelfAttention = nn.unified.CausalSelfAttention;

const is_cuda_available = builtin.os.tag == .linux;

pub fn main(init: std.process.Init.Minimal) !void {
    var args = init.args.iterate();
    _ = args.skip();
    const mode = args.next() orelse "cpu";

    if (is_cuda_available and std.mem.eql(u8, mode, "cuda")) {
        try charLMDemoCuda();
    } else {
        try charLMDemo();
    }
}

// --- Shared constants and helpers ---

const CHAR_VOCAB_SIZE = 28;
const CHAR_EMBED_DIM = 64;
const CHAR_SEQ_LEN = 32;
const CHAR_FF_DIM = 4 * CHAR_EMBED_DIM;

fn charEncode(c: u8) u32 {
    return switch (c) {
        ' ' => 0,
        '.' => 1,
        'a'...'z' => @as(u32, c - 'a') + 2,
        else => 0,
    };
}

fn charDecode(idx: u32) u8 {
    return switch (idx) {
        0 => ' ',
        1 => '.',
        else => if (idx >= 2 and idx <= 27) @as(u8, @intCast(idx - 2)) + 'a' else '?',
    };
}

fn argmax(comptime n: usize, data: []const f32) u32 {
    var best: u32 = 0;
    var best_val: f32 = data[0];
    for (1..n) |i| {
        if (data[i] > best_val) {
            best_val = data[i];
            best = @intCast(i);
        }
    }
    return best;
}

/// CharLM model: Embedding + 1-layer Transformer (causal) + Linear output
/// Pre-norm architecture: LN -> Attention -> Residual -> LN -> FF -> Residual -> LN -> Output
const CharLMModel = struct {
    tok_emb: Embedding(CHAR_VOCAB_SIZE, CHAR_EMBED_DIM),
    pos_emb: compute.ParamHandle, // [SEQ_LEN, EMBED_DIM]
    ln1: LayerNorm(CHAR_EMBED_DIM),
    attn: CausalSelfAttention(CHAR_EMBED_DIM, CHAR_SEQ_LEN),
    ln2: LayerNorm(CHAR_EMBED_DIM),
    ff1: Linear(CHAR_EMBED_DIM, CHAR_FF_DIM),
    ff2: Linear(CHAR_FF_DIM, CHAR_EMBED_DIM),
    ln_f: LayerNorm(CHAR_EMBED_DIM),
    out_proj: Linear(CHAR_EMBED_DIM, CHAR_VOCAB_SIZE),

    fn init(module: *Module) CharLMModel {
        return .{
            .tok_emb = Embedding(CHAR_VOCAB_SIZE, CHAR_EMBED_DIM).init(module),
            .pos_emb = module.addParam(&.{ CHAR_SEQ_LEN, CHAR_EMBED_DIM }, .xavier),
            .ln1 = LayerNorm(CHAR_EMBED_DIM).init(module),
            .attn = CausalSelfAttention(CHAR_EMBED_DIM, CHAR_SEQ_LEN).init(module),
            .ln2 = LayerNorm(CHAR_EMBED_DIM).init(module),
            .ff1 = Linear(CHAR_EMBED_DIM, CHAR_FF_DIM).init(module),
            .ff2 = Linear(CHAR_FF_DIM, CHAR_EMBED_DIM).init(module),
            .ln_f = LayerNorm(CHAR_EMBED_DIM).init(module),
            .out_proj = Linear(CHAR_EMBED_DIM, CHAR_VOCAB_SIZE).init(module),
        };
    }

    fn forward(self: CharLMModel, ctx: anytype, indices: []const u32) @TypeOf(ctx.param(self.pos_emb)) {
        const batch_size = 1;
        // 1. Token embedding + position embedding
        const tok = self.tok_emb.forward(ctx, indices);
        const pos = ctx.param(self.pos_emb);
        const h0 = ctx.add(tok, pos);

        // 2. Pre-norm causal self-attention + residual
        const ln1 = self.ln1.forward(ctx, h0);
        const sa = self.attn.forward(ctx, ln1, batch_size);
        const res1 = ctx.add(h0, sa);

        // 3. Pre-norm FF + residual
        const ln2 = self.ln2.forward(ctx, res1);
        const ff = ctx.gelu(self.ff1.forward(ctx, ln2));
        const res2 = ctx.add(res1, self.ff2.forward(ctx, ff));

        // 4. Final LN + output projection
        const h_final = self.ln_f.forward(ctx, res2);
        return self.out_proj.forward(ctx, h_final);
    }
};

fn prepareCorpus(
    all_input_ids: *[128][CHAR_SEQ_LEN]u32,
    all_target_ids: *[128][CHAR_SEQ_LEN]u32,
) usize {
    const base_text = "hello world. ";
    const corpus_repeats = 40;
    const corpus_len = base_text.len * corpus_repeats;
    var corpus: [corpus_len]u8 = undefined;
    for (0..corpus_repeats) |r| {
        @memcpy(corpus[r * base_text.len .. (r + 1) * base_text.len], base_text);
    }

    const num_sequences = (corpus_len - CHAR_SEQ_LEN - 1) / (CHAR_SEQ_LEN / 2);
    const actual_num_seq = @min(num_sequences, 128);
    for (0..actual_num_seq) |s| {
        for (0..CHAR_SEQ_LEN) |t| {
            const pos = s * (CHAR_SEQ_LEN / 2) + t;
            all_input_ids[s][t] = charEncode(corpus[pos]);
            all_target_ids[s][t] = charEncode(corpus[pos + 1]);
        }
    }

    std.debug.print("  Corpus: \"{s}...\" ({d} chars)\n", .{ corpus[0..base_text.len], corpus_len });
    std.debug.print("  Training sequences: {d} (seq_len={d}, overlap=50%%)\n", .{ actual_num_seq, CHAR_SEQ_LEN });
    return actual_num_seq;
}

fn charLMDemo() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== CharLM (CPU: Transformer + Adam) ===\n\n", .{});

    var all_input_ids: [128][CHAR_SEQ_LEN]u32 = undefined;
    var all_target_ids: [128][CHAR_SEQ_LEN]u32 = undefined;
    const actual_num_seq = prepareCorpus(&all_input_ids, &all_target_ids);

    var module = Module.init(allocator);
    defer module.deinit();
    const model = CharLMModel.init(&module);

    var rt = try DiffCpuRuntime.init(&module, allocator);
    defer rt.deinit();
    rt.initParams();

    const total_params = module.totalParamElements();
    std.debug.print("  Model: 1-layer Transformer, {d} params (~{d}KB)\n\n", .{ total_params, total_params * 4 / 1024 });

    const sizes = try module.paramSizes(allocator);
    defer allocator.free(sizes);
    var adam = try AdamState.init(allocator, sizes);
    defer adam.deinit();

    const num_epochs = 200;
    var timer = try nn.Timer.start();

    for (0..num_epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (0..actual_num_seq) |seq_idx| {
            rt.resetArena();
            rt.zeroGrad();

            const logits = model.forward(&rt, &all_input_ids[seq_idx]);
            const loss = rt.crossEntropyLossWithIndices(logits, &all_target_ids[seq_idx]);

            rt.backward(loss);
            rt.applyAdam(&adam, 0.001, 0.9, 0.999, 1e-8, 0);

            epoch_loss += loss.data[0];
        }

        if (epoch % 50 == 0 or epoch == num_epochs - 1) {
            const avg_loss = epoch_loss / @as(f32, @floatFromInt(actual_num_seq));
            std.debug.print("  Epoch {d:>4}: loss = {d:.4}\n", .{ epoch, avg_loss });
        }
    }

    const elapsed_ms = timer.read() / 1_000_000;
    std.debug.print("\n  Training time: {d}ms\n", .{elapsed_ms});

    generateText(&rt, model);
}

fn charLMDemoCuda() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== CharLM (CUDA: Transformer + Adam) ===\n\n", .{});

    var all_input_ids: [128][CHAR_SEQ_LEN]u32 = undefined;
    var all_target_ids: [128][CHAR_SEQ_LEN]u32 = undefined;
    const actual_num_seq = prepareCorpus(&all_input_ids, &all_target_ids);

    var module = Module.init(allocator);
    defer module.deinit();
    const model = CharLMModel.init(&module);

    const cuda = nn.cuda;
    var cuda_ctx = try cuda.CudaContext.init(0);
    defer cuda_ctx.deinit();

    var rt = try DiffCudaRuntime.init(&module, &cuda_ctx, allocator);
    defer rt.deinit();
    rt.initParams();

    const total_params = module.totalParamElements();
    std.debug.print("  Model: 1-layer Transformer, {d} params (~{d}KB)\n\n", .{ total_params, total_params * 4 / 1024 });

    const sizes = try module.paramSizes(allocator);
    defer allocator.free(sizes);
    var adam = try GpuAdamState.init(allocator, &cuda_ctx, sizes);
    defer adam.deinit();

    const num_epochs = 200;
    var timer = try nn.Timer.start();

    for (0..num_epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (0..actual_num_seq) |seq_idx| {
            rt.resetArena();
            rt.zeroGrad();

            const logits = model.forward(&rt, &all_input_ids[seq_idx]);
            const loss = rt.crossEntropyLossWithIndices(logits, &all_target_ids[seq_idx]);

            rt.backward(loss);
            rt.applyAdam(&adam, 0.001, 0.9, 0.999, 1e-8, 0);

            epoch_loss += rt.copyScalarToHost(loss);
        }

        if (epoch % 50 == 0 or epoch == num_epochs - 1) {
            const avg_loss = epoch_loss / @as(f32, @floatFromInt(actual_num_seq));
            std.debug.print("  Epoch {d:>4}: loss = {d:.4}\n", .{ epoch, avg_loss });
        }
    }

    const elapsed_ms = timer.read() / 1_000_000;
    std.debug.print("\n  Training time: {d}ms\n", .{elapsed_ms});

    generateTextCuda(&rt, model);
}

fn generateText(rt: *DiffCpuRuntime, model: CharLMModel) void {
    std.debug.print("\n  Generated text:\n    \"", .{});

    const seed = "hello world. hello world. hello";
    comptime std.debug.assert(seed.len <= CHAR_SEQ_LEN);
    var gen_ctx: [CHAR_SEQ_LEN]u32 = undefined;
    for (&gen_ctx) |*v| v.* = 0;
    for (seed, 0..) |c, i| {
        gen_ctx[CHAR_SEQ_LEN - seed.len + i] = charEncode(c);
    }
    for (seed) |c| std.debug.print("{c}", .{c});

    const gen_len = 60;
    for (0..gen_len) |_| {
        rt.resetArena();
        const logits = model.forward(rt, &gen_ctx);
        const last_logits = logits.data[(CHAR_SEQ_LEN - 1) * CHAR_VOCAB_SIZE .. CHAR_SEQ_LEN * CHAR_VOCAB_SIZE];
        const pred = argmax(CHAR_VOCAB_SIZE, last_logits);
        std.debug.print("{c}", .{charDecode(pred)});
        for (0..CHAR_SEQ_LEN - 1) |i| gen_ctx[i] = gen_ctx[i + 1];
        gen_ctx[CHAR_SEQ_LEN - 1] = pred;
    }
    std.debug.print("\"\n", .{});
}

fn generateTextCuda(rt: *DiffCudaRuntime, model: CharLMModel) void {
    std.debug.print("\n  Generated text:\n    \"", .{});

    const seed = "hello world. hello world. hello";
    comptime std.debug.assert(seed.len <= CHAR_SEQ_LEN);
    var gen_ctx: [CHAR_SEQ_LEN]u32 = undefined;
    for (&gen_ctx) |*v| v.* = 0;
    for (seed, 0..) |c, i| {
        gen_ctx[CHAR_SEQ_LEN - seed.len + i] = charEncode(c);
    }
    for (seed) |c| std.debug.print("{c}", .{c});

    const gen_len = 60;
    for (0..gen_len) |_| {
        rt.resetArena();
        const logits = model.forward(rt, &gen_ctx);
        var last_logits: [CHAR_VOCAB_SIZE]f32 = undefined;
        // Copy just the last row of logits
        const total = logits.totalElements();
        var all_logits: [CHAR_SEQ_LEN * CHAR_VOCAB_SIZE]f32 = undefined;
        rt.copyToHost(logits, all_logits[0..total]);
        @memcpy(&last_logits, all_logits[(CHAR_SEQ_LEN - 1) * CHAR_VOCAB_SIZE .. CHAR_SEQ_LEN * CHAR_VOCAB_SIZE]);
        const pred = argmax(CHAR_VOCAB_SIZE, &last_logits);
        std.debug.print("{c}", .{charDecode(pred)});
        for (0..CHAR_SEQ_LEN - 1) |i| gen_ctx[i] = gen_ctx[i + 1];
        gen_ctx[CHAR_SEQ_LEN - 1] = pred;
    }
    std.debug.print("\"\n", .{});
}
