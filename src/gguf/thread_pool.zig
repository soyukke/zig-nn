const std = @import("std");
const dequant = @import("dequant.zig");
const simd_dot = @import("simd_dot.zig");
const Q8Block = simd_dot.Q8Block;

pub const QuantType = dequant.QuantType;

// ============================================================
// ThreadPool: persistent worker threads + Q8 整数ドット積
// ============================================================
//
// matmul() は以下の手順で動作:
// 1. メインスレッドで f32 入力を Q8 ブロックに量子化 (1 回だけ)
// 2. Q8 ブロックを全ワーカーで共有
// 3. 各ワーカーが i8×i8→i32 整数ドット積で行チャンクを計算
//
// ワーカーはスピンウェイトで待機し、generation counter で同期。

pub const ThreadPool = struct {
    const MAX_WORKERS = 15; // worker threads (main thread = +1)
    const MAX_Q8_BLOCKS = 256; // 最大 8192 要素 (256 × 32)

    const WorkDesc = struct {
        weight: []const u8 = &.{},
        q8_input: []const Q8Block = &.{},
        output: []f32 = &.{},
        row_start: usize = 0,
        row_end: usize = 0,
        row_bytes: usize = 0,
        num_blocks: usize = 0,
        quant_type: QuantType = .q4_0,
    };

    workers: [MAX_WORKERS]std.Thread = undefined,
    n_workers: usize = 0,
    work: [MAX_WORKERS]WorkDesc = [_]WorkDesc{.{}} ** MAX_WORKERS,
    q8_buf: [MAX_Q8_BLOCKS]Q8Block = undefined,

    // Synchronization fields (accessed via atomic builtins)
    gen: u32 = 0, // generation counter - incremented to signal new work
    pending: u32 = 0, // workers still computing
    stop: u32 = 0, // shutdown flag (0=running, 1=stop)

    /// ヒープ上に ThreadPool を確保しワーカーを起動
    pub fn init(n_threads: usize, allocator: std.mem.Allocator) !*ThreadPool {
        const pool = try allocator.create(ThreadPool);
        pool.* = .{};
        const n = @min(if (n_threads > 1) n_threads - 1 else 0, MAX_WORKERS);
        for (0..n) |i| {
            pool.workers[i] = std.Thread.spawn(.{}, worker_loop, .{ pool, i }) catch break;
            pool.n_workers += 1;
        }
        return pool;
    }

    pub fn deinit(pool: *ThreadPool, allocator: std.mem.Allocator) void {
        @atomicStore(u32, &pool.stop, 1, .release);
        _ = @atomicRmw(u32, &pool.gen, .Add, 1, .release);
        for (0..pool.n_workers) |i| {
            pool.workers[i].join();
        }
        allocator.destroy(pool);
    }

    fn worker_loop(pool: *ThreadPool, id: usize) void {
        var my_gen: u32 = 0;
        while (true) {
            // Spin until new generation or shutdown
            while (@atomicLoad(u32, &pool.gen, .acquire) == my_gen) {
                if (@atomicLoad(u32, &pool.stop, .acquire) != 0) return;
                std.atomic.spinLoopHint();
            }
            my_gen +%= 1;

            // Check shutdown after waking
            if (@atomicLoad(u32, &pool.stop, .acquire) != 0) return;

            const w = &pool.work[id];
            if (w.row_end > w.row_start) {
                matmul_chunk_q8(
                    w.weight,
                    w.q8_input,
                    w.output,
                    w.row_start,
                    w.row_end,
                    w.row_bytes,
                    w.quant_type,
                );
            }

            _ = @atomicRmw(u32, &pool.pending, .Sub, 1, .release);
        }
    }

    /// matmul: 入力を Q8 量子化 → 整数ドット積でマルチスレッド計算
    pub fn matmul(
        pool: *ThreadPool,
        weight: []const u8,
        input: []const f32,
        output: []f32,
        out_dim: usize,
        in_dim: usize,
        qt: QuantType,
    ) void {
        const row_bytes: usize = switch (qt) {
            .q4_0 => dequant.tensor_bytes(.q4_0, in_dim),
            .q4_1 => dequant.tensor_bytes(.q4_1, in_dim),
            .q8_0 => dequant.tensor_bytes(.q8_0, in_dim),
        };

        // (1) 入力を Q8 に量子化 (1回だけ、メインスレッド)
        const num_blocks = in_dim / 32;
        const q8_slice = pool.q8_buf[0..num_blocks];
        simd_dot.quantize_row_q8(input[0..in_dim], q8_slice);

        // Total threads = workers + main; min 256 rows/thread
        const total_threads = pool.n_workers + 1;
        const effective = @max(1, @min(total_threads, out_dim / 256));

        if (effective <= 1 or pool.n_workers == 0) {
            matmul_chunk_q8(weight, q8_slice, output, 0, out_dim, row_bytes, qt);
            return;
        }

        const rows_per = out_dim / effective;
        const n_active = effective - 1; // workers to use (main = +1)

        // Set up work for active workers (chunks 1..N)
        for (0..n_active) |i| {
            const start = (i + 1) * rows_per;
            const end = if (i == n_active - 1) out_dim else (i + 2) * rows_per;
            pool.work[i] = .{
                .weight = weight,
                .q8_input = q8_slice,
                .output = output,
                .row_start = start,
                .row_end = end,
                .row_bytes = row_bytes,
                .num_blocks = num_blocks,
                .quant_type = qt,
            };
        }
        // Clear inactive workers
        for (n_active..pool.n_workers) |i| {
            pool.work[i].row_start = 0;
            pool.work[i].row_end = 0;
        }

        // Signal all workers
        @atomicStore(u32, &pool.pending, @as(u32, @intCast(pool.n_workers)), .release);
        _ = @atomicRmw(u32, &pool.gen, .Add, 1, .release);

        // Main thread does first chunk
        matmul_chunk_q8(weight, q8_slice, output, 0, rows_per, row_bytes, qt);

        // Wait for all workers (including inactive ones that just decrement)
        while (@atomicLoad(u32, &pool.pending, .acquire) != 0) {
            std.atomic.spinLoopHint();
        }
    }
};

/// Q8 整数ドット積を使った matmul チャンク (ワーカースレッド用)
fn matmul_chunk_q8(
    weight: []const u8,
    q8_input: []const Q8Block,
    output: []f32,
    row_start: usize,
    row_end: usize,
    row_bytes: usize,
    quant_type: QuantType,
) void {
    for (row_start..row_end) |j| {
        const row = weight[j * row_bytes ..][0..row_bytes];
        output[j] = switch (quant_type) {
            .q4_0 => simd_dot.dot_q4_0_q8(row, q8_input),
            .q4_1 => simd_dot.dot_q4_1_q8(row, q8_input),
            .q8_0 => simd_dot.dot_q8_0_q8(row, q8_input),
        };
    }
}
