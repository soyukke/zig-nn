const std = @import("std");

// Re-export: 新モジュール
pub const simd_dot = @import("simd_dot.zig");
pub const thread_pool = @import("thread_pool.zig");
pub const ThreadPool = thread_pool.ThreadPool;

/// GGML テンソルの量子化型
pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    // 4, 5 は deprecated
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    _,
};

/// 量子化型ごとのブロックサイズ（要素数）
pub fn blockSize(t: GGMLType) usize {
    return switch (t) {
        .f32, .f16 => 1,
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
        _ => 0,
    };
}

/// 1ブロックあたりのバイト数
pub fn bytesPerBlock(t: GGMLType) usize {
    return switch (t) {
        .f32 => 4,
        .f16 => 2,
        .q4_0 => 18, // 2 (f16 scale) + 16 (32 nibbles)
        .q4_1 => 20, // 2 (f16 scale) + 2 (f16 min) + 16
        .q8_0 => 34, // 2 (f16 scale) + 32 (int8)
        .q8_1 => 36, // 2 (f16 scale) + 2 (f16 sum) + 32 (int8)
        .q5_0 => 22, // 2 (f16 scale) + 4 (qh bitmask) + 16
        .q5_1 => 24, // 2 (f16 scale) + 2 (f16 min) + 4 (qh) + 16
        _ => 0,
    };
}

/// 量子化ブロック内の要素数 (全型共通)
pub const BLOCK_SIZE: usize = 32;
/// Q4_0: 1 ブロックあたりのバイト数
pub const Q4_0_BPB: usize = 18;
/// Q4_1: 1 ブロックあたりのバイト数
pub const Q4_1_BPB: usize = 20;
/// Q8_0: 1 ブロックあたりのバイト数
pub const Q8_0_BPB: usize = 34;

/// num_elements 分のテンソルデータに必要なバイト数
pub fn tensorBytes(t: GGMLType, num_elements: usize) usize {
    const bs = blockSize(t);
    if (bs == 0) return 0;
    const num_blocks = num_elements / bs;
    return num_blocks * bytesPerBlock(t);
}

// ============================================================
// 逆量子化ルーチン
// ============================================================

/// F32: バイト列をそのまま f32 として読み出し
pub fn dequantizeF32(src: []const u8, dst: []f32) void {
    const count = dst.len;
    for (0..count) |i| {
        dst[i] = @bitCast(src[i * 4 ..][0..4].*);
    }
}

/// F16 → F32 変換
pub fn dequantizeF16(src: []const u8, dst: []f32) void {
    const count = dst.len;
    for (0..count) |i| {
        const bits: u16 = @bitCast(src[i * 2 ..][0..2].*);
        const val: f16 = @bitCast(bits);
        dst[i] = @floatCast(val);
    }
}

/// Q4_0 ブロック逆量子化
/// ブロック: f16 scale (2 bytes) + 16 bytes (32 x 4bit nibbles)
/// 復元: (nibble - 8) * scale
pub fn dequantizeQ4_0(src: []const u8, dst: []f32, num_elements: usize) void {
    const num_blocks = num_elements / BLOCK_SIZE;

    for (0..num_blocks) |bi| {
        const block = src[bi * Q4_0_BPB ..][0..Q4_0_BPB];
        const scale_bits: u16 = @bitCast(block[0..2].*);
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));

        for (0..16) |j| {
            const byte = block[2 + j];
            // 下位ニブル → dst[j], 上位ニブル → dst[j + 16]
            const lo: i32 = @as(i32, byte & 0x0F) - 8;
            const hi: i32 = @as(i32, byte >> 4) - 8;
            dst[bi * BLOCK_SIZE + j] = @as(f32, @floatFromInt(lo)) * scale;
            dst[bi * BLOCK_SIZE + j + 16] = @as(f32, @floatFromInt(hi)) * scale;
        }
    }
}

/// Q8_0 ブロック逆量子化
/// ブロック: f16 scale (2 bytes) + 32 bytes (int8 x 32)
/// 復元: q * scale
pub fn dequantizeQ8_0(src: []const u8, dst: []f32, num_elements: usize) void {
    const num_blocks = num_elements / BLOCK_SIZE;

    for (0..num_blocks) |bi| {
        const block = src[bi * Q8_0_BPB ..][0..Q8_0_BPB];
        const scale_bits: u16 = @bitCast(block[0..2].*);
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));

        for (0..32) |j| {
            const q: i8 = @bitCast(block[2 + j]);
            dst[bi * BLOCK_SIZE + j] = @as(f32, @floatFromInt(q)) * scale;
        }
    }
}

// ============================================================
// Q4_0 直接ドット積（逆量子化なし）
// ============================================================

/// Q4_0 行 × f32 ベクトルのドット積 (SIMD)
/// weight_row: Q4_0 エンコード済みの 1 行分 (in_dim 要素 = in_dim/32 ブロック)
/// input: f32 ベクトル (in_dim 要素)
/// in_dim: 入力次元（32 の倍数であること）
pub fn dotQ4_0_f32(weight_row: []const u8, input: []const f32, in_dim: usize) f32 {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    const num_blocks = in_dim / BLOCK_SIZE;
    var sum: f32 = 0;

    for (0..num_blocks) |bi| {
        const block = weight_row[bi * Q4_0_BPB ..][0..Q4_0_BPB];
        const scale_bits: u16 = @bitCast(block[0..2].*);
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
        const base = bi * BLOCK_SIZE;
        const inp = input[base..][0..BLOCK_SIZE];

        // ニブルを f32 に展開
        var vals: [32]f32 = undefined;
        for (0..16) |j| {
            const byte = block[2 + j];
            vals[j] = @floatFromInt(@as(i32, byte & 0x0F) - 8);
            vals[j + 16] = @floatFromInt(@as(i32, byte >> 4) - 8);
        }

        // SIMD ドット積
        var acc: @Vector(vl, f32) = @splat(0);
        var k: usize = 0;
        while (k + vl <= 32) : (k += vl) {
            const vv: @Vector(vl, f32) = vals[k..][0..vl].*;
            const vi: @Vector(vl, f32) = inp[k..][0..vl].*;
            acc += vv * vi;
        }
        sum += @reduce(.Add, acc) * scale;
    }
    return sum;
}

/// Q4_0 行列 × f32 ベクトル → f32 出力ベクトル
/// weight: Q4_0 エンコード済み行列 (out_dim 行 × in_dim 列)
/// input: f32 ベクトル (in_dim 要素)
/// output: f32 出力 (out_dim 要素)
pub fn matmulQ4_0_f32(
    weight: []const u8,
    input: []const f32,
    output: []f32,
    out_dim: usize,
    in_dim: usize,
) void {
    const row_bytes = tensorBytes(.q4_0, in_dim);
    for (0..out_dim) |j| {
        const row = weight[j * row_bytes ..][0..row_bytes];
        output[j] = dotQ4_0_f32(row, input, in_dim);
    }
}

// ============================================================
// Q4_1 直接ドット積（逆量子化なし）
// ============================================================

/// Q4_1 ブロック逆量子化
/// ブロック: f16 scale (2 bytes) + f16 min (2 bytes) + 16 bytes (32 x 4bit nibbles)
/// 復元: nibble * scale + min
pub fn dequantizeQ4_1(src: []const u8, dst: []f32, num_elements: usize) void {
    const num_blocks = num_elements / BLOCK_SIZE;

    for (0..num_blocks) |bi| {
        const block = src[bi * Q4_1_BPB ..][0..Q4_1_BPB];
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[0..2].*)))));
        const min_val: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[2..4].*)))));

        for (0..16) |j| {
            const byte = block[4 + j];
            const lo: f32 = @floatFromInt(@as(u32, byte & 0x0F));
            const hi: f32 = @floatFromInt(@as(u32, byte >> 4));
            dst[bi * BLOCK_SIZE + j] = lo * scale + min_val;
            dst[bi * BLOCK_SIZE + j + 16] = hi * scale + min_val;
        }
    }
}

/// Q4_1 行 × f32 ベクトルのドット積 (SIMD)
pub fn dotQ4_1_f32(weight_row: []const u8, input: []const f32, in_dim: usize) f32 {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    const num_blocks = in_dim / BLOCK_SIZE;
    var sum: f32 = 0;

    for (0..num_blocks) |bi| {
        const block = weight_row[bi * Q4_1_BPB ..][0..Q4_1_BPB];
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[0..2].*)))));
        const min_val: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[2..4].*)))));
        const base = bi * BLOCK_SIZE;
        const inp = input[base..][0..BLOCK_SIZE];

        // ニブルを f32 に展開
        var vals: [32]f32 = undefined;
        for (0..16) |j| {
            const byte = block[4 + j];
            vals[j] = @floatFromInt(@as(u32, byte & 0x0F));
            vals[j + 16] = @floatFromInt(@as(u32, byte >> 4));
        }

        // SIMD ドット積: sum += scale * dot(vals, inp) + min * sum(inp)
        var acc: @Vector(vl, f32) = @splat(0);
        var inp_sum: @Vector(vl, f32) = @splat(0);
        var k: usize = 0;
        while (k + vl <= 32) : (k += vl) {
            const vv: @Vector(vl, f32) = vals[k..][0..vl].*;
            const vi: @Vector(vl, f32) = inp[k..][0..vl].*;
            acc += vv * vi;
            inp_sum += vi;
        }
        sum += @reduce(.Add, acc) * scale + @reduce(.Add, inp_sum) * min_val;
    }
    return sum;
}

/// Q4_1 行列 × f32 ベクトル → f32 出力ベクトル
pub fn matmulQ4_1_f32(
    weight: []const u8,
    input: []const f32,
    output: []f32,
    out_dim: usize,
    in_dim: usize,
) void {
    const row_bytes = tensorBytes(.q4_1, in_dim);
    for (0..out_dim) |j| {
        const row = weight[j * row_bytes ..][0..row_bytes];
        output[j] = dotQ4_1_f32(row, input, in_dim);
    }
}

// ============================================================
// Q8_0 直接ドット積（逆量子化なし）
// ============================================================

/// Q8_0 行 × f32 ベクトルのドット積 (SIMD)
pub fn dotQ8_0_f32(weight_row: []const u8, input: []const f32, in_dim: usize) f32 {
    const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
    const num_blocks = in_dim / BLOCK_SIZE;
    var sum: f32 = 0;

    for (0..num_blocks) |bi| {
        const block = weight_row[bi * Q8_0_BPB ..][0..Q8_0_BPB];
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[0..2].*)))));
        const base = bi * BLOCK_SIZE;
        const inp = input[base..][0..BLOCK_SIZE];

        // int8 を f32 に展開
        var vals: [32]f32 = undefined;
        for (0..32) |j| {
            const q: i8 = @bitCast(block[2 + j]);
            vals[j] = @floatFromInt(q);
        }

        // SIMD ドット積
        var acc: @Vector(vl, f32) = @splat(0);
        var k: usize = 0;
        while (k + vl <= 32) : (k += vl) {
            const vv: @Vector(vl, f32) = vals[k..][0..vl].*;
            const vi: @Vector(vl, f32) = inp[k..][0..vl].*;
            acc += vv * vi;
        }
        sum += @reduce(.Add, acc) * scale;
    }
    return sum;
}

/// Q8_0 行列 × f32 ベクトル → f32 出力ベクトル
pub fn matmulQ8_0_f32(
    weight: []const u8,
    input: []const f32,
    output: []f32,
    out_dim: usize,
    in_dim: usize,
) void {
    const row_bytes = tensorBytes(.q8_0, in_dim);
    for (0..out_dim) |j| {
        const row = weight[j * row_bytes ..][0..row_bytes];
        output[j] = dotQ8_0_f32(row, input, in_dim);
    }
}

// ============================================================
// マルチスレッド matmul
// ============================================================

const MAX_THREADS = 16;

pub const QuantType = enum { q4_0, q4_1, q8_0 };

/// matmul の部分行チャンク (レガシー matmulParallel 用)
fn matmulChunkF32(
    weight: []const u8,
    input: []const f32,
    output: []f32,
    row_start: usize,
    row_end: usize,
    row_bytes: usize,
    in_dim: usize,
    quant_type: QuantType,
) void {
    for (row_start..row_end) |j| {
        const row = weight[j * row_bytes ..][0..row_bytes];
        output[j] = switch (quant_type) {
            .q4_0 => dotQ4_0_f32(row, input, in_dim),
            .q4_1 => dotQ4_1_f32(row, input, in_dim),
            .q8_0 => dotQ8_0_f32(row, input, in_dim),
        };
    }
}

/// マルチスレッド matmul (レガシー: spawn/join 方式、GPT-2 後方互換): out_dim の行ループを複数スレッドで分割
pub fn matmulParallel(
    weight: []const u8,
    input: []const f32,
    output: []f32,
    out_dim: usize,
    in_dim: usize,
    quant_type: QuantType,
    n_threads: usize,
) void {
    const row_bytes: usize = switch (quant_type) {
        .q4_0 => tensorBytes(.q4_0, in_dim),
        .q4_1 => tensorBytes(.q4_1, in_dim),
        .q8_0 => tensorBytes(.q8_0, in_dim),
    };

    // スレッドあたり最低 256 行を確保 (spawn オーバーヘッド対策)
    const effective_threads = @max(1, @min(@min(n_threads, MAX_THREADS), out_dim / 256));

    if (effective_threads <= 1) {
        matmulChunkF32(weight, input, output, 0, out_dim, row_bytes, in_dim, quant_type);
        return;
    }

    const rows_per_thread = out_dim / effective_threads;
    var threads: [MAX_THREADS]std.Thread = undefined;
    var spawned: usize = 0;

    // ワーカースレッドを起動 (2番目以降のチャンク)
    for (1..effective_threads) |t| {
        const start = t * rows_per_thread;
        const end = if (t == effective_threads - 1)
            out_dim
        else
            (t + 1) * rows_per_thread;
        threads[t - 1] = std.Thread.spawn(.{}, matmulChunkF32, .{
            weight, input, output, start, end, row_bytes, in_dim, quant_type,
        }) catch break;
        spawned += 1;
    }

    // メインスレッドが最初のチャンクを処理
    matmulChunkF32(weight, input, output, 0, rows_per_thread, row_bytes, in_dim, quant_type);

    // 全スレッドを join
    for (0..spawned) |t| {
        threads[t].join();
    }

    // spawn に失敗したスレッドの分をメインスレッドで補完
    if (spawned < effective_threads - 1) {
        const remaining_start = (spawned + 1) * rows_per_thread;
        matmulChunkF32(
            weight,
            input,
            output,
            remaining_start,
            out_dim,
            row_bytes,
            in_dim,
            quant_type,
        );
    }
}

/// 型に応じた統合逆量子化関数
pub fn dequantize(
    type_: GGMLType,
    src: []const u8,
    dst: []f32,
    num_elements: usize,
) !void {
    switch (type_) {
        .f32 => dequantizeF32(src, dst),
        .f16 => dequantizeF16(src, dst),
        .q4_0 => dequantizeQ4_0(src, dst, num_elements),
        .q4_1 => dequantizeQ4_1(src, dst, num_elements),
        .q8_0 => dequantizeQ8_0(src, dst, num_elements),
        else => return error.UnsupportedQuantType,
    }
}

// ============================================================
// テスト
// ============================================================

test "dequantize F32" {
    // f32 values: 1.0, -2.5, 3.14
    const src = std.mem.toBytes(@as(f32, 1.0)) ++
        std.mem.toBytes(@as(f32, -2.5)) ++
        std.mem.toBytes(@as(f32, 3.14));
    var dst: [3]f32 = undefined;
    dequantizeF32(&src, &dst);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), dst[2], 0.01);
}

test "dequantize F16" {
    const val1: f16 = 1.0;
    const val2: f16 = -0.5;
    const src = std.mem.toBytes(@as(u16, @bitCast(val1))) ++
        std.mem.toBytes(@as(u16, @bitCast(val2)));
    var dst: [2]f32 = undefined;
    dequantizeF16(&src, &dst);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), dst[1], 1e-3);
}

test "dequantize Q4_0 basic" {
    // 1 block = 32 elements
    // scale = 1.0 (f16), nibbles all = 8 (value 0 after subtracting 8)
    var block: [18]u8 = undefined;
    const scale: f16 = 1.0;
    const scale_bytes = std.mem.toBytes(@as(u16, @bitCast(scale)));
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // All nibbles = 8: lo=8, hi=8 → byte = 0x88
    @memset(block[2..18], 0x88);

    var dst: [32]f32 = undefined;
    dequantizeQ4_0(&block, &dst, 32);
    // (8 - 8) * 1.0 = 0
    for (&dst) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    }
}

test "dequantize Q4_0 non-zero" {
    var block: [18]u8 = undefined;
    const scale: f16 = 2.0;
    const scale_bytes = std.mem.toBytes(@as(u16, @bitCast(scale)));
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // byte[0]: lo=0xF (15-8=7), hi=0x0 (0-8=-8) → byte = 0x0F
    block[2] = 0x0F;
    @memset(block[3..18], 0x88); // rest are zero

    var dst: [32]f32 = undefined;
    dequantizeQ4_0(&block, &dst, 32);
    // dst[0] = (15-8) * 2.0 = 14.0
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), dst[0], 0.01);
    // dst[16] = (0-8) * 2.0 = -16.0
    try std.testing.expectApproxEqAbs(@as(f32, -16.0), dst[16], 0.01);
}

test "dequantize Q8_0 basic" {
    var block: [34]u8 = undefined;
    const scale: f16 = 0.5;
    const scale_bytes = std.mem.toBytes(@as(u16, @bitCast(scale)));
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // q[0] = 4 (int8), q[1] = -3 (int8 = 0xFD)
    block[2] = 4;
    block[3] = @bitCast(@as(i8, -3));
    @memset(block[4..34], 0);

    var dst: [32]f32 = undefined;
    dequantizeQ8_0(&block, &dst, 32);
    // 4 * 0.5 = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[0], 0.01);
    // -3 * 0.5 = -1.5
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), dst[1], 0.01);
    // 0 * 0.5 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dst[2], 1e-6);
}

test "dotQ4_0_f32 basic" {
    // 1 block (32 elements), scale = 1.0, all nibbles = 9 → (9-8)=1
    var block: [18]u8 = undefined;
    const scale: f16 = 1.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    // lo=9, hi=9 → byte = 0x99
    @memset(block[2..18], 0x99);

    // input = all 1.0
    var input: [32]f32 = undefined;
    @memset(&input, 1.0);

    // Each element is (9-8)*1.0 = 1.0, dot with 1.0 = 32 * 1.0 = 32.0
    const result = dotQ4_0_f32(&block, &input, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 0.01);
}

test "dotQ4_0_f32 matches dequantize" {
    // Create a Q4_0 block with known values
    var block: [18]u8 = undefined;
    const scale: f16 = 2.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    block[2] = 0x0F; // lo=15, hi=0
    block[3] = 0x17; // lo=7, hi=1
    @memset(block[4..18], 0x88); // rest: (8-8)*scale = 0

    // Dequantize for reference
    var dequantized: [32]f32 = undefined;
    dequantizeQ4_0(&block, &dequantized, 32);

    // input = all 1.0
    var input: [32]f32 = undefined;
    @memset(&input, 1.0);

    // Reference: sum of all dequantized values (since input is 1.0)
    var expected: f32 = 0;
    for (dequantized) |v| expected += v;

    const result = dotQ4_0_f32(&block, &input, 32);
    try std.testing.expectApproxEqAbs(expected, result, 0.01);
}

test "matmulQ4_0_f32 basic" {
    // 2 rows × 32 cols (1 block per row)
    var weight: [36]u8 = undefined; // 2 × 18 bytes
    const scale1: f16 = 1.0;
    const scale2: f16 = 0.5;

    // Row 0: scale=1.0, all nibbles = 9 → val = 1.0
    weight[0..2].* = @bitCast(@as(u16, @bitCast(scale1)));
    @memset(weight[2..18], 0x99);

    // Row 1: scale=0.5, all nibbles = 10 → val = (10-8)*0.5 = 1.0
    weight[18..20].* = @bitCast(@as(u16, @bitCast(scale2)));
    @memset(weight[20..36], 0xAA);

    var input: [32]f32 = undefined;
    @memset(&input, 1.0);
    var output: [2]f32 = undefined;

    matmulQ4_0_f32(&weight, &input, &output, 2, 32);

    // Row 0: each val = 1.0 * 1.0, sum = 32.0
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), output[0], 0.01);
    // Row 1: each val = 1.0 * 1.0, sum = 32.0
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), output[1], 0.01);
}

test "dequantize Q4_1 basic" {
    var block: [20]u8 = undefined;
    const scale: f16 = 1.0;
    const min_val: f16 = 0.5;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    block[2..4].* = @bitCast(@as(u16, @bitCast(min_val)));
    // All nibbles = 0 → value = 0 * 1.0 + 0.5 = 0.5
    @memset(block[4..20], 0x00);

    var dst: [32]f32 = undefined;
    dequantizeQ4_1(&block, &dst, 32);
    for (&dst) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), v, 1e-3);
    }
}

test "dotQ4_1_f32 basic" {
    var block: [20]u8 = undefined;
    const scale: f16 = 1.0;
    const min_val: f16 = 0.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    block[2..4].* = @bitCast(@as(u16, @bitCast(min_val)));
    // lo=1, hi=1 → byte = 0x11
    @memset(block[4..20], 0x11);

    var input: [32]f32 = undefined;
    @memset(&input, 1.0);

    // Each val = 1*1.0 + 0 = 1.0, dot with 1.0 = 32.0
    const result = dotQ4_1_f32(&block, &input, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 0.1);
}

test "dotQ4_1_f32 with min" {
    var block: [20]u8 = undefined;
    const scale: f16 = 2.0;
    const min_val: f16 = 1.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    block[2..4].* = @bitCast(@as(u16, @bitCast(min_val)));
    // All nibbles = 0 → val = 0*2 + 1 = 1.0
    @memset(block[4..20], 0x00);

    var input: [32]f32 = undefined;
    @memset(&input, 1.0);

    // Each val = 1.0, dot with 1.0 = 32.0
    const result = dotQ4_1_f32(&block, &input, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 0.1);
}

test "dotQ8_0_f32 basic" {
    var block: [34]u8 = undefined;
    const scale: f16 = 1.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    // q[i] = 2 for all i
    @memset(block[2..34], 2);

    var input: [32]f32 = undefined;
    @memset(&input, 1.0);

    // Each val = 2 * 1.0 = 2.0, dot with 1.0 = 64.0
    const result = dotQ8_0_f32(&block, &input, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), result, 0.01);
}

test "dotQ8_0_f32 matches dequantize" {
    var block: [34]u8 = undefined;
    const scale: f16 = 0.5;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    block[2] = 4;
    block[3] = @bitCast(@as(i8, -3));
    @memset(block[4..34], 0);

    var dequantized: [32]f32 = undefined;
    dequantizeQ8_0(&block, &dequantized, 32);

    var input: [32]f32 = undefined;
    @memset(&input, 1.0);

    var expected: f32 = 0;
    for (dequantized) |v| expected += v;

    const result = dotQ8_0_f32(&block, &input, 32);
    try std.testing.expectApproxEqAbs(expected, result, 0.01);
}

test "matmulQ8_0_f32 basic" {
    // 2 rows × 32 cols
    var weight: [68]u8 = undefined; // 2 × 34 bytes
    const scale1: f16 = 1.0;
    const scale2: f16 = 0.5;

    // Row 0: scale=1.0, all q=1
    weight[0..2].* = @bitCast(@as(u16, @bitCast(scale1)));
    @memset(weight[2..34], 1);

    // Row 1: scale=0.5, all q=2
    weight[34..36].* = @bitCast(@as(u16, @bitCast(scale2)));
    @memset(weight[36..68], 2);

    var input: [32]f32 = undefined;
    @memset(&input, 1.0);
    var output: [2]f32 = undefined;

    matmulQ8_0_f32(&weight, &input, &output, 2, 32);

    // Row 0: 1 * 1.0 * 32 = 32.0
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), output[0], 0.01);
    // Row 1: 2 * 0.5 * 32 = 32.0
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), output[1], 0.01);
}

test "tensorBytes calculation" {
    try std.testing.expectEqual(@as(usize, 128), tensorBytes(.f32, 32));
    try std.testing.expectEqual(@as(usize, 64), tensorBytes(.f16, 32));
    try std.testing.expectEqual(@as(usize, 18), tensorBytes(.q4_0, 32));
    try std.testing.expectEqual(@as(usize, 36), tensorBytes(.q4_0, 64));
    try std.testing.expectEqual(@as(usize, 20), tensorBytes(.q4_1, 32));
    try std.testing.expectEqual(@as(usize, 34), tensorBytes(.q8_0, 32));
}
