const std = @import("std");
const dequant = @import("dequant.zig");

// ============================================================
// Q8 動的量子化 + 整数ドット積カーネル
// ============================================================
//
// f32 入力ベクトルを Q8 ブロックに 1 回だけ量子化し、
// 量子化済み weight (Q4_0/Q4_1/Q8_0) と i8×i8→i32 の
// 整数ドット積で matmul を高速化する。
//
// - int-to-float 変換が毎ブロック不要 (スカラー乗算 1 回のみ)
// - i8 は f32 の 1/4 サイズ → キャッシュ効率向上
// - Apple Silicon SDOT 命令 (i8×i8→i32) の活用可能性あり

/// Q8 量子化ブロック (32 要素)
pub const Q8Block = struct {
    q: [32]i8, // 量子化された値
    scale: f32, // スケールファクター (amax / 127)
    sum: i32, // sum(q[0..32]) — Q4_1 の min バイアス項に必要
};

/// f32 ベクトルを Q8 ブロック列に量子化
/// input.len は 32 の倍数であること
pub fn quantize_row_q8(input: []const f32, out: []Q8Block) void {
    const num_blocks = input.len / dequant.BLOCK_SIZE;

    for (0..num_blocks) |bi| {
        const base = bi * dequant.BLOCK_SIZE;
        const chunk = input[base..][0..dequant.BLOCK_SIZE];

        // SIMD で amax (最大絶対値) 計算
        const vl = comptime std.simd.suggestVectorLength(f32) orelse 4;
        var amax_v: @Vector(vl, f32) = @splat(0);
        var k: usize = 0;
        while (k + vl <= dequant.BLOCK_SIZE) : (k += vl) {
            const v: @Vector(vl, f32) = chunk[k..][0..vl].*;
            amax_v = @max(amax_v, @abs(v));
        }
        var amax: f32 = @reduce(.Max, amax_v);
        while (k < dequant.BLOCK_SIZE) : (k += 1) {
            amax = @max(amax, @abs(chunk[k]));
        }

        const scale: f32 = if (amax > 0) amax / 127.0 else 1.0;
        const inv_scale: f32 = 1.0 / scale;

        var blk: Q8Block = undefined;
        blk.scale = scale;
        var sum_acc: i32 = 0;

        for (0..dequant.BLOCK_SIZE) |j| {
            const v = chunk[j] * inv_scale;
            // round to nearest (std.math.round returns f32)
            const rounded = std.math.round(v);
            const q: i8 = @intFromFloat(std.math.clamp(rounded, -127, 127));
            blk.q[j] = q;
            sum_acc += @as(i32, q);
        }
        blk.sum = sum_acc;

        out[bi] = blk;
    }
}

// ============================================================
// 整数ドット積カーネル
// ============================================================

/// i8[32] × i8[32] → i32 整数ドット積 (SIMD)
inline fn dot_i8x32(a: *const [32]i8, b: *const [32]i8) i32 {
    // @Vector(16, i32) で 16 要素ずつ処理 (32 = 16 × 2)
    const a_lo: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), a[0..16].*));
    const b_lo: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), b[0..16].*));
    const a_hi: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), a[16..32].*));
    const b_hi: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), b[16..32].*));

    const prod_lo = a_lo * b_lo;
    const prod_hi = a_hi * b_hi;
    return @reduce(.Add, prod_lo) + @reduce(.Add, prod_hi);
}

/// Q4_0 weight row × Q8 input blocks → f32 ドット積
/// weight_row: Q4_0 エンコード済み 1 行分
/// q8_input: quantizeRowQ8 で生成した Q8 ブロック列
pub fn dot_q4_0_q8(weight_row: []const u8, q8_input: []const Q8Block) f32 {
    const num_blocks = q8_input.len;
    var sum: f32 = 0;

    for (0..num_blocks) |bi| {
        const block = weight_row[bi * dequant.Q4_0_BPB ..][0..dequant.Q4_0_BPB];
        const w_scale: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[0..2].*)))));

        // ニブルを i8[32] に展開: (nibble - 8) → range -8..7
        var w_i8: [32]i8 = undefined;
        for (0..16) |j| {
            const byte = block[2 + j];
            w_i8[j] = @as(i8, @intCast(@as(i32, byte & 0x0F) - 8));
            w_i8[j + 16] = @as(i8, @intCast(@as(i32, byte >> 4) - 8));
        }

        const int_dot = dot_i8x32(&w_i8, &q8_input[bi].q);
        sum += w_scale * q8_input[bi].scale * @as(f32, @floatFromInt(int_dot));
    }
    return sum;
}

/// Q4_1 weight row × Q8 input blocks → f32 ドット積
/// Q4_1: val = nibble * scale + min
pub fn dot_q4_1_q8(weight_row: []const u8, q8_input: []const Q8Block) f32 {
    const num_blocks = q8_input.len;
    var sum: f32 = 0;

    for (0..num_blocks) |bi| {
        const block = weight_row[bi * dequant.Q4_1_BPB ..][0..dequant.Q4_1_BPB];
        const w_scale: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[0..2].*)))));
        const w_min: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[2..4].*)))));

        // ニブルを i8[32] に展開: nibble → range 0..15 (offset なし)
        var w_i8: [32]i8 = undefined;
        for (0..16) |j| {
            const byte = block[4 + j];
            w_i8[j] = @intCast(byte & 0x0F);
            w_i8[j + 16] = @intCast(byte >> 4);
        }

        const int_dot = dot_i8x32(&w_i8, &q8_input[bi].q);
        const q_scale = q8_input[bi].scale;
        // dot = w_scale * q_scale * int_dot + w_min * q_scale * q_sum
        sum += w_scale * q_scale * @as(f32, @floatFromInt(int_dot)) +
            w_min * q_scale * @as(f32, @floatFromInt(q8_input[bi].sum));
    }
    return sum;
}

/// Q8_0 weight row × Q8 input blocks → f32 ドット積
pub fn dot_q8_0_q8(weight_row: []const u8, q8_input: []const Q8Block) f32 {
    const num_blocks = q8_input.len;
    var sum: f32 = 0;

    for (0..num_blocks) |bi| {
        const block = weight_row[bi * dequant.Q8_0_BPB ..][0..dequant.Q8_0_BPB];
        const w_scale: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @bitCast(block[0..2].*)))));

        // weight の i8[32] を直接参照
        const w_i8: *const [32]i8 = @ptrCast(block[2..34]);

        const int_dot = dot_i8x32(w_i8, &q8_input[bi].q);
        sum += w_scale * q8_input[bi].scale * @as(f32, @floatFromInt(int_dot));
    }
    return sum;
}

// ============================================================
// テスト
// ============================================================

test "quantizeRowQ8 basic" {
    // 32 要素の入力
    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) - 16.0;
    }
    // range: -16 .. 15

    var q8: [1]Q8Block = undefined;
    quantize_row_q8(&input, &q8);

    // scale = 16.0 / 127.0 ≈ 0.126
    try std.testing.expect(q8[0].scale > 0);
    try std.testing.expect(q8[0].scale < 0.2);

    // input[16] = 0 → q should be ~0
    try std.testing.expectEqual(@as(i8, 0), q8[0].q[16]);

    // input[0] = -16 → q should be -127
    try std.testing.expectEqual(@as(i8, -127), q8[0].q[0]);
}

test "quantizeRowQ8 zero input" {
    var input: [32]f32 = undefined;
    @memset(&input, 0);

    var q8: [1]Q8Block = undefined;
    quantize_row_q8(&input, &q8);

    // All zeros
    for (0..32) |i| {
        try std.testing.expectEqual(@as(i8, 0), q8[0].q[i]);
    }
    try std.testing.expectEqual(@as(i32, 0), q8[0].sum);
}

test "dotQ4_0_q8 matches dotQ4_0_f32" {
    // 1 block (32 elements), various nibble values
    var block: [18]u8 = undefined;
    const scale: f16 = 2.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    block[2] = 0x0F; // lo=15, hi=0
    block[3] = 0x17; // lo=7, hi=1
    @memset(block[4..18], 0x99); // rest: (9-8)=1

    // f32 input
    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) * 0.1 - 1.5;
    }

    // f32 版の結果
    const expected = dequant.dot_q4_0_f32(&block, &input, 32);

    // Q8 版
    var q8: [1]Q8Block = undefined;
    quantize_row_q8(&input, &q8);
    const actual = dot_q4_0_q8(&block, &q8);

    // 量子化誤差があるので tolerance を緩めに
    try std.testing.expectApproxEqAbs(expected, actual, @abs(expected) * 0.05 + 0.5);
}

test "dotQ4_0_q8 unit values" {
    // scale=1.0, all nibbles = 9 → (9-8) = 1
    var block: [18]u8 = undefined;
    const scale: f16 = 1.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    @memset(block[2..18], 0x99);

    var input: [32]f32 = undefined;
    @memset(&input, 1.0);

    // f32 reference: each val = 1.0 * 1.0 = 1.0, sum = 32.0
    const expected = dequant.dot_q4_0_f32(&block, &input, 32);

    var q8: [1]Q8Block = undefined;
    quantize_row_q8(&input, &q8);
    const actual = dot_q4_0_q8(&block, &q8);

    try std.testing.expectApproxEqAbs(expected, actual, 0.5);
}

test "dotQ4_1_q8 matches dotQ4_1_f32" {
    var block: [20]u8 = undefined;
    const scale: f16 = 2.0;
    const min_val: f16 = 1.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    block[2..4].* = @bitCast(@as(u16, @bitCast(min_val)));
    @memset(block[4..20], 0x33); // nibbles = 3

    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) * 0.2 - 3.0;
    }

    const expected = dequant.dot_q4_1_f32(&block, &input, 32);

    var q8: [1]Q8Block = undefined;
    quantize_row_q8(&input, &q8);
    const actual = dot_q4_1_q8(&block, &q8);

    try std.testing.expectApproxEqAbs(expected, actual, @abs(expected) * 0.05 + 1.0);
}

test "dotQ8_0_q8 matches dotQ8_0_f32" {
    var block: [34]u8 = undefined;
    const scale: f16 = 0.5;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    block[2] = 4;
    block[3] = @bitCast(@as(i8, -3));
    block[4] = 10;
    @memset(block[5..34], 1);

    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) * 0.3 - 4.0;
    }

    const expected = dequant.dot_q8_0_f32(&block, &input, 32);

    var q8: [1]Q8Block = undefined;
    quantize_row_q8(&input, &q8);
    const actual = dot_q8_0_q8(&block, &q8);

    try std.testing.expectApproxEqAbs(expected, actual, @abs(expected) * 0.05 + 0.5);
}

test "dotQ8_0_q8 unit values" {
    var block: [34]u8 = undefined;
    const scale: f16 = 1.0;
    block[0..2].* = @bitCast(@as(u16, @bitCast(scale)));
    // q[i] = 2 for all i
    @memset(block[2..34], 2);

    var input: [32]f32 = undefined;
    @memset(&input, 1.0);

    // Expected: each val = 2 * 1.0 = 2.0, dot with 1.0 = 64.0
    const expected = dequant.dot_q8_0_f32(&block, &input, 32);

    var q8: [1]Q8Block = undefined;
    quantize_row_q8(&input, &q8);
    const actual = dot_q8_0_q8(&block, &q8);

    try std.testing.expectApproxEqAbs(expected, actual, 1.0);
}

test "multi-block dotQ4_0_q8" {
    // 2 blocks = 64 elements
    var weight: [36]u8 = undefined; // 2 × 18
    const scale1: f16 = 1.0;
    const scale2: f16 = 0.5;
    weight[0..2].* = @bitCast(@as(u16, @bitCast(scale1)));
    @memset(weight[2..18], 0x99);
    weight[18..20].* = @bitCast(@as(u16, @bitCast(scale2)));
    @memset(weight[20..36], 0xAA); // (10-8)=2

    var input: [64]f32 = undefined;
    @memset(&input, 1.0);

    const expected = dequant.dot_q4_0_f32(&weight, &input, 64);

    var q8: [2]Q8Block = undefined;
    quantize_row_q8(&input, &q8);
    const actual = dot_q4_0_q8(&weight, &q8);

    try std.testing.expectApproxEqAbs(expected, actual, 1.0);
}
