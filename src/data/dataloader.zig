/// BatchIterator: データ型非依存のバッチインデックスイテレータ
///
/// シャッフル + バッチングのみを担当し、データ本体はユーザーが管理する。
///
/// Usage:
///   var iter = BatchIterator.init(allocator, 100, 32, true);
///   defer iter.deinit();
///   while (iter.next()) |batch_indices| {
///       // batch_indices: []const usize — 最大 batch_size 個のインデックス
///   }
///   iter.reset(); // 次の epoch
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const BatchIterator = struct {
    n_samples: usize,
    batch_size: usize,
    indices: []usize,
    current: usize,
    prng: std.Random.DefaultPrng,
    allocator: Allocator,
    shuffle: bool,

    pub const DEFAULT_SEED: u64 = 42;

    /// デフォルト seed (42) で初期化する薄いラッパ。
    pub fn init(
        allocator: Allocator,
        n_samples: usize,
        batch_size: usize,
        do_shuffle: bool,
    ) !BatchIterator {
        return init_with_seed(allocator, n_samples, batch_size, do_shuffle, DEFAULT_SEED);
    }

    /// seed を明示指定して初期化する。同じ seed なら shuffle 順が bitwise 再現する。
    pub fn init_with_seed(
        allocator: Allocator,
        n_samples: usize,
        batch_size: usize,
        do_shuffle: bool,
        seed: u64,
    ) !BatchIterator {
        const indices = try allocator.alloc(usize, n_samples);
        for (indices, 0..) |*idx, i| idx.* = i;

        var self = BatchIterator{
            .n_samples = n_samples,
            .batch_size = batch_size,
            .indices = indices,
            .current = 0,
            .prng = std.Random.DefaultPrng.init(seed),
            .allocator = allocator,
            .shuffle = do_shuffle,
        };

        if (do_shuffle) self.shuffle_indices();
        return self;
    }

    pub fn deinit(self: *BatchIterator) void {
        self.allocator.free(self.indices);
    }

    /// epoch 先頭: re-shuffle + current=0
    pub fn reset(self: *BatchIterator) void {
        self.current = 0;
        if (self.shuffle) self.shuffle_indices();
    }

    /// 次のバッチインデックスを返す。全バッチ走査済みなら null。
    pub fn next(self: *BatchIterator) ?[]const usize {
        if (self.current >= self.n_samples) return null;
        const end = @min(self.current + self.batch_size, self.n_samples);
        const batch = self.indices[self.current..end];
        self.current = end;
        return batch;
    }

    /// 合計バッチ数 (端数バッチ含む)
    pub fn num_batches(self: *const BatchIterator) usize {
        if (self.n_samples == 0) return 0;
        return (self.n_samples + self.batch_size - 1) / self.batch_size;
    }

    fn shuffle_indices(self: *BatchIterator) void {
        var random = self.prng.random();
        // Fisher-Yates shuffle
        var i: usize = self.n_samples;
        while (i > 1) {
            i -= 1;
            const j = random.intRangeAtMost(usize, 0, i);
            const tmp = self.indices[i];
            self.indices[i] = self.indices[j];
            self.indices[j] = tmp;
        }
    }
};

// ── Tests ──

const testing = std.testing;

test "BatchIterator: basic iteration" {
    var iter = try BatchIterator.init(testing.allocator, 10, 3, false);
    defer iter.deinit();

    // No shuffle: indices should be sequential
    const b1 = iter.next().?;
    try testing.expectEqual(@as(usize, 3), b1.len);
    try testing.expectEqual(@as(usize, 0), b1[0]);
    try testing.expectEqual(@as(usize, 1), b1[1]);
    try testing.expectEqual(@as(usize, 2), b1[2]);

    const b2 = iter.next().?;
    try testing.expectEqual(@as(usize, 3), b2.len);
    try testing.expectEqual(@as(usize, 3), b2[0]);

    const b3 = iter.next().?;
    try testing.expectEqual(@as(usize, 3), b3.len);

    // Last batch: remainder (1 element)
    const b4 = iter.next().?;
    try testing.expectEqual(@as(usize, 1), b4.len);
    try testing.expectEqual(@as(usize, 9), b4[0]);

    // No more batches
    try testing.expectEqual(@as(?[]const usize, null), iter.next());
}

test "BatchIterator: numBatches" {
    var iter = try BatchIterator.init(testing.allocator, 10, 3, false);
    defer iter.deinit();

    try testing.expectEqual(@as(usize, 4), iter.num_batches()); // ceil(10/3) = 4
}

test "BatchIterator: shuffle covers all indices" {
    var iter = try BatchIterator.init(testing.allocator, 20, 5, true);
    defer iter.deinit();

    var seen = [_]bool{false} ** 20;
    var count: usize = 0;
    while (iter.next()) |batch| {
        for (batch) |idx| {
            try testing.expect(idx < 20);
            seen[idx] = true;
            count += 1;
        }
    }
    try testing.expectEqual(@as(usize, 20), count);
    for (seen) |s| try testing.expect(s);
}

test "BatchIterator: reset re-shuffles" {
    var iter = try BatchIterator.init(testing.allocator, 10, 10, true);
    defer iter.deinit();

    const first_batch = iter.next().?;
    var first_order: [10]usize = undefined;
    @memcpy(&first_order, first_batch);

    iter.reset();
    const second_batch = iter.next().?;

    // After reset + re-shuffle, order should (very likely) differ
    var same = true;
    for (0..10) |i| {
        if (first_order[i] != second_batch[i]) {
            same = false;
            break;
        }
    }
    try testing.expect(!same);
}

test "BatchIterator: init_with_seed reproduces shuffle order bitwise" {
    var a = try BatchIterator.init_with_seed(testing.allocator, 64, 8, true, 12345);
    defer a.deinit();

    var b = try BatchIterator.init_with_seed(testing.allocator, 64, 8, true, 12345);
    defer b.deinit();

    // 2 つのイテレータの全インデックス列が一致
    try testing.expectEqualSlices(usize, a.indices, b.indices);

    // next() で取り出したバッチも順に一致
    while (true) {
        const na = a.next();
        const nb = b.next();
        if (na == null and nb == null) break;
        try testing.expect(na != null and nb != null);
        try testing.expectEqualSlices(usize, na.?, nb.?);
    }
}

test "BatchIterator: different seeds give different orders" {
    var a = try BatchIterator.init_with_seed(testing.allocator, 64, 8, true, 1);
    defer a.deinit();

    var b = try BatchIterator.init_with_seed(testing.allocator, 64, 8, true, 2);
    defer b.deinit();

    // 64 要素 shuffle で seed 1 vs 2 が完全一致する確率は 1/64! ≈ 0
    try testing.expect(!std.mem.eql(usize, a.indices, b.indices));
}

test "BatchIterator: multiple seeds — each self-consistent, mutually distinct" {
    const seeds = [_]u64{ 0, 7, 42, 2024, 999_999 };
    const N: usize = 64;

    var orders: [seeds.len][2][N]usize = undefined;

    for (seeds, 0..) |s, i| {
        for (0..2) |k| {
            var iter = try BatchIterator.init_with_seed(testing.allocator, N, 8, true, s);
            defer iter.deinit();

            @memcpy(&orders[i][k], iter.indices);
        }
    }

    // (a) 同じ seed の 2 回は bitwise 一致
    for (0..seeds.len) |i| {
        try testing.expectEqualSlices(usize, &orders[i][0], &orders[i][1]);
    }
    // (b) 異なる seed の組み合わせでは不一致 (64! 通り中 1 つに当たる確率はゼロとみなす)
    for (0..seeds.len) |i| {
        for (i + 1..seeds.len) |j| {
            try testing.expect(!std.mem.eql(usize, &orders[i][0], &orders[j][0]));
        }
    }
}

test "BatchIterator: reset keeps seed determinism across epochs" {
    var a = try BatchIterator.init_with_seed(testing.allocator, 32, 8, true, 7);
    defer a.deinit();

    var b = try BatchIterator.init_with_seed(testing.allocator, 32, 8, true, 7);
    defer b.deinit();

    // 1 epoch 回す
    while (a.next()) |_| {}
    while (b.next()) |_| {}

    a.reset();
    b.reset();

    // 2 epoch 目も順序一致 (reset 後の再 shuffle も決定論的)
    try testing.expectEqualSlices(usize, a.indices, b.indices);
}

test "BatchIterator: exact divisible batch size" {
    var iter = try BatchIterator.init(testing.allocator, 8, 4, false);
    defer iter.deinit();

    try testing.expectEqual(@as(usize, 2), iter.num_batches());
    const b1 = iter.next().?;
    try testing.expectEqual(@as(usize, 4), b1.len);
    const b2 = iter.next().?;
    try testing.expectEqual(@as(usize, 4), b2.len);
    try testing.expectEqual(@as(?[]const usize, null), iter.next());
}
