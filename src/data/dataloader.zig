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

    pub fn init(allocator: Allocator, n_samples: usize, batch_size: usize, do_shuffle: bool) !BatchIterator {
        const indices = try allocator.alloc(usize, n_samples);
        for (indices, 0..) |*idx, i| idx.* = i;

        var self = BatchIterator{
            .n_samples = n_samples,
            .batch_size = batch_size,
            .indices = indices,
            .current = 0,
            .prng = std.Random.DefaultPrng.init(42),
            .allocator = allocator,
            .shuffle = do_shuffle,
        };

        if (do_shuffle) self.shuffleIndices();
        return self;
    }

    pub fn deinit(self: *BatchIterator) void {
        self.allocator.free(self.indices);
    }

    /// epoch 先頭: re-shuffle + current=0
    pub fn reset(self: *BatchIterator) void {
        self.current = 0;
        if (self.shuffle) self.shuffleIndices();
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
    pub fn numBatches(self: *const BatchIterator) usize {
        if (self.n_samples == 0) return 0;
        return (self.n_samples + self.batch_size - 1) / self.batch_size;
    }

    fn shuffleIndices(self: *BatchIterator) void {
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
    try testing.expectEqual(@as(usize, 4), iter.numBatches()); // ceil(10/3) = 4
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

test "BatchIterator: exact divisible batch size" {
    var iter = try BatchIterator.init(testing.allocator, 8, 4, false);
    defer iter.deinit();

    try testing.expectEqual(@as(usize, 2), iter.numBatches());
    const b1 = iter.next().?;
    try testing.expectEqual(@as(usize, 4), b1.len);
    const b2 = iter.next().?;
    try testing.expectEqual(@as(usize, 4), b2.len);
    try testing.expectEqual(@as(?[]const usize, null), iter.next());
}
