const std = @import("std");
const Allocator = std.mem.Allocator;

/// 計算グラフのノード。自動微分のための逆伝播情報を保持する。
/// T でスカラー型 (f16, f32, f64) をパラメータ化する。
pub fn GraphNode(comptime T: type) type {
    return struct {
        const Self = @This();

        /// このノードの勾配データ (backward時に蓄積される)
        grad: ?[]T,
        /// 勾配データの要素数
        num_elements: usize,

        /// 逆伝播関数。grad_output(このノードの勾配)から親ノードの勾配を計算する。
        backward_fn: ?*const fn (self: *Self) void,

        /// 入力ノードへの参照 (最大3入力: layerNorm等の三項演算用)
        parents: [3]?*Self,

        /// forward時のキャッシュ (backward計算に必要なデータ)
        context: ?*anyopaque,

        /// requires_grad フラグ
        requires_grad: bool,

        /// トポロジカルソート用visited
        visited: bool,

        pub fn init(num_elements: usize, requires_grad: bool) Self {
            return .{
                .grad = null,
                .num_elements = num_elements,
                .backward_fn = null,
                .parents = .{ null, null, null },
                .context = null,
                .requires_grad = requires_grad,
                .visited = false,
            };
        }

        /// 勾配を蓄積 (+=)。初回は確保してゼロ初期化する。
        pub fn accumulateGrad(self: *Self, allocator: Allocator, incoming: []const T) !void {
            if (self.grad == null) {
                self.grad = try allocator.alloc(T, self.num_elements);
                @memset(self.grad.?, 0);
            }
            const g = self.grad.?;
            for (g, incoming) |*dst, src| {
                dst.* += src;
            }
        }

        /// 勾配をゼロにリセット
        pub fn zeroGrad(self: *Self) void {
            if (self.grad) |g| {
                @memset(g, 0);
            }
        }

        /// 勾配メモリを解放
        pub fn deinitGrad(self: *Self, allocator: Allocator) void {
            if (self.grad) |g| {
                allocator.free(g);
                self.grad = null;
            }
        }
    };
}

// ============================================================
// テスト
// ============================================================

test "GraphNode init" {
    const node = GraphNode(f32).init(10, true);
    try std.testing.expect(node.requires_grad);
    try std.testing.expectEqual(@as(usize, 10), node.num_elements);
    try std.testing.expect(node.grad == null);
    try std.testing.expect(node.backward_fn == null);
}

test "GraphNode accumulateGrad" {
    const allocator = std.testing.allocator;
    var node = GraphNode(f32).init(3, true);
    defer node.deinitGrad(allocator);

    const grad1 = [_]f32{ 1, 2, 3 };
    try node.accumulateGrad(allocator, &grad1);

    try std.testing.expectEqual(@as(f32, 1), node.grad.?[0]);
    try std.testing.expectEqual(@as(f32, 2), node.grad.?[1]);
    try std.testing.expectEqual(@as(f32, 3), node.grad.?[2]);

    // 2回目: 累積される
    const grad2 = [_]f32{ 10, 20, 30 };
    try node.accumulateGrad(allocator, &grad2);

    try std.testing.expectEqual(@as(f32, 11), node.grad.?[0]);
    try std.testing.expectEqual(@as(f32, 22), node.grad.?[1]);
    try std.testing.expectEqual(@as(f32, 33), node.grad.?[2]);
}

test "GraphNode zeroGrad" {
    const allocator = std.testing.allocator;
    var node = GraphNode(f32).init(3, true);
    defer node.deinitGrad(allocator);

    const grad = [_]f32{ 1, 2, 3 };
    try node.accumulateGrad(allocator, &grad);
    node.zeroGrad();

    try std.testing.expectEqual(@as(f32, 0), node.grad.?[0]);
    try std.testing.expectEqual(@as(f32, 0), node.grad.?[1]);
    try std.testing.expectEqual(@as(f32, 0), node.grad.?[2]);
}
