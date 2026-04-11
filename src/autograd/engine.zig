const std = @import("std");
const Allocator = std.mem.Allocator;
const GraphNodeMod = @import("../core/graph.zig");

/// 自動微分エンジン。
/// 計算グラフをトポロジカルソートし、逆順にbackwardを実行する。
pub fn GradEngine(comptime T: type) type {
    const Node = GraphNodeMod.GraphNode(T);

    return struct {
        const Self = @This();

        allocator: Allocator,
        topo_order: std.ArrayList(*Node),

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .topo_order = .empty,
            };
        }

        pub fn deinit(self: *Self) void {
            self.topo_order.deinit(self.allocator);
        }

        /// rootノードから逆伝播を実行する。
        /// rootは通常loss (スカラー) で、勾配は1.0で初期化される。
        pub fn backward(self: *Self, root: *Node) !void {
            // 1. rootの勾配を1.0で初期化
            if (root.grad == null) {
                root.grad = try self.allocator.alloc(T, root.num_elements);
            }
            for (root.grad.?) |*g| {
                g.* = 1;
            }

            // 2. トポロジカルソート (rootからDFS)
            self.topo_order.clearRetainingCapacity();
            try self.topoSort(root);

            // 3. 全ノードの勾配バッファを確保（中間ノードも含む）
            for (self.topo_order.items) |node| {
                if (node.grad == null and node.requires_grad) {
                    node.grad = try self.allocator.alloc(T, node.num_elements);
                    @memset(node.grad.?, 0);
                }
            }

            // 4. 逆順に走査して勾配を伝播
            const items = self.topo_order.items;
            var i = items.len;
            while (i > 0) {
                i -= 1;
                const node = items[i];
                if (node.backward_fn) |bfn| {
                    bfn(node);
                }
            }

            // 4. visitedフラグをリセット
            for (items) |node| {
                node.visited = false;
            }
        }

        fn topoSort(self: *Self, node: *Node) !void {
            if (node.visited) return;
            node.visited = true;
            for (node.parents) |maybe_parent| {
                if (maybe_parent) |parent| {
                    try self.topoSort(parent);
                }
            }
            try self.topo_order.append(self.allocator, node);
        }
    };
}

// ============================================================
// テスト
// ============================================================

test "GradEngine backward simple chain" {
    // y = 2 * x を手動で構築してbackwardをテスト
    const allocator = std.testing.allocator;
    const Node = GraphNodeMod.GraphNode(f32);

    // xノード (リーフ)
    var x_node = Node.init(1, true);
    x_node.grad = try allocator.alloc(f32, 1);
    defer allocator.free(x_node.grad.?);
    x_node.grad.?[0] = 0;

    // yノード (y = 2*x, dy/dx = 2)
    var y_node = Node.init(1, true);
    y_node.parents[0] = &x_node;

    const BackwardCtx = struct {
        fn backward(node: *Node) void {
            // dy/dx = 2, 勾配を親に伝播
            if (node.parents[0]) |parent| {
                if (parent.grad) |g| {
                    g[0] += node.grad.?[0] * 2.0;
                }
            }
        }
    };
    y_node.backward_fn = BackwardCtx.backward;

    var engine = GradEngine(f32).init(allocator);
    defer engine.deinit();

    try engine.backward(&y_node);

    // y_nodeの勾配 = 1.0 (root)
    try std.testing.expectEqual(@as(f32, 1.0), y_node.grad.?[0]);
    // x_nodeの勾配 = 2.0 (dy/dx = 2)
    try std.testing.expectEqual(@as(f32, 2.0), x_node.grad.?[0]);

    // clean up
    allocator.free(y_node.grad.?);
}

test "GradEngine backward chain: z = (x + y) * 2" {
    const allocator = std.testing.allocator;
    const Node = GraphNodeMod.GraphNode(f32);

    // x, y: リーフノード
    var x_node = Node.init(1, true);
    x_node.grad = try allocator.alloc(f32, 1);
    defer allocator.free(x_node.grad.?);
    x_node.grad.?[0] = 0;

    var y_node = Node.init(1, true);
    y_node.grad = try allocator.alloc(f32, 1);
    defer allocator.free(y_node.grad.?);
    y_node.grad.?[0] = 0;

    // s = x + y
    var s_node = Node.init(1, true);
    s_node.parents = .{ &x_node, &y_node, null };
    s_node.grad = try allocator.alloc(f32, 1);
    defer allocator.free(s_node.grad.?);
    s_node.grad.?[0] = 0;

    s_node.backward_fn = struct {
        fn backward(node: *Node) void {
            // ds/dx = 1, ds/dy = 1
            inline for (0..2) |pi| {
                if (node.parents[pi]) |parent| {
                    if (parent.grad) |g| {
                        g[0] += node.grad.?[0] * 1.0;
                    }
                }
            }
        }
    }.backward;

    // z = s * 2
    var z_node = Node.init(1, true);
    z_node.parents[0] = &s_node;

    z_node.backward_fn = struct {
        fn backward(node: *Node) void {
            if (node.parents[0]) |parent| {
                if (parent.grad) |g| {
                    g[0] += node.grad.?[0] * 2.0;
                }
            }
        }
    }.backward;

    var engine = GradEngine(f32).init(allocator);
    defer engine.deinit();

    try engine.backward(&z_node);

    // dz/dx = dz/ds * ds/dx = 2 * 1 = 2
    try std.testing.expectEqual(@as(f32, 2.0), x_node.grad.?[0]);
    // dz/dy = dz/ds * ds/dy = 2 * 1 = 2
    try std.testing.expectEqual(@as(f32, 2.0), y_node.grad.?[0]);

    allocator.free(z_node.grad.?);
}
