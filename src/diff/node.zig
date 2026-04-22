/// diff/node.zig: バックエンド非依存の計算グラフノード型
///
/// DiffCpuRuntime と DiffCudaRuntime で共通の計算グラフノードを提供する。
/// DataPtr を comptime パラメータ化し、CPU ([]f32) と CUDA (CUdeviceptr) で共有。
const runtime_kernels = @import("../runtime_kernels.zig");

pub const CPU_MAX_NDIM = runtime_kernels.MAX_NDIM;

pub fn DiffNodeGeneric(comptime DataPtr: type, comptime max_ndim: usize) type {
    return struct {
        const Self = @This();

        data: DataPtr,
        shape: [max_ndim]usize,
        ndim: usize,
        grad: ?DataPtr,
        backward_fn: ?*const fn (*Self) void,
        parents: [3]?*Self,
        context: ?*anyopaque,
        requires_grad: bool,
        visited: bool,
        is_param: bool,
        param_index: ?usize,

        pub fn totalElements(self: *const Self) usize {
            var size: usize = 1;
            for (0..self.ndim) |i| size *= self.shape[i];
            return size;
        }

        pub fn lastDim(self: *const Self) usize {
            return self.shape[self.ndim - 1];
        }

        pub fn numRows(self: *const Self) usize {
            return self.totalElements() / self.lastDim();
        }
    };
}

// ── Autograd graph utilities ──

const std = @import("std");
const Allocator = std.mem.Allocator;

/// トポロジカルソート (DFS) — バックエンド非依存
pub fn topoSort(
    comptime Node: type,
    node: *Node,
    buf: *std.ArrayListUnmanaged(*Node),
    allocator: Allocator,
) void {
    if (node.visited) return;
    node.visited = true;

    for (&node.parents) |maybe_parent| {
        if (maybe_parent) |parent| {
            topoSort(Node, parent, buf, allocator);
        }
    }

    buf.append(allocator, node) catch unreachable;
}

/// 逆伝播のステップ 2,4,5 を共通化。
/// ステップ 1 (loss 勾配初期化) と 3 (中間ノードの勾配確保) はバックエンド依存のため
/// 呼び出し側で行う。
pub fn backwardPass(
    comptime Node: type,
    topo_buf: *std.ArrayListUnmanaged(*Node),
    param_nodes: []Node,
) void {
    // 4. Reverse traversal: call backward_fn
    var idx = topo_buf.items.len;
    while (idx > 0) {
        idx -= 1;
        const node = topo_buf.items[idx];
        if (node.backward_fn) |bfn| {
            bfn(node);
        }
    }

    // 5. Reset visited flags
    for (topo_buf.items) |node| {
        node.visited = false;
    }
    for (param_nodes) |*node| {
        node.visited = false;
    }
}

// ── Tests ──

const testing = std.testing;

test "DiffNodeGeneric: CPU型 ([]f32) のインスタンス化と基本メソッド" {
    const CpuNode = DiffNodeGeneric([]f32, 4);
    var node: CpuNode = .{
        .data = &.{},
        .shape = .{ 2, 3, 4, 1 },
        .ndim = 3,
        .grad = null,
        .backward_fn = null,
        .parents = .{ null, null, null },
        .context = null,
        .requires_grad = false,
        .visited = false,
        .is_param = false,
        .param_index = null,
    };
    try testing.expectEqual(@as(usize, 24), node.totalElements());
    try testing.expectEqual(@as(usize, 4), node.lastDim());
    try testing.expectEqual(@as(usize, 6), node.numRows());
}

test "DiffNodeGeneric: CUDA型 (u64) のインスタンス化と基本メソッド" {
    // CUdeviceptr は実質 u64
    const GpuNode = DiffNodeGeneric(u64, 4);
    var node: GpuNode = .{
        .data = 0,
        .shape = .{ 5, 10, 1, 1 },
        .ndim = 2,
        .grad = null,
        .backward_fn = null,
        .parents = .{ null, null, null },
        .context = null,
        .requires_grad = true,
        .visited = false,
        .is_param = true,
        .param_index = 42,
    };
    try testing.expectEqual(@as(usize, 50), node.totalElements());
    try testing.expectEqual(@as(usize, 10), node.lastDim());
    try testing.expectEqual(@as(usize, 5), node.numRows());
}

test "DiffNodeGeneric: backward_fn と parents の接続" {
    const CpuNode = DiffNodeGeneric([]f32, 4);

    var parent_node: CpuNode = .{
        .data = &.{},
        .shape = .{ 3, 1, 1, 1 },
        .ndim = 1,
        .grad = null,
        .backward_fn = null,
        .parents = .{ null, null, null },
        .context = null,
        .requires_grad = true,
        .visited = false,
        .is_param = false,
        .param_index = null,
    };

    const bwd = struct {
        fn backward(self: *CpuNode) void {
            self.visited = true;
        }
    }.backward;

    var child_node: CpuNode = .{
        .data = &.{},
        .shape = .{ 3, 1, 1, 1 },
        .ndim = 1,
        .grad = null,
        .backward_fn = bwd,
        .parents = .{ &parent_node, null, null },
        .context = null,
        .requires_grad = true,
        .visited = false,
        .is_param = false,
        .param_index = null,
    };

    // backward_fn を呼ぶとノードに作用できる
    child_node.backward_fn.?(&child_node);
    try testing.expect(child_node.visited);
    // parents の参照が正しい
    try testing.expectEqual(&parent_node, child_node.parents[0].?);
}

test "DiffNodeGeneric: 1次元 shape" {
    const Node = DiffNodeGeneric([]f32, 4);
    var node: Node = .{
        .data = &.{},
        .shape = .{ 7, 1, 1, 1 },
        .ndim = 1,
        .grad = null,
        .backward_fn = null,
        .parents = .{ null, null, null },
        .context = null,
        .requires_grad = false,
        .visited = false,
        .is_param = false,
        .param_index = null,
    };
    try testing.expectEqual(@as(usize, 7), node.totalElements());
    try testing.expectEqual(@as(usize, 7), node.lastDim());
    try testing.expectEqual(@as(usize, 1), node.numRows());
}

test "DiffNodeGeneric: 4次元 shape" {
    const Node = DiffNodeGeneric(u64, 4);
    var node: Node = .{
        .data = 0,
        .shape = .{ 2, 3, 4, 5 },
        .ndim = 4,
        .grad = null,
        .backward_fn = null,
        .parents = .{ null, null, null },
        .context = null,
        .requires_grad = false,
        .visited = false,
        .is_param = false,
        .param_index = null,
    };
    try testing.expectEqual(@as(usize, 120), node.totalElements());
    try testing.expectEqual(@as(usize, 5), node.lastDim());
    try testing.expectEqual(@as(usize, 24), node.numRows());
}

// ── topoSort / backwardPass tests ──

fn makeTestNode(comptime Node: type, rg: bool) Node {
    return .{
        .data = &.{},
        .shape = .{ 1, 1, 1, 1 },
        .ndim = 1,
        .grad = null,
        .backward_fn = null,
        .parents = .{ null, null, null },
        .context = null,
        .requires_grad = rg,
        .visited = false,
        .is_param = false,
        .param_index = null,
    };
}

test "topoSort: 線形グラフ (A → B → C) の順序" {
    const Node = DiffNodeGeneric([]f32, 4);
    var a = makeTestNode(Node, true);
    var b = makeTestNode(Node, true);
    b.parents[0] = &a;
    var c = makeTestNode(Node, true);
    c.parents[0] = &b;

    var buf: std.ArrayListUnmanaged(*Node) = .empty;
    defer buf.deinit(testing.allocator);

    topoSort(Node, &c, &buf, testing.allocator);

    // トポロジカル順: a, b, c
    try testing.expectEqual(@as(usize, 3), buf.items.len);
    try testing.expectEqual(&a, buf.items[0]);
    try testing.expectEqual(&b, buf.items[1]);
    try testing.expectEqual(&c, buf.items[2]);
}

test "topoSort: ダイヤモンドグラフ (A → B, A → C, B+C → D)" {
    const Node = DiffNodeGeneric([]f32, 4);
    var a = makeTestNode(Node, true);
    var b = makeTestNode(Node, true);
    b.parents[0] = &a;
    var c = makeTestNode(Node, true);
    c.parents[0] = &a;
    var d = makeTestNode(Node, true);
    d.parents[0] = &b;
    d.parents[1] = &c;

    var buf: std.ArrayListUnmanaged(*Node) = .empty;
    defer buf.deinit(testing.allocator);

    topoSort(Node, &d, &buf, testing.allocator);

    // A は 1回だけ出現する
    try testing.expectEqual(@as(usize, 4), buf.items.len);
    try testing.expectEqual(&a, buf.items[0]);
    // D は末尾
    try testing.expectEqual(&d, buf.items[3]);
}

test "topoSort: visited フラグが設定される" {
    const Node = DiffNodeGeneric([]f32, 4);
    var a = makeTestNode(Node, true);
    var b = makeTestNode(Node, true);
    b.parents[0] = &a;

    var buf: std.ArrayListUnmanaged(*Node) = .empty;
    defer buf.deinit(testing.allocator);

    topoSort(Node, &b, &buf, testing.allocator);

    try testing.expect(a.visited);
    try testing.expect(b.visited);
}

test "backwardPass: backward_fn の逆順呼び出しと visited リセット" {
    const Node = DiffNodeGeneric([]f32, 4);

    // 呼び出し順を記録するためのカウンター
    var call_order: [3]usize = .{ 0, 0, 0 };
    var call_count: usize = 0;

    // backward_fn からカウンターにアクセスするために context を使う
    const Ctx = struct {
        order: *[3]usize,
        count: *usize,
        index: usize,
    };

    const bwd = struct {
        fn backward(self: *Node) void {
            const ctx: *Ctx = @ptrCast(@alignCast(self.context.?));
            ctx.order[ctx.count.*] = ctx.index;
            ctx.count.* += 1;
        }
    }.backward;

    var ctx0 = Ctx{ .order = &call_order, .count = &call_count, .index = 0 };
    var ctx1 = Ctx{ .order = &call_order, .count = &call_count, .index = 1 };
    var ctx2 = Ctx{ .order = &call_order, .count = &call_count, .index = 2 };

    var nodes: [3]Node = .{
        makeTestNode(Node, true),
        makeTestNode(Node, true),
        makeTestNode(Node, true),
    };
    nodes[0].backward_fn = bwd;
    nodes[0].context = @ptrCast(&ctx0);
    nodes[0].visited = true;
    nodes[1].backward_fn = bwd;
    nodes[1].context = @ptrCast(&ctx1);
    nodes[1].visited = true;
    nodes[2].backward_fn = bwd;
    nodes[2].context = @ptrCast(&ctx2);
    nodes[2].visited = true;

    // topo_buf = [0, 1, 2] — backwardPass は逆順 (2, 1, 0) で呼ぶ
    var buf: std.ArrayListUnmanaged(*Node) = .empty;
    defer buf.deinit(testing.allocator);

    buf.append(testing.allocator, &nodes[0]) catch unreachable;
    buf.append(testing.allocator, &nodes[1]) catch unreachable;
    buf.append(testing.allocator, &nodes[2]) catch unreachable;

    var param_nodes: [0]Node = .{};
    backwardPass(Node, &buf, &param_nodes);

    // 逆順: 2, 1, 0
    try testing.expectEqual(@as(usize, 2), call_order[0]);
    try testing.expectEqual(@as(usize, 1), call_order[1]);
    try testing.expectEqual(@as(usize, 0), call_order[2]);

    // visited がリセットされている
    for (&nodes) |*n| {
        try testing.expect(!n.visited);
    }
}

test "backwardPass: param_nodes の visited もリセットされる" {
    const Node = DiffNodeGeneric([]f32, 4);
    var param = makeTestNode(Node, true);
    param.is_param = true;
    param.visited = true;

    var buf: std.ArrayListUnmanaged(*Node) = .empty;
    defer buf.deinit(testing.allocator);

    var param_nodes = [_]Node{param};
    backwardPass(Node, &buf, &param_nodes);

    try testing.expect(!param_nodes[0].visited);
}
