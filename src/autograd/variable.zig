const std = @import("std");
const Allocator = std.mem.Allocator;
const TensorMod = @import("../core/tensor.zig");
const GraphNodeMod = @import("../core/graph.zig");

/// tupleを配列に変換
fn tupleToArray(comptime tuple: anytype) [tuple.len]usize {
    comptime {
        var arr: [tuple.len]usize = undefined;
        for (0..tuple.len) |i| {
            arr[i] = tuple[i];
        }
        return arr;
    }
}

/// 勾配追跡付きテンソル (tuple API)。
pub fn Variable(comptime T: type, comptime shape: anytype) type {
    return VariableNd(T, shape.len, tupleToArray(shape));
}

/// 勾配追跡付きテンソル（[N]usize配列ベースの内部実装）。
/// 型の同一性が保証される。
pub fn VariableNd(comptime T: type, comptime N: usize, comptime shape_arr: [N]usize) type {
    const Tens = TensorMod.TensorNd(T, N, shape_arr);
    const Node = GraphNodeMod.GraphNode(T);

    return struct {
        const Self = @This();

        pub const Scalar = T;
        pub const tensor_shape = shape_arr;
        pub const num_elements = Tens.num_elements;
        pub const num_dims = N;

        tensor: Tens,
        node: *Node,
        owns_node: bool,
        allocator: Allocator,

        pub fn init(tensor: Tens, allocator: Allocator, requires_grad: bool) !Self {
            const node = try allocator.create(Node);
            node.* = Node.init(num_elements, requires_grad);
            return .{
                .tensor = tensor,
                .node = node,
                .owns_node = true,
                .allocator = allocator,
            };
        }

        pub fn fromSlice(allocator: Allocator, src: []const T, requires_grad: bool) !Self {
            const tensor = try Tens.fromSlice(allocator, src);
            return init(tensor, allocator, requires_grad);
        }

        pub fn zeros(allocator: Allocator, requires_grad: bool) !Self {
            const tensor = try Tens.zeros(allocator);
            return init(tensor, allocator, requires_grad);
        }

        pub fn deinit(self: *Self) void {
            self.tensor.deinit();
            if (self.owns_node) {
                self.node.deinitGrad(self.allocator);
                self.allocator.destroy(self.node);
            }
        }

        pub fn grad(self: Self) ?[]const T {
            return self.node.grad;
        }

        pub fn zeroGrad(self: *Self) void {
            self.node.zeroGrad();
        }

        pub fn data(self: Self) []T {
            return self.tensor.slice();
        }

        pub fn constData(self: Self) []const T {
            return self.tensor.constSlice();
        }
    };
}

// ============================================================
// テスト
// ============================================================

test "Variable init and deinit" {
    const allocator = std.testing.allocator;
    const Tens = TensorMod.Tensor(f32, .{ 2, 3 });
    const tensor = try Tens.fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 });

    var v = try Variable(f32, .{ 2, 3 }).init(tensor, allocator, true);
    defer v.deinit();

    try std.testing.expect(v.node.requires_grad);
    try std.testing.expectEqual(@as(usize, 6), Variable(f32, .{ 2, 3 }).num_elements);
    try std.testing.expectEqual(@as(f32, 1), v.constData()[0]);
}

test "Variable fromSlice" {
    const allocator = std.testing.allocator;

    var v = try Variable(f32, .{ 2, 3 }).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, true);
    defer v.deinit();

    try std.testing.expectEqual(@as(f32, 6), v.constData()[5]);
}

test "Variable grad" {
    const allocator = std.testing.allocator;

    var v = try Variable(f32, .{3}).fromSlice(allocator, &.{ 1, 2, 3 }, true);
    defer v.deinit();

    try std.testing.expect(v.grad() == null);

    try v.node.accumulateGrad(allocator, &.{ 0.1, 0.2, 0.3 });
    const g = v.grad().?;
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), g[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), g[1], 1e-6);
}
