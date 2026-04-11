/// GPU Variable: Metal GPU バッファ付きの Variable ラッパー
///
/// UMA (Unified Memory Architecture) を活用し、MTLBuffer の contents() ポインタを
/// tensor data / grad として使う。CPU/GPU 両方からアクセス可能。
///
/// 既存の Variable/GraphNode/GradEngine と互換性を持つ。
/// backward_fn 内で context 経由で MetalContext にアクセスできる。
const std = @import("std");
const Allocator = std.mem.Allocator;
const GraphNodeMod = @import("../core/graph.zig");
const metal = @import("../backend/metal.zig");

const id = metal.id;

/// GPU 上にバッファを持つ Variable
/// T: スカラー型 (f32), shape: コンパイル時形状
pub fn GpuVariable(comptime T: type, comptime shape: anytype) type {
    comptime {
        var n_elem: usize = 1;
        for (0..shape.len) |i| {
            n_elem *= shape[i];
        }

        return GpuVariableInner(T, n_elem);
    }
}

fn GpuVariableInner(comptime T: type, comptime n: usize) type {
    const Node = GraphNodeMod.GraphNode(T);

    return struct {
        const Self = @This();
        pub const num_elements = n;

        data: [*]T,           // UMA ポインタ (MTLBuffer contents)
        data_buf: id,         // MTLBuffer for data
        grad_buf: ?id,        // MTLBuffer for grad (null = 未確保)
        node: *Node,          // 計算グラフノード
        metal_ctx: *metal.MetalContext,
        allocator: Allocator, // node 確保用

        /// GPU バッファ付き Variable を作成 (ゼロ初期化)
        pub fn init(mtl: *metal.MetalContext, allocator: Allocator, requires_grad: bool) !Self {
            const buf_size = n * @sizeOf(T);
            const data_buf = try mtl.createBuffer(buf_size);
            const data_ptr = metal.MetalContext.bufferContents(T, data_buf);
            @memset(data_ptr[0..n], 0);

            const node = try allocator.create(Node);
            node.* = Node.init(n, requires_grad);

            return .{
                .data = data_ptr,
                .data_buf = data_buf,
                .grad_buf = null,
                .node = node,
                .metal_ctx = mtl,
                .allocator = allocator,
            };
        }

        /// 既存データからコピーして初期化
        pub fn fromSlice(mtl: *metal.MetalContext, allocator: Allocator, src: []const T, requires_grad: bool) !Self {
            var self = try init(mtl, allocator, requires_grad);
            @memcpy(self.data[0..n], src[0..n]);
            return self;
        }

        /// Xavier 初期化 (Linear 用)
        pub fn xavierInit(mtl: *metal.MetalContext, allocator: Allocator, fan_in: usize, fan_out: usize, rng: std.Random) !Self {
            var self = try init(mtl, allocator, true);
            const scale: T = @sqrt(2.0 / @as(T, @floatFromInt(fan_in + fan_out)));
            for (self.data[0..n]) |*d| {
                d.* = (rng.float(T) * 2.0 - 1.0) * scale;
            }
            return self;
        }

        /// 勾配バッファを確保 (GPU MTLBuffer + node.grad に UMA ポインタ設定)
        pub fn allocGrad(self: *Self) !void {
            if (self.grad_buf != null) return;
            const grad_buf_size = n * @sizeOf(T);
            const buf = try self.metal_ctx.createBuffer(grad_buf_size);
            const grad_ptr = metal.MetalContext.bufferContents(T, buf);
            @memset(grad_ptr[0..n], 0);
            self.grad_buf = buf;
            self.node.grad = grad_ptr[0..n];
        }

        /// データスライス取得
        pub fn dataSlice(self: Self) []T {
            return self.data[0..n];
        }

        pub fn constDataSlice(self: Self) []const T {
            return self.data[0..n];
        }

        pub fn zeroGrad(self: *Self) void {
            self.node.zeroGrad();
        }

        pub fn deinit(self: *Self) void {
            // MTLBuffer を release
            metal.objRelease(self.data_buf);
            if (self.grad_buf) |gb| {
                metal.objRelease(gb);
            }
            // node.grad は MTLBuffer 経由なので free しない → null にして destroy
            self.node.grad = null;
            self.allocator.destroy(self.node);
        }
    };
}
