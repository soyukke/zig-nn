/// GraphModule: MPSGraph パラメータ自動管理 (レガシー互換ラッパー)
///
/// 新しい統一モジュールは compute.Module + MpsRuntime を使用する。
/// このファイルはレガシー互換のために残す。
const std = @import("std");
const Allocator = std.mem.Allocator;
const metal_mod = @import("../backend/metal.zig");
const MetalContext = metal_mod.MetalContext;
const metal_id = metal_mod.id;
const mps_graph = @import("../backend/mps_graph.zig");
const MPSGraphContext = mps_graph.MPSGraphContext;
const MPSDataTypeFloat32 = mps_graph.MPSDataTypeFloat32;

// Re-export from compute.zig
const compute = @import("../compute.zig");
pub const ParamInit = compute.ParamInit;
pub const AdamState = compute.AdamState;

pub const GraphParam = struct {
    tensor: metal_id,
    shape: []const usize,
    init_kind: ParamInit,
};

pub const GraphModule = struct {
    ctx: *MPSGraphContext,
    allocator: Allocator,
    params: std.ArrayListUnmanaged(GraphParam),

    pub fn init(ctx: *MPSGraphContext, allocator: Allocator) GraphModule {
        return .{
            .ctx = ctx,
            .allocator = allocator,
            .params = .empty,
        };
    }

    pub fn deinit(self: *GraphModule) void {
        self.params.deinit(self.allocator);
    }

    /// パラメータを登録: placeholder 作成 + shape 記録
    /// 統一レイヤー互換: ParamHandle (usize) を返す
    pub fn addParam(self: *GraphModule, shape: []const usize, init_kind: ParamInit) compute.ParamHandle {
        const tensor = self.ctx.placeholder(shape, MPSDataTypeFloat32);
        const gp = GraphParam{
            .tensor = tensor,
            .shape = shape,
            .init_kind = init_kind,
        };
        self.params.append(self.allocator, gp) catch unreachable;
        return .{ .index = self.params.items.len - 1 };
    }

    /// ParamHandle → MPSGraph placeholder tensor
    pub fn param(self: *GraphModule, handle: compute.ParamHandle) metal_id {
        return self.params.items[handle.index].tensor;
    }

    // --- MPSGraphContext proxy methods (統一レイヤーの forward で使用) ---

    pub fn matmul(self: *GraphModule, a: metal_id, b: metal_id) metal_id {
        return self.ctx.matmul(a, b);
    }
    pub fn add(self: *GraphModule, a: metal_id, b: metal_id) metal_id {
        return self.ctx.add(a, b);
    }
    pub fn mul(self: *GraphModule, a: metal_id, b: metal_id) metal_id {
        return self.ctx.mul(a, b);
    }
    pub fn sub(self: *GraphModule, a: metal_id, b: metal_id) metal_id {
        return self.ctx.sub(a, b);
    }
    pub fn tanh_(self: *GraphModule, x: metal_id) metal_id {
        return self.ctx.tanh_(x);
    }
    pub fn sigmoid(self: *GraphModule, x: metal_id) metal_id {
        return self.ctx.sigmoid(x);
    }
    pub fn gelu(self: *GraphModule, x: metal_id) metal_id {
        return self.ctx.gelu(x);
    }
    pub fn silu(self: *GraphModule, x: metal_id) metal_id {
        return self.ctx.silu(x);
    }
    pub fn softmax(self: *GraphModule, x: metal_id, axis: i64) metal_id {
        return self.ctx.softmax(x, axis);
    }
    pub fn logSoftmax(self: *GraphModule, x: metal_id, axis: i64) metal_id {
        return self.ctx.logSoftmax(x, axis);
    }
    pub fn reshape(self: *GraphModule, x: metal_id, shape: []const usize) metal_id {
        return self.ctx.reshape(x, shape);
    }
    pub fn transpose(self: *GraphModule, x: metal_id, dim1: u64, dim2: u64) metal_id {
        return self.ctx.transpose(x, dim1, dim2);
    }
    pub fn constantScalar(self: *GraphModule, val: f64, dtype: u32) metal_id {
        return self.ctx.constantScalar(val, dtype);
    }
    pub fn constantData(self: *GraphModule, data: [*]const u8, len: usize, shape: []const usize, dtype: u32) metal_id {
        return self.ctx.constantData(data, len, shape, dtype);
    }
    pub fn square(self: *GraphModule, x: metal_id) metal_id {
        return self.ctx.square(x);
    }
    pub fn negative(self: *GraphModule, x: metal_id) metal_id {
        return self.ctx.negative(x);
    }
    pub fn reductionMean(self: *GraphModule, x: metal_id, axes: []const i64) metal_id {
        return self.ctx.reductionMean(x, axes);
    }
    pub fn reductionSum(self: *GraphModule, x: metal_id, axis: i64) metal_id {
        return self.ctx.reductionSum(x, axis);
    }
    pub fn layerNorm(self: *GraphModule, x: metal_id, gamma: metal_id, beta: metal_id, eps: f32, axis: i64) metal_id {
        return self.ctx.layerNorm(x, gamma, beta, eps, axis);
    }
    pub fn placeholder(self: *GraphModule, shape: []const usize, dtype: u32) metal_id {
        return self.ctx.placeholder(shape, dtype);
    }
    pub fn gradients(self: *GraphModule, loss: metal_id, params: []const metal_id) []metal_id {
        return self.ctx.gradients(loss, params);
    }

    /// 全パラメータの placeholder 配列
    pub fn allParamTensors(self: *GraphModule) ![]metal_id {
        const result = try self.allocator.alloc(metal_id, self.params.items.len);
        for (self.params.items, 0..) |p, i| {
            result[i] = p.tensor;
        }
        return result;
    }

    /// パラメータサイズ配列 (要素数)
    pub fn paramSizes(self: *GraphModule) ![]usize {
        const result = try self.allocator.alloc(usize, self.params.items.len);
        for (self.params.items, 0..) |p, i| {
            var size: usize = 1;
            for (p.shape) |d| size *= d;
            result[i] = size;
        }
        return result;
    }

    /// MTLBuffer 確保 + 初期化 (Xavier/zeros/ones)
    pub fn initParamBuffers(self: *GraphModule, mtl: *MetalContext) ![]metal_id {
        const count = self.params.items.len;
        const bufs = try self.allocator.alloc(metal_id, count);

        var rng_state = std.Random.DefaultPrng.init(42);
        const rng = rng_state.random();

        for (self.params.items, 0..) |p, i| {
            var size: usize = 1;
            for (p.shape) |d| size *= d;

            const buf = try mtl.createBuffer(size * @sizeOf(f32));
            bufs[i] = buf;

            const ptr = MetalContext.bufferContents(f32, buf);
            const data = ptr[0..size];

            switch (p.init_kind) {
                .ones => @memset(data, 1.0),
                .zeros => @memset(data, 0.0),
                .xavier => {
                    const fan_in: f32 = @floatFromInt(p.shape[0]);
                    const scale = @sqrt(1.0 / fan_in);
                    for (data) |*val| {
                        val.* = (rng.float(f32) * 2.0 - 1.0) * scale;
                    }
                },
                .kaiming => {
                    const fan_in: f32 = @floatFromInt(p.shape[0]);
                    const scale = @sqrt(2.0 / fan_in);
                    for (data) |*val| {
                        val.* = rng.floatNorm(f32) * scale;
                    }
                },
                .kaiming_fan => |fi| {
                    const fan_in: f32 = @floatFromInt(fi);
                    const scale = @sqrt(2.0 / fan_in);
                    for (data) |*val| {
                        val.* = rng.floatNorm(f32) * scale;
                    }
                },
                .normal => |cfg| {
                    for (data) |*val| {
                        val.* = rng.floatNorm(f32) * cfg.std_dev + cfg.mean;
                    }
                },
            }
        }
        return bufs;
    }

    /// Adam state 確保
    pub fn initAdamState(self: *GraphModule) !AdamState {
        const sizes = try self.paramSizes();
        defer self.allocator.free(sizes);
        return AdamState.init(self.allocator, sizes);
    }

    /// feeds 構築ヘルパー (param feeds のみ)
    pub fn buildParamFeeds(self: *GraphModule, param_bufs: []const metal_id) ![]MPSGraphContext.Feed {
        const count = self.params.items.len;
        const feeds = try self.allocator.alloc(MPSGraphContext.Feed, count);
        for (self.params.items, 0..) |p, i| {
            feeds[i] = .{
                .tensor = p.tensor,
                .buffer = param_bufs[i],
                .shape = p.shape,
                .dtype = MPSDataTypeFloat32,
            };
        }
        return feeds;
    }

    /// Gradient buffers 確保
    pub fn initGradBuffers(self: *GraphModule, mtl: *MetalContext) ![]metal_id {
        const count = self.params.items.len;
        const bufs = try self.allocator.alloc(metal_id, count);
        for (self.params.items, 0..) |p, i| {
            var size: usize = 1;
            for (p.shape) |d| size *= d;
            bufs[i] = try mtl.createBuffer(size * @sizeOf(f32));
        }
        return bufs;
    }

    /// gradient 読み出し + Adam 適用
    pub fn applyAdam(
        self: *GraphModule,
        results: []const metal_id,
        grad_offset: usize,
        param_bufs: []metal_id,
        adam: *AdamState,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) !void {
        adam.step += 1;
        const count = self.params.items.len;

        for (0..count) |i| {
            var size: usize = 1;
            for (self.params.items[i].shape) |d| size *= d;

            const grad_data = try self.allocator.alloc(f32, size);
            defer self.allocator.free(grad_data);

            MPSGraphContext.readTensorData(results[grad_offset + i], grad_data);

            const param_data = MetalContext.bufferContents(f32, param_bufs[i]);
            compute.adamStep(param_data[0..size], grad_data, adam.m[i], adam.v[i], lr, beta1, beta2, eps, wd, adam.step);
        }
    }
};

const CHECKPOINT_MAGIC: u32 = 0x4D504752; // "MPGR"
const CHECKPOINT_VERSION: u32 = 1;

/// GraphModule のパラメータ + Adam state をファイルに保存
pub fn saveCheckpoint(
    module: *const GraphModule,
    param_bufs: []const metal_id,
    adam: *const AdamState,
    path: []const u8,
) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    const count: u32 = @intCast(module.params.items.len);

    // Header
    try file.writeAll(std.mem.asBytes(&std.mem.nativeToLittle(u32, CHECKPOINT_MAGIC)));
    try file.writeAll(std.mem.asBytes(&std.mem.nativeToLittle(u32, CHECKPOINT_VERSION)));
    try file.writeAll(std.mem.asBytes(&std.mem.nativeToLittle(u32, count)));
    try file.writeAll(std.mem.asBytes(&std.mem.nativeToLittle(u32, adam.step)));

    // Each param: num_elements, weight_data, adam_m, adam_v
    for (module.params.items, 0..) |p, i| {
        var size: usize = 1;
        for (p.shape) |d| size *= d;

        const n: u32 = @intCast(size);
        try file.writeAll(std.mem.asBytes(&std.mem.nativeToLittle(u32, n)));

        const ptr = MetalContext.bufferContents(f32, param_bufs[i]);
        try file.writeAll(std.mem.sliceAsBytes(ptr[0..size]));
        try file.writeAll(std.mem.sliceAsBytes(adam.m[i]));
        try file.writeAll(std.mem.sliceAsBytes(adam.v[i]));
    }
}

/// ファイルから GraphModule のパラメータ + Adam state を復元
pub fn loadCheckpoint(
    module: *const GraphModule,
    param_bufs: []metal_id,
    adam: *AdamState,
    path: []const u8,
) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const readExact = struct {
        fn read(f: std.fs.File, out: []u8) !void {
            var total: usize = 0;
            while (total < out.len) {
                const n = try f.read(out[total..]);
                if (n == 0) return error.EndOfStream;
                total += n;
            }
        }
    }.read;

    const readU32 = struct {
        fn read(f: std.fs.File) !u32 {
            var bytes: [4]u8 = undefined;
            var total: usize = 0;
            while (total < 4) {
                const n = try f.read(bytes[total..]);
                if (n == 0) return error.EndOfStream;
                total += n;
            }
            return std.mem.littleToNative(u32, @bitCast(bytes));
        }
    }.read;

    // Header
    const magic = try readU32(file);
    if (magic != CHECKPOINT_MAGIC) return error.InvalidCheckpoint;
    const version = try readU32(file);
    if (version != CHECKPOINT_VERSION) return error.UnsupportedVersion;
    const num_params = try readU32(file);
    if (num_params != module.params.items.len) return error.ParamCountMismatch;
    adam.step = try readU32(file);

    // Each param
    for (module.params.items, 0..) |p, i| {
        var size: usize = 1;
        for (p.shape) |d| size *= d;

        const n = try readU32(file);
        if (n != size) return error.ParamSizeMismatch;

        const ptr = MetalContext.bufferContents(f32, param_bufs[i]);
        try readExact(file, std.mem.sliceAsBytes(ptr[0..size]));
        try readExact(file, std.mem.sliceAsBytes(adam.m[i]));
        try readExact(file, std.mem.sliceAsBytes(adam.v[i]));
    }
}
