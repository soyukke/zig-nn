const std = @import("std");
const Allocator = std.mem.Allocator;
const AdamMod = @import("../optim/adam.zig");

fn isStruct(comptime F: type) bool {
    return @typeInfo(F) == .@"struct";
}

/// Variable (trainable parameter) かどうかを判定
fn isVariable(comptime F: type) bool {
    if (!isStruct(F)) return false;
    return @hasDecl(F, "Scalar") and @hasField(F, "node") and @hasDecl(F, "zeroGrad");
}

/// サブ層かどうかを判定 (deinit/allocGrad/zeroGrad を持つが Variable ではない)
fn isSubModule(comptime F: type) bool {
    if (!isStruct(F)) return false;
    if (isVariable(F)) return false;
    return @hasDecl(F, "deinit") and @hasDecl(F, "allocGrad") and @hasDecl(F, "zeroGrad");
}

/// 固定長配列で要素がサブ層かどうか ([N]LayerType)
fn isSubModuleArray(comptime F: type) bool {
    return @typeInfo(F) == .array and isSubModule(@typeInfo(F).array.child);
}

/// struct から最初の Variable のスカラー型を抽出
fn extractScalar(comptime S: type) type {
    for (std.meta.fields(S)) |field| {
        if (isVariable(field.type)) return field.type.Scalar;
        if (isSubModule(field.type)) return extractScalar(field.type);
        if (isSubModuleArray(field.type)) return extractScalar(@typeInfo(field.type).array.child);
    }
    @compileError("No Variable found in module");
}

/// パラメータ数を comptime カウント
fn countParams(comptime S: type) usize {
    var count: usize = 0;
    for (std.meta.fields(S)) |field| {
        if (isVariable(field.type)) {
            count += 1;
        } else if (isSubModule(field.type)) {
            count += countParams(field.type);
        } else if (isSubModuleArray(field.type)) {
            count += @typeInfo(field.type).array.len * countParams(@typeInfo(field.type).array.child);
        }
    }
    return count;
}

/// パラメータを再帰的に収集
fn collectParamsRecursive(
    comptime S: type,
    comptime T: type,
    self: *S,
    params: []AdamMod.Adam(T).Param,
    idx: *usize,
) void {
    inline for (std.meta.fields(S)) |field| {
        if (comptime isVariable(field.type)) {
            const v = &@field(self, field.name);
            params[idx.*] = .{
                .data = v.data(),
                .grad = &v.node.grad,
            };
            idx.* += 1;
        }
        if (comptime isSubModule(field.type)) {
            collectParamsRecursive(field.type, T, &@field(self, field.name), params, idx);
        }
        if (comptime isSubModuleArray(field.type)) {
            for (&@field(self, field.name)) |*item| {
                collectParamsRecursive(@typeInfo(field.type).array.child, T, item, params, idx);
            }
        }
    }
}

/// Module mixin: comptime reflection でパラメータ管理を自動化。
///
/// Variable フィールドと isSubModule なサブ層を自動検出し、
/// deinit/allocGrad/zeroGrad/params を自動生成する。
///
/// 使用例:
///   const M = Module(@This());
///   pub const param_count = M.param_count;
///   pub fn deinit(self: *Self) void { M.moduleDeinit(self); }
///   pub fn allocGrad(self: *Self, allocator: Allocator) !void { try M.moduleAllocGrad(self, allocator); }
///   pub fn zeroGrad(self: *Self) void { M.moduleZeroGrad(self); }
///   pub fn params(self: *Self) [param_count]nn.Adam(T).Param { return M.moduleParams(self); }
pub fn Module(comptime Self: type) type {
    const T = extractScalar(Self);
    const ParamType = AdamMod.Adam(T).Param;

    return struct {
        pub const param_count = countParams(Self);

        /// 全 Variable と全サブ層の deinit を呼ぶ
        pub fn moduleDeinit(self: *Self) void {
            inline for (std.meta.fields(Self)) |field| {
                // NOTE: Zig 0.15 では inline for 内の else if は pruning されないため
                // 独立した if (comptime ...) ブロックを使用する
                if (comptime isVariable(field.type)) {
                    @field(self, field.name).deinit();
                }
                if (comptime isSubModule(field.type)) {
                    @field(self, field.name).deinit();
                }
                if (comptime isSubModuleArray(field.type)) {
                    for (&@field(self, field.name)) |*item| item.deinit();
                }
            }
        }

        /// 全 Variable の勾配バッファを確保し、サブ層の allocGrad を呼ぶ
        pub fn moduleAllocGrad(self: *Self, allocator: Allocator) !void {
            inline for (std.meta.fields(Self)) |field| {
                if (comptime isVariable(field.type)) {
                    const v = &@field(self, field.name);
                    if (v.node.grad == null) {
                        v.node.grad = try allocator.alloc(field.type.Scalar, field.type.num_elements);
                        @memset(v.node.grad.?, 0);
                    }
                }
                if (comptime isSubModule(field.type)) {
                    try @field(self, field.name).allocGrad(allocator);
                }
                if (comptime isSubModuleArray(field.type)) {
                    for (&@field(self, field.name)) |*item| try item.allocGrad(allocator);
                }
            }
        }

        /// 全 Variable と全サブ層の勾配をゼロにリセット
        pub fn moduleZeroGrad(self: *Self) void {
            inline for (std.meta.fields(Self)) |field| {
                if (comptime isVariable(field.type)) {
                    @field(self, field.name).zeroGrad();
                }
                if (comptime isSubModule(field.type)) {
                    @field(self, field.name).zeroGrad();
                }
                if (comptime isSubModuleArray(field.type)) {
                    for (&@field(self, field.name)) |*item| item.zeroGrad();
                }
            }
        }

        /// 全パラメータを Adam.Param 配列として返す
        pub fn moduleParams(self: *Self) [param_count]ParamType {
            var p: [param_count]ParamType = undefined;
            var idx: usize = 0;
            collectParamsRecursive(Self, T, self, &p, &idx);
            return p;
        }
    };
}

// ============================================================
// テスト
// ============================================================

const VariableMod = @import("../autograd/variable.zig");
const TensorMod = @import("../core/tensor.zig");

test "Module - param_count for simple struct" {
    const SimpleLayer = struct {
        weight: VariableMod.Variable(f32, .{ 3, 2 }),
        bias: VariableMod.Variable(f32, .{2}),
    };
    const M = Module(SimpleLayer);
    try std.testing.expectEqual(2, M.param_count);
}

test "Module - param_count with non-param fields" {
    const LayerWithExtra = struct {
        weight: VariableMod.Variable(f32, .{ 4, 3 }),
        bias: VariableMod.Variable(f32, .{3}),
        eps: f32,
        training: bool,
    };
    const M = Module(LayerWithExtra);
    try std.testing.expectEqual(2, M.param_count);
}

test "Module - param_count with sub-module" {
    const Linear = @import("linear.zig").Linear;
    const Nested = struct {
        fc1: Linear(f32, 4, 3),
        fc2: Linear(f32, 3, 2),
    };
    const M = Module(Nested);
    // Linear has 2 params (weight + bias), so 2 Linear = 4
    try std.testing.expectEqual(4, M.param_count);
}

test "Module - param_count with sub-module array" {
    const Linear = @import("linear.zig").Linear;
    const Model = struct {
        layers: [3]Linear(f32, 4, 4),
    };
    const M = Module(Model);
    // 3 Linear × 2 params = 6
    try std.testing.expectEqual(6, M.param_count);
}

test "Module - moduleDeinit" {
    const allocator = std.testing.allocator;
    const SimpleLayer = struct {
        weight: VariableMod.Variable(f32, .{ 3, 2 }),
        bias: VariableMod.Variable(f32, .{2}),
    };
    const M = Module(SimpleLayer);

    const w_tensor = try TensorMod.Tensor(f32, .{ 3, 2 }).zeros(allocator);
    const b_tensor = try TensorMod.Tensor(f32, .{2}).zeros(allocator);
    var layer = SimpleLayer{
        .weight = try VariableMod.Variable(f32, .{ 3, 2 }).init(w_tensor, allocator, true),
        .bias = try VariableMod.Variable(f32, .{2}).init(b_tensor, allocator, true),
    };
    M.moduleDeinit(&layer);
    // No leak = success
}

test "Module - moduleAllocGrad and moduleZeroGrad" {
    const allocator = std.testing.allocator;
    const SimpleLayer = struct {
        weight: VariableMod.Variable(f32, .{ 3, 2 }),
        bias: VariableMod.Variable(f32, .{2}),
    };
    const M = Module(SimpleLayer);

    const w_tensor = try TensorMod.Tensor(f32, .{ 3, 2 }).zeros(allocator);
    const b_tensor = try TensorMod.Tensor(f32, .{2}).zeros(allocator);
    var layer = SimpleLayer{
        .weight = try VariableMod.Variable(f32, .{ 3, 2 }).init(w_tensor, allocator, true),
        .bias = try VariableMod.Variable(f32, .{2}).init(b_tensor, allocator, true),
    };
    defer M.moduleDeinit(&layer);

    // allocGrad
    try M.moduleAllocGrad(&layer, allocator);
    try std.testing.expect(layer.weight.node.grad != null);
    try std.testing.expect(layer.bias.node.grad != null);
    try std.testing.expectEqual(@as(usize, 6), layer.weight.node.grad.?.len);
    try std.testing.expectEqual(@as(usize, 2), layer.bias.node.grad.?.len);

    // Set some gradient values
    layer.weight.node.grad.?[0] = 1.0;
    layer.bias.node.grad.?[0] = 2.0;

    // zeroGrad
    M.moduleZeroGrad(&layer);
    try std.testing.expectEqual(@as(f32, 0), layer.weight.node.grad.?[0]);
    try std.testing.expectEqual(@as(f32, 0), layer.bias.node.grad.?[0]);
}

test "Module - moduleParams" {
    const allocator = std.testing.allocator;
    const SimpleLayer = struct {
        weight: VariableMod.Variable(f32, .{ 3, 2 }),
        bias: VariableMod.Variable(f32, .{2}),
    };
    const M = Module(SimpleLayer);

    const w_tensor = try TensorMod.Tensor(f32, .{ 3, 2 }).zeros(allocator);
    const b_tensor = try TensorMod.Tensor(f32, .{2}).zeros(allocator);
    var layer = SimpleLayer{
        .weight = try VariableMod.Variable(f32, .{ 3, 2 }).init(w_tensor, allocator, true),
        .bias = try VariableMod.Variable(f32, .{2}).init(b_tensor, allocator, true),
    };
    defer M.moduleDeinit(&layer);

    try M.moduleAllocGrad(&layer, allocator);

    const params = M.moduleParams(&layer);
    try std.testing.expectEqual(2, params.len);
    try std.testing.expectEqual(@as(usize, 6), params[0].data.len);
    try std.testing.expectEqual(@as(usize, 2), params[1].data.len);

    // Verify params point to the actual data
    layer.weight.data()[0] = 42.0;
    try std.testing.expectEqual(@as(f32, 42.0), params[0].data[0]);
}

test "Module - moduleParams with nested Linear" {
    const allocator = std.testing.allocator;
    const Linear = @import("linear.zig").Linear;

    const Model = struct {
        fc1: Linear(f32, 4, 3),
        fc2: Linear(f32, 3, 2),
    };
    const M = Module(Model);

    var model = Model{
        .fc1 = try Linear(f32, 4, 3).init(allocator),
        .fc2 = try Linear(f32, 3, 2).init(allocator),
    };
    defer M.moduleDeinit(&model);

    try M.moduleAllocGrad(&model, allocator);

    const params = M.moduleParams(&model);
    try std.testing.expectEqual(4, params.len);
    // fc1.weight: 4*3=12, fc1.bias: 3, fc2.weight: 3*2=6, fc2.bias: 2
    try std.testing.expectEqual(@as(usize, 12), params[0].data.len);
    try std.testing.expectEqual(@as(usize, 3), params[1].data.len);
    try std.testing.expectEqual(@as(usize, 6), params[2].data.len);
    try std.testing.expectEqual(@as(usize, 2), params[3].data.len);
}

test "Module - allocGrad idempotent" {
    const allocator = std.testing.allocator;
    const SimpleLayer = struct {
        weight: VariableMod.Variable(f32, .{4}),
    };
    const M = Module(SimpleLayer);

    const tensor = try TensorMod.Tensor(f32, .{4}).zeros(allocator);
    var layer = SimpleLayer{
        .weight = try VariableMod.Variable(f32, .{4}).init(tensor, allocator, true),
    };
    defer M.moduleDeinit(&layer);

    // Call allocGrad twice - should not allocate again
    try M.moduleAllocGrad(&layer, allocator);
    const grad_ptr = layer.weight.node.grad.?.ptr;
    try M.moduleAllocGrad(&layer, allocator);
    try std.testing.expectEqual(grad_ptr, layer.weight.node.grad.?.ptr);
}
