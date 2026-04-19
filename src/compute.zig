/// compute.zig: バックエンド非依存のパラメータ管理
///
/// Module はモデルのパラメータメタデータ (shape, init_kind) を管理する。
/// 実際のデータ割り当ては MpsRuntime / CpuRuntime が行う。
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ParamHandle = struct { index: usize };

/// MPS data type constants (used by constantScalar/constantData dtype parameter)
pub const MPSDataTypeFloat32: u32 = 0x10000020; // 268435488
pub const MPSDataTypeInt32: u32 = 0x20000020;

pub const ParamInit = union(enum) {
    xavier,
    /// Kaiming He init: N(0, sqrt(2/fan_in)), fan_in = shape[0] (Linear 向け)
    kaiming,
    /// Kaiming He init with explicit fan_in (Conv2d 等で使用: `.{ .kaiming_fan = in_ch * k * k }`)
    kaiming_fan: usize,
    zeros,
    ones,
    normal: struct { mean: f32 = 0, std_dev: f32 = 0.01 },
};

pub const ParamMeta = struct {
    shape: []const usize,
    init_kind: ParamInit,
};

pub const Module = struct {
    params: std.ArrayListUnmanaged(ParamMeta),
    allocator: Allocator,

    pub fn init(allocator: Allocator) Module {
        return .{
            .params = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Module) void {
        self.params.deinit(self.allocator);
    }

    /// パラメータを登録: shape + init_kind を記録し、ParamHandle を返す
    pub fn addParam(self: *Module, shape: []const usize, init_kind: ParamInit) ParamHandle {
        const index = self.params.items.len;
        self.params.append(self.allocator, .{
            .shape = shape,
            .init_kind = init_kind,
        }) catch unreachable;
        return .{ .index = index };
    }

    /// 登録されたパラメータ数
    pub fn paramCount(self: *const Module) usize {
        return self.params.items.len;
    }

    /// パラメータの要素数
    pub fn paramSize(self: *const Module, handle: ParamHandle) usize {
        const meta = self.params.items[handle.index];
        var size: usize = 1;
        for (meta.shape) |d| size *= d;
        return size;
    }

    /// 全パラメータの総要素数
    pub fn totalParamElements(self: *const Module) usize {
        var sum: usize = 0;
        for (self.params.items) |meta| {
            var s: usize = 1;
            for (meta.shape) |d| s *= d;
            sum += s;
        }
        return sum;
    }

    /// 全パラメータの要素数配列
    pub fn paramSizes(self: *const Module, allocator: Allocator) ![]usize {
        const result = try allocator.alloc(usize, self.params.items.len);
        for (self.params.items, 0..) |meta, i| {
            var size: usize = 1;
            for (meta.shape) |d| size *= d;
            result[i] = size;
        }
        return result;
    }
};

/// Adam optimizer state (CPU/GPU 共用)
pub const AdamState = struct {
    m: [][]f32,
    v: [][]f32,
    step: u32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, param_sizes: []const usize) !AdamState {
        const count = param_sizes.len;
        const m = try allocator.alloc([]f32, count);
        const v = try allocator.alloc([]f32, count);
        for (0..count) |i| {
            m[i] = try allocator.alloc(f32, param_sizes[i]);
            @memset(m[i], 0);
            v[i] = try allocator.alloc(f32, param_sizes[i]);
            @memset(v[i], 0);
        }
        return .{ .m = m, .v = v, .step = 0, .allocator = allocator };
    }

    pub fn deinit(self: *AdamState) void {
        for (self.m) |m| self.allocator.free(m);
        self.allocator.free(self.m);
        for (self.v) |v| self.allocator.free(v);
        self.allocator.free(self.v);
    }
};

/// Adam step (CPU実装、1パラメータ分) — SIMD 最適化
pub fn adamStep(param: []f32, grad: []const f32, m: []f32, v: []f32, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, step: u32) void {
    const t_f: f32 = @floatFromInt(step);
    const bc1 = 1.0 - std.math.pow(f32, beta1, t_f);
    const bc2 = 1.0 - std.math.pow(f32, beta2, t_f);

    const VEC_LEN = std.simd.suggestVectorLength(f32) orelse 4;
    const n = param.len;
    const vec_n = n - (n % VEC_LEN);

    const v_beta1: @Vector(VEC_LEN, f32) = @splat(beta1);
    const v_1_minus_beta1: @Vector(VEC_LEN, f32) = @splat(1.0 - beta1);
    const v_beta2: @Vector(VEC_LEN, f32) = @splat(beta2);
    const v_1_minus_beta2: @Vector(VEC_LEN, f32) = @splat(1.0 - beta2);
    const v_bc1: @Vector(VEC_LEN, f32) = @splat(bc1);
    const v_bc2: @Vector(VEC_LEN, f32) = @splat(bc2);
    const v_lr: @Vector(VEC_LEN, f32) = @splat(lr);
    const v_eps: @Vector(VEC_LEN, f32) = @splat(eps);
    const v_wd: @Vector(VEC_LEN, f32) = @splat(wd);

    var i: usize = 0;
    while (i < vec_n) : (i += VEC_LEN) {
        const p_vec: @Vector(VEC_LEN, f32) = param[i..][0..VEC_LEN].*;
        const g_vec: @Vector(VEC_LEN, f32) = grad[i..][0..VEC_LEN].*;
        var m_vec: @Vector(VEC_LEN, f32) = m[i..][0..VEC_LEN].*;
        var v_vec: @Vector(VEC_LEN, f32) = v[i..][0..VEC_LEN].*;

        const g_wd = g_vec + v_wd * p_vec;
        m_vec = v_beta1 * m_vec + v_1_minus_beta1 * g_wd;
        v_vec = v_beta2 * v_vec + v_1_minus_beta2 * g_wd * g_wd;
        const m_hat = m_vec / v_bc1;
        const v_hat = v_vec / v_bc2;

        m[i..][0..VEC_LEN].* = m_vec;
        v[i..][0..VEC_LEN].* = v_vec;
        param[i..][0..VEC_LEN].* = p_vec - v_lr * m_hat / (@sqrt(v_hat) + v_eps);
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        const g_wd = grad[i] + wd * param[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g_wd;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g_wd * g_wd;
        const m_hat = m[i] / bc1;
        const v_hat = v[i] / bc2;
        param[i] -= lr * m_hat / (@sqrt(v_hat) + eps);
    }
}

// ── Learning Rate Schedulers ──

/// Cosine annealing: lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*step/total_steps))
pub fn cosineAnnealingLR(step: u32, total_steps: u32, lr_min: f32, lr_max: f32) f32 {
    if (total_steps == 0) return lr_min;
    const ratio: f32 = @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(total_steps));
    const clamped = @min(ratio, 1.0);
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + @cos(std.math.pi * clamped));
}

/// Linear warmup: step < warmup → lr_max * step/warmup, else lr_max
pub fn linearWarmupLR(step: u32, warmup_steps: u32, lr_max: f32) f32 {
    if (step >= warmup_steps) return lr_max;
    return lr_max * @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(warmup_steps));
}

/// Warmup + cosine decay
pub fn warmupCosineDecayLR(step: u32, warmup_steps: u32, total_steps: u32, lr_min: f32, lr_max: f32) f32 {
    if (step < warmup_steps) return linearWarmupLR(step, warmup_steps, lr_max);
    const decay_step = step - warmup_steps;
    const decay_total = if (total_steps > warmup_steps) total_steps - warmup_steps else 1;
    return cosineAnnealingLR(decay_step, decay_total, lr_min, lr_max);
}

const CHECKPOINT_MAGIC: u32 = 0x4D504752; // "MPGR"
const CHECKPOINT_VERSION: u32 = 1;

fn readExactPositional(f: std.Io.File, io: std.Io, offset: *u64, out: []u8) !void {
    const n = try f.readPositionalAll(io, out, offset.*);
    if (n != out.len) return error.EndOfStream;
    offset.* += out.len;
}

fn readU32Positional(f: std.Io.File, io: std.Io, offset: *u64) !u32 {
    var bytes: [4]u8 = undefined;
    try readExactPositional(f, io, offset, &bytes);
    return std.mem.littleToNative(u32, @bitCast(bytes));
}

fn writeExactPositional(f: std.Io.File, io: std.Io, offset: *u64, bytes: []const u8) !void {
    try f.writePositionalAll(io, bytes, offset.*);
    offset.* += bytes.len;
}

fn writeU32Positional(f: std.Io.File, io: std.Io, offset: *u64, v: u32) !void {
    const le = std.mem.nativeToLittle(u32, v);
    try writeExactPositional(f, io, offset, std.mem.asBytes(&le));
}

/// Checkpoint 保存 (パラメータデータ + Adam state)
/// param_data: 各パラメータの f32 スライス
pub fn saveCheckpoint(
    module: *const Module,
    io: std.Io,
    param_data: []const []const f32,
    adam: *const AdamState,
    path: []const u8,
) !void {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, path, .{});
    defer file.close(io);

    var offset: u64 = 0;
    const count: u32 = @intCast(module.params.items.len);

    try writeU32Positional(file, io, &offset, CHECKPOINT_MAGIC);
    try writeU32Positional(file, io, &offset, CHECKPOINT_VERSION);
    try writeU32Positional(file, io, &offset, count);
    try writeU32Positional(file, io, &offset, adam.step);

    // Each param: num_elements, weight_data, adam_m, adam_v
    for (module.params.items, 0..) |meta, i| {
        var size: usize = 1;
        for (meta.shape) |d| size *= d;

        const n: u32 = @intCast(size);
        try writeU32Positional(file, io, &offset, n);
        try writeExactPositional(file, io, &offset, std.mem.sliceAsBytes(param_data[i]));
        try writeExactPositional(file, io, &offset, std.mem.sliceAsBytes(adam.m[i]));
        try writeExactPositional(file, io, &offset, std.mem.sliceAsBytes(adam.v[i]));
    }
}

/// Checkpoint 読み込み
/// param_data: 各パラメータの f32 スライス (書き込み先)
pub fn loadCheckpoint(
    module: *const Module,
    io: std.Io,
    param_data: [][]f32,
    adam: *AdamState,
    path: []const u8,
) !void {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{});
    defer file.close(io);

    var offset: u64 = 0;

    // Header
    const magic = try readU32Positional(file, io, &offset);
    if (magic != CHECKPOINT_MAGIC) return error.InvalidCheckpoint;
    const version = try readU32Positional(file, io, &offset);
    if (version != CHECKPOINT_VERSION) return error.UnsupportedVersion;
    const num_params = try readU32Positional(file, io, &offset);
    if (num_params != module.params.items.len) return error.ParamCountMismatch;
    adam.step = try readU32Positional(file, io, &offset);

    // Each param
    for (module.params.items, 0..) |meta, i| {
        var size: usize = 1;
        for (meta.shape) |d| size *= d;

        const n = try readU32Positional(file, io, &offset);
        if (n != size) return error.ParamSizeMismatch;

        try readExactPositional(file, io, &offset, std.mem.sliceAsBytes(param_data[i]));
        try readExactPositional(file, io, &offset, std.mem.sliceAsBytes(adam.m[i]));
        try readExactPositional(file, io, &offset, std.mem.sliceAsBytes(adam.v[i]));
    }
}

// ── Tests ──

const testing = std.testing;

test "cosineAnnealingLR" {
    // step=0 → lr_max
    try testing.expectApproxEqAbs(cosineAnnealingLR(0, 100, 0.0, 1.0), 1.0, 1e-5);
    // step=total → lr_min
    try testing.expectApproxEqAbs(cosineAnnealingLR(100, 100, 0.0, 1.0), 0.0, 1e-5);
    // step=50 → midpoint
    try testing.expectApproxEqAbs(cosineAnnealingLR(50, 100, 0.0, 1.0), 0.5, 1e-5);
}

test "linearWarmupLR" {
    try testing.expectApproxEqAbs(linearWarmupLR(0, 10, 1.0), 0.0, 1e-5);
    try testing.expectApproxEqAbs(linearWarmupLR(5, 10, 1.0), 0.5, 1e-5);
    try testing.expectApproxEqAbs(linearWarmupLR(10, 10, 1.0), 1.0, 1e-5);
    try testing.expectApproxEqAbs(linearWarmupLR(20, 10, 1.0), 1.0, 1e-5);
}

test "warmupCosineDecayLR" {
    // During warmup
    try testing.expectApproxEqAbs(warmupCosineDecayLR(0, 10, 110, 0.0, 1.0), 0.0, 1e-5);
    try testing.expectApproxEqAbs(warmupCosineDecayLR(5, 10, 110, 0.0, 1.0), 0.5, 1e-5);
    // At warmup end → lr_max
    try testing.expectApproxEqAbs(warmupCosineDecayLR(10, 10, 110, 0.0, 1.0), 1.0, 1e-5);
    // At end → lr_min
    try testing.expectApproxEqAbs(warmupCosineDecayLR(110, 10, 110, 0.0, 1.0), 0.0, 1e-5);
}
