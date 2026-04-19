/// diff_cpu_runtime.zig: 微分可能 CPU ランタイム
///
/// CpuRuntime と同じ duck-typed ops インターフェースを実装しつつ、
/// forward 時に計算グラフを構築し backward() で自動微分する。
/// 統一モジュールの forward(ctx: anytype, ...) が DiffCpuRuntime を ctx として
/// 受け取れば、同じ forward コードで CPU training が可能。
const std = @import("std");
const Allocator = std.mem.Allocator;
const compute = @import("compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const AdamState = compute.AdamState;
const cpu_backend = @import("backend/cpu.zig");
const kernels = @import("runtime_kernels.zig");
const diff_node = @import("diff_node.zig");

pub const MAX_NDIM = kernels.MAX_NDIM;

pub const DiffNode = diff_node.DiffNodeGeneric([]f32, kernels.MAX_NDIM);

// ── Backward function types ──
const BackwardFn = *const fn (*DiffNode) void;

pub const DiffTensor = *DiffNode;

pub const DiffCpuRuntime = struct {
    allocator: Allocator,
    module: *const Module,
    param_nodes: []DiffNode, // 永続: パラメータ
    param_grads: [][]f32, // 永続: パラメータ勾配
    arena: std.heap.ArenaAllocator, // 中間テンソル + ノード
    topo_buf: std.ArrayListUnmanaged(*DiffNode),
    prng: std.Random.DefaultPrng,
    training: bool,

    pub fn init(module: *const Module, allocator: Allocator) !DiffCpuRuntime {
        const count = module.paramCount();
        const param_nodes = try allocator.alloc(DiffNode, count);
        const param_grads = try allocator.alloc([]f32, count);

        for (module.params.items, 0..) |meta, i| {
            const size = module.paramSize(.{ .index = i });
            const data = try allocator.alloc(f32, size);
            const grad = try allocator.alloc(f32, size);
            @memset(grad, 0);

            param_nodes[i] = .{
                .data = data,
                .shape = kernels.initShapeArray(meta.shape),
                .ndim = meta.shape.len,
                .grad = grad,
                .backward_fn = null,
                .parents = .{ null, null, null },
                .context = null,
                .requires_grad = true,
                .visited = false,
                .is_param = true,
                .param_index = i,
            };
            param_grads[i] = grad;
        }

        return .{
            .allocator = allocator,
            .module = module,
            .param_nodes = param_nodes,
            .param_grads = param_grads,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .topo_buf = .empty,
            .prng = std.Random.DefaultPrng.init(42),
            .training = true,
        };
    }

    pub fn deinit(self: *DiffCpuRuntime) void {
        for (self.param_nodes, 0..) |node, i| {
            self.allocator.free(node.data);
            self.allocator.free(self.param_grads[i]);
        }
        self.allocator.free(self.param_nodes);
        self.allocator.free(self.param_grads);
        self.arena.deinit();
        self.topo_buf.deinit(self.allocator);
    }

    pub fn resetArena(self: *DiffCpuRuntime) void {
        _ = self.arena.reset(.retain_capacity);
        // Reset visited flags on params
        for (self.param_nodes) |*node| {
            node.visited = false;
        }
    }

    pub fn zeroGrad(self: *DiffCpuRuntime) void {
        for (self.param_grads) |grad| {
            @memset(grad, 0);
        }
        // Re-assign grad pointers (in case they diverged)
        for (self.param_nodes, 0..) |*node, i| {
            node.grad = self.param_grads[i];
        }
    }

    fn arenaAlloc(self: *DiffCpuRuntime) Allocator {
        return self.arena.allocator();
    }

    // ── Tensor/Node creation helpers ──

    pub fn makeNode(self: *DiffCpuRuntime, data: []f32, shape_slice: []const usize, requires_grad: bool) *DiffNode {
        const node = self.arenaAlloc().create(DiffNode) catch unreachable;
        node.* = .{
            .data = data,
            .shape = kernels.initShapeArray(shape_slice),
            .ndim = shape_slice.len,
            .grad = null,
            .backward_fn = null,
            .parents = .{ null, null, null },
            .context = null,
            .requires_grad = requires_grad,
            .visited = false,
            .is_param = false,
            .param_index = null,
        };
        return node;
    }

    pub fn makeTensor(self: *DiffCpuRuntime, data: []f32, shape: []const usize) DiffTensor {
        return self.makeNode(data, shape, false);
    }

    pub fn allocData(self: *DiffCpuRuntime, size: usize) []f32 {
        return self.arenaAlloc().alloc(f32, size) catch unreachable;
    }

    fn allocGrad(self: *DiffCpuRuntime, size: usize) []f32 {
        const g = self.arenaAlloc().alloc(f32, size) catch unreachable;
        @memset(g, 0);
        return g;
    }

    // ── Context allocation helper ──

    fn allocContext(self: *DiffCpuRuntime, comptime T: type) *T {
        return self.arenaAlloc().create(T) catch unreachable;
    }

    // ── Unary op helper ──

    /// context 不要の単純 unary op のボイラープレートを吸収する。
    /// fwd は comptime 関数ポインタなので確実にインライン展開される。
    fn unaryOp(
        self: *DiffCpuRuntime,
        x: DiffTensor,
        comptime fwd: fn ([]const f32, []f32) void,
        bwd: *const fn (*DiffNode) void,
    ) DiffTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        fwd(x.data[0..total], out[0..total]);
        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = bwd;
        }
        return node;
    }

    // ── Param access ──

    pub fn param(self: *DiffCpuRuntime, handle: ParamHandle) DiffTensor {
        return &self.param_nodes[handle.index];
    }

    // ── Leaf ops ──

    pub fn constantScalar(self: *DiffCpuRuntime, val: f64, dtype: u32) DiffTensor {
        _ = dtype;
        const out = self.allocData(1);
        out[0] = @floatCast(val);
        return self.makeNode(out, &.{1}, false);
    }

    pub fn constantData(self: *DiffCpuRuntime, data: [*]const u8, len: usize, new_shape: []const usize, dtype: u32) DiffTensor {
        _ = dtype;
        const n_floats = len / @sizeOf(f32);
        const out = self.allocData(n_floats);
        const src: [*]const f32 = @ptrCast(@alignCast(data));
        @memcpy(out, src[0..n_floats]);
        return self.makeNode(out, new_shape, false);
    }

    pub fn stopGradient(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        // Create a new node with same data but requires_grad=false
        const node = self.makeNode(x.data, x.shape[0..x.ndim], false);
        return node;
    }

    // ── Unary ops ──

    pub fn negative(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        return self.unaryOp(x, kernels.negativeForward, &backwardNegative);
    }

    fn backwardNegative(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        const ga = pa.grad.?;
        for (0..total) |i| ga[i] -= go[i];
    }

    pub fn gelu(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        return self.unaryOp(x, kernels.geluForward, &backwardGelu);
    }

    fn backwardGelu(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        const ga = pa.grad.?;
        const sqrt_2_over_pi: f32 = 0.7978845608028654;
        for (0..total) |i| {
            const v = pa.data[i];
            const inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
            const tanh_val = std.math.tanh(inner);
            const sech2 = 1.0 - tanh_val * tanh_val;
            const inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * v * v);
            // d/dx gelu(x) = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * inner'
            ga[i] += go[i] * (0.5 * (1.0 + tanh_val) + 0.5 * v * sech2 * inner_deriv);
        }
    }

    pub fn silu(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        const sig_cache = self.allocData(total);
        kernels.siluForward(x.data[0..total], out[0..total], sig_cache[0..total]);
        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(SiluContext);
            ctx.* = .{ .sig_cache = sig_cache };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardSilu;
        }
        return node;
    }

    const SiluContext = struct {
        sig_cache: []f32,
    };

    fn backwardSilu(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        const ga = pa.grad.?;
        const ctx: *SiluContext = @ptrCast(@alignCast(self_node.context.?));
        const sig = ctx.sig_cache;
        for (0..total) |i| {
            const v = pa.data[i];
            // d/dx (x * sig(x)) = sig + x * sig * (1 - sig)
            ga[i] += go[i] * (sig[i] + v * sig[i] * (1.0 - sig[i]));
        }
    }

    /// Fused add + silu: silu(a + b) — delegates to add then silu
    pub fn addSilu(self: *DiffCpuRuntime, a: DiffTensor, b: DiffTensor) DiffTensor {
        return self.silu(self.add(a, b));
    }

    pub fn square(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        return self.unaryOp(x, kernels.squareForward, &backwardSquare);
    }

    fn backwardSquare(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| {
                ga[i] += go[i] * 2.0 * pa.data[i];
            }
        }
    }

    // ── exp / log / abs / sqrt / clamp ──

    /// exp(x) — d/dx exp(x) = exp(x) = out
    pub fn exp(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        return self.unaryOp(x, kernels.expForward, &backwardExp);
    }

    fn backwardExp(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| ga[i] += go[i] * self_node.data[i]; // out = exp(x)
        }
    }

    /// log(x) — d/dx log(x) = 1/x
    pub fn log(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        for (0..total) |i| out[i] = @log(x.data[i]);
        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardLog;
        }
        return node;
    }

    fn backwardLog(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| ga[i] += go[i] / pa.data[i];
        }
    }

    /// abs(x) — d/dx |x| = sign(x)
    pub fn abs(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        for (0..total) |i| out[i] = @abs(x.data[i]);
        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardAbs;
        }
        return node;
    }

    fn backwardAbs(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| {
                const sign: f32 = if (pa.data[i] > 0) 1.0 else if (pa.data[i] < 0) -1.0 else 0.0;
                ga[i] += go[i] * sign;
            }
        }
    }

    /// sqrt(x) — d/dx sqrt(x) = 0.5 / sqrt(x)
    pub fn sqrt(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        for (0..total) |i| out[i] = @sqrt(x.data[i]);
        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardSqrt;
        }
        return node;
    }

    fn backwardSqrt(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| ga[i] += go[i] * 0.5 / self_node.data[i]; // 0.5 / sqrt(x)
        }
    }

    /// clamp(x, min, max) — gradient passes through if min <= x <= max
    pub fn clamp(self: *DiffCpuRuntime, x: DiffTensor, min_val: f32, max_val: f32) DiffTensor {
        const total = x.totalElements();
        const out = self.allocData(total);
        for (0..total) |i| out[i] = @min(@max(x.data[i], min_val), max_val);
        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(ClampContext);
            ctx.* = .{ .min_val = min_val, .max_val = max_val };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardClamp;
        }
        return node;
    }

    const ClampContext = struct { min_val: f32, max_val: f32 };

    fn backwardClamp(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        const ctx: *const ClampContext = @ptrCast(@alignCast(self_node.context.?));
        if (pa.grad) |ga| {
            for (0..total) |i| {
                const in_range = pa.data[i] >= ctx.min_val and pa.data[i] <= ctx.max_val;
                ga[i] += if (in_range) go[i] else 0.0;
            }
        }
    }

    /// element-wise div: a / b — d/da = 1/b, d/db = -a/b^2
    pub fn div(self: *DiffCpuRuntime, a: DiffTensor, b: DiffTensor) DiffTensor {
        const total = a.totalElements();
        const out = self.allocData(total);
        for (0..total) |i| out[i] = a.data[i] / b.data[i];
        const rg = a.requires_grad or b.requires_grad;
        const node = self.makeNode(out, a.shape[0..a.ndim], rg);
        if (rg) {
            node.parents[0] = a;
            node.parents[1] = b;
            node.backward_fn = &backwardDiv;
        }
        return node;
    }

    fn backwardDiv(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| ga[i] += go[i] / pb.data[i];
        }
        if (pb.grad) |gb| {
            for (0..total) |i| gb[i] += -go[i] * pa.data[i] / (pb.data[i] * pb.data[i]);
        }
    }

    pub fn tanh_(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        return self.unaryOp(x, kernels.tanhForward, &backwardTanh);
    }

    fn backwardTanh(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| {
                const y = self_node.data[i]; // tanh output
                ga[i] += go[i] * (1.0 - y * y);
            }
        }
    }

    pub fn sigmoid(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        return self.unaryOp(x, kernels.sigmoidForward, &backwardSigmoid);
    }

    fn backwardSigmoid(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| {
                const y = self_node.data[i]; // sigmoid output
                ga[i] += go[i] * y * (1.0 - y);
            }
        }
    }

    // ── Binary ops ──

    pub fn add(self: *DiffCpuRuntime, a: DiffTensor, b: DiffTensor) DiffTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out = self.allocData(a_total);
            for (0..a_total) |i| out[i] = a.data[i] + b.data[i];
            const node = self.makeNode(out, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardAddSame;
            }
            return node;
        }

        // Broadcast: b is smaller
        if (b_total < a_total and a_total % b_total == 0) {
            const out = self.allocData(a_total);
            for (0..a_total) |i| out[i] = a.data[i] + b.data[i % b_total];
            const node = self.makeNode(out, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardAddBroadcastB;
            }
            return node;
        }

        // Broadcast: a is smaller
        if (a_total < b_total and b_total % a_total == 0) {
            const out = self.allocData(b_total);
            for (0..b_total) |i| out[i] = a.data[i % a_total] + b.data[i];
            const node = self.makeNode(out, b.shape[0..b.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardAddBroadcastA;
            }
            return node;
        }

        @panic("add: incompatible shapes for broadcast");
    }

    fn backwardAddSame(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const total = self_node.totalElements();
        if (pa.grad) |ga| {
            for (0..total) |i| ga[i] += go[i];
        }
        if (pb.grad) |gb| {
            for (0..total) |i| gb[i] += go[i];
        }
    }

    fn backwardAddBroadcastB(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |ga| {
            for (0..a_total) |i| ga[i] += go[i];
        }
        if (pb.grad) |gb| {
            for (0..a_total) |i| gb[i % b_total] += go[i];
        }
    }

    fn backwardAddBroadcastA(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |ga| {
            for (0..b_total) |i| ga[i % a_total] += go[i];
        }
        if (pb.grad) |gb| {
            for (0..b_total) |i| gb[i] += go[i];
        }
    }

    pub fn mul(self: *DiffCpuRuntime, a: DiffTensor, b: DiffTensor) DiffTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out = self.allocData(a_total);
            for (0..a_total) |i| out[i] = a.data[i] * b.data[i];
            const node = self.makeNode(out, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMulSame;
            }
            return node;
        }

        // Broadcast: b is scalar or smaller
        if (b_total <= a_total and a_total % b_total == 0) {
            const out = self.allocData(a_total);
            for (0..a_total) |i| out[i] = a.data[i] * b.data[i % b_total];
            const node = self.makeNode(out, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMulBroadcastB;
            }
            return node;
        }

        // Broadcast: a is scalar or smaller
        if (a_total < b_total and b_total % a_total == 0) {
            const out = self.allocData(b_total);
            for (0..b_total) |i| out[i] = a.data[i % a_total] * b.data[i];
            const node = self.makeNode(out, b.shape[0..b.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMulBroadcastA;
            }
            return node;
        }

        @panic("mul: incompatible shapes for broadcast");
    }

    fn backwardMulSame(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const total = self_node.totalElements();
        if (pa.grad) |ga| {
            for (0..total) |i| ga[i] += go[i] * pb.data[i];
        }
        if (pb.grad) |gb| {
            for (0..total) |i| gb[i] += go[i] * pa.data[i];
        }
    }

    fn backwardMulBroadcastB(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |ga| {
            for (0..a_total) |i| ga[i] += go[i] * pb.data[i % b_total];
        }
        if (pb.grad) |gb| {
            for (0..a_total) |i| gb[i % b_total] += go[i] * pa.data[i];
        }
    }

    fn backwardMulBroadcastA(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |ga| {
            for (0..b_total) |i| ga[i % a_total] += go[i] * pb.data[i];
        }
        if (pb.grad) |gb| {
            for (0..b_total) |i| gb[i] += go[i] * pa.data[i % a_total];
        }
    }

    pub fn sub(self: *DiffCpuRuntime, a: DiffTensor, b: DiffTensor) DiffTensor {
        const a_total = a.totalElements();
        const b_total = b.totalElements();
        const rg = a.requires_grad or b.requires_grad;

        if (a_total == b_total) {
            const out = self.allocData(a_total);
            for (0..a_total) |i| out[i] = a.data[i] - b.data[i];
            const node = self.makeNode(out, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardSubSame;
            }
            return node;
        }

        if (b_total < a_total and a_total % b_total == 0) {
            const out = self.allocData(a_total);
            for (0..a_total) |i| out[i] = a.data[i] - b.data[i % b_total];
            const node = self.makeNode(out, a.shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardSubBroadcastB;
            }
            return node;
        }

        @panic("sub: incompatible shapes for broadcast");
    }

    fn backwardSubSame(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const total = self_node.totalElements();
        if (pa.grad) |ga| {
            for (0..total) |i| ga[i] += go[i];
        }
        if (pb.grad) |gb| {
            for (0..total) |i| gb[i] -= go[i];
        }
    }

    fn backwardSubBroadcastB(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();
        if (pa.grad) |ga| {
            for (0..a_total) |i| ga[i] += go[i];
        }
        if (pb.grad) |gb| {
            for (0..a_total) |i| gb[i % b_total] -= go[i];
        }
    }

    // ── Matmul ──

    pub fn matmul(self: *DiffCpuRuntime, a: DiffTensor, b: DiffTensor) DiffTensor {
        const rg = a.requires_grad or b.requires_grad;

        if (a.ndim == 2 and b.ndim == 2) {
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[1];
            const out = self.allocData(M * N);
            cpu_backend.matmul(f32, a.data.ptr, b.data.ptr, out.ptr, M, K, N);
            const node = self.makeNode(out, &.{ M, N }, rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMatmul2D;
            }
            return node;
        }

        if (a.ndim == 3 and b.ndim == 3) {
            const B = a.shape[0];
            const M = a.shape[1];
            const K = a.shape[2];
            const N = b.shape[2];
            const out = self.allocData(B * M * N);
            for (0..B) |batch| {
                cpu_backend.matmul(
                    f32,
                    a.data[batch * M * K ..].ptr,
                    b.data[batch * K * N ..].ptr,
                    out[batch * M * N ..].ptr,
                    M,
                    K,
                    N,
                );
            }
            const node = self.makeNode(out, &.{ B, M, N }, rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMatmul3D;
            }
            return node;
        }

        // 2D @ 3D: treat as batch with shared 2D
        if (a.ndim == 2 and b.ndim == 3) {
            const B = b.shape[0];
            const M = a.shape[0];
            const K = a.shape[1];
            const N = b.shape[2];
            const out = self.allocData(B * M * N);
            for (0..B) |batch| {
                cpu_backend.matmul(
                    f32,
                    a.data.ptr,
                    b.data[batch * K * N ..].ptr,
                    out[batch * M * N ..].ptr,
                    M,
                    K,
                    N,
                );
            }
            const node = self.makeNode(out, &.{ B, M, N }, rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                node.backward_fn = &backwardMatmul2D3D;
            }
            return node;
        }

        @panic("matmul: unsupported shape combination (expected 2D or 3D)");
    }

    fn backwardMatmul2D(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const M = pa.shape[0];
        const K = pa.shape[1];
        const N = pb.shape[1];

        // dA += go @ B^T
        if (pa.grad) |ga| {
            cpu_backend.matmulTransBAccum(f32, go.ptr, pb.data.ptr, ga.ptr, M, N, K);
        }
        // dB += A^T @ go
        if (pb.grad) |gb| {
            cpu_backend.matmulTransAAccum(f32, pa.data.ptr, go.ptr, gb.ptr, K, M, N);
        }
    }

    fn backwardMatmul3D(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const B = pa.shape[0];
        const M = pa.shape[1];
        const K = pa.shape[2];
        const N = pb.shape[2];

        for (0..B) |batch| {
            if (pa.grad) |ga| {
                cpu_backend.matmulTransBAccum(
                    f32,
                    go[batch * M * N ..].ptr,
                    pb.data[batch * K * N ..].ptr,
                    ga[batch * M * K ..].ptr,
                    M,
                    N,
                    K,
                );
            }
            if (pb.grad) |gb| {
                cpu_backend.matmulTransAAccum(
                    f32,
                    pa.data[batch * M * K ..].ptr,
                    go[batch * M * N ..].ptr,
                    gb[batch * K * N ..].ptr,
                    K,
                    M,
                    N,
                );
            }
        }
    }

    fn backwardMatmul2D3D(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const B = pb.shape[0];
        const M = pa.shape[0];
        const K = pa.shape[1];
        const N = pb.shape[2];

        // dA (2D) += sum_b go[b] @ B[b]^T
        if (pa.grad) |ga| {
            for (0..B) |batch| {
                cpu_backend.matmulTransBAccum(
                    f32,
                    go[batch * M * N ..].ptr,
                    pb.data[batch * K * N ..].ptr,
                    ga.ptr,
                    M,
                    N,
                    K,
                );
            }
        }
        // dB[b] (3D) += A^T @ go[b]
        if (pb.grad) |gb| {
            for (0..B) |batch| {
                cpu_backend.matmulTransAAccum(
                    f32,
                    pa.data.ptr,
                    go[batch * M * N ..].ptr,
                    gb[batch * K * N ..].ptr,
                    K,
                    M,
                    N,
                );
            }
        }
    }

    // ── Shape ops ──

    pub fn reshape(self: *DiffCpuRuntime, x: DiffTensor, new_shape: []const usize) DiffTensor {
        // Data is shared (no copy), reshape just changes the view
        const node = self.makeNode(x.data, new_shape, x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            // Save original shape in context for backward
            const ctx = self.allocContext(ReshapeContext);
            ctx.* = .{ .orig_ndim = x.ndim, .orig_shape = x.shape };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardReshape;
        }
        return node;
    }

    const ReshapeContext = struct {
        orig_ndim: usize,
        orig_shape: [MAX_NDIM]usize,
    };

    fn backwardReshape(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        // Reshape grad shares the same memory, just accumulate
        if (pa.grad) |ga| {
            const total = self_node.totalElements();
            for (0..total) |i| ga[i] += go[i];
        }
    }

    pub fn transpose(self: *DiffCpuRuntime, x: DiffTensor, d1: u64, d2: u64) DiffTensor {
        const dim1: usize = @intCast(d1);
        const dim2: usize = @intCast(d2);

        if (x.ndim == 3 and dim1 == 1 and dim2 == 2) {
            const B = x.shape[0];
            const R = x.shape[1];
            const C = x.shape[2];
            const total = B * R * C;
            const out = self.allocData(total);
            for (0..B) |b| {
                for (0..R) |i| {
                    for (0..C) |j| {
                        out[b * C * R + j * R + i] = x.data[b * R * C + i * C + j];
                    }
                }
            }
            const node = self.makeNode(out, &.{ B, C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backwardTranspose3D;
            }
            return node;
        }

        if (x.ndim == 2 and dim1 == 0 and dim2 == 1) {
            const R = x.shape[0];
            const C = x.shape[1];
            const out = self.allocData(R * C);
            for (0..R) |i| {
                for (0..C) |j| {
                    out[j * R + i] = x.data[i * C + j];
                }
            }
            const node = self.makeNode(out, &.{ C, R }, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backwardTranspose2D;
            }
            return node;
        }

        @panic("transpose: unsupported ndim (expected 2D or 3D)");
    }

    fn backwardTranspose3D(self_node: *DiffNode) void {
        // self_node shape: [B, C, R], parent shape: [B, R, C]
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const B = pa.shape[0];
            const R = pa.shape[1];
            const C = pa.shape[2];
            for (0..B) |b| {
                for (0..R) |i| {
                    for (0..C) |j| {
                        // go[b,j,i] → ga[b,i,j]
                        ga[b * R * C + i * C + j] += go[b * C * R + j * R + i];
                    }
                }
            }
        }
    }

    fn backwardTranspose2D(self_node: *DiffNode) void {
        // self_node shape: [C, R], parent shape: [R, C]
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const R = pa.shape[0];
            const C = pa.shape[1];
            for (0..R) |i| {
                for (0..C) |j| {
                    ga[i * C + j] += go[j * R + i];
                }
            }
        }
    }

    // ── Statistical ops ──

    pub fn softmax(self: *DiffCpuRuntime, x: DiffTensor, axis: i64) DiffTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        const out = self.allocData(total);
        @memcpy(out, x.data[0..total]);
        kernels.softmaxForward(out, rows, cols);

        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            node.backward_fn = &backwardSoftmax;
        }
        return node;
    }

    fn backwardSoftmax(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        const s = self_node.data; // softmax output
        if (pa.grad) |ga| {
            const total = self_node.totalElements();
            const cols = self_node.lastDim();
            const rows = total / cols;
            for (0..rows) |i| {
                // dot = sum(go * s) for this row
                var dot: f32 = 0;
                for (0..cols) |j| dot += go[i * cols + j] * s[i * cols + j];
                // ga += s * (go - dot)
                for (0..cols) |j| {
                    ga[i * cols + j] += s[i * cols + j] * (go[i * cols + j] - dot);
                }
            }
        }
    }

    pub fn logSoftmax(self: *DiffCpuRuntime, x: DiffTensor, axis: i64) DiffTensor {
        _ = axis;
        const total = x.totalElements();
        const cols = x.lastDim();
        const rows = total / cols;
        const out = self.allocData(total);
        const softmax_cache = self.allocData(total);
        kernels.logSoftmaxForward(x.data[0..total], out, rows, cols, softmax_cache);

        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(LogSoftmaxContext);
            ctx.* = .{ .softmax_cache = softmax_cache };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardLogSoftmax;
        }
        return node;
    }

    const LogSoftmaxContext = struct {
        softmax_cache: []f32,
    };

    fn backwardLogSoftmax(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        const ctx: *LogSoftmaxContext = @ptrCast(@alignCast(self_node.context.?));
        const s = ctx.softmax_cache;
        if (pa.grad) |ga| {
            const total = self_node.totalElements();
            const cols = self_node.lastDim();
            const rows = total / cols;
            for (0..rows) |i| {
                // sum_go = sum(go) for this row
                var sum_go: f32 = 0;
                for (0..cols) |j| sum_go += go[i * cols + j];
                // ga += go - s * sum_go
                for (0..cols) |j| {
                    ga[i * cols + j] += go[i * cols + j] - s[i * cols + j] * sum_go;
                }
            }
        }
    }

    pub fn reductionSum(self: *DiffCpuRuntime, x: DiffTensor, axis: i64) DiffTensor {
        const actual_axis: usize = if (axis < 0) @intCast(@as(i64, @intCast(x.ndim)) + axis) else @intCast(axis);

        if (x.ndim == 2) {
            const rows = x.shape[0];
            const cols = x.shape[1];
            if (actual_axis == 1) {
                const out = self.allocData(rows);
                kernels.reductionSumRows(x.data[0 .. rows * cols], out, rows, cols);
                const node = self.makeNode(out, &.{ rows, 1 }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    node.backward_fn = &backwardReductionSumAxis1;
                }
                return node;
            } else {
                const out = self.allocData(cols);
                kernels.reductionSumCols(x.data[0 .. rows * cols], out, rows, cols);
                const node = self.makeNode(out, &.{ 1, cols }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    node.backward_fn = &backwardReductionSumAxis0;
                }
                return node;
            }
        }

        if (x.ndim == 1) {
            const out = self.allocData(1);
            out[0] = kernels.reductionSum1D(x.data[0..x.totalElements()]);
            const node = self.makeNode(out, &.{1}, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backwardReductionSum1D;
            }
            return node;
        }

        // ndim >= 3: flatten around the reduction axis
        if (x.ndim >= 3) {
            const total = x.totalElements();
            // Compute before/axis_dim/after
            var before: usize = 1;
            for (0..actual_axis) |d| before *= x.shape[d];
            const axis_dim = x.shape[actual_axis];
            var after: usize = 1;
            for (actual_axis + 1..x.ndim) |d| after *= x.shape[d];

            if (actual_axis == x.ndim - 1) {
                // reduce last dim: reshape to [total/last, last] then sum axis=1
                const flat = self.reshape(x, &.{ total / axis_dim, axis_dim });
                const reduced = self.reductionSum(flat, 1);
                var new_shape: [8]usize = undefined;
                for (0..x.ndim - 1) |d| new_shape[d] = x.shape[d];
                new_shape[x.ndim - 1] = 1;
                return self.reshape(reduced, new_shape[0..x.ndim]);
            } else if (actual_axis == 0) {
                // reduce first dim: reshape to [first, total/first] then sum axis=0
                const flat = self.reshape(x, &.{ axis_dim, total / axis_dim });
                const reduced = self.reductionSum(flat, 0);
                var new_shape: [8]usize = undefined;
                new_shape[0] = 1;
                for (1..x.ndim) |d| new_shape[d] = x.shape[d];
                return self.reshape(reduced, new_shape[0..x.ndim]);
            } else {
                // middle axis: [before, axis_dim, after] → transpose(1,2) → [before, after, axis_dim]
                // → reshape [before*after, axis_dim] → sum axis=1 → reshape output
                const r3 = self.reshape(x, &.{ before, axis_dim, after });
                const t3 = self.transpose(r3, 1, 2); // [before, after, axis_dim]
                const flat = self.reshape(t3, &.{ before * after, axis_dim });
                const reduced = self.reductionSum(flat, 1); // [before*after, 1]
                // output shape: original with shape[axis] = 1
                var new_shape: [8]usize = undefined;
                for (0..x.ndim) |d| new_shape[d] = x.shape[d];
                new_shape[actual_axis] = 1;
                return self.reshape(reduced, new_shape[0..x.ndim]);
            }
        }

        @panic("reductionSum: unsupported ndim/axis combination");
    }

    fn backwardReductionSumAxis1(self_node: *DiffNode) void {
        // [rows, cols] → [rows, 1]: broadcast go back
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            for (0..rows) |i| {
                for (0..cols) |j| {
                    ga[i * cols + j] += go[i];
                }
            }
        }
    }

    fn backwardReductionSumAxis0(self_node: *DiffNode) void {
        // [rows, cols] → [1, cols]: broadcast go back
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            for (0..rows) |i| {
                for (0..cols) |j| {
                    ga[i * cols + j] += go[j];
                }
            }
        }
    }

    fn backwardReductionSum1D(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const total = pa.totalElements();
            for (0..total) |i| ga[i] += go[0];
        }
    }

    pub fn reductionMean(self: *DiffCpuRuntime, x: DiffTensor, axis: i64) DiffTensor {
        const actual_axis: usize = if (axis < 0) @intCast(@as(i64, @intCast(x.ndim)) + axis) else @intCast(axis);

        if (x.ndim == 2) {
            const rows = x.shape[0];
            const cols = x.shape[1];
            if (actual_axis == 1) {
                const out = self.allocData(rows);
                for (0..rows) |i| {
                    var s: f32 = 0;
                    for (0..cols) |j| s += x.data[i * cols + j];
                    out[i] = s / @as(f32, @floatFromInt(cols));
                }
                const node = self.makeNode(out, &.{ rows, 1 }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    node.backward_fn = &backwardReductionMeanAxis1;
                }
                return node;
            } else {
                const out = self.allocData(cols);
                @memset(out, 0);
                for (0..rows) |i| {
                    for (0..cols) |j| out[j] += x.data[i * cols + j];
                }
                const rows_f: f32 = @floatFromInt(rows);
                for (0..cols) |j| out[j] /= rows_f;
                const node = self.makeNode(out, &.{ 1, cols }, x.requires_grad);
                if (x.requires_grad) {
                    node.parents[0] = x;
                    node.backward_fn = &backwardReductionMeanAxis0;
                }
                return node;
            }
        }

        if (x.ndim == 1) {
            const total_elem = x.totalElements();
            const out = self.allocData(1);
            var s: f32 = 0;
            for (x.data[0..total_elem]) |v| s += v;
            out[0] = s / @as(f32, @floatFromInt(total_elem));
            const node = self.makeNode(out, &.{1}, x.requires_grad);
            if (x.requires_grad) {
                node.parents[0] = x;
                node.backward_fn = &backwardReductionMean1D;
            }
            return node;
        }

        // ndim >= 3: flatten around the reduction axis
        if (x.ndim >= 3) {
            const total = x.totalElements();
            var before: usize = 1;
            for (0..actual_axis) |d| before *= x.shape[d];
            const axis_dim = x.shape[actual_axis];
            var after: usize = 1;
            for (actual_axis + 1..x.ndim) |d| after *= x.shape[d];

            if (actual_axis == x.ndim - 1) {
                const flat = self.reshape(x, &.{ total / axis_dim, axis_dim });
                const reduced = self.reductionMean(flat, 1);
                var new_shape: [8]usize = undefined;
                for (0..x.ndim - 1) |d| new_shape[d] = x.shape[d];
                new_shape[x.ndim - 1] = 1;
                return self.reshape(reduced, new_shape[0..x.ndim]);
            } else if (actual_axis == 0) {
                const flat = self.reshape(x, &.{ axis_dim, total / axis_dim });
                const reduced = self.reductionMean(flat, 0);
                var new_shape: [8]usize = undefined;
                new_shape[0] = 1;
                for (1..x.ndim) |d| new_shape[d] = x.shape[d];
                return self.reshape(reduced, new_shape[0..x.ndim]);
            } else {
                // middle axis: [before, axis_dim, after] → transpose(1,2) → [before, after, axis_dim]
                // → reshape [before*after, axis_dim] → mean axis=1 → reshape output
                const r3 = self.reshape(x, &.{ before, axis_dim, after });
                const t3 = self.transpose(r3, 1, 2);
                const flat = self.reshape(t3, &.{ before * after, axis_dim });
                const reduced = self.reductionMean(flat, 1);
                var new_shape: [8]usize = undefined;
                for (0..x.ndim) |d| new_shape[d] = x.shape[d];
                new_shape[actual_axis] = 1;
                return self.reshape(reduced, new_shape[0..x.ndim]);
            }
        }

        @panic("reductionMean: unsupported ndim/axis combination");
    }

    fn backwardReductionMeanAxis1(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            const cols_f: f32 = @floatFromInt(cols);
            for (0..rows) |i| {
                for (0..cols) |j| {
                    ga[i * cols + j] += go[i] / cols_f;
                }
            }
        }
    }

    fn backwardReductionMeanAxis0(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const rows = pa.shape[0];
            const cols = pa.shape[1];
            const rows_f: f32 = @floatFromInt(rows);
            for (0..rows) |i| {
                for (0..cols) |j| {
                    ga[i * cols + j] += go[j] / rows_f;
                }
            }
        }
    }

    fn backwardReductionMean1D(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const total = pa.totalElements();
            const n_f: f32 = @floatFromInt(total);
            for (0..total) |i| ga[i] += go[0] / n_f;
        }
    }

    // ── LayerNorm ──

    pub fn layerNorm(self: *DiffCpuRuntime, x: DiffTensor, gamma: DiffTensor, beta: DiffTensor, eps: f32, axis: i64) DiffTensor {
        _ = axis;
        const total = x.totalElements();
        const dim = x.lastDim();
        const rows = total / dim;
        const out = self.allocData(total);
        const x_norm = self.allocData(total);
        const inv_stds = self.allocData(rows);
        kernels.layerNormForward(x.data[0..total], out, gamma.data[0..dim], beta.data[0..dim], rows, dim, eps, x_norm, inv_stds);

        const rg = x.requires_grad or gamma.requires_grad or beta.requires_grad;
        const node = self.makeNode(out, x.shape[0..x.ndim], rg);
        if (rg) {
            node.parents[0] = x;
            node.parents[1] = gamma;
            node.parents[2] = beta;
            const ctx = self.allocContext(LayerNormContext);
            ctx.* = .{ .x_norm = x_norm, .inv_stds = inv_stds };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardLayerNorm;
        }
        return node;
    }

    const LayerNormContext = struct {
        x_norm: []f32,
        inv_stds: []f32,
    };

    fn backwardLayerNorm(self_node: *DiffNode) void {
        const px = self_node.parents[0].?;
        const pgamma = self_node.parents[1].?;
        const pbeta = self_node.parents[2].?;
        const go = self_node.grad.?;
        const ctx: *LayerNormContext = @ptrCast(@alignCast(self_node.context.?));
        const x_norm = ctx.x_norm;
        const inv_stds = ctx.inv_stds;

        const total = self_node.totalElements();
        const dim = self_node.lastDim();
        const rows = total / dim;
        const dim_f: f32 = @floatFromInt(dim);

        // dBeta: sum of go over rows
        if (pbeta.grad) |gbeta| {
            for (0..rows) |i| {
                for (0..dim) |j| {
                    gbeta[j] += go[i * dim + j];
                }
            }
        }

        // dGamma: sum of go * x_norm over rows
        if (pgamma.grad) |ggamma| {
            for (0..rows) |i| {
                for (0..dim) |j| {
                    ggamma[j] += go[i * dim + j] * x_norm[i * dim + j];
                }
            }
        }

        // dX: full layerNorm backward
        if (px.grad) |gx| {
            for (0..rows) |i| {
                const inv_std = inv_stds[i];
                // dy = go * gamma
                // mean_dy = mean(dy), mean_dy_xn = mean(dy * x_norm)
                var mean_dy: f32 = 0;
                var mean_dy_xn: f32 = 0;
                for (0..dim) |j| {
                    const dy = go[i * dim + j] * pgamma.data[j];
                    mean_dy += dy;
                    mean_dy_xn += dy * x_norm[i * dim + j];
                }
                mean_dy /= dim_f;
                mean_dy_xn /= dim_f;

                for (0..dim) |j| {
                    const dy = go[i * dim + j] * pgamma.data[j];
                    gx[i * dim + j] += inv_std * (dy - mean_dy - x_norm[i * dim + j] * mean_dy_xn);
                }
            }
        }
    }

    // ── ReLU ──

    pub fn relu(self: *DiffCpuRuntime, x: DiffTensor) DiffTensor {
        return self.unaryOp(x, kernels.reluForward, &backwardRelu);
    }

    fn backwardRelu(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const total = self_node.totalElements();
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            for (0..total) |i| {
                if (pa.data[i] > 0) ga[i] += go[i];
            }
        }
    }

    // ── Dropout ──

    pub fn dropout(self: *DiffCpuRuntime, x: DiffTensor, rate: f32) DiffTensor {
        if (!self.training) return x;

        const total = x.totalElements();
        const out = self.allocData(total);
        const mask = self.allocData(total);
        const rng = self.prng.random();
        const inv_keep = 1.0 / (1.0 - rate);

        for (0..total) |i| {
            if (rng.float(f32) >= rate) {
                mask[i] = inv_keep;
                out[i] = x.data[i] * inv_keep;
            } else {
                mask[i] = 0;
                out[i] = 0;
            }
        }

        const node = self.makeNode(out, x.shape[0..x.ndim], x.requires_grad);
        if (x.requires_grad) {
            node.parents[0] = x;
            const ctx = self.allocContext(DropoutContext);
            ctx.* = .{ .mask = mask };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardDropout;
        }
        return node;
    }

    const DropoutContext = struct {
        mask: []f32,
    };

    fn backwardDropout(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const ctx: *DropoutContext = @ptrCast(@alignCast(self_node.context.?));
            const total = self_node.totalElements();
            for (0..total) |i| {
                ga[i] += go[i] * ctx.mask[i];
            }
        }
    }

    // ── Training mode ──

    pub fn eval(self: *DiffCpuRuntime) void {
        self.training = false;
    }

    pub fn train(self: *DiffCpuRuntime) void {
        self.training = true;
    }

    // ── Gather (Embedding lookup) ──

    /// table: [vocab_size, embed_dim], indices: u32 配列 (non-differentiable)
    /// 出力: [num_indices, embed_dim]
    pub fn gather(self: *DiffCpuRuntime, table: DiffTensor, indices: []const u32) DiffTensor {
        const embed_dim = table.shape[1];
        const num_indices = indices.len;
        const out = self.allocData(num_indices * embed_dim);

        for (0..num_indices) |i| {
            const idx = indices[i];
            const src = table.data[idx * embed_dim .. (idx + 1) * embed_dim];
            @memcpy(out[i * embed_dim .. (i + 1) * embed_dim], src);
        }

        const node = self.makeNode(out, &.{ num_indices, embed_dim }, table.requires_grad);
        if (table.requires_grad) {
            node.parents[0] = table;
            const ctx = self.allocContext(GatherContext);
            ctx.* = .{ .indices = indices };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardGather;
        }
        return node;
    }

    const GatherContext = struct {
        indices: []const u32,
    };

    fn backwardGather(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?; // table
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const ctx: *GatherContext = @ptrCast(@alignCast(self_node.context.?));
            const embed_dim = pa.shape[1];
            // scatter_add: 各 index に対応する行に勾配を加算
            for (0..ctx.indices.len) |i| {
                const idx = ctx.indices[i];
                for (0..embed_dim) |j| {
                    ga[idx * embed_dim + j] += go[i * embed_dim + j];
                }
            }
        }
    }

    // ── MSE Loss ──

    /// pred と target の MSE: mean((pred - target)^2)
    /// target は non-differentiable な f32 スライス
    pub fn mseLoss(self: *DiffCpuRuntime, pred: DiffTensor, target: []const f32) DiffTensor {
        const total = pred.totalElements();
        var sum_sq: f32 = 0;
        for (0..total) |i| {
            const diff = pred.data[i] - target[i];
            sum_sq += diff * diff;
        }
        const n_f: f32 = @floatFromInt(total);
        const out = self.allocData(1);
        out[0] = sum_sq / n_f;

        const node = self.makeNode(out, &.{1}, pred.requires_grad);
        if (pred.requires_grad) {
            node.parents[0] = pred;
            const ctx = self.allocContext(MseLossContext);
            ctx.* = .{ .target = target };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardMseLoss;
        }
        return node;
    }

    const MseLossContext = struct {
        target: []const f32,
    };

    fn backwardMseLoss(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const ctx: *MseLossContext = @ptrCast(@alignCast(self_node.context.?));
            const total = pa.totalElements();
            const n_f: f32 = @floatFromInt(total);
            const scale = go[0] * 2.0 / n_f;
            for (0..total) |i| {
                ga[i] += scale * (pa.data[i] - ctx.target[i]);
            }
        }
    }

    // ── Cross-Entropy Loss with Integer Indices ──

    /// logits: [batch, num_classes], indices: [batch] (u32 class labels)
    /// forward: -mean(log_softmax(logits)[i, indices[i]])
    /// backward: softmax(logits) - one_hot(indices), scaled by 1/batch
    pub fn crossEntropyLossWithIndices(self: *DiffCpuRuntime, logits: DiffTensor, indices: []const u32) DiffTensor {
        const batch = logits.shape[0];
        const num_classes = logits.shape[1];

        // Compute log_softmax and softmax (cache for backward)
        const softmax_cache = self.allocData(batch * num_classes);
        var total_loss: f32 = 0;

        for (0..batch) |i| {
            // Max for numerical stability
            var max_val: f32 = -std.math.inf(f32);
            for (0..num_classes) |j| {
                const v = logits.data[i * num_classes + j];
                if (v > max_val) max_val = v;
            }
            // exp and sum
            var sum_exp: f32 = 0;
            for (0..num_classes) |j| {
                const e = @exp(logits.data[i * num_classes + j] - max_val);
                softmax_cache[i * num_classes + j] = e;
                sum_exp += e;
            }
            // normalize to get softmax
            for (0..num_classes) |j| {
                softmax_cache[i * num_classes + j] /= sum_exp;
            }
            // NLL: -log(softmax[target])
            const log_prob = @log(softmax_cache[i * num_classes + indices[i]] + 1e-10);
            total_loss -= log_prob;
        }

        const batch_f: f32 = @floatFromInt(batch);
        const out = self.allocData(1);
        out[0] = total_loss / batch_f;

        const node = self.makeNode(out, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.allocContext(CrossEntropyContext);
            ctx.* = .{ .softmax_cache = softmax_cache, .indices = indices };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardCrossEntropy;
        }
        return node;
    }

    const CrossEntropyContext = struct {
        softmax_cache: []f32,
        indices: []const u32,
    };

    fn backwardCrossEntropy(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?; // logits
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const ctx: *CrossEntropyContext = @ptrCast(@alignCast(self_node.context.?));
            const batch = pa.shape[0];
            const num_classes = pa.shape[1];
            const batch_f: f32 = @floatFromInt(batch);
            const scale = go[0] / batch_f;
            // grad = (softmax - one_hot) / batch
            for (0..batch) |i| {
                for (0..num_classes) |j| {
                    var g = ctx.softmax_cache[i * num_classes + j];
                    if (j == ctx.indices[i]) g -= 1.0;
                    ga[i * num_classes + j] += scale * g;
                }
            }
        }
    }

    // ── BCE Loss with Logits ──

    /// Binary cross-entropy with logits (numerically stable)
    /// forward: mean(max(x,0) - x*target + log(1+exp(-|x|)))
    /// backward: (sigmoid(x) - target) / n
    pub fn bceLossWithLogits(self: *DiffCpuRuntime, logits: DiffTensor, target: []const f32) DiffTensor {
        const total = logits.totalElements();
        var loss_sum: f32 = 0;
        for (0..total) |i| {
            const x = logits.data[i];
            const t = target[i];
            // Numerically stable: max(x,0) - x*t + log(1+exp(-|x|))
            const pos_part = if (x > 0) x else 0;
            loss_sum += pos_part - x * t + @log(1.0 + @exp(-@abs(x)));
        }
        const n_f: f32 = @floatFromInt(total);
        const out = self.allocData(1);
        out[0] = loss_sum / n_f;

        const node = self.makeNode(out, &.{1}, logits.requires_grad);
        if (logits.requires_grad) {
            node.parents[0] = logits;
            const ctx = self.allocContext(BceLossContext);
            ctx.* = .{ .target = target };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardBceLoss;
        }
        return node;
    }

    const BceLossContext = struct {
        target: []const f32,
    };

    fn backwardBceLoss(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad.?;
        if (pa.grad) |ga| {
            const ctx: *BceLossContext = @ptrCast(@alignCast(self_node.context.?));
            const total = pa.totalElements();
            const n_f: f32 = @floatFromInt(total);
            const scale = go[0] / n_f;
            for (0..total) |i| {
                const sig = 1.0 / (1.0 + @exp(-pa.data[i]));
                ga[i] += scale * (sig - ctx.target[i]);
            }
        }
    }

    // ── Backward (topological sort + reverse traversal) ──

    pub fn backward(self: *DiffCpuRuntime, loss: DiffTensor) void {
        // 1. Set loss gradient to 1.0 (CPU: fill loop)
        const loss_total = loss.totalElements();
        if (loss.grad == null) {
            loss.grad = self.allocGrad(loss_total);
        }
        for (0..loss_total) |i| loss.grad.?[i] = 1.0;

        // 2. Topological sort (DFS)
        self.topo_buf.clearRetainingCapacity();
        diff_node.topoSort(DiffNode, loss, &self.topo_buf, self.allocator);

        // 3. Allocate grad buffers for intermediate nodes (CPU: allocGrad)
        for (self.topo_buf.items) |node| {
            if (node.grad == null and node.requires_grad) {
                const total = node.totalElements();
                node.grad = self.allocGrad(total);
            }
        }

        // 4-5. Reverse traversal + reset visited
        diff_node.backwardPass(DiffNode, &self.topo_buf, self.param_nodes);
    }

    // ── Adam optimizer ──

    pub fn applyAdam(self: *DiffCpuRuntime, adam: *AdamState, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32) void {
        adam.step += 1;
        const count = self.module.paramCount();
        for (0..count) |i| {
            compute.adamStep(
                self.param_nodes[i].data,
                self.param_grads[i],
                adam.m[i],
                adam.v[i],
                lr,
                beta1,
                beta2,
                eps,
                wd,
                adam.step,
            );
        }
    }

    /// Gradient clipping + Adam 適用
    pub fn applyAdamClipped(self: *DiffCpuRuntime, adam: *AdamState, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, max_grad_norm: f32) void {
        const count = self.module.paramCount();

        // 1. Compute total gradient norm
        var total_norm_sq: f64 = 0;
        for (0..count) |i| {
            for (self.param_grads[i]) |v| total_norm_sq += @as(f64, v) * @as(f64, v);
        }

        // 2. Clip gradients
        const total_norm: f32 = @floatCast(@sqrt(total_norm_sq));
        const clip_coef = if (total_norm > max_grad_norm)
            max_grad_norm / (total_norm + 1e-6)
        else
            @as(f32, 1.0);

        if (clip_coef < 1.0) {
            for (0..count) |i| {
                for (self.param_grads[i]) |*v| v.* *= clip_coef;
            }
        }

        // 3. Apply Adam
        self.applyAdam(adam, lr, beta1, beta2, eps, wd);
    }

    // ── Param data access ──

    pub fn paramData(self: *const DiffCpuRuntime, index: usize) []f32 {
        return self.param_nodes[index].data;
    }

    pub fn paramGrad(self: *const DiffCpuRuntime, index: usize) []f32 {
        return self.param_grads[index];
    }

    // ── Tensor creation helpers ──

    /// 全要素 0 のテンソルを作成
    pub fn zeros(self: *DiffCpuRuntime, new_shape: []const usize) DiffTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const out = self.allocData(size);
        @memset(out, 0);
        return self.makeNode(out, new_shape, false);
    }

    /// 全要素 1 のテンソルを作成
    pub fn ones(self: *DiffCpuRuntime, new_shape: []const usize) DiffTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const out = self.allocData(size);
        @memset(out, 1.0);
        return self.makeNode(out, new_shape, false);
    }

    /// 標準正規分布 N(0,1) テンソルを作成 (Box-Muller法)
    pub fn randn(self: *DiffCpuRuntime, new_shape: []const usize) DiffTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const out = self.allocData(size);
        const rng = self.prng.random();
        var i: usize = 0;
        while (i + 1 < size) : (i += 2) {
            const uniform1 = rng.float(f32) * (1.0 - std.math.floatEps(f32)) + std.math.floatEps(f32);
            const uniform2 = rng.float(f32);
            const r = @sqrt(-2.0 * @log(uniform1));
            out[i] = r * @cos(2.0 * std.math.pi * uniform2);
            out[i + 1] = r * @sin(2.0 * std.math.pi * uniform2);
        }
        if (size % 2 == 1) {
            const uniform1 = rng.float(f32) * (1.0 - std.math.floatEps(f32)) + std.math.floatEps(f32);
            const uniform2 = rng.float(f32);
            out[size - 1] = @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * std.math.pi * uniform2);
        }
        return self.makeNode(out, new_shape, false);
    }

    /// uniform [-1, 1] テンソルを作成
    pub fn rand(self: *DiffCpuRuntime, new_shape: []const usize) DiffTensor {
        var size: usize = 1;
        for (new_shape) |d| size *= d;
        const out = self.allocData(size);
        const rng = self.prng.random();
        for (out) |*v| {
            v.* = rng.float(f32) * 2.0 - 1.0;
        }
        return self.makeNode(out, new_shape, false);
    }

    // ── Param initialization ──

    pub fn initParams(self: *DiffCpuRuntime) void {
        var rng_state = std.Random.DefaultPrng.init(42);
        const rng = rng_state.random();

        for (self.module.params.items, 0..) |meta, i| {
            const data = self.param_nodes[i].data;
            switch (meta.init_kind) {
                .ones => @memset(data, 1.0),
                .zeros => @memset(data, 0.0),
                .xavier => {
                    const fan_in: f32 = @floatFromInt(meta.shape[0]);
                    const scale = @sqrt(1.0 / fan_in);
                    for (data) |*val| {
                        val.* = (rng.float(f32) * 2.0 - 1.0) * scale;
                    }
                },
                .kaiming => {
                    const fan_in: f32 = @floatFromInt(meta.shape[0]);
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
    }

    /// MpsRuntime のパラメータデータを CPU にロード
    pub fn loadFromMps(self: *DiffCpuRuntime, mps: anytype) void {
        const metal_mod = @import("backend/metal.zig");
        const MetalContext = metal_mod.MetalContext;
        for (0..self.module.paramCount()) |i| {
            const size = self.module.paramSize(.{ .index = i });
            const ptr = MetalContext.bufferContents(f32, mps.param_bufs[i]);
            @memcpy(self.param_nodes[i].data[0..size], ptr[0..size]);
        }
    }

    /// CpuRuntime からパラメータをロード
    pub fn loadFromCpu(self: *DiffCpuRuntime, cpu: anytype) void {
        for (0..self.module.paramCount()) |i| {
            const size = self.module.paramSize(.{ .index = i });
            @memcpy(self.param_nodes[i].data[0..size], cpu.paramData(i)[0..size]);
        }
    }

    // ── Conv2d / MaxPool2d ──

    const Conv2dContext = struct {
        col: []f32, // im2col result [batch*OH*OW, in_ch*k*k]
        batch: usize,
        in_ch: usize,
        h: usize,
        w: usize,
        oh: usize,
        ow: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    };

    /// Conv2d forward: input [batch, in_ch*H*W], weight [out_ch, in_ch*k*k], bias [out_ch]
    /// Returns [batch, out_ch*OH*OW] flattened
    pub fn conv2d(
        self: *DiffCpuRuntime,
        input: DiffTensor,
        weight: DiffTensor,
        bias: DiffTensor,
        stride_val: usize,
        padding_val: usize,
        kernel_size: usize,
        in_ch: usize,
        out_ch: usize,
        h: usize,
        w: usize,
    ) DiffTensor {
        const batch = input.shape[0];
        const oh = (h + 2 * padding_val - kernel_size) / stride_val + 1;
        const ow = (w + 2 * padding_val - kernel_size) / stride_val + 1;
        const col_row = batch * oh * ow;
        const col_col = in_ch * kernel_size * kernel_size;

        // im2col
        const col = self.allocData(col_row * col_col);
        @memset(col, 0);
        for (0..batch) |b| {
            for (0..oh) |i| {
                for (0..ow) |j| {
                    const row_idx = b * oh * ow + i * ow + j;
                    for (0..in_ch) |c| {
                        for (0..kernel_size) |ki| {
                            for (0..kernel_size) |kj| {
                                const ih = i * stride_val + ki;
                                const iw = j * stride_val + kj;
                                if (ih >= padding_val and ih < h + padding_val and
                                    iw >= padding_val and iw < w + padding_val)
                                {
                                    const src_h = ih - padding_val;
                                    const src_w = iw - padding_val;
                                    col[row_idx * col_col + c * kernel_size * kernel_size + ki * kernel_size + kj] =
                                        input.data[b * (in_ch * h * w) + c * (h * w) + src_h * w + src_w];
                                }
                            }
                        }
                    }
                }
            }
        }

        // col [col_row, col_col] @ weight^T [col_col, out_ch] → out [col_row, out_ch]
        const out_total = col_row * out_ch;
        const out = self.allocData(out_total);
        // weight is [out_ch, col_col], we need col @ weight^T
        cpu_backend.matmulTransB(f32, col.ptr, weight.data.ptr, out.ptr, col_row, col_col, out_ch);

        // Add bias: out[i, j] += bias[j]
        for (0..col_row) |r| {
            for (0..out_ch) |c| {
                out[r * out_ch + c] += bias.data[c];
            }
        }

        // Reshape to [batch, out_ch * oh * ow]
        const node = self.makeNode(out, &.{ batch, out_ch * oh * ow }, input.requires_grad or weight.requires_grad or bias.requires_grad);
        if (node.requires_grad) {
            node.parents[0] = input;
            node.parents[1] = weight;
            node.parents[2] = bias;
            const ctx = self.allocContext(Conv2dContext);
            ctx.* = .{
                .col = col,
                .batch = batch,
                .in_ch = in_ch,
                .h = h,
                .w = w,
                .oh = oh,
                .ow = ow,
                .kernel_size = kernel_size,
                .stride = stride_val,
                .padding = padding_val,
            };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardConv2d;
        }
        return node;
    }

    fn backwardConv2d(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?; // input
        const pw = self_node.parents[1].?; // weight
        const pb = self_node.parents[2].?; // bias
        const go = self_node.grad.?;
        const ctx: *const Conv2dContext = @ptrCast(@alignCast(self_node.context.?));

        const out_ch = pw.shape[0];
        const col_col = pw.shape[1]; // in_ch * k * k
        const col_row = ctx.batch * ctx.oh * ctx.ow;

        // Reshape go to [col_row, out_ch]
        // dBias = sum(go, axis=0)
        if (pb.grad) |gb| {
            for (0..col_row) |r| {
                for (0..out_ch) |c| {
                    gb[c] += go[r * out_ch + c];
                }
            }
        }

        // dWeight = go^T @ col → [out_ch, col_col]
        if (pw.grad) |gw| {
            cpu_backend.matmulTransAAccum(f32, go.ptr, ctx.col.ptr, gw.ptr, out_ch, col_row, col_col);
        }

        // dCol = go @ weight → [col_row, col_col]
        if (pa.grad) |ga| {
            // Temporary buffer for d_col (allocated from arena would be ideal but we have no self ref)
            // We need col2im: scatter d_col back to input gradients
            // d_col = go @ weight
            for (0..col_row) |r| {
                for (0..col_col) |c| {
                    var sum: f32 = 0;
                    for (0..out_ch) |k| {
                        sum += go[r * out_ch + k] * pw.data[k * col_col + c];
                    }
                    // col2im: scatter back
                    const batch_idx = r / (ctx.oh * ctx.ow);
                    const spatial = r % (ctx.oh * ctx.ow);
                    const oi = spatial / ctx.ow;
                    const oj = spatial % ctx.ow;
                    const ch = c / (ctx.kernel_size * ctx.kernel_size);
                    const k_off = c % (ctx.kernel_size * ctx.kernel_size);
                    const ki = k_off / ctx.kernel_size;
                    const kj = k_off % ctx.kernel_size;
                    const ih = oi * ctx.stride + ki;
                    const iw = oj * ctx.stride + kj;
                    if (ih >= ctx.padding and ih < ctx.h + ctx.padding and
                        iw >= ctx.padding and iw < ctx.w + ctx.padding)
                    {
                        const src_h = ih - ctx.padding;
                        const src_w = iw - ctx.padding;
                        ga[batch_idx * (ctx.in_ch * ctx.h * ctx.w) + ch * (ctx.h * ctx.w) + src_h * ctx.w + src_w] += sum;
                    }
                }
            }
        }
    }

    const MaxPool2dContext = struct {
        argmax: []usize,
        batch: usize,
        channels: usize,
        h: usize,
        w: usize,
        oh: usize,
        ow: usize,
        pool_size: usize,
        stride: usize,
    };

    /// MaxPool2d forward: input [batch, ch*H*W], returns [batch, ch*OH*OW]
    pub fn maxPool2d(
        self: *DiffCpuRuntime,
        input: DiffTensor,
        pool_size: usize,
        stride_val: usize,
        channels: usize,
        h: usize,
        w: usize,
    ) DiffTensor {
        const batch = input.shape[0];
        const oh = (h - pool_size) / stride_val + 1;
        const ow = (w - pool_size) / stride_val + 1;
        const out_total = batch * channels * oh * ow;

        const out = self.allocData(out_total);
        const argmax = self.arenaAlloc().alloc(usize, out_total) catch unreachable;

        for (0..batch) |b| {
            for (0..channels) |c| {
                for (0..oh) |i| {
                    for (0..ow) |j| {
                        var max_val: f32 = -std.math.inf(f32);
                        var max_idx: usize = 0;
                        for (0..pool_size) |pi| {
                            for (0..pool_size) |pj| {
                                const ih = i * stride_val + pi;
                                const iw = j * stride_val + pj;
                                const idx = b * (channels * h * w) + c * (h * w) + ih * w + iw;
                                if (input.data[idx] > max_val) {
                                    max_val = input.data[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                        const out_idx = b * (channels * oh * ow) + c * (oh * ow) + i * ow + j;
                        out[out_idx] = max_val;
                        argmax[out_idx] = max_idx;
                    }
                }
            }
        }

        const node = self.makeNode(out, &.{ batch, channels * oh * ow }, input.requires_grad);
        if (input.requires_grad) {
            node.parents[0] = input;
            const ctx = self.allocContext(MaxPool2dContext);
            ctx.* = .{
                .argmax = argmax,
                .batch = batch,
                .channels = channels,
                .h = h,
                .w = w,
                .oh = oh,
                .ow = ow,
                .pool_size = pool_size,
                .stride = stride_val,
            };
            node.context = @ptrCast(ctx);
            node.backward_fn = &backwardMaxPool2d;
        }
        return node;
    }

    fn backwardMaxPool2d(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad orelse return;
        const gx = pa.grad orelse return;
        const ctx: *const MaxPool2dContext = @ptrCast(@alignCast(self_node.context.?));

        const out_total = self_node.totalElements();
        for (0..out_total) |i| {
            gx[ctx.argmax[i]] += go[i];
        }
    }

    // ── Concat / Split ──

    const ConcatContext = struct {
        a_last_dim: usize,
        axis: i64,
    };

    /// 2つのテンソルを指定軸で連結
    /// axis=0: 最初の次元で連結, axis=-1: 最後の次元で連結
    pub fn concat(self: *DiffCpuRuntime, a: DiffTensor, b: DiffTensor) DiffTensor {
        return self.concatAxis(a, b, -1);
    }

    pub fn concatAxis(self: *DiffCpuRuntime, a: DiffTensor, b: DiffTensor, axis: i64) DiffTensor {
        const rg = a.requires_grad or b.requires_grad;

        if (axis == -1 or axis == @as(i64, @intCast(a.ndim)) - 1) {
            // Last-axis concat
            const a_cols = a.lastDim();
            const b_cols = b.lastDim();
            const rows = a.numRows();
            const out_cols = a_cols + b_cols;
            const out = self.allocData(rows * out_cols);

            for (0..rows) |r| {
                @memcpy(out[r * out_cols ..][0..a_cols], a.data[r * a_cols ..][0..a_cols]);
                @memcpy(out[r * out_cols + a_cols ..][0..b_cols], b.data[r * b_cols ..][0..b_cols]);
            }

            var out_shape: [MAX_NDIM]usize = a.shape;
            out_shape[a.ndim - 1] = out_cols;
            const node = self.makeNode(out, out_shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(ConcatContext);
                ctx.* = .{ .a_last_dim = a_cols, .axis = -1 };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardConcatLastAxis;
            }
            return node;
        }

        if (axis == 0) {
            // First-axis concat: just concatenate data
            const a_total = a.totalElements();
            const b_total = b.totalElements();
            const out = self.allocData(a_total + b_total);
            @memcpy(out[0..a_total], a.data[0..a_total]);
            @memcpy(out[a_total..][0..b_total], b.data[0..b_total]);

            var out_shape: [MAX_NDIM]usize = a.shape;
            out_shape[0] = a.shape[0] + b.shape[0];
            const node = self.makeNode(out, out_shape[0..a.ndim], rg);
            if (rg) {
                node.parents[0] = a;
                node.parents[1] = b;
                const ctx = self.allocContext(ConcatContext);
                ctx.* = .{ .a_last_dim = a_total, .axis = 0 };
                node.context = @ptrCast(ctx);
                node.backward_fn = &backwardConcatAxis0;
            }
            return node;
        }

        @panic("concatAxis: unsupported ndim or axis combination");
    }

    fn backwardConcatLastAxis(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const ctx: *const ConcatContext = @ptrCast(@alignCast(self_node.context.?));
        const a_cols = ctx.a_last_dim;
        const out_cols = self_node.lastDim();
        const b_cols = out_cols - a_cols;
        const rows = self_node.numRows();

        if (pa.grad) |ga| {
            for (0..rows) |r| {
                for (0..a_cols) |c| {
                    ga[r * a_cols + c] += go[r * out_cols + c];
                }
            }
        }
        if (pb.grad) |gb| {
            for (0..rows) |r| {
                for (0..b_cols) |c| {
                    gb[r * b_cols + c] += go[r * out_cols + a_cols + c];
                }
            }
        }
    }

    fn backwardConcatAxis0(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const pb = self_node.parents[1].?;
        const go = self_node.grad.?;
        const a_total = pa.totalElements();
        const b_total = pb.totalElements();

        if (pa.grad) |ga| {
            for (0..a_total) |i| ga[i] += go[i];
        }
        if (pb.grad) |gb| {
            for (0..b_total) |i| gb[i] += go[a_total + i];
        }
    }

    const SplitContext = struct {
        split_pos: usize,
        axis: i64,
        sibling: *DiffNode,
    };

    /// テンソルを指定位置で2つに分割 (concatの逆操作)
    /// axis=-1: 最後の次元で分割, axis=0: 最初の次元で分割
    pub fn split(self: *DiffCpuRuntime, x: DiffTensor, split_pos: usize, axis: i64) [2]DiffTensor {
        if (axis == -1 or axis == @as(i64, @intCast(x.ndim)) - 1) {
            const cols = x.lastDim();
            const rows = x.numRows();
            const a_cols = split_pos;
            const b_cols = cols - split_pos;

            const out_a = self.allocData(rows * a_cols);
            const out_b = self.allocData(rows * b_cols);

            for (0..rows) |r| {
                @memcpy(out_a[r * a_cols ..][0..a_cols], x.data[r * cols ..][0..a_cols]);
                @memcpy(out_b[r * b_cols ..][0..b_cols], x.data[r * cols + a_cols ..][0..b_cols]);
            }

            var shape_a: [MAX_NDIM]usize = x.shape;
            shape_a[x.ndim - 1] = a_cols;
            var shape_b: [MAX_NDIM]usize = x.shape;
            shape_b[x.ndim - 1] = b_cols;

            const node_a = self.makeNode(out_a, shape_a[0..x.ndim], x.requires_grad);
            const node_b = self.makeNode(out_b, shape_b[0..x.ndim], x.requires_grad);

            if (x.requires_grad) {
                node_a.parents[0] = x;
                const ctx_a = self.allocContext(SplitContext);
                ctx_a.* = .{ .split_pos = split_pos, .axis = -1, .sibling = node_b };
                node_a.context = @ptrCast(ctx_a);
                node_a.backward_fn = &backwardSplitLastAxisFirst;

                node_b.parents[0] = x;
                const ctx_b = self.allocContext(SplitContext);
                ctx_b.* = .{ .split_pos = split_pos, .axis = -1, .sibling = node_a };
                node_b.context = @ptrCast(ctx_b);
                node_b.backward_fn = &backwardSplitLastAxisSecond;
            }

            return .{ node_a, node_b };
        }

        if (axis == 0) {
            const a_rows = split_pos;
            const b_rows = x.shape[0] - split_pos;
            var stride: usize = 1;
            for (1..x.ndim) |d| stride *= x.shape[d];
            const a_total = a_rows * stride;
            const b_total = b_rows * stride;

            const out_a = self.allocData(a_total);
            const out_b = self.allocData(b_total);
            @memcpy(out_a[0..a_total], x.data[0..a_total]);
            @memcpy(out_b[0..b_total], x.data[a_total..][0..b_total]);

            var shape_a: [MAX_NDIM]usize = x.shape;
            shape_a[0] = a_rows;
            var shape_b: [MAX_NDIM]usize = x.shape;
            shape_b[0] = b_rows;

            const node_a = self.makeNode(out_a, shape_a[0..x.ndim], x.requires_grad);
            const node_b = self.makeNode(out_b, shape_b[0..x.ndim], x.requires_grad);

            if (x.requires_grad) {
                node_a.parents[0] = x;
                const ctx_a = self.allocContext(SplitContext);
                ctx_a.* = .{ .split_pos = a_total, .axis = 0, .sibling = node_b };
                node_a.context = @ptrCast(ctx_a);
                node_a.backward_fn = &backwardSplitAxis0First;

                node_b.parents[0] = x;
                const ctx_b = self.allocContext(SplitContext);
                ctx_b.* = .{ .split_pos = a_total, .axis = 0, .sibling = node_a };
                node_b.context = @ptrCast(ctx_b);
                node_b.backward_fn = &backwardSplitAxis0Second;
            }

            return .{ node_a, node_b };
        }

        @panic("splitAxis: unsupported ndim or axis combination");
    }

    fn backwardSplitLastAxisFirst(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad orelse return;
        const gx = pa.grad orelse return;
        const ctx: *const SplitContext = @ptrCast(@alignCast(self_node.context.?));
        const a_cols = ctx.split_pos;
        const x_cols = pa.lastDim();
        const rows = pa.numRows();

        for (0..rows) |r| {
            for (0..a_cols) |c| {
                gx[r * x_cols + c] += go[r * a_cols + c];
            }
        }
    }

    fn backwardSplitLastAxisSecond(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad orelse return;
        const gx = pa.grad orelse return;
        const ctx: *const SplitContext = @ptrCast(@alignCast(self_node.context.?));
        const a_cols = ctx.split_pos;
        const x_cols = pa.lastDim();
        const b_cols = x_cols - a_cols;
        const rows = pa.numRows();

        for (0..rows) |r| {
            for (0..b_cols) |c| {
                gx[r * x_cols + a_cols + c] += go[r * b_cols + c];
            }
        }
    }

    fn backwardSplitAxis0First(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad orelse return;
        const gx = pa.grad orelse return;
        const ctx: *const SplitContext = @ptrCast(@alignCast(self_node.context.?));
        const a_total = ctx.split_pos;

        for (0..a_total) |i| gx[i] += go[i];
    }

    fn backwardSplitAxis0Second(self_node: *DiffNode) void {
        const pa = self_node.parents[0].?;
        const go = self_node.grad orelse return;
        const gx = pa.grad orelse return;
        const ctx: *const SplitContext = @ptrCast(@alignCast(self_node.context.?));
        const a_total = ctx.split_pos;
        const b_total = self_node.totalElements();

        for (0..b_total) |i| gx[a_total + i] += go[i];
    }

    /// Checkpoint 保存
    pub fn saveCheckpoint(self: *const DiffCpuRuntime, io: std.Io, adam: *const AdamState, path: []const u8) !void {
        const count = self.module.paramCount();
        const slices = try self.allocator.alloc([]const f32, count);
        defer self.allocator.free(slices);
        for (0..count) |i| {
            slices[i] = self.param_nodes[i].data;
        }
        try compute.saveCheckpoint(self.module, io, slices, adam, path);
    }

    /// Checkpoint 読み込み
    pub fn loadCheckpoint(self: *DiffCpuRuntime, io: std.Io, adam: *AdamState, path: []const u8) !void {
        const count = self.module.paramCount();
        const slices = try self.allocator.alloc([]f32, count);
        defer self.allocator.free(slices);
        for (0..count) |i| {
            slices[i] = self.param_nodes[i].data;
        }
        try compute.loadCheckpoint(self.module, io, slices, adam, path);
    }
};

test {
    _ = @import("diff_cpu_runtime_test.zig");
}
