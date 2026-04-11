/// mps_runtime.zig: MPSGraph GPU ランタイム
///
/// compute.Module + MPSGraphContext を束ねて、統一モデルの forward() に渡す GPU バックエンド。
/// 全 op は MPSGraphContext への1行デリゲート。param() は placeholder tensor を返す。
const std = @import("std");
const Allocator = std.mem.Allocator;
const compute = @import("compute.zig");
const Module = compute.Module;
const ParamHandle = compute.ParamHandle;
const AdamState = compute.AdamState;
const metal_mod = @import("backend/metal.zig");
const MetalContext = metal_mod.MetalContext;
const mps_graph = @import("backend/mps_graph.zig");
const MPSGraphContext = mps_graph.MPSGraphContext;
const MPSDataTypeFloat32 = mps_graph.MPSDataTypeFloat32;
const metal_id = metal_mod.id;

pub const MpsRuntime = struct {
    ctx: *MPSGraphContext,
    module: *const Module,
    allocator: Allocator,
    /// Param placeholders (1:1 with module.params)
    tensors: []metal_id,
    /// MTLBuffer for each param (weights)
    param_bufs: []metal_id,
    training: bool = true,
    /// Dropout rate placeholder (scalar [1])
    dropout_rate_ph: ?metal_id = null,
    /// MTLBuffer for dropout rate value
    dropout_rate_buf: ?metal_id = null,
    /// 実際の dropout 率 (train 復帰用)
    dropout_actual_rate: f32 = 0,
    /// MetalContext 参照 (buffer 作成用)
    mtl: ?*MetalContext = null,

    pub fn init(ctx: *MPSGraphContext, module: *const Module, allocator: Allocator) !MpsRuntime {
        const count = module.paramCount();
        const tensors = try allocator.alloc(metal_id, count);
        for (module.params.items, 0..) |meta, i| {
            tensors[i] = ctx.placeholder(meta.shape, MPSDataTypeFloat32);
        }
        return .{
            .ctx = ctx,
            .module = module,
            .allocator = allocator,
            .tensors = tensors,
            .param_bufs = try allocator.alloc(metal_id, count),
        };
    }

    pub fn deinit(self: *MpsRuntime) void {
        self.allocator.free(self.tensors);
        self.allocator.free(self.param_bufs);
    }

    // ── Param access ──

    pub fn param(self: *const MpsRuntime, handle: ParamHandle) metal_id {
        return self.tensors[handle.index];
    }

    // ── Ops (delegates to MPSGraphContext) ──

    pub fn add(self: *MpsRuntime, a: metal_id, b: metal_id) metal_id {
        return self.ctx.add(a, b);
    }

    pub fn mul(self: *MpsRuntime, a: metal_id, b: metal_id) metal_id {
        return self.ctx.mul(a, b);
    }

    pub fn sub(self: *MpsRuntime, a: metal_id, b: metal_id) metal_id {
        return self.ctx.sub(a, b);
    }

    pub fn matmul(self: *MpsRuntime, a: metal_id, b: metal_id) metal_id {
        return self.ctx.matmul(a, b);
    }

    pub fn gelu(self: *MpsRuntime, x: metal_id) metal_id {
        return self.ctx.gelu(x);
    }

    pub fn silu(self: *MpsRuntime, x: metal_id) metal_id {
        return self.ctx.silu(x);
    }

    pub fn tanh_(self: *MpsRuntime, x: metal_id) metal_id {
        return self.ctx.tanh_(x);
    }

    pub fn sigmoid(self: *MpsRuntime, x: metal_id) metal_id {
        return self.ctx.sigmoid(x);
    }

    pub fn square(self: *MpsRuntime, x: metal_id) metal_id {
        return self.ctx.square(x);
    }

    pub fn reductionMean(self: *MpsRuntime, x: metal_id, axis: i64) metal_id {
        return self.ctx.reductionMean(x, &.{axis});
    }

    pub fn reshape(self: *MpsRuntime, x: metal_id, new_shape: []const usize) metal_id {
        return self.ctx.reshape(x, new_shape);
    }

    pub fn transpose(self: *MpsRuntime, x: metal_id, d1: u64, d2: u64) metal_id {
        return self.ctx.transpose(x, d1, d2);
    }

    pub fn softmax(self: *MpsRuntime, x: metal_id, axis: i64) metal_id {
        return self.ctx.softmax(x, axis);
    }

    pub fn logSoftmax(self: *MpsRuntime, x: metal_id, axis: i64) metal_id {
        return self.ctx.logSoftmax(x, axis);
    }

    pub fn layerNorm(self: *MpsRuntime, x: metal_id, gamma: metal_id, beta: metal_id, eps: f32, axis: i64) metal_id {
        return self.ctx.layerNorm(x, gamma, beta, eps, axis);
    }

    pub fn constantScalar(self: *MpsRuntime, val: f64, dtype: u32) metal_id {
        return self.ctx.constantScalar(val, dtype);
    }

    pub fn constantData(self: *MpsRuntime, data: [*]const u8, len: usize, new_shape: []const usize, dtype: u32) metal_id {
        return self.ctx.constantData(data, len, new_shape, dtype);
    }

    pub fn negative(self: *MpsRuntime, x: metal_id) metal_id {
        return self.ctx.negative(x);
    }

    pub fn reductionSum(self: *MpsRuntime, x: metal_id, axis: i64) metal_id {
        return self.ctx.reductionSum(x, axis);
    }

    pub fn stopGradient(self: *MpsRuntime, x: metal_id) metal_id {
        return self.ctx.stopGradient(x);
    }

    pub fn relu(self: *MpsRuntime, x: metal_id) metal_id {
        // ReLU = max(x, 0)
        const zero = self.ctx.constantScalar(0.0, MPSDataTypeFloat32);
        return self.ctx.maximum(x, zero);
    }

    pub fn gather(self: *MpsRuntime, table: metal_id, indices: []const u32) metal_id {
        // Create constant tensor from indices
        const idx_tensor = self.ctx.constantData(
            @ptrCast(indices.ptr),
            indices.len * @sizeOf(u32),
            &.{indices.len},
            mps_graph.MPSDataTypeInt32,
        );
        return self.ctx.gather(table, idx_tensor, 0);
    }

    pub fn mseLoss(self: *MpsRuntime, pred: metal_id, target: []const f32) metal_id {
        const target_tensor = self.ctx.constantData(
            @ptrCast(target.ptr),
            target.len * @sizeOf(f32),
            &.{target.len},
            MPSDataTypeFloat32,
        );
        return self.ctx.mseLoss(pred, target_tensor);
    }

    pub fn crossEntropyLossWithIndices(self: *MpsRuntime, logits: metal_id, indices: []const u32) metal_id {
        // Convert u32 indices to one-hot labels for MPSGraph crossEntropyLoss
        const idx_tensor = self.ctx.constantData(
            @ptrCast(indices.ptr),
            indices.len * @sizeOf(u32),
            &.{indices.len},
            mps_graph.MPSDataTypeInt32,
        );
        const labels = self.ctx.oneHot(idx_tensor, 0);
        return self.ctx.crossEntropyLoss(logits, labels, -1);
    }

    pub fn dropout(self: *MpsRuntime, x: metal_id, rate: f32) metal_id {
        if (rate == 0.0) return x;
        // Lazy init: 最初の dropout 呼び出し時に placeholder + buffer 作成
        if (self.dropout_rate_ph == null) {
            self.dropout_rate_ph = self.ctx.placeholder(&.{1}, MPSDataTypeFloat32);
            self.dropout_rate_buf = (self.mtl orelse unreachable).createBuffer(@sizeOf(f32)) catch unreachable;
            self.dropout_actual_rate = rate;
            // 初期値: training mode → actual rate
            MetalContext.bufferContents(f32, self.dropout_rate_buf.?)[0] = rate;
        }
        return self.ctx.dropout(x, self.dropout_rate_ph.?);
    }

    pub fn eval(self: *MpsRuntime) void {
        self.training = false;
        if (self.dropout_rate_buf) |buf|
            MetalContext.bufferContents(f32, buf)[0] = 0.0;
    }

    pub fn train(self: *MpsRuntime) void {
        self.training = true;
        if (self.dropout_rate_buf) |buf|
            MetalContext.bufferContents(f32, buf)[0] = self.dropout_actual_rate;
    }

    pub fn bceLossWithLogits(self: *MpsRuntime, logits: metal_id, target: []const f32) metal_id {
        const target_tensor = self.ctx.constantData(
            @ptrCast(target.ptr),
            target.len * @sizeOf(f32),
            &.{target.len},
            MPSDataTypeFloat32,
        );
        // BCE = mean(max(x,0) - x*t + log(1+exp(-|x|)))
        // Construct: sigmoid(logits) → -target*log(sig) - (1-target)*log(1-sig) → mean
        const sig = self.ctx.sigmoid(logits);
        const log_sig = self.ctx.log(sig);
        const one = self.ctx.constantScalar(1.0, MPSDataTypeFloat32);
        const one_minus_sig = self.ctx.sub(one, sig);
        const log_one_minus_sig = self.ctx.log(one_minus_sig);
        const one_minus_target = self.ctx.sub(one, target_tensor);
        // -target*log(sig) - (1-target)*log(1-sig)
        const term1 = self.ctx.mul(target_tensor, log_sig);
        const term2 = self.ctx.mul(one_minus_target, log_one_minus_sig);
        const neg_loss = self.ctx.add(term1, term2);
        const loss = self.ctx.negative(neg_loss);
        // mean over all elements
        return self.ctx.reductionMean(loss, &.{-1});
    }

    // ── MPS-only: param initialization ──

    /// MTLBuffer 確保 + Xavier/zeros/ones 初期化
    pub fn initParams(self: *MpsRuntime, mtl: *MetalContext) !void {
        self.mtl = mtl;
        var rng_state = std.Random.DefaultPrng.init(42);
        const rng = rng_state.random();

        for (self.module.params.items, 0..) |meta, i| {
            var size: usize = 1;
            for (meta.shape) |d| size *= d;

            const buf = try mtl.createBuffer(size * @sizeOf(f32));
            self.param_bufs[i] = buf;

            const ptr = MetalContext.bufferContents(f32, buf);
            const data = ptr[0..size];

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

    /// Adam state 確保
    pub fn initAdamState(self: *const MpsRuntime) !AdamState {
        const sizes = try self.module.paramSizes(self.allocator);
        defer self.allocator.free(sizes);
        return AdamState.init(self.allocator, sizes);
    }

    // ── MPS-only: graph execution ──

    /// Auto-diff: loss に対する全パラメータの勾配
    pub fn gradients(self: *MpsRuntime, loss_tensor: metal_id) []metal_id {
        return self.ctx.gradients(loss_tensor, self.tensors);
    }

    /// Param feeds を構築 (self の param_bufs を使用)
    pub fn buildParamFeeds(self: *const MpsRuntime) ![]MPSGraphContext.Feed {
        return self.buildParamFeedsFrom(self.param_bufs);
    }

    /// Param feeds を構築 (外部の param_bufs を使用 — 推論グラフでの共有用)
    pub fn buildParamFeedsFrom(self: *const MpsRuntime, ext_param_bufs: []metal_id) ![]MPSGraphContext.Feed {
        const count = self.module.paramCount();
        const extra: usize = if (self.dropout_rate_ph != null) 1 else 0;
        const feeds = try self.allocator.alloc(MPSGraphContext.Feed, count + extra);
        for (self.module.params.items, 0..) |meta, i| {
            feeds[i] = .{
                .tensor = self.tensors[i],
                .buffer = ext_param_bufs[i],
                .shape = meta.shape,
                .dtype = MPSDataTypeFloat32,
            };
        }
        if (self.dropout_rate_ph) |ph| {
            feeds[count] = .{
                .tensor = ph,
                .buffer = self.dropout_rate_buf.?,
                .shape = &.{1},
                .dtype = MPSDataTypeFloat32,
            };
        }
        return feeds;
    }

    /// 全 param placeholder 配列
    pub fn allParamTensors(self: *const MpsRuntime) []metal_id {
        return self.tensors;
    }

    /// Graph 実行
    pub fn run(self: *MpsRuntime, feeds: []const MPSGraphContext.Feed, targets: []const metal_id) []metal_id {
        return self.ctx.run(feeds, targets);
    }

    /// Gradient 読み出し + Adam 適用
    pub fn applyAdam(
        self: *MpsRuntime,
        results: []const metal_id,
        grad_offset: usize,
        adam: *AdamState,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) !void {
        adam.step += 1;
        const count = self.module.paramCount();

        for (0..count) |i| {
            const size = self.module.paramSize(.{ .index = i });

            const grad_data = try self.allocator.alloc(f32, size);
            defer self.allocator.free(grad_data);

            MPSGraphContext.readTensorData(results[grad_offset + i], grad_data);

            const param_ptr = MetalContext.bufferContents(f32, self.param_bufs[i]);
            compute.adamStep(param_ptr[0..size], grad_data, adam.m[i], adam.v[i], lr, beta1, beta2, eps, wd, adam.step);
        }
    }

    /// Gradient clipping + Adam 適用
    pub fn applyAdamClipped(
        self: *MpsRuntime,
        results: []const metal_id,
        grad_offset: usize,
        adam: *AdamState,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        max_grad_norm: f32,
    ) !void {
        adam.step += 1;
        const count = self.module.paramCount();

        // 1. Read all gradients
        const grad_bufs = try self.allocator.alloc([]f32, count);
        defer {
            for (grad_bufs) |g| self.allocator.free(g);
            self.allocator.free(grad_bufs);
        }

        var total_norm_sq: f64 = 0;
        for (0..count) |i| {
            const size = self.module.paramSize(.{ .index = i });
            grad_bufs[i] = try self.allocator.alloc(f32, size);
            MPSGraphContext.readTensorData(results[grad_offset + i], grad_bufs[i]);
            for (grad_bufs[i]) |v| total_norm_sq += @as(f64, v) * @as(f64, v);
        }

        // 2. Clip gradients
        const total_norm: f32 = @floatCast(@sqrt(total_norm_sq));
        const clip_coef = if (total_norm > max_grad_norm)
            max_grad_norm / (total_norm + 1e-6)
        else
            @as(f32, 1.0);

        if (clip_coef < 1.0) {
            for (0..count) |i| {
                for (grad_bufs[i]) |*v| v.* *= clip_coef;
            }
        }

        // 3. Apply Adam
        for (0..count) |i| {
            const size = self.module.paramSize(.{ .index = i });
            const param_ptr = MetalContext.bufferContents(f32, self.param_bufs[i]);
            compute.adamStep(param_ptr[0..size], grad_bufs[i], adam.m[i], adam.v[i], lr, beta1, beta2, eps, wd, adam.step);
        }
    }

    /// パラメータデータ (f32 slices) を取得 — checkpoint 保存用
    pub fn getParamDataSlices(self: *const MpsRuntime) ![][]const f32 {
        const count = self.module.paramCount();
        const slices = try self.allocator.alloc([]const f32, count);
        for (0..count) |i| {
            const size = self.module.paramSize(.{ .index = i });
            const ptr = MetalContext.bufferContents(f32, self.param_bufs[i]);
            slices[i] = ptr[0..size];
        }
        return slices;
    }

    /// Checkpoint 保存
    pub fn saveCheckpoint(self: *const MpsRuntime, adam: *const AdamState, path: []const u8) !void {
        const slices = try self.getParamDataSlices();
        defer self.allocator.free(slices);
        try compute.saveCheckpoint(self.module, slices, adam, path);
    }

    /// Checkpoint 読み込み
    pub fn loadCheckpoint(self: *MpsRuntime, adam: *AdamState, path: []const u8) !void {
        const count = self.module.paramCount();
        const slices = try self.allocator.alloc([]f32, count);
        defer self.allocator.free(slices);
        for (0..count) |i| {
            const size = self.module.paramSize(.{ .index = i });
            const ptr = MetalContext.bufferContents(f32, self.param_bufs[i]);
            slices[i] = ptr[0..size];
        }
        try compute.loadCheckpoint(self.module, slices, adam, path);
    }
};
