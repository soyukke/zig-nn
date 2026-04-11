/// Trainer: PyTorch ライクな training ループヘルパー
///
/// Module + DiffCpuRuntime + AdamState + Config を1つにまとめ、
/// training ループのボイラープレートを削減する。
///
/// Usage:
///   const MLP = Sequential(.{ Linear_(2, 16), ReLU, Linear_(16, 1) });
///   var trainer = try Trainer(MLP).init(allocator, .{ .lr = 1e-3 });
///   defer trainer.deinit();
///
///   for (0..epochs) |_| {
///       trainer.zeroGrad();
///       const out = trainer.forward(trainer.tensor(&input, &.{1, 2}));
///       const loss = trainer.mseLoss(out, &target);
///       trainer.backward(loss);
///       trainer.step();
///   }
const std = @import("std");
const Allocator = std.mem.Allocator;
const compute = @import("compute.zig");
const Module = compute.Module;
const AdamState = compute.AdamState;
const diff_cpu = @import("diff_cpu_runtime.zig");
const DiffCpuRuntime = diff_cpu.DiffCpuRuntime;
const DiffTensor = diff_cpu.DiffTensor;

pub fn Trainer(comptime ModelType: type) type {
    return struct {
        allocator: Allocator,
        module: *Module,
        model: ModelType,
        rt: *DiffCpuRuntime,
        adam: AdamState,
        config: Config,

        pub const Config = struct {
            lr: f32 = 1e-3,
            beta1: f32 = 0.9,
            beta2: f32 = 0.999,
            eps: f32 = 1e-8,
            weight_decay: f32 = 0,
            max_grad_norm: ?f32 = null,
        };

        pub fn init(allocator: Allocator, config: Config) !@This() {
            // Heap-allocate Module and Runtime for pointer stability
            const module = try allocator.create(Module);
            module.* = Module.init(allocator);
            const model = ModelType.init(module);

            const rt = try allocator.create(DiffCpuRuntime);
            rt.* = try DiffCpuRuntime.init(module, allocator);
            rt.initParams();

            const sizes = try module.paramSizes(allocator);
            defer allocator.free(sizes);
            const adam = try AdamState.init(allocator, sizes);

            return .{
                .allocator = allocator,
                .module = module,
                .model = model,
                .rt = rt,
                .adam = adam,
                .config = config,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.adam.deinit();
            self.rt.deinit();
            self.allocator.destroy(self.rt);
            self.module.deinit();
            self.allocator.destroy(self.module);
        }

        // ── Core training loop ──

        /// 勾配をゼロ化 + 中間テンソルを解放 (毎 iteration の先頭で呼ぶ)
        pub fn zeroGrad(self: *@This()) void {
            self.rt.zeroGrad();
            self.rt.resetArena();
        }

        /// モデルの forward pass
        pub fn forward(self: *@This(), x: DiffTensor) DiffTensor {
            return self.model.forward(self.rt, x);
        }

        /// 損失からの逆伝播
        pub fn backward(self: *@This(), loss: DiffTensor) void {
            self.rt.backward(loss);
        }

        /// Optimizer step (Adam, gradient clipping 対応)
        pub fn step(self: *@This()) void {
            const c = self.config;
            if (c.max_grad_norm) |max_norm| {
                self.rt.applyAdamClipped(&self.adam, c.lr, c.beta1, c.beta2, c.eps, c.weight_decay, max_norm);
            } else {
                self.rt.applyAdam(&self.adam, c.lr, c.beta1, c.beta2, c.eps, c.weight_decay);
            }
        }

        // ── Mode ──

        pub fn eval(self: *@This()) void {
            self.rt.eval();
        }

        pub fn train(self: *@This()) void {
            self.rt.train();
        }

        // ── Tensor creation ──

        pub fn tensor(self: *@This(), data: []f32, shape: []const usize) DiffTensor {
            return self.rt.makeTensor(data, shape);
        }

        pub fn param(self: *@This(), handle: compute.ParamHandle) DiffTensor {
            return self.rt.param(handle);
        }

        // ── Loss functions ──

        pub fn mseLoss(self: *@This(), pred: DiffTensor, target: []const f32) DiffTensor {
            return self.rt.mseLoss(pred, target);
        }

        pub fn crossEntropyLoss(self: *@This(), logits: DiffTensor, indices: []const u32) DiffTensor {
            return self.rt.crossEntropyLossWithIndices(logits, indices);
        }

        pub fn bceLoss(self: *@This(), logits: DiffTensor, target: []const f32) DiffTensor {
            return self.rt.bceLossWithLogits(logits, target);
        }

        // ── Custom loss ──

        /// 事前計算された loss から backward + step を一括実行
        pub fn backwardAndStep(self: *@This(), loss: DiffTensor) f32 {
            self.backward(loss);
            self.step();
            return loss.data[0];
        }

        // ── Checkpoint ──

        pub fn save(self: *@This(), path: []const u8) !void {
            try self.rt.saveCheckpoint(&self.adam, path);
        }

        pub fn load(self: *@This(), path: []const u8) !void {
            try self.rt.loadCheckpoint(&self.adam, path);
        }

        // ── Learning rate ──

        pub fn setLr(self: *@This(), lr: f32) void {
            self.config.lr = lr;
        }

        pub fn getLr(self: *const @This()) f32 {
            return self.config.lr;
        }

        /// Cosine annealing LR scheduler step
        pub fn cosineAnnealingStep(self: *@This(), current_step: u32, total_steps: u32, lr_min: f32, lr_max: f32) void {
            self.config.lr = compute.cosineAnnealingLR(current_step, total_steps, lr_min, lr_max);
        }

        /// Warmup + cosine decay LR scheduler step
        pub fn warmupCosineStep(self: *@This(), current_step: u32, warmup_steps: u32, total_steps: u32, lr_min: f32, lr_max: f32) void {
            self.config.lr = compute.warmupCosineDecayLR(current_step, warmup_steps, total_steps, lr_min, lr_max);
        }

        // ── Utilities ──

        pub fn paramCount(self: *const @This()) usize {
            return self.module.paramCount();
        }

        /// 全パラメータの総要素数
        pub fn totalParamElements(self: *const @This()) usize {
            return self.module.totalParamElements();
        }
    };
}

// ── Tests ──

const testing = std.testing;
const Sequential = @import("nn/graph_sequential.zig").Sequential;
const ReLU = @import("nn/graph_sequential.zig").ReLU;
const Linear_ = @import("nn/graph_linear.zig").Linear;

test "Trainer: XOR-like training loop" {
    const MLP = Sequential(.{
        Linear_(2, 8),
        ReLU,
        Linear_(8, 1),
    });

    var trainer = try Trainer(MLP).init(testing.allocator, .{ .lr = 1e-2 });
    defer trainer.deinit();

    try testing.expectEqual(@as(usize, 4), trainer.paramCount()); // W1, b1, W2, b2

    // Simple training: minimize MSE on one sample
    var input_data = [_]f32{ 1.0, 0.0 };
    const target = [_]f32{1.0};

    var initial_loss: f32 = undefined;
    var final_loss: f32 = undefined;

    for (0..100) |i| {
        trainer.zeroGrad();
        const out = trainer.forward(trainer.tensor(&input_data, &.{ 1, 2 }));
        const loss = trainer.mseLoss(out, &target);
        if (i == 0) initial_loss = loss.data[0];
        if (i == 99) final_loss = loss.data[0];
        trainer.backward(loss);
        trainer.step();
    }

    // Loss should decrease
    try testing.expect(final_loss < initial_loss);
}

test "Trainer: gradient clipping" {
    const Model = Sequential(.{Linear_(2, 1)});

    var trainer = try Trainer(Model).init(testing.allocator, .{
        .lr = 1e-2,
        .max_grad_norm = 1.0,
    });
    defer trainer.deinit();

    var input_data = [_]f32{ 10.0, 10.0 }; // large input to produce large gradients
    const target = [_]f32{0.0};

    trainer.zeroGrad();
    const out = trainer.forward(trainer.tensor(&input_data, &.{ 1, 2 }));
    const loss = trainer.mseLoss(out, &target);
    trainer.backward(loss);
    trainer.step(); // should not crash with large gradients
    try testing.expect(loss.data[0] > 0);
}

test "Trainer: eval/train mode" {
    const Model = Sequential(.{Linear_(2, 1)});

    var trainer = try Trainer(Model).init(testing.allocator, .{});
    defer trainer.deinit();

    trainer.eval();
    try testing.expect(!trainer.rt.training);
    trainer.train();
    try testing.expect(trainer.rt.training);
}

test "Trainer: save and load checkpoint" {
    const Model = Sequential(.{Linear_(2, 4)});

    var trainer = try Trainer(Model).init(testing.allocator, .{ .lr = 1e-2 });
    defer trainer.deinit();

    // Train a few steps
    var input_data = [_]f32{ 1.0, 2.0 };
    const target = [_]f32{ 1, 0, 0, 0 };
    for (0..10) |_| {
        trainer.zeroGrad();
        const out = trainer.forward(trainer.tensor(&input_data, &.{ 1, 2 }));
        const loss = trainer.mseLoss(out, &target);
        trainer.backward(loss);
        trainer.step();
    }

    // Get output before save
    trainer.zeroGrad();
    const out_before = trainer.forward(trainer.tensor(&input_data, &.{ 1, 2 }));
    var saved_out: [4]f32 = undefined;
    @memcpy(&saved_out, out_before.data[0..4]);

    // Save checkpoint
    const path = "/tmp/zig_trainer_test_ckpt.bin";
    try trainer.save(path);

    // Corrupt params by training more
    for (0..20) |_| {
        trainer.zeroGrad();
        const out = trainer.forward(trainer.tensor(&input_data, &.{ 1, 2 }));
        const loss = trainer.mseLoss(out, &target);
        trainer.backward(loss);
        trainer.step();
    }

    // Load checkpoint
    try trainer.load(path);

    // Verify output matches saved state
    trainer.zeroGrad();
    const out_after = trainer.forward(trainer.tensor(&input_data, &.{ 1, 2 }));
    for (0..4) |i| {
        try testing.expectApproxEqAbs(saved_out[i], out_after.data[i], 1e-6);
    }
}

test "Trainer: backwardAndStep with custom loss" {
    const Model = Sequential(.{
        Linear_(2, 4),
        ReLU,
        Linear_(4, 1),
    });

    var trainer = try Trainer(Model).init(testing.allocator, .{ .lr = 1e-2 });
    defer trainer.deinit();

    var input_data = [_]f32{ 1.0, 0.0 };
    const target_val: f32 = 1.0;

    var initial_loss: f32 = undefined;
    var final_loss: f32 = undefined;

    for (0..100) |i| {
        trainer.zeroGrad();
        const out = trainer.forward(trainer.tensor(&input_data, &.{ 1, 2 }));
        // Manual L1 loss: mean(|pred - target|) ≈ square(pred - target) for gradient test
        const target_tensor = trainer.rt.constantData(
            @ptrCast(std.mem.asBytes(&[_]f32{target_val})),
            @sizeOf(f32),
            &.{1},
            0,
        );
        const diff_tensor = trainer.rt.add(out, trainer.rt.negative(target_tensor));
        const loss = trainer.rt.reductionMean(trainer.rt.square(diff_tensor), -1);
        const lv = trainer.backwardAndStep(loss);
        if (i == 0) initial_loss = lv;
        if (i == 99) final_loss = lv;
    }

    try testing.expect(final_loss < initial_loss);
}

test "Trainer: crossEntropyLoss" {
    const Model = Sequential(.{Linear_(3, 4)});

    var trainer = try Trainer(Model).init(testing.allocator, .{ .lr = 1e-2 });
    defer trainer.deinit();

    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [2, 3]
    const indices = [_]u32{ 1, 3 };

    var initial_loss: f32 = undefined;
    var final_loss: f32 = undefined;

    for (0..50) |i| {
        trainer.zeroGrad();
        const out = trainer.forward(trainer.tensor(&input_data, &.{ 2, 3 }));
        const loss = trainer.crossEntropyLoss(out, &indices);
        if (i == 0) initial_loss = loss.data[0];
        if (i == 49) final_loss = loss.data[0];
        trainer.backward(loss);
        trainer.step();
    }

    try testing.expect(final_loss < initial_loss);
}
