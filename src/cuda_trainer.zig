/// CudaTrainer: GPU 版 Trainer
///
/// Trainer と同じ PyTorch ライクな training ループヘルパーを
/// DiffCudaRuntime + GpuAdamState で GPU 上で実行する。
///
/// Usage:
///   const MLP = Sequential(.{ Linear_(2, 16), ReLU, Linear_(16, 1) });
///   var cuda_ctx = try CudaContext.init();
///   var trainer = try CudaTrainer(MLP).init(allocator, &cuda_ctx, .{ .lr = 1e-3 });
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
const cuda_mod = @import("backend/cuda.zig");
const CudaContext = cuda_mod.CudaContext;
const diff_cuda = @import("diff_cuda_runtime.zig");
const DiffCudaRuntime = diff_cuda.DiffCudaRuntime;
const DiffCudaTensor = diff_cuda.DiffCudaTensor;
const GpuAdamState = diff_cuda.GpuAdamState;

pub fn CudaTrainer(comptime ModelType: type) type {
    return struct {
        allocator: Allocator,
        module: *Module,
        model: ModelType,
        rt: *DiffCudaRuntime,
        adam: GpuAdamState,
        config: Config,

        pub const Config = struct {
            lr: f32 = 1e-3,
            beta1: f32 = 0.9,
            beta2: f32 = 0.999,
            eps: f32 = 1e-8,
            weight_decay: f32 = 0,
            max_grad_norm: ?f32 = null,
        };

        pub fn init(allocator: Allocator, cuda_ctx: *CudaContext, config: Config) !@This() {
            const module = try allocator.create(Module);
            module.* = Module.init(allocator);
            const model = ModelType.init(module);

            const rt = try allocator.create(DiffCudaRuntime);
            rt.* = try DiffCudaRuntime.init(module, cuda_ctx, allocator);
            rt.initParams();

            const sizes = try module.paramSizes(allocator);
            defer allocator.free(sizes);
            const adam = try GpuAdamState.init(allocator, cuda_ctx, sizes);

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

        pub fn zeroGrad(self: *@This()) void {
            self.rt.zeroGrad();
            self.rt.resetArena();
        }

        pub fn forward(self: *@This(), x: DiffCudaTensor) DiffCudaTensor {
            return self.model.forward(self.rt, x);
        }

        pub fn backward(self: *@This(), loss: DiffCudaTensor) void {
            self.rt.backward(loss);
        }

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

        pub fn tensor(self: *@This(), data: []f32, shape: []const usize) DiffCudaTensor {
            return self.rt.makeTensor(data, shape);
        }

        pub fn param(self: *@This(), handle: compute.ParamHandle) DiffCudaTensor {
            return self.rt.param(handle);
        }

        // ── Loss functions ──

        pub fn mseLoss(self: *@This(), pred: DiffCudaTensor, target: []const f32) DiffCudaTensor {
            return self.rt.mseLoss(pred, target);
        }

        pub fn crossEntropyLoss(self: *@This(), logits: DiffCudaTensor, indices: []const u32) DiffCudaTensor {
            return self.rt.crossEntropyLossWithIndices(logits, indices);
        }

        pub fn bceLoss(self: *@This(), logits: DiffCudaTensor, target: []const f32) DiffCudaTensor {
            return self.rt.bceLossWithLogits(logits, target);
        }

        // ── Custom loss ──

        pub fn backwardAndStep(self: *@This(), loss: DiffCudaTensor) f32 {
            self.backward(loss);
            self.step();
            return self.rt.copyScalarToHost(loss);
        }

        // ── Learning rate ──

        pub fn setLr(self: *@This(), lr: f32) void {
            self.config.lr = lr;
        }

        pub fn getLr(self: *const @This()) f32 {
            return self.config.lr;
        }

        pub fn cosineAnnealingStep(self: *@This(), current_step: u32, total_steps: u32, lr_min: f32, lr_max: f32) void {
            self.config.lr = compute.cosineAnnealingLR(current_step, total_steps, lr_min, lr_max);
        }

        pub fn warmupCosineStep(self: *@This(), current_step: u32, warmup_steps: u32, total_steps: u32, lr_min: f32, lr_max: f32) void {
            self.config.lr = compute.warmupCosineDecayLR(current_step, warmup_steps, total_steps, lr_min, lr_max);
        }

        // ── Utilities ──

        pub fn paramCount(self: *const @This()) usize {
            return self.module.paramCount();
        }

        pub fn totalParamElements(self: *const @This()) usize {
            return self.module.totalParamElements();
        }
    };
}
