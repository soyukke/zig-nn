const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");

pub const std_options = nn.log.std_options;
const log = nn.log.example;

const compute = nn.compute;
const Module = compute.Module;
const AdamState = compute.AdamState;
const DiffCpuRuntime = nn.unified.DiffCpuRuntime;
const DiffTensor = nn.unified.DiffTensor;
const DiffCudaRuntime = nn.unified.DiffCudaRuntime;
const GpuAdamState = nn.unified.GpuAdamState;
const Linear = nn.unified.Linear;
const diffusion = nn.diffusion;

const is_cuda_available = builtin.os.tag == .linux;

const DIFF_BATCH = 256;
const DIFF_HIDDEN = 128;
const DIFF_TIME_DIM = 64;
const DIFF_T = 200;
const DIFF_NUM_SAMPLES = 1024;

pub fn main(init: std.process.Init.Minimal) !void {
    var args = init.args.iterate();
    _ = args.skip();
    const mode = args.next() orelse "cpu";

    if (is_cuda_available and std.mem.eql(u8, mode, "cuda")) {
        try diffusionDemoCuda();
    } else {
        try diffusionDemo();
    }
}

/// DDPM Model: MLP with sinusoidal time embedding injection
const DdpmModel = struct {
    time_fc1: Linear(DIFF_TIME_DIM, DIFF_HIDDEN),
    time_fc2: Linear(DIFF_HIDDEN, DIFF_HIDDEN),
    fc1: Linear(2, DIFF_HIDDEN),
    fc2: Linear(DIFF_HIDDEN, DIFF_HIDDEN),
    fc3: Linear(DIFF_HIDDEN, DIFF_HIDDEN),
    fc_out: Linear(DIFF_HIDDEN, 2),
    tp1: Linear(DIFF_HIDDEN, DIFF_HIDDEN),
    tp2: Linear(DIFF_HIDDEN, DIFF_HIDDEN),
    tp3: Linear(DIFF_HIDDEN, DIFF_HIDDEN),

    fn init(module: *Module) DdpmModel {
        return .{
            .time_fc1 = Linear(DIFF_TIME_DIM, DIFF_HIDDEN).init(module),
            .time_fc2 = Linear(DIFF_HIDDEN, DIFF_HIDDEN).init(module),
            .fc1 = Linear(2, DIFF_HIDDEN).init(module),
            .fc2 = Linear(DIFF_HIDDEN, DIFF_HIDDEN).init(module),
            .fc3 = Linear(DIFF_HIDDEN, DIFF_HIDDEN).init(module),
            .fc_out = Linear(DIFF_HIDDEN, 2).init(module),
            .tp1 = Linear(DIFF_HIDDEN, DIFF_HIDDEN).init(module),
            .tp2 = Linear(DIFF_HIDDEN, DIFF_HIDDEN).init(module),
            .tp3 = Linear(DIFF_HIDDEN, DIFF_HIDDEN).init(module),
        };
    }

    fn forward(self: DdpmModel, ctx: anytype, x_t: anytype, time_emb: anytype) @TypeOf(x_t) {
        // Time MLP
        const t_h1 = ctx.silu(self.time_fc1.forward(ctx, time_emb));
        const t_hidden = self.time_fc2.forward(ctx, t_h1);

        // MLP with time injection (fused add+silu)
        const h1 = ctx.addSilu(self.fc1.forward(ctx, x_t), self.tp1.forward(ctx, t_hidden));
        const h2 = ctx.addSilu(self.fc2.forward(ctx, h1), self.tp2.forward(ctx, t_hidden));
        const h3 = ctx.addSilu(self.fc3.forward(ctx, h2), self.tp3.forward(ctx, t_hidden));
        return self.fc_out.forward(ctx, h3);
    }
};

fn generateGaussianMixture(x_out: []f32, rng: std.Random) void {
    const centers = [_][2]f32{ .{ 2.0, 0.0 }, .{ 0.0, 2.0 }, .{ -2.0, 0.0 }, .{ 0.0, -2.0 } };
    const sigma: f32 = 0.3;
    const n_points = x_out.len / 2;

    for (0..n_points) |i| {
        const cluster = i % 4;
        const uniform1 = rng.float(f32) * 0.99998 + 0.00001;
        const uniform2 = rng.float(f32);
        const r = @sqrt(-2.0 * @log(uniform1));
        x_out[i * 2 + 0] = centers[cluster][0] + sigma * r * @cos(2.0 * std.math.pi * uniform2);
        x_out[i * 2 + 1] = centers[cluster][1] + sigma * r * @sin(2.0 * std.math.pi * uniform2);
    }
}

fn prepareBatch(
    rng: std.Random,
    dataset: []const f32,
    schedule: anytype,
    batch_x0: *[DIFF_BATCH * 2]f32,
    x_t_data: *[DIFF_BATCH * 2]f32,
    epsilon: *[DIFF_BATCH * 2]f32,
    time_emb_data: *[DIFF_BATCH * DIFF_TIME_DIM]f32,
) void {
    for (0..DIFF_BATCH) |i| {
        const idx = rng.intRangeAtMost(usize, 0, DIFF_NUM_SAMPLES - 1);
        batch_x0[i * 2 + 0] = dataset[idx * 2 + 0];
        batch_x0[i * 2 + 1] = dataset[idx * 2 + 1];
    }

    var timesteps: [DIFF_BATCH]usize = undefined;
    for (0..DIFF_BATCH) |i| timesteps[i] = rng.intRangeAtMost(usize, 0, DIFF_T - 1);

    diffusion.boxMullerGaussian(rng, epsilon);

    for (0..DIFF_BATCH) |i| {
        const t = timesteps[i];
        x_t_data[i * 2 + 0] = schedule.sqrt_alpha_bars[t] * batch_x0[i * 2 + 0] + schedule.sqrt_one_minus_alpha_bars[t] * epsilon[i * 2 + 0];
        x_t_data[i * 2 + 1] = schedule.sqrt_alpha_bars[t] * batch_x0[i * 2 + 1] + schedule.sqrt_one_minus_alpha_bars[t] * epsilon[i * 2 + 1];
    }

    for (0..DIFF_BATCH) |i| {
        diffusion.sinusoidalEmbedding(DIFF_TIME_DIM, timesteps[i], time_emb_data[i * DIFF_TIME_DIM .. (i + 1) * DIFF_TIME_DIM]);
    }
}

fn diffusionDemo() !void {
    const allocator = std.heap.page_allocator;
    log.info("=== DDPM Diffusion (CPU) ===", .{});

    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();
    var dataset: [DIFF_NUM_SAMPLES * 2]f32 = undefined;
    generateGaussianMixture(&dataset, rng);
    log.info("dataset: {d} points, 4 Gaussian clusters", .{DIFF_NUM_SAMPLES});

    const schedule = diffusion.NoiseSchedule(DIFF_T).initLinear(1e-4, 0.02);

    var module = Module.init(allocator);
    defer module.deinit();
    const model = DdpmModel.init(&module);

    var rt = try DiffCpuRuntime.init(&module, allocator);
    defer rt.deinit();
    rt.initParams();

    const total_params = module.totalParamElements();
    log.info("model parameters: {d} (~{d}KB)", .{ total_params, total_params * 4 / 1024 });
    log.info("diffusion steps: T={d}, batch={d}", .{ DIFF_T, DIFF_BATCH });

    const sizes = try module.paramSizes(allocator);
    defer allocator.free(sizes);
    var adam = try AdamState.init(allocator, sizes);
    defer adam.deinit();

    const num_epochs = 1000;
    var timer = nn.Timer.start() catch unreachable;

    for (0..num_epochs) |epoch| {
        rt.resetArena();
        rt.zeroGrad();

        var batch_x0: [DIFF_BATCH * 2]f32 = undefined;
        var x_t_data: [DIFF_BATCH * 2]f32 = undefined;
        var epsilon: [DIFF_BATCH * 2]f32 = undefined;
        var time_emb_data: [DIFF_BATCH * DIFF_TIME_DIM]f32 = undefined;
        prepareBatch(rng, &dataset, schedule, &batch_x0, &x_t_data, &epsilon, &time_emb_data);

        const x_t = rt.makeTensor(&x_t_data, &.{ DIFF_BATCH, 2 });
        const time_emb = rt.makeTensor(&time_emb_data, &.{ DIFF_BATCH, DIFF_TIME_DIM });
        const eps_pred = model.forward(&rt, x_t, time_emb);
        const loss = rt.mseLoss(rt.reshape(eps_pred, &.{DIFF_BATCH * 2}), &epsilon);

        rt.backward(loss);
        rt.applyAdam(&adam, 1e-3, 0.9, 0.999, 1e-8, 0);

        if (epoch % 500 == 0 or epoch == num_epochs - 1) {
            log.info("epoch {d:>4}: loss = {d:.6}", .{ epoch, loss.data[0] });
        }
    }

    const train_ms = timer.read() / 1_000_000;
    log.info("training time: {d}ms", .{train_ms});

    sampleAndPrint(&rt, model, rng, schedule);
}

fn diffusionDemoCuda() !void {
    const allocator = std.heap.page_allocator;
    log.info("=== DDPM Diffusion (CUDA) ===", .{});

    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();
    var dataset: [DIFF_NUM_SAMPLES * 2]f32 = undefined;
    generateGaussianMixture(&dataset, rng);
    log.info("dataset: {d} points, 4 Gaussian clusters", .{DIFF_NUM_SAMPLES});

    const schedule = diffusion.NoiseSchedule(DIFF_T).initLinear(1e-4, 0.02);

    var module = Module.init(allocator);
    defer module.deinit();
    const model = DdpmModel.init(&module);

    const cuda = nn.cuda;
    var cuda_ctx = try cuda.CudaContext.init(0);
    defer cuda_ctx.deinit();

    var rt = try DiffCudaRuntime.init(&module, &cuda_ctx, allocator);
    defer rt.deinit();
    rt.initParams();

    const total_params = module.totalParamElements();
    log.info("model parameters: {d} (~{d}KB)", .{ total_params, total_params * 4 / 1024 });
    log.info("diffusion steps: T={d}, batch={d}", .{ DIFF_T, DIFF_BATCH });

    const sizes = try module.paramSizes(allocator);
    defer allocator.free(sizes);
    var adam = try GpuAdamState.init(allocator, &cuda_ctx, sizes);
    defer adam.deinit();

    const num_epochs = 1000;
    var timer = nn.Timer.start() catch unreachable;

    for (0..num_epochs) |epoch| {
        rt.resetArena();
        rt.zeroGrad();

        var batch_x0: [DIFF_BATCH * 2]f32 = undefined;
        var x_t_data: [DIFF_BATCH * 2]f32 = undefined;
        var epsilon: [DIFF_BATCH * 2]f32 = undefined;
        var time_emb_data: [DIFF_BATCH * DIFF_TIME_DIM]f32 = undefined;
        prepareBatch(rng, &dataset, schedule, &batch_x0, &x_t_data, &epsilon, &time_emb_data);

        const x_t = rt.makeTensor(&x_t_data, &.{ DIFF_BATCH, 2 });
        const time_emb = rt.makeTensor(&time_emb_data, &.{ DIFF_BATCH, DIFF_TIME_DIM });
        const eps_pred = model.forward(&rt, x_t, time_emb);
        const loss = rt.mseLoss(rt.reshape(eps_pred, &.{DIFF_BATCH * 2}), &epsilon);

        rt.backward(loss);
        rt.applyAdam(&adam, 1e-3, 0.9, 0.999, 1e-8, 0);

        if (epoch % 500 == 0 or epoch == num_epochs - 1) {
            const loss_val = rt.copyScalarToHost(loss);
            log.info("epoch {d:>4}: loss = {d:.6}", .{ epoch, loss_val });
        }
    }

    const train_ms = timer.read() / 1_000_000;
    log.info("training time: {d}ms", .{train_ms});

    sampleAndPrintCuda(&rt, model, rng, schedule);
}

fn sampleAndPrint(rt: *DiffCpuRuntime, model: DdpmModel, rng: std.Random, schedule: anytype) void {
    log.info("sampling 64 points via DDPM reverse process...", .{});
    const N_GEN = 64;
    var x_gen: [N_GEN * 2]f32 = undefined;
    diffusion.boxMullerGaussian(rng, &x_gen);

    var t_idx: usize = DIFF_T;
    while (t_idx > 0) {
        t_idx -= 1;
        rt.resetArena();

        var time_emb_gen: [N_GEN * DIFF_TIME_DIM]f32 = undefined;
        for (0..N_GEN) |i| {
            diffusion.sinusoidalEmbedding(DIFF_TIME_DIM, t_idx, time_emb_gen[i * DIFF_TIME_DIM .. (i + 1) * DIFF_TIME_DIM]);
        }

        const xg = rt.makeTensor(&x_gen, &.{ N_GEN, 2 });
        const te = rt.makeTensor(&time_emb_gen, &.{ N_GEN, DIFF_TIME_DIM });
        const gen_pred = model.forward(rt, xg, te);

        var noise_z: [N_GEN * 2]f32 = undefined;
        diffusion.boxMullerGaussian(rng, &noise_z);

        for (0..N_GEN) |i| {
            for (0..2) |d| {
                const idx = i * 2 + d;
                const inv_sqrt_alpha = 1.0 / @sqrt(schedule.alphas[t_idx]);
                const coeff = schedule.betas[t_idx] / schedule.sqrt_one_minus_alpha_bars[t_idx];
                const sigma = @sqrt(schedule.betas[t_idx]);
                x_gen[idx] = inv_sqrt_alpha * (x_gen[idx] - coeff * gen_pred.data[idx]);
                if (t_idx > 0) x_gen[idx] += sigma * noise_z[idx];
            }
        }
    }

    printResults(&x_gen);
}

fn sampleAndPrintCuda(rt: *DiffCudaRuntime, model: DdpmModel, rng: std.Random, schedule: anytype) void {
    log.info("sampling 64 points via DDPM reverse process...", .{});
    const N_GEN = 64;
    var x_gen: [N_GEN * 2]f32 = undefined;
    diffusion.boxMullerGaussian(rng, &x_gen);

    var t_idx: usize = DIFF_T;
    while (t_idx > 0) {
        t_idx -= 1;
        rt.resetArena();

        var time_emb_gen: [N_GEN * DIFF_TIME_DIM]f32 = undefined;
        for (0..N_GEN) |i| {
            diffusion.sinusoidalEmbedding(DIFF_TIME_DIM, t_idx, time_emb_gen[i * DIFF_TIME_DIM .. (i + 1) * DIFF_TIME_DIM]);
        }

        const xg = rt.makeTensor(&x_gen, &.{ N_GEN, 2 });
        const te = rt.makeTensor(&time_emb_gen, &.{ N_GEN, DIFF_TIME_DIM });
        const gen_pred = model.forward(rt, xg, te);

        var pred_host: [N_GEN * 2]f32 = undefined;
        rt.copyToHost(gen_pred, &pred_host);

        var noise_z: [N_GEN * 2]f32 = undefined;
        diffusion.boxMullerGaussian(rng, &noise_z);

        for (0..N_GEN) |i| {
            for (0..2) |d| {
                const idx = i * 2 + d;
                const inv_sqrt_alpha = 1.0 / @sqrt(schedule.alphas[t_idx]);
                const coeff = schedule.betas[t_idx] / schedule.sqrt_one_minus_alpha_bars[t_idx];
                const sigma = @sqrt(schedule.betas[t_idx]);
                x_gen[idx] = inv_sqrt_alpha * (x_gen[idx] - coeff * pred_host[idx]);
                if (t_idx > 0) x_gen[idx] += sigma * noise_z[idx];
            }
        }
    }

    printResults(&x_gen);
}

fn printResults(x_gen: []const f32) void {
    const N_GEN = 64;
    log.debug("generated samples (x, y):", .{});
    for (0..N_GEN) |i| {
        log.debug("  ({d:>7.3}, {d:>7.3})", .{ x_gen[i * 2], x_gen[i * 2 + 1] });
    }

    log.info("cluster analysis:", .{});
    const centers = [_][2]f32{ .{ 2.0, 0.0 }, .{ 0.0, 2.0 }, .{ -2.0, 0.0 }, .{ 0.0, -2.0 } };
    var cluster_counts = [_]usize{ 0, 0, 0, 0 };
    for (0..N_GEN) |i| {
        var best_c: usize = 0;
        var best_dist: f32 = std.math.floatMax(f32);
        for (centers, 0..) |c, ci| {
            const dx = x_gen[i * 2] - c[0];
            const dy = x_gen[i * 2 + 1] - c[1];
            const dist = dx * dx + dy * dy;
            if (dist < best_dist) {
                best_dist = dist;
                best_c = ci;
            }
        }
        cluster_counts[best_c] += 1;
    }
    for (centers, cluster_counts, 0..) |c, count, ci| {
        log.info("  cluster {d} ({d:>5.1}, {d:>5.1}): {d} points", .{ ci, c[0], c[1], count });
    }
}
