const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");
const sequential = nn.unified.sequential;
const linear = nn.unified.linear;
const ReLU = nn.unified.ReLU;
const trainer = nn.unified.trainer;

pub const std_options = nn.log.std_options;
const log = nn.log.example;

const is_cuda_available = builtin.os.tag == .linux;

pub fn main(init: std.process.Init.Minimal) !void {
    var args = init.args.iterate();
    _ = args.skip();
    const mode = args.next() orelse "cpu";

    if (is_cuda_available and std.mem.eql(u8, mode, "cuda")) {
        const cuda = nn.cuda;
        var cuda_ctx = try cuda.CudaContext.init(0);
        defer cuda_ctx.deinit();
        var trainer = try trainer(SpiralModel, .cuda).init(std.heap.page_allocator, &cuda_ctx, .{ .lr = 0.01 });
        defer trainer.deinit();
        try spiralDemo(&trainer, "CUDA");
    } else {
        var trainer = try trainer(SpiralModel, .cpu).init(std.heap.page_allocator, {}, .{ .lr = 0.01 });
        defer trainer.deinit();
        try spiralDemo(&trainer, "CPU");
    }
}

const SPIRAL_N_PER_CLASS = 50;
const SPIRAL_N_CLASSES = 3;
const SPIRAL_TOTAL = SPIRAL_N_PER_CLASS * SPIRAL_N_CLASSES; // 150

fn generateSpiralData(
    x_out: *[SPIRAL_TOTAL * 2]f32,
    y_out: *[SPIRAL_TOTAL]u32,
) void {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    for (0..SPIRAL_N_CLASSES) |cls| {
        for (0..SPIRAL_N_PER_CLASS) |i| {
            const idx = cls * SPIRAL_N_PER_CLASS + i;
            const r: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(SPIRAL_N_PER_CLASS));
            const base_angle: f32 = @as(f32, @floatFromInt(cls)) * 4.0 + r * 4.0;
            const noise: f32 = (rng.float(f32) - 0.5) * 0.3;
            const angle = base_angle + noise;

            x_out[idx * 2 + 0] = r * @cos(angle);
            x_out[idx * 2 + 1] = r * @sin(angle);
            y_out[idx] = @intCast(cls);
        }
    }
}

fn computeAccuracy(comptime n: usize, comptime classes: usize, logits: []const f32, targets: []const u32) f32 {
    var correct: usize = 0;
    for (0..n) |i| {
        const pred = argmax(classes, logits[i * classes .. (i + 1) * classes]);
        if (pred == targets[i]) correct += 1;
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(n));
}

fn argmax(comptime n: usize, data: []const f32) u32 {
    var best: u32 = 0;
    var best_val: f32 = data[0];
    for (1..n) |i| {
        if (data[i] > best_val) {
            best_val = data[i];
            best = @intCast(i);
        }
    }
    return best;
}

/// Spiral model: Linear(2,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,3)
const SpiralModel = sequential(.{ linear(2, 32), ReLU, linear(32, 32), ReLU, linear(32, 3) });

fn spiralDemo(trainer: anytype, device_name: []const u8) !void {
    log.info("=== Spiral Classification ({s}: CrossEntropy + Adam) ===", .{device_name});

    var x_data: [SPIRAL_TOTAL * 2]f32 = undefined;
    var y_data: [SPIRAL_TOTAL]u32 = undefined;
    generateSpiralData(&x_data, &y_data);

    log.info("data: {d} samples, {d} classes", .{ SPIRAL_TOTAL, SPIRAL_N_CLASSES });

    const num_epochs = 500;
    var timer = try nn.Timer.start();
    for (0..num_epochs) |epoch| {
        trainer.zero_grad();
        const logits = trainer.forward(trainer.tensor(&x_data, &.{ SPIRAL_TOTAL, 2 }));
        const loss = trainer.cross_entropy_loss(logits, &y_data);
        trainer.backward(loss);
        trainer.step();

        if (epoch % 100 == 0 or epoch == num_epochs - 1) {
            const loss_val = trainer.loss_value(loss);
            var logits_buf: [SPIRAL_TOTAL * 3]f32 = undefined;
            const logits_data = trainer.data_slice(logits, &logits_buf);
            const acc = computeAccuracy(SPIRAL_TOTAL, 3, logits_data, &y_data);
            log.info("epoch {d:>3}: loss = {d:.4}, accuracy = {d:.1}%", .{
                epoch, loss_val, acc * 100.0,
            });
        }
    }
    const elapsed_ms = timer.read() / 1_000_000;
    log.info("training time: {d}ms", .{elapsed_ms});

    trainer.zero_grad();
    const logits = trainer.forward(trainer.tensor(&x_data, &.{ SPIRAL_TOTAL, 2 }));
    var logits_buf: [SPIRAL_TOTAL * 3]f32 = undefined;
    const logits_data = trainer.data_slice(logits, &logits_buf);
    const final_acc = computeAccuracy(SPIRAL_TOTAL, 3, logits_data, &y_data);
    log.info("final accuracy: {d:.1}%", .{final_acc * 100.0});
    printSamplePredictions(logits_data, &x_data, &y_data);
}

fn printSamplePredictions(logits_data: []const f32, x_data: []const f32, y_data: []const u32) void {
    log.info("sample predictions:", .{});
    const show_indices = [_]usize{ 0, 25, 49, 50, 75, 99, 100, 125, 149 };
    for (show_indices) |idx| {
        const pred_class = argmax(3, logits_data[idx * 3 .. (idx + 1) * 3]);
        const correct: []const u8 = if (pred_class == y_data[idx]) "OK" else "NG";
        log.info("  [{d:>3}] ({d:>6.3}, {d:>6.3}) -> pred={d} target={d} {s}", .{
            idx, x_data[idx * 2], x_data[idx * 2 + 1], pred_class, y_data[idx], correct,
        });
    }
}
