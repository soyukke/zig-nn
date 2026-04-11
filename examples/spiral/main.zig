const std = @import("std");
const nn = @import("nn");
const Sequential = nn.unified.Sequential;
const Linear = nn.unified.Linear;
const ReLU = nn.unified.ReLU;
const Trainer = nn.unified.Trainer;

pub fn main() !void {
    try spiralDemo();
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
const SpiralModel = Sequential(.{ Linear(2, 32), ReLU, Linear(32, 32), ReLU, Linear(32, 3) });

fn spiralDemo() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== Spiral Classification (Trainer API: CrossEntropy + Adam) ===\n\n", .{});

    // Generate data
    var x_data: [SPIRAL_TOTAL * 2]f32 = undefined;
    var y_data: [SPIRAL_TOTAL]u32 = undefined;
    generateSpiralData(&x_data, &y_data);

    std.debug.print("  Data: {d} samples, {d} classes\n", .{ SPIRAL_TOTAL, SPIRAL_N_CLASSES });

    // 1. Trainer (Module + Runtime + Adam を一括管理)
    var trainer = try Trainer(SpiralModel).init(allocator, .{ .lr = 0.01 });
    defer trainer.deinit();

    // 2. Training loop
    const num_epochs = 500;
    for (0..num_epochs) |epoch| {
        trainer.zeroGrad();
        const logits = trainer.forward(trainer.tensor(&x_data, &.{ SPIRAL_TOTAL, 2 })); // [150, 3]
        const loss = trainer.crossEntropyLoss(logits, &y_data);
        trainer.backward(loss);
        trainer.step();

        if (epoch % 100 == 0 or epoch == num_epochs - 1) {
            const acc = computeAccuracy(SPIRAL_TOTAL, 3, logits.data[0 .. SPIRAL_TOTAL * 3], &y_data);
            std.debug.print("  Epoch {d:>3}: loss = {d:.4}, accuracy = {d:.1}%\n", .{
                epoch, loss.data[0], acc * 100.0,
            });
        }
    }

    // 3. Final accuracy
    trainer.zeroGrad();
    const logits = trainer.forward(trainer.tensor(&x_data, &.{ SPIRAL_TOTAL, 2 }));
    const final_acc = computeAccuracy(SPIRAL_TOTAL, 3, logits.data[0 .. SPIRAL_TOTAL * 3], &y_data);
    std.debug.print("\n  Final accuracy: {d:.1}%\n", .{final_acc * 100.0});

    // Show a few predictions
    std.debug.print("\n  Sample predictions:\n", .{});
    const show_indices = [_]usize{ 0, 25, 49, 50, 75, 99, 100, 125, 149 };
    for (show_indices) |idx| {
        const pred_class = argmax(3, logits.data[idx * 3 .. (idx + 1) * 3]);
        const correct: []const u8 = if (pred_class == y_data[idx]) "OK" else "NG";
        std.debug.print("    [{d:>3}] ({d:>6.3}, {d:>6.3}) -> pred={d} target={d} {s}\n", .{
            idx, x_data[idx * 2], x_data[idx * 2 + 1], pred_class, y_data[idx], correct,
        });
    }
}
