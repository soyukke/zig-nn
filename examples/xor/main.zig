const std = @import("std");
const nn = @import("nn");
const Sequential = nn.unified.Sequential;
const Linear = nn.unified.Linear;
const ReLU = nn.unified.ReLU;
const Trainer = nn.unified.Trainer;

pub fn main() !void {
    try xorDemo();
}

/// XOR model: Linear(2,8) -> ReLU -> Linear(8,1)
const XorModel = Sequential(.{ Linear(2, 8), ReLU, Linear(8, 1) });

fn xorDemo() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== XOR Training (Trainer API: MSE + Adam) ===\n\n", .{});

    // 1. Trainer (Module + Runtime + Adam を一括管理)
    var trainer = try Trainer(XorModel).init(allocator, .{ .lr = 0.01 });
    defer trainer.deinit();

    // 2. Training data
    var inputs = [_]f32{ 0, 0, 0, 1, 1, 0, 1, 1 };
    const targets = [_]f32{ 0, 1, 1, 0 };

    // 3. Training loop
    const num_epochs = 2000;
    for (0..num_epochs) |epoch| {
        trainer.zeroGrad();
        const output = trainer.forward(trainer.tensor(&inputs, &.{ 4, 2 })); // [4, 1]
        const pred = trainer.rt.reshape(output, &.{4});
        const loss = trainer.mseLoss(pred, &targets);
        trainer.backward(loss);
        trainer.step();

        if (epoch % 400 == 0 or epoch == num_epochs - 1) {
            std.debug.print("  Epoch {d:>4}: loss = {d:.6}\n", .{ epoch, loss.data[0] });
        }
    }

    // 4. Final predictions
    std.debug.print("\n  Predictions:\n", .{});
    trainer.zeroGrad();
    const output = trainer.forward(trainer.tensor(&inputs, &.{ 4, 2 }));
    const pred = trainer.rt.reshape(output, &.{4});

    const xor_in = [_][2]f32{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
    const xor_tgt = [_]f32{ 0, 1, 1, 0 };
    for (xor_in, xor_tgt, 0..) |inp, tgt, i| {
        const p = pred.data[i];
        const mark: []const u8 = if (@abs(p - tgt) < 0.3) " OK" else " NG";
        std.debug.print("    [{d}, {d}] -> {d:.4} (target: {d}){s}\n", .{ inp[0], inp[1], p, tgt, mark });
    }
}
