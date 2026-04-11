const std = @import("std");
const nn = @import("nn");
const Sequential = nn.unified.Sequential;
const Linear = nn.unified.Linear;
const ReLU = nn.unified.ReLU;
const Trainer = nn.unified.Trainer;
const CudaTrainer = nn.unified.CudaTrainer;

pub fn main() !void {
    var args = std.process.args();
    _ = args.skip();
    const mode = args.next() orelse "cpu";

    if (std.mem.eql(u8, mode, "cuda")) {
        try xorDemoCuda();
    } else {
        try xorDemo();
    }
}

/// XOR model: Linear(2,8) -> ReLU -> Linear(8,1)
const XorModel = Sequential(.{ Linear(2, 8), ReLU, Linear(8, 1) });

fn xorDemo() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== XOR Training (CPU: MSE + Adam) ===\n\n", .{});

    var trainer = try Trainer(XorModel).init(allocator, .{ .lr = 0.01 });
    defer trainer.deinit();

    var inputs = [_]f32{ 0, 0, 0, 1, 1, 0, 1, 1 };
    const targets = [_]f32{ 0, 1, 1, 0 };

    const num_epochs = 2000;
    for (0..num_epochs) |epoch| {
        trainer.zeroGrad();
        const output = trainer.forward(trainer.tensor(&inputs, &.{ 4, 2 }));
        const pred = trainer.rt.reshape(output, &.{4});
        const loss = trainer.mseLoss(pred, &targets);
        trainer.backward(loss);
        trainer.step();

        if (epoch % 400 == 0 or epoch == num_epochs - 1) {
            std.debug.print("  Epoch {d:>4}: loss = {d:.6}\n", .{ epoch, loss.data[0] });
        }
    }

    std.debug.print("\n  Predictions:\n", .{});
    trainer.zeroGrad();
    const output = trainer.forward(trainer.tensor(&inputs, &.{ 4, 2 }));
    const pred = trainer.rt.reshape(output, &.{4});
    printPredictions(pred.data[0..4]);
}

fn xorDemoCuda() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== XOR Training (CUDA: MSE + Adam) ===\n\n", .{});

    const cuda = nn.cuda;
    var cuda_ctx = try cuda.CudaContext.init(0);
    defer cuda_ctx.deinit();

    var trainer = try CudaTrainer(XorModel).init(allocator, &cuda_ctx, .{ .lr = 0.01 });
    defer trainer.deinit();

    var inputs = [_]f32{ 0, 0, 0, 1, 1, 0, 1, 1 };
    const targets = [_]f32{ 0, 1, 1, 0 };

    const num_epochs = 2000;
    for (0..num_epochs) |epoch| {
        trainer.zeroGrad();
        const output = trainer.forward(trainer.tensor(&inputs, &.{ 4, 2 }));
        const pred = trainer.rt.reshape(output, &.{4});
        const loss = trainer.mseLoss(pred, &targets);
        trainer.backward(loss);
        trainer.step();

        if (epoch % 400 == 0 or epoch == num_epochs - 1) {
            const loss_val = trainer.rt.copyScalarToHost(loss);
            std.debug.print("  Epoch {d:>4}: loss = {d:.6}\n", .{ epoch, loss_val });
        }
    }

    std.debug.print("\n  Predictions:\n", .{});
    trainer.zeroGrad();
    const output = trainer.forward(trainer.tensor(&inputs, &.{ 4, 2 }));
    const pred = trainer.rt.reshape(output, &.{4});
    var pred_host: [4]f32 = undefined;
    trainer.rt.copyToHost(pred, &pred_host);
    printPredictions(&pred_host);
}

fn printPredictions(pred: []const f32) void {
    const xor_in = [_][2]f32{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
    const xor_tgt = [_]f32{ 0, 1, 1, 0 };
    for (xor_in, xor_tgt, 0..) |inp, tgt, i| {
        const p = pred[i];
        const mark: []const u8 = if (@abs(p - tgt) < 0.3) " OK" else " NG";
        std.debug.print("    [{d}, {d}] -> {d:.4} (target: {d}){s}\n", .{ inp[0], inp[1], p, tgt, mark });
    }
}
