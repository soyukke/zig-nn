const std = @import("std");
const nn = @import("nn");
const Sequential = nn.unified.Sequential;
const Linear = nn.unified.Linear;
const ReLU = nn.unified.ReLU;
const Trainer = nn.unified.Trainer;

pub fn main() !void {
    var args = std.process.args();
    _ = args.skip();
    const mode = args.next() orelse "cpu";

    if (std.mem.eql(u8, mode, "cuda")) {
        const cuda = nn.cuda;
        var cuda_ctx = try cuda.CudaContext.init(0);
        defer cuda_ctx.deinit();
        var trainer = try Trainer(XorModel, .cuda).init(std.heap.page_allocator, &cuda_ctx, .{ .lr = 0.01 });
        defer trainer.deinit();
        try xorDemo(&trainer, "CUDA");
    } else {
        var trainer = try Trainer(XorModel, .cpu).init(std.heap.page_allocator, {}, .{ .lr = 0.01 });
        defer trainer.deinit();
        try xorDemo(&trainer, "CPU");
    }
}

/// XOR model: Linear(2,8) -> ReLU -> Linear(8,1)
const XorModel = Sequential(.{ Linear(2, 8), ReLU, Linear(8, 1) });

fn xorDemo(trainer: anytype, device_name: []const u8) !void {
    std.debug.print("=== XOR Training ({s}: MSE + Adam) ===\n\n", .{device_name});

    var inputs = [_]f32{ 0, 0, 0, 1, 1, 0, 1, 1 };
    const targets = [_]f32{ 0, 1, 1, 0 };

    const num_epochs = 2000;
    var timer = try std.time.Timer.start();
    for (0..num_epochs) |epoch| {
        trainer.zeroGrad();
        const output = trainer.forward(trainer.tensor(&inputs, &.{ 4, 2 }));
        const pred = trainer.reshape(output, &.{4});
        const loss = trainer.mseLoss(pred, &targets);
        trainer.backward(loss);
        trainer.step();

        if (epoch % 400 == 0 or epoch == num_epochs - 1) {
            const loss_val = trainer.lossValue(loss);
            std.debug.print("  Epoch {d:>4}: loss = {d:.6}\n", .{ epoch, loss_val });
        }
    }
    const elapsed_ms = timer.read() / 1_000_000;
    std.debug.print("\n  Training time: {d}ms\n", .{elapsed_ms});

    std.debug.print("\n  Predictions:\n", .{});
    trainer.zeroGrad();
    const output = trainer.forward(trainer.tensor(&inputs, &.{ 4, 2 }));
    const pred = trainer.reshape(output, &.{4});
    var pred_buf: [4]f32 = undefined;
    const pred_data = trainer.dataSlice(pred, &pred_buf);
    printPredictions(pred_data);
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
