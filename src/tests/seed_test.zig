/// tests/seed_test.zig: シード固定の網羅テスト
///
/// 目的: 推論モード / 学習モード の両方で、
///   (A) 同じ seed なら bitwise に同じ結果が得られる
///   (B) 異なる seed 同士では異なる結果になる
///   (C) かつ (A)(B) は 1 つの seed だけではなく複数の seed で成立する
/// を保証する。
///
/// trainer.zig 全体を test binary に引き込むと、scope 外の壊れた checkpoint テスト
/// (std.Io 移行の追従漏れ) に巻き込まれるため、seed 固定のテストだけをここに隔離する。
/// build.zig の test-diff-cpu にぶら下げて `just test-diff-cpu` で走らせる。
const std = @import("std");
const testing = std.testing;

const compute = @import("../compute.zig");
const Module = compute.Module;

const diff_cpu = @import("../diff/cpu_runtime.zig");
const DiffCpuRuntime = diff_cpu.DiffCpuRuntime;

const trainer_mod = @import("../trainer.zig");
const trainer = trainer_mod.trainer;

const graph_seq = @import("../nn/graph_sequential.zig");
const sequential = graph_seq.sequential;
const ReLU = graph_seq.ReLU;
const graph_linear = @import("../nn/graph_linear.zig");
const Linear_ = graph_linear.linear;

const dataloader = @import("../data/dataloader.zig");
const BatchIterator = dataloader.BatchIterator;

// ════════════════════════════════════════════════════════════════
// helpers
// ════════════════════════════════════════════════════════════════

/// 全パラメータの生データを連結したスナップショットを確保。caller が free。
fn snapshot_params(
    allocator: std.mem.Allocator,
    session: anytype,
) ![]f32 {
    var total: usize = 0;
    for (session.rt.param_nodes) |n| total += n.total_elements();
    const buf = try allocator.alloc(f32, total);
    var off: usize = 0;
    for (session.rt.param_nodes) |n| {
        const size = n.total_elements();
        @memcpy(buf[off .. off + size], n.data[0..size]);
        off += size;
    }
    return buf;
}

// ════════════════════════════════════════════════════════════════
// 基本: 単一 seed での往復確認
// ════════════════════════════════════════════════════════════════

test "seed: init_params is bitwise deterministic for same seed" {
    const Model = sequential(.{ Linear_(4, 8), ReLU, Linear_(8, 4) });

    var a = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = 123 });
    defer a.deinit();

    var b = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = 123 });
    defer b.deinit();

    const snap_a = try snapshot_params(testing.allocator, &a);
    defer testing.allocator.free(snap_a);

    const snap_b = try snapshot_params(testing.allocator, &b);
    defer testing.allocator.free(snap_b);

    try testing.expectEqualSlices(f32, snap_a, snap_b);
}

test "seed: different seeds give different init_params" {
    const Model = sequential(.{Linear_(4, 8)});

    var a = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = 1 });
    defer a.deinit();

    var b = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = 2 });
    defer b.deinit();

    const snap_a = try snapshot_params(testing.allocator, &a);
    defer testing.allocator.free(snap_a);

    const snap_b = try snapshot_params(testing.allocator, &b);
    defer testing.allocator.free(snap_b);

    try testing.expect(!std.mem.eql(f32, snap_a, snap_b));
}

test "seed: training 1 step is bitwise reproducible" {
    const Model = sequential(.{ Linear_(4, 8), ReLU, Linear_(8, 2) });

    var a = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .lr = 1e-2, .seed = 42 });
    defer a.deinit();

    var b = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .lr = 1e-2, .seed = 42 });
    defer b.deinit();

    var input_a = [_]f32{ 0.1, -0.2, 0.3, -0.4 };
    var input_b = [_]f32{ 0.1, -0.2, 0.3, -0.4 };
    const target = [_]f32{ 1, 0 };

    a.zero_grad();
    const out_a = a.forward(a.tensor(&input_a, &.{ 1, 4 }));
    const loss_a = a.mse_loss(out_a, &target);
    a.backward(loss_a);
    a.step();

    b.zero_grad();
    const out_b = b.forward(b.tensor(&input_b, &.{ 1, 4 }));
    const loss_b = b.mse_loss(out_b, &target);
    b.backward(loss_b);
    b.step();

    const snap_a = try snapshot_params(testing.allocator, &a);
    defer testing.allocator.free(snap_a);

    const snap_b = try snapshot_params(testing.allocator, &b);
    defer testing.allocator.free(snap_b);

    try testing.expectEqualSlices(f32, snap_a, snap_b);
}

test "seed: inference (eval mode) is bitwise reproducible" {
    const Model = sequential(.{ Linear_(3, 5), ReLU, Linear_(5, 2) });

    var a = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = 999 });
    defer a.deinit();

    var b = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = 999 });
    defer b.deinit();

    a.eval();
    b.eval();

    var input_a = [_]f32{ 0.5, -0.1, 0.2 };
    var input_b = [_]f32{ 0.5, -0.1, 0.2 };

    a.zero_grad();
    b.zero_grad();
    const out_a = a.forward(a.tensor(&input_a, &.{ 1, 3 }));
    const out_b = b.forward(b.tensor(&input_b, &.{ 1, 3 }));

    var da: [2]f32 = undefined;
    var db: [2]f32 = undefined;
    @memcpy(&da, out_a.data[0..2]);
    @memcpy(&db, out_b.data[0..2]);

    try testing.expectEqualSlices(f32, &da, &db);
}

test "seed: dropout mask is bitwise reproducible under same seed" {
    var module_a = Module.init(testing.allocator);
    defer module_a.deinit();

    var rt_a = try DiffCpuRuntime.init(&module_a, testing.allocator);
    defer rt_a.deinit();

    rt_a.set_seed(77);

    var module_b = Module.init(testing.allocator);
    defer module_b.deinit();

    var rt_b = try DiffCpuRuntime.init(&module_b, testing.allocator);
    defer rt_b.deinit();

    rt_b.set_seed(77);

    try testing.expect(rt_a.training);
    try testing.expect(rt_b.training);

    var x_a = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    var x_b = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const tensor_a = rt_a.make_node(&x_a, &.{16}, false);
    const tensor_b = rt_b.make_node(&x_b, &.{16}, false);

    const y_a = rt_a.dropout(tensor_a, 0.5);
    const y_b = rt_b.dropout(tensor_b, 0.5);
    try testing.expectEqualSlices(f32, y_a.data[0..16], y_b.data[0..16]);

    // 2 回目も PRNG 状態が同じ起点から進行していることを確認
    const y_a2 = rt_a.dropout(tensor_a, 0.5);
    const y_b2 = rt_b.dropout(tensor_b, 0.5);
    try testing.expectEqualSlices(f32, y_a2.data[0..16], y_b2.data[0..16]);
}

test "seed: set_seed resets dropout stream to the same start" {
    var module = Module.init(testing.allocator);
    defer module.deinit();

    var rt = try DiffCpuRuntime.init(&module, testing.allocator);
    defer rt.deinit();

    var x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const tensor_in = rt.make_node(&x, &.{8}, false);

    rt.set_seed(321);
    const y1 = rt.dropout(tensor_in, 0.3);
    var first: [8]f32 = undefined;
    @memcpy(&first, y1.data[0..8]);

    rt.set_seed(321);
    const y2 = rt.dropout(tensor_in, 0.3);
    try testing.expectEqualSlices(f32, &first, y2.data[0..8]);
}

test "seed: no seed config keeps backward compatibility (default seed=42)" {
    const Model = sequential(.{Linear_(4, 4)});

    var a = try trainer(Model, .cpu).init(testing.allocator, {}, .{});
    defer a.deinit();

    var b = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = 42 });
    defer b.deinit();

    const snap_a = try snapshot_params(testing.allocator, &a);
    defer testing.allocator.free(snap_a);

    const snap_b = try snapshot_params(testing.allocator, &b);
    defer testing.allocator.free(snap_b);

    try testing.expectEqualSlices(f32, snap_a, snap_b);
}

// ════════════════════════════════════════════════════════════════
// 複数 seed: 各 seed で self-consistent、かつ seed 間で相互に不一致
// (A と B の両方が複数の seed で同時に成り立つことを明示的に保証する)
// ════════════════════════════════════════════════════════════════

test "seed: multiple seeds — each self-consistent, mutually distinct (init_params)" {
    const Model = sequential(.{ Linear_(4, 8), ReLU, Linear_(8, 4) });
    const seeds = [_]u64{ 0, 7, 42, 2024, 999_999 };

    var snaps: [seeds.len][2][]f32 = undefined;
    defer {
        for (0..seeds.len) |i| {
            for (0..2) |k| testing.allocator.free(snaps[i][k]);
        }
    }

    for (seeds, 0..) |s, i| {
        for (0..2) |k| {
            var session = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = s });
            defer session.deinit();

            snaps[i][k] = try snapshot_params(testing.allocator, &session);
        }
    }

    // (A) 各 seed 内では 2 回分が bitwise 一致
    for (0..seeds.len) |i| {
        try testing.expectEqualSlices(f32, snaps[i][0], snaps[i][1]);
    }
    // (B) 異なる seed 同士では不一致 (全ペア確認)
    for (0..seeds.len) |i| {
        for (i + 1..seeds.len) |j| {
            try testing.expect(!std.mem.eql(f32, snaps[i][0], snaps[j][0]));
        }
    }
}

test "seed: multiple seeds — training 1 step reproducible per seed" {
    const Model = sequential(.{ Linear_(4, 6), ReLU, Linear_(6, 2) });
    const seeds = [_]u64{ 1, 13, 42, 12345 };

    var snaps: [seeds.len][2][]f32 = undefined;
    defer {
        for (0..seeds.len) |i| {
            for (0..2) |k| testing.allocator.free(snaps[i][k]);
        }
    }

    const target = [_]f32{ 1, 0 };

    for (seeds, 0..) |s, i| {
        for (0..2) |k| {
            var session = try trainer(Model, .cpu).init(testing.allocator, {}, .{
                .lr = 1e-2,
                .seed = s,
            });
            defer session.deinit();

            var input = [_]f32{ 0.1, -0.2, 0.3, -0.4 };
            session.zero_grad();
            const out = session.forward(session.tensor(&input, &.{ 1, 4 }));
            const loss = session.mse_loss(out, &target);
            session.backward(loss);
            session.step();

            snaps[i][k] = try snapshot_params(testing.allocator, &session);
        }
    }

    for (0..seeds.len) |i| {
        try testing.expectEqualSlices(f32, snaps[i][0], snaps[i][1]);
    }
    for (0..seeds.len) |i| {
        for (i + 1..seeds.len) |j| {
            try testing.expect(!std.mem.eql(f32, snaps[i][0], snaps[j][0]));
        }
    }
}

test "seed: multiple seeds — eval (inference) reproducible per seed" {
    const Model = sequential(.{ Linear_(3, 5), ReLU, Linear_(5, 2) });
    const seeds = [_]u64{ 3, 14, 159, 2653 };

    var outs: [seeds.len][2][2]f32 = undefined;

    for (seeds, 0..) |s, i| {
        for (0..2) |k| {
            var session = try trainer(Model, .cpu).init(testing.allocator, {}, .{ .seed = s });
            defer session.deinit();

            session.eval();

            var input = [_]f32{ 0.5, -0.1, 0.2 };
            session.zero_grad();
            const out = session.forward(session.tensor(&input, &.{ 1, 3 }));
            @memcpy(&outs[i][k], out.data[0..2]);
        }
    }

    for (0..seeds.len) |i| {
        try testing.expectEqualSlices(f32, &outs[i][0], &outs[i][1]);
    }
    for (0..seeds.len) |i| {
        for (i + 1..seeds.len) |j| {
            try testing.expect(!std.mem.eql(f32, &outs[i][0], &outs[j][0]));
        }
    }
}

test "seed: multiple seeds — dropout mask reproducible per seed" {
    const seeds = [_]u64{ 11, 22, 33, 77, 128 };

    var masks: [seeds.len][2][16]f32 = undefined;

    for (seeds, 0..) |s, i| {
        for (0..2) |k| {
            var module = Module.init(testing.allocator);
            defer module.deinit();

            var rt = try DiffCpuRuntime.init(&module, testing.allocator);
            defer rt.deinit();

            rt.set_seed(s);

            var x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
            const t = rt.make_node(&x, &.{16}, false);
            const y = rt.dropout(t, 0.5);
            @memcpy(&masks[i][k], y.data[0..16]);
        }
    }

    for (0..seeds.len) |i| {
        try testing.expectEqualSlices(f32, &masks[i][0], &masks[i][1]);
    }
    for (0..seeds.len) |i| {
        for (i + 1..seeds.len) |j| {
            try testing.expect(!std.mem.eql(f32, &masks[i][0], &masks[j][0]));
        }
    }
}

// ════════════════════════════════════════════════════════════════
// BatchIterator: 複数 seed で self-consistent / mutually distinct
// (dataloader.zig にも同様の test はあるが、test-diff-cpu に含めるため
//  seed_test.zig にも載せ、通常 test-diff-cpu 走行で確実に走るようにする)
// ════════════════════════════════════════════════════════════════

test "seed: BatchIterator multiple seeds self-consistent & distinct" {
    const seeds = [_]u64{ 0, 7, 42, 2024, 999_999 };
    const N: usize = 64;

    var orders: [seeds.len][2][N]usize = undefined;

    for (seeds, 0..) |s, i| {
        for (0..2) |k| {
            var iter = try BatchIterator.init_with_seed(testing.allocator, N, 8, true, s);
            defer iter.deinit();

            @memcpy(&orders[i][k], iter.indices);
        }
    }

    for (0..seeds.len) |i| {
        try testing.expectEqualSlices(usize, &orders[i][0], &orders[i][1]);
    }
    for (0..seeds.len) |i| {
        for (i + 1..seeds.len) |j| {
            try testing.expect(!std.mem.eql(usize, &orders[i][0], &orders[j][0]));
        }
    }
}
