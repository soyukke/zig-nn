/// diff_runtime_test_helpers.zig: バックエンド共通の勾配テスト基盤
///
/// GradientChecker(Adapter) は Adapter が提供する makeInput / readData / readGrad を通じて
/// CPU / CUDA / Metal 問わず同じ数値勾配チェックを実行する。
///
/// ## Adapter contract (comptime duck-typing)
///
///   pub const Runtime  — ランタイム型 (allocator: Allocator フィールドを持つこと)
///   pub const Tensor   — テンソル型 (*DiffNodeGeneric(...))
///   pub fn makeInput(rt: *Runtime, data: []f32, shape: []const usize, requires_grad: bool) Tensor
///   pub fn readData(rt: *Runtime, tensor: Tensor, dst: []f32) void
///   pub fn readGrad(rt: *Runtime, tensor: Tensor, dst: []f32) ?[]f32
///
/// ## 新しいバックエンド (例: Metal/MPS) の追加手順
///
/// 1. Adapter を定義する (diff_mps_runtime_test.zig):
///
///    const MpsAdapter = struct {
///        pub const Runtime = DiffMpsRuntime;
///        pub const Tensor = DiffMpsTensor;
///
///        pub fn makeInput(rt: *Runtime, data: []f32, shape: []const usize, requires_grad: bool) Tensor {
///            const node = rt.makeTensor(data, shape); // H2D コピー
///            node.requires_grad = requires_grad;
///            return node;
///        }
///        pub fn readData(rt: *Runtime, tensor: Tensor, dst: []f32) void {
///            rt.copyToHost(tensor, dst); // D2H コピー
///        }
///        pub fn readGrad(rt: *Runtime, tensor: Tensor, dst: []f32) ?[]f32 {
///            if (tensor.grad) |grad_ptr| {
///                // grad の D2H コピー (バックエンド依存)
///                rt.copyGradToHost(grad_ptr, dst, tensor.totalElements());
///                return dst;
///            }
///            return null;
///        }
///    };
///
/// 2. テストを宣言する:
///
///    test "diff_mps_runtime: gelu gradient" {
///        // ... MPS runtime セットアップ ...
///        try helpers.testGeluGrad(MpsAdapter, &rt);
///    }
///
/// 3. build.zig に test-diff-mps ターゲットを追加する
///
/// 実装例: diff_cpu_runtime_test.zig (CpuAdapter), diff_cuda_runtime_test.zig (CudaAdapter)
const std = @import("std");
const testing = std.testing;

pub fn GradientChecker(comptime Adapter: type) type {
    const Runtime = Adapter.Runtime;
    const Tensor = Adapter.Tensor;

    return struct {
        /// 数値勾配と解析的勾配を比較して backward の正しさを検証する。
        ///
        /// loss = sum(weights * f(x)) を定義し、
        /// - 数値勾配: (loss(x+eps) - loss(x-eps)) / 2eps  (f64 精度)
        /// - 解析的勾配: backward() で得た x.grad
        /// を比較する。
        pub fn checkGrad(
            comptime f: fn (*Runtime, Tensor) Tensor,
            runtime: *Runtime,
            x_data: []f32,
            shape: []const usize,
            eps: f32,
            tol: f32,
        ) !void {
            const n = x_data.len;
            const allocator = runtime.allocator;
            const num_grad = try allocator.alloc(f32, n);
            defer allocator.free(num_grad);

            // Probe run: 出力サイズ取得 + deterministic な重みを生成
            runtime.resetArena();
            const probe_input = Adapter.makeInput(runtime, x_data, shape, true);
            const probe_out = f(runtime, probe_input);
            const out_n = probe_out.totalElements();
            const weights = try allocator.alloc(f32, out_n);
            defer allocator.free(weights);
            for (weights, 0..) |*w, idx| {
                w.* = @as(f32, @floatFromInt(idx + 1)) * 0.3 + 0.1;
            }

            // 出力読み取り用バッファ
            const out_buf = try allocator.alloc(f32, out_n);
            defer allocator.free(out_buf);

            // ── 数値勾配 ──
            // loss = sum(weights * f(x))、f64 精度で有限差分
            for (0..n) |i| {
                const orig = x_data[i];

                x_data[i] = orig + eps;
                runtime.resetArena();
                const y_plus = f(runtime, Adapter.makeInput(runtime, x_data, shape, true));
                Adapter.readData(runtime, y_plus, out_buf);
                var loss_plus: f64 = 0;
                for (out_buf, 0..) |v, j| loss_plus += @as(f64, v) * @as(f64, weights[j]);

                x_data[i] = orig - eps;
                runtime.resetArena();
                const y_minus = f(runtime, Adapter.makeInput(runtime, x_data, shape, true));
                Adapter.readData(runtime, y_minus, out_buf);
                var loss_minus: f64 = 0;
                for (out_buf, 0..) |v, j| loss_minus += @as(f64, v) * @as(f64, weights[j]);

                num_grad[i] = @floatCast((loss_plus - loss_minus) / (2.0 * @as(f64, eps)));
                x_data[i] = orig;
            }

            // ── 解析的勾配 ──
            // loss = sum(y * weights) を runtime ops で構築 → backward
            runtime.resetArena();
            const x_node = Adapter.makeInput(runtime, x_data, shape, true);
            const y_node = f(runtime, x_node);

            // weights テンソル (requires_grad=false) を y_node と同じ shape で作成
            const w_node = Adapter.makeInput(runtime, weights, y_node.shape[0..y_node.ndim], false);
            const product = runtime.mul(y_node, w_node);
            const flat_shape = [_]usize{out_n};
            const flat = runtime.reshape(product, &flat_shape);
            const loss = runtime.reductionSum(flat, 0);
            runtime.backward(loss);

            // 勾配を読み取って比較
            const ana_grad_buf = try allocator.alloc(f32, n);
            defer allocator.free(ana_grad_buf);
            const ana_grad = Adapter.readGrad(runtime, x_node, ana_grad_buf) orelse
                return error.TestExpectedApproxEqAbs;

            for (0..n) |i| {
                const diff = @abs(ana_grad[i] - num_grad[i]);
                const scale = @max(@abs(ana_grad[i]), @abs(num_grad[i]));
                const abs_tol: f32 = 3e-3;
                if (diff > @max(tol * scale, abs_tol)) {
                    std.debug.print("Gradient mismatch at [{d}]: analytical={d:.6}, numerical={d:.6}, diff={d:.6}\n", .{ i, ana_grad[i], num_grad[i], diff });
                    return error.TestExpectedApproxEqAbs;
                }
            }
        }
    };
}

// ════════════════════════════════════════════════════════════════
// 共通テストケース: パラメータ不要な単項・二項演算
// ════════════════════════════════════════════════════════════════

pub fn testGeluGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -1.0, 0.0, 0.5, 1.0, 2.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.gelu(x);
            }
        }.f,
        rt,
        &data,
        &.{5},
        1e-4,
        1e-2,
    );
}

pub fn testSiluGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.silu(x);
            }
        }.f,
        rt,
        &data,
        &.{5},
        1e-4,
        1e-2,
    );
}

pub fn testSquareGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -1.0, 0.5, 1.0, 2.0, 3.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.square(x);
            }
        }.f,
        rt,
        &data,
        &.{6},
        1e-4,
        1e-2,
    );
}

pub fn testReductionMeanGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.reductionMean(x, -1);
            }
        }.f,
        rt,
        &data,
        &.{ 2, 3 },
        1e-4,
        1e-2,
    );
}

pub fn testTanhGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -1.0, 0.0, 0.5, 1.0, 2.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.tanh_(x);
            }
        }.f,
        rt,
        &data,
        &.{6},
        1e-4,
        1e-2,
    );
}

pub fn testSigmoidGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -1.0, 0.0, 0.5, 1.0, 2.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.sigmoid(x);
            }
        }.f,
        rt,
        &data,
        &.{6},
        1e-4,
        1e-2,
    );
}

pub fn testNegativeGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, -2.0, 3.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.negative(x);
            }
        }.f,
        rt,
        &data,
        &.{3},
        1e-4,
        1e-2,
    );
}

pub fn testSoftmaxGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.softmax(x, -1);
            }
        }.f,
        rt,
        &data,
        &.{ 2, 3 },
        1e-4,
        1e-2,
    );
}

pub fn testReluGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -0.5, 0.1, 0.5, 1.0, 2.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.relu(x);
            }
        }.f,
        rt,
        &data,
        &.{6},
        1e-4,
        1e-2,
    );
}

pub fn testReductionSumGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.reductionSum(x, -1);
            }
        }.f,
        rt,
        &data,
        &.{ 2, 3 },
        1e-4,
        1e-2,
    );
}

pub fn testTranspose3DGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.transpose(x, 1, 2);
            }
        }.f,
        rt,
        &data,
        &.{ 1, 2, 3 },
        1e-4,
        1e-2,
    );
}

pub fn testReshapeGrad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.reshape(x, &.{ 2, 3 });
            }
        }.f,
        rt,
        &data,
        &.{6},
        1e-4,
        1e-2,
    );
}

// ════════════════════════════════════════════════════════════════
// 境界サイズテスト: threadgroup / tiling 境界での正当性検証
//
// GPU カーネルは threadgroup サイズ (256) や tiling ブロック (64) の
// 境界で端数処理のバグが出やすい。小データと合わせて検証する。
//
// NOTE: softmax は要素数が増えると exp の累積で数値勾配の有限差分が
// 不正確になるため tolerance を緩めている (5e-2)。
// ════════════════════════════════════════════════════════════════

fn makeRandomData(buf: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (buf) |*v| v.* = r.float(f32) * 2.0 - 1.0;
}

pub fn testGeluBoundary(comptime A: type, rt: *A.Runtime, comptime n: usize) !void {
    var data: [n]f32 = undefined;
    makeRandomData(&data, n);
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.gelu(x);
            }
        }.f,
        rt,
        &data,
        &.{n},
        1e-4,
        1e-2,
    );
}

pub fn testMatmulBoundary(comptime A: type, rt: *A.Runtime, comptime M: usize, comptime K: usize, comptime N: usize) !void {
    var data: [M * K]f32 = undefined;
    makeRandomData(&data, M * K + N);
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                const w = ctx.param(.{ .index = 0 });
                return ctx.matmul(x, w);
            }
        }.f,
        rt,
        &data,
        &.{ M, K },
        1e-3,
        5e-2,
    );
}

pub fn testSoftmaxBoundary(comptime A: type, rt: *A.Runtime, comptime rows: usize, comptime cols: usize) !void {
    var data: [rows * cols]f32 = undefined;
    makeRandomData(&data, rows * cols);
    try GradientChecker(A).checkGrad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.softmax(x, -1);
            }
        }.f,
        rt,
        &data,
        &.{ rows, cols },
        1e-3,
        5e-2,
    );
}
