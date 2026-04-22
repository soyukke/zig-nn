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
///        pub fn makeInput(
///            rt: *Runtime,
///            data: []f32,
///            shape: []const usize,
///            requires_grad: bool,
///        ) Tensor {
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
const log = @import("../log.zig").gradcheck;

pub fn gradient_checker(comptime Adapter: type) type {
    const Runtime = Adapter.Runtime;
    const Tensor = Adapter.Tensor;

    return struct {
        /// 数値勾配と解析的勾配を比較して backward の正しさを検証する。
        ///
        /// loss = sum(weights * f(x)) を定義し、
        /// - 数値勾配: (loss(x+eps) - loss(x-eps)) / 2eps  (f64 精度)
        /// - 解析的勾配: backward() で得た x.grad
        /// を比較する。
        pub fn check_grad(
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
            const out_n = probe_output_size(f, runtime, x_data, shape);
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
            compute_numerical_grad(
                f,
                runtime,
                x_data,
                shape,
                eps,
                weights,
                out_buf,
                num_grad,
            );

            // ── 解析的勾配 ──
            // 勾配を読み取って比較
            const ana_grad_buf = try allocator.alloc(f32, n);
            defer allocator.free(ana_grad_buf);

            const ana_grad = compute_analytical_grad(
                f,
                runtime,
                x_data,
                shape,
                weights,
                out_n,
                ana_grad_buf,
            ) orelse return error.TestExpectedApproxEqAbs;

            try compare_grads(ana_grad, num_grad, tol);
        }

        fn probe_output_size(
            comptime f: fn (*Runtime, Tensor) Tensor,
            runtime: *Runtime,
            x_data: []f32,
            shape: []const usize,
        ) usize {
            runtime.reset_arena();
            const probe_input = Adapter.make_input(runtime, x_data, shape, true);
            const probe_out = f(runtime, probe_input);
            return probe_out.total_elements();
        }

        fn compute_numerical_grad(
            comptime f: fn (*Runtime, Tensor) Tensor,
            runtime: *Runtime,
            x_data: []f32,
            shape: []const usize,
            eps: f32,
            weights: []const f32,
            out_buf: []f32,
            num_grad: []f32,
        ) void {
            const n = x_data.len;
            for (0..n) |i| {
                const orig = x_data[i];

                x_data[i] = orig + eps;
                runtime.reset_arena();
                const y_plus = f(runtime, Adapter.make_input(runtime, x_data, shape, true));
                Adapter.read_data(runtime, y_plus, out_buf);
                var loss_plus: f64 = 0;
                for (out_buf, 0..) |v, j| loss_plus += @as(f64, v) * @as(f64, weights[j]);

                x_data[i] = orig - eps;
                runtime.reset_arena();
                const y_minus = f(runtime, Adapter.make_input(runtime, x_data, shape, true));
                Adapter.read_data(runtime, y_minus, out_buf);
                var loss_minus: f64 = 0;
                for (out_buf, 0..) |v, j| loss_minus += @as(f64, v) * @as(f64, weights[j]);

                num_grad[i] = @floatCast((loss_plus - loss_minus) / (2.0 * @as(f64, eps)));
                x_data[i] = orig;
            }
        }

        fn compute_analytical_grad(
            comptime f: fn (*Runtime, Tensor) Tensor,
            runtime: *Runtime,
            x_data: []f32,
            shape: []const usize,
            weights: []f32,
            out_n: usize,
            ana_grad_buf: []f32,
        ) ?[]f32 {
            // loss = sum(y * weights) を runtime ops で構築 → backward
            runtime.reset_arena();
            const x_node = Adapter.make_input(runtime, x_data, shape, true);
            const y_node = f(runtime, x_node);

            // weights テンソル (requires_grad=false) を y_node と同じ shape で作成
            const w_node = Adapter.make_input(
                runtime,
                weights,
                y_node.shape[0..y_node.ndim],
                false,
            );
            const product = runtime.mul(y_node, w_node);
            const flat_shape = [_]usize{out_n};
            const flat = runtime.reshape(product, &flat_shape);
            const loss = runtime.reduction_sum(flat, 0);
            runtime.backward(loss);

            return Adapter.read_grad(runtime, x_node, ana_grad_buf);
        }

        fn compare_grads(ana_grad: []const f32, num_grad: []const f32, tol: f32) !void {
            for (0..ana_grad.len) |i| {
                const diff = @abs(ana_grad[i] - num_grad[i]);
                const scale = @max(@abs(ana_grad[i]), @abs(num_grad[i]));
                const abs_tol: f32 = 3e-3;
                if (diff > @max(tol * scale, abs_tol)) {
                    log.err(
                        "mismatch at [{d}]: analytical={d:.6}, numerical={d:.6}, diff={d:.6}",
                        .{ i, ana_grad[i], num_grad[i], diff },
                    );
                    return error.TestExpectedApproxEqAbs;
                }
            }
        }
    };
}

// ════════════════════════════════════════════════════════════════
// 共通テストケース: パラメータ不要な単項・二項演算
// ════════════════════════════════════════════════════════════════

pub fn test_gelu_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -1.0, 0.0, 0.5, 1.0, 2.0 };
    try gradient_checker(A).check_grad(
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

pub fn test_silu_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    try gradient_checker(A).check_grad(
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

pub fn test_square_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -1.0, 0.5, 1.0, 2.0, 3.0 };
    try gradient_checker(A).check_grad(
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

pub fn test_reduction_mean_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try gradient_checker(A).check_grad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.reduction_mean(x, -1);
            }
        }.f,
        rt,
        &data,
        &.{ 2, 3 },
        1e-4,
        1e-2,
    );
}

pub fn test_tanh_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -1.0, 0.0, 0.5, 1.0, 2.0 };
    try gradient_checker(A).check_grad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.tanh(x);
            }
        }.f,
        rt,
        &data,
        &.{6},
        1e-4,
        1e-2,
    );
}

pub fn test_sigmoid_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -1.0, 0.0, 0.5, 1.0, 2.0 };
    try gradient_checker(A).check_grad(
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

pub fn test_negative_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, -2.0, 3.0 };
    try gradient_checker(A).check_grad(
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

pub fn test_softmax_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0 };
    try gradient_checker(A).check_grad(
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

pub fn test_relu_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ -2.0, -0.5, 0.1, 0.5, 1.0, 2.0 };
    try gradient_checker(A).check_grad(
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

pub fn test_reduction_sum_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try gradient_checker(A).check_grad(
        struct {
            fn f(ctx: *A.Runtime, x: A.Tensor) A.Tensor {
                return ctx.reduction_sum(x, -1);
            }
        }.f,
        rt,
        &data,
        &.{ 2, 3 },
        1e-4,
        1e-2,
    );
}

pub fn test_transpose3_d_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try gradient_checker(A).check_grad(
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

pub fn test_reshape_grad(comptime A: type, rt: *A.Runtime) !void {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try gradient_checker(A).check_grad(
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

fn make_random_data(buf: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (buf) |*v| v.* = r.float(f32) * 2.0 - 1.0;
}

pub fn test_gelu_boundary(comptime A: type, rt: *A.Runtime, comptime n: usize) !void {
    var data: [n]f32 = undefined;
    make_random_data(&data, n);
    try gradient_checker(A).check_grad(
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

pub fn test_matmul_boundary(
    comptime A: type,
    rt: *A.Runtime,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
) !void {
    var data: [M * K]f32 = undefined;
    make_random_data(&data, M * K + N);
    try gradient_checker(A).check_grad(
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

pub fn test_softmax_boundary(
    comptime A: type,
    rt: *A.Runtime,
    comptime rows: usize,
    comptime cols: usize,
) !void {
    var data: [rows * cols]f32 = undefined;
    make_random_data(&data, rows * cols);
    try gradient_checker(A).check_grad(
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
