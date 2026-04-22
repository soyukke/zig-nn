// diff/common/softmax.zig: Softmax / LogSoftmax backward の数式一極集中
//
// どちらも「最後の軸 (cols) に沿って行ごとに softmax を取る」前提の実装。
// 3D 以上は呼び出し側で reshape して rows × cols に flatten してから渡す。
//
// softmax backward:
//   ga[i,j] += s[i,j] * (go[i,j] - dot_i)
//     where dot_i = sum_j(go[i,j] * s[i,j])
//
// logSoftmax backward:
//   ga[i,j] += go[i,j] - s[i,j] * sum_go_i
//     where sum_go_i = sum_j(go[i,j])
//   s は softmax(x) = exp(logSoftmax(x))
//
// どちらも parent.grad への accumulate (+=)。

/// softmax 出力 s (rows * cols) と出力勾配 go から parent 勾配に accumulate。
pub fn softmaxBackward(
    ga: [*]f32,
    go: [*]const f32,
    s: [*]const f32,
    rows: usize,
    cols: usize,
) void {
    for (0..rows) |i| {
        const base = i * cols;
        var dot: f32 = 0;
        for (0..cols) |j| dot += go[base + j] * s[base + j];
        for (0..cols) |j| {
            ga[base + j] += s[base + j] * (go[base + j] - dot);
        }
    }
}

/// logSoftmax 出力 log_s (= log-softmax values) から softmax を復元しつつ backward 計算。
/// s は呼び出し側で用意しなくて良い（exp で都度計算、state 不要）。
pub fn logSoftmaxBackward(
    ga: [*]f32,
    go: [*]const f32,
    log_s: [*]const f32,
    rows: usize,
    cols: usize,
) void {
    for (0..rows) |i| {
        const base = i * cols;
        var sum_go: f32 = 0;
        for (0..cols) |j| sum_go += go[base + j];
        for (0..cols) |j| {
            const s_j = @exp(log_s[base + j]);
            ga[base + j] += go[base + j] - s_j * sum_go;
        }
    }
}

// ── tests: 数値微分との一致 (1D 入力: 1行 × N列) ──

test "softmaxBackward matches numerical gradient" {
    const std = @import("std");
    const testing = std.testing;
    const n = 4;
    const x = [_]f32{ 0.3, -0.5, 1.2, 0.0 };

    // compute softmax forward
    var s: [n]f32 = undefined;
    {
        var max_v: f32 = x[0];
        for (x[1..]) |v| max_v = @max(max_v, v);
        var sum: f32 = 0;
        for (0..n) |i| {
            s[i] = @exp(x[i] - max_v);
            sum += s[i];
        }
        for (0..n) |i| s[i] /= sum;
    }

    // arbitrary loss = sum_i w_i * s_i
    const w = [_]f32{ 1.0, 2.0, -0.5, 0.7 };
    var go: [n]f32 = undefined;
    for (0..n) |i| go[i] = w[i]; // dL/ds_i = w_i

    var ga = [_]f32{ 0, 0, 0, 0 };
    softmaxBackward(&ga, &go, &s, 1, n);

    // numerical: dL/dx_k via finite diff on softmax forward
    const eps: f32 = 1e-3;
    for (0..n) |k| {
        var xp = x;
        xp[k] += eps;
        var sp: [n]f32 = undefined;
        {
            var max_v: f32 = xp[0];
            for (xp[1..]) |v| max_v = @max(max_v, v);
            var sum: f32 = 0;
            for (0..n) |i| {
                sp[i] = @exp(xp[i] - max_v);
                sum += sp[i];
            }
            for (0..n) |i| sp[i] /= sum;
        }
        var lp: f32 = 0;
        for (0..n) |i| lp += w[i] * sp[i];

        var xm = x;
        xm[k] -= eps;
        var sm: [n]f32 = undefined;
        {
            var max_v: f32 = xm[0];
            for (xm[1..]) |v| max_v = @max(max_v, v);
            var sum: f32 = 0;
            for (0..n) |i| {
                sm[i] = @exp(xm[i] - max_v);
                sum += sm[i];
            }
            for (0..n) |i| sm[i] /= sum;
        }
        var lm: f32 = 0;
        for (0..n) |i| lm += w[i] * sm[i];

        const num = (lp - lm) / (2.0 * eps);
        try testing.expectApproxEqAbs(num, ga[k], 1e-2);
    }
}
