// diff/common/unary.zig: Pointwise unary op の数式一極集中
//
// forward / backward の数式をここに集約し、各 backend (cpu/mps) は buffer アクセス
// のみを担当する。PyTorch ATen の TensorIterator + derivatives.yaml に相当する
// "数式の single source of truth"。
//
// 各 Kind は fwd(x) -> y と deriv(x, y) -> dy/dx を持つ。
//   - fwd:   要素ごとの forward 変換
//   - deriv: x (入力) と y (forward の結果) から導関数を計算
//            y を利用できる場合 (sigmoid, tanh, exp, sqrt 等) は再計算不要で速い
//
// 使い方 (CPU/MPS runtime 側):
//   for (0..n) |i| dst[i] = Kind.fwd(src[i]);           // forward loop
//   for (0..n) |i| ga[i] += go[i] * Kind.deriv(x[i], y[i]); // backward loop
//
// CUDA 側は pre-compiled kernel を使うため、この module の pure 関数は直接は
// 呼ばれない。ただし Kind.fwd / Kind.deriv を「検証可能な reference 実装」として
// CUDA kernel の数値一致テストに流用できる。

const std = @import("std");

pub const Kind = struct {
    /// forward: y = fwd(x)
    fwd: *const fn (x: f32) f32,
    /// backward derivative: dy/dx, 必要に応じて x と y の両方を参照できる
    deriv: *const fn (x: f32, y: f32) f32,
};

// ── 実装: 各 unary op の fwd と deriv ──
//
// 命名規則: Pascal case (Negative, Square, ...). Runtime からは
//   @import("../common/unary.zig").Negative
// のように参照する。

pub const Negative: Kind = .{ .fwd = negFwd, .deriv = negDeriv };
fn negFwd(x: f32) f32 {
    return -x;
}
fn negDeriv(_: f32, _: f32) f32 {
    return -1.0;
}

pub const Square: Kind = .{ .fwd = sqFwd, .deriv = sqDeriv };
fn sqFwd(x: f32) f32 {
    return x * x;
}
fn sqDeriv(x: f32, _: f32) f32 {
    return 2.0 * x;
}

pub const Sigmoid: Kind = .{ .fwd = sigFwd, .deriv = sigDeriv };
fn sigFwd(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
fn sigDeriv(_: f32, y: f32) f32 {
    return y * (1.0 - y);
}

pub const Tanh: Kind = .{ .fwd = tanhFwd, .deriv = tanhDeriv };
fn tanhFwd(x: f32) f32 {
    return std.math.tanh(x);
}
fn tanhDeriv(_: f32, y: f32) f32 {
    return 1.0 - y * y;
}

pub const Relu: Kind = .{ .fwd = reluFwd, .deriv = reluDeriv };
fn reluFwd(x: f32) f32 {
    return if (x > 0) x else 0;
}
fn reluDeriv(x: f32, _: f32) f32 {
    return if (x > 0) @as(f32, 1.0) else @as(f32, 0.0);
}

pub const Exp: Kind = .{ .fwd = expFwd, .deriv = expDeriv };
fn expFwd(x: f32) f32 {
    return @exp(x);
}
fn expDeriv(_: f32, y: f32) f32 {
    return y; // d/dx exp(x) = exp(x) = y
}

pub const Log: Kind = .{ .fwd = logFwd, .deriv = logDeriv };
fn logFwd(x: f32) f32 {
    return @log(x);
}
fn logDeriv(x: f32, _: f32) f32 {
    return 1.0 / x;
}

pub const Abs: Kind = .{ .fwd = absFwd, .deriv = absDeriv };
fn absFwd(x: f32) f32 {
    return @abs(x);
}
fn absDeriv(x: f32, _: f32) f32 {
    return if (x >= 0) @as(f32, 1.0) else @as(f32, -1.0);
}

pub const Sqrt: Kind = .{ .fwd = sqrtFwd, .deriv = sqrtDeriv };
fn sqrtFwd(x: f32) f32 {
    return @sqrt(x);
}
fn sqrtDeriv(_: f32, y: f32) f32 {
    return 0.5 / y; // d/dx sqrt(x) = 1 / (2*sqrt(x)) = 1 / (2*y)
}

/// SiLU (Swish): x * sigmoid(x)
///   fwd:   y = x / (1 + exp(-x))
///   deriv: sig + x * sig * (1 - sig), where sig = 1/(1+exp(-x))
///
/// 旧実装は forward で sig を cache していたが、deriv 内で再計算する方針に統一する。
/// backward パスで exp() 1 回分の追加コストが発生するが、コードはシンプルかつ
/// 他 unary op と同じ driver を使える。
pub const Silu: Kind = .{ .fwd = siluFwd, .deriv = siluDeriv };
fn siluFwd(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}
fn siluDeriv(x: f32, _: f32) f32 {
    const sig = 1.0 / (1.0 + @exp(-x));
    return sig + x * sig * (1.0 - sig);
}

/// GELU (tanh approximation)
///   fwd:   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///   deriv: 0.5 * (1 + tanh) + 0.5 * x * sech^2 * inner'
pub const Gelu: Kind = .{ .fwd = geluFwd, .deriv = geluDeriv };
const GELU_C: f32 = 0.7978845608028654; // sqrt(2/pi)
const GELU_K: f32 = 0.044715;
fn geluFwd(x: f32) f32 {
    const inner = GELU_C * (x + GELU_K * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}
fn geluDeriv(x: f32, _: f32) f32 {
    const inner = GELU_C * (x + GELU_K * x * x * x);
    const tanh_val = std.math.tanh(inner);
    const sech2 = 1.0 - tanh_val * tanh_val;
    const inner_deriv = GELU_C * (1.0 + 3.0 * GELU_K * x * x);
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_deriv;
}

// ── tests: fwd/deriv の自己整合性 (数値微分との一致) ──

test "unary deriv matches numerical gradient" {
    const testing = std.testing;
    const eps: f32 = 1e-3;
    const kinds = [_]struct { k: Kind, name: []const u8, x: f32 }{
        .{ .k = Negative, .name = "Negative", .x = 0.7 },
        .{ .k = Square, .name = "Square", .x = 0.7 },
        .{ .k = Sigmoid, .name = "Sigmoid", .x = 0.7 },
        .{ .k = Tanh, .name = "Tanh", .x = 0.7 },
        .{ .k = Relu, .name = "Relu", .x = 0.7 },
        .{ .k = Exp, .name = "Exp", .x = 0.7 },
        .{ .k = Log, .name = "Log", .x = 0.7 },
        .{ .k = Abs, .name = "Abs", .x = 0.7 },
        .{ .k = Sqrt, .name = "Sqrt", .x = 0.7 },
        .{ .k = Gelu, .name = "Gelu", .x = 0.7 },
        .{ .k = Silu, .name = "Silu", .x = 0.7 },
    };
    for (kinds) |kd| {
        const y = kd.k.fwd(kd.x);
        const analytical = kd.k.deriv(kd.x, y);
        const numerical = (kd.k.fwd(kd.x + eps) - kd.k.fwd(kd.x - eps)) / (2.0 * eps);
        try testing.expectApproxEqAbs(numerical, analytical, 1e-2);
    }
}
