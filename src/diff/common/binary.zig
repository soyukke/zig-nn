// diff/common/binary.zig: Pointwise binary op の数式一極集中
//
// add/sub/mul/div の forward / derivative をここに集約する。
// 各 backend runtime は buffer アクセス + shape dispatch (same / broadcast) を担当。
//
// 数式は 3 関数セット:
//   fwd(a, b)      -> y
//   deriv_a(a,b,y) -> dy/da
//   deriv_b(a,b,y) -> dy/db
//
// Broadcasting: 出力サイズ n_out に対し、a は index を n_a で mod, b は n_b で mod。
// これによって same-shape (n_a == n_b == n_out) と broadcast どちらも同じ formula。
// backward は mod index で grad を accumulate するため、broadcast 次元で自然に
// "broadcast reduce" が起きる。
//
// 例:
//   add: fwd = a + b,   deriv_a = 1,         deriv_b = 1
//   sub: fwd = a - b,   deriv_a = 1,         deriv_b = -1
//   mul: fwd = a * b,   deriv_a = b,         deriv_b = a
//   div: fwd = a / b,   deriv_a = 1/b,       deriv_b = -a / b^2

pub const Kind = struct {
    /// forward: y = fwd(a, b)
    fwd: *const fn (a: f32, b: f32) f32,
    /// backward: dy/da(a, b, y). y は再計算回避 (必要なら使用)。
    deriv_a: *const fn (a: f32, b: f32, y: f32) f32,
    /// backward: dy/db(a, b, y)
    deriv_b: *const fn (a: f32, b: f32, y: f32) f32,
};

// ── Add ──

pub const Add: Kind = .{ .fwd = add_fwd, .deriv_a = one3, .deriv_b = one3 };
fn add_fwd(a: f32, b: f32) f32 {
    return a + b;
}

// ── Sub ──

pub const Sub: Kind = .{ .fwd = sub_fwd, .deriv_a = one3, .deriv_b = minus_one3 };
fn sub_fwd(a: f32, b: f32) f32 {
    return a - b;
}

// ── Mul ──

pub const Mul: Kind = .{ .fwd = mul_fwd, .deriv_a = mul_deriv_a, .deriv_b = mul_deriv_b };
fn mul_fwd(a: f32, b: f32) f32 {
    return a * b;
}
fn mul_deriv_a(_: f32, b: f32, _: f32) f32 {
    return b;
}
fn mul_deriv_b(a: f32, _: f32, _: f32) f32 {
    return a;
}

// ── Div ──

pub const Div: Kind = .{ .fwd = div_fwd, .deriv_a = div_deriv_a, .deriv_b = div_deriv_b };
fn div_fwd(a: f32, b: f32) f32 {
    return a / b;
}
fn div_deriv_a(_: f32, b: f32, _: f32) f32 {
    return 1.0 / b;
}
fn div_deriv_b(a: f32, b: f32, _: f32) f32 {
    return -a / (b * b);
}

// ── 共通の定数 deriv ──

fn one3(_: f32, _: f32, _: f32) f32 {
    return 1.0;
}
fn minus_one3(_: f32, _: f32, _: f32) f32 {
    return -1.0;
}

// ── tests: 数値微分との一致 ──

test "binary deriv matches numerical gradient" {
    const testing = @import("std").testing;
    const eps: f32 = 1e-3;

    const cases = [_]struct { k: Kind, a: f32, b: f32 }{
        .{ .k = Add, .a = 0.7, .b = 0.3 },
        .{ .k = Sub, .a = 0.7, .b = 0.3 },
        .{ .k = Mul, .a = 0.7, .b = 0.3 },
        .{ .k = Div, .a = 0.7, .b = 0.3 },
    };

    for (cases) |c| {
        const y = c.k.fwd(c.a, c.b);
        const da_analytical = c.k.deriv_a(c.a, c.b, y);
        const db_analytical = c.k.deriv_b(c.a, c.b, y);

        const da_numerical = (c.k.fwd(c.a + eps, c.b) - c.k.fwd(c.a - eps, c.b)) / (2.0 * eps);
        const db_numerical = (c.k.fwd(c.a, c.b + eps) - c.k.fwd(c.a, c.b - eps)) / (2.0 * eps);

        try testing.expectApproxEqAbs(da_numerical, da_analytical, 1e-2);
        try testing.expectApproxEqAbs(db_numerical, db_analytical, 1e-2);
    }
}
