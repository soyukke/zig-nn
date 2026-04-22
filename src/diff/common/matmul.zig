// diff/common/matmul.zig: Matmul backward の数式一極集中
//
// PyTorch ATen の `derivatives.yaml` に倣い、matmul backward を
// 2 つの matmul の合成として宣言的に表現する:
//
//   y = a @ b
//   ga = grad_out @ b^T        ← これも matmul (+ 転置)
//   gb = a^T @ grad_out        ← これも matmul (+ 転置)
//
// PyTorch 流の "backward = forward op 合成" の精神そのまま。ただし実装上は
// 中間 transpose テンソルを確保せず、cpu_backend の fused (transpose + matmul
// + accumulate) プリミティブ 1 BLAS 呼び出しで完結させる（=概念的には
// "go @ b^T" と "a^T @ go" を新 DiffTensor として生成するのと等価だが、
// 中間確保ゼロ・fused BLAS）。
//
// CPU runtime (`[]f32`) と MPS runtime (UMA バッファを `bufPtr` で `[*]f32` に
// 変換) の両方から同じ実装を再利用する。cpu_backend は Accelerate / OpenBLAS
// 経由で SIMD + BLAS 最適化されている。

const cpu_backend = @import("../../backend/cpu.zig");

/// 2D matmul の backward。
///
/// 形状: a (M×K), b (K×N), y = a @ b (M×N), go (M×N)
///   - ga (M×K) が非 null なら ga += go @ b^T
///   - gb (K×N) が非 null なら gb += a^T @ go
pub fn backward2D(
    go: [*]const f32,
    a: [*]const f32,
    b: [*]const f32,
    ga: ?[*]f32,
    gb: ?[*]f32,
    M: usize,
    K: usize,
    N: usize,
) void {
    // ga += go @ b^T  (PyTorch: grad.mm(b.t()))
    if (ga) |g| cpu_backend.matmulTransBAccum(f32, go, b, g, M, N, K);
    // gb += a^T @ go  (PyTorch: a.t().mm(grad))
    if (gb) |g| cpu_backend.matmulTransAAccum(f32, a, go, g, K, M, N);
}

/// 3D batched matmul の backward。a と b の両方が B 個のバッチを持つ。
/// 各バッチで backward2D を呼ぶだけ。
pub fn backward3D(
    go: [*]const f32,
    a: [*]const f32,
    b: [*]const f32,
    ga: ?[*]f32,
    gb: ?[*]f32,
    B: usize,
    M: usize,
    K: usize,
    N: usize,
) void {
    for (0..B) |batch| {
        const go_b = go + batch * M * N;
        const a_b = a + batch * M * K;
        const b_b = b + batch * K * N;
        const ga_b: ?[*]f32 = if (ga) |g| g + batch * M * K else null;
        const gb_b: ?[*]f32 = if (gb) |g| g + batch * K * N else null;
        backward2D(go_b, a_b, b_b, ga_b, gb_b, M, K, N);
    }
}

/// 2D @ 3D batched matmul の backward。
/// a は batch 非依存 (2D M×K, 全バッチで共有)、b と go は 3D (B×K×N, B×M×N)。
///   - ga (2D M×K) は全バッチの Σ go[b] @ b[b]^T を accumulate
///   - gb (3D B×K×N) は各バッチ毎に a^T @ go[b]
pub fn backward2Dx3D(
    go: [*]const f32,
    a: [*]const f32,
    b: [*]const f32,
    ga: ?[*]f32,
    gb: ?[*]f32,
    B: usize,
    M: usize,
    K: usize,
    N: usize,
) void {
    for (0..B) |batch| {
        const go_b = go + batch * M * N;
        if (ga) |g| {
            // ga は全 batch 共有、各 batch の寄与を accumulate
            cpu_backend.matmulTransBAccum(f32, go_b, b + batch * K * N, g, M, N, K);
        }
        if (gb) |g| {
            cpu_backend.matmulTransAAccum(f32, a, go_b, g + batch * K * N, K, M, N);
        }
    }
}

// ── tests: y = a @ b の数値微分との一致 ──

test "backward2D matches analytical gradient (2x3 @ 3x2)" {
    const std = @import("std");
    const testing = std.testing;

    // a = [[1,2,3],[4,5,6]]  (2x3)
    // b = [[1,0],[0,1],[1,1]]  (3x2)
    // y = a @ b = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 1, 1, 1 };

    // go = all ones (dL/dy = 1)  → dL/da = 1 @ b^T, dL/db = a^T @ 1
    const go = [_]f32{ 1, 1, 1, 1 };

    var ga = [_]f32{ 0, 0, 0, 0, 0, 0 };
    var gb = [_]f32{ 0, 0, 0, 0, 0, 0 };

    backward2D(&go, &a, &b, &ga, &gb, 2, 3, 2);

    // ga expected: go (2x2 all-1) @ b^T (2x3) = [[1,1,2],[1,1,2]]
    try testing.expectEqualSlices(f32, &.{ 1, 1, 2, 1, 1, 2 }, &ga);

    // gb expected: a^T (3x2) @ go (2x2) = [[1+4,1+4],[2+5,2+5],[3+6,3+6]]
    //                                    = [[5,5],[7,7],[9,9]]
    try testing.expectEqualSlices(f32, &.{ 5, 5, 7, 7, 9, 9 }, &gb);
}

test "backward2D skips null grads" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 0, 0, 1 };
    const go = [_]f32{ 1, 1, 1, 1 };

    // ga only
    var ga = [_]f32{ 0, 0, 0, 0 };
    backward2D(&go, &a, &b, &ga, null, 2, 2, 2);
    // go (2x2 all-1) @ b^T (I^T = I) = all-1 matrix
    try @import("std").testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, &ga);

    // gb only
    var gb = [_]f32{ 0, 0, 0, 0 };
    backward2D(&go, &a, &b, null, &gb, 2, 2, 2);
    // a^T @ go = [[1+3,1+3],[2+4,2+4]] = [[4,4],[6,6]]
    try @import("std").testing.expectEqualSlices(f32, &.{ 4, 4, 6, 6 }, &gb);
}
