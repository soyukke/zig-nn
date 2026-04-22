/// バックエンド抽象化。
/// comptime dispatch: ビルド時にバックエンドが確定するため、
/// VTableのオーバーヘッドなしで最適なコードが生成される。
const cpu = @import("cpu.zig");
const simd = @import("simd.zig");

pub const BackendType = enum {
    cpu,
    simd,
    // metal,  // 将来追加
};

/// コンパイル時にバックエンドを選択するディスパッチャー。
/// 全ての演算は comptime T: type でf16/f32/f64に対応する。
pub fn Backend(comptime backend_type: BackendType) type {
    return struct {
        /// 行列積: C = A @ B
        /// A: (m x k), B: (k x n), C: (m x n)  row-major
        pub fn matmul(
            comptime T: type,
            a: [*]const T,
            b: [*]const T,
            c: [*]T,
            m: usize,
            k: usize,
            n: usize,
        ) void {
            switch (backend_type) {
                .cpu => cpu.matmul(T, a, b, c, m, k, n),
                .simd => simd.matmul(T, a, b, c, m, k, n),
            }
        }

        /// element-wise add: c[i] = a[i] + b[i]
        pub fn add(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, len: usize) void {
            switch (backend_type) {
                .cpu => cpu.add(T, a, b, c, len),
                .simd => simd.add(T, a, b, c, len),
            }
        }

        /// element-wise mul: c[i] = a[i] * b[i]
        pub fn mul(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, len: usize) void {
            switch (backend_type) {
                .cpu => cpu.mul(T, a, b, c, len),
                .simd => simd.mul(T, a, b, c, len),
            }
        }

        /// element-wise sub: c[i] = a[i] - b[i]
        pub fn sub(comptime T: type, a: [*]const T, b: [*]const T, c: [*]T, len: usize) void {
            switch (backend_type) {
                .cpu => cpu.sub(T, a, b, c, len),
                .simd => simd.sub(T, a, b, c, len),
            }
        }

        /// ReLU: c[i] = max(0, a[i])
        pub fn relu(comptime T: type, a: [*]const T, c: [*]T, len: usize) void {
            switch (backend_type) {
                .cpu => cpu.relu(T, a, c, len),
                .simd => simd.relu(T, a, c, len),
            }
        }

        /// scale: c[i] = a[i] * scalar
        pub fn scale(comptime T: type, a: [*]const T, scalar: T, c: [*]T, len: usize) void {
            switch (backend_type) {
                .cpu => cpu.scale(T, a, scalar, c, len),
                .simd => simd.scale(T, a, scalar, c, len),
            }
        }
    };
}
