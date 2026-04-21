// test_diff_cuda.zig: CUDA 微分ランタイムテストのエントリ（Linux + -Dcuda=true）
//
// build.zig の test-diff-cuda から使う。このファイルの root が src/ になることで、
// src/diff/*_test.zig が src/ 配下の他ファイル（compute.zig など）を @import できる。
test {
    _ = @import("diff/cuda_runtime_test.zig");
}
