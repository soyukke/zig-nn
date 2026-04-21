// test_diff_mps.zig: MPS 微分ランタイムテストのエントリ（macOS）
//
// build.zig の test-diff-mps から使う。このファイルの root が src/ になることで、
// src/diff/*_test.zig が src/ 配下の他ファイル（compute.zig など）を @import できる。
test {
    _ = @import("diff/mps_runtime_test.zig");
}
