// test_diff_cpu.zig: CPU 微分ランタイムテストのエントリ
//
// build.zig の test-diff-cpu から使う。このファイルの root が src/ になることで、
// src/diff/*_test.zig が src/ 配下の他ファイル（compute.zig など）を @import できる。
test {
    _ = @import("diff/cpu_runtime_test.zig");
}
