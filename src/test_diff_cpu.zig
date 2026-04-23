// test_diff_cpu.zig: CPU 微分ランタイムテストのエントリ
//
// build.zig の test-diff-cpu から使う。このファイルの root が src/ になることで、
// src/diff/*_test.zig が src/ 配下の他ファイル（compute.zig など）を @import できる。
//
// 注: trainer.zig / data/dataloader.zig のファイル内 test ブロックは root.zig 経由の
// refAllDecls では拾われない (struct decl のみ再帰するため)。
// seed 固定の網羅テストは tests/seed_test.zig に隔離し、ここから明示的に import する。
test {
    _ = @import("diff/cpu_runtime_test.zig");
    _ = @import("tests/seed_test.zig");
    _ = @import("data/dataloader.zig");
}
