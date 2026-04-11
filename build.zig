const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("nn", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // CUDA GPU backend (Linux only)
    if (target.result.os.tag == .linux) {
        mod.linkSystemLibrary("cuda", .{});
        mod.linkSystemLibrary("cublas", .{});
    }

    // Metal GPU backend (macOS only)
    if (target.result.os.tag == .macos) {
        const fw_path: std.Build.LazyPath = .{ .cwd_relative = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks" };
        mod.addFrameworkPath(fw_path);
        mod.linkFramework("Metal", .{});
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Accelerate", .{});
        mod.linkFramework("MetalPerformanceShaders", .{});
        mod.linkFramework("MetalPerformanceShadersGraph", .{});
    }

    // Library tests
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);

    // Standalone test for diff_cpu_runtime only
    const diff_cpu_mod = b.addModule("diff_cpu_test", .{
        .root_source_file = b.path("src/diff_cpu_runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    if (target.result.os.tag == .macos) {
        const fw = std.Build.LazyPath{ .cwd_relative = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks" };
        diff_cpu_mod.addFrameworkPath(fw);
        diff_cpu_mod.linkFramework("Accelerate", .{});
    }
    const diff_cpu_tests = b.addTest(.{ .root_module = diff_cpu_mod });
    const run_diff_cpu_tests = b.addRunArtifact(diff_cpu_tests);
    const diff_cpu_test_step = b.step("test-diff-cpu", "Run diff_cpu_runtime tests only");
    diff_cpu_test_step.dependOn(&run_diff_cpu_tests.step);

    // Examples
    const examples = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "xor", .path = "examples/xor/main.zig" },
        .{ .name = "spiral", .path = "examples/spiral/main.zig" },
        .{ .name = "diffusion", .path = "examples/diffusion/main.zig" },
        .{ .name = "charlm", .path = "examples/charlm/main.zig" },
        .{ .name = "gpt2", .path = "examples/gpt2/main.zig" },
        .{ .name = "gemma3", .path = "examples/gemma3/main.zig" },

    };
    for (examples) |ex| {
        const exe = b.addExecutable(.{
            .name = ex.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(ex.path),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "nn", .module = mod }},
            }),
        });
        const run = b.addRunArtifact(exe);
        if (b.args) |args| run.addArgs(args);
        const step = b.step(ex.name, b.fmt("Run {s} example", .{ex.name}));
        step.dependOn(&run.step);
    }
}
