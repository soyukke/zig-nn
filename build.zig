const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("nn", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const enable_cuda = b.option(bool, "cuda", "Enable CUDA GPU backend (requires CUDA toolkit)") orelse false;

    const openblas_path = b.graph.environ_map.get("OPENBLAS_PATH");
    const cuda_path = b.graph.environ_map.get("CUDA_PATH");
    const cublas_path = b.graph.environ_map.get("CUBLAS_PATH");

    // macOS SDK framework path (Nix apple-sdk の SDKROOT を優先し、無ければ Xcode を利用)
    const macos_fw_path: ?[]const u8 = blk: {
        if (b.graph.environ_map.get("SDKROOT")) |sdkroot| {
            break :blk b.fmt("{s}/System/Library/Frameworks", .{sdkroot});
        }
        break :blk "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks";
    };

    const addMacosFrameworkPath = struct {
        fn apply(m: *std.Build.Module, p: ?[]const u8) void {
            if (p) |path| m.addFrameworkPath(.{ .cwd_relative = path });
        }
    }.apply;

    // Linux backend dependencies
    if (target.result.os.tag == .linux) {
        mod.linkSystemLibrary("c", .{});

        // OpenBLAS (CBLAS) for CPU backend
        if (openblas_path) |p| {
            mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{p}) });
            mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{p}) });
        }
        mod.linkSystemLibrary("openblas", .{});

        // CUDA GPU backend (opt-in via -Dcuda=true)
        if (enable_cuda) {
            // WSL2 driver library path (libcuda.so)
            mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/wsl/lib" });

            if (cuda_path) |p| {
                mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{p}) });
                mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{p}) });
            }

            if (cublas_path) |p| {
                mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{p}) });
                mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{p}) });
            }

            mod.linkSystemLibrary("cuda", .{});
            mod.linkSystemLibrary("cublas", .{});
        }
    }

    // Metal GPU backend (macOS only)
    if (target.result.os.tag == .macos) {
        addMacosFrameworkPath(mod, macos_fw_path);
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
        .root_source_file = b.path("src/diff_cpu_runtime_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    if (target.result.os.tag == .linux) {
        diff_cpu_mod.linkSystemLibrary("c", .{});
        if (openblas_path) |p| {
            diff_cpu_mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{p}) });
            diff_cpu_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{p}) });
        }
        diff_cpu_mod.linkSystemLibrary("openblas", .{});
    }
    if (target.result.os.tag == .macos) {
        addMacosFrameworkPath(diff_cpu_mod, macos_fw_path);
        diff_cpu_mod.linkFramework("Accelerate", .{});
    }
    const diff_cpu_tests = b.addTest(.{ .root_module = diff_cpu_mod });
    const run_diff_cpu_tests = b.addRunArtifact(diff_cpu_tests);
    const diff_cpu_test_step = b.step("test-diff-cpu", "Run diff_cpu_runtime tests only");
    diff_cpu_test_step.dependOn(&run_diff_cpu_tests.step);

    // Standalone test for diff_mps_runtime (macOS only)
    if (target.result.os.tag == .macos) {
        const diff_mps_mod = b.addModule("diff_mps_test", .{
            .root_source_file = b.path("src/diff_mps_runtime_test.zig"),
            .target = target,
            .optimize = optimize,
        });
        addMacosFrameworkPath(diff_mps_mod, macos_fw_path);
        diff_mps_mod.linkFramework("Metal", .{});
        diff_mps_mod.linkFramework("Foundation", .{});
        diff_mps_mod.linkFramework("Accelerate", .{});
        diff_mps_mod.linkFramework("MetalPerformanceShaders", .{});
        diff_mps_mod.linkFramework("MetalPerformanceShadersGraph", .{});
        const diff_mps_tests = b.addTest(.{ .root_module = diff_mps_mod });
        const run_diff_mps_tests = b.addRunArtifact(diff_mps_tests);
        const diff_mps_test_step = b.step("test-diff-mps", "Run diff_mps_runtime tests only (macOS Metal)");
        diff_mps_test_step.dependOn(&run_diff_mps_tests.step);
    }

    // Standalone test for diff_cuda_runtime (requires -Dcuda=true)
    if (enable_cuda and target.result.os.tag == .linux) {
        const diff_cuda_mod = b.addModule("diff_cuda_test", .{
            .root_source_file = b.path("src/diff_cuda_runtime_test.zig"),
            .target = target,
            .optimize = optimize,
        });
        diff_cuda_mod.linkSystemLibrary("c", .{});
        if (openblas_path) |p| {
            diff_cuda_mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{p}) });
            diff_cuda_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{p}) });
        }
        diff_cuda_mod.linkSystemLibrary("openblas", .{});
        diff_cuda_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/wsl/lib" });
        if (cuda_path) |p| {
            diff_cuda_mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{p}) });
            diff_cuda_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{p}) });
        }
        if (cublas_path) |p| {
            diff_cuda_mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{p}) });
            diff_cuda_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{p}) });
        }
        diff_cuda_mod.linkSystemLibrary("cuda", .{});
        diff_cuda_mod.linkSystemLibrary("cublas", .{});
        const diff_cuda_tests = b.addTest(.{ .root_module = diff_cuda_mod });
        const run_diff_cuda_tests = b.addRunArtifact(diff_cuda_tests);
        const diff_cuda_test_step = b.step("test-diff-cuda", "Run diff_cuda_runtime tests only (requires GPU)");
        diff_cuda_test_step.dependOn(&run_diff_cuda_tests.step);
    }

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
