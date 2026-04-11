const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const nn_mod = b.addModule("nn", .{
        .root_source_file = b.path("../../src/root.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "gpt2",
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = nn_mod },
            },
        }),
    });

    if (target.result.os.tag == .macos) {
        const fw_path: std.Build.LazyPath = .{ .cwd_relative = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks" };
        exe.root_module.addFrameworkPath(fw_path);
        nn_mod.addFrameworkPath(fw_path);
        exe.root_module.linkFramework("Metal", .{});
        exe.root_module.linkFramework("Foundation", .{});
        exe.root_module.linkFramework("Accelerate", .{});
        exe.root_module.linkFramework("MetalPerformanceShaders", .{});
        exe.root_module.linkFramework("MetalPerformanceShadersGraph", .{});
        nn_mod.linkFramework("Metal", .{});
        nn_mod.linkFramework("Foundation", .{});
        nn_mod.linkFramework("Accelerate", .{});
        nn_mod.linkFramework("MetalPerformanceShaders", .{});
        nn_mod.linkFramework("MetalPerformanceShadersGraph", .{});
    }

    b.installArtifact(exe);

    const run_step = b.step("run", "Run GPT-2 demo");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}
