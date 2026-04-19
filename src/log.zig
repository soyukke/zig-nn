//! 共通 logger 基盤。
//!
//! 利用側 (executable / test root) は次を 1 行入れるだけで有効化できる:
//!     pub const std_options = @import("nn").log.std_options;
//!
//! - レベルは comptime 既定 `.info`。env `NN_LOG_LEVEL=err|warn|info|debug` で
//!   runtime に絞ることができる (debug を許可するには ReleaseFast 以外で起動)。
//! - scoped helper は feature 単位で `nn_log.metal.info(...)` のように使う。
//! - フォーマットは `[level][scope] msg` で統一する。

const std = @import("std");

pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = nnLogFn,
};

/// Test root などで level を絞った std_options を組み立てたいときに使う。
/// 例: `pub const std_options = @import("log.zig").stdOptionsAtLevel(.warn);`
pub fn stdOptionsAtLevel(comptime level: std.log.Level) std.Options {
    return .{ .log_level = level, .logFn = nnLogFn };
}

pub const default = std.log.scoped(.nn);
pub const cpu = std.log.scoped(.cpu);
pub const metal = std.log.scoped(.metal);
pub const cuda = std.log.scoped(.cuda);
pub const gguf = std.log.scoped(.gguf);
pub const gemma3 = std.log.scoped(.gemma3);
pub const trainer = std.log.scoped(.trainer);
pub const example = std.log.scoped(.example);

var initialized: bool = false;
var runtime_level: std.log.Level = .info;

fn parseLevel(s: []const u8) ?std.log.Level {
    if (std.ascii.eqlIgnoreCase(s, "err")) return .err;
    if (std.ascii.eqlIgnoreCase(s, "error")) return .err;
    if (std.ascii.eqlIgnoreCase(s, "warn")) return .warn;
    if (std.ascii.eqlIgnoreCase(s, "warning")) return .warn;
    if (std.ascii.eqlIgnoreCase(s, "info")) return .info;
    if (std.ascii.eqlIgnoreCase(s, "debug")) return .debug;
    return null;
}

fn ensureInit() void {
    if (@atomicLoad(bool, &initialized, .acquire)) return;
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "NN_LOG_LEVEL")) |buf| {
        defer std.heap.page_allocator.free(buf);
        if (parseLevel(buf)) |lv| runtime_level = lv;
    } else |_| {}
    @atomicStore(bool, &initialized, true, .release);
}

fn levelStr(comptime level: std.log.Level) []const u8 {
    return switch (level) {
        .err => "err ",
        .warn => "warn",
        .info => "info",
        .debug => "dbg ",
    };
}

fn nnLogFn(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    ensureInit();
    if (@intFromEnum(level) > @intFromEnum(runtime_level)) return;

    const lvl = comptime levelStr(level);
    const scp = comptime if (scope == .default) "nn" else @tagName(scope);
    std.debug.print("[" ++ lvl ++ "][" ++ scp ++ "] " ++ format ++ "\n", args);
}
