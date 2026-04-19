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
pub const gradcheck = std.log.scoped(.gradcheck);

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

/// Profile dump 用の artifact ファイルハンドル。
/// `defer artifact.close()` で file と path を解放する。
pub const ProfileArtifact = struct {
    file: std.fs.File,
    path: []u8,
    allocator: std.mem.Allocator,

    pub fn close(self: *ProfileArtifact) void {
        self.file.close();
        self.allocator.free(self.path);
    }
};

fn isProfileEnabled() bool {
    const v = std.process.getEnvVarOwned(std.heap.page_allocator, "NN_PROFILE") catch return false;
    defer std.heap.page_allocator.free(v);
    if (v.len == 0) return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.ascii.eqlIgnoreCase(v, "false")) return false;
    return true;
}

/// Profile dump を artifact ファイルに書き出すための writer を開く。
/// `NN_PROFILE` env が立っていない場合は null を返す (no-op gate)。
/// 出力先: `NN_PROFILE_DIR` env か、既定 `zig-out/profiles`。
/// ファイル名: `<prefix>-<unix_ts>.txt`。
pub fn openProfileArtifact(prefix: []const u8) !?ProfileArtifact {
    if (!isProfileEnabled()) return null;
    const allocator = std.heap.page_allocator;

    const dir_owned: ?[]u8 = std.process.getEnvVarOwned(allocator, "NN_PROFILE_DIR") catch null;
    defer if (dir_owned) |d| allocator.free(d);
    const dir: []const u8 = if (dir_owned) |d| d else "zig-out/profiles";

    std.fs.cwd().makePath(dir) catch {};

    const ts = std.time.timestamp();
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}-{d}.txt", .{ dir, prefix, ts });
    errdefer allocator.free(path);

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    return .{ .file = file, .path = path, .allocator = allocator };
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
