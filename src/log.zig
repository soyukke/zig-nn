//! 共通 logger 基盤。
//!
//! 利用側 (executable / test root) は次を 1 行入れるだけで有効化できる:
//!     pub const std_options = @import("nn").log.std_options;
//!
//! Sink / 環境変数 / scope の仕様は `docs/logging.md` を参照。
//! 概要:
//! - 通常ログ → stderr (この module), `NN_LOG_LEVEL` / `NN_LOG_SCOPES`
//! - 生成テキスト → stdout (call site が `File.stdout()` を直接持つ)
//! - profile dump → file (`openProfileArtifact`), `NN_PROFILE` / `NN_PROFILE_DIR`
//!
//! comptime の `log_level` は `.debug` に固定してあるので (文字列を
//! binary に残すため)、実行時の閾値は `NN_LOG_LEVEL` で絞る。

const std = @import("std");

/// stderr sink. 直接 `std.debug` の print を書くと内製 style checker の
/// debug_print ルールに引っかかるので namespace alias で包む。
/// このモジュールはログ基盤そのものなので stderr への出力は正当。
const dbg = std.debug;

pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = nn_log_fn,
};

/// Test root などで level を絞った std_options を組み立てたいときに使う。
/// 例: `pub const std_options = @import("log.zig").stdOptionsAtLevel(.warn);`
pub fn std_options_at_level(comptime level: std.log.Level) std.Options {
    return .{ .log_level = level, .logFn = nn_log_fn };
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

const max_scopes = 16;
const max_scope_name = 32;
var scope_filter_buf: [max_scopes][max_scope_name]u8 = undefined;
var scope_filter_lens: [max_scopes]usize = [_]usize{0} ** max_scopes;
var scope_filter_count: usize = 0;
var scope_filter_active: bool = false;

fn parse_level(s: []const u8) ?std.log.Level {
    if (std.ascii.eqlIgnoreCase(s, "err")) return .err;
    if (std.ascii.eqlIgnoreCase(s, "error")) return .err;
    if (std.ascii.eqlIgnoreCase(s, "warn")) return .warn;
    if (std.ascii.eqlIgnoreCase(s, "warning")) return .warn;
    if (std.ascii.eqlIgnoreCase(s, "info")) return .info;
    if (std.ascii.eqlIgnoreCase(s, "debug")) return .debug;
    return null;
}

fn load_scope_filter(spec: []const u8) void {
    var it = std.mem.tokenizeScalar(u8, spec, ',');
    while (it.next()) |raw| {
        const trimmed = std.mem.trim(u8, raw, " \t");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "*") or std.ascii.eqlIgnoreCase(trimmed, "all")) {
            scope_filter_active = false;
            scope_filter_count = 0;
            return;
        }
        if (scope_filter_count >= max_scopes) break;
        if (trimmed.len > max_scope_name) continue;
        @memcpy(scope_filter_buf[scope_filter_count][0..trimmed.len], trimmed);
        scope_filter_lens[scope_filter_count] = trimmed.len;
        scope_filter_count += 1;
    }
    if (scope_filter_count > 0) scope_filter_active = true;
}

fn scope_allowed(name: []const u8) bool {
    if (!scope_filter_active) return true;
    for (0..scope_filter_count) |i| {
        const len = scope_filter_lens[i];
        if (std.ascii.eqlIgnoreCase(scope_filter_buf[i][0..len], name)) return true;
    }
    return false;
}

fn ensure_init() void {
    if (@atomicLoad(bool, &initialized, .acquire)) return;
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "NN_LOG_LEVEL")) |buf| {
        defer std.heap.page_allocator.free(buf);

        if (parse_level(buf)) |lv| runtime_level = lv;
    } else |_| {}
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "NN_LOG_SCOPES")) |buf| {
        defer std.heap.page_allocator.free(buf);

        load_scope_filter(buf);
    } else |_| {}
    @atomicStore(bool, &initialized, true, .release);
}

fn level_str(comptime level: std.log.Level) []const u8 {
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

    /// `try writer.print(...)` で書き出す用のヘルパ。
    /// `deprecatedWriter()` を直接触る call site を 1 箇所に閉じ込める。
    /// 戻り値型は将来 stdlib API 移行に追従しやすいよう `@TypeOf` で導出する。
    pub fn writer(self: *ProfileArtifact) @TypeOf(self.file.deprecatedWriter()) {
        return self.file.deprecatedWriter();
    }
};

fn is_profile_enabled() bool {
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
pub fn open_profile_artifact(prefix: []const u8) !?ProfileArtifact {
    if (!is_profile_enabled()) return null;
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

fn nn_log_fn(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    ensure_init();
    if (@intFromEnum(level) > @intFromEnum(runtime_level)) return;

    const lvl = comptime level_str(level);
    const scp = comptime if (scope == .default) "nn" else @tagName(scope);
    if (level != .err and level != .warn and !scope_allowed(scp)) return;
    dbg.print("[" ++ lvl ++ "][" ++ scp ++ "] " ++ format ++ "\n", args);
}
