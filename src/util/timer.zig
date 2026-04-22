const std = @import("std");
const builtin = @import("builtin");

// Zig 0.16 で std.time.Timer が削除されたため、同等 API を提供する互換シム。
pub const Timer = struct {
    start_ns: u64,

    pub fn start() error{TimerUnsupported}!Timer {
        return .{ .start_ns = now_nanos() };
    }

    pub fn read(self: *Timer) u64 {
        return now_nanos() -% self.start_ns;
    }

    pub fn lap(self: *Timer) u64 {
        const now = now_nanos();
        const elapsed = now -% self.start_ns;
        self.start_ns = now;
        return elapsed;
    }

    pub fn reset(self: *Timer) void {
        self.start_ns = now_nanos();
    }
};

pub fn now_nanos() u64 {
    if (builtin.os.tag == .windows) {
        return 0;
    }
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    const sec: u64 = @intCast(ts.sec);
    const nsec: u64 = @intCast(ts.nsec);
    return sec * std.time.ns_per_s + nsec;
}
