/// Conv2d / MaxPool2d レイヤー (DiffCpuRuntime 用)
///
/// Conv2d: im2col ベースの畳み込み。weight [out_ch, in_ch*k*k], bias [out_ch]
/// MaxPool2d: パラメータなしの max pooling。
///
/// forward(self, ctx, input) で呼べる。h, w は comptime パラメータ。
const std = @import("std");
const compute = @import("../compute.zig");

pub fn Conv2d(
    comptime in_ch: usize,
    comptime out_ch: usize,
    comptime kernel_size: usize,
    comptime stride: usize,
    comptime padding: usize,
) type {
    return struct {
        w: compute.ParamHandle,
        b: compute.ParamHandle,
        input_h: usize = 0,
        input_w: usize = 0,

        pub fn init(module: anytype) @This() {
            return .{
                .w = module.addParam(&.{ out_ch, in_ch * kernel_size * kernel_size }, .xavier),
                .b = module.addParam(&.{out_ch}, .zeros),
            };
        }

        /// 入力の空間サイズを設定 (forward の前に呼ぶ)
        pub fn setInputSize(self: *@This(), h: usize, input_w: usize) void {
            self.input_h = h;
            self.input_w = input_w;
        }

        /// Sequential 対応の forward(self, ctx, input)。事前に setInputSize() が必要。
        pub fn forward(self: @This(), ctx: anytype, input: anytype) @TypeOf(input) {
            std.debug.assert(self.input_h > 0 and self.input_w > 0);
            return ctx.conv2d(input, ctx.param(self.w), ctx.param(self.b), stride, padding, kernel_size, in_ch, out_ch, self.input_h, self.input_w);
        }

        /// 明示的 h, w 指定の forward (backward compatibility)
        pub fn forwardWithSize(self: @This(), ctx: anytype, input: anytype, h: usize, input_w: usize) @TypeOf(input) {
            return ctx.conv2d(input, ctx.param(self.w), ctx.param(self.b), stride, padding, kernel_size, in_ch, out_ch, h, input_w);
        }

        /// 出力の空間サイズを計算
        pub fn outputSize(h: usize, input_w: usize) struct { oh: usize, ow: usize } {
            return .{
                .oh = (h + 2 * padding - kernel_size) / stride + 1,
                .ow = (input_w + 2 * padding - kernel_size) / stride + 1,
            };
        }
    };
}

pub fn MaxPool2d(comptime pool_size: usize, comptime stride: usize) type {
    return struct {
        channels: usize = 0,
        input_h: usize = 0,
        input_w: usize = 0,

        pub fn init(_: anytype) @This() {
            return .{};
        }

        /// 入力パラメータを設定 (forward の前に呼ぶ)
        pub fn setInputSize(self: *@This(), channels: usize, h: usize, input_w: usize) void {
            self.channels = channels;
            self.input_h = h;
            self.input_w = input_w;
        }

        /// Sequential 対応の forward(self, ctx, input)。事前に setInputSize() が必要。
        pub fn forward(self: @This(), ctx: anytype, input: anytype) @TypeOf(input) {
            std.debug.assert(self.channels > 0 and self.input_h > 0 and self.input_w > 0);
            return ctx.maxPool2d(input, pool_size, stride, self.channels, self.input_h, self.input_w);
        }

        /// 明示的パラメータ指定の forward (backward compatibility)
        pub fn forwardWithSize(self: @This(), ctx: anytype, input: anytype, channels: usize, h: usize, input_w: usize) @TypeOf(input) {
            _ = self;
            return ctx.maxPool2d(input, pool_size, stride, channels, h, input_w);
        }

        pub fn outputSize(h: usize, input_w: usize) struct { oh: usize, ow: usize } {
            return .{
                .oh = (h - pool_size) / stride + 1,
                .ow = (input_w - pool_size) / stride + 1,
            };
        }
    };
}
