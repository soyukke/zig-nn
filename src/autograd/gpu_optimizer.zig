/// GPU Adam / AdamW オプティマイザ
///
/// Fused adam_step カーネルで m/v 更新 + bias correction + weight 更新を
/// 1 dispatch で実行。全パラメータの勾配バッファも GPU 上にある。
const std = @import("std");
const Allocator = std.mem.Allocator;
const metal = @import("../backend/metal.zig");

const MetalContext = metal.MetalContext;
const id = metal.id;

pub fn GpuAdam(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const GpuParam = struct {
            data: []T,        // UMA ポインタ (MTLBuffer contents)
            data_buf: id,     // MTLBuffer for weights
            grad: *?[]T,      // 勾配 (UMA ポインタ)
            grad_buf: ?id,    // MTLBuffer for grads
            count: usize,
        };

        params: []const GpuParam,
        lr: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        t: usize,
        // m (first moment) and v (second moment) MTL buffers
        m_bufs: []id,
        v_bufs: []id,
        metal_ctx: *MetalContext,
        allocator: Allocator,

        pub fn init(
            params: []const GpuParam,
            mtl: *MetalContext,
            allocator: Allocator,
            lr: T,
            beta1: T,
            beta2: T,
            epsilon: T,
            weight_decay: T,
        ) !Self {
            const m_bufs = try allocator.alloc(id, params.len);
            const v_bufs = try allocator.alloc(id, params.len);

            for (params, m_bufs, v_bufs) |p, *mb, *vb| {
                const buf_size = p.count * @sizeOf(T);
                mb.* = try mtl.createBuffer(buf_size);
                vb.* = try mtl.createBuffer(buf_size);
                // ゼロ初期化 (UMA)
                const m_ptr = MetalContext.bufferContents(T, mb.*);
                @memset(m_ptr[0..p.count], 0);
                const v_ptr = MetalContext.bufferContents(T, vb.*);
                @memset(v_ptr[0..p.count], 0);
            }

            return .{
                .params = params,
                .lr = lr,
                .beta1 = beta1,
                .beta2 = beta2,
                .epsilon = epsilon,
                .weight_decay = weight_decay,
                .t = 0,
                .m_bufs = m_bufs,
                .v_bufs = v_bufs,
                .metal_ctx = mtl,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.m_bufs) |mb| metal.objRelease(mb);
            for (self.v_bufs) |vb| metal.objRelease(vb);
            self.allocator.free(self.m_bufs);
            self.allocator.free(self.v_bufs);
        }

        pub fn step(self: *Self) void {
            self.t += 1;
            const t_f: T = @floatFromInt(self.t);
            const bc1 = 1.0 - std.math.pow(T, self.beta1, t_f);
            const bc2 = 1.0 - std.math.pow(T, self.beta2, t_f);

            // 全パラメータの Adam step を 1 コマンドバッファに統合
            const cmd_buf = self.metal_ctx.newCommandBuffer();
            const encoder = MetalContext.newComputeEncoder(cmd_buf);

            for (self.params, self.m_bufs, self.v_bufs) |p, m_buf, v_buf| {
                const g = p.grad.* orelse continue;
                _ = g;
                const grad_buf = p.grad_buf orelse continue;

                self.metal_ctx.dispatchAdamStep(
                    encoder,
                    p.data_buf,
                    grad_buf,
                    m_buf,
                    v_buf,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    self.weight_decay,
                    bc1,
                    bc2,
                    @intCast(p.count),
                );
                MetalContext.memoryBarrier(encoder);
            }

            MetalContext.endEncoding(encoder);
            MetalContext.commit(cmd_buf);
            MetalContext.waitUntilCompleted(cmd_buf);
        }

        pub fn zeroGrad(self: *Self) void {
            for (self.params) |p| {
                if (p.grad.*) |g| {
                    @memset(g, 0);
                }
            }
        }
    };
}
