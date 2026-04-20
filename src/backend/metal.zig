/// Metal GPU バックエンド
///
/// Objective-C ランタイム経由で Metal API を呼び出す。
/// Apple Silicon の Unified Memory Architecture (UMA) を活用し、
/// MTLResourceStorageModeShared でゼロコピーバッファ共有を実現。
const std = @import("std");
const Allocator = std.mem.Allocator;
const Timer = @import("../util/timer.zig").Timer;
const log_mod = @import("../log.zig");
const log = log_mod.metal;

// Objective-C ランタイム
const objc = @cImport({
    @cInclude("objc/runtime.h");
    @cInclude("objc/message.h");
});

// Metal C 関数
extern "c" fn MTLCreateSystemDefaultDevice() ?*anyopaque;

// ============================================================
// Objective-C ランタイムヘルパー
// ============================================================

pub const id = *anyopaque;
const SEL = *anyopaque;
const NSUInteger = u64;

const MTLSize = extern struct {
    width: NSUInteger,
    height: NSUInteger,
    depth: NSUInteger,
};

fn sel(name: [*:0]const u8) SEL {
    return objc.sel_registerName(name).?;
}

fn getClass(name: [*:0]const u8) id {
    return objc.objc_getClass(name).?;
}

// ============================================================
// 明示的な関数ポインタ型 (可変引数を使わない)
// ============================================================

const objc_msgSend_ptr = @extern(*const anyopaque, .{ .name = "objc_msgSend" });

// 引数0個 → id
fn send0(target: id, selector: SEL) ?*anyopaque {
    const F = *const fn (id, SEL) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector);
}

// 引数0個 → void
fn send0v(target: id, selector: SEL) void {
    const F = *const fn (id, SEL) callconv(.c) void;
    @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector);
}

// 引数0個 → NSUInteger
fn send0u(target: id, selector: SEL) NSUInteger {
    const F = *const fn (id, SEL) callconv(.c) NSUInteger;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector);
}

// 引数1個 (id) → id
fn send1(target: id, selector: SEL, a1: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, SEL, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1);
}

// 引数1個 (cstr) → id   NSString stringWithUTF8String:
fn send1s(target: id, selector: SEL, a1: [*]const u8) ?*anyopaque {
    const F = *const fn (id, SEL, [*]const u8) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1);
}

// 引数1個 → [*:0]const u8  (UTF8String)
fn send0str(target: id, selector: SEL) ?[*:0]const u8 {
    const F = *const fn (id, SEL) callconv(.c) ?[*:0]const u8;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector);
}

// 引数1個 (id) → void
fn send1v(target: id, selector: SEL, a1: ?*anyopaque) void {
    const F = *const fn (id, SEL, ?*anyopaque) callconv(.c) void;
    @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1);
}

// 引数2個 (u64, u64) → id  newBufferWithLength:options:
fn send2uu(target: id, selector: SEL, a1: NSUInteger, a2: NSUInteger) ?*anyopaque {
    const F = *const fn (id, SEL, NSUInteger, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2);
}

// 引数2個 (id, *err) → id  newComputePipelineStateWithFunction:error:
fn send2ie(target: id, selector: SEL, a1: ?*anyopaque, a2: *?*anyopaque) ?*anyopaque {
    const F = *const fn (id, SEL, ?*anyopaque, *?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2);
}

// 引数3個 (ptr, u64, u64) → id  newBufferWithBytes:length:options:
fn send3puu(target: id, selector: SEL, a1: *const anyopaque, a2: NSUInteger, a3: NSUInteger) ?*anyopaque {
    const F = *const fn (id, SEL, *const anyopaque, NSUInteger, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数3個 (id, id, *err) → id  newLibraryWithSource:options:error:
fn send3iie(target: id, selector: SEL, a1: ?*anyopaque, a2: ?*anyopaque, a3: *?*anyopaque) ?*anyopaque {
    const F = *const fn (id, SEL, ?*anyopaque, ?*anyopaque, *?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数3個 (id, u64, u64) → void  setBuffer:offset:atIndex:
fn send3iuuv(target: id, selector: SEL, a1: ?*anyopaque, a2: NSUInteger, a3: NSUInteger) void {
    const F = *const fn (id, SEL, ?*anyopaque, NSUInteger, NSUInteger) callconv(.c) void;
    @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数3個 (ptr, u64, u64) → void  setBytes:length:atIndex:
fn send3puuv(target: id, selector: SEL, a1: *const anyopaque, a2: NSUInteger, a3: NSUInteger) void {
    const F = *const fn (id, SEL, *const anyopaque, NSUInteger, NSUInteger) callconv(.c) void;
    @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数2個 (MTLSize, MTLSize) → void  dispatchThreads:threadsPerThreadgroup:
fn send2ssv(target: id, selector: SEL, a1: MTLSize, a2: MTLSize) void {
    const F = *const fn (id, SEL, MTLSize, MTLSize) callconv(.c) void;
    @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2);
}

pub fn objRelease(target: id) void {
    send0v(target, sel("release"));
}

fn nsString(str: [*]const u8) id {
    const NSString = getClass("NSString");
    return send1s(NSString, sel("stringWithUTF8String:"), str).?;
}

// ============================================================
// MetalContext
// ============================================================

pub const MetalContext = struct {
    device: id,
    command_queue: id,
    library: id,
    pipelines: Pipelines,
    training_pipelines: ?TrainingPipelines,
    training_library: ?id,
    // Batch mode: forward/optimizer dispatches を 1 コマンドバッファに統合
    batch_cmd_buf: ?id = null,
    batch_encoder: ?id = null,
    // Backward batch: grad を MTLBuffer で管理
    backward_grad_state: ?BackwardGradState = null,
    // MPS MatrixMultiplication キャッシュ
    mps_cache: ?MPSCache = null,
    // GPU プロファイルモード: カテゴリが変わった時だけ flush して計測
    profile_mode: bool = false,
    profile_stats: ProfileStats = .{},
    profile_current_cat: u8 = 255, // 現在のカテゴリ (255=none)
    profile_timer: ?Timer = null, // カテゴリ開始タイマー

    pub const ProfileStats = struct {
        mps_matmul_ns: u64 = 0,
        batched_matmul_ns: u64 = 0,
        layernorm_ns: u64 = 0,
        softmax_ns: u64 = 0,
        gelu_tanh_ns: u64 = 0,
        embedding_ns: u64 = 0,
        add_bias_ns: u64 = 0,
        scale_ns: u64 = 0,
        concat_ns: u64 = 0,
        loss_ns: u64 = 0,
        other_ns: u64 = 0,
        mps_matmul_count: u32 = 0,
        batched_matmul_count: u32 = 0,
        layernorm_count: u32 = 0,
        softmax_count: u32 = 0,
        gelu_tanh_count: u32 = 0,
        embedding_count: u32 = 0,
        add_bias_count: u32 = 0,
        scale_count: u32 = 0,
        concat_count: u32 = 0,
        loss_count: u32 = 0,
        other_count: u32 = 0,

        pub fn reset(self: *ProfileStats) void {
            self.* = .{};
        }

        pub fn totalNs(self: ProfileStats) u64 {
            return self.mps_matmul_ns + self.batched_matmul_ns + self.layernorm_ns +
                self.softmax_ns + self.gelu_tanh_ns + self.embedding_ns +
                self.add_bias_ns + self.scale_ns + self.concat_ns + self.loss_ns + self.other_ns;
        }

        pub fn print(self: ProfileStats) void {
            var maybe_artifact = log_mod.openProfileArtifact("metal-gpu") catch null;
            if (maybe_artifact) |*artifact| {
                defer artifact.close();
                const w = artifact.writer();
                const total = self.totalNs();
                const ms = std.time.ns_per_ms;
                w.print("[GPU profile] total={d}ms\n", .{total / ms}) catch return;
                self.writeLine(w, "mps_matmul", self.mps_matmul_ns, self.mps_matmul_count, total) catch return;
                self.writeLine(w, "batched_matmul", self.batched_matmul_ns, self.batched_matmul_count, total) catch return;
                self.writeLine(w, "layernorm", self.layernorm_ns, self.layernorm_count, total) catch return;
                self.writeLine(w, "softmax", self.softmax_ns, self.softmax_count, total) catch return;
                self.writeLine(w, "gelu_tanh", self.gelu_tanh_ns, self.gelu_tanh_count, total) catch return;
                self.writeLine(w, "embedding", self.embedding_ns, self.embedding_count, total) catch return;
                self.writeLine(w, "add_bias", self.add_bias_ns, self.add_bias_count, total) catch return;
                self.writeLine(w, "scale", self.scale_ns, self.scale_count, total) catch return;
                self.writeLine(w, "concat", self.concat_ns, self.concat_count, total) catch return;
                self.writeLine(w, "loss", self.loss_ns, self.loss_count, total) catch return;
                self.writeLine(w, "other", self.other_ns, self.other_count, total) catch return;
                log.info("profile written: {s}", .{artifact.path});
            }
        }

        fn writeLine(_: ProfileStats, w: anytype, name: []const u8, ns: u64, count: u32, total: u64) !void {
            const pct = if (total > 0) (ns * 100) / total else 0;
            try w.print("  {s}: {d}ms ({d}%) [{d} calls]\n", .{ name, ns / std.time.ns_per_ms, pct, count });
        }

        /// GpuProfileCat enum value でカウント加算
        pub fn addCount(self: *ProfileStats, cat: anytype, n: u32) void {
            const idx = @intFromEnum(cat);
            switch (idx) {
                0 => self.batched_matmul_count += n,
                1 => self.layernorm_count += n,
                2 => self.softmax_count += n,
                3 => self.gelu_tanh_count += n,
                4 => self.embedding_count += n,
                5 => self.add_bias_count += n,
                6 => self.scale_count += n,
                7 => self.concat_count += n,
                8 => self.loss_count += n,
                else => self.other_count += n,
            }
        }

        /// GpuProfileCat enum value で時間加算
        pub fn addNs(self: *ProfileStats, cat: anytype, ns: u64) void {
            self.addNsByIdx(@intFromEnum(cat), ns);
        }

        /// u8 index で時間加算 (254 = mps_matmul)
        pub fn addNsByIdx(self: *ProfileStats, idx: u8, ns: u64) void {
            switch (idx) {
                0 => self.batched_matmul_ns += ns,
                1 => self.layernorm_ns += ns,
                2 => self.softmax_ns += ns,
                3 => self.gelu_tanh_ns += ns,
                4 => self.embedding_ns += ns,
                5 => self.add_bias_ns += ns,
                6 => self.scale_ns += ns,
                7 => self.concat_ns += ns,
                8 => self.loss_ns += ns,
                254 => self.mps_matmul_ns += ns,
                else => self.other_ns += ns,
            }
        }
    };

    /// プロファイルモード: 現在のカテゴリを flush (commit/wait)
    pub fn profileFlush(self: *MetalContext) void {
        if (self.profile_current_cat == 255) return;
        if (self.batch_encoder) |encoder| {
            memoryBarrier(encoder);
            endEncoding(encoder);
            commit(self.batch_cmd_buf.?);
            waitUntilCompleted(self.batch_cmd_buf.?);
            if (self.profile_timer) |*timer| {
                self.profile_stats.addNsByIdx(self.profile_current_cat, timer.read());
            }
            self.batch_cmd_buf = self.newCommandBuffer();
            self.batch_encoder = newComputeEncoder(self.batch_cmd_buf.?);
        }
        self.profile_current_cat = 255;
        self.profile_timer = null;
    }

    pub const MPSCacheKey = struct {
        result_rows: u32,
        result_cols: u32,
        interior_cols: u32,
        transpose_left: bool,
        transpose_right: bool,
        beta_is_one: bool,
    };

    pub const MPSCache = std.AutoHashMap(MPSCacheKey, id);

    pub fn initMPSCache(self: *MetalContext, allocator: Allocator) void {
        self.mps_cache = MPSCache.init(allocator);
    }

    pub fn deinitMPSCache(self: *MetalContext) void {
        if (self.mps_cache) |*cache| {
            var it = cache.valueIterator();
            while (it.next()) |v| {
                objRelease(v.*);
            }
            cache.deinit();
            self.mps_cache = null;
        }
    }

    pub const TrainingPipelines = struct {
        // Phase 1: DDPM (MLP)
        matmul_f32: id,
        add_f32: id,
        add_bias_f32: id,
        silu_forward: id,
        mse_loss_diff: id,
        mse_loss_reduce: id,
        matmul_f32_backward_a: id,
        matmul_f32_backward_b: id,
        add_backward_accum: id,
        add_bias_backward: id,
        silu_backward: id,
        mse_loss_backward: id,
        adam_step: id,
        zero_buffer: id,
        // Phase 2: Transformer
        relu_forward: id,
        relu_backward: id,
        gelu_forward: id,
        gelu_backward: id,
        softmax_f32: id,
        softmax_backward: id,
        causal_softmax_f32: id,
        layernorm_forward: id,
        layernorm_backward_x: id,
        layernorm_backward_params: id,
        cross_entropy_forward: id,
        cross_entropy_reduce: id,
        cross_entropy_backward: id,
        embedding_forward: id,
        embedding_backward: id,
        scale_f32: id,
        scale_backward: id,
        matmul_f32_trans_b: id,
        matmul_f32_accum: id,
        // Phase 3: QLoRA
        matmul_q4_0_trans_batched: id,
        matmul_q4_1_trans_batched: id,
        matmul_q8_0_trans_batched: id,
        rmsnorm_forward_training: id,
        rmsnorm_backward_x: id,
        rmsnorm_backward_weight: id,
        rope_forward_training: id,
        rope_backward: id,
        dequant_q8_0_batch_scaled: id,
        // Phase 4: Sequence ops
        tanh_forward: id,
        tanh_backward: id,
        concat_last_dim: id,
        concat_last_dim_backward: id,
        // Phase 5: Batched matmul for attention
        batched_matmul_f32: id,
        batched_matmul_trans_b_f32: id,
        batched_matmul_backward_a_f32: id,
        batched_matmul_backward_b_f32: id,
        batched_matmul_trans_b_backward_a_f32: id,
        batched_matmul_trans_b_backward_b_f32: id,
        // Phase 6: Fused kernels
        matmul_addbias_gelu_f32: id,
        gelu_bias_backward: id,
        matmul_addbias_tanh_f32: id,
        tanh_bias_backward: id,
        batched_matmul_trans_b_scale_f32: id,
    };

    const Pipelines = struct {
        matmul_q4_0: id,
        matmul_q4_1: id,
        matmul_q8_0: id,
        matmul_q4_0_batched: id,
        matmul_q4_1_batched: id,
        matmul_q8_0_batched: id,
        rmsnorm: id,
        rmsnorm_inplace: id,
        rmsnorm_residual: id,
        rope: id,
        gelu: id,
        gelu_mul: id,
        softmax: id,
        add_inplace: id,
        mul_inplace: id,
        scale_inplace: id,
        dequant_q8_0_row: id,
        dequant_q8_0_row_scaled: id,
        gqa_attention_decode: id,
        write_kv_cache: id,
    };

    pub fn init() !MetalContext {
        const device = MTLCreateSystemDefaultDevice() orelse return error.NoMetalDevice;

        // Command queue
        const queue = send0(device, sel("newCommandQueue")) orelse return error.MetalInitFailed;

        // MSL ソースをコンパイル
        const msl_source = @embedFile("shaders/nn_kernels.metal");
        const ns_source = nsString(msl_source.ptr);

        var err: ?*anyopaque = null;
        const library = send3iie(
            device,
            sel("newLibraryWithSource:options:error:"),
            ns_source,
            null,
            &err,
        ) orelse {
            if (err) |e| {
                const desc = send0(e, sel("localizedDescription"));
                if (desc) |d| {
                    const cstr = send0str(d, sel("UTF8String"));
                    if (cstr) |s| {
                        log.err("shader compilation error: {s}", .{s});
                    }
                }
            }
            return error.ShaderCompilationFailed;
        };

        var ctx = MetalContext{
            .device = device,
            .command_queue = queue,
            .library = library,
            .pipelines = undefined,
            .training_pipelines = null,
            .training_library = null,
        };

        // パイプライン作成
        ctx.pipelines = .{
            .matmul_q4_0 = try ctx.createPipeline("matmul_q4_0"),
            .matmul_q4_1 = try ctx.createPipeline("matmul_q4_1"),
            .matmul_q8_0 = try ctx.createPipeline("matmul_q8_0"),
            .matmul_q4_0_batched = try ctx.createPipeline("matmul_q4_0_batched"),
            .matmul_q4_1_batched = try ctx.createPipeline("matmul_q4_1_batched"),
            .matmul_q8_0_batched = try ctx.createPipeline("matmul_q8_0_batched"),
            .rmsnorm = try ctx.createPipeline("rmsnorm"),
            .rmsnorm_inplace = try ctx.createPipeline("rmsnorm_inplace"),
            .rmsnorm_residual = try ctx.createPipeline("rmsnorm_residual"),
            .rope = try ctx.createPipeline("rope"),
            .gelu = try ctx.createPipeline("gelu"),
            .gelu_mul = try ctx.createPipeline("gelu_mul"),
            .softmax = try ctx.createPipeline("softmax"),
            .add_inplace = try ctx.createPipeline("add_inplace"),
            .mul_inplace = try ctx.createPipeline("mul_inplace"),
            .scale_inplace = try ctx.createPipeline("scale_inplace"),
            .dequant_q8_0_row = try ctx.createPipeline("dequant_q8_0_row"),
            .dequant_q8_0_row_scaled = try ctx.createPipeline("dequant_q8_0_row_scaled"),
            .gqa_attention_decode = try ctx.createPipeline("gqa_attention_decode"),
            .write_kv_cache = try ctx.createPipeline("write_kv_cache"),
        };

        // デバイス名を表示
        const name = send0(device, sel("name"));
        if (name) |n| {
            const cstr = send0str(n, sel("UTF8String"));
            if (cstr) |s| {
                log.info("device: {s}", .{s});
            }
        }

        return ctx;
    }

    fn createPipeline(self: *MetalContext, name: [*:0]const u8) !id {
        const ns_name = nsString(name);
        const func = send1(self.library, sel("newFunctionWithName:"), ns_name) orelse {
            log.err("function '{s}' not found", .{name});
            return error.MetalFunctionNotFound;
        };
        defer objRelease(func);

        var err: ?*anyopaque = null;
        const pipeline = send2ie(
            self.device,
            sel("newComputePipelineStateWithFunction:error:"),
            func,
            &err,
        ) orelse {
            return error.MetalPipelineCreationFailed;
        };
        return pipeline;
    }

    pub fn deinit(self: *MetalContext) void {
        if (self.training_pipelines) |tp| {
            inline for (std.meta.fields(TrainingPipelines)) |field| {
                objRelease(@field(tp, field.name));
            }
        }
        if (self.training_library) |lib| {
            objRelease(lib);
        }
        inline for (std.meta.fields(Pipelines)) |field| {
            objRelease(@field(self.pipelines, field.name));
        }
        objRelease(self.library);
        objRelease(self.command_queue);
        objRelease(self.device);
    }

    // ============================================================
    // GPU メモリクエリ
    // ============================================================

    pub fn currentAllocatedSize(self: *MetalContext) usize {
        return @intCast(send0u(self.device, sel("currentAllocatedSize")));
    }

    // ============================================================
    // バッファ管理
    // ============================================================

    pub fn createBuffer(self: *MetalContext, size: usize) !id {
        const buf = send2uu(
            self.device,
            sel("newBufferWithLength:options:"),
            @as(NSUInteger, @intCast(size)),
            @as(NSUInteger, 0),
        ) orelse return error.MetalBufferCreationFailed;
        return buf;
    }

    pub fn createBufferWithData(self: *MetalContext, data: []const u8) !id {
        const buf = send3puu(
            self.device,
            sel("newBufferWithBytes:length:options:"),
            data.ptr,
            @as(NSUInteger, @intCast(data.len)),
            @as(NSUInteger, 0),
        ) orelse return error.MetalBufferCreationFailed;
        return buf;
    }

    pub fn bufferContents(comptime T: type, buf: id) [*]T {
        const ptr = send0(buf, sel("contents")).?;
        return @ptrCast(@alignCast(ptr));
    }

    // ============================================================
    // メモリバリア & フェンス
    // ============================================================

    // 引数1個 (NSUInteger) → void
    fn send1uv(target: id, selector: SEL, a1: NSUInteger) void {
        const F = *const fn (id, SEL, NSUInteger) callconv(.c) void;
        @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1);
    }

    pub fn memoryBarrier(encoder: id) void {
        // MTLBarrierScopeBuffers = 1 << 0 = 1
        send1uv(encoder, sel("memoryBarrierWithScope:"), 1);
    }

    // MTLFence: エンコーダ内で dispatch 間の順序保証
    // memoryBarrier よりも軽量な可能性がある (パイプライン全体をストールさせない)
    pub fn newFence(self: *MetalContext) id {
        return send0(self.device, sel("newFence")).?;
    }

    pub fn updateFence(encoder: id, fence: id) void {
        send1v(encoder, sel("updateFence:"), fence);
    }

    pub fn waitForFence(encoder: id, fence: id) void {
        send1v(encoder, sel("waitForFence:"), fence);
    }

    // ============================================================
    // カーネルディスパッチ共通ヘルパー
    // ============================================================

    /// Parallel reduction 用: 2冪に切り上げ (reduction の s >>= 1 が全要素をカバーするため)
    fn ceilPow2(v: u64) u64 {
        if (v <= 1) return 1;
        return std.math.ceilPowerOfTwo(u64, v) catch v;
    }

    fn setPipeline(encoder: id, pipeline: id) void {
        send1v(encoder, sel("setComputePipelineState:"), pipeline);
    }

    fn setBuffer(encoder: id, buf: id, offset: u64, index: u64) void {
        send3iuuv(encoder, sel("setBuffer:offset:atIndex:"), buf, offset, index);
    }

    fn setBytes(encoder: id, ptr: *const anyopaque, length: u64, index: u64) void {
        send3puuv(encoder, sel("setBytes:length:atIndex:"), ptr, length, index);
    }

    fn dispatch1D(encoder: id, total: u64, group_size: u64) void {
        const threads_per_grid = MTLSize{ .width = total, .height = 1, .depth = 1 };
        const threads_per_group = MTLSize{ .width = group_size, .height = 1, .depth = 1 };
        send2ssv(encoder, sel("dispatchThreads:threadsPerThreadgroup:"), threads_per_grid, threads_per_group);
    }

    fn dispatchThreadgroups(encoder: id, groups: MTLSize, threads_per_group: MTLSize) void {
        send2ssv(encoder, sel("dispatchThreadgroups:threadsPerThreadgroup:"), groups, threads_per_group);
    }

    fn setThreadgroupMemoryLength(encoder: id, length: u64, index: u64) void {
        const F = *const fn (id, SEL, u64, u64) callconv(.c) void;
        @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(
            encoder,
            sel("setThreadgroupMemoryLength:atIndex:"),
            length,
            index,
        );
    }

    // ============================================================
    // カーネルディスパッチ
    // ============================================================

    pub fn dispatchMatmul(
        self: *MetalContext,
        encoder: id,
        weight_buf: id,
        input_buf: id,
        output_buf: id,
        out_dim: u32,
        in_dim: u32,
        quant_type: QuantType,
    ) void {
        const pipeline = switch (quant_type) {
            .q4_0 => self.pipelines.matmul_q4_0,
            .q4_1 => self.pipelines.matmul_q4_1,
            .q8_0 => self.pipelines.matmul_q8_0,
        };

        const num_blocks = in_dim / 32;
        const row_bytes: u32 = switch (quant_type) {
            .q4_0 => num_blocks * 18,
            .q4_1 => num_blocks * 20,
            .q8_0 => num_blocks * 34,
        };

        const MatmulParams = extern struct {
            out_dim: u32,
            in_dim: u32,
            num_blocks: u32,
            row_bytes: u32,
        };
        const params = MatmulParams{
            .out_dim = out_dim,
            .in_dim = in_dim,
            .num_blocks = num_blocks,
            .row_bytes = row_bytes,
        };

        setPipeline(encoder, pipeline);
        setBuffer(encoder, weight_buf, 0, 0);
        setBuffer(encoder, input_buf, 0, 1);
        setBuffer(encoder, output_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(MatmulParams), 3);

        // SIMD group dispatch - per quant type configuration
        const rows_per_tg: u64 = switch (quant_type) {
            .q4_0 => 2 * 4, // NSG=2, NR0=4 → 8 rows/TG
            .q4_1 => 4,     // NSG=4, NR0=1 → 4 rows/TG
            .q8_0 => 4,     // NSG=4, NR0=1 → 4 rows/TG
        };
        const n_sg: u64 = switch (quant_type) {
            .q4_0 => 2,
            .q4_1 => 4,
            .q8_0 => 4,
        };
        const tg_size: u64 = n_sg * 32;
        const n_groups: u64 = (@as(u64, out_dim) + rows_per_tg - 1) / rows_per_tg;
        dispatchThreadgroups(encoder, .{ .width = n_groups, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchMatmulBatched(
        self: *MetalContext,
        encoder: id,
        weight_buf: id,
        input_buf: id,
        output_buf: id,
        out_dim: u32,
        in_dim: u32,
        m_rows: u32,
        quant_type: QuantType,
    ) void {
        const pipeline = switch (quant_type) {
            .q4_0 => self.pipelines.matmul_q4_0_batched,
            .q4_1 => self.pipelines.matmul_q4_1_batched,
            .q8_0 => self.pipelines.matmul_q8_0_batched,
        };

        const num_blocks = in_dim / 32;
        const row_bytes: u32 = switch (quant_type) {
            .q4_0 => num_blocks * 18,
            .q4_1 => num_blocks * 20,
            .q8_0 => num_blocks * 34,
        };

        const BatchedMatmulParams = extern struct {
            out_dim: u32,
            in_dim: u32,
            num_blocks: u32,
            row_bytes: u32,
            M: u32,
        };
        const params = BatchedMatmulParams{
            .out_dim = out_dim,
            .in_dim = in_dim,
            .num_blocks = num_blocks,
            .row_bytes = row_bytes,
            .M = m_rows,
        };

        setPipeline(encoder, pipeline);
        setBuffer(encoder, weight_buf, 0, 0);
        setBuffer(encoder, input_buf, 0, 1);
        setBuffer(encoder, output_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(BatchedMatmulParams), 3);

        // All batched kernels use N_SG=4, 1 row/SG → 4 rows per threadgroup
        const rows_per_tg: u64 = 4;
        const tg_size: u64 = 4 * 32; // 4 SG × 32 lanes
        const n_groups_x: u64 = (@as(u64, out_dim) + rows_per_tg - 1) / rows_per_tg;
        const n_groups_y: u64 = @as(u64, m_rows);
        dispatchThreadgroups(encoder, .{ .width = n_groups_x, .height = n_groups_y, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchRMSNorm(
        self: *MetalContext,
        encoder: id,
        input_buf: id,
        weight_buf: id,
        output_buf: id,
        dim: u32,
        rows: u32,
        eps: f32,
    ) void {
        const RMSNormParams = extern struct { dim: u32, rows: u32, eps: f32 };
        const params = RMSNormParams{ .dim = dim, .rows = rows, .eps = eps };

        setPipeline(encoder, self.pipelines.rmsnorm);
        setBuffer(encoder, input_buf, 0, 0);
        setBuffer(encoder, weight_buf, 0, 1);
        setBuffer(encoder, output_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(RMSNormParams), 3);

        const tg_size: u64 = @min(@as(u64, dim), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchRMSNormInPlace(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        x_offset: u64,
        weight_buf: id,
        dim: u32,
        eps: f32,
    ) void {
        setPipeline(encoder, self.pipelines.rmsnorm_inplace);
        setBuffer(encoder, x_buf, x_offset, 0);
        setBuffer(encoder, weight_buf, 0, 1);
        setBytes(encoder, @ptrCast(&dim), 4, 2);
        setBytes(encoder, @ptrCast(&eps), 4, 3);

        const tg_size: u64 = @min(@as(u64, dim), 256);
        dispatchThreadgroups(encoder, .{ .width = 1, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchRoPE(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        freqs_buf: id,
        half_dim: u32,
        n_heads: u32,
        pos: f32,
    ) void {
        const RoPEParams = extern struct { half_dim: u32, n_heads: u32, pos: f32 };
        const params = RoPEParams{ .half_dim = half_dim, .n_heads = n_heads, .pos = pos };

        setPipeline(encoder, self.pipelines.rope);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, freqs_buf, 0, 1);
        setBytes(encoder, @ptrCast(&params), @sizeOf(RoPEParams), 2);

        const total: u64 = @as(u64, n_heads) * @as(u64, half_dim);
        dispatch1D(encoder, total, @min(total, 256));
    }

    pub fn dispatchGELU(self: *MetalContext, encoder: id, x_buf: id, n: u32) void {
        setPipeline(encoder, self.pipelines.gelu);
        setBuffer(encoder, x_buf, 0, 0);
        setBytes(encoder, @ptrCast(&n), 4, 1);
        dispatch1D(encoder, n, @min(@as(u64, n), 256));
    }

    pub fn dispatchGELUMul(self: *MetalContext, encoder: id, gate_buf: id, up_buf: id, n: u32) void {
        setPipeline(encoder, self.pipelines.gelu_mul);
        setBuffer(encoder, gate_buf, 0, 0);
        setBuffer(encoder, up_buf, 0, 1);
        setBytes(encoder, @ptrCast(&n), 4, 2);
        dispatch1D(encoder, n, @min(@as(u64, n), 256));
    }

    pub fn dispatchRMSNormResidual(
        self: *MetalContext,
        encoder: id,
        input_buf: id,
        weight_buf: id,
        residual_buf: id,
        dim: u32,
        rows: u32,
        eps: f32,
    ) void {
        const RMSNormParams = extern struct { dim: u32, rows: u32, eps: f32 };
        const params = RMSNormParams{ .dim = dim, .rows = rows, .eps = eps };

        setPipeline(encoder, self.pipelines.rmsnorm_residual);
        setBuffer(encoder, input_buf, 0, 0);
        setBuffer(encoder, weight_buf, 0, 1);
        setBuffer(encoder, residual_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(RMSNormParams), 3);

        const tg_size: u64 = @min(@as(u64, dim), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchAdd(self: *MetalContext, encoder: id, a_buf: id, b_buf: id, n: u32) void {
        setPipeline(encoder, self.pipelines.add_inplace);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBytes(encoder, @ptrCast(&n), 4, 2);
        dispatch1D(encoder, n, @min(@as(u64, n), 256));
    }

    pub fn dispatchMul(self: *MetalContext, encoder: id, a_buf: id, b_buf: id, n: u32) void {
        setPipeline(encoder, self.pipelines.mul_inplace);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBytes(encoder, @ptrCast(&n), 4, 2);
        dispatch1D(encoder, n, @min(@as(u64, n), 256));
    }

    pub fn dispatchScale(self: *MetalContext, encoder: id, x_buf: id, scalar: f32, n: u32) void {
        setPipeline(encoder, self.pipelines.scale_inplace);
        setBuffer(encoder, x_buf, 0, 0);
        setBytes(encoder, @ptrCast(&scalar), 4, 1);
        setBytes(encoder, @ptrCast(&n), 4, 2);
        dispatch1D(encoder, n, @min(@as(u64, n), 256));
    }

    pub fn dispatchDequantQ8Row(
        self: *MetalContext,
        encoder: id,
        weight_buf: id,
        output_buf: id,
        token_id: u32,
        embed_dim: u32,
    ) void {
        setPipeline(encoder, self.pipelines.dequant_q8_0_row);
        setBuffer(encoder, weight_buf, 0, 0);
        setBuffer(encoder, output_buf, 0, 1);
        setBytes(encoder, @ptrCast(&token_id), 4, 2);
        setBytes(encoder, @ptrCast(&embed_dim), 4, 3);
        dispatch1D(encoder, embed_dim, @min(@as(u64, embed_dim), 256));
    }

    pub fn dispatchDequantQ8RowScaled(
        self: *MetalContext,
        encoder: id,
        weight_buf: id,
        output_buf: id,
        token_id: u32,
        embed_dim: u32,
        embed_scale: f32,
    ) void {
        setPipeline(encoder, self.pipelines.dequant_q8_0_row_scaled);
        setBuffer(encoder, weight_buf, 0, 0);
        setBuffer(encoder, output_buf, 0, 1);
        setBytes(encoder, @ptrCast(&token_id), 4, 2);
        setBytes(encoder, @ptrCast(&embed_dim), 4, 3);
        setBytes(encoder, @ptrCast(&embed_scale), 4, 4);
        dispatch1D(encoder, embed_dim, @min(@as(u64, embed_dim), 256));
    }

    pub fn dispatchAttentionDecode(
        self: *MetalContext,
        encoder: id,
        q_buf: id,
        k_cache_buf: id,
        v_cache_buf: id,
        output_buf: id,
        scores_buf: id,
        n_head: u32,
        head_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        kv_start: u32,
        kv_end: u32,
    ) void {
        const AttentionParams = extern struct {
            n_head: u32,
            head_dim: u32,
            q_dim: u32,
            kv_dim: u32,
            kv_start: u32,
            kv_end: u32,
            scale: f32,
        };
        const params = AttentionParams{
            .n_head = n_head,
            .head_dim = head_dim,
            .q_dim = q_dim,
            .kv_dim = kv_dim,
            .kv_start = kv_start,
            .kv_end = kv_end,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        };

        setPipeline(encoder, self.pipelines.gqa_attention_decode);
        setBuffer(encoder, q_buf, 0, 0);
        setBuffer(encoder, k_cache_buf, 0, 1);
        setBuffer(encoder, v_cache_buf, 0, 2);
        setBuffer(encoder, output_buf, 0, 3);
        setBuffer(encoder, scores_buf, 0, 4);
        setBytes(encoder, @ptrCast(&params), @sizeOf(AttentionParams), 5);

        // Reduction (softmax) は2の冪 threadgroup サイズが必須
        // shared[256] に合わせて常に256を使用。kv_len < 256 のスレッドは
        // identity 値 (-INF for max, 0 for sum) を提供するので正しく動作する。
        const tg_size: u64 = 256;
        dispatchThreadgroups(encoder, .{ .width = n_head, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchWriteKVCache(
        self: *MetalContext,
        encoder: id,
        k_new_buf: id,
        v_new_buf: id,
        k_cache_buf: id,
        v_cache_buf: id,
        pos: u32,
        kv_dim: u32,
    ) void {
        setPipeline(encoder, self.pipelines.write_kv_cache);
        setBuffer(encoder, k_new_buf, 0, 0);
        setBuffer(encoder, v_new_buf, 0, 1);
        setBuffer(encoder, k_cache_buf, 0, 2);
        setBuffer(encoder, v_cache_buf, 0, 3);
        setBytes(encoder, @ptrCast(&pos), 4, 4);
        setBytes(encoder, @ptrCast(&kv_dim), 4, 5);
        dispatch1D(encoder, kv_dim, @min(@as(u64, kv_dim), 256));
    }

    // ============================================================
    // コマンドバッファ操作
    // ============================================================

    pub fn newCommandBuffer(self: *MetalContext) id {
        // commandBufferWithUnretainedReferences: ARC オーバーヘッド削減
        // バッファの retain/release をスキップ → 342 encoder で大量の setBuffer 呼び出しのコスト削減
        // 呼び出し元が waitUntilCompleted まで全バッファを保持する前提
        return send0(self.command_queue, sel("commandBufferWithUnretainedReferences")).?;
    }

    pub fn newComputeEncoder(cmd_buf: id) id {
        return send0(cmd_buf, sel("computeCommandEncoder")).?;
    }

    pub fn endEncoding(encoder: id) void {
        send0v(encoder, sel("endEncoding"));
    }

    pub fn commit(cmd_buf: id) void {
        send0v(cmd_buf, sel("commit"));
    }

    pub fn waitUntilCompleted(cmd_buf: id) void {
        send0v(cmd_buf, sel("waitUntilCompleted"));
    }

    // ============================================================
    // Batch mode: 複数 dispatch を 1 コマンドバッファに統合
    // ============================================================

    /// バッチモード開始: 共有コマンドバッファ + エンコーダを作成
    /// gpuExec は dispatch のみ行い、commit/wait をスキップする
    pub fn beginBatch(self: *MetalContext) void {
        std.debug.assert(self.batch_cmd_buf == null);
        self.batch_cmd_buf = self.newCommandBuffer();
        self.batch_encoder = newComputeEncoder(self.batch_cmd_buf.?);
    }

    /// バッチモード終了: memoryBarrier + endEncoding + commit + wait
    pub fn endBatch(self: *MetalContext) void {
        // プロファイルモード: 最後のカテゴリの時間を計測
        if (self.profile_mode) {
            self.profileFlush();
        }
        const encoder = self.batch_encoder.?;
        const cmd_buf = self.batch_cmd_buf.?;
        memoryBarrier(encoder);
        endEncoding(encoder);
        commit(cmd_buf);
        waitUntilCompleted(cmd_buf);
        self.batch_encoder = null;
        self.batch_cmd_buf = null;
    }

    // ============================================================
    // Backward Batch: grad を MTLBuffer で管理、CPU round-trip 排除
    // ============================================================

    /// ノード → grad MTLBuffer のマッピング + 一時バッファ管理
    pub const BackwardGradState = struct {
        map: std.AutoHashMap(usize, id),
        temp_bufs: std.ArrayListUnmanaged(id),
        allocator: std.mem.Allocator,
    };

    /// Backward バッチモード開始
    /// weight ノードの grad_buf を事前登録してから呼ぶこと
    pub fn beginBackwardBatch(self: *MetalContext, allocator: std.mem.Allocator) void {
        std.debug.assert(self.backward_grad_state == null);
        self.beginBatch();
        self.backward_grad_state = .{
            .map = std.AutoHashMap(usize, id).init(allocator),
            .temp_bufs = .{},
            .allocator = allocator,
        };
    }

    /// weight ノードの grad_buf を事前登録 (backward 前に呼ぶ)
    pub fn registerGradBuf(self: *MetalContext, node_ptr: *anyopaque, buf: id) void {
        var state = &self.backward_grad_state.?;
        state.map.put(@intFromPtr(node_ptr), buf) catch {};
    }

    /// ノードの grad MTLBuffer を取得 (なければ zero 初期化で新規作成)
    pub fn getOrAllocGradBuf(self: *MetalContext, node_ptr: *anyopaque, num_bytes: usize) id {
        var state = &self.backward_grad_state.?;
        const key = @intFromPtr(node_ptr);
        if (state.map.get(key)) |buf| return buf;

        const buf = self.createBuffer(num_bytes) catch unreachable;
        // Zero-initialize via UMA
        const ptr = bufferContents(u8, buf);
        @memset(ptr[0..num_bytes], 0);
        state.map.put(key, buf) catch {};
        state.temp_bufs.append(state.allocator, buf) catch {};
        return buf;
    }

    /// 既にバッファがなければ alias として登録 (dispatch 不要にする)
    /// 既にバッファがあれば false を返す (accumulation dispatch が必要)
    /// temp_bufs には追加しない (alias_buf は別ノードが所有)
    pub fn tryAliasGradBuf(self: *MetalContext, node_ptr: *anyopaque, alias_buf: id) bool {
        var state = &self.backward_grad_state.?;
        const key = @intFromPtr(node_ptr);
        if (state.map.get(key) != null) return false;
        state.map.put(key, alias_buf) catch {};
        return true;
    }

    /// 一時バッファを backward バッチの管理リストに追加 (endBackwardBatch で解放)
    pub fn addTempBuf(self: *MetalContext, buf: id) void {
        var state = &self.backward_grad_state.?;
        state.temp_bufs.append(state.allocator, buf) catch {};
    }

    /// Backward バッチモード終了: commit/wait + 一時バッファ解放
    pub fn endBackwardBatch(self: *MetalContext) void {
        self.endBatch();
        var state = &self.backward_grad_state.?;
        for (state.temp_bufs.items) |buf| {
            objRelease(buf);
        }
        state.temp_bufs.deinit(state.allocator);
        state.map.deinit();
        self.backward_grad_state = null;
    }

    // ============================================================
    // Training Pipelines (遅延初期化)
    // ============================================================

    pub fn initTrainingPipelines(self: *MetalContext) !void {
        if (self.training_pipelines != null) return;

        // 学習用 MSL ソースをコンパイル
        const msl_source = @embedFile("shaders/nn_training_kernels.metal");
        const ns_source = nsString(msl_source.ptr);

        var err: ?*anyopaque = null;
        const lib = send3iie(
            self.device,
            sel("newLibraryWithSource:options:error:"),
            ns_source,
            null,
            &err,
        ) orelse {
            if (err) |e| {
                const desc = send0(e, sel("localizedDescription"));
                if (desc) |d| {
                    const cstr = send0str(d, sel("UTF8String"));
                    if (cstr) |s| {
                        log.err("training shader compilation error: {s}", .{s});
                    }
                }
            }
            return error.ShaderCompilationFailed;
        };
        self.training_library = lib;

        self.training_pipelines = .{
            .matmul_f32 = try self.createTrainingPipeline(lib, "matmul_f32"),
            .add_f32 = try self.createTrainingPipeline(lib, "add_f32"),
            .add_bias_f32 = try self.createTrainingPipeline(lib, "add_bias_f32"),
            .silu_forward = try self.createTrainingPipeline(lib, "silu_forward"),
            .mse_loss_diff = try self.createTrainingPipeline(lib, "mse_loss_diff"),
            .mse_loss_reduce = try self.createTrainingPipeline(lib, "mse_loss_reduce"),
            .matmul_f32_backward_a = try self.createTrainingPipeline(lib, "matmul_f32_backward_a"),
            .matmul_f32_backward_b = try self.createTrainingPipeline(lib, "matmul_f32_backward_b"),
            .add_backward_accum = try self.createTrainingPipeline(lib, "add_backward_accum"),
            .add_bias_backward = try self.createTrainingPipeline(lib, "add_bias_backward"),
            .silu_backward = try self.createTrainingPipeline(lib, "silu_backward"),
            .mse_loss_backward = try self.createTrainingPipeline(lib, "mse_loss_backward"),
            .adam_step = try self.createTrainingPipeline(lib, "adam_step"),
            .zero_buffer = try self.createTrainingPipeline(lib, "zero_buffer"),
            // Phase 2: Transformer
            .relu_forward = try self.createTrainingPipeline(lib, "relu_forward"),
            .relu_backward = try self.createTrainingPipeline(lib, "relu_backward"),
            .gelu_forward = try self.createTrainingPipeline(lib, "gelu_forward"),
            .gelu_backward = try self.createTrainingPipeline(lib, "gelu_backward"),
            .softmax_f32 = try self.createTrainingPipeline(lib, "softmax_f32"),
            .softmax_backward = try self.createTrainingPipeline(lib, "softmax_backward"),
            .causal_softmax_f32 = try self.createTrainingPipeline(lib, "causal_softmax_f32"),
            .layernorm_forward = try self.createTrainingPipeline(lib, "layernorm_forward"),
            .layernorm_backward_x = try self.createTrainingPipeline(lib, "layernorm_backward_x"),
            .layernorm_backward_params = try self.createTrainingPipeline(lib, "layernorm_backward_params"),
            .cross_entropy_forward = try self.createTrainingPipeline(lib, "cross_entropy_forward"),
            .cross_entropy_reduce = try self.createTrainingPipeline(lib, "cross_entropy_reduce"),
            .cross_entropy_backward = try self.createTrainingPipeline(lib, "cross_entropy_backward"),
            .embedding_forward = try self.createTrainingPipeline(lib, "embedding_forward"),
            .embedding_backward = try self.createTrainingPipeline(lib, "embedding_backward"),
            .scale_f32 = try self.createTrainingPipeline(lib, "scale_f32"),
            .scale_backward = try self.createTrainingPipeline(lib, "scale_backward"),
            .matmul_f32_trans_b = try self.createTrainingPipeline(lib, "matmul_f32_trans_b"),
            .matmul_f32_accum = try self.createTrainingPipeline(lib, "matmul_f32_accum"),
            // Phase 3: QLoRA
            .matmul_q4_0_trans_batched = try self.createTrainingPipeline(lib, "matmul_q4_0_trans_batched"),
            .matmul_q4_1_trans_batched = try self.createTrainingPipeline(lib, "matmul_q4_1_trans_batched"),
            .matmul_q8_0_trans_batched = try self.createTrainingPipeline(lib, "matmul_q8_0_trans_batched"),
            .rmsnorm_forward_training = try self.createTrainingPipeline(lib, "rmsnorm_forward_training"),
            .rmsnorm_backward_x = try self.createTrainingPipeline(lib, "rmsnorm_backward_x"),
            .rmsnorm_backward_weight = try self.createTrainingPipeline(lib, "rmsnorm_backward_weight"),
            .rope_forward_training = try self.createTrainingPipeline(lib, "rope_forward_training"),
            .rope_backward = try self.createTrainingPipeline(lib, "rope_backward"),
            .dequant_q8_0_batch_scaled = try self.createTrainingPipeline(lib, "dequant_q8_0_batch_scaled"),
            // Phase 4: Sequence ops
            .tanh_forward = try self.createTrainingPipeline(lib, "tanh_forward"),
            .tanh_backward = try self.createTrainingPipeline(lib, "tanh_backward"),
            .concat_last_dim = try self.createTrainingPipeline(lib, "concat_last_dim"),
            .concat_last_dim_backward = try self.createTrainingPipeline(lib, "concat_last_dim_backward"),
            // Phase 5: Batched matmul
            .batched_matmul_f32 = try self.createTrainingPipeline(lib, "batched_matmul_f32"),
            .batched_matmul_trans_b_f32 = try self.createTrainingPipeline(lib, "batched_matmul_trans_b_f32"),
            .batched_matmul_backward_a_f32 = try self.createTrainingPipeline(lib, "batched_matmul_backward_a_f32"),
            .batched_matmul_backward_b_f32 = try self.createTrainingPipeline(lib, "batched_matmul_backward_b_f32"),
            .batched_matmul_trans_b_backward_a_f32 = try self.createTrainingPipeline(lib, "batched_matmul_trans_b_backward_a_f32"),
            .batched_matmul_trans_b_backward_b_f32 = try self.createTrainingPipeline(lib, "batched_matmul_trans_b_backward_b_f32"),
            // Phase 6: Fused kernels
            .matmul_addbias_gelu_f32 = try self.createTrainingPipeline(lib, "matmul_addbias_gelu_f32"),
            .gelu_bias_backward = try self.createTrainingPipeline(lib, "gelu_bias_backward"),
            .matmul_addbias_tanh_f32 = try self.createTrainingPipeline(lib, "matmul_addbias_tanh_f32"),
            .tanh_bias_backward = try self.createTrainingPipeline(lib, "tanh_bias_backward"),
            .batched_matmul_trans_b_scale_f32 = try self.createTrainingPipeline(lib, "batched_matmul_trans_b_scale_f32"),
        };
    }

    fn createTrainingPipeline(self: *MetalContext, lib: id, name: [*:0]const u8) !id {
        const ns_name = nsString(name);
        const func = send1(lib, sel("newFunctionWithName:"), ns_name) orelse {
            log.err("training function '{s}' not found", .{name});
            return error.MetalFunctionNotFound;
        };
        defer objRelease(func);

        var err: ?*anyopaque = null;
        const pipeline = send2ie(
            self.device,
            sel("newComputePipelineStateWithFunction:error:"),
            func,
            &err,
        ) orelse {
            return error.MetalPipelineCreationFailed;
        };
        return pipeline;
    }

    // ============================================================
    // Training Dispatch 関数群
    // ============================================================

    const TrainingMatmulParams = extern struct { M: u32, K: u32, N: u32 };
    const TrainingBiasParams = extern struct { rows: u32, cols: u32 };

    pub fn dispatchMatmulF32(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        c_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingMatmulParams{ .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.matmul_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, c_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingMatmulParams), 3);
        // Threadgroups: ceil(N/64) x ceil(M/64), threadgroup 16x16
        const gx: u64 = (@as(u64, n_dim) + 63) / 64;
        const gy: u64 = (@as(u64, m_dim) + 63) / 64;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchAddF32(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        c_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.add_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, c_buf, 0, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchAddBiasF32(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        bias_buf: id,
        z_buf: id,
        rows: u32,
        cols: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBiasParams{ .rows = rows, .cols = cols };
        setPipeline(encoder, tp.add_bias_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, bias_buf, 0, 1);
        setBuffer(encoder, z_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBiasParams), 3);
        const total: u64 = @as(u64, rows) * @as(u64, cols);
        dispatch1D(encoder, total, @min(total, 256));
    }

    pub fn dispatchSiluForward(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        out_buf: id,
        sig_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.silu_forward);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, out_buf, 0, 1);
        setBuffer(encoder, sig_buf, 0, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchMseLossDiff(
        self: *MetalContext,
        encoder: id,
        pred_buf: id,
        target_buf: id,
        diff_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.mse_loss_diff);
        setBuffer(encoder, pred_buf, 0, 0);
        setBuffer(encoder, target_buf, 0, 1);
        setBuffer(encoder, diff_buf, 0, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchMseLossReduce(
        self: *MetalContext,
        encoder: id,
        diff_buf: id,
        loss_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.mse_loss_reduce);
        setBuffer(encoder, diff_buf, 0, 0);
        setBuffer(encoder, loss_buf, 0, 1);
        setBytes(encoder, @ptrCast(&count), 4, 2);
        // Single threadgroup reduction
        const tg_size: u64 = 256;
        dispatchThreadgroups(encoder, .{ .width = 1, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchMatmulBackwardA(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        b_buf: id,
        grad_a_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingMatmulParams{ .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.matmul_f32_backward_a);
        setBuffer(encoder, grad_out_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, grad_a_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingMatmulParams), 3);
        // Output: (M, K), BM=64 block tiling
        const gx: u64 = (@as(u64, k_dim) + 63) / 64;
        const gy: u64 = (@as(u64, m_dim) + 63) / 64;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchMatmulBackwardB(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        grad_out_buf: id,
        grad_b_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingMatmulParams{ .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.matmul_f32_backward_b);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, grad_b_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingMatmulParams), 3);
        // Output: (K, N), BM=64 block tiling
        const gx: u64 = (@as(u64, n_dim) + 63) / 64;
        const gy: u64 = (@as(u64, k_dim) + 63) / 64;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchAddBackwardAccum(
        self: *MetalContext,
        encoder: id,
        src_buf: id,
        grad_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.add_backward_accum);
        setBuffer(encoder, src_buf, 0, 0);
        setBuffer(encoder, grad_buf, 0, 1);
        setBytes(encoder, @ptrCast(&count), 4, 2);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchAddBiasBackward(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        grad_bias_buf: id,
        rows: u32,
        cols: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBiasParams{ .rows = rows, .cols = cols };
        setPipeline(encoder, tp.add_bias_backward);
        setBuffer(encoder, grad_out_buf, 0, 0);
        setBuffer(encoder, grad_bias_buf, 0, 1);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBiasParams), 2);
        dispatch1D(encoder, cols, @min(@as(u64, cols), 256));
    }

    pub fn dispatchSiluBackward(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        x_buf: id,
        sig_buf: id,
        grad_x_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.silu_backward);
        setBuffer(encoder, grad_out_buf, 0, 0);
        setBuffer(encoder, x_buf, 0, 1);
        setBuffer(encoder, sig_buf, 0, 2);
        setBuffer(encoder, grad_x_buf, 0, 3);
        setBytes(encoder, @ptrCast(&count), 4, 4);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchMseLossBackward(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        diff_buf: id,
        grad_pred_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.mse_loss_backward);
        setBuffer(encoder, grad_out_buf, 0, 0);
        setBuffer(encoder, diff_buf, 0, 1);
        setBuffer(encoder, grad_pred_buf, 0, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    const AdamKernelParams = extern struct {
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        count: u32,
    };

    pub fn dispatchAdamStep(
        self: *MetalContext,
        encoder: id,
        weights_buf: id,
        grads_buf: id,
        m_buf: id,
        v_buf: id,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = AdamKernelParams{
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .epsilon = epsilon,
            .weight_decay = weight_decay,
            .bc1 = bc1,
            .bc2 = bc2,
            .count = count,
        };
        setPipeline(encoder, tp.adam_step);
        setBuffer(encoder, weights_buf, 0, 0);
        setBuffer(encoder, grads_buf, 0, 1);
        setBuffer(encoder, m_buf, 0, 2);
        setBuffer(encoder, v_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(AdamKernelParams), 4);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchZeroBuffer(
        self: *MetalContext,
        encoder: id,
        buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.zero_buffer);
        setBuffer(encoder, buf, 0, 0);
        setBytes(encoder, @ptrCast(&count), 4, 1);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    // ============================================================
    // Phase 2: Transformer Training Dispatch 関数群
    // ============================================================

    const TrainingSoftmaxParams = extern struct { rows: u32, cols: u32 };
    const TrainingCausalSoftmaxParams = extern struct { rows: u32, cols: u32, num_heads: u32, seq_len: u32 };
    const TrainingLayerNormParams = extern struct { rows: u32, cols: u32, epsilon: f32 };
    const TrainingCrossEntropyParams = extern struct { batch_size: u32, num_classes: u32 };
    const TrainingEmbeddingParams = extern struct { num_tokens: u32, embed_dim: u32 };

    // --- ReLU ---

    pub fn dispatchReluForward(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        out_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.relu_forward);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, out_buf, 0, 1);
        setBytes(encoder, @ptrCast(&count), 4, 2);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchReluBackward(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        grad_out_buf: id,
        grad_in_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.relu_backward);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, grad_in_buf, 0, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    // --- GELU ---

    pub fn dispatchGeluForward(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        out_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.gelu_forward);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, out_buf, 0, 1);
        setBytes(encoder, @ptrCast(&count), 4, 2);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchGeluBackward(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        grad_out_buf: id,
        grad_in_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.gelu_backward);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, grad_in_buf, 0, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    // --- Softmax ---

    pub fn dispatchSoftmaxF32(
        self: *MetalContext,
        encoder: id,
        input_buf: id,
        output_buf: id,
        rows: u32,
        cols: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingSoftmaxParams{ .rows = rows, .cols = cols };
        setPipeline(encoder, tp.softmax_f32);
        setBuffer(encoder, input_buf, 0, 0);
        setBuffer(encoder, output_buf, 0, 1);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingSoftmaxParams), 2);
        // Parallel reduction requires power-of-2 threadgroup size;
        // extra threads use identity values (-INFINITY for max, 0 for sum)
        const tg_size: u64 = @min(ceilPow2(@as(u64, cols)), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchSoftmaxBackward(
        self: *MetalContext,
        encoder: id,
        out_buf: id,
        grad_out_buf: id,
        grad_in_buf: id,
        rows: u32,
        cols: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingSoftmaxParams{ .rows = rows, .cols = cols };
        setPipeline(encoder, tp.softmax_backward);
        setBuffer(encoder, out_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, grad_in_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingSoftmaxParams), 3);
        const tg_size: u64 = @min(ceilPow2(@as(u64, cols)), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchCausalSoftmaxF32(
        self: *MetalContext,
        encoder: id,
        input_buf: id,
        output_buf: id,
        rows: u32,
        cols: u32,
        num_heads: u32,
        seq_len: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingCausalSoftmaxParams{ .rows = rows, .cols = cols, .num_heads = num_heads, .seq_len = seq_len };
        setPipeline(encoder, tp.causal_softmax_f32);
        setBuffer(encoder, input_buf, 0, 0);
        setBuffer(encoder, output_buf, 0, 1);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingCausalSoftmaxParams), 2);
        const tg_size: u64 = @min(@as(u64, cols), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    // --- LayerNorm ---

    pub fn dispatchLayerNormForward(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        gamma_buf: id,
        beta_buf: id,
        out_buf: id,
        mean_buf: id,
        inv_std_buf: id,
        rows: u32,
        cols: u32,
        epsilon: f32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingLayerNormParams{ .rows = rows, .cols = cols, .epsilon = epsilon };
        setPipeline(encoder, tp.layernorm_forward);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, gamma_buf, 0, 1);
        setBuffer(encoder, beta_buf, 0, 2);
        setBuffer(encoder, out_buf, 0, 3);
        setBuffer(encoder, mean_buf, 0, 4);
        setBuffer(encoder, inv_std_buf, 0, 5);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingLayerNormParams), 6);
        const tg_size: u64 = @min(@as(u64, cols), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchLayerNormBackwardX(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        gamma_buf: id,
        grad_out_buf: id,
        mean_buf: id,
        inv_std_buf: id,
        grad_x_buf: id,
        rows: u32,
        cols: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingLayerNormParams{ .rows = rows, .cols = cols, .epsilon = 0 };
        setPipeline(encoder, tp.layernorm_backward_x);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, gamma_buf, 0, 1);
        setBuffer(encoder, grad_out_buf, 0, 2);
        setBuffer(encoder, mean_buf, 0, 3);
        setBuffer(encoder, inv_std_buf, 0, 4);
        setBuffer(encoder, grad_x_buf, 0, 5);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingLayerNormParams), 6);
        const tg_size: u64 = @min(@as(u64, cols), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchLayerNormBackwardParams(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        grad_out_buf: id,
        mean_buf: id,
        inv_std_buf: id,
        grad_gamma_buf: id,
        grad_beta_buf: id,
        rows: u32,
        cols: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingLayerNormParams{ .rows = rows, .cols = cols, .epsilon = 0 };
        setPipeline(encoder, tp.layernorm_backward_params);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, mean_buf, 0, 2);
        setBuffer(encoder, inv_std_buf, 0, 3);
        setBuffer(encoder, grad_gamma_buf, 0, 4);
        setBuffer(encoder, grad_beta_buf, 0, 5);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingLayerNormParams), 6);
        dispatch1D(encoder, cols, @min(@as(u64, cols), 256));
    }

    // --- Cross-Entropy Loss ---

    pub fn dispatchCrossEntropyForward(
        self: *MetalContext,
        encoder: id,
        logits_buf: id,
        targets_buf: id,
        softmax_buf: id,
        loss_buf: id,
        batch_size: u32,
        num_classes: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingCrossEntropyParams{ .batch_size = batch_size, .num_classes = num_classes };
        setPipeline(encoder, tp.cross_entropy_forward);
        setBuffer(encoder, logits_buf, 0, 0);
        setBuffer(encoder, targets_buf, 0, 1);
        setBuffer(encoder, softmax_buf, 0, 2);
        setBuffer(encoder, loss_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingCrossEntropyParams), 4);
        const tg_size: u64 = @min(@as(u64, num_classes), 256);
        dispatchThreadgroups(encoder, .{ .width = batch_size, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchCrossEntropyReduce(
        self: *MetalContext,
        encoder: id,
        losses_buf: id,
        total_loss_buf: id,
        batch_size: u32,
        targets_buf: id,
        num_classes: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.cross_entropy_reduce);
        setBuffer(encoder, losses_buf, 0, 0);
        setBuffer(encoder, total_loss_buf, 0, 1);
        setBytes(encoder, @ptrCast(&batch_size), 4, 2);
        setBuffer(encoder, targets_buf, 0, 3);
        setBytes(encoder, @ptrCast(&num_classes), 4, 4);
        const tg_size: u64 = @min(@as(u64, batch_size), 256);
        dispatchThreadgroups(encoder, .{ .width = 1, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchCrossEntropyBackward(
        self: *MetalContext,
        encoder: id,
        softmax_buf: id,
        targets_buf: id,
        grad_logits_buf: id,
        grad_out_buf: id,
        batch_size: u32,
        num_classes: u32,
        valid_count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingCrossEntropyParams{ .batch_size = batch_size, .num_classes = num_classes };
        setPipeline(encoder, tp.cross_entropy_backward);
        setBuffer(encoder, softmax_buf, 0, 0);
        setBuffer(encoder, targets_buf, 0, 1);
        setBuffer(encoder, grad_logits_buf, 0, 2);
        setBuffer(encoder, grad_out_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingCrossEntropyParams), 4);
        setBytes(encoder, @ptrCast(&valid_count), 4, 5);
        const total: u64 = @as(u64, batch_size) * @as(u64, num_classes);
        dispatch1D(encoder, total, @min(total, 256));
    }

    // --- Embedding ---

    pub fn dispatchEmbeddingForward(
        self: *MetalContext,
        encoder: id,
        weight_buf: id,
        indices_buf: id,
        out_buf: id,
        num_tokens: u32,
        embed_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingEmbeddingParams{ .num_tokens = num_tokens, .embed_dim = embed_dim };
        setPipeline(encoder, tp.embedding_forward);
        setBuffer(encoder, weight_buf, 0, 0);
        setBuffer(encoder, indices_buf, 0, 1);
        setBuffer(encoder, out_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingEmbeddingParams), 3);
        const total: u64 = @as(u64, num_tokens) * @as(u64, embed_dim);
        dispatch1D(encoder, total, @min(total, 256));
    }

    pub fn dispatchEmbeddingBackward(
        self: *MetalContext,
        encoder: id,
        indices_buf: id,
        grad_out_buf: id,
        grad_weight_buf: id,
        num_tokens: u32,
        embed_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingEmbeddingParams{ .num_tokens = num_tokens, .embed_dim = embed_dim };
        setPipeline(encoder, tp.embedding_backward);
        setBuffer(encoder, indices_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, grad_weight_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingEmbeddingParams), 3);
        const total: u64 = @as(u64, num_tokens) * @as(u64, embed_dim);
        dispatch1D(encoder, total, @min(total, 256));
    }

    // --- Scale ---

    pub fn dispatchScaleF32(
        self: *MetalContext,
        encoder: id,
        input_buf: id,
        output_buf: id,
        scale_val: f32,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.scale_f32);
        setBuffer(encoder, input_buf, 0, 0);
        setBuffer(encoder, output_buf, 0, 1);
        setBytes(encoder, @ptrCast(&scale_val), 4, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchScaleBackward(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        grad_in_buf: id,
        scale_val: f32,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.scale_backward);
        setBuffer(encoder, grad_out_buf, 0, 0);
        setBuffer(encoder, grad_in_buf, 0, 1);
        setBytes(encoder, @ptrCast(&scale_val), 4, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    // --- Matmul variants for attention ---

    pub fn dispatchMatmulF32TransB(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        c_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingMatmulParams{ .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.matmul_f32_trans_b);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, c_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingMatmulParams), 3);
        const gx: u64 = (@as(u64, n_dim) + 63) / 64;
        const gy: u64 = (@as(u64, m_dim) + 63) / 64;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchMatmulF32Accum(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        c_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingMatmulParams{ .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.matmul_f32_accum);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, c_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingMatmulParams), 3);
        const gx: u64 = (@as(u64, n_dim) + 63) / 64;
        const gy: u64 = (@as(u64, m_dim) + 63) / 64;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }
    // ============================================================
    // Phase 3: QLoRA Training Dispatch 関数群
    // ============================================================

    const BatchedQuantParams = extern struct {
        out_dim: u32,
        in_dim: u32,
        num_blocks: u32,
        row_bytes: u32,
        M: u32,
    };

    const RMSNormTrainParams = extern struct {
        rows: u32,
        dim: u32,
        eps: f32,
    };

    const RoPETrainParams = extern struct {
        seq_len: u32,
        n_heads: u32,
        half_dim: u32,
    };

    const BatchedDequantParams = extern struct {
        num_tokens: u32,
        embed_dim: u32,
    };

    pub fn dispatchQuantTransBatched(
        self: *MetalContext,
        encoder: id,
        weight_buf: id,
        grad_out_buf: id,
        grad_x_buf: id,
        out_dim: u32,
        in_dim: u32,
        m_rows: u32,
        quant_type: QuantType,
    ) void {
        const tp = self.training_pipelines.?;
        const pipeline = switch (quant_type) {
            .q4_0 => tp.matmul_q4_0_trans_batched,
            .q4_1 => tp.matmul_q4_1_trans_batched,
            .q8_0 => tp.matmul_q8_0_trans_batched,
        };

        const num_blocks = in_dim / 32;
        const row_bytes: u32 = switch (quant_type) {
            .q4_0 => num_blocks * 18,
            .q4_1 => num_blocks * 20,
            .q8_0 => num_blocks * 34,
        };

        const params = BatchedQuantParams{
            .out_dim = out_dim,
            .in_dim = in_dim,
            .num_blocks = num_blocks,
            .row_bytes = row_bytes,
            .M = m_rows,
        };

        setPipeline(encoder, pipeline);
        setBuffer(encoder, weight_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, grad_x_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(BatchedQuantParams), 3);

        // 16x16 threadgroup: x=in_dim, y=M
        const gx: u64 = (@as(u64, in_dim) + 15) / 16;
        const gy: u64 = (@as(u64, m_rows) + 15) / 16;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchRMSNormForwardTraining(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        weight_buf: id,
        out_buf: id,
        inv_rms_buf: id,
        rows: u32,
        dim: u32,
        eps: f32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = RMSNormTrainParams{ .rows = rows, .dim = dim, .eps = eps };
        setPipeline(encoder, tp.rmsnorm_forward_training);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, weight_buf, 0, 1);
        setBuffer(encoder, out_buf, 0, 2);
        setBuffer(encoder, inv_rms_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(RMSNormTrainParams), 4);
        // Parallel reduction requires power-of-2 threadgroup size; MSL fills extra
        // lanes with local_ss=0 because the per-lane stride loop skips them.
        const tg_size: u64 = @min(ceilPow2(@as(u64, dim)), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchRMSNormBackwardX(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        weight_buf: id,
        grad_out_buf: id,
        inv_rms_buf: id,
        grad_x_buf: id,
        rows: u32,
        dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = RMSNormTrainParams{ .rows = rows, .dim = dim, .eps = 0 };
        setPipeline(encoder, tp.rmsnorm_backward_x);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, weight_buf, 0, 1);
        setBuffer(encoder, grad_out_buf, 0, 2);
        setBuffer(encoder, inv_rms_buf, 0, 3);
        setBuffer(encoder, grad_x_buf, 0, 4);
        setBytes(encoder, @ptrCast(&params), @sizeOf(RMSNormTrainParams), 5);
        const tg_size: u64 = @min(ceilPow2(@as(u64, dim)), 256);
        dispatchThreadgroups(encoder, .{ .width = rows, .height = 1, .depth = 1 }, .{ .width = tg_size, .height = 1, .depth = 1 });
    }

    pub fn dispatchRMSNormBackwardWeight(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        grad_out_buf: id,
        inv_rms_buf: id,
        grad_weight_buf: id,
        rows: u32,
        dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = RMSNormTrainParams{ .rows = rows, .dim = dim, .eps = 0 };
        setPipeline(encoder, tp.rmsnorm_backward_weight);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, inv_rms_buf, 0, 2);
        setBuffer(encoder, grad_weight_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(RMSNormTrainParams), 4);
        dispatch1D(encoder, dim, @min(@as(u64, dim), 256));
    }

    pub fn dispatchRoPEForwardTraining(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        freqs_buf: id,
        sin_cache_buf: id,
        cos_cache_buf: id,
        seq_len: u32,
        n_heads: u32,
        half_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = RoPETrainParams{ .seq_len = seq_len, .n_heads = n_heads, .half_dim = half_dim };
        setPipeline(encoder, tp.rope_forward_training);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, freqs_buf, 0, 1);
        setBuffer(encoder, sin_cache_buf, 0, 2);
        setBuffer(encoder, cos_cache_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(RoPETrainParams), 4);
        const total: u64 = @as(u64, seq_len) * @as(u64, n_heads) * @as(u64, half_dim);
        dispatch1D(encoder, total, @min(total, 256));
    }

    pub fn dispatchRoPEBackward(
        self: *MetalContext,
        encoder: id,
        grad_buf: id,
        sin_cache_buf: id,
        cos_cache_buf: id,
        seq_len: u32,
        n_heads: u32,
        half_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = RoPETrainParams{ .seq_len = seq_len, .n_heads = n_heads, .half_dim = half_dim };
        setPipeline(encoder, tp.rope_backward);
        setBuffer(encoder, grad_buf, 0, 0);
        setBuffer(encoder, sin_cache_buf, 0, 1);
        setBuffer(encoder, cos_cache_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(RoPETrainParams), 3);
        const total: u64 = @as(u64, seq_len) * @as(u64, n_heads) * @as(u64, half_dim);
        dispatch1D(encoder, total, @min(total, 256));
    }

    pub fn dispatchDequantQ8BatchScaled(
        self: *MetalContext,
        encoder: id,
        weight_buf: id,
        token_ids_buf: id,
        output_buf: id,
        num_tokens: u32,
        embed_dim: u32,
        embed_scale: f32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = BatchedDequantParams{ .num_tokens = num_tokens, .embed_dim = embed_dim };
        setPipeline(encoder, tp.dequant_q8_0_batch_scaled);
        setBuffer(encoder, weight_buf, 0, 0);
        setBuffer(encoder, token_ids_buf, 0, 1);
        setBuffer(encoder, output_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(BatchedDequantParams), 3);
        setBytes(encoder, @ptrCast(&embed_scale), 4, 4);
        const total: u64 = @as(u64, num_tokens) * @as(u64, embed_dim);
        dispatch1D(encoder, total, @min(total, 256));
    }

    // ============================================================
    // Phase 4: Sequence dispatch functions
    // ============================================================

    pub fn dispatchTanhForward(
        self: *MetalContext,
        encoder: id,
        x_buf: id,
        out_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.tanh_forward);
        setBuffer(encoder, x_buf, 0, 0);
        setBuffer(encoder, out_buf, 0, 1);
        setBytes(encoder, @ptrCast(&count), 4, 2);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    pub fn dispatchTanhBackward(
        self: *MetalContext,
        encoder: id,
        out_buf: id,
        grad_out_buf: id,
        grad_in_buf: id,
        count: u32,
    ) void {
        const tp = self.training_pipelines.?;
        setPipeline(encoder, tp.tanh_backward);
        setBuffer(encoder, out_buf, 0, 0);
        setBuffer(encoder, grad_out_buf, 0, 1);
        setBuffer(encoder, grad_in_buf, 0, 2);
        setBytes(encoder, @ptrCast(&count), 4, 3);
        dispatch1D(encoder, count, @min(@as(u64, count), 256));
    }

    const ConcatParams = extern struct { rows: u32, cols_a: u32, cols_b: u32 };

    pub fn dispatchConcatLastDim(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        out_buf: id,
        rows: u32,
        cols_a: u32,
        cols_b: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = ConcatParams{ .rows = rows, .cols_a = cols_a, .cols_b = cols_b };
        setPipeline(encoder, tp.concat_last_dim);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, out_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(ConcatParams), 3);
        const cols_total: u64 = @as(u64, cols_a) + @as(u64, cols_b);
        dispatchThreadgroups(encoder, .{ .width = @intCast((cols_total + 15) / 16), .height = @intCast((@as(u64, rows) + 15) / 16), .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchConcatLastDimBackward(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        grad_a_buf: id,
        grad_b_buf: id,
        rows: u32,
        cols_a: u32,
        cols_b: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = ConcatParams{ .rows = rows, .cols_a = cols_a, .cols_b = cols_b };
        setPipeline(encoder, tp.concat_last_dim_backward);
        setBuffer(encoder, grad_out_buf, 0, 0);
        setBuffer(encoder, grad_a_buf, 0, 1);
        setBuffer(encoder, grad_b_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(ConcatParams), 3);
        const cols_total: u64 = @as(u64, cols_a) + @as(u64, cols_b);
        dispatchThreadgroups(encoder, .{ .width = @intCast((cols_total + 15) / 16), .height = @intCast((@as(u64, rows) + 15) / 16), .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    // ============================================================
    // Phase 5: Batched Matmul Dispatch 関数群
    // ============================================================

    const TrainingBatchedMatmulParams = extern struct { batch: u32, M: u32, K: u32, N: u32 };

    pub fn dispatchBatchedMatmulF32(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        c_buf: id,
        batch: u32,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBatchedMatmulParams{ .batch = batch, .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.batched_matmul_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, c_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBatchedMatmulParams), 3);
        const gx: u64 = (@as(u64, n_dim) + 15) / 16;
        const gy: u64 = (@as(u64, m_dim) + 15) / 16;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = batch }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchBatchedMatmulTransBF32(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        c_buf: id,
        batch: u32,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBatchedMatmulParams{ .batch = batch, .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.batched_matmul_trans_b_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, c_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBatchedMatmulParams), 3);
        const gx: u64 = (@as(u64, n_dim) + 15) / 16;
        const gy: u64 = (@as(u64, m_dim) + 15) / 16;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = batch }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchBatchedMatmulBackwardA(
        self: *MetalContext,
        encoder: id,
        dc_buf: id,
        b_buf: id,
        da_buf: id,
        batch: u32,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBatchedMatmulParams{ .batch = batch, .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.batched_matmul_backward_a_f32);
        setBuffer(encoder, dc_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, da_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBatchedMatmulParams), 3);
        const gx: u64 = (@as(u64, k_dim) + 15) / 16;
        const gy: u64 = (@as(u64, m_dim) + 15) / 16;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = batch }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchBatchedMatmulBackwardB(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        dc_buf: id,
        db_buf: id,
        batch: u32,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBatchedMatmulParams{ .batch = batch, .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.batched_matmul_backward_b_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, dc_buf, 0, 1);
        setBuffer(encoder, db_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBatchedMatmulParams), 3);
        const gx: u64 = (@as(u64, n_dim) + 15) / 16;
        const gy: u64 = (@as(u64, k_dim) + 15) / 16;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = batch }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchBatchedMatmulTransBBackwardA(
        self: *MetalContext,
        encoder: id,
        dc_buf: id,
        b_buf: id,
        da_buf: id,
        batch: u32,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBatchedMatmulParams{ .batch = batch, .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.batched_matmul_trans_b_backward_a_f32);
        setBuffer(encoder, dc_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, da_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBatchedMatmulParams), 3);
        const gx: u64 = (@as(u64, k_dim) + 15) / 16;
        const gy: u64 = (@as(u64, m_dim) + 15) / 16;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = batch }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchBatchedMatmulTransBBackwardB(
        self: *MetalContext,
        encoder: id,
        dc_buf: id,
        a_buf: id,
        db_buf: id,
        batch: u32,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBatchedMatmulParams{ .batch = batch, .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.batched_matmul_trans_b_backward_b_f32);
        setBuffer(encoder, dc_buf, 0, 0);
        setBuffer(encoder, a_buf, 0, 1);
        setBuffer(encoder, db_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBatchedMatmulParams), 3);
        const gx: u64 = (@as(u64, k_dim) + 15) / 16;
        const gy: u64 = (@as(u64, n_dim) + 15) / 16;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = batch }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    // ============================================================
    // Fused Kernel Dispatches
    // ============================================================

    const TrainingFusedMatmulBiasParams = extern struct {
        M: u32,
        K: u32,
        N: u32,
    };

    const TrainingBatchedMatmulScaleParams = extern struct {
        batch: u32,
        M: u32,
        K: u32,
        N: u32,
        scale: f32,
    };

    pub fn dispatchMatmulAddbiasGeluF32(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        bias_buf: id,
        out_buf: id,
        pre_act_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingFusedMatmulBiasParams{ .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.matmul_addbias_gelu_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, bias_buf, 0, 2);
        setBuffer(encoder, out_buf, 0, 3);
        setBuffer(encoder, pre_act_buf, 0, 4);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingFusedMatmulBiasParams), 5);
        const gx: u64 = (@as(u64, n_dim) + 63) / 64;
        const gy: u64 = (@as(u64, m_dim) + 63) / 64;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchGeluBiasBackward(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        pre_act_buf: id,
        grad_pre_act_buf: id,
        grad_bias_buf: id,
        rows: u32,
        cols: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBiasParams{ .rows = rows, .cols = cols };
        setPipeline(encoder, tp.gelu_bias_backward);
        setBuffer(encoder, grad_out_buf, 0, 0);
        setBuffer(encoder, pre_act_buf, 0, 1);
        setBuffer(encoder, grad_pre_act_buf, 0, 2);
        setBuffer(encoder, grad_bias_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBiasParams), 4);
        dispatchThreadgroups(encoder, .{ .width = (@as(u64, cols) + 255) / 256, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
    }

    pub fn dispatchMatmulAddbiasGeluBackward(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        pre_act_buf: id,
        a_buf: id,
        b_buf: id,
        grad_pre_act_buf: id,
        grad_a_buf: id,
        grad_b_buf: id,
        grad_bias_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        // 1. gelu_bias_backward: grad_pre_act + grad_bias
        self.dispatchGeluBiasBackward(encoder, grad_out_buf, pre_act_buf, grad_pre_act_buf, grad_bias_buf, m_dim, n_dim);
        memoryBarrier(encoder);
        // 2. matmul_backward_a: grad_A += grad_pre_act @ B^T
        self.dispatchMatmulBackwardA(encoder, grad_pre_act_buf, b_buf, grad_a_buf, m_dim, k_dim, n_dim);
        memoryBarrier(encoder);
        // 3. matmul_backward_b: grad_B += A^T @ grad_pre_act
        self.dispatchMatmulBackwardB(encoder, a_buf, grad_pre_act_buf, grad_b_buf, m_dim, k_dim, n_dim);
    }

    pub fn dispatchMatmulAddbiasGeluBackwardNoGradA(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        pre_act_buf: id,
        a_buf: id,
        _: id,
        grad_pre_act_buf: id,
        grad_b_buf: id,
        grad_bias_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        self.dispatchGeluBiasBackward(encoder, grad_out_buf, pre_act_buf, grad_pre_act_buf, grad_bias_buf, m_dim, n_dim);
        memoryBarrier(encoder);
        self.dispatchMatmulBackwardB(encoder, a_buf, grad_pre_act_buf, grad_b_buf, m_dim, k_dim, n_dim);
    }

    pub fn dispatchMatmulAddbiasGeluBackwardGradAOnly(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        pre_act_buf: id,
        b_buf: id,
        grad_pre_act_buf: id,
        grad_a_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        self.dispatchGeluBiasBackward(encoder, grad_out_buf, pre_act_buf, grad_pre_act_buf,
            // no bias grad needed, but we still pass a dummy — use pre_act_buf as scratch
            // Actually we need a proper buffer. Let's skip bias in this variant.
            // For now, use the full backward.
            grad_pre_act_buf, // dummy - will be overwritten anyway
            m_dim, n_dim);
        memoryBarrier(encoder);
        self.dispatchMatmulBackwardA(encoder, grad_pre_act_buf, b_buf, grad_a_buf, m_dim, k_dim, n_dim);
    }

    pub fn dispatchMatmulAddbiasTanhF32(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        bias_buf: id,
        out_buf: id,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingFusedMatmulBiasParams{ .M = m_dim, .K = k_dim, .N = n_dim };
        setPipeline(encoder, tp.matmul_addbias_tanh_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, bias_buf, 0, 2);
        setBuffer(encoder, out_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingFusedMatmulBiasParams), 4);
        const gx: u64 = (@as(u64, n_dim) + 63) / 64;
        const gy: u64 = (@as(u64, m_dim) + 63) / 64;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = 1 }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    pub fn dispatchTanhBiasBackward(
        self: *MetalContext,
        encoder: id,
        grad_out_buf: id,
        tanh_out_buf: id,
        grad_pre_act_buf: id,
        grad_bias_buf: id,
        rows: u32,
        cols: u32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBiasParams{ .rows = rows, .cols = cols };
        setPipeline(encoder, tp.tanh_bias_backward);
        setBuffer(encoder, grad_out_buf, 0, 0);
        setBuffer(encoder, tanh_out_buf, 0, 1);
        setBuffer(encoder, grad_pre_act_buf, 0, 2);
        setBuffer(encoder, grad_bias_buf, 0, 3);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBiasParams), 4);
        dispatchThreadgroups(encoder, .{ .width = (@as(u64, cols) + 255) / 256, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
    }

    pub fn dispatchBatchedMatmulTransBScaleF32(
        self: *MetalContext,
        encoder: id,
        a_buf: id,
        b_buf: id,
        c_buf: id,
        batch: u32,
        m_dim: u32,
        k_dim: u32,
        n_dim: u32,
        scale_val: f32,
    ) void {
        const tp = self.training_pipelines.?;
        const params = TrainingBatchedMatmulScaleParams{ .batch = batch, .M = m_dim, .K = k_dim, .N = n_dim, .scale = scale_val };
        setPipeline(encoder, tp.batched_matmul_trans_b_scale_f32);
        setBuffer(encoder, a_buf, 0, 0);
        setBuffer(encoder, b_buf, 0, 1);
        setBuffer(encoder, c_buf, 0, 2);
        setBytes(encoder, @ptrCast(&params), @sizeOf(TrainingBatchedMatmulScaleParams), 3);
        const gx: u64 = (@as(u64, n_dim) + 15) / 16;
        const gy: u64 = (@as(u64, m_dim) + 15) / 16;
        dispatchThreadgroups(encoder, .{ .width = gx, .height = gy, .depth = batch }, .{ .width = 16, .height = 16, .depth = 1 });
    }

    // ============================================================
    // MPS MatrixMultiplication
    // ============================================================

    // objc_msgSend variants for MPS
    // (id, SEL, u64, u64, u64, u64) → id : MPSMatrixDescriptor
    fn send4uuuu(target: id, selector: SEL, a1: NSUInteger, a2: NSUInteger, a3: NSUInteger, a4: NSUInteger) ?*anyopaque {
        const F = *const fn (id, SEL, NSUInteger, NSUInteger, NSUInteger, NSUInteger) callconv(.c) ?*anyopaque;
        return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3, a4);
    }

    // (id, SEL, id, BOOL, BOOL, u64, u64, u64, f64, f64) → id : MPSMatrixMultiplication initWithDevice:...
    fn sendMpsMatmulInit(target: id, selector: SEL, device: id, transpose_left: bool, transpose_right: bool, result_rows: NSUInteger, result_columns: NSUInteger, interior_columns: NSUInteger, alpha: f64, beta: f64) ?*anyopaque {
        const F = *const fn (id, SEL, id, u8, u8, NSUInteger, NSUInteger, NSUInteger, f64, f64) callconv(.c) ?*anyopaque;
        return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, device, @intFromBool(transpose_left), @intFromBool(transpose_right), result_rows, result_columns, interior_columns, alpha, beta);
    }

    // (id, SEL, id, id, id, id) → void : encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:
    fn sendMpsEncode(target: id, selector: SEL, cmd_buf: id, left: id, right: id, result: id) void {
        const F = *const fn (id, SEL, id, id, id, id) callconv(.c) void;
        @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, cmd_buf, left, right, result);
    }

    /// MPSMatrix を作成 (buffer + descriptor)
    fn createMPSMatrix(buf: id, rows: NSUInteger, columns: NSUInteger) id {
        const MPSMatrixDescriptor = getClass("MPSMatrixDescriptor");
        const row_bytes = columns * @sizeOf(f32);
        // matrixDescriptorWithRows:columns:rowBytes:dataType:
        // MPSDataTypeFloat32 = 0x10000000 | 32 = 268435488
        const desc = send4uuuu(
            MPSMatrixDescriptor,
            sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"),
            rows,
            columns,
            row_bytes,
            268435488, // MPSDataTypeFloat32
        ).?;

        const MPSMatrix = getClass("MPSMatrix");
        const alloc_obj = send0(MPSMatrix, sel("alloc")).?;
        // initWithBuffer:descriptor:
        const matrix = send2ii(alloc_obj, sel("initWithBuffer:descriptor:"), buf, desc).?;
        return matrix;
    }

    // (id, SEL, id, id) → id
    fn send2ii(target: id, selector: SEL, a1: id, a2: id) ?*anyopaque {
        const F = *const fn (id, SEL, id, id) callconv(.c) ?*anyopaque;
        return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2);
    }

    /// MPS matmul: C = alpha * op(A) * op(B) + beta * C
    /// C: (result_rows × result_cols)
    /// op(A): (result_rows × interior_cols), op(B): (interior_cols × result_cols)
    /// A storage: transL=false → (result_rows, interior_cols), transL=true → (interior_cols, result_rows)
    /// B storage: transR=false → (interior_cols, result_cols), transR=true → (result_cols, interior_cols)
    /// batch mode 時は encoder を一時停止して MPS encode → 再開
    pub fn dispatchMPSMatmul(
        self: *MetalContext,
        a_buf: id,
        b_buf: id,
        c_buf: id,
        result_rows: u32,
        result_cols: u32,
        interior_cols: u32,
        transpose_left: bool,
        transpose_right: bool,
        alpha: f64,
        beta: f64,
    ) void {
        const rr: NSUInteger = @intCast(result_rows);
        const rc: NSUInteger = @intCast(result_cols);
        const ic: NSUInteger = @intCast(interior_cols);

        // A のストレージ次元 (メモリレイアウト)
        const a_store_rows: NSUInteger = if (transpose_left) ic else rr;
        const a_store_cols: NSUInteger = if (transpose_left) rr else ic;
        // B のストレージ次元 (メモリレイアウト)
        const b_store_rows: NSUInteger = if (transpose_right) rc else ic;
        const b_store_cols: NSUInteger = if (transpose_right) ic else rc;

        const mat_a = createMPSMatrix(a_buf, a_store_rows, a_store_cols);
        const mat_b = createMPSMatrix(b_buf, b_store_rows, b_store_cols);
        const mat_c = createMPSMatrix(c_buf, rr, rc);

        // MPSMatrixMultiplication: キャッシュから取得 or 新規作成
        const cache_key = MPSCacheKey{
            .result_rows = result_rows,
            .result_cols = result_cols,
            .interior_cols = interior_cols,
            .transpose_left = transpose_left,
            .transpose_right = transpose_right,
            .beta_is_one = (beta == 1.0),
        };
        const mps_mul = if (self.mps_cache) |*cache| blk: {
            if (cache.get(cache_key)) |cached| {
                break :blk cached;
            }
            const MPSMatrixMultiplication = getClass("MPSMatrixMultiplication");
            const alloc_obj = send0(MPSMatrixMultiplication, sel("alloc")).?;
            const new_mul = sendMpsMatmulInit(
                alloc_obj,
                sel("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:"),
                self.device,
                transpose_left,
                transpose_right,
                rr,
                rc,
                ic,
                alpha,
                beta,
            ).?;
            cache.put(cache_key, new_mul) catch {};
            break :blk new_mul;
        } else blk: {
            const MPSMatrixMultiplication = getClass("MPSMatrixMultiplication");
            const alloc_obj = send0(MPSMatrixMultiplication, sel("alloc")).?;
            break :blk sendMpsMatmulInit(
                alloc_obj,
                sel("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:"),
                self.device,
                transpose_left,
                transpose_right,
                rr,
                rc,
                ic,
                alpha,
                beta,
            ).?;
        };

        // Encoder 切り替え: batch mode 時は encoder を endEncoding → MPS encode → 新 encoder
        const encode_sel = sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:");
        if (self.profile_mode and self.batch_encoder != null) {
            // Profile mode: 前のカテゴリを flush → MPS をカテゴリとして開始
            const mps_cat_id: u8 = 255 - 1; // special: mps_matmul
            if (self.profile_current_cat != mps_cat_id) {
                self.profileFlush();
                self.profile_current_cat = mps_cat_id;
                self.profile_timer = Timer.start() catch null;
            }
            // MPS encode (通常の batch mode と同じパス)
            memoryBarrier(self.batch_encoder.?);
            endEncoding(self.batch_encoder.?);
            sendMpsEncode(mps_mul, encode_sel, self.batch_cmd_buf.?, mat_a, mat_b, mat_c);
            self.batch_encoder = newComputeEncoder(self.batch_cmd_buf.?);
            self.profile_stats.mps_matmul_count += 1;
        } else if (self.batch_encoder) |encoder| {
            memoryBarrier(encoder);
            endEncoding(encoder);
            sendMpsEncode(mps_mul, encode_sel, self.batch_cmd_buf.?, mat_a, mat_b, mat_c);
            self.batch_encoder = newComputeEncoder(self.batch_cmd_buf.?);
        } else {
            const cmd_buf = self.newCommandBuffer();
            sendMpsEncode(mps_mul, encode_sel, cmd_buf, mat_a, mat_b, mat_c);
            commit(cmd_buf);
            waitUntilCompleted(cmd_buf);
        }

        // MPSMatrix は毎回 release (バッファが異なる)
        // MPSMatrixMultiplication はキャッシュ時は release しない
        if (self.mps_cache == null) {
            objRelease(mps_mul);
        }
        objRelease(mat_a);
        objRelease(mat_b);
        objRelease(mat_c);
    }
};

// ============================================================
// 量子化型
// ============================================================

pub const QuantType = enum {
    q4_0,
    q4_1,
    q8_0,
};
