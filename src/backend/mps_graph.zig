/// MPSGraph Wrapper
///
/// Objective-C ランタイム経由で MetalPerformanceShadersGraph API を呼び出す。
/// MPSGraph でコンパイル済みグラフを構築し、forward+auto-diff+Adam を
/// 1回の `run` で実行する。
const std = @import("std");
const Allocator = std.mem.Allocator;
const metal = @import("metal.zig");

const id = metal.id;

// Objective-C ランタイム
const objc = @cImport({
    @cInclude("objc/runtime.h");
    @cInclude("objc/message.h");
});

const objc_msgSend_ptr = @extern(*const anyopaque, .{ .name = "objc_msgSend" });

const NSUInteger = u64;
const NSInteger = i64;

// ============================================================
// Objective-C ランタイムヘルパー
// ============================================================

fn sel(name: [*:0]const u8) *anyopaque {
    return objc.sel_registerName(name).?;
}

fn getClass(name: [*:0]const u8) id {
    return objc.objc_getClass(name).?;
}

// 引数0個 → id
fn send0(target: id, selector: *anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector);
}

// 引数0個 → void
fn send0v(target: id, selector: *anyopaque) void {
    const F = *const fn (id, *anyopaque) callconv(.c) void;
    @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector);
}

// 引数1個 (id) → id
fn send1(target: id, selector: *anyopaque, a1: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1);
}

// 引数1個 (cstr) → id
fn send1s(target: id, selector: *anyopaque, a1: [*]const u8) ?*anyopaque {
    const F = *const fn (id, *anyopaque, [*]const u8) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1);
}

// 引数2個 (id, id) → id
fn send2(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2);
}

// 引数3個 (id, id, id) → id
fn send3(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: ?*anyopaque, a3: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数4個 (id, id, id, id) → id
fn send4(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: ?*anyopaque, a3: ?*anyopaque, a4: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3, a4);
}

// 引数2個 (id, NSInteger) → id  (softmax axis, transpose dim, reductionSum axis)
fn send_id_i64(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: NSInteger) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, NSInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2);
}

// 引数3個 (id, NSInteger, id) → id  (softmax with axis and name)
fn send_id_i64_id(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: NSInteger, a3: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, NSInteger, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数3個 (id, NSUInteger, NSUInteger) → id  (transpose dim:withDim:)
fn send_id_u64_u64(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: NSUInteger, a3: NSUInteger) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, NSUInteger, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数4個 (id, NSUInteger, NSUInteger, id) → id  (transpose with name)
fn send_id_u64_u64_id(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: NSUInteger, a3: NSUInteger, a4: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, NSUInteger, NSUInteger, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3, a4);
}

// 引数3個 (id, id, NSInteger) → id  (concat dim)
fn send_id_id_i64(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: ?*anyopaque, a3: NSInteger) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, NSInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数4個 (id, id, NSInteger, id) → id  (concat with name)
fn send_id_id_i64_id(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: ?*anyopaque, a3: NSInteger, a4: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, NSInteger, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3, a4);
}

// 引数2個 (ptr, NSUInteger) → id  (dataWithBytes:length:, arrayWithObjects:count:)
fn send_ptr_u64(target: id, selector: *anyopaque, a1: *const anyopaque, a2: NSUInteger) ?*anyopaque {
    const F = *const fn (id, *anyopaque, *const anyopaque, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2);
}

// 引数3個 (id, NSUInteger, id) → id  (placeholderWithShape:dataType:name:)
fn send_id_u64_id(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: NSUInteger, a3: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, NSUInteger, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数3個 (id, id, NSUInteger) → id  (constantWithData:shape:dataType: with id args)
fn send_id_id_u64(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: ?*anyopaque, a3: NSUInteger) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3);
}

// 引数4個 (id, id, NSInteger, NSInteger, id) → id  (gather with axis, batchDims, name)
fn send_id_id_i64_i64_id(target: id, selector: *anyopaque, a1: ?*anyopaque, a2: ?*anyopaque, a3: NSInteger, a4: NSInteger, a5: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, NSInteger, NSInteger, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, a1, a2, a3, a4, a5);
}

// 引数6個: SDPA (id, id, id, id, f32, id) → id
fn send_sdpa(target: id, selector: *anyopaque, q: ?*anyopaque, k: ?*anyopaque, v: ?*anyopaque, mask: ?*anyopaque, scale_val: f32, name: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, f32, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, q, k, v, mask, scale_val, name);
}

// 引数6個: normalization(tensor, mean, var, gamma, beta, eps, name) → id
fn send_norm(target: id, selector: *anyopaque, tensor: ?*anyopaque, mean_t: ?*anyopaque, var_t: ?*anyopaque, gamma: ?*anyopaque, beta: ?*anyopaque, eps: f32, name: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, f32, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, tensor, mean_t, var_t, gamma, beta, eps, name);
}

// crossEntropy: (tensor, labels, axis, reductionType, name) → id
fn send_ce(target: id, selector: *anyopaque, src: ?*anyopaque, labels: ?*anyopaque, axis: NSInteger, reduction: NSUInteger, name: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, NSInteger, NSUInteger, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, src, labels, axis, reduction, name);
}

// NSNumber numberWithFloat:
fn send_f32(target: id, selector: *anyopaque, val: f32) ?*anyopaque {
    const F = *const fn (id, *anyopaque, f32) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, val);
}

// NSNumber numberWithDouble:
fn send_f64(target: id, selector: *anyopaque, val: f64) ?*anyopaque {
    const F = *const fn (id, *anyopaque, f64) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, val);
}

// NSNumber numberWithUnsignedInteger:
fn send_u64(target: id, selector: *anyopaque, val: NSUInteger) ?*anyopaque {
    const F = *const fn (id, *anyopaque, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, val);
}

// NSNumber numberWithInteger:
fn send_i64(target: id, selector: *anyopaque, val: NSInteger) ?*anyopaque {
    const F = *const fn (id, *anyopaque, NSInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, val);
}

// MPSGraphTensorData initWithMTLBuffer:shape:dataType:
fn send_id_id_u64_init(target: id, selector: *anyopaque, buf: ?*anyopaque, shape_arr: ?*anyopaque, dtype: NSUInteger) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, buf, shape_arr, dtype);
}

// runWithMTLCommandQueue:feeds:targetTensors:targetOperations: → NSDictionary
fn send_run(target: id, selector: *anyopaque, queue: ?*anyopaque, feeds: ?*anyopaque, targets: ?*anyopaque, ops: ?*anyopaque) ?*anyopaque {
    const F = *const fn (id, *anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(target, selector, queue, feeds, targets, ops);
}

// ============================================================
// Foundation ヘルパー
// ============================================================

/// NSString from C string
fn nsString(str: [*:0]const u8) id {
    const NSString = getClass("NSString");
    return send1s(NSString, sel("stringWithUTF8String:"), str).?;
}

/// NSNumber from u64
fn nsNumberU64(val: u64) id {
    const NSNumber = getClass("NSNumber");
    return send_u64(NSNumber, sel("numberWithUnsignedInteger:"), val).?;
}

/// NSNumber from i64
fn nsNumberI64(val: i64) id {
    const NSNumber = getClass("NSNumber");
    return send_i64(NSNumber, sel("numberWithInteger:"), val).?;
}

/// NSNumber from f32
fn nsNumberF32(val: f32) id {
    const NSNumber = getClass("NSNumber");
    return send_f32(NSNumber, sel("numberWithFloat:"), val).?;
}

/// NSNumber from f64
fn nsNumberF64(val: f64) id {
    const NSNumber = getClass("NSNumber");
    return send_f64(NSNumber, sel("numberWithDouble:"), val).?;
}

/// NSArray from items
fn nsArray(items: []const id, allocator: Allocator) id {
    const ptrs = allocator.alloc(?*anyopaque, items.len) catch unreachable;
    defer allocator.free(ptrs);
    for (items, 0..) |item, i| {
        ptrs[i] = item;
    }
    return nsArrayFromPtrs(ptrs.ptr, items.len);
}

/// NSData from bytes
fn nsData(ptr: [*]const u8, length: usize) id {
    const NSData = getClass("NSData");
    return send_ptr_u64(
        NSData,
        sel("dataWithBytes:length:"),
        @ptrCast(ptr),
        length,
    ) orelse unreachable;
}

/// NSDictionary from keys+values
fn nsDictionary(keys: []const id, vals: []const id, allocator: Allocator) id {
    const NSDictionary = getClass("NSDictionary");
    const k_ptrs = allocator.alloc(?*anyopaque, keys.len) catch unreachable;
    defer allocator.free(k_ptrs);
    const v_ptrs = allocator.alloc(?*anyopaque, vals.len) catch unreachable;
    defer allocator.free(v_ptrs);
    for (keys, 0..) |key, i| k_ptrs[i] = key;
    for (vals, 0..) |val, i| v_ptrs[i] = val;
    // dictionaryWithObjects:forKeys:count:
    const F = *const fn (id, *anyopaque, *const anyopaque, *const anyopaque, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(
        NSDictionary,
        sel("dictionaryWithObjects:forKeys:count:"),
        @ptrCast(v_ptrs.ptr),
        @ptrCast(k_ptrs.ptr),
        keys.len,
    ) orelse unreachable;
}

/// NSArray of NSNumber from shape
fn nsShapeArray(shape: []const usize, allocator: Allocator) id {
    const nums = allocator.alloc(id, shape.len) catch unreachable;
    defer allocator.free(nums);
    for (shape, 0..) |dim, i| {
        nums[i] = nsNumberI64(@intCast(dim));
    }
    return nsArray(nums, allocator);
}

/// NSArray of NSNumber from i64 array
fn nsI64Array(vals: []const i64, allocator: Allocator) id {
    const nums = allocator.alloc(id, vals.len) catch unreachable;
    defer allocator.free(nums);
    for (vals, 0..) |val, i| {
        nums[i] = nsNumberI64(val);
    }
    return nsArray(nums, allocator);
}

/// NSArray with 2-arg arrayWithObjects:count: (correct signature)
fn nsArrayFromPtrs(ptrs: [*]const ?*anyopaque, count: usize) id {
    const NSArray = getClass("NSArray");
    const F = *const fn (id, *anyopaque, [*]const ?*anyopaque, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(
        NSArray,
        sel("arrayWithObjects:count:"),
        ptrs,
        count,
    ) orelse unreachable;
}

/// MPSGraphTensorData from MTLBuffer + shape
fn mpsTensorData(buf: id, shape: []const usize, dtype: u32, allocator: Allocator) id {
    const MPSGraphTensorData = getClass("MPSGraphTensorData");
    const alloc_obj = send0(MPSGraphTensorData, sel("alloc")).?;
    const shape_arr = nsShapeArray(shape, allocator);
    return send_id_id_u64_init(
        alloc_obj,
        sel("initWithMTLBuffer:shape:dataType:"),
        buf,
        shape_arr,
        dtype,
    ) orelse unreachable;
}

/// NSDictionary objectForKey:
pub fn dictGet(dict: id, key: id) ?id {
    return @as(?id, send1(dict, sel("objectForKey:"), key));
}

/// MTLBuffer contents
fn bufferContents(buf: id) [*]u8 {
    return @ptrCast(send0(buf, sel("contents")).?);
}

// ============================================================
// MPSDataType 定数
// ============================================================

pub const MPSDataTypeFloat32: u32 = 0x10000020; // 268435488
pub const MPSDataTypeFloat16: u32 = 0x10000010;
pub const MPSDataTypeInt32: u32 = 0x20000020;
pub const MPSDataTypeUInt32: u32 = 0x40000020;

// MPSGraphTensorNamedDataLayout
// MPSGraphReductionMode
const MPSGraphReductionModeMean: NSUInteger = 1;

// ============================================================
// MPSGraphContext
// ============================================================

pub const MPSGraphContext = struct {
    graph: id, // MPSGraph
    device: id, // MTLDevice
    queue: id, // MTLCommandQueue
    allocator: Allocator,

    pub fn init(device: id, queue: id, allocator: Allocator) MPSGraphContext {
        const MPSGraph = getClass("MPSGraph");
        const alloc_obj = send0(MPSGraph, sel("alloc")).?;
        const graph = send0(alloc_obj, sel("init")).?;

        return .{
            .graph = graph,
            .device = device,
            .queue = queue,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MPSGraphContext) void {
        metal.objRelease(self.graph);
    }

    // ============================================================
    // Placeholder / Constant
    // ============================================================

    /// placeholder with shape and dtype
    pub fn placeholder(self: *MPSGraphContext, shape: []const usize, dtype: u32) id {
        const shape_arr = nsShapeArray(shape, self.allocator);
        // placeholderWithShape:dataType:name: takes (NSArray*, MPSDataType, NSString*)
        return send_id_u64_id(
            self.graph,
            sel("placeholderWithShape:dataType:name:"),
            shape_arr,
            dtype,
            null, // name
        ) orelse unreachable;
    }

    /// constant scalar
    pub fn constantScalar(self: *MPSGraphContext, val: f64, dtype: u32) id {
        // constantWithScalar:dataType:
        const F = *const fn (id, *anyopaque, f64, NSUInteger) callconv(.c) ?*anyopaque;
        return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(
            self.graph,
            sel("constantWithScalar:dataType:"),
            val,
            dtype,
        ) orelse unreachable;
    }

    /// constant from data buffer
    pub fn constantData(self: *MPSGraphContext, data: [*]const u8, len: usize, shape: []const usize, dtype: u32) id {
        const ns_data = nsData(data, len);
        const shape_arr = nsShapeArray(shape, self.allocator);
        // constantWithData:shape:dataType: takes (NSData*, NSArray*, MPSDataType)
        return send_id_id_u64(
            self.graph,
            sel("constantWithData:shape:dataType:"),
            ns_data,
            shape_arr,
            dtype,
        ) orelse unreachable;
    }

    // ============================================================
    // Binary ops
    // ============================================================

    pub fn matmul(self: *MPSGraphContext, a: id, b: id) id {
        return send3(self.graph, sel("matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:"), a, b, null) orelse unreachable;
    }

    pub fn add(self: *MPSGraphContext, a: id, b: id) id {
        return send3(self.graph, sel("additionWithPrimaryTensor:secondaryTensor:name:"), a, b, null) orelse unreachable;
    }

    pub fn mul(self: *MPSGraphContext, a: id, b: id) id {
        return send3(self.graph, sel("multiplicationWithPrimaryTensor:secondaryTensor:name:"), a, b, null) orelse unreachable;
    }

    pub fn sub(self: *MPSGraphContext, a: id, b: id) id {
        return send3(self.graph, sel("subtractionWithPrimaryTensor:secondaryTensor:name:"), a, b, null) orelse unreachable;
    }

    pub fn div(self: *MPSGraphContext, a: id, b: id) id {
        return send3(self.graph, sel("divisionWithPrimaryTensor:secondaryTensor:name:"), a, b, null) orelse unreachable;
    }

    // ============================================================
    // Unary ops
    // ============================================================

    pub fn tanh_(self: *MPSGraphContext, x: id) id {
        return send2(self.graph, sel("tanhWithTensor:name:"), x, null) orelse unreachable;
    }

    pub fn sigmoid(self: *MPSGraphContext, x: id) id {
        return send2(self.graph, sel("sigmoidWithTensor:name:"), x, null) orelse unreachable;
    }

    pub fn erf(self: *MPSGraphContext, x: id) id {
        return send2(self.graph, sel("erfWithTensor:name:"), x, null) orelse unreachable;
    }

    pub fn sqrt_(self: *MPSGraphContext, x: id) id {
        return send2(self.graph, sel("squareRootWithTensor:name:"), x, null) orelse unreachable;
    }

    pub fn square(self: *MPSGraphContext, x: id) id {
        return send2(self.graph, sel("squareWithTensor:name:"), x, null) orelse unreachable;
    }

    pub fn maximum(self: *MPSGraphContext, a: id, b: id) id {
        return send3(self.graph, sel("maximumWithPrimaryTensor:secondaryTensor:name:"), a, b, null) orelse unreachable;
    }

    pub fn negative(self: *MPSGraphContext, x: id) id {
        return send2(self.graph, sel("negativeWithTensor:name:"), x, null) orelse unreachable;
    }

    pub fn exp(self: *MPSGraphContext, x: id) id {
        return send2(self.graph, sel("exponentWithTensor:name:"), x, null) orelse unreachable;
    }

    // ============================================================
    // Composite activations
    // ============================================================

    /// GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    /// Uses tanh approximation to ensure auto-diff compatibility (erf gradient may not be implemented)
    pub fn gelu(self: *MPSGraphContext, x: id) id {
        const half = self.constantScalar(0.5, MPSDataTypeFloat32);
        const one = self.constantScalar(1.0, MPSDataTypeFloat32);
        const coeff = self.constantScalar(0.044715, MPSDataTypeFloat32);
        const sqrt_2_over_pi = self.constantScalar(std.math.sqrt(2.0 / std.math.pi), MPSDataTypeFloat32);
        // x^3
        const x3 = self.mul(self.mul(x, x), x);
        // inner = sqrt(2/π) * (x + 0.044715 * x^3)
        const inner = self.mul(sqrt_2_over_pi, self.add(x, self.mul(coeff, x3)));
        // 0.5 * x * (1 + tanh(inner))
        return self.mul(half, self.mul(x, self.add(one, self.tanh_(inner))));
    }

    /// Dropout: dropoutTensor:rateTensor:name: (inverted dropout, auto-diff 対応)
    pub fn dropout(self: *MPSGraphContext, x: id, rate_tensor: id) id {
        return send3(self.graph, sel("dropoutTensor:rateTensor:name:"), x, rate_tensor, null) orelse unreachable;
    }

    /// SiLU: x * sigmoid(x)
    pub fn silu(self: *MPSGraphContext, x: id) id {
        const sig = self.sigmoid(x);
        return self.mul(x, sig);
    }

    // ============================================================
    // Activation / Softmax
    // ============================================================

    pub fn softmax(self: *MPSGraphContext, x: id, axis: i64) id {
        return send_id_i64_id(self.graph, sel("softMaxWithTensor:axis:name:"), x, axis, null) orelse unreachable;
    }

    /// LogSoftmax: log(softmax(x, axis)) with eps for numerical stability
    pub fn logSoftmax(self: *MPSGraphContext, x: id, axis: i64) id {
        const sm = self.softmax(x, axis);
        const eps = self.constantScalar(1e-10, MPSDataTypeFloat32);
        return self.log(self.add(sm, eps));
    }

    /// Natural logarithm
    pub fn log(self: *MPSGraphContext, x: id) id {
        return send2(self.graph, sel("logarithmWithTensor:name:"), x, null) orelse unreachable;
    }

    // ============================================================
    // SDPA (Scaled Dot-Product Attention)
    // ============================================================

    pub fn sdpa(self: *MPSGraphContext, q: id, k: id, v: id, mask: ?id, scale_val: f32) id {
        return send_sdpa(
            self.graph,
            sel("scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:"),
            q,
            k,
            v,
            mask,
            scale_val,
            null,
        ) orelse unreachable;
    }

    // ============================================================
    // Normalization
    // ============================================================

    /// LayerNorm: fully manual implementation using basic ops for auto-diff compatibility
    /// result = (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn layerNorm(self: *MPSGraphContext, x: id, gamma: id, beta: id, eps: f32, axis: i64) id {
        const axes_arr = nsI64Array(&.{axis}, self.allocator);
        // mean
        const mean_t = send3(self.graph, sel("meanOfTensor:axes:name:"), x, axes_arr, null) orelse unreachable;
        // variance = mean((x - mean)^2)
        const x_centered = self.sub(x, mean_t);
        const sq = self.square(x_centered);
        const var_t = send3(self.graph, sel("meanOfTensor:axes:name:"), sq, axes_arr, null) orelse unreachable;
        // x_norm = (x - mean) / sqrt(var + eps)
        const eps_c = self.constantScalar(@floatCast(eps), MPSDataTypeFloat32);
        const var_eps = self.add(var_t, eps_c);
        const std_dev = self.sqrt_(var_eps);
        const x_norm = self.div(x_centered, std_dev);
        // result = x_norm * gamma + beta
        return self.add(self.mul(x_norm, gamma), beta);
    }

    // ============================================================
    // Shape ops
    // ============================================================

    pub fn reshape(self: *MPSGraphContext, x: id, shape: []const usize) id {
        const shape_arr = nsShapeArray(shape, self.allocator);
        return send3(self.graph, sel("reshapeTensor:withShape:name:"), x, shape_arr, null) orelse unreachable;
    }

    pub fn transpose(self: *MPSGraphContext, x: id, dim1: u64, dim2: u64) id {
        return send_id_u64_u64_id(self.graph, sel("transposeTensor:dimension:withDimension:name:"), x, dim1, dim2, null) orelse unreachable;
    }

    pub fn concat(self: *MPSGraphContext, a: id, b: id, axis: i64) id {
        return send_id_id_i64_id(self.graph, sel("concatTensor:withTensor:dimension:name:"), a, b, axis, null) orelse unreachable;
    }

    // ============================================================
    // Embedding (gather)
    // ============================================================

    pub fn gather(self: *MPSGraphContext, table: id, indices: id, axis: i64) id {
        return send_id_id_i64_i64_id(
            self.graph,
            sel("gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:name:"),
            table,
            indices,
            axis,
            0, // batchDimensions
            null,
        ) orelse unreachable;
    }

    // ============================================================
    // Reduction
    // ============================================================

    pub fn reductionSum(self: *MPSGraphContext, x: id, axis: i64) id {
        return send_id_i64_id(self.graph, sel("reductionSumWithTensor:axis:name:"), x, axis, null) orelse unreachable;
    }

    pub fn reductionMax(self: *MPSGraphContext, x: id, axis: i64) id {
        return send_id_i64_id(self.graph, sel("reductionMaximumWithTensor:axis:name:"), x, axis, null) orelse unreachable;
    }

    pub fn reductionMean(self: *MPSGraphContext, x: id, axes: []const i64) id {
        const axes_arr = nsI64Array(axes, self.allocator);
        return send3(self.graph, sel("meanOfTensor:axes:name:"), x, axes_arr, null) orelse unreachable;
    }

    // ============================================================
    // Loss
    // ============================================================

    /// MSE loss = mean(square(pred - target))
    pub fn mseLoss(self: *MPSGraphContext, pred: id, target: id) id {
        const diff = self.sub(pred, target);
        const sq = self.square(diff);
        // Reduce all dimensions
        const flat = self.reshape(sq, &.{std.math.maxInt(usize)});
        _ = flat;
        // meanOfTensor:axes:name: with axis 0 on flattened
        // Actually just use full reduction on original tensor
        return self.reductionMean(sq, &.{ 0, 1 });
    }

    /// MSE loss for specific shape
    pub fn mseLossFlat(self: *MPSGraphContext, pred: id, target: id, total_elements: usize) id {
        const diff = self.sub(pred, target);
        const sq = self.square(diff);
        // Reshape to 1D and take mean
        const flat = self.reshape(sq, &.{total_elements});
        return self.reductionMean(flat, &.{0});
    }

    /// Cross-entropy loss
    pub fn crossEntropyLoss(self: *MPSGraphContext, logits: id, labels: id, axis: i64) id {
        // softMaxCrossEntropyWithSourceTensor:labelsTensor:axis:reductionType:name:
        return send_ce(
            self.graph,
            sel("softMaxCrossEntropyWithSourceTensor:labelsTensor:axis:reductionType:name:"),
            logits,
            labels,
            axis,
            MPSGraphReductionModeMean,
            null,
        ) orelse unreachable;
    }

    // NOTE: stopGradient (stopGradientWithTensor:name:) は MPSGraph API に存在しない。
    // auto-diff は gradients() に渡したパラメータのみ微分するため、
    // non-param placeholder (embedding table 等) への gradient flow は問題にならない。

    // ============================================================
    // Auto-diff
    // ============================================================

    /// Compute gradients dict (returns raw NSDictionary)
    pub fn gradientsDict(self: *MPSGraphContext, loss: id, params: []const id) id {
        const params_ptrs = self.allocator.alloc(?*anyopaque, params.len) catch unreachable;
        defer self.allocator.free(params_ptrs);
        for (params, 0..) |p, i| {
            params_ptrs[i] = p;
        }
        const params_arr = nsArrayFromPtrs(params_ptrs.ptr, params.len);
        return send3(
            self.graph,
            sel("gradientForPrimaryTensor:withTensors:name:"),
            loss,
            params_arr,
            null,
        ) orelse unreachable;
    }

    /// Compute gradients of loss with respect to params
    pub fn gradients(self: *MPSGraphContext, loss: id, params: []const id) []id {
        // Build NSArray of param tensors
        const params_ptrs = self.allocator.alloc(?*anyopaque, params.len) catch unreachable;
        defer self.allocator.free(params_ptrs);
        for (params, 0..) |p, i| {
            params_ptrs[i] = p;
        }
        const params_arr = nsArrayFromPtrs(params_ptrs.ptr, params.len);

        // gradientForPrimaryTensor:withTensors:name: → NSDictionary<MPSGraphTensor, MPSGraphTensor>
        const grad_dict = send3(
            self.graph,
            sel("gradientForPrimaryTensor:withTensors:name:"),
            loss,
            params_arr,
            null,
        ) orelse unreachable;

        // Extract gradient tensors in same order
        const result = self.allocator.alloc(id, params.len) catch unreachable;
        var missing_count: usize = 0;
        for (params, 0..) |p, i| {
            if (dictGet(grad_dict, p)) |grad| {
                result[i] = grad;
            } else {
                std.debug.print("WARNING: no gradient for param index {d} (placeholder ptr=0x{x})\n", .{ i, @intFromPtr(p) });
                missing_count += 1;
                result[i] = p; // placeholder — will not be used
            }
        }
        if (missing_count > 0) {
            std.debug.print("ERROR: {d}/{d} params have no gradient path to loss\n", .{ missing_count, params.len });
            @panic("MPSGraph auto-diff failed: some params not connected to loss");
        }
        return result;
    }

    // ============================================================
    // Execution
    // ============================================================

    pub const Feed = struct {
        tensor: id, // MPSGraphTensor (placeholder)
        buffer: id, // MTLBuffer
        shape: []const usize,
        dtype: u32,
    };

    /// Create an NSAutoreleasePool. Call drainAutoreleasePool() when done.
    /// Wrap trainStep or any code that calls run()/readTensorData() in a pool
    /// to prevent ObjC object leaks.
    pub fn createAutoreleasePool() id {
        return send0(send0(getClass("NSAutoreleasePool"), sel("alloc")).?, sel("init")).?;
    }

    /// Drain (release) an autorelease pool created by createAutoreleasePool().
    pub fn drainAutoreleasePool(pool: id) void {
        send0v(pool, sel("drain"));
    }

    /// Run the graph
    pub fn run(self: *MPSGraphContext, feeds: []const Feed, targets: []const id) []id {
        // Build feeds NSDictionary<MPSGraphTensor, MPSGraphTensorData>
        const n_feeds = feeds.len;
        const feed_keys = self.allocator.alloc(id, n_feeds) catch unreachable;
        defer self.allocator.free(feed_keys);
        const feed_vals = self.allocator.alloc(id, n_feeds) catch unreachable;
        defer self.allocator.free(feed_vals);

        for (feeds, 0..) |f, i| {
            feed_keys[i] = f.tensor;
            feed_vals[i] = mpsTensorData(f.buffer, f.shape, f.dtype, self.allocator);
        }
        const feeds_dict = nsDictionary(feed_keys, feed_vals, self.allocator);

        // Build targets NSArray
        const target_ptrs = self.allocator.alloc(?*anyopaque, targets.len) catch unreachable;
        defer self.allocator.free(target_ptrs);
        for (targets, 0..) |t, i| {
            target_ptrs[i] = t;
        }
        const targets_arr = nsArrayFromPtrs(target_ptrs.ptr, targets.len);

        // runWithMTLCommandQueue:feeds:targetTensors:targetOperations:
        const result_dict = send_run(
            self.graph,
            sel("runWithMTLCommandQueue:feeds:targetTensors:targetOperations:"),
            self.queue,
            feeds_dict,
            targets_arr,
            null, // no target operations
        ) orelse unreachable;

        // Extract results in same order as targets
        const result = self.allocator.alloc(id, targets.len) catch unreachable;
        for (targets, 0..) |t, i| {
            // result_dict: NSDictionary<MPSGraphTensor, MPSGraphTensorData>
            result[i] = dictGet(result_dict, t) orelse unreachable;
        }

        // Release mpsTensorData objects (alloc+init'd, each retains its MTLBuffer)
        for (feed_vals) |fv| metal.objRelease(fv);

        return result;
    }

    /// Get float data from MPSGraphTensorData result
    pub fn tensorDataToBuffer(self: *MPSGraphContext, tensor_data: id) id {
        _ = self;
        // mpsndarray → MTLBuffer via mpsndarray().buffer or just use UMA
        // For MPSGraphTensorData: use -mpsndarray to get MPSNDArray, then buffer
        const ndarray = send0(tensor_data, sel("mpsndarray")) orelse unreachable;
        _ = ndarray;
        // MPSNDArray doesn't have a direct buffer accessor.
        // Instead, read data via readBytes:strideBytes: or use MTLBuffer-backed tensor data.
        // Since we use MTLBuffer-backed feeds, results should also be MTLBuffer-backed.
        // Use readBytes approach for now.
        return tensor_data;
    }

    /// Read f32 values from MPSGraphTensorData into a slice
    pub fn readTensorData(tensor_data: id, out: []f32) void {
        // readBytes:strideBytes: on MPSNDArray
        const ndarray = send0(tensor_data, sel("mpsndarray")) orelse unreachable;
        const F = *const fn (id, *anyopaque, [*]u8, ?*anyopaque) callconv(.c) void;
        @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(
            ndarray,
            sel("readBytes:strideBytes:"),
            @ptrCast(out.ptr),
            null, // nil strideBytes = contiguous
        );
    }

    // ============================================================
    // Broadcast helper
    // ============================================================

    /// Broadcast add for (1, d) + (n, d) patterns
    /// MPSGraph の add はブロードキャスト対応なのでそのまま使える
    pub fn broadcastAdd(self: *MPSGraphContext, a: id, b: id) id {
        return self.add(a, b);
    }

    // ============================================================
    // One-hot encoding
    // ============================================================

    /// Create one-hot tensor for cross-entropy labels
    pub fn oneHot(self: *MPSGraphContext, indices: id, depth: usize) id {
        const depth_val = self.constantScalar(@floatFromInt(depth), MPSDataTypeFloat32);
        _ = depth_val;
        // MPSGraph doesn't have oneHot directly. Use scatter or manual construction.
        // For softMaxCrossEntropy, labels should be one-hot or class indices depending on API.
        // The MPSGraph softMaxCrossEntropyWithSourceTensor expects one-hot labels.
        // We need to construct one-hot from indices.

        // oneHotWithIndicesTensor:depth:axis:dataType:name: (available in MPSGraph)
        const F = *const fn (id, *anyopaque, ?*anyopaque, NSUInteger, NSInteger, NSUInteger, ?*anyopaque) callconv(.c) ?*anyopaque;
        return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(
            self.graph,
            sel("oneHotWithIndicesTensor:depth:axis:dataType:name:"),
            indices,
            depth,
            -1, // last axis
            MPSDataTypeFloat32,
            null,
        ) orelse unreachable;
    }

    // ============================================================
    // Cast
    // ============================================================

    pub fn cast(self: *MPSGraphContext, x: id, dtype: u32) id {
        const F = *const fn (id, *anyopaque, ?*anyopaque, NSUInteger, ?*anyopaque) callconv(.c) ?*anyopaque;
        return @as(F, @ptrCast(@alignCast(objc_msgSend_ptr)))(
            self.graph,
            sel("castTensor:toType:name:"),
            x,
            dtype,
            null,
        ) orelse unreachable;
    }

    // ============================================================
    // Tile (broadcast)
    // ============================================================

    pub fn tile(self: *MPSGraphContext, x: id, multiples: []const usize) id {
        const mult_arr = nsShapeArray(multiples, self.allocator);
        return send3(self.graph, sel("tileTensor:withMultiplier:name:"), x, mult_arr, null) orelse unreachable;
    }
};
