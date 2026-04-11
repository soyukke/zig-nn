/// CUDA GPU バックエンド
///
/// CUDA Driver API + cuBLAS 経由で NVIDIA GPU を利用する。
/// Metal バックエンドと対称なインターフェースを提供。
/// 現時点では骨格のみ（extern 宣言 + CudaContext 構造体）。
const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================
// CUDA Driver API 型定義
// ============================================================

pub const CUresult = c_int;
pub const CUdevice = c_int;
pub const CUcontext = *anyopaque;
pub const CUstream = *anyopaque;
pub const CUmodule = *anyopaque;
pub const CUfunction = *anyopaque;
pub const CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;

// ============================================================
// cuBLAS 型定義
// ============================================================

pub const cublasStatus_t = c_int;
pub const cublasHandle_t = *anyopaque;
pub const cublasOperation_t = c_int;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;

// ============================================================
// CUDA Driver API extern 宣言
// ============================================================

extern "c" fn cuInit(flags: c_uint) CUresult;
extern "c" fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
extern "c" fn cuDeviceGetName(name: [*]u8, len: c_int, dev: CUdevice) CUresult;
extern "c" fn cuCtxCreate_v2(ctx: *CUcontext, flags: c_uint, dev: CUdevice) CUresult;
extern "c" fn cuCtxDestroy_v2(ctx: CUcontext) CUresult;
extern "c" fn cuStreamCreate(stream: *CUstream, flags: c_uint) CUresult;
extern "c" fn cuStreamDestroy_v2(stream: CUstream) CUresult;
extern "c" fn cuStreamSynchronize(stream: CUstream) CUresult;
extern "c" fn cuMemAlloc_v2(dptr: *CUdeviceptr, size: usize) CUresult;
extern "c" fn cuMemFree_v2(dptr: CUdeviceptr) CUresult;
extern "c" fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const anyopaque, size: usize) CUresult;
extern "c" fn cuMemcpyDtoH_v2(dst: *anyopaque, src: CUdeviceptr, size: usize) CUresult;
extern "c" fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, size: usize) CUresult;
extern "c" fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) CUresult;
extern "c" fn cuModuleUnload(module: CUmodule) CUresult;
extern "c" fn cuModuleGetFunction(func: *CUfunction, module: CUmodule, name: [*:0]const u8) CUresult;
extern "c" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    stream: ?CUstream,
    kernelParams: ?[*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) CUresult;

// ============================================================
// cuBLAS extern 宣言
// ============================================================

extern "c" fn cublasCreate_v2(handle: *cublasHandle_t) cublasStatus_t;
extern "c" fn cublasDestroy_v2(handle: cublasHandle_t) cublasStatus_t;
extern "c" fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) cublasStatus_t;
extern "c" fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    A: CUdeviceptr,
    lda: c_int,
    B: CUdeviceptr,
    ldb: c_int,
    beta: *const f32,
    C: CUdeviceptr,
    ldc: c_int,
) cublasStatus_t;

// ============================================================
// CudaContext
// ============================================================

pub const CudaContext = struct {
    device: CUdevice,
    context: CUcontext,
    stream: CUstream,
    cublas_handle: cublasHandle_t,
    module: ?CUmodule,

    pub const CudaError = error{
        InitFailed,
        DeviceNotFound,
        ContextCreationFailed,
        StreamCreationFailed,
        CublasInitFailed,
        MemAllocFailed,
        MemcpyFailed,
        ModuleLoadFailed,
        KernelLaunchFailed,
        SgemmFailed,
    };

    /// GPU デバイスを初期化し、CUDA コンテキスト・ストリーム・cuBLAS ハンドルを作成する
    pub fn init(device_ordinal: c_int) CudaError!CudaContext {
        // CUDA 初期化
        if (cuInit(0) != CUDA_SUCCESS) return error.InitFailed;

        // デバイス取得
        var device: CUdevice = undefined;
        if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS) return error.DeviceNotFound;

        // デバイス名をログ出力
        var name_buf: [256]u8 = undefined;
        if (cuDeviceGetName(&name_buf, 256, device) == CUDA_SUCCESS) {
            const name_slice = std.mem.sliceTo(&name_buf, 0);
            std.debug.print("[CUDA] Device: {s}\n", .{name_slice});
        }

        // コンテキスト作成
        var ctx: CUcontext = undefined;
        if (cuCtxCreate_v2(&ctx, 0, device) != CUDA_SUCCESS) return error.ContextCreationFailed;

        // ストリーム作成
        var stream: CUstream = undefined;
        if (cuStreamCreate(&stream, 0) != CUDA_SUCCESS) return error.StreamCreationFailed;

        // cuBLAS 初期化
        var cublas: cublasHandle_t = undefined;
        if (cublasCreate_v2(&cublas) != CUBLAS_STATUS_SUCCESS) return error.CublasInitFailed;
        if (cublasSetStream_v2(cublas, stream) != CUBLAS_STATUS_SUCCESS) return error.CublasInitFailed;

        return CudaContext{
            .device = device,
            .context = ctx,
            .stream = stream,
            .cublas_handle = cublas,
            .module = null,
        };
    }

    /// リソースを解放する
    pub fn deinit(self: *CudaContext) void {
        _ = cublasDestroy_v2(self.cublas_handle);
        if (self.module) |m| _ = cuModuleUnload(m);
        _ = cuStreamDestroy_v2(self.stream);
        _ = cuCtxDestroy_v2(self.context);
    }

    /// GPU メモリを確保する
    pub fn allocBuffer(self: *CudaContext, size: usize) CudaError!CUdeviceptr {
        _ = self;
        var dptr: CUdeviceptr = undefined;
        if (cuMemAlloc_v2(&dptr, size) != CUDA_SUCCESS) return error.MemAllocFailed;
        return dptr;
    }

    /// GPU メモリを解放する
    pub fn freeBuffer(_: *CudaContext, dptr: CUdeviceptr) void {
        _ = cuMemFree_v2(dptr);
    }

    /// ホスト → デバイスへメモリコピー
    pub fn copyHostToDevice(_: *CudaContext, dst: CUdeviceptr, src: *const anyopaque, size: usize) CudaError!void {
        if (cuMemcpyHtoD_v2(dst, src, size) != CUDA_SUCCESS) return error.MemcpyFailed;
    }

    /// デバイス → ホストへメモリコピー
    pub fn copyDeviceToHost(_: *CudaContext, dst: *anyopaque, src: CUdeviceptr, size: usize) CudaError!void {
        if (cuMemcpyDtoH_v2(dst, src, size) != CUDA_SUCCESS) return error.MemcpyFailed;
    }

    /// デバイス → デバイスへメモリコピー
    pub fn copyDeviceToDevice(_: *CudaContext, dst: CUdeviceptr, src: CUdeviceptr, size: usize) CudaError!void {
        if (cuMemcpyDtoD_v2(dst, src, size) != CUDA_SUCCESS) return error.MemcpyFailed;
    }

    /// PTX モジュールをロードする
    pub fn loadModule(self: *CudaContext, ptx_image: *const anyopaque) CudaError!void {
        var module: CUmodule = undefined;
        if (cuModuleLoadData(&module, ptx_image) != CUDA_SUCCESS) return error.ModuleLoadFailed;
        self.module = module;
    }

    /// カーネル関数を取得する
    pub fn getFunction(self: *CudaContext, name: [*:0]const u8) CudaError!CUfunction {
        const m = self.module orelse return error.ModuleLoadFailed;
        var func: CUfunction = undefined;
        if (cuModuleGetFunction(&func, m, name) != CUDA_SUCCESS) return error.ModuleLoadFailed;
        return func;
    }

    /// カーネルを起動する
    pub fn launchKernel(
        self: *CudaContext,
        func: CUfunction,
        grid: [3]c_uint,
        block: [3]c_uint,
        shared_mem: c_uint,
        kernel_params: ?[*]?*anyopaque,
    ) CudaError!void {
        if (cuLaunchKernel(
            func,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem,
            self.stream,
            kernel_params,
            null,
        ) != CUDA_SUCCESS) return error.KernelLaunchFailed;
    }

    /// ストリームの完了を待つ
    pub fn synchronize(self: *CudaContext) void {
        _ = cuStreamSynchronize(self.stream);
    }

    /// cuBLAS SGEMM: C = alpha * op(A) * op(B) + beta * C
    pub fn sgemm(
        self: *CudaContext,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a_ptr: CUdeviceptr,
        lda: c_int,
        b_ptr: CUdeviceptr,
        ldb: c_int,
        beta: f32,
        c_ptr: CUdeviceptr,
        ldc: c_int,
    ) CudaError!void {
        if (cublasSgemm_v2(
            self.cublas_handle,
            transa,
            transb,
            m,
            n,
            k,
            &alpha,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            &beta,
            c_ptr,
            ldc,
        ) != CUBLAS_STATUS_SUCCESS) return error.SgemmFailed;
    }
};
