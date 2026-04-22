/// CudaContext 構造体
///
/// GPU デバイスの初期化、メモリ管理、カーネル起動、cuBLAS 演算を提供する。
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda_blas = @import("cuda_blas.zig");
const log = @import("../log.zig").cuda;

const CUresult = cuda_driver.CUresult;
const CUdevice = cuda_driver.CUdevice;
const CUcontext = cuda_driver.CUcontext;
const CUstream = cuda_driver.CUstream;
const CUmodule = cuda_driver.CUmodule;
const CUfunction = cuda_driver.CUfunction;
const CUdeviceptr = cuda_driver.CUdeviceptr;
const CUDA_SUCCESS = cuda_driver.CUDA_SUCCESS;

const cublasStatus_t = cuda_blas.cublasStatus_t;
const cublasHandle_t = cuda_blas.cublasHandle_t;
const cublasOperation_t = cuda_blas.cublasOperation_t;
const CUBLAS_STATUS_SUCCESS = cuda_blas.CUBLAS_STATUS_SUCCESS;

/// Size-bucketed free list pool for GPU memory.
/// Avoids repeated cuMemAlloc/cuMemFree for intermediate buffers.
/// Bucket i holds buffers of size 2^(i + MIN_BUCKET) bytes.
pub const GpuMemPool = struct {
    const NUM_BUCKETS = 25; // 2^4 (16B) ~ 2^28 (256MB)
    const MIN_BUCKET: u5 = 4;
    buckets: [NUM_BUCKETS]std.ArrayListUnmanaged(CUdeviceptr),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) GpuMemPool {
        var pool: GpuMemPool = undefined;
        pool.allocator = allocator;
        for (&pool.buckets) |*b| b.* = .empty;
        return pool;
    }

    pub fn deinit(self: *GpuMemPool) void {
        for (&self.buckets) |*b| {
            for (b.items) |dptr| {
                _ = cuda_driver.cuMemFree_v2(dptr);
            }
            b.deinit(self.allocator);
        }
    }

    pub fn bucket_index(size: usize) usize {
        if (size == 0) return 0;
        // Round up to next power of 2, then find log2
        var s = size - 1;
        var bits: usize = 0;
        while (s > 0) {
            s >>= 1;
            bits += 1;
        }
        if (bits < MIN_BUCKET) return 0;
        const idx = bits - MIN_BUCKET;
        if (idx >= NUM_BUCKETS) return NUM_BUCKETS - 1;
        return idx;
    }

    pub fn bucket_size(idx: usize) usize {
        return @as(usize, 1) << @intCast(idx + MIN_BUCKET);
    }

    /// Try to acquire a buffer from the pool. Returns null if none available.
    pub fn acquire(self: *GpuMemPool, size: usize) ?CUdeviceptr {
        const idx = bucket_index(size);
        if (self.buckets[idx].items.len > 0) {
            return self.buckets[idx].pop();
        }
        return null;
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(self: *GpuMemPool, dptr: CUdeviceptr, size: usize) void {
        const idx = bucket_index(size);
        self.buckets[idx].append(self.allocator, dptr) catch {
            // If we can't track it, just free it
            _ = cuda_driver.cuMemFree_v2(dptr);
        };
    }
};

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
        if (cuda_driver.cuInit(0) != CUDA_SUCCESS) return error.InitFailed;

        var device: CUdevice = undefined;
        if (cuda_driver.cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS) {
            return error.DeviceNotFound;
        }

        var name_buf: [256]u8 = undefined;
        if (cuda_driver.cuDeviceGetName(&name_buf, 256, device) == CUDA_SUCCESS) {
            const name_slice = std.mem.sliceTo(&name_buf, 0);
            log.info("device: {s}", .{name_slice});
        }

        var ctx: CUcontext = undefined;
        if (cuda_driver.cuCtxCreate_v2(&ctx, 0, device) != CUDA_SUCCESS) {
            return error.ContextCreationFailed;
        }

        var stream: CUstream = undefined;
        if (cuda_driver.cuStreamCreate(&stream, 0) != CUDA_SUCCESS) {
            return error.StreamCreationFailed;
        }

        var cublas: cublasHandle_t = undefined;
        if (cuda_blas.cublasCreate_v2(&cublas) != CUBLAS_STATUS_SUCCESS) {
            return error.CublasInitFailed;
        }
        if (cuda_blas.cublasSetStream_v2(cublas, stream) != CUBLAS_STATUS_SUCCESS) {
            return error.CublasInitFailed;
        }

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
        _ = cuda_blas.cublasDestroy_v2(self.cublas_handle);
        if (self.module) |m| _ = cuda_driver.cuModuleUnload(m);
        _ = cuda_driver.cuStreamDestroy_v2(self.stream);
        _ = cuda_driver.cuCtxDestroy_v2(self.context);
    }

    /// GPU メモリを確保する
    pub fn alloc_buffer(self: *CudaContext, size: usize) CudaError!CUdeviceptr {
        _ = self;
        var dptr: CUdeviceptr = undefined;
        if (cuda_driver.cuMemAlloc_v2(&dptr, size) != CUDA_SUCCESS) return error.MemAllocFailed;
        return dptr;
    }

    /// GPU メモリを解放する
    pub fn free_buffer(_: *CudaContext, dptr: CUdeviceptr) void {
        _ = cuda_driver.cuMemFree_v2(dptr);
    }

    /// ホスト → デバイスへメモリコピー
    pub fn copy_host_to_device(
        _: *CudaContext,
        dst: CUdeviceptr,
        src: *const anyopaque,
        size: usize,
    ) CudaError!void {
        if (cuda_driver.cuMemcpyHtoD_v2(dst, src, size) != CUDA_SUCCESS) return error.MemcpyFailed;
    }

    /// デバイス → ホストへメモリコピー
    pub fn copy_device_to_host(
        _: *CudaContext,
        dst: *anyopaque,
        src: CUdeviceptr,
        size: usize,
    ) CudaError!void {
        if (cuda_driver.cuMemcpyDtoH_v2(dst, src, size) != CUDA_SUCCESS) return error.MemcpyFailed;
    }

    /// デバイス → デバイスへメモリコピー
    pub fn copy_device_to_device(
        _: *CudaContext,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        size: usize,
    ) CudaError!void {
        if (cuda_driver.cuMemcpyDtoD_v2(dst, src, size) != CUDA_SUCCESS) return error.MemcpyFailed;
    }

    /// PTX モジュールをロードする
    pub fn load_module(self: *CudaContext, ptx_image: *const anyopaque) CudaError!void {
        var module: CUmodule = undefined;
        if (cuda_driver.cuModuleLoadData(&module, ptx_image) != CUDA_SUCCESS) {
            return error.ModuleLoadFailed;
        }
        self.module = module;
    }

    /// カーネル関数を取得する
    pub fn get_function(self: *CudaContext, name: [*:0]const u8) CudaError!CUfunction {
        const m = self.module orelse return error.ModuleLoadFailed;
        var func: CUfunction = undefined;
        if (cuda_driver.cuModuleGetFunction(&func, m, name) != CUDA_SUCCESS) {
            return error.ModuleLoadFailed;
        }
        return func;
    }

    /// カーネルを起動する
    pub fn launch_kernel(
        self: *CudaContext,
        func: CUfunction,
        grid: [3]c_uint,
        block: [3]c_uint,
        shared_mem: c_uint,
        kernel_params: ?[*]?*anyopaque,
    ) CudaError!void {
        if (cuda_driver.cuLaunchKernel(
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
        _ = cuda_driver.cuStreamSynchronize(self.stream);
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
        if (cuda_blas.cublasSgemm_v2(
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

    /// cuBLAS SGEMM with beta=1.0 for gradient accumulation: C += alpha * op(A) * op(B)
    pub fn sgemm_accum(
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
        c_ptr: CUdeviceptr,
        ldc: c_int,
    ) CudaError!void {
        return self.sgemm(transa, transb, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, 1.0, c_ptr, ldc);
    }

    /// Strided batched SGEMM
    pub fn sgemm_strided_batched(
        self: *CudaContext,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a_ptr: CUdeviceptr,
        lda: c_int,
        stride_a: c_longlong,
        b_ptr: CUdeviceptr,
        ldb: c_int,
        stride_b: c_longlong,
        beta: f32,
        c_ptr: CUdeviceptr,
        ldc: c_int,
        stride_c: c_longlong,
        batch_count: c_int,
    ) CudaError!void {
        if (cuda_blas.cublasSgemmStridedBatched(
            self.cublas_handle,
            transa,
            transb,
            m,
            n,
            k,
            &alpha,
            a_ptr,
            lda,
            stride_a,
            b_ptr,
            ldb,
            stride_b,
            &beta,
            c_ptr,
            ldc,
            stride_c,
            batch_count,
        ) != CUBLAS_STATUS_SUCCESS) return error.SgemmFailed;
    }

    /// TF32 Tensor Core math mode を有効化 (Ampere+)
    pub fn set_tensor_math_mode(self: *CudaContext) void {
        _ = cuda_blas.cublasSetMathMode(self.cublas_handle, cuda_blas.CUBLAS_TENSOR_OP_MATH);
    }

    /// GPU メモリをゼロクリア
    pub fn memset_zero(_: *CudaContext, dptr: CUdeviceptr, num_floats: usize) CudaError!void {
        if (cuda_driver.cuMemsetD32_v2(dptr, 0, num_floats) != CUDA_SUCCESS) {
            return error.MemcpyFailed;
        }
    }

    /// GPU メモリをゼロクリア (非同期)
    pub fn memset_zero_async(
        self: *CudaContext,
        dptr: CUdeviceptr,
        num_floats: usize,
    ) CudaError!void {
        if (cuda_driver.cuMemsetD32Async(dptr, 0, num_floats, self.stream) != CUDA_SUCCESS) {
            return error.MemcpyFailed;
        }
    }
};
