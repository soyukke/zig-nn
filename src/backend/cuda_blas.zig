/// cuBLAS 型定義 + extern 宣言
const cuda_driver = @import("cuda_driver.zig");
const CUstream = cuda_driver.CUstream;
const CUdeviceptr = cuda_driver.CUdeviceptr;

// ============================================================
// cuBLAS 型定義
// ============================================================

pub const cublasStatus_t = c_int;
pub const cublasHandle_t = *anyopaque;
pub const cublasOperation_t = c_int;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;

pub const cublasMath_t = c_int;
pub const CUBLAS_DEFAULT_MATH: cublasMath_t = 0;
pub const CUBLAS_TENSOR_OP_MATH: cublasMath_t = 1;

// ============================================================
// cuBLAS extern 宣言
// ============================================================

pub extern "c" fn cublasCreate_v2(handle: *cublasHandle_t) cublasStatus_t;
pub extern "c" fn cublasDestroy_v2(handle: cublasHandle_t) cublasStatus_t;
pub extern "c" fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) cublasStatus_t;
pub extern "c" fn cublasSgemm_v2(
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

pub extern "c" fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    A: CUdeviceptr,
    lda: c_int,
    strideA: c_longlong,
    B: CUdeviceptr,
    ldb: c_int,
    strideB: c_longlong,
    beta: *const f32,
    C: CUdeviceptr,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) cublasStatus_t;

pub extern "c" fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) cublasStatus_t;
