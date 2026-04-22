/// CUDA GPU バックエンド
///
/// CUDA Driver API + cuBLAS 経由で NVIDIA GPU を利用する。
/// 各サブモジュールのシンボルを再エクスポートする facade モジュール。
const cuda_driver = @import("cuda_driver.zig");
const cuda_blas = @import("cuda_blas.zig");
const cuda_context = @import("cuda_context.zig");

// CUDA Driver API 型定義
pub const CUresult = cuda_driver.CUresult;
pub const CUdevice = cuda_driver.CUdevice;
pub const CUcontext = cuda_driver.CUcontext;
pub const CUstream = cuda_driver.CUstream;
pub const CUmodule = cuda_driver.CUmodule;
pub const CUfunction = cuda_driver.CUfunction;
pub const CUdeviceptr = cuda_driver.CUdeviceptr;
pub const CUDA_SUCCESS = cuda_driver.CUDA_SUCCESS;

// cuBLAS 型定義
pub const cublasStatus_t = cuda_blas.cublasStatus_t;
pub const cublasHandle_t = cuda_blas.cublasHandle_t;
pub const cublasOperation_t = cuda_blas.cublasOperation_t;
pub const CUBLAS_STATUS_SUCCESS = cuda_blas.CUBLAS_STATUS_SUCCESS;
pub const CUBLAS_OP_N = cuda_blas.CUBLAS_OP_N;
pub const CUBLAS_OP_T = cuda_blas.CUBLAS_OP_T;
pub const cublasMath_t = cuda_blas.cublasMath_t;
pub const CUBLAS_DEFAULT_MATH = cuda_blas.CUBLAS_DEFAULT_MATH;
pub const CUBLAS_TENSOR_OP_MATH = cuda_blas.CUBLAS_TENSOR_OP_MATH;

// CudaContext
pub const CudaContext = cuda_context.CudaContext;
pub const GpuMemPool = cuda_context.GpuMemPool;
