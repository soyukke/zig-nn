/// CUDA Driver API 型定義 + extern 宣言

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
// CUDA Driver API extern 宣言
// ============================================================

pub extern "c" fn cuInit(flags: c_uint) CUresult;
pub extern "c" fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
pub extern "c" fn cuDeviceGetName(name: [*]u8, len: c_int, dev: CUdevice) CUresult;
pub extern "c" fn cuCtxCreate_v2(ctx: *CUcontext, flags: c_uint, dev: CUdevice) CUresult;
pub extern "c" fn cuCtxDestroy_v2(ctx: CUcontext) CUresult;
pub extern "c" fn cuStreamCreate(stream: *CUstream, flags: c_uint) CUresult;
pub extern "c" fn cuStreamDestroy_v2(stream: CUstream) CUresult;
pub extern "c" fn cuStreamSynchronize(stream: CUstream) CUresult;
pub extern "c" fn cuMemAlloc_v2(dptr: *CUdeviceptr, size: usize) CUresult;
pub extern "c" fn cuMemFree_v2(dptr: CUdeviceptr) CUresult;
pub extern "c" fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const anyopaque, size: usize) CUresult;
pub extern "c" fn cuMemcpyDtoH_v2(dst: *anyopaque, src: CUdeviceptr, size: usize) CUresult;
pub extern "c" fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, size: usize) CUresult;
pub extern "c" fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) CUresult;
pub extern "c" fn cuModuleUnload(module: CUmodule) CUresult;
pub extern "c" fn cuModuleGetFunction(
    func: *CUfunction,
    module: CUmodule,
    name: [*:0]const u8,
) CUresult;
pub extern "c" fn cuLaunchKernel(
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
pub extern "c" fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: c_uint, N: usize) CUresult;
pub extern "c" fn cuMemsetD32Async(
    dstDevice: CUdeviceptr,
    ui: c_uint,
    N: usize,
    stream: ?CUstream,
) CUresult;
