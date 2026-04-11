// CUDA kernels for nn library inference + training
// Build: nvcc -ptx -arch=sm_90 -I src/backend src/backend/cuda_kernels.cu -o cuda_kernels.ptx
//
// All kernels are split into category files under kernels/.
// This file aggregates them into a single translation unit via #include.

extern "C" {

#include "kernels/forward_elementwise.cu"
#include "kernels/forward_structured.cu"
#include "kernels/backward_elementwise.cu"
#include "kernels/backward_structured.cu"
#include "kernels/loss_kernels.cu"
#include "kernels/utility_kernels.cu"
#include "kernels/reduction_kernels.cu"
#include "kernels/optimizer_kernels.cu"

} // extern "C"
