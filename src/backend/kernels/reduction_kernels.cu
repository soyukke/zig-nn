// Reduction kernels: sum_rows/sum_cols/sum_1d

// Reduction sum axis=1: out[i] = sum(x[i, :])
// Each block handles one row
__global__ void reduction_sum_rows_kernel(float* out, const float* x, int rows, int cols) {
    extern __shared__ float shmem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        local_sum += x[row * cols + j];
    }
    shmem[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[row] = shmem[0];
}

// Reduction sum axis=0: out[j] = sum(x[:, j])
__global__ void reduction_sum_cols_kernel(float* out, const float* x, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            sum += x[i * cols + j];
        }
        out[j] = sum;
    }
}

// Reduction sum 1D: out[0] = sum(x[:])
__global__ void reduction_sum_1d_kernel(float* out, const float* x, int n) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    shmem[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, shmem[0]);
}
