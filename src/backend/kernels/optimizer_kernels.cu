// Optimizer kernels: adam_step/norm_sq/scale_grad

// Fused GPU Adam optimizer
__global__ void adam_step_kernel(float* param, const float* grad,
                                 float* m, float* v,
                                 float lr, float beta1, float beta2,
                                 float eps, float wd,
                                 float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        float p = param[i];

        // Weight decay
        if (wd != 0.0f) p -= lr * wd * p;

        // Update moments
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;

        // Bias correction
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;

        // Update param
        param[i] = p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// Compute L2 norm squared (partial sums per block)
__global__ void norm_sq_kernel(float* partial_sums, const float* data, int n) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    shmem[tid] = (i < n) ? data[i] * data[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(partial_sums, shmem[0]);
}

// Scale gradients: grad *= scale
__global__ void scale_grad_kernel(float* grad, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad[i] *= scale;
    }
}
