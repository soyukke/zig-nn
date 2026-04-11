// CUDA kernels for nn library inference
// Build: nvcc -ptx -arch=sm_90 cuda_kernels.cu -o cuda_kernels.ptx

extern "C" {

// Element-wise add with broadcast (a_total >= b_total, b_total divides a_total)
__global__ void add_broadcast(float* out, const float* a, const float* b,
                              int a_total, int b_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_total) {
        out[i] = a[i] + b[i % b_total];
    }
}

// Element-wise multiply with broadcast
__global__ void mul_broadcast(float* out, const float* a, const float* b,
                              int a_total, int b_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_total) {
        out[i] = a[i] * b[i % b_total];
    }
}

// Tanh GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void gelu_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        out[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

// SiLU: x * sigmoid(x)
__global__ void silu_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

// Row-wise softmax with shared memory reduction
// Each block handles one row of `cols` elements.
// Launch: grid=(rows), block=(min(cols, 1024))
__global__ void softmax_kernel(float* data, int rows, int cols) {
    extern __shared__ float shmem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    float* row_data = data + row * cols;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // 1. Find max (thread-level reduction)
    float local_max = -1e30f;
    for (int j = tid; j < cols; j += nthreads) {
        float v = row_data[j];
        if (v > local_max) local_max = v;
    }
    shmem[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && shmem[tid + s] > shmem[tid])
            shmem[tid] = shmem[tid + s];
        __syncthreads();
    }
    float row_max = shmem[0];
    __syncthreads();

    // 2. Compute exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        float e = expf(row_data[j] - row_max);
        row_data[j] = e;
        local_sum += e;
    }
    shmem[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    float total = shmem[0];
    __syncthreads();

    // 3. Normalize
    float inv_total = 1.0f / total;
    for (int j = tid; j < cols; j += nthreads) {
        row_data[j] *= inv_total;
    }
}

// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
// Each block handles one row of `cols` elements.
__global__ void layernorm_kernel(float* out, const float* x,
                                 const float* gamma, const float* beta,
                                 int rows, int cols, float eps) {
    extern __shared__ float shmem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* row_in = x + row * cols;
    float* row_out = out + row * cols;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Mean
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        local_sum += row_in[j];
    }
    shmem[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    float mean = shmem[0] / (float)cols;
    __syncthreads();

    // Variance
    float local_var = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        float d = row_in[j] - mean;
        local_var += d * d;
    }
    shmem[tid] = local_var;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(shmem[0] / (float)cols + eps);
    __syncthreads();

    // Normalize
    for (int j = tid; j < cols; j += nthreads) {
        row_out[j] = (row_in[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

// 2D transpose: out[j*rows+i] = x[i*cols+j]
__global__ void transpose_2d_kernel(float* out, const float* x, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        int r = i / cols;
        int c = i % cols;
        out[c * rows + r] = x[i];
    }
}

// Fused AdaLN modulation: out[b,s,d] = norm[b,s,d] * scale[b,d] + beta[b,d]
// scale and beta are [B, D], broadcast over S dimension
__global__ void modulate_adaln_kernel(float* out, const float* norm,
                                      const float* scale, const float* beta_data,
                                      int B, int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (i < total) {
        int b = i / (S * D);
        int d = i % D;
        out[i] = norm[i] * scale[b * D + d] + beta_data[b * D + d];
    }
}

// Embedding gather: out[i*embed_dim + d] = table[indices[i]*embed_dim + d]
__global__ void gather_kernel(float* out, const float* table, const int* indices,
                              int num_indices, int embed_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices * embed_dim) {
        int idx = i / embed_dim;
        int d = i % embed_dim;
        out[i] = table[indices[idx] * embed_dim + d];
    }
}

// Scale: out = x * s
__global__ void scale_kernel(float* out, const float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * s;
    }
}

} // extern "C"
