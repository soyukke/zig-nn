// Backward structured kernels: mul/div backward, add/sub broadcast reduction,
// softmax/layernorm backward, scatter_add

// Mul backward (same shapes): ga += go*b, gb += go*a
__global__ void mul_backward_same_kernel(float* ga, float* gb,
                                         const float* go, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (ga) ga[i] += go[i] * b[i];
        if (gb) gb[i] += go[i] * a[i];
    }
}

// Mul backward broadcast b: ga += go * b[i%b_total], gb[i%b_total] += go*a (atomicAdd for reduction)
__global__ void mul_backward_broadcast_b_ga_kernel(float* ga, const float* go, const float* b,
                                                    int a_total, int b_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_total) {
        ga[i] += go[i] * b[i % b_total];
    }
}

__global__ void mul_backward_broadcast_b_gb_kernel(float* gb, const float* go, const float* a,
                                                    int a_total, int b_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_total) {
        atomicAdd(&gb[i % b_total], go[i] * a[i]);
    }
}

// Add backward broadcast: reduce go to smaller shape gb
__global__ void reduce_add_to_broadcast_kernel(float* gb, const float* go,
                                                int out_total, int b_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_total) {
        atomicAdd(&gb[i % b_total], go[i]);
    }
}

// Sub backward broadcast: reduce -go to smaller shape gb
__global__ void reduce_sub_to_broadcast_kernel(float* gb, const float* go,
                                                int out_total, int b_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_total) {
        atomicAdd(&gb[i % b_total], -go[i]);
    }
}

// Div backward: ga += go / b, gb += -go * a / (b*b)
__global__ void div_backward_kernel(float* ga, float* gb,
                                    const float* go, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (ga) ga[i] += go[i] / b[i];
        if (gb) gb[i] += -go[i] * a[i] / (b[i] * b[i]);
    }
}

// Softmax backward: ga += s * (go - dot(go, s))
// Each block handles one row
__global__ void softmax_backward_kernel(float* ga, const float* go, const float* s,
                                        int rows, int cols) {
    extern __shared__ float shmem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* s_row = s + row * cols;
    const float* go_row = go + row * cols;
    float* ga_row = ga + row * cols;

    // dot = sum(go * s)
    float local_dot = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        local_dot += go_row[j] * s_row[j];
    }
    shmem[tid] = local_dot;
    __syncthreads();
    for (int step = nthreads / 2; step > 0; step >>= 1) {
        if (tid < step) shmem[tid] += shmem[tid + step];
        __syncthreads();
    }
    float dot = shmem[0];
    __syncthreads();

    // ga += s * (go - dot)
    for (int j = tid; j < cols; j += nthreads) {
        ga_row[j] += s_row[j] * (go_row[j] - dot);
    }
}

// LayerNorm backward dx: gx += inv_std * (dy - mean_dy - x_norm * mean_dy_xn)
// Each block handles one row
__global__ void layernorm_backward_dx_kernel(float* gx, const float* go,
                                              const float* gamma, const float* x_norm,
                                              const float* inv_stds,
                                              int rows, int cols) {
    extern __shared__ float shmem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    float* shmem2 = shmem + nthreads;

    float inv_std = inv_stds[row];
    float inv_cols = 1.0f / (float)cols;

    // Compute dy = go * gamma, then mean_dy and mean_dy_xn
    float local_mean_dy = 0.0f;
    float local_mean_dy_xn = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        float dy = go[row * cols + j] * gamma[j];
        local_mean_dy += dy;
        local_mean_dy_xn += dy * x_norm[row * cols + j];
    }
    shmem[tid] = local_mean_dy;
    shmem2[tid] = local_mean_dy_xn;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] += shmem[tid + s];
            shmem2[tid] += shmem2[tid + s];
        }
        __syncthreads();
    }
    float mean_dy = shmem[0] * inv_cols;
    float mean_dy_xn = shmem2[0] * inv_cols;
    __syncthreads();

    // gx += inv_std * (dy - mean_dy - x_norm * mean_dy_xn)
    for (int j = tid; j < cols; j += nthreads) {
        float dy = go[row * cols + j] * gamma[j];
        gx[row * cols + j] += inv_std * (dy - mean_dy - x_norm[row * cols + j] * mean_dy_xn);
    }
}

// LayerNorm backward dgamma/dbeta: reduction across rows
// ggamma[j] += sum_i go[i,j] * x_norm[i,j]
// gbeta[j]  += sum_i go[i,j]
// Each thread handles one column across all rows
__global__ void layernorm_backward_dgamma_dbeta_kernel(float* ggamma, float* gbeta,
                                                        const float* go, const float* x_norm,
                                                        int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float sum_gamma = 0.0f;
        float sum_beta = 0.0f;
        for (int i = 0; i < rows; i++) {
            float g = go[i * cols + j];
            sum_gamma += g * x_norm[i * cols + j];
            sum_beta += g;
        }
        if (ggamma) ggamma[j] += sum_gamma;
        if (gbeta) gbeta[j] += sum_beta;
    }
}

// Scatter-add (gather backward): ga_table[indices[i]*D + d] += go[i*D + d]
__global__ void scatter_add_kernel(float* ga_table, const float* go, const int* indices,
                                   int num_indices, int embed_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices * embed_dim) {
        int idx = i / embed_dim;
        int d = i % embed_dim;
        atomicAdd(&ga_table[indices[idx] * embed_dim + d], go[i]);
    }
}
