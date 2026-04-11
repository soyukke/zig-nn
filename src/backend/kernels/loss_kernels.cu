// Loss kernels: cross_entropy/mse/bce forward + backward

// Cross-entropy forward: computes softmax + NLL loss
// Each block handles one batch row
// out_loss[0] is atomically accumulated
__global__ void cross_entropy_forward_kernel(float* out_loss, float* softmax_cache,
                                              const float* logits, const int* indices,
                                              int batch, int num_classes) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const float* row_logits = logits + row * num_classes;
    float* row_softmax = softmax_cache + row * num_classes;

    // Find max for stability
    float max_val = -1e30f;
    for (int j = 0; j < num_classes; j++) {
        if (row_logits[j] > max_val) max_val = row_logits[j];
    }

    // Exp and sum
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        float e = expf(row_logits[j] - max_val);
        row_softmax[j] = e;
        sum_exp += e;
    }

    // Normalize
    for (int j = 0; j < num_classes; j++) {
        row_softmax[j] /= sum_exp;
    }

    // NLL
    float nll = -logf(row_softmax[indices[row]] + 1e-10f);
    atomicAdd(out_loss, nll / (float)batch);
}

// Cross-entropy backward: ga += scale * (softmax - one_hot) / batch
__global__ void cross_entropy_backward_kernel(float* ga, const float* go,
                                               const float* softmax_cache, const int* indices,
                                               int batch, int num_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_classes;
    if (i < total) {
        int row = i / num_classes;
        int col = i % num_classes;
        float g = softmax_cache[i];
        if (col == indices[row]) g -= 1.0f;
        ga[i] += go[0] * g / (float)batch;
    }
}

// MSE loss forward: loss = sum((pred - target)^2) / n
__global__ void mse_forward_kernel(float* out_loss, const float* pred, const float* target, int n) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < n) {
        float d = pred[i] - target[i];
        local_sum = d * d;
    }
    shmem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out_loss, shmem[0] / (float)n);
}

// MSE backward: ga += go[0] * 2 * (pred - target) / n
__global__ void mse_backward_kernel(float* ga, const float* go,
                                     const float* pred, const float* target, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ga[i] += go[0] * 2.0f * (pred[i] - target[i]) / (float)n;
    }
}

// BCE with logits forward: loss = mean(max(x,0) - x*t + log(1+exp(-|x|)))
__global__ void bce_forward_kernel(float* out_loss, const float* logits, const float* target, int n) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < n) {
        float x = logits[i];
        float t = target[i];
        float pos = x > 0.0f ? x : 0.0f;
        local_sum = pos - x * t + logf(1.0f + expf(-fabsf(x)));
    }
    shmem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out_loss, shmem[0] / (float)n);
}

// BCE backward: ga += go[0] * (sigmoid(x) - target) / n
__global__ void bce_backward_kernel(float* ga, const float* go,
                                     const float* logits, const float* target, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sig = 1.0f / (1.0f + expf(-logits[i]));
        ga[i] += go[0] * (sig - target[i]) / (float)n;
    }
}
