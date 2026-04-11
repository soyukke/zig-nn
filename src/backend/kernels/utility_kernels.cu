// Utility kernels: fill/accum_grad/accum_scaled/sub_broadcast/dropout_fwd/silu_fwd_cache

// Fill: dst[i] = val
__global__ void fill_kernel(float* dst, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = val;
    }
}

// Accumulate gradient: dst += src
__global__ void accum_grad_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i];
    }
}

// Accumulate scaled: dst += src * scale
__global__ void accum_scaled_kernel(float* dst, const float* src, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i] * s;
    }
}

// Sub broadcast: out = a - b[i % b_total]
__global__ void sub_broadcast(float* out, const float* a, const float* b,
                              int a_total, int b_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_total) {
        out[i] = a[i] - b[i % b_total];
    }
}

// Dropout forward: out = x * mask, mask = (rand > rate) ? inv_keep : 0
// Uses a simple hash-based PRNG per element
__global__ void dropout_kernel(float* out, float* mask, const float* x,
                               unsigned long long seed, float rate, float inv_keep, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Simple hash-based PRNG (xorshift)
        unsigned long long h = seed ^ ((unsigned long long)i * 6364136223846793005ULL + 1442695040888963407ULL);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        float r = (float)(h >> 40) / (float)(1ULL << 24); // [0, 1)
        if (r >= rate) {
            mask[i] = inv_keep;
            out[i] = x[i] * inv_keep;
        } else {
            mask[i] = 0.0f;
            out[i] = 0.0f;
        }
    }
}

// SiLU forward + store sigmoid cache
__global__ void silu_fwd_cache_kernel(float* out, float* sig_cache, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float sig = 1.0f / (1.0f + expf(-v));
        sig_cache[i] = sig;
        out[i] = v * sig;
    }
}
