// Forward elementwise kernels: add/mul/gelu/silu/relu/sigmoid/tanh/exp/log/square/sqrt/abs/clamp/negative/scale

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

// ReLU: max(0, x)
__global__ void relu_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

// Sigmoid: 1 / (1 + exp(-x))
__global__ void sigmoid_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// Tanh
__global__ void tanh_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tanhf(x[i]);
    }
}

// Exp
__global__ void exp_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = expf(x[i]);
    }
}

// Log
__global__ void log_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = logf(x[i]);
    }
}

// Square: x^2
__global__ void square_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * x[i];
    }
}

// Sqrt
__global__ void sqrt_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sqrtf(x[i]);
    }
}

// Abs
__global__ void abs_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fabsf(x[i]);
    }
}

// Clamp: min(max(x, min_val), max_val)
__global__ void clamp_kernel(float* out, const float* x, float min_val, float max_val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = fminf(fmaxf(v, min_val), max_val);
    }
}

// Negative: -x
__global__ void negative_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = -x[i];
    }
}

// Fused Add + SiLU forward with sigmoid cache
// out = (a + b_broadcast) * sigmoid(a + b_broadcast)
// sig_cache stores sigmoid for efficient backward
__global__ void add_silu_fwd_cache_kernel(float* out, float* sig_cache,
    const float* a, const float* b, int a_total, int b_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_total) {
        float v = a[i] + b[i % b_total];
        float sig = 1.0f / (1.0f + expf(-v));
        sig_cache[i] = sig;
        out[i] = v * sig;
    }
}

// Scale: out = x * s
__global__ void scale_kernel(float* out, const float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * s;
    }
}
