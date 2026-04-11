// Backward elementwise kernels: gelu/silu/relu/sigmoid/tanh/exp/log/square/sqrt/abs/clamp/dropout backward

// GELU backward: ga += go * gelu'(x)
// gelu'(x) = 0.5*(1+tanh(inner)) + 0.5*x*sech2(inner)*inner'
__global__ void gelu_backward_kernel(float* ga, const float* go, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float inner_deriv = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * v * v);
        ga[i] += go[i] * (0.5f * (1.0f + tanh_val) + 0.5f * v * sech2 * inner_deriv);
    }
}

// SiLU backward: ga += go * (sig + x*sig*(1-sig)), where sig = sigmoid(x)
// sig_cache stores pre-computed sigmoid values
__global__ void silu_backward_kernel(float* ga, const float* go, const float* x,
                                     const float* sig_cache, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sig = sig_cache[i];
        float v = x[i];
        ga[i] += go[i] * (sig + v * sig * (1.0f - sig));
    }
}

// ReLU backward: ga += go * (x > 0)
__global__ void relu_backward_kernel(float* ga, const float* go, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (x[i] > 0.0f) ga[i] += go[i];
    }
}

// Sigmoid backward: ga += go * y * (1 - y), where y = sigmoid(x) = out
__global__ void sigmoid_backward_kernel(float* ga, const float* go, const float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float y = out[i];
        ga[i] += go[i] * y * (1.0f - y);
    }
}

// Tanh backward: ga += go * (1 - y^2), where y = tanh(x) = out
__global__ void tanh_backward_kernel(float* ga, const float* go, const float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float y = out[i];
        ga[i] += go[i] * (1.0f - y * y);
    }
}

// Exp backward: ga += go * out (since d/dx exp(x) = exp(x) = out)
__global__ void exp_backward_kernel(float* ga, const float* go, const float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ga[i] += go[i] * out[i];
    }
}

// Log backward: ga += go / x
__global__ void log_backward_kernel(float* ga, const float* go, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ga[i] += go[i] / x[i];
    }
}

// Square backward: ga += go * 2 * x
__global__ void square_backward_kernel(float* ga, const float* go, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ga[i] += go[i] * 2.0f * x[i];
    }
}

// Sqrt backward: ga += go * 0.5 / out (since d/dx sqrt(x) = 0.5/sqrt(x) = 0.5/out)
__global__ void sqrt_backward_kernel(float* ga, const float* go, const float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ga[i] += go[i] * 0.5f / out[i];
    }
}

// Abs backward: ga += go * sign(x)
__global__ void abs_backward_kernel(float* ga, const float* go, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sign = x[i] > 0.0f ? 1.0f : (x[i] < 0.0f ? -1.0f : 0.0f);
        ga[i] += go[i] * sign;
    }
}

// Clamp backward: ga += go if min <= x <= max, else 0
__global__ void clamp_backward_kernel(float* ga, const float* go, const float* x,
                                      float min_val, float max_val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (x[i] >= min_val && x[i] <= max_val) ga[i] += go[i];
    }
}

// Dropout backward: ga += go * mask
__global__ void dropout_backward_kernel(float* ga, const float* go, const float* mask, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ga[i] += go[i] * mask[i];
    }
}
