/// Metal Training Kernels for GPU-based neural network training
/// All kernels operate on f32 tensors for training precision.

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Params structures
// ============================================================

struct MatmulParams {
    uint M;
    uint K;
    uint N;
};

struct BiasParams {
    uint rows;
    uint cols;
};

struct SiluParams {
    uint count;
};

struct MseLossParams {
    uint count;
};

struct AdamParams {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float bc1; // 1 - beta1^t
    float bc2; // 1 - beta2^t
    uint count;
};

// ============================================================
// Forward Kernels
// ============================================================

/// Tiled f32 matrix multiply with register tiling: C = A @ B
/// A: (M, K), B: (K, N), C: (M, N)
/// Block: BM×BN output tile per threadgroup, BK inner tile
/// Threadgroup: 16×16 = 256 threads, each computes RM×RN outputs
#define BM 64
#define BN 64
#define BK 16
#define RM 4  // register tile rows per thread
#define RN 4  // register tile cols per thread
// Legacy defines for non-matmul kernels that still reference TM/TN/TK
#define TM 16
#define TN 16
#define TK 16

kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    // Thread position within 16×16 threadgroup
    const uint tid = lid.y * 16 + lid.x;  // 0..255
    const uint thread_row = tid / (BN / RN);  // 0..15 → which RM-row block
    const uint thread_col = tid % (BN / RN);  // 0..15 → which RN-col block

    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];

    float acc[RM][RN] = {};  // zero-initialized

    for (uint bk = 0; bk < K; bk += BK) {
        // Cooperative load: 256 threads load BM×BK = 64×16 = 1024 elements
        // Each thread loads 1024/256 = 4 elements of A
        for (uint i = 0; i < (BM * BK) / 256; i++) {
            uint idx = tid + i * 256;
            uint ar = idx / BK;
            uint ac = idx % BK;
            uint global_r = tgid.y * BM + ar;
            uint global_c = bk + ac;
            As[ar][ac] = (global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f;
        }
        // Each thread loads 1024/256 = 4 elements of B
        for (uint i = 0; i < (BK * BN) / 256; i++) {
            uint idx = tid + i * 256;
            uint br = idx / BN;
            uint bc = idx % BN;
            uint global_r = bk + br;
            uint global_c = tgid.x * BN + bc;
            Bs[br][bc] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute RM×RN outputs per thread
        for (uint k = 0; k < BK; k++) {
            float a_vals[RM];
            float b_vals[RN];
            for (uint ri = 0; ri < RM; ri++)
                a_vals[ri] = As[thread_row * RM + ri][k];
            for (uint ci = 0; ci < RN; ci++)
                b_vals[ci] = Bs[k][thread_col * RN + ci];
            for (uint ri = 0; ri < RM; ri++)
                for (uint ci = 0; ci < RN; ci++)
                    acc[ri][ci] += a_vals[ri] * b_vals[ci];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write RM×RN outputs
    for (uint ri = 0; ri < RM; ri++) {
        for (uint ci = 0; ci < RN; ci++) {
            uint global_r = tgid.y * BM + thread_row * RM + ri;
            uint global_c = tgid.x * BN + thread_col * RN + ci;
            if (global_r < M && global_c < N)
                C[global_r * N + global_c] = acc[ri][ci];
        }
    }
}

/// Element-wise add: C = A + B
kernel void add_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        C[idx] = A[idx] + B[idx];
    }
}

/// Bias add: Z[i,j] = A[i,j] + bias[j]
kernel void add_bias_f32(
    device const float* A [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* Z [[buffer(2)]],
    constant BiasParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < params.rows * params.cols) {
        uint j = idx % params.cols;
        Z[idx] = A[idx] + bias[j];
    }
}

/// SiLU forward: out = x * sigmoid(x), sig_out = sigmoid(x)
kernel void silu_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    device float* sig_out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        float s = 1.0f / (1.0f + exp(-x[idx]));
        sig_out[idx] = s;
        out[idx] = x[idx] * s;
    }
}

/// MSE loss diff: diff[i] = pred[i] - target[i]
kernel void mse_loss_diff(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* diff [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        diff[idx] = pred[idx] - target[idx];
    }
}

/// MSE loss reduce: loss = sum(diff^2) / count
/// Uses threadgroup reduction
kernel void mse_loss_reduce(
    device const float* diff [[buffer(0)]],
    device float* loss [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[256];

    float sum = 0.0f;
    for (uint i = tid; i < count; i += tg_size) {
        float d = diff[i];
        sum += d * d;
    }
    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        loss[0] = shared[0] / float(count);
    }
}


// ============================================================
// Backward Kernels
// ============================================================

/// grad_A += grad_out @ B^T (register tiling, accumulate)
/// grad_out: (M, N), B: (K, N)  => grad_A: (M, K)
kernel void matmul_f32_backward_a(
    device const float* grad_out [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* grad_A [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    // grad_A = grad_out @ B^T: grad_out(M,N) @ B^T(N,K) = (M,K)
    const uint M = params.M;
    const uint inner = params.N;  // N is inner dimension
    const uint K = params.K;

    const uint tid = lid.y * 16 + lid.x;
    const uint thread_row = tid / (BN / RN);
    const uint thread_col = tid % (BN / RN);

    threadgroup float Gs[BM][BK];   // grad_out tile
    threadgroup float Bts[BK][BN];  // B^T tile

    float acc[RM][RN] = {};

    for (uint bn = 0; bn < inner; bn += BK) {
        // Load grad_out: G[row, bn+col]
        for (uint i = 0; i < (BM * BK) / 256; i++) {
            uint idx = tid + i * 256;
            uint ar = idx / BK;
            uint ac = idx % BK;
            uint gr = tgid.y * BM + ar;
            uint gc = bn + ac;
            Gs[ar][ac] = (gr < M && gc < inner) ? grad_out[gr * inner + gc] : 0.0f;
        }
        // Load B^T: B^T[bn+r, col] = B[col, bn+r] = B[col * inner + bn + r]
        // Output cols are in K dimension
        for (uint i = 0; i < (BK * BN) / 256; i++) {
            uint idx = tid + i * 256;
            uint br = idx / BN;
            uint bc = idx % BN;
            uint gr = bn + br;
            uint gc = tgid.x * BN + bc;
            Bts[br][bc] = (gr < inner && gc < K) ? B[gc * inner + gr] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK; k++) {
            float a_vals[RM], b_vals[RN];
            for (uint ri = 0; ri < RM; ri++) a_vals[ri] = Gs[thread_row * RM + ri][k];
            for (uint ci = 0; ci < RN; ci++) b_vals[ci] = Bts[k][thread_col * RN + ci];
            for (uint ri = 0; ri < RM; ri++)
                for (uint ci = 0; ci < RN; ci++)
                    acc[ri][ci] += a_vals[ri] * b_vals[ci];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint ri = 0; ri < RM; ri++) {
        for (uint ci = 0; ci < RN; ci++) {
            uint gr = tgid.y * BM + thread_row * RM + ri;
            uint gc = tgid.x * BN + thread_col * RN + ci;
            if (gr < M && gc < K)
                grad_A[gr * K + gc] += acc[ri][ci];
        }
    }
}

/// grad_B += A^T @ grad_out (register tiling, accumulate)
/// A: (M, K), grad_out: (M, N) => grad_B: (K, N)
kernel void matmul_f32_backward_b(
    device const float* A [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_B [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    // grad_B = A^T @ grad_out: A^T(K,M) @ grad_out(M,N) = (K,N)
    const uint inner = params.M;  // M is inner dimension
    const uint K = params.K;
    const uint N = params.N;

    const uint tid = lid.y * 16 + lid.x;
    const uint thread_row = tid / (BN / RN);
    const uint thread_col = tid % (BN / RN);

    threadgroup float Ats[BM][BK];  // A^T tile
    threadgroup float Gs[BK][BN];   // grad_out tile

    float acc[RM][RN] = {};

    for (uint bm = 0; bm < inner; bm += BK) {
        // Load A^T: A^T[row, bm+col] = A[bm+col, row] = A[(bm+col)*K + row]
        for (uint i = 0; i < (BM * BK) / 256; i++) {
            uint idx = tid + i * 256;
            uint ar = idx / BK;
            uint ac = idx % BK;
            uint gr = tgid.y * BM + ar;  // row in K
            uint gc = bm + ac;           // col in M
            Ats[ar][ac] = (gr < K && gc < inner) ? A[gc * K + gr] : 0.0f;
        }
        // Load grad_out: G[bm+r, col]
        for (uint i = 0; i < (BK * BN) / 256; i++) {
            uint idx = tid + i * 256;
            uint br = idx / BN;
            uint bc = idx % BN;
            uint gr = bm + br;
            uint gc = tgid.x * BN + bc;
            Gs[br][bc] = (gr < inner && gc < N) ? grad_out[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK; k++) {
            float a_vals[RM], b_vals[RN];
            for (uint ri = 0; ri < RM; ri++) a_vals[ri] = Ats[thread_row * RM + ri][k];
            for (uint ci = 0; ci < RN; ci++) b_vals[ci] = Gs[k][thread_col * RN + ci];
            for (uint ri = 0; ri < RM; ri++)
                for (uint ci = 0; ci < RN; ci++)
                    acc[ri][ci] += a_vals[ri] * b_vals[ci];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint ri = 0; ri < RM; ri++) {
        for (uint ci = 0; ci < RN; ci++) {
            uint gr = tgid.y * BM + thread_row * RM + ri;
            uint gc = tgid.x * BN + thread_col * RN + ci;
            if (gr < K && gc < N)
                grad_B[gr * N + gc] += acc[ri][ci];
        }
    }
}

/// grad[i] += src[i] (element-wise accumulate)
kernel void add_backward_accum(
    device const float* src [[buffer(0)]],
    device float* grad [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        grad[idx] += src[idx];
    }
}

/// grad_bias[j] += sum_i(grad_out[i,j])
/// grad_out: (rows, cols) => grad_bias: (cols)
kernel void add_bias_backward(
    device const float* grad_out [[buffer(0)]],
    device float* grad_bias [[buffer(1)]],
    constant BiasParams& params [[buffer(2)]],
    uint j [[thread_position_in_grid]]
) {
    if (j < params.cols) {
        float sum = 0.0f;
        for (uint i = 0; i < params.rows; i++) {
            sum += grad_out[i * params.cols + j];
        }
        grad_bias[j] += sum;
    }
}

/// SiLU backward: grad_x += go * sig * (1 + x * (1 - sig))
kernel void silu_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* x_data [[buffer(1)]],
    device const float* sig_data [[buffer(2)]],
    device float* grad_x [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        float sig = sig_data[idx];
        float x = x_data[idx];
        float go = grad_out[idx];
        grad_x[idx] += go * sig * (1.0f + x * (1.0f - sig));
    }
}

/// MSE loss backward: grad_pred += grad_out[0] * 2 * diff / n
kernel void mse_loss_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* diff [[buffer(1)]],
    device float* grad_pred [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        float scale = 2.0f / float(count);
        grad_pred[idx] += grad_out[0] * scale * diff[idx];
    }
}


// ============================================================
// Optimizer Kernels
// ============================================================

/// Fused Adam step: update m, v, and weights in one kernel
kernel void adam_step(
    device float* weights [[buffer(0)]],
    device float* grads [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant AdamParams& params [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < params.count) {
        float g = grads[idx];
        float w = weights[idx];

        // AdamW: decoupled weight decay
        if (params.weight_decay > 0.0f) {
            w -= params.lr * params.weight_decay * w;
        }

        // Update moments
        float m_val = params.beta1 * m[idx] + (1.0f - params.beta1) * g;
        float v_val = params.beta2 * v[idx] + (1.0f - params.beta2) * g * g;

        m[idx] = m_val;
        v[idx] = v_val;

        // Bias correction
        float m_hat = m_val / params.bc1;
        float v_hat = v_val / params.bc2;

        // Weight update
        w -= params.lr * m_hat / (sqrt(v_hat) + params.epsilon);
        weights[idx] = w;
    }
}

/// Zero a buffer
kernel void zero_buffer(
    device float* buf [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        buf[idx] = 0.0f;
    }
}


// ============================================================
// Phase 2: Transformer Training Kernels
// ============================================================

// --- Params structures ---

struct SoftmaxParams {
    uint rows;
    uint cols;
};

struct CausalSoftmaxParams {
    uint rows;
    uint cols;
    uint num_heads; // For GQA: position = row / num_heads
    uint seq_len;   // For batched: position wraps within seq_len * num_heads
};

struct LayerNormParams {
    uint rows;
    uint cols;
    float epsilon;
};

struct CrossEntropyParams {
    uint batch_size;
    uint num_classes;
};

struct EmbeddingParams {
    uint num_tokens;
    uint embed_dim;
};

// --- ReLU ---

kernel void relu_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        out[idx] = max(0.0f, x[idx]);
    }
}

kernel void relu_backward(
    device const float* x [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        grad_in[idx] += grad_out[idx] * (x[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

// --- GELU (tanh approximation) ---

kernel void gelu_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        float v = x[idx];
        float c = 0.7978845608f; // sqrt(2/pi)
        float inner = c * (v + 0.044715f * v * v * v);
        out[idx] = 0.5f * v * (1.0f + precise::tanh(inner));
    }
}

kernel void gelu_backward(
    device const float* x [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        float v = x[idx];
        float c = 0.7978845608f;
        float inner = c * (v + 0.044715f * v * v * v);
        float t = precise::tanh(inner);
        float sech2 = 1.0f - t * t;
        float inner_deriv = c * (1.0f + 3.0f * 0.044715f * v * v);
        float gelu_grad = 0.5f * (1.0f + t) + 0.5f * v * sech2 * inner_deriv;
        grad_in[idx] += grad_out[idx] * gelu_grad;
    }
}

// --- Softmax (per-row) ---

kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint cols = params.cols;
    uint base = row * cols;
    if (row >= params.rows) return;

    // 1. Find max
    threadgroup float shared[256];
    float local_max = -INFINITY;
    for (uint j = lid; j < cols; j += tg_size) {
        local_max = max(local_max, input[base + j]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    // 2. Compute exp and sum
    float local_sum = 0.0f;
    for (uint j = lid; j < cols; j += tg_size) {
        float e = exp(input[base + j] - row_max);
        output[base + j] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];

    // 3. Normalize
    for (uint j = lid; j < cols; j += tg_size) {
        output[base + j] *= inv_sum;
    }
}

kernel void softmax_backward(
    device const float* out [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint cols = params.cols;
    uint base = row * cols;
    if (row >= params.rows) return;

    // dot = sum(grad_out * out)
    threadgroup float shared[256];
    float local_dot = 0.0f;
    for (uint j = lid; j < cols; j += tg_size) {
        local_dot += grad_out[base + j] * out[base + j];
    }
    shared[lid] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = shared[0];

    // grad_in[j] += out[j] * (grad_out[j] - dot)
    for (uint j = lid; j < cols; j += tg_size) {
        grad_in[base + j] += out[base + j] * (grad_out[base + j] - dot);
    }
}

/// Causal softmax: apply upper-triangle mask (-inf) then softmax (per-row)
/// For decoder self-attention where row i can only attend to cols 0..i
kernel void causal_softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant CausalSoftmaxParams& params [[buffer(2)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint cols = params.cols;
    uint row = row_id;
    uint base = row * cols;
    if (row >= params.rows) return;

    // For batched + GQA: position wraps within seq_len * num_heads per batch
    // pos = (row % (seq_len * num_heads)) / num_heads
    uint pos = (row % (params.seq_len * params.num_heads)) / params.num_heads;

    threadgroup float shared[256];

    // 1. Find max (with causal mask)
    float local_max = -INFINITY;
    for (uint j = lid; j < cols; j += tg_size) {
        float val = (j <= pos) ? input[base + j] : -INFINITY;
        local_max = max(local_max, val);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    // 2. Compute exp and sum
    float local_sum = 0.0f;
    for (uint j = lid; j < cols; j += tg_size) {
        float val = (j <= pos) ? exp(input[base + j] - row_max) : 0.0f;
        output[base + j] = val;
        local_sum += val;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / max(shared[0], 1e-10f);

    // 3. Normalize
    for (uint j = lid; j < cols; j += tg_size) {
        output[base + j] *= inv_sum;
    }
}

// --- LayerNorm ---

/// LayerNorm forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
/// Saves mean and inv_std per row for backward
kernel void layernorm_forward(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* mean_out [[buffer(4)]],
    device float* inv_std_out [[buffer(5)]],
    constant LayerNormParams& params [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint cols = params.cols;
    uint base = row * cols;
    if (row >= params.rows) return;

    threadgroup float shared[256];

    // 1. Compute mean
    float local_sum = 0.0f;
    for (uint j = lid; j < cols; j += tg_size) {
        local_sum += x[base + j];
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(cols);
    if (lid == 0) mean_out[row] = mean;

    // 2. Compute variance
    float local_var = 0.0f;
    for (uint j = lid; j < cols; j += tg_size) {
        float diff = x[base + j] - mean;
        local_var += diff * diff;
    }
    shared[lid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared[0] / float(cols) + params.epsilon);
    if (lid == 0) inv_std_out[row] = inv_std;

    // 3. Normalize + scale + shift
    for (uint j = lid; j < cols; j += tg_size) {
        float norm = (x[base + j] - mean) * inv_std;
        out[base + j] = gamma[j] * norm + beta[j];
    }
}

/// LayerNorm backward for input x:
/// grad_x[j] = inv_std * (ds[j] - mean(ds) - x_hat[j] * mean(ds * x_hat))
/// where ds = gamma * grad_out, x_hat = (x - mean) * inv_std
kernel void layernorm_backward_x(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* grad_out [[buffer(2)]],
    device const float* mean_buf [[buffer(3)]],
    device const float* inv_std_buf [[buffer(4)]],
    device float* grad_x [[buffer(5)]],
    constant LayerNormParams& params [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint cols = params.cols;
    uint base = row * cols;
    if (row >= params.rows) return;
    float mean = mean_buf[row];
    float inv_std = inv_std_buf[row];

    threadgroup float shared[256];

    // mean_ds = mean(gamma * grad_out)
    float local_sum_ds = 0.0f;
    float local_sum_ds_xhat = 0.0f;
    for (uint j = lid; j < cols; j += tg_size) {
        float ds = gamma[j] * grad_out[base + j];
        float x_hat = (x[base + j] - mean) * inv_std;
        local_sum_ds += ds;
        local_sum_ds_xhat += ds * x_hat;
    }

    // Reduce mean_ds
    shared[lid] = local_sum_ds;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_ds = shared[0] / float(cols);

    // Reduce mean_ds_xhat
    shared[lid] = local_sum_ds_xhat;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_ds_xhat = shared[0] / float(cols);

    // Compute grad_x
    for (uint j = lid; j < cols; j += tg_size) {
        float ds = gamma[j] * grad_out[base + j];
        float x_hat = (x[base + j] - mean) * inv_std;
        grad_x[base + j] += inv_std * (ds - mean_ds - x_hat * mean_ds_xhat);
    }
}

/// LayerNorm backward for gamma and beta (reduce over rows/batch dimension):
/// grad_gamma[j] += sum_i(grad_out[i,j] * x_hat[i,j])
/// grad_beta[j]  += sum_i(grad_out[i,j])
kernel void layernorm_backward_params(
    device const float* x [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device const float* mean_buf [[buffer(2)]],
    device const float* inv_std_buf [[buffer(3)]],
    device float* grad_gamma [[buffer(4)]],
    device float* grad_beta [[buffer(5)]],
    constant LayerNormParams& params [[buffer(6)]],
    uint j [[thread_position_in_grid]]
) {
    if (j >= params.cols) return;
    uint rows = params.rows;
    uint cols = params.cols;

    float sum_gg = 0.0f;
    float sum_gb = 0.0f;
    for (uint i = 0; i < rows; i++) {
        float go = grad_out[i * cols + j];
        float x_hat = (x[i * cols + j] - mean_buf[i]) * inv_std_buf[i];
        sum_gg += go * x_hat;
        sum_gb += go;
    }
    grad_gamma[j] += sum_gg;
    grad_beta[j] += sum_gb;
}

// --- Cross-Entropy Loss ---

/// Combined softmax + NLL loss forward (numerically stable)
/// logits: (batch, num_classes), targets: (batch,) uint indices
/// Outputs: softmax_out (batch, num_classes), loss_per_sample (batch,)
/// Padding: target >= num_classes means ignore (loss=0, softmax still computed)
kernel void cross_entropy_forward(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* softmax_out [[buffer(2)]],
    device float* loss_per_sample [[buffer(3)]],
    constant CrossEntropyParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint nc = params.num_classes;
    uint base = row * nc;
    if (row >= params.batch_size) return;

    threadgroup float shared[256];

    // 1. Max
    float local_max = -INFINITY;
    for (uint j = lid; j < nc; j += tg_size) {
        local_max = max(local_max, logits[base + j]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    // 2. Exp + sum
    float local_sum = 0.0f;
    for (uint j = lid; j < nc; j += tg_size) {
        float e = exp(logits[base + j] - row_max);
        softmax_out[base + j] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];

    // 3. Normalize to get softmax
    for (uint j = lid; j < nc; j += tg_size) {
        softmax_out[base + j] *= inv_sum;
    }

    // 4. NLL loss for this sample (skip if target is padding sentinel)
    if (lid == 0) {
        uint target = targets[row];
        if (target >= nc) {
            loss_per_sample[row] = 0.0f;  // padding: no loss
        } else {
            loss_per_sample[row] = -log(max(softmax_out[base + target], 1e-7f));
        }
    }
}

/// Reduce per-sample losses to scalar: loss = sum(loss_per_sample) / valid_count
/// Supports padding: zero-loss samples are excluded from count
kernel void cross_entropy_reduce(
    device const float* losses [[buffer(0)]],
    device float* total_loss [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    device const uint* targets [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[256];
    threadgroup float shared_count[256];
    float local_sum = 0.0f;
    float local_count = 0.0f;
    for (uint i = lid; i < batch_size; i += tg_size) {
        if (targets[i] < num_classes) {
            local_sum += losses[i];
            local_count += 1.0f;
        }
    }
    shared[lid] = local_sum;
    shared_count[lid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
            shared_count[lid] += shared_count[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        float count = max(shared_count[0], 1.0f);
        total_loss[0] = shared[0] / count;
    }
}

/// Cross-entropy backward:
/// grad_logits[i,j] += grad_out * (softmax[i,j] - 1{j==target[i]}) / valid_count
/// Padding: target >= num_classes means ignore (zero gradient for that row)
kernel void cross_entropy_backward(
    device const float* softmax_vals [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* grad_logits [[buffer(2)]],
    device const float* grad_out [[buffer(3)]],
    constant CrossEntropyParams& params [[buffer(4)]],
    constant uint& valid_count [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_classes;
    if (idx >= total) return;

    uint nc = params.num_classes;
    uint row = idx / nc;
    uint col = idx % nc;

    // Skip padding positions
    if (targets[row] >= nc) return;

    float sm = softmax_vals[idx];
    float one_hot = (col == targets[row]) ? 1.0f : 0.0f;
    float inv_count = 1.0f / max(float(valid_count), 1.0f);
    grad_logits[idx] += grad_out[0] * (sm - one_hot) * inv_count;
}

// --- Embedding ---

/// Embedding forward: out[i * embed_dim + j] = weight[indices[i] * embed_dim + j]
kernel void embedding_forward(
    device const float* weight [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = params.num_tokens * params.embed_dim;
    if (idx >= total) return;

    uint token_idx = idx / params.embed_dim;
    uint dim_idx = idx % params.embed_dim;
    uint vocab_idx = indices[token_idx];
    out[idx] = weight[vocab_idx * params.embed_dim + dim_idx];
}

/// Embedding backward: scatter_add grad to weight grad
/// Uses atomic CAS loop for float add (compatible with all Metal versions)
void atomicAddFloat(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float old_val = as_type<float>(expected);
        float new_val = old_val + val;
        uint new_bits = as_type<uint>(new_val);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, new_bits,
                                                   memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

kernel void embedding_backward(
    device const uint* indices [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic_uint* grad_weight [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = params.num_tokens * params.embed_dim;
    if (idx >= total) return;

    uint token_idx = idx / params.embed_dim;
    uint dim_idx = idx % params.embed_dim;
    uint vocab_idx = indices[token_idx];
    atomicAddFloat(&grad_weight[vocab_idx * params.embed_dim + dim_idx], grad_out[idx]);
}

// --- Scale ---

/// Element-wise scale: out[i] = input[i] * scale
kernel void scale_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        output[idx] = input[idx] * scale;
    }
}

/// Scale backward: grad_in[i] += grad_out[i] * scale
kernel void scale_backward(
    device const float* grad_out [[buffer(0)]],
    device float* grad_in [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        grad_in[idx] += grad_out[idx] * scale;
    }
}

// --- Matmul variants for attention ---

/// C = A @ B^T (register tiling)
/// A: (M, K), B: (N, K) => C: (M, N)
/// Used for attention: scores = Q @ K^T
kernel void matmul_f32_trans_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint tid = lid.y * 16 + lid.x;
    const uint thread_row = tid / (BN / RN);
    const uint thread_col = tid % (BN / RN);

    threadgroup float As[BM][BK];
    threadgroup float Bts[BK][BN];  // B^T loaded as: B^T[k, n] = B[n * K + k]

    float acc[RM][RN] = {};

    for (uint bk = 0; bk < K; bk += BK) {
        for (uint i = 0; i < (BM * BK) / 256; i++) {
            uint idx = tid + i * 256;
            uint ar = idx / BK;
            uint ac = idx % BK;
            uint gr = tgid.y * BM + ar;
            uint gc = bk + ac;
            As[ar][ac] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        // Load B^T: B^T[k, n] = B[n, k] = B[n * K + k]
        for (uint i = 0; i < (BK * BN) / 256; i++) {
            uint idx = tid + i * 256;
            uint br = idx / BN;
            uint bc = idx % BN;
            uint gr = bk + br;       // k dimension
            uint gc = tgid.x * BN + bc;  // n dimension
            Bts[br][bc] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK; k++) {
            float a_vals[RM], b_vals[RN];
            for (uint ri = 0; ri < RM; ri++) a_vals[ri] = As[thread_row * RM + ri][k];
            for (uint ci = 0; ci < RN; ci++) b_vals[ci] = Bts[k][thread_col * RN + ci];
            for (uint ri = 0; ri < RM; ri++)
                for (uint ci = 0; ci < RN; ci++)
                    acc[ri][ci] += a_vals[ri] * b_vals[ci];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint ri = 0; ri < RM; ri++) {
        for (uint ci = 0; ci < RN; ci++) {
            uint gr = tgid.y * BM + thread_row * RM + ri;
            uint gc = tgid.x * BN + thread_col * RN + ci;
            if (gr < M && gc < N)
                C[gr * N + gc] = acc[ri][ci];
        }
    }
}

// ============================================================
// Phase 3: QLoRA Training Kernels
// ============================================================

// --- Batched Quantized Matmul Params ---

struct BatchedQuantParams {
    uint out_dim;
    uint in_dim;
    uint num_blocks;   // in_dim / 32
    uint row_bytes;    // bytes per weight row
    uint M;            // number of input rows
};

// --- Transposed Quantized Matmul: grad_x = grad_out @ W^T ---
// W: (out_dim, in_dim) quantized, grad_out: (M, out_dim), grad_x: (M, in_dim)
// grad_x[m][k] += sum_j(grad_out[m][j] * W[j][k])
// Dequantize W on-the-fly and compute dot product along out_dim

// Q4_0 transposed: each thread computes one (m, k) element
kernel void matmul_q4_0_trans_batched(
    const device uchar* weight [[ buffer(0) ]],    // (out_dim, in_dim) Q4_0
    const device float* grad_out [[ buffer(1) ]],   // (M, out_dim)
    device float* grad_x [[ buffer(2) ]],            // (M, in_dim)
    constant BatchedQuantParams& params [[ buffer(3) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint2 lid [[ thread_position_in_threadgroup ]]
) {
    const uint k = tgid.x * 16 + lid.x;  // in_dim column
    const uint m = tgid.y * 16 + lid.y;  // batch row
    if (k >= params.in_dim || m >= params.M) return;

    // W^T[k, j] = W[j, k] — need to access column k across all rows j
    const uint block_idx = k / 32;
    const uint in_block = k % 32;
    const uint byte_idx = in_block < 16 ? in_block : in_block - 16;
    const bool is_hi = in_block >= 16;

    float acc = 0.0f;
    for (uint j = 0; j < params.out_dim; j++) {
        // Dequantize W[j][k]
        const device uchar* block = weight + j * params.row_bytes + block_idx * 18;
        float d = float(*reinterpret_cast<const device half*>(block));
        uchar byte_val = block[2 + byte_idx];
        float q = is_hi ? float(byte_val >> 4) : float(byte_val & 0xF);
        float w_val = d * (q - 8.0f);

        acc += grad_out[m * params.out_dim + j] * w_val;
    }

    grad_x[m * params.in_dim + k] += acc;
}

// Q4_1 transposed: each thread computes one (m, k) element
kernel void matmul_q4_1_trans_batched(
    const device uchar* weight [[ buffer(0) ]],    // (out_dim, in_dim) Q4_1
    const device float* grad_out [[ buffer(1) ]],   // (M, out_dim)
    device float* grad_x [[ buffer(2) ]],            // (M, in_dim)
    constant BatchedQuantParams& params [[ buffer(3) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint2 lid [[ thread_position_in_threadgroup ]]
) {
    const uint k = tgid.x * 16 + lid.x;
    const uint m = tgid.y * 16 + lid.y;
    if (k >= params.in_dim || m >= params.M) return;

    const uint block_idx = k / 32;
    const uint in_block = k % 32;
    const uint byte_idx = in_block < 16 ? in_block : in_block - 16;
    const bool is_hi = in_block >= 16;

    float acc = 0.0f;
    for (uint j = 0; j < params.out_dim; j++) {
        const device uchar* block = weight + j * params.row_bytes + block_idx * 20;
        float w_scale = float(*reinterpret_cast<const device half*>(block));
        float w_min = float(*reinterpret_cast<const device half*>(block + 2));
        uchar byte_val = block[4 + byte_idx];
        float q = is_hi ? float(byte_val >> 4) : float(byte_val & 0xF);
        float w_val = w_scale * q + w_min;

        acc += grad_out[m * params.out_dim + j] * w_val;
    }

    grad_x[m * params.in_dim + k] += acc;
}

// Q8_0 transposed: each thread computes one (m, k) element
kernel void matmul_q8_0_trans_batched(
    const device uchar* weight [[ buffer(0) ]],    // (out_dim, in_dim) Q8_0
    const device float* grad_out [[ buffer(1) ]],   // (M, out_dim)
    device float* grad_x [[ buffer(2) ]],            // (M, in_dim)
    constant BatchedQuantParams& params [[ buffer(3) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint2 lid [[ thread_position_in_threadgroup ]]
) {
    const uint k = tgid.x * 16 + lid.x;
    const uint m = tgid.y * 16 + lid.y;
    if (k >= params.in_dim || m >= params.M) return;

    const uint block_idx = k / 32;
    const uint in_block = k % 32;

    float acc = 0.0f;
    for (uint j = 0; j < params.out_dim; j++) {
        const device uchar* block = weight + j * params.row_bytes + block_idx * 34;
        float w_scale = float(*reinterpret_cast<const device half*>(block));
        float q = float(reinterpret_cast<const device char*>(block + 2)[in_block]);
        float w_val = w_scale * q;

        acc += grad_out[m * params.out_dim + j] * w_val;
    }

    grad_x[m * params.in_dim + k] += acc;
}

// --- RMSNorm Training (forward saves inv_rms, backward computes grad_x and grad_weight) ---

struct RMSNormTrainParams {
    uint rows;
    uint dim;
    float eps;
};

// RMSNorm forward (training): y = x * inv_rms * weight, save inv_rms
kernel void rmsnorm_forward_training(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    device float* inv_rms_out [[ buffer(3) ]],
    constant RMSNormTrainParams& params [[ buffer(4) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint lid [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    if (row >= params.rows) return;
    uint dim = params.dim;
    uint base = row * dim;

    threadgroup float shared[256];

    // Sum of squares
    float local_ss = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float v = x[base + i];
        local_ss += v * v;
    }
    shared[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared[0] / float(dim) + params.eps);
    if (lid == 0) inv_rms_out[row] = inv_rms;

    for (uint i = lid; i < dim; i += tg_size) {
        out[base + i] = x[base + i] * inv_rms * weight[i];
    }
}

// RMSNorm backward for x:
// grad_x[i] = inv_rms * weight[i] * grad_out[i]
//           - inv_rms^3 * x[i] * sum(x[j] * weight[j] * grad_out[j]) / dim
kernel void rmsnorm_backward_x(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device const float* grad_out [[ buffer(2) ]],
    device const float* inv_rms_buf [[ buffer(3) ]],
    device float* grad_x [[ buffer(4) ]],
    constant RMSNormTrainParams& params [[ buffer(5) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint lid [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    if (row >= params.rows) return;
    uint dim = params.dim;
    uint base = row * dim;
    float inv_rms = inv_rms_buf[row];

    threadgroup float shared[256];

    // sum_xwg = sum(x[i] * weight[i] * grad_out[i])
    float local_sum = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        local_sum += x[base + i] * weight[i] * grad_out[base + i];
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_xwg = shared[0];

    // grad_x[i] = inv_rms * weight[i] * grad_out[i] - inv_rms^3 * x[i] * sum_xwg / dim
    float coeff = inv_rms * inv_rms * inv_rms * sum_xwg / float(dim);
    for (uint i = lid; i < dim; i += tg_size) {
        grad_x[base + i] += inv_rms * weight[i] * grad_out[base + i] - coeff * x[base + i];
    }
}

// RMSNorm backward for weight (reduce over batch/rows):
// grad_weight[i] += sum_row(inv_rms[row] * x[row,i] * grad_out[row,i])
kernel void rmsnorm_backward_weight(
    device const float* x [[ buffer(0) ]],
    device const float* grad_out [[ buffer(1) ]],
    device const float* inv_rms_buf [[ buffer(2) ]],
    device float* grad_weight [[ buffer(3) ]],
    constant RMSNormTrainParams& params [[ buffer(4) ]],
    uint i [[ thread_position_in_grid ]]
) {
    if (i >= params.dim) return;
    uint dim = params.dim;

    float sum = 0.0f;
    for (uint row = 0; row < params.rows; row++) {
        sum += inv_rms_buf[row] * x[row * dim + i] * grad_out[row * dim + i];
    }
    grad_weight[i] += sum;
}

// --- RoPE Training (forward saves sin/cos, backward applies inverse rotation) ---

struct RoPETrainParams {
    uint seq_len;      // M (number of positions)
    uint n_heads;
    uint half_dim;     // head_dim / 2
};

// RoPE forward (training): apply rotation and save sin/cos for backward
// x: (M, n_heads * head_dim), freqs: (half_dim,), positions: [0..M-1]
kernel void rope_forward_training(
    device float* x [[ buffer(0) ]],              // input/output (in-place)
    device const float* freqs [[ buffer(1) ]],    // (half_dim,) precomputed freq table
    device float* sin_cache [[ buffer(2) ]],      // (M * half_dim,) saved for backward
    device float* cos_cache [[ buffer(3) ]],      // (M * half_dim,) saved for backward
    constant RoPETrainParams& params [[ buffer(4) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint total = params.seq_len * params.n_heads * params.half_dim;
    if (tid >= total) return;

    uint half_dim = params.half_dim;
    uint head_dim = half_dim * 2;
    uint n_heads = params.n_heads;

    uint temp = tid;
    uint i = temp % half_dim;
    temp /= half_dim;
    uint h = temp % n_heads;
    uint m = temp / n_heads;  // position

    float theta = float(m) * freqs[i];
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    // Save sin/cos for backward
    uint sc_idx = m * half_dim + i;
    sin_cache[sc_idx] = sin_t;
    cos_cache[sc_idx] = cos_t;

    // Apply rotation
    uint base = m * (n_heads * head_dim) + h * head_dim + i * 2;
    float x0 = x[base];
    float x1 = x[base + 1];
    x[base]     = x0 * cos_t - x1 * sin_t;
    x[base + 1] = x0 * sin_t + x1 * cos_t;
}

// RoPE backward: inverse rotation
// grad_x0 = grad_y0 * cos + grad_y1 * sin
// grad_x1 = -grad_y0 * sin + grad_y1 * cos
kernel void rope_backward(
    device float* grad [[ buffer(0) ]],            // input/output grad (in-place)
    device const float* sin_cache [[ buffer(1) ]], // (M * half_dim,)
    device const float* cos_cache [[ buffer(2) ]], // (M * half_dim,)
    constant RoPETrainParams& params [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint total = params.seq_len * params.n_heads * params.half_dim;
    if (tid >= total) return;

    uint half_dim = params.half_dim;
    uint head_dim = half_dim * 2;
    uint n_heads = params.n_heads;

    uint temp = tid;
    uint i = temp % half_dim;
    temp /= half_dim;
    uint h = temp % n_heads;
    uint m = temp / n_heads;

    uint sc_idx = m * half_dim + i;
    float sin_t = sin_cache[sc_idx];
    float cos_t = cos_cache[sc_idx];

    uint base = m * (n_heads * head_dim) + h * head_dim + i * 2;
    float g0 = grad[base];
    float g1 = grad[base + 1];
    // Inverse rotation (transpose of rotation matrix)
    grad[base]     = g0 * cos_t + g1 * sin_t;
    grad[base + 1] = -g0 * sin_t + g1 * cos_t;
}

// --- Batched embedding dequant (Q8_0): multiple tokens at once ---

struct BatchedDequantParams {
    uint num_tokens;
    uint embed_dim;
};

// Dequantize multiple Q8_0 embedding rows (scaled) for sequence input
kernel void dequant_q8_0_batch_scaled(
    const device uchar* weight [[ buffer(0) ]],
    device const uint* token_ids [[ buffer(1) ]],
    device float* output [[ buffer(2) ]],
    constant BatchedDequantParams& params [[ buffer(3) ]],
    constant float& embed_scale [[ buffer(4) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint total = params.num_tokens * params.embed_dim;
    if (tid >= total) return;

    uint tok_idx = tid / params.embed_dim;
    uint dim_idx = tid % params.embed_dim;
    uint token_id = token_ids[tok_idx];

    uint num_blocks = params.embed_dim / 32;
    uint row_bytes = num_blocks * 34;
    const device uchar* row = weight + token_id * row_bytes;

    uint block_idx = dim_idx / 32;
    uint in_block = dim_idx % 32;
    const device uchar* block = row + block_idx * 34;
    float scale = float(*reinterpret_cast<const device half*>(block));
    float q = float(reinterpret_cast<const device char*>(block + 2)[in_block]);
    output[tid] = scale * q * embed_scale;
}

/// C += A @ B (register tiling, accumulate)
/// Used for backward: grad_Q += grad_scores @ K
kernel void matmul_f32_accum(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint tid = lid.y * 16 + lid.x;
    const uint thread_row = tid / (BN / RN);
    const uint thread_col = tid % (BN / RN);

    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];

    float acc[RM][RN] = {};

    for (uint bk = 0; bk < K; bk += BK) {
        for (uint i = 0; i < (BM * BK) / 256; i++) {
            uint idx = tid + i * 256;
            uint ar = idx / BK;
            uint ac = idx % BK;
            uint gr = tgid.y * BM + ar;
            uint gc = bk + ac;
            As[ar][ac] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = 0; i < (BK * BN) / 256; i++) {
            uint idx = tid + i * 256;
            uint br = idx / BN;
            uint bc = idx % BN;
            uint gr = bk + br;
            uint gc = tgid.x * BN + bc;
            Bs[br][bc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK; k++) {
            float a_vals[RM], b_vals[RN];
            for (uint ri = 0; ri < RM; ri++) a_vals[ri] = As[thread_row * RM + ri][k];
            for (uint ci = 0; ci < RN; ci++) b_vals[ci] = Bs[k][thread_col * RN + ci];
            for (uint ri = 0; ri < RM; ri++)
                for (uint ci = 0; ci < RN; ci++)
                    acc[ri][ci] += a_vals[ri] * b_vals[ci];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint ri = 0; ri < RM; ri++) {
        for (uint ci = 0; ci < RN; ci++) {
            uint gr = tgid.y * BM + thread_row * RM + ri;
            uint gc = tgid.x * BN + thread_col * RN + ci;
            if (gr < M && gc < N)
                C[gr * N + gc] += acc[ri][ci];
        }
    }
}

// ============================================================
// Fused Kernels
// ============================================================

// --- matmul + addBias + gelu (forward) ---
// out = gelu(A @ B + bias), pre_act = A @ B + bias
// A: (M, K), B: (K, N), bias: (N,), out: (M, N), pre_act: (M, N)
struct FusedMatmulBiasParams {
    uint M;
    uint K;
    uint N;
};

kernel void matmul_addbias_gelu_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* pre_act [[buffer(4)]],
    constant FusedMatmulBiasParams& params [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint tid = lid.y * 16 + lid.x;
    const uint thread_row = tid / (BN / RN);
    const uint thread_col = tid % (BN / RN);

    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];

    float acc[RM][RN] = {};

    for (uint bk = 0; bk < K; bk += BK) {
        for (uint i = 0; i < (BM * BK) / 256; i++) {
            uint idx = tid + i * 256;
            uint ar = idx / BK;
            uint ac = idx % BK;
            uint gr = tgid.y * BM + ar;
            uint gc = bk + ac;
            As[ar][ac] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = 0; i < (BK * BN) / 256; i++) {
            uint idx = tid + i * 256;
            uint br = idx / BN;
            uint bc = idx % BN;
            uint gr = bk + br;
            uint gc = tgid.x * BN + bc;
            Bs[br][bc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK; k++) {
            float a_vals[RM], b_vals[RN];
            for (uint ri = 0; ri < RM; ri++) a_vals[ri] = As[thread_row * RM + ri][k];
            for (uint ci = 0; ci < RN; ci++) b_vals[ci] = Bs[k][thread_col * RN + ci];
            for (uint ri = 0; ri < RM; ri++)
                for (uint ci = 0; ci < RN; ci++)
                    acc[ri][ci] += a_vals[ri] * b_vals[ci];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write: add bias + gelu
    const float c = 0.7978845608f;
    for (uint ri = 0; ri < RM; ri++) {
        for (uint ci = 0; ci < RN; ci++) {
            uint gr = tgid.y * BM + thread_row * RM + ri;
            uint gc = tgid.x * BN + thread_col * RN + ci;
            if (gr < M && gc < N) {
                float v = acc[ri][ci] + bias[gc];
                pre_act[gr * N + gc] = v;
                float inner = c * (v + 0.044715f * v * v * v);
                out[gr * N + gc] = 0.5f * v * (1.0f + precise::tanh(inner));
            }
        }
    }
}

// --- gelu_bias backward ---
// grad_pre_act = grad_out * gelu'(pre_act)
// grad_bias[j] += sum_i(grad_pre_act[i,j])
kernel void gelu_bias_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* pre_act [[buffer(1)]],
    device float* grad_pre_act [[buffer(2)]],
    device float* grad_bias [[buffer(3)]],
    constant BiasParams& params [[buffer(4)]],
    uint j [[thread_position_in_grid]]
) {
    if (j >= params.cols) return;
    const float c = 0.7978845608f;
    float bias_sum = 0.0f;
    for (uint i = 0; i < params.rows; i++) {
        uint idx = i * params.cols + j;
        float v = pre_act[idx];
        float inner = c * (v + 0.044715f * v * v * v);
        float t = precise::tanh(inner);
        float sech2 = 1.0f - t * t;
        float inner_deriv = c * (1.0f + 3.0f * 0.044715f * v * v);
        float gelu_grad = 0.5f * (1.0f + t) + 0.5f * v * sech2 * inner_deriv;
        float gp = grad_out[idx] * gelu_grad;
        grad_pre_act[idx] = gp;
        bias_sum += gp;
    }
    grad_bias[j] += bias_sum;
}

// --- matmul + addBias + tanh (forward) ---
// out = tanh(A @ B + bias)
kernel void matmul_addbias_tanh_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant FusedMatmulBiasParams& params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint tid = lid.y * 16 + lid.x;
    const uint thread_row = tid / (BN / RN);
    const uint thread_col = tid % (BN / RN);

    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];

    float acc[RM][RN] = {};

    for (uint bk = 0; bk < K; bk += BK) {
        for (uint i = 0; i < (BM * BK) / 256; i++) {
            uint idx = tid + i * 256;
            uint ar = idx / BK;
            uint ac = idx % BK;
            uint gr = tgid.y * BM + ar;
            uint gc = bk + ac;
            As[ar][ac] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = 0; i < (BK * BN) / 256; i++) {
            uint idx = tid + i * 256;
            uint br = idx / BN;
            uint bc = idx % BN;
            uint gr = bk + br;
            uint gc = tgid.x * BN + bc;
            Bs[br][bc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK; k++) {
            float a_vals[RM], b_vals[RN];
            for (uint ri = 0; ri < RM; ri++) a_vals[ri] = As[thread_row * RM + ri][k];
            for (uint ci = 0; ci < RN; ci++) b_vals[ci] = Bs[k][thread_col * RN + ci];
            for (uint ri = 0; ri < RM; ri++)
                for (uint ci = 0; ci < RN; ci++)
                    acc[ri][ci] += a_vals[ri] * b_vals[ci];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint ri = 0; ri < RM; ri++) {
        for (uint ci = 0; ci < RN; ci++) {
            uint gr = tgid.y * BM + thread_row * RM + ri;
            uint gc = tgid.x * BN + thread_col * RN + ci;
            if (gr < M && gc < N)
                out[gr * N + gc] = precise::tanh(acc[ri][ci] + bias[gc]);
        }
    }
}

// --- tanh_bias backward ---
// grad_pre_act = grad_out * (1 - tanh_out^2)
// grad_bias[j] += sum_i(grad_pre_act[i,j])
kernel void tanh_bias_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* tanh_out [[buffer(1)]],
    device float* grad_pre_act [[buffer(2)]],
    device float* grad_bias [[buffer(3)]],
    constant BiasParams& params [[buffer(4)]],
    uint j [[thread_position_in_grid]]
) {
    if (j >= params.cols) return;
    float bias_sum = 0.0f;
    for (uint i = 0; i < params.rows; i++) {
        uint idx = i * params.cols + j;
        float t = tanh_out[idx];
        float gp = grad_out[idx] * (1.0f - t * t);
        grad_pre_act[idx] = gp;
        bias_sum += gp;
    }
    grad_bias[j] += bias_sum;
}

// --- batchedMatmulTransB + scale (forward) ---
// C[b] = scale * (A[b] @ B[b]^T)
struct BatchedMatmulScaleParams {
    uint batch;
    uint M;
    uint K;
    uint N;
    float scale;
};

kernel void batched_matmul_trans_b_scale_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulScaleParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint b = tgid.z;
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;
    const float scale_val = params.scale;

    const uint row = tgid.y * TM + lid.y;
    const uint col = tgid.x * TN + lid.x;

    if (b >= params.batch) return;

    const uint a_offset = b * M * K;
    const uint b_offset = b * N * K;
    const uint c_offset = b * M * N;

    threadgroup float As[TM][TK];
    threadgroup float Bs[TK][TN];

    float acc = 0.0f;

    for (uint tk = 0; tk < K; tk += TK) {
        if (row < M && (tk + lid.x) < K)
            As[lid.y][lid.x] = A[a_offset + row * K + tk + lid.x];
        else
            As[lid.y][lid.x] = 0.0f;

        if ((tk + lid.y) < K && col < N)
            Bs[lid.y][lid.x] = B[b_offset + col * K + tk + lid.y];
        else
            Bs[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++)
            acc += As[lid.y][k] * Bs[k][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[c_offset + row * N + col] = acc * scale_val;
}

// ============================================================
// Phase 4: SeqDiffuSeq ops
// ============================================================

// --- Tanh ---

kernel void tanh_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        out[idx] = precise::tanh(x[idx]);
    }
}

kernel void tanh_backward(
    device const float* out [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        float o = out[idx];
        grad_in[idx] += grad_out[idx] * (1.0f - o * o);
    }
}

// --- ConcatLastDim ---

struct ConcatParams {
    uint rows;
    uint cols_a;
    uint cols_b;
};

kernel void concat_last_dim(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ConcatParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint cols_a = params.cols_a;
    uint cols_b = params.cols_b;
    uint cols_total = cols_a + cols_b;
    if (row >= params.rows || col >= cols_total) return;
    if (col < cols_a) {
        out[row * cols_total + col] = a[row * cols_a + col];
    } else {
        out[row * cols_total + col] = b[row * cols_b + (col - cols_a)];
    }
}

kernel void concat_last_dim_backward(
    device const float* grad_out [[buffer(0)]],
    device float* grad_a [[buffer(1)]],
    device float* grad_b [[buffer(2)]],
    constant ConcatParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint cols_a = params.cols_a;
    uint cols_b = params.cols_b;
    uint cols_total = cols_a + cols_b;
    if (row >= params.rows || col >= cols_total) return;
    if (col < cols_a) {
        grad_a[row * cols_a + col] += grad_out[row * cols_total + col];
    } else {
        grad_b[row * cols_b + (col - cols_a)] += grad_out[row * cols_total + col];
    }
}

// ============================================================
// Phase 5: Batched Matmul for attention (grid.z = batch)
// ============================================================

struct BatchedMatmulParams {
    uint batch;
    uint M;
    uint K;
    uint N;
};

/// Batched matmul: C[b] = A[b] @ B[b]
/// A: (batch, M, K), B: (batch, K, N), C: (batch, M, N)
kernel void batched_matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint b = tgid.z;
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint row = tgid.y * TM + lid.y;
    const uint col = tgid.x * TN + lid.x;

    if (b >= params.batch) return;

    const uint a_offset = b * M * K;
    const uint b_offset = b * K * N;
    const uint c_offset = b * M * N;

    threadgroup float As[TM][TK];
    threadgroup float Bs[TK][TN];

    float acc = 0.0f;

    for (uint tk = 0; tk < K; tk += TK) {
        if (row < M && (tk + lid.x) < K)
            As[lid.y][lid.x] = A[a_offset + row * K + tk + lid.x];
        else
            As[lid.y][lid.x] = 0.0f;

        if ((tk + lid.y) < K && col < N)
            Bs[lid.y][lid.x] = B[b_offset + (tk + lid.y) * N + col];
        else
            Bs[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++)
            acc += As[lid.y][k] * Bs[k][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[c_offset + row * N + col] = acc;
}

/// Batched matmul transB: C[b] = A[b] @ B[b]^T
/// A: (batch, M, K), B: (batch, N, K), C: (batch, M, N)
kernel void batched_matmul_trans_b_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint b = tgid.z;
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint row = tgid.y * TM + lid.y;
    const uint col = tgid.x * TN + lid.x;

    if (b >= params.batch) return;

    const uint a_offset = b * M * K;
    const uint b_offset = b * N * K;  // B is (N, K) per batch
    const uint c_offset = b * M * N;

    threadgroup float As[TM][TK];
    threadgroup float Bs[TK][TN];

    float acc = 0.0f;

    for (uint tk = 0; tk < K; tk += TK) {
        if (row < M && (tk + lid.x) < K)
            As[lid.y][lid.x] = A[a_offset + row * K + tk + lid.x];
        else
            As[lid.y][lid.x] = 0.0f;

        // B^T: B[b][col][tk+lid.y] → transposed read
        if ((tk + lid.y) < K && col < N)
            Bs[lid.y][lid.x] = B[b_offset + col * K + tk + lid.y];
        else
            Bs[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++)
            acc += As[lid.y][k] * Bs[k][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[c_offset + row * N + col] = acc;
}

/// Backward for batched matmul: dA[b] = dC[b] @ B[b]^T
/// dC: (batch, M, N), B: (batch, K, N) → dA: (batch, M, K)
kernel void batched_matmul_backward_a_f32(
    device const float* dC [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* dA [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint b = tgid.z;
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint row = tgid.y * TM + lid.y;  // row in M
    const uint col = tgid.x * TN + lid.x;  // col in K

    if (b >= params.batch) return;

    const uint dc_offset = b * M * N;
    const uint b_offset = b * K * N;
    const uint da_offset = b * M * K;

    threadgroup float dCs[TM][TK];
    threadgroup float BTs[TK][TN];

    float acc = 0.0f;

    // dA = dC @ B^T: dC(M,N) @ B^T(N,K) = dA(M,K)
    for (uint tn = 0; tn < N; tn += TK) {
        if (row < M && (tn + lid.x) < N)
            dCs[lid.y][lid.x] = dC[dc_offset + row * N + tn + lid.x];
        else
            dCs[lid.y][lid.x] = 0.0f;

        // B^T[tn+lid.y][col] = B[col][tn+lid.y]
        if ((tn + lid.y) < N && col < K)
            BTs[lid.y][lid.x] = B[b_offset + col * N + tn + lid.y];
        else
            BTs[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++)
            acc += dCs[lid.y][k] * BTs[k][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < K)
        dA[da_offset + row * K + col] += acc;
}

/// Backward for batched matmul: dB[b] = A[b]^T @ dC[b]
/// A: (batch, M, K), dC: (batch, M, N) → dB: (batch, K, N)
kernel void batched_matmul_backward_b_f32(
    device const float* A [[buffer(0)]],
    device const float* dC [[buffer(1)]],
    device float* dB [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint b = tgid.z;
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint row = tgid.y * TM + lid.y;  // row in K
    const uint col = tgid.x * TN + lid.x;  // col in N

    if (b >= params.batch) return;

    const uint a_offset = b * M * K;
    const uint dc_offset = b * M * N;
    const uint db_offset = b * K * N;

    threadgroup float ATs[TM][TK];
    threadgroup float dCs[TK][TN];

    float acc = 0.0f;

    // dB = A^T @ dC: A^T(K,M) @ dC(M,N) = dB(K,N)
    for (uint tm = 0; tm < M; tm += TK) {
        // A^T[row][tm+lid.x] = A[tm+lid.x][row]
        if (row < K && (tm + lid.x) < M)
            ATs[lid.y][lid.x] = A[a_offset + (tm + lid.x) * K + row];
        else
            ATs[lid.y][lid.x] = 0.0f;

        if ((tm + lid.y) < M && col < N)
            dCs[lid.y][lid.x] = dC[dc_offset + (tm + lid.y) * N + col];
        else
            dCs[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++)
            acc += ATs[lid.y][k] * dCs[k][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < K && col < N)
        dB[db_offset + row * N + col] += acc;
}

/// Backward for batched matmulTransB: dA[b] = dC[b] @ B[b]
/// dC: (batch, M, N), B: (batch, N, K) → dA: (batch, M, K)
kernel void batched_matmul_trans_b_backward_a_f32(
    device const float* dC [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* dA [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint b = tgid.z;
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint row = tgid.y * TM + lid.y;
    const uint col = tgid.x * TN + lid.x;

    if (b >= params.batch) return;

    const uint dc_offset = b * M * N;
    const uint b_offset = b * N * K;
    const uint da_offset = b * M * K;

    threadgroup float dCs[TM][TK];
    threadgroup float Bs[TK][TN];

    float acc = 0.0f;

    // dA = dC @ B: dC(M,N) @ B(N,K) = dA(M,K)
    for (uint tn = 0; tn < N; tn += TK) {
        if (row < M && (tn + lid.x) < N)
            dCs[lid.y][lid.x] = dC[dc_offset + row * N + tn + lid.x];
        else
            dCs[lid.y][lid.x] = 0.0f;

        if ((tn + lid.y) < N && col < K)
            Bs[lid.y][lid.x] = B[b_offset + (tn + lid.y) * K + col];
        else
            Bs[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++)
            acc += dCs[lid.y][k] * Bs[k][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < K)
        dA[da_offset + row * K + col] += acc;
}

/// Backward for batched matmulTransB: dB[b] = dC[b]^T @ A[b]
/// dC: (batch, M, N), A: (batch, M, K) → dB: (batch, N, K)
kernel void batched_matmul_trans_b_backward_b_f32(
    device const float* dC [[buffer(0)]],
    device const float* A [[buffer(1)]],
    device float* dB [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint b = tgid.z;
    const uint M = params.M;
    const uint K = params.K;
    const uint N = params.N;

    const uint row = tgid.y * TM + lid.y;  // row in N
    const uint col = tgid.x * TN + lid.x;  // col in K

    if (b >= params.batch) return;

    const uint dc_offset = b * M * N;
    const uint a_offset = b * M * K;
    const uint db_offset = b * N * K;

    threadgroup float dCTs[TM][TK];
    threadgroup float As[TK][TN];

    float acc = 0.0f;

    // dB = dC^T @ A: dC^T(N,M) @ A(M,K) = dB(N,K)
    for (uint tm = 0; tm < M; tm += TK) {
        // dC^T[row][tm+lid.x] = dC[tm+lid.x][row]
        if (row < N && (tm + lid.x) < M)
            dCTs[lid.y][lid.x] = dC[dc_offset + (tm + lid.x) * N + row];
        else
            dCTs[lid.y][lid.x] = 0.0f;

        if ((tm + lid.y) < M && col < K)
            As[lid.y][lid.x] = A[a_offset + (tm + lid.y) * K + col];
        else
            As[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++)
            acc += dCTs[lid.y][k] * As[k][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < N && col < K)
        dB[db_offset + row * K + col] += acc;
}
