#include <metal_stdlib>
using namespace metal;

// ============================================================
// 共通構造体
// ============================================================

struct MatmulParams {
    uint out_dim;
    uint in_dim;
    uint num_blocks;   // in_dim / 32
    uint row_bytes;    // bytes per weight row
};

struct RMSNormParams {
    uint dim;
    uint rows;
    float eps;
};

struct RoPEParams {
    uint half_dim;
    uint n_heads;
    float pos;
};

struct AttentionParams {
    uint n_head;
    uint head_dim;
    uint q_dim;
    uint kv_dim;
    uint kv_start;
    uint kv_end;
    float scale;
};

// ============================================================
// SIMD group 並列 量子化 MatMul カーネル
// ============================================================
// N_SG simd groups per threadgroup, 各 simd group (32 lanes) が 1 出力行を計算
// 各 lane がブロックのサブセットを処理し、simd_sum で集約

// Q4_0: llama.cpp スタイル - uint16 mask trick + NR0=4 マルチロー
// NR0 行を各 SIMD group が同時処理、入力をレジスタにキャッシュして再利用
// nibble → float 変換はシフト不要: pre-scaled yl × masked uint16 で直接ドット積

#define NR0_Q4 4   // rows per SIMD group
#define NSG_Q4 2   // SIMD groups per threadgroup
#define NQ_Q4 16   // blocks per iteration (32 lanes / 2 lanes per block)

inline float block_q4_0_dot_y(const device uchar* block, float sumy, thread float* yl, int il) {
    float d = float(*reinterpret_cast<const device half*>(block));
    const device ushort* qs = reinterpret_cast<const device ushort*>(block + 2 + il);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int i = 0; i < 8; i += 2) {
        acc0 += yl[i + 0] * float(qs[i / 2] & 0x000Fu);
        acc1 += yl[i + 1] * float(qs[i / 2] & 0x0F00u);
        acc2 += yl[i + 8] * float(qs[i / 2] & 0x00F0u);
        acc3 += yl[i + 9] * float(qs[i / 2] & 0xF000u);
    }
    return d * (sumy * (-8.0f) + acc0 + acc1 + acc2 + acc3);
}

kernel void matmul_q4_0(
    const device uchar* weight [[ buffer(0) ]],
    const device float* input  [[ buffer(1) ]],
    device float* output       [[ buffer(2) ]],
    constant MatmulParams& params [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    const uint base_row = (tgid * NSG_Q4 + sgid) * NR0_Q4;
    if (base_row >= params.out_dim) return;

    const uint nb = params.num_blocks;

    // ix: block index (0-15), il: byte offset within nibble data (0 or 8)
    const uint ix = lane / 2;
    const uint il = (lane % 2) * 8;

    // Pointers to NR0_Q4 weight rows
    const device uchar* ax0 = weight + (base_row + 0) * params.row_bytes;
    const device uchar* ax1 = weight + (base_row + 1) * params.row_bytes;
    const device uchar* ax2 = weight + (base_row + 2) * params.row_bytes;
    const device uchar* ax3 = weight + (base_row + 3) * params.row_bytes;

    float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;

    const device float* yb = input + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ_Q4) {
        // Pre-compute yl[16] with scaling trick (eliminates shift ops)
        float sumy0 = 0.0f, sumy1 = 0.0f;
        float yl[16];

        for (uint i = 0; i < 8; i += 2) {
            sumy0  += yb[i + 0] + yb[i + 1];
            yl[i+0] = yb[i + 0];
            yl[i+1] = yb[i + 1] / 256.0f;

            sumy1  += yb[i + 16] + yb[i + 17];
            yl[i+8] = yb[i + 16] / 16.0f;
            yl[i+9] = yb[i + 17] / 4096.0f;
        }

        float sumy = sumy0 + sumy1;

        // Dot product for each of NR0_Q4 rows (reusing yl)
        sumf0 += block_q4_0_dot_y(ax0 + ib * 18, sumy, yl, il);
        if (base_row + 1 < params.out_dim) sumf1 += block_q4_0_dot_y(ax1 + ib * 18, sumy, yl, il);
        if (base_row + 2 < params.out_dim) sumf2 += block_q4_0_dot_y(ax2 + ib * 18, sumy, yl, il);
        if (base_row + 3 < params.out_dim) sumf3 += block_q4_0_dot_y(ax3 + ib * 18, sumy, yl, il);

        yb += 32 * NQ_Q4;  // advance by NQ_Q4 blocks
    }

    // Reduction
    float s0 = simd_sum(sumf0);
    float s1 = simd_sum(sumf1);
    float s2 = simd_sum(sumf2);
    float s3 = simd_sum(sumf3);
    if (lane == 0) {
        output[base_row] = s0;
        if (base_row + 1 < params.out_dim) output[base_row + 1] = s1;
        if (base_row + 2 < params.out_dim) output[base_row + 2] = s2;
        if (base_row + 3 < params.out_dim) output[base_row + 3] = s3;
    }
}

// Q4_1: f16 scale (2B) + f16 min (2B) + 16B nibbles = 20B per block
// uchar4 + dot() ベクトル化, N_SG=4, 1 row/SG
kernel void matmul_q4_1(
    const device uchar* weight [[ buffer(0) ]],
    const device float* input  [[ buffer(1) ]],
    device float* output       [[ buffer(2) ]],
    constant MatmulParams& params [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    uint row = tgid * 4 + sgid;  // N_SG=4
    if (row >= params.out_dim) return;

    const device uchar* row_data = weight + row * params.row_bytes;
    float partial = 0.0f;

    for (uint bi = lane; bi < params.num_blocks; bi += 32) {
        const device uchar* block = row_data + bi * 20;
        float w_scale = float(*reinterpret_cast<const device half*>(block));
        float w_min = float(*reinterpret_cast<const device half*>(block + 2));
        const device uchar4* nibs4 = reinterpret_cast<const device uchar4*>(block + 4);

        float dot_val = 0.0f;
        float input_sum = 0.0f;

        for (uint k = 0; k < 4; k++) {
            float4 in_lo = *reinterpret_cast<const device float4*>(input + bi * 32 + k * 4);
            float4 in_hi = *reinterpret_cast<const device float4*>(input + bi * 32 + 16 + k * 4);
            uchar4 bytes = nibs4[k];
            float4 lo = float4(bytes.x & 0xF, bytes.y & 0xF, bytes.z & 0xF, bytes.w & 0xF);
            float4 hi = float4(bytes.x >> 4, bytes.y >> 4, bytes.z >> 4, bytes.w >> 4);
            dot_val += dot(lo, in_lo) + dot(hi, in_hi);
            input_sum += dot(float4(1.0f), in_lo) + dot(float4(1.0f), in_hi);
        }
        partial += w_scale * dot_val + w_min * input_sum;
    }

    float sum = simd_sum(partial);
    if (lane == 0) {
        output[row] = sum;
    }
}

// Q8_0: f16 scale (2B) + 32B int8 = 34B per block
// Simple 1 row/SG, N_SG=4 (optimal for large out_dim like logits 262K)
kernel void matmul_q8_0(
    const device uchar* weight [[ buffer(0) ]],
    const device float* input  [[ buffer(1) ]],
    device float* output       [[ buffer(2) ]],
    constant MatmulParams& params [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    uint row = tgid * 4 + sgid;  // N_SG=4
    if (row >= params.out_dim) return;

    const device uchar* row_data = weight + row * params.row_bytes;
    float partial = 0.0f;

    for (uint bi = lane; bi < params.num_blocks; bi += 32) {
        const device uchar* block = row_data + bi * 34;
        float w_scale = float(*reinterpret_cast<const device half*>(block));
        const device char* w_q = reinterpret_cast<const device char*>(block + 2);

        float dot = 0.0f;
        for (uint j = 0; j < 32; j++) {
            dot += float(w_q[j]) * input[bi * 32 + j];
        }
        partial += w_scale * dot;
    }

    float sum = simd_sum(partial);
    if (lane == 0) {
        output[row] = sum;
    }
}

// ============================================================
// RMSNorm カーネル
// ============================================================
// 各 threadgroup が1行を処理

kernel void rmsnorm(
    const device float* input   [[ buffer(0) ]],
    const device float* weight  [[ buffer(1) ]],
    device float* output        [[ buffer(2) ]],
    constant RMSNormParams& params [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    uint row = tgid;
    if (row >= params.rows) return;

    const device float* row_in = input + row * params.dim;
    device float* row_out = output + row * params.dim;

    // 二乗和を並列リダクション
    threadgroup float shared_sum[256];
    float local_ss = 0.0f;
    for (uint i = lid; i < params.dim; i += tg_size) {
        float v = row_in[i];
        local_ss += v * v;
    }
    shared_sum[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(params.dim) + params.eps);

    // 正規化
    for (uint i = lid; i < params.dim; i += tg_size) {
        row_out[i] = row_in[i] * rms_inv * weight[i];
    }
}

// RMSNorm in-place (per-head norm用)
kernel void rmsnorm_inplace(
    device float* x             [[ buffer(0) ]],
    const device float* weight  [[ buffer(1) ]],
    constant uint& dim          [[ buffer(2) ]],
    constant float& eps         [[ buffer(3) ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    threadgroup float shared_sum[256];
    float local_ss = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float v = x[i];
        local_ss += v * v;
    }
    shared_sum[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(dim) + eps);

    for (uint i = lid; i < dim; i += tg_size) {
        x[i] = x[i] * rms_inv * weight[i];
    }
}

// ============================================================
// RMSNorm + Residual Add 融合カーネル
// ============================================================
// residual[i] += RMSNorm(input[i]) * weight[i]
// post-attention/post-FFN norm + residual add を1カーネルに融合

kernel void rmsnorm_residual(
    const device float* input   [[ buffer(0) ]],
    const device float* weight  [[ buffer(1) ]],
    device float* residual      [[ buffer(2) ]],
    constant RMSNormParams& params [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    uint row = tgid;
    if (row >= params.rows) return;

    const device float* row_in = input + row * params.dim;
    device float* row_res = residual + row * params.dim;

    threadgroup float shared_sum[256];
    float local_ss = 0.0f;
    for (uint i = lid; i < params.dim; i += tg_size) {
        float v = row_in[i];
        local_ss += v * v;
    }
    shared_sum[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(params.dim) + params.eps);

    for (uint i = lid; i < params.dim; i += tg_size) {
        row_res[i] += row_in[i] * rms_inv * weight[i];
    }
}

// ============================================================
// RoPE カーネル
// ============================================================
// 各スレッドが1ペアを処理

kernel void rope(
    device float* x             [[ buffer(0) ]],
    const device float* freqs   [[ buffer(1) ]],
    constant RoPEParams& params [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint total_pairs = params.n_heads * params.half_dim;
    if (tid >= total_pairs) return;

    uint head = tid / params.half_dim;
    uint i = tid % params.half_dim;

    float theta = params.pos * freqs[i];
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    uint base = head * (params.half_dim * 2) + i * 2;
    float x0 = x[base];
    float x1 = x[base + 1];
    x[base]     = x0 * cos_t - x1 * sin_t;
    x[base + 1] = x0 * sin_t + x1 * cos_t;
}

// ============================================================
// GELU カーネル (tanh近似)
// ============================================================

kernel void gelu(
    device float* x     [[ buffer(0) ]],
    constant uint& n    [[ buffer(1) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= n) return;
    float v = x[tid];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    // Metal fast::tanh は |inner| > ~44 で NaN を返す (exp(2x) overflow)
    // clamp で防止。tanh(15) ≈ 1.0 なので精度損失なし。
    inner = clamp(inner, -15.0f, 15.0f);
    x[tid] = 0.5f * v * (1.0f + tanh(inner));
}

// ============================================================
// GELU + Mul 融合カーネル
// ============================================================
// gate = GELU(gate) * up を1カーネルで実行

kernel void gelu_mul(
    device float* gate      [[ buffer(0) ]],
    const device float* up  [[ buffer(1) ]],
    constant uint& n        [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= n) return;
    float v = gate[tid];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    inner = clamp(inner, -15.0f, 15.0f);
    gate[tid] = 0.5f * v * (1.0f + tanh(inner)) * up[tid];
}

// ============================================================
// Softmax カーネル (1 threadgroup で1ベクトル)
// ============================================================

kernel void softmax(
    device float* x         [[ buffer(0) ]],
    constant uint& n        [[ buffer(1) ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    threadgroup float shared[256];

    // max
    float local_max = -INFINITY;
    for (uint i = lid; i < n; i += tg_size) {
        local_max = max(local_max, x[i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // exp + sum
    float local_sum = 0.0f;
    for (uint i = lid; i < n; i += tg_size) {
        float e = exp(x[i] - max_val);
        x[i] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // normalize
    for (uint i = lid; i < n; i += tg_size) {
        x[i] *= inv_sum;
    }
}

// ============================================================
// Element-wise カーネル
// ============================================================

kernel void add_inplace(
    device float* a         [[ buffer(0) ]],
    const device float* b   [[ buffer(1) ]],
    constant uint& n        [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= n) return;
    a[tid] += b[tid];
}

kernel void mul_inplace(
    device float* a         [[ buffer(0) ]],
    const device float* b   [[ buffer(1) ]],
    constant uint& n        [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= n) return;
    a[tid] *= b[tid];
}

// scale: x *= scalar
kernel void scale_inplace(
    device float* x         [[ buffer(0) ]],
    constant float& scalar  [[ buffer(1) ]],
    constant uint& n        [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= n) return;
    x[tid] *= scalar;
}

// ============================================================
// Batched Quantized MatMul カーネル (M > 1 入力行対応)
// ============================================================
// weight: (out_dim, in_dim) quantized
// input:  (M, in_dim) f32
// output: (M, out_dim) f32
// output[m][row] = dot(weight_row, input[m])

struct BatchedMatmulParams {
    uint out_dim;
    uint in_dim;
    uint num_blocks;   // in_dim / 32
    uint row_bytes;    // bytes per weight row
    uint M;            // number of input rows (batch/seq_len)
};

// Q4_0 batched: シンプル版 (N_SG=4, 1 row/SG)
// M 入力行を height 次元で並列化
kernel void matmul_q4_0_batched(
    const device uchar* weight [[ buffer(0) ]],
    const device float* input  [[ buffer(1) ]],
    device float* output       [[ buffer(2) ]],
    constant BatchedMatmulParams& params [[ buffer(3) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    uint row = tgid.x * 4 + sgid;  // weight row (out_dim)
    uint m = tgid.y;                // input row (batch)
    if (row >= params.out_dim || m >= params.M) return;

    const device uchar* row_data = weight + row * params.row_bytes;
    const device float* in_row = input + m * params.in_dim;
    float partial = 0.0f;

    for (uint bi = lane; bi < params.num_blocks; bi += 32) {
        const device uchar* block = row_data + bi * 18;
        float d = float(*reinterpret_cast<const device half*>(block));
        const device uchar* qs = block + 2;

        float dot_val = 0.0f;
        for (uint j = 0; j < 16; j++) {
            uchar byte = qs[j];
            float lo = float(byte & 0xF) - 8.0f;
            float hi = float(byte >> 4) - 8.0f;
            dot_val += lo * in_row[bi * 32 + j] + hi * in_row[bi * 32 + 16 + j];
        }
        partial += d * dot_val;
    }

    float sum = simd_sum(partial);
    if (lane == 0) {
        output[m * params.out_dim + row] = sum;
    }
}

// Q4_1 batched: N_SG=4, 1 row/SG, M 次元を height で並列化
kernel void matmul_q4_1_batched(
    const device uchar* weight [[ buffer(0) ]],
    const device float* input  [[ buffer(1) ]],
    device float* output       [[ buffer(2) ]],
    constant BatchedMatmulParams& params [[ buffer(3) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    uint row = tgid.x * 4 + sgid;  // weight row (out_dim)
    uint m = tgid.y;                // input row (batch)
    if (row >= params.out_dim || m >= params.M) return;

    const device uchar* row_data = weight + row * params.row_bytes;
    const device float* in_row = input + m * params.in_dim;
    float partial = 0.0f;

    for (uint bi = lane; bi < params.num_blocks; bi += 32) {
        const device uchar* block = row_data + bi * 20;
        float w_scale = float(*reinterpret_cast<const device half*>(block));
        float w_min = float(*reinterpret_cast<const device half*>(block + 2));
        const device uchar4* nibs4 = reinterpret_cast<const device uchar4*>(block + 4);

        float dot_val = 0.0f;
        float input_sum = 0.0f;

        for (uint k = 0; k < 4; k++) {
            float4 in_lo = *reinterpret_cast<const device float4*>(in_row + bi * 32 + k * 4);
            float4 in_hi = *reinterpret_cast<const device float4*>(in_row + bi * 32 + 16 + k * 4);
            uchar4 bytes = nibs4[k];
            float4 lo = float4(bytes.x & 0xF, bytes.y & 0xF, bytes.z & 0xF, bytes.w & 0xF);
            float4 hi = float4(bytes.x >> 4, bytes.y >> 4, bytes.z >> 4, bytes.w >> 4);
            dot_val += dot(lo, in_lo) + dot(hi, in_hi);
            input_sum += dot(float4(1.0f), in_lo) + dot(float4(1.0f), in_hi);
        }
        partial += w_scale * dot_val + w_min * input_sum;
    }

    float sum = simd_sum(partial);
    if (lane == 0) {
        output[m * params.out_dim + row] = sum;
    }
}

// Q8_0 batched: N_SG=4, 1 row/SG, M 次元を height で並列化
kernel void matmul_q8_0_batched(
    const device uchar* weight [[ buffer(0) ]],
    const device float* input  [[ buffer(1) ]],
    device float* output       [[ buffer(2) ]],
    constant BatchedMatmulParams& params [[ buffer(3) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    uint row = tgid.x * 4 + sgid;  // weight row (out_dim)
    uint m = tgid.y;                // input row (batch)
    if (row >= params.out_dim || m >= params.M) return;

    const device uchar* row_data = weight + row * params.row_bytes;
    const device float* in_row = input + m * params.in_dim;
    float partial = 0.0f;

    for (uint bi = lane; bi < params.num_blocks; bi += 32) {
        const device uchar* block = row_data + bi * 34;
        float w_scale = float(*reinterpret_cast<const device half*>(block));
        const device char* w_q = reinterpret_cast<const device char*>(block + 2);

        float d = 0.0f;
        for (uint j = 0; j < 32; j++) {
            d += float(w_q[j]) * in_row[bi * 32 + j];
        }
        partial += w_scale * d;
    }

    float sum = simd_sum(partial);
    if (lane == 0) {
        output[m * params.out_dim + row] = sum;
    }
}

// ============================================================
// Embedding dequantize (Q8_0)
// ============================================================
// 1 token の embedding を逆量子化

kernel void dequant_q8_0_row(
    const device uchar* weight  [[ buffer(0) ]],
    device float* output        [[ buffer(1) ]],
    constant uint& token_id     [[ buffer(2) ]],
    constant uint& embed_dim    [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= embed_dim) return;

    uint num_blocks = embed_dim / 32;
    uint row_bytes = num_blocks * 34; // Q8_0: 34 bytes per block
    const device uchar* row = weight + token_id * row_bytes;

    uint block_idx = tid / 32;
    uint in_block = tid % 32;
    const device uchar* block = row + block_idx * 34;
    float scale = float(*reinterpret_cast<const device half*>(block));
    float q = float(reinterpret_cast<const device char*>(block + 2)[in_block]);
    output[tid] = scale * q;
}

// Fused: dequant Q8_0 + scale (embedding * sqrt(embed_dim))
kernel void dequant_q8_0_row_scaled(
    const device uchar* weight  [[ buffer(0) ]],
    device float* output        [[ buffer(1) ]],
    constant uint& token_id     [[ buffer(2) ]],
    constant uint& embed_dim    [[ buffer(3) ]],
    constant float& embed_scale [[ buffer(4) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= embed_dim) return;

    uint num_blocks = embed_dim / 32;
    uint row_bytes = num_blocks * 34;
    const device uchar* row = weight + token_id * row_bytes;

    uint block_idx = tid / 32;
    uint in_block = tid % 32;
    const device uchar* block = row + block_idx * 34;
    float scale = float(*reinterpret_cast<const device half*>(block));
    float q = float(reinterpret_cast<const device char*>(block + 2)[in_block]);
    output[tid] = scale * q * embed_scale;
}

// ============================================================
// GQA Cached Attention (decode, single token)
// ============================================================
// 各 threadgroup が 1 Q ヘッドを処理

kernel void gqa_attention_decode(
    const device float* q       [[ buffer(0) ]],   // (Q_DIM,)
    const device float* k_cache [[ buffer(1) ]],   // (CTX, KV_DIM)
    const device float* v_cache [[ buffer(2) ]],   // (CTX, KV_DIM)
    device float* output        [[ buffer(3) ]],   // (Q_DIM,)
    device float* scores_buf    [[ buffer(4) ]],   // (kv_len,) scratch
    constant AttentionParams& params [[ buffer(5) ]],
    uint head_id [[ threadgroup_position_in_grid ]],
    uint lid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    if (head_id >= params.n_head) return;

    uint head_dim = params.head_dim;
    uint kv_dim = params.kv_dim;
    uint q_off = head_id * head_dim;
    uint kv_len = params.kv_end - params.kv_start;

    threadgroup float shared[256];

    // 1. QK dot products → scores
    for (uint ki_rel = lid; ki_rel < kv_len; ki_rel += tg_size) {
        uint ki = params.kv_start + ki_rel;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q[q_off + d] * k_cache[ki * kv_dim + d];
        }
        scores_buf[head_id * kv_len + ki_rel] = dot * params.scale;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // 2. Softmax over scores for this head
    device float* my_scores = scores_buf + head_id * kv_len;

    // max
    float local_max = -INFINITY;
    for (uint i = lid; i < kv_len; i += tg_size) {
        local_max = max(local_max, my_scores[i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // exp + sum
    float local_sum = 0.0f;
    for (uint i = lid; i < kv_len; i += tg_size) {
        float e = exp(my_scores[i] - max_val);
        my_scores[i] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < kv_len; i += tg_size) {
        my_scores[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // 3. Weighted sum of values
    for (uint d = lid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint ki_rel = 0; ki_rel < kv_len; ki_rel++) {
            uint ki = params.kv_start + ki_rel;
            acc += my_scores[ki_rel] * v_cache[ki * kv_dim + d];
        }
        output[q_off + d] = acc;
    }
}

// ============================================================
// KV Cache 書き込み
// ============================================================

kernel void write_kv_cache(
    const device float* k_new   [[ buffer(0) ]],
    const device float* v_new   [[ buffer(1) ]],
    device float* k_cache       [[ buffer(2) ]],
    device float* v_cache       [[ buffer(3) ]],
    constant uint& pos          [[ buffer(4) ]],
    constant uint& kv_dim       [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= kv_dim) return;
    k_cache[pos * kv_dim + tid] = k_new[tid];
    v_cache[pos * kv_dim + tid] = v_new[tid];
}
