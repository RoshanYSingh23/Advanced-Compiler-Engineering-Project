#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_M 32
#define TILE_N 32
#define BLOCK_THREADS 256
#define NUM_LAYOUTS 8
#define STRESS_ITERS 2048

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

const char* layout_name(int layout) {
    switch (layout) {
        case 0: return "naive";
        case 1: return "padded";
        case 2: return "f2_swizzle_3_0_0";
        case 3: return "hexcute_swizzle_2_3_3";
        case 4: return "cute_swizzle_3_3_3";
        case 5: return "isl_affine_a1";
        case 6: return "isl_affine_a3";
        case 7: return "isl_blocked_affine";
        default: return "unknown";
    }
}

template<int TK, int LAYOUT>
__device__ __forceinline__ int stride_a() {
    return (LAYOUT == 1) ? (TK + 1) : TK;
}

template<int TK, int LAYOUT>
__device__ __forceinline__ int swizzle_col(int row, int col) {
    if constexpr (LAYOUT == 0) {
        return col;
    } else if constexpr (LAYOUT == 1) {
        return col;
    } else if constexpr (LAYOUT == 2) {
        return col ^ (((row >> 0) & 7) << 0);
    } else if constexpr (LAYOUT == 3) {
        return col ^ (((row >> 3) & 3) << 3);
    } else if constexpr (LAYOUT == 4) {
        return col ^ (((row >> 3) & 7) << 3);
    } else if constexpr (LAYOUT == 5) {
        return (col + row) % TK;
    } else if constexpr (LAYOUT == 6) {
        return (col + 3 * row) % TK;
    } else {
        return (col + row + (row / 8)) % TK;
    }
}

template<int TK, int LAYOUT, int TAG>
__global__ void hw_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int STRIDE_A = (LAYOUT == 1) ? (TK + 1) : TK;

    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = As + TILE_M * STRIDE_A;

    int tid = threadIdx.x;
    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int out_idx[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) out_idx[i] = tid + i * BLOCK_THREADS;

    for (int bk = 0; bk < K; bk += TK) {
        for (int i = tid; i < TILE_M * TK; i += BLOCK_THREADS) {
            int r = i / TK;
            int c = i % TK;
            int sc = swizzle_col<TK, LAYOUT>(r, c);
            int gr = block_m + r;
            int gc = bk + c;
            As[r * STRIDE_A + sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        for (int i = tid; i < TK * TILE_N; i += BLOCK_THREADS) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int gr = bk + r;
            int gc = block_n + c;
            Bs[r * TILE_N + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int oi = out_idx[i];
                if (oi < TILE_M * TILE_N) {
                    int lm = oi / TILE_N;
                    int ln = oi % TILE_N;
                    int asc = swizzle_col<TK, LAYOUT>(lm, kk);
                    acc[i] += As[lm * STRIDE_A + asc] * Bs[kk * TILE_N + ln];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int oi = out_idx[i];
        if (oi < TILE_M * TILE_N) {
            int lm = oi / TILE_N;
            int ln = oi % TILE_N;
            int gm = block_m + lm;
            int gn = block_n + ln;
            if (gm < M && gn < N) {
                C[gm * N + gn] = acc[i];
            }
        }
    }
}

__global__ void hw_layernorm_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 0.999f + 0.001f;
}

__global__ void hw_gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
    }
}

__global__ void hw_softmax_like_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        x[row * cols + c] = v / (1.f + fabsf(v));
    }
}

bool valid_layout_for_tile(int tile_k, int layout) {
    if (layout == 4 && tile_k < 64) return false;
    return true;
}

size_t smem_bytes(int tile_k, int layout) {
    int stride = (layout == 1) ? tile_k + 1 : tile_k;
    return (TILE_M * stride + tile_k * TILE_N) * sizeof(float);
}

template<int TK, int LAYOUT, int TAG>
void launch_static(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    hw_gemm_kernel<TK, LAYOUT, TAG><<<grid, BLOCK_THREADS, smem_bytes(TK, LAYOUT)>>>(A, B, C, M, N, K);
}

#define DISPATCH_LAYOUT(TK, TAG) \
    switch (layout) { \
        case 0: launch_static<TK,0,TAG>(A,B,C,M,N,K); break; \
        case 1: launch_static<TK,1,TAG>(A,B,C,M,N,K); break; \
        case 2: launch_static<TK,2,TAG>(A,B,C,M,N,K); break; \
        case 3: launch_static<TK,3,TAG>(A,B,C,M,N,K); break; \
        case 4: launch_static<TK,4,TAG>(A,B,C,M,N,K); break; \
        case 5: launch_static<TK,5,TAG>(A,B,C,M,N,K); break; \
        case 6: launch_static<TK,6,TAG>(A,B,C,M,N,K); break; \
        case 7: launch_static<TK,7,TAG>(A,B,C,M,N,K); break; \
    }

void launch_gemm(int tile_k, int layout, int tag,
                 const float* A, const float* B, float* C,
                 int M, int N, int K) {
    if (!valid_layout_for_tile(tile_k, layout) || K % tile_k != 0) return;
    if (tag == 0) {
        if (tile_k == 32) { DISPATCH_LAYOUT(32, 0); }
        else if (tile_k == 64) { DISPATCH_LAYOUT(64, 0); }
        else if (tile_k == 128) { DISPATCH_LAYOUT(128, 0); }
    } else if (tag == 1) {
        if (tile_k == 32) { DISPATCH_LAYOUT(32, 1); }
        else if (tile_k == 64) { DISPATCH_LAYOUT(64, 1); }
        else if (tile_k == 128) { DISPATCH_LAYOUT(128, 1); }
    } else if (tag == 2) {
        if (tile_k == 32) { DISPATCH_LAYOUT(32, 2); }
        else if (tile_k == 64) { DISPATCH_LAYOUT(64, 2); }
        else if (tile_k == 128) { DISPATCH_LAYOUT(128, 2); }
    } else {
        if (tile_k == 32) { DISPATCH_LAYOUT(32, 3); }
        else if (tile_k == 64) { DISPATCH_LAYOUT(64, 3); }
        else if (tile_k == 128) { DISPATCH_LAYOUT(128, 3); }
    }
}

#undef DISPATCH_LAYOUT

void init_array(float* d, size_t count, int seed) {
    float* h = (float*)malloc(count * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        h[i] = (float)(((int)((i * 17 + seed * 31) % 257) - 128) / 128.0);
    }
    CHECK_CUDA(cudaMemcpy(d, h, count * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)A[m * K + k] * (double)B[k * N + n];
            C[m * N + n] = (float)acc;
        }
    }
}

int run_correctness() {
    const int M = 64, N = 64, K = 128;
    size_t sA = (size_t)M * K, sB = (size_t)K * N, sC = (size_t)M * N;
    float *hA = (float*)malloc(sA * sizeof(float));
    float *hB = (float*)malloc(sB * sizeof(float));
    float *hC = (float*)malloc(sC * sizeof(float));
    float *hRef = (float*)malloc(sC * sizeof(float));
    for (size_t i = 0; i < sA; ++i) hA[i] = (float)(((i * 13) % 101) - 50) / 64.0f;
    for (size_t i = 0; i < sB; ++i) hB[i] = (float)(((i * 7) % 97) - 48) / 64.0f;
    cpu_gemm(hA, hB, hRef, M, N, K);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, sB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, sC * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, sA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sB * sizeof(float), cudaMemcpyHostToDevice));

    int failures = 0;
    for (int tile_k : {32, 64, 128}) {
        for (int layout = 0; layout < NUM_LAYOUTS; ++layout) {
            if (!valid_layout_for_tile(tile_k, layout)) continue;
            CHECK_CUDA(cudaMemset(dC, 0, sC * sizeof(float)));
            launch_gemm(tile_k, layout, 0, dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(hC, dC, sC * sizeof(float), cudaMemcpyDeviceToHost));
            double max_err = 0.0;
            for (size_t i = 0; i < sC; ++i) {
                double e = fabs((double)hC[i] - (double)hRef[i]);
                if (e > max_err) max_err = e;
            }
            printf("correctness,tile_k=%d,layout=%s,max_abs_error=%.6g\n",
                   tile_k, layout_name(layout), max_err);
            if (max_err > 1e-3) failures++;
        }
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hRef);
    return failures == 0 ? 0 : 1;
}

struct Shape {
    const char* model;
    const char* layer;
    int tag;
    int M, N, K;
};

const char* stress_pattern_name(int pattern) {
    switch (pattern) {
        case 10: return "stress_col0_row_lane";
        case 11: return "stress_col0_row_2lane";
        case 12: return "stress_col_lane_row_lane";
        case 13: return "stress_col_3lane_row_lane";
        default: return "stress_unknown";
    }
}

template<int TK, int LAYOUT, int PATTERN>
__global__ void hw_stress_kernel(float* out) {
    constexpr int STRIDE_A = (LAYOUT == 1) ? (TK + 1) : TK;
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    for (int i = tid; i < TILE_M * TK; i += BLOCK_THREADS) {
        int r = i / TK;
        int c = i % TK;
        int sc = swizzle_col<TK, LAYOUT>(r, c);
        smem[r * STRIDE_A + sc] = (float)((r * 17 + c * 3 + 1) & 255) * 0.001f;
    }
    __syncthreads();

    int lane = tid & 31;
    int row = lane;
    int col = 0;
    if constexpr (PATTERN == 11) {
        row = (2 * lane) & 31;
        col = 0;
    } else if constexpr (PATTERN == 12) {
        row = lane;
        col = lane % TK;
    } else if constexpr (PATTERN == 13) {
        row = lane;
        col = (3 * lane) % TK;
    }

    int sc = swizzle_col<TK, LAYOUT>(row, col);
    volatile float* vsmem = smem;
    float acc = 0.f;
    #pragma unroll 1
    for (int it = 0; it < STRESS_ITERS; ++it) {
        acc += vsmem[row * STRIDE_A + sc];
    }
    out[blockIdx.x * blockDim.x + tid] = acc;
}

#define LAUNCH_STRESS_LAYOUT(TK, PATTERN) \
    switch (layout) { \
        case 0: hw_stress_kernel<TK,0,PATTERN><<<64, BLOCK_THREADS, TILE_M * ((TK) + 0) * sizeof(float)>>>(out); break; \
        case 1: hw_stress_kernel<TK,1,PATTERN><<<64, BLOCK_THREADS, TILE_M * ((TK) + 1) * sizeof(float)>>>(out); break; \
        case 2: hw_stress_kernel<TK,2,PATTERN><<<64, BLOCK_THREADS, TILE_M * ((TK) + 0) * sizeof(float)>>>(out); break; \
        case 3: hw_stress_kernel<TK,3,PATTERN><<<64, BLOCK_THREADS, TILE_M * ((TK) + 0) * sizeof(float)>>>(out); break; \
        case 4: hw_stress_kernel<TK,4,PATTERN><<<64, BLOCK_THREADS, TILE_M * ((TK) + 0) * sizeof(float)>>>(out); break; \
        case 5: hw_stress_kernel<TK,5,PATTERN><<<64, BLOCK_THREADS, TILE_M * ((TK) + 0) * sizeof(float)>>>(out); break; \
        case 6: hw_stress_kernel<TK,6,PATTERN><<<64, BLOCK_THREADS, TILE_M * ((TK) + 0) * sizeof(float)>>>(out); break; \
        case 7: hw_stress_kernel<TK,7,PATTERN><<<64, BLOCK_THREADS, TILE_M * ((TK) + 0) * sizeof(float)>>>(out); break; \
    }

void launch_stress(int tile_k, int layout, int pattern, float* out) {
    if (!valid_layout_for_tile(tile_k, layout)) return;
    if (pattern == 10) {
        if (tile_k == 32) { LAUNCH_STRESS_LAYOUT(32, 10); }
        else if (tile_k == 64) { LAUNCH_STRESS_LAYOUT(64, 10); }
        else if (tile_k == 128) { LAUNCH_STRESS_LAYOUT(128, 10); }
    } else if (pattern == 11) {
        if (tile_k == 32) { LAUNCH_STRESS_LAYOUT(32, 11); }
        else if (tile_k == 64) { LAUNCH_STRESS_LAYOUT(64, 11); }
        else if (tile_k == 128) { LAUNCH_STRESS_LAYOUT(128, 11); }
    } else if (pattern == 12) {
        if (tile_k == 32) { LAUNCH_STRESS_LAYOUT(32, 12); }
        else if (tile_k == 64) { LAUNCH_STRESS_LAYOUT(64, 12); }
        else if (tile_k == 128) { LAUNCH_STRESS_LAYOUT(128, 12); }
    } else if (pattern == 13) {
        if (tile_k == 32) { LAUNCH_STRESS_LAYOUT(32, 13); }
        else if (tile_k == 64) { LAUNCH_STRESS_LAYOUT(64, 13); }
        else if (tile_k == 128) { LAUNCH_STRESS_LAYOUT(128, 13); }
    }
}

#undef LAUNCH_STRESS_LAYOUT

int run_stress_kernels() {
    printf("mode,stress-kernels,iters=%d\n", STRESS_ITERS);
    float* out;
    CHECK_CUDA(cudaMalloc(&out, 64 * BLOCK_THREADS * sizeof(float)));
    for (int tile_k : {32, 64, 128}) {
        for (int pattern : {10, 11, 12, 13}) {
            for (int layout = 0; layout < NUM_LAYOUTS; ++layout) {
                if (!valid_layout_for_tile(tile_k, layout)) continue;
                printf("launch,stress_pattern=%s,tile_k=%d,layout=%s\n",
                       stress_pattern_name(pattern), tile_k, layout_name(layout));
                launch_stress(tile_k, layout, pattern, out);
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
    }
    CHECK_CUDA(cudaFree(out));
    return 0;
}

Shape profile_shapes[] = {
    {"bert", "qkv_proj", 0, 512, 2304, 768},
    {"bert", "attn_score", 1, 512, 512, 64},
    {"bert", "ffn_down", 2, 512, 768, 3072},
    {"gpt2", "qkv_proj", 3, 1024, 2304, 768},
};

int run_profile_kernels() {
    printf("mode,profile-kernels\n");
    for (const Shape& sh : profile_shapes) {
        size_t sA = (size_t)sh.M * sh.K;
        size_t sB = (size_t)sh.K * sh.N;
        size_t sC = (size_t)sh.M * sh.N;
        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, sA * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dB, sB * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dC, sC * sizeof(float)));
        init_array(dA, sA, sh.tag + 1);
        init_array(dB, sB, sh.tag + 11);
        for (int tile_k : {32, 64, 128}) {
            if (sh.K % tile_k != 0) continue;
            for (int layout = 0; layout < NUM_LAYOUTS; ++layout) {
                if (!valid_layout_for_tile(tile_k, layout)) continue;
                printf("launch,model=%s,layer=%s,tile_k=%d,layout=%s\n",
                       sh.model, sh.layer, tile_k, layout_name(layout));
                launch_gemm(tile_k, layout, sh.tag, dA, dB, dC, sh.M, sh.N, sh.K);
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));
    }
    return 0;
}

void run_pointwise(float* x, int n, int rows, int cols) {
    int blocks = (n + 255) / 256;
    hw_layernorm_kernel<<<blocks, 256>>>(x, n);
    hw_gelu_kernel<<<blocks, 256>>>(x, n);
    hw_softmax_like_kernel<<<rows, 256>>>(x, rows, cols);
}

int run_custom_model(const char* model, int layout, int tile_k, int layers) {
    int seq = strcmp(model, "gpt2") == 0 ? 1024 : 512;
    int hidden = 768;
    int inter = 3072;
    int qkv = 2304;
    int attn_cols = seq;
    int head_dim = 64;

    size_t max_elems = (size_t)seq * inter;
    float *buf_a, *buf_b, *buf_c, *w_qkv, *w_out, *w_up, *w_down, *w_attn;
    CHECK_CUDA(cudaMalloc(&buf_a, max_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buf_b, max_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buf_c, max_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_qkv, (size_t)hidden * qkv * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_out, (size_t)hidden * hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_up, (size_t)hidden * inter * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_down, (size_t)inter * hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_attn, (size_t)seq * head_dim * sizeof(float)));
    init_array(buf_a, (size_t)seq * hidden, 1);
    init_array(w_qkv, (size_t)hidden * qkv, 2);
    init_array(w_out, (size_t)hidden * hidden, 3);
    init_array(w_up, (size_t)hidden * inter, 4);
    init_array(w_down, (size_t)inter * hidden, 5);
    init_array(w_attn, (size_t)seq * head_dim, 6);

    printf("mode,custom-model,model=%s,layout=%s,tile_k=%d,layers=%d\n",
           model, layout_name(layout), tile_k, layers);

    for (int l = 0; l < layers; ++l) {
        launch_gemm(tile_k, layout, 0, buf_a, w_qkv, buf_b, seq, qkv, hidden);
        CHECK_CUDA(cudaDeviceSynchronize());
        launch_gemm(tile_k, layout, 1, buf_a, buf_a, buf_c, seq, attn_cols, hidden);
        CHECK_CUDA(cudaDeviceSynchronize());
        run_pointwise(buf_c, (size_t)seq * attn_cols, seq, attn_cols);
        CHECK_CUDA(cudaDeviceSynchronize());
        launch_gemm(tile_k, layout, 2, buf_c, w_attn, buf_b, seq, head_dim, seq);
        CHECK_CUDA(cudaDeviceSynchronize());
        launch_gemm(tile_k, layout, 3, buf_a, w_out, buf_b, seq, hidden, hidden);
        CHECK_CUDA(cudaDeviceSynchronize());
        run_pointwise(buf_b, (size_t)seq * hidden, seq, hidden);
        CHECK_CUDA(cudaDeviceSynchronize());
        launch_gemm(tile_k, layout, 0, buf_a, w_up, buf_b, seq, inter, hidden);
        CHECK_CUDA(cudaDeviceSynchronize());
        hw_gelu_kernel<<<((seq * inter) + 255) / 256, 256>>>(buf_b, seq * inter);
        CHECK_CUDA(cudaDeviceSynchronize());
        launch_gemm(tile_k, layout, 3, buf_b, w_down, buf_a, seq, hidden, inter);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(buf_a));
    CHECK_CUDA(cudaFree(buf_b));
    CHECK_CUDA(cudaFree(buf_c));
    CHECK_CUDA(cudaFree(w_qkv));
    CHECK_CUDA(cudaFree(w_out));
    CHECK_CUDA(cudaFree(w_up));
    CHECK_CUDA(cudaFree(w_down));
    CHECK_CUDA(cudaFree(w_attn));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s --correctness | --profile-kernels | --stress-kernels | --model bert|gpt2 --layout N --tile-k N [--layers N]\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "--correctness") == 0) {
        return run_correctness();
    }
    if (strcmp(argv[1], "--profile-kernels") == 0) {
        return run_profile_kernels();
    }
    if (strcmp(argv[1], "--stress-kernels") == 0) {
        return run_stress_kernels();
    }
    if (strcmp(argv[1], "--model") == 0) {
        const char* model = argc > 2 ? argv[2] : "bert";
        int layout = 5;
        int tile_k = 64;
        int layers = 12;
        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "--layout") == 0 && i + 1 < argc) layout = atoi(argv[++i]);
            else if (strcmp(argv[i], "--tile-k") == 0 && i + 1 < argc) tile_k = atoi(argv[++i]);
            else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = atoi(argv[++i]);
        }
        return run_custom_model(model, layout, tile_k, layers);
    }
    return 2;
}
