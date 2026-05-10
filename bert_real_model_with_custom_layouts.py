from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCH_COMPILE_THREADS", "1")

import torch
import torch.nn as nn

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M  32
#define TILE_N  32
#define BLK     256

template<int LAYOUT>
__device__ __forceinline__ int swiz(int TK, int row, int col) {
    if constexpr (LAYOUT == 0) return col;
    if constexpr (LAYOUT == 1) return col;
    if constexpr (LAYOUT == 2) return col ^ (((row >> 0) & 7) << 0);
    if constexpr (LAYOUT == 3) return col ^ (((row >> 3) & 3) << 3);
    if constexpr (LAYOUT == 4) return col ^ (((row >> 3) & 7) << 3);
    if constexpr (LAYOUT == 5) return (col + row) % TK;
    if constexpr (LAYOUT == 6) return (col + 3 * row) % TK;
    if constexpr (LAYOUT == 7) return (col + row + row / 8) % TK;
    return col;
}

template<int LAYOUT>
__global__ void layout_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, int TK)
{
    int sa = (LAYOUT == 1) ? (TK + 1) : TK;
    extern __shared__ float sm[];
    float* As = sm;
    float* Bs = As + TILE_M * sa;

    int tid = threadIdx.x;
    int bm  = blockIdx.x * TILE_M;
    int bn  = blockIdx.y * TILE_N;

    float acc[4] = {};
    int oidx[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) oidx[i] = tid + i * BLK;

    for (int bk = 0; bk < K; bk += TK) {
        for (int i = tid; i < TILE_M * TK; i += BLK) {
            int r = i / TK, c = i % TK;
            int sc = swiz<LAYOUT>(TK, r, c);
            int gr = bm + r, gc = bk + c;
            As[r * sa + sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        for (int i = tid; i < TK * TILE_N; i += BLK) {
            int r = i / TILE_N, c = i % TILE_N;
            int gr = bk + r, gc = bn + c;
            Bs[r * TILE_N + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int oi = oidx[i];
                if (oi < TILE_M * TILE_N) {
                    int lm = oi / TILE_N, ln = oi % TILE_N;
                    int asc = swiz<LAYOUT>(TK, lm, kk);
                    acc[i] += As[lm * sa + asc] * Bs[kk * TILE_N + ln];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int oi = oidx[i];
        if (oi < TILE_M * TILE_N) {
            int lm = oi / TILE_N, ln = oi % TILE_N;
            int gm = bm + lm, gn = bn + ln;
            if (gm < M && gn < N) C[gm * N + gn] = acc[i];
        }
    }
}

template<int LAYOUT>
void launch(const float* A, const float* B, float* C,
            int M, int N, int K, int TK) {
    int sa = (LAYOUT == 1) ? TK + 1 : TK;
    size_t smem = (size_t)(TILE_M * sa + TK * TILE_N) * sizeof(float);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    layout_gemm<LAYOUT><<<grid, BLK, smem>>>(A, B, C, M, N, K, TK);
}

torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B,
                            int64_t layout, int64_t tile_k) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "only float32 supported");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);

    int Kp = ((K + (int)tile_k - 1) / (int)tile_k) * (int)tile_k;
    torch::Tensor Ap = A, Bp = B;
    if (Kp != K) {
        Ap = torch::zeros({M, Kp}, A.options());
        Ap.narrow(1, 0, K).copy_(A);
        Bp = torch::zeros({Kp, N}, B.options());
        Bp.narrow(0, 0, K).copy_(B);
    }
    auto C = torch::zeros({M, N}, A.options());
    const float* a = Ap.data_ptr<float>();
    const float* b = Bp.data_ptr<float>();
    float* c = C.data_ptr<float>();

    switch (layout) {
        case 0: launch<0>(a, b, c, M, N, Kp, tile_k); break;
        case 1: launch<1>(a, b, c, M, N, Kp, tile_k); break;
        case 2: launch<2>(a, b, c, M, N, Kp, tile_k); break;
        case 3: launch<3>(a, b, c, M, N, Kp, tile_k); break;
        case 4: launch<4>(a, b, c, M, N, Kp, tile_k); break;
        case 5: launch<5>(a, b, c, M, N, Kp, tile_k); break;
        case 6: launch<6>(a, b, c, M, N, Kp, tile_k); break;
        case 7: launch<7>(a, b, c, M, N, Kp, tile_k); break;
    }
    return C;
}

"""

CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B,
                            int64_t layout, int64_t tile_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_matmul", &custom_matmul,
          "Layout-aware tiled GEMM for bank-conflict profiling");
}
"""

_ext = None

def get_ext():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load_inline
    print("JIT-compiling custom GEMM extension (first run takes ~60s)...", flush=True)
    _ext = load_inline(
        name="layout_gemm_ext",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        verbose=os.environ.get("VERBOSE", "0") == "1",
        extra_cuda_cflags=["-std=c++17", "-O3"],
    )
    print("Extension compiled.", flush=True)
    return _ext


LAYOUT_NAMES = {
    0: "naive",
    1: "padded",
    2: "f2_swizzle_3_0_0",
    3: "hexcute_swizzle_2_3_3",
    4: "cute_swizzle_3_3_3",
    5: "isl_affine_a1",
    6: "isl_affine_a3",
    7: "isl_blocked_affine",
}


class CustomLinear(nn.Module):
    """Drop-in replacement for nn.Linear using our layout-aware CUDA GEMM."""
    def __init__(self, original: nn.Linear, layout: int, tile_k: int):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.layout = layout
        self.tile_k = tile_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = get_ext()
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).float()
        w = self.weight.float().t().contiguous()
        out = ext.custom_matmul(x2d, w, self.layout, self.tile_k)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.reshape(*shape[:-1], out.size(-1))


def patch_model(model: nn.Module, layout: int, tile_k: int) -> nn.Module:
    """Replace every nn.Linear in *model* with CustomLinear."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, CustomLinear(child, layout, tile_k))
        else:
            patch_model(child, layout, tile_k)
    return model


def build_bert(seq_len: int, batch: int, device: torch.device):
    from transformers import BertConfig, BertModel
    cfg = BertConfig()
    model = BertModel(cfg).eval().to(device)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    mask = torch.ones_like(ids)
    ttids = torch.zeros_like(ids)
    return model, dict(input_ids=ids, attention_mask=mask, token_type_ids=ttids)


def build_gpt2(seq_len: int, batch: int, device: torch.device):
    from transformers import GPT2Config, GPT2Model
    cfg = GPT2Config()
    model = GPT2Model(cfg).eval().to(device)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    mask = torch.ones((batch, seq_len), device=device, dtype=torch.long)
    return model, dict(input_ids=ids, attention_mask=mask)


def run(args):
    device = torch.device("cuda")

    if args.model == "bert":
        model, inputs = build_bert(args.seq_len, args.batch, device)
    else:
        model, inputs = build_gpt2(args.seq_len, args.batch, device)

    model = patch_model(model, args.layout, args.tile_k)

    print(f"Model: {args.model}, Layout: {args.layout} ({LAYOUT_NAMES.get(args.layout, '?')}), "
          f"tile_k: {args.tile_k}, seq_len: {args.seq_len}, batch: {args.batch}")

    for _ in range(args.warmup):
        with torch.no_grad():
            model(**inputs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad():
            out = model(**inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    info = dict(
        model=args.model,
        layout=args.layout,
        layout_name=LAYOUT_NAMES.get(args.layout, "unknown"),
        tile_k=args.tile_k,
        seq_len=args.seq_len,
        batch=args.batch,
        iters=args.iters,
        elapsed_s=elapsed,
        per_iter_ms=elapsed * 1000.0 / args.iters,
        output_shape=list(out.last_hidden_state.shape),
        device=torch.cuda.get_device_name(),
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


def main():
    p = argparse.ArgumentParser(description="Real BERT/GPT-2 with custom shared-memory layouts")
    p.add_argument("--model", choices=["bert", "gpt2"], default="bert")
    p.add_argument("--layout", type=int, default=0,
                   help="Layout ID: 0=naive,2=F2,3=HexCute,5=ISL_a1,7=ISL_blocked")
    p.add_argument("--tile-k", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=int(os.environ.get("SEQ_LEN", "128")))
    p.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", "1")))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--out", default="")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
