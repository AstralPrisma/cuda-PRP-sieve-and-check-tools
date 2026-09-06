// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 AstralPrisma (A.P.)
// Integer two-prime NTT backend, derived from GSRPS 2.3.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace gfpps_ntt {
inline int block_cap = 96;
inline int ntt_block_limit() { return block_cap; }
inline int limited_ntt_grid_y(int items, int threads) {
    return std::max(1, std::min(block_cap, (items + threads - 1) / threads));
}
inline void cuda_check(cudaError_t status, const char* message) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
}
uint32_t pow_mod_host(uint32_t a, uint64_t e, uint32_t p) {
    uint64_t r = 1, x = a;
    while (e != 0) {
        if ((e & 1) != 0) r = (r * x) % p;
        x = (x * x) % p;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

__device__ __forceinline__ uint32_t mul_mod_mont_2prime(uint32_t a, uint32_t b, uint32_t p) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t nprime0 = 998244351u;   // -p0^-1 mod 2^32
    constexpr uint32_t nprime1 = 1004535807u;  // -p1^-1 mod 2^32
    const uint32_t nprime = (p == p0) ? nprime0 : nprime1;
    const uint32_t lo = a * b;
    const uint32_t hi = __umulhi(a, b);
    const uint32_t m = lo * nprime;
    const uint32_t mp_lo = m * p;
    const uint32_t mp_hi = __umulhi(m, p);
    const uint32_t sum_lo = lo + mp_lo;
    uint32_t u = hi + mp_hi + static_cast<uint32_t>(sum_lo < lo);
    if (u >= p) u -= p;
    return u;
}

__device__ __forceinline__ void ntt_stage_one_apply_mont(uint32_t* a, const uint32_t* roots,
                                                          int len, int idx, uint32_t p) {
    const int half = len >> 1;
    const int block = idx / half;
    const int j = idx - block * half;
    const int pos = block * len + j;
    const uint32_t u = a[pos];
    const uint32_t v = mul_mod_mont_2prime(a[pos + half], roots[j], p);
    uint32_t x = u + v;
    if (x >= p) x -= p;
    const uint32_t y = (u >= v) ? (u - v) : (u + p - v);
    a[pos] = x;
    a[pos + half] = y;
}

__device__ __forceinline__ void ntt_stage2_apply_mont(uint32_t* a,
                                                       const uint32_t* roots1,
                                                       const uint32_t* roots2,
                                                       int len, int idx, uint32_t p) {
    const int quarter = len >> 2;
    const int block = idx / quarter;
    const int j = idx - block * quarter;
    const int pos = block * len + j;
    const uint32_t w1 = roots1[j];
    const uint32_t a0 = a[pos];
    const uint32_t a1 = mul_mod_mont_2prime(a[pos + quarter], w1, p);
    const uint32_t a2 = a[pos + 2 * quarter];
    const uint32_t a3 = mul_mod_mont_2prime(a[pos + 3 * quarter], w1, p);
    uint32_t b0 = a0 + a1;
    if (b0 >= p) b0 -= p;
    const uint32_t b1 = (a0 >= a1) ? (a0 - a1) : (a0 + p - a1);
    uint32_t b2 = a2 + a3;
    if (b2 >= p) b2 -= p;
    const uint32_t b3 = (a2 >= a3) ? (a2 - a3) : (a2 + p - a3);
    const uint32_t t0 = mul_mod_mont_2prime(b2, roots2[j], p);
    const uint32_t t1 = mul_mod_mont_2prime(b3, roots2[j + quarter], p);
    uint32_t c0 = b0 + t0;
    if (c0 >= p) c0 -= p;
    const uint32_t c2 = (b0 >= t0) ? (b0 - t0) : (b0 + p - t0);
    uint32_t c1 = b1 + t1;
    if (c1 >= p) c1 -= p;
    const uint32_t c3 = (b1 >= t1) ? (b1 - t1) : (b1 + p - t1);
    a[pos] = c0;
    a[pos + quarter] = c1;
    a[pos + 2 * quarter] = c2;
    a[pos + 3 * quarter] = c3;
}

__device__ __forceinline__ void ntt_stage_one_dif_apply_mont(uint32_t* a, const uint32_t* roots,
                                                              int len, int idx, uint32_t p) {
    const int half = len >> 1;
    const int block = idx / half;
    const int j = idx - block * half;
    const int pos = block * len + j;
    const uint32_t u = a[pos];
    const uint32_t v = a[pos + half];
    uint32_t sum = u + v;
    if (sum >= p) sum -= p;
    const uint32_t diff = (u >= v) ? (u - v) : (u + p - v);
    a[pos] = sum;
    a[pos + half] = mul_mod_mont_2prime(diff, roots[j], p);
}

// Two descending DIF stages: len followed by len/2. The output remains in the
// same array and a complete forward transform ends in bit-reversed order.
__device__ __forceinline__ void ntt_stage2_dif_apply_mont(uint32_t* a,
                                                           const uint32_t* roots_big,
                                                           const uint32_t* roots_small,
                                                           int len, int idx, uint32_t p) {
    const int quarter = len >> 2;
    const int block = idx / quarter;
    const int j = idx - block * quarter;
    const int pos = block * len + j;
    const uint32_t x0 = a[pos];
    const uint32_t x1 = a[pos + quarter];
    const uint32_t x2 = a[pos + 2 * quarter];
    const uint32_t x3 = a[pos + 3 * quarter];

    uint32_t a0 = x0 + x2;
    if (a0 >= p) a0 -= p;
    uint32_t a1 = x1 + x3;
    if (a1 >= p) a1 -= p;
    const uint32_t d0 = (x0 >= x2) ? (x0 - x2) : (x0 + p - x2);
    const uint32_t d1 = (x1 >= x3) ? (x1 - x3) : (x1 + p - x3);
    const uint32_t a2 = mul_mod_mont_2prime(d0, roots_big[j], p);
    const uint32_t a3 = mul_mod_mont_2prime(d1, roots_big[j + quarter], p);
    const uint32_t ws = roots_small[j];

    uint32_t y0 = a0 + a1;
    if (y0 >= p) y0 -= p;
    const uint32_t d2 = (a0 >= a1) ? (a0 - a1) : (a0 + p - a1);
    const uint32_t y1 = mul_mod_mont_2prime(d2, ws, p);
    uint32_t y2 = a2 + a3;
    if (y2 >= p) y2 -= p;
    const uint32_t d3 = (a2 >= a3) ? (a2 - a3) : (a2 + p - a3);
    const uint32_t y3 = mul_mod_mont_2prime(d3, ws, p);
    a[pos] = y0;
    a[pos + quarter] = y1;
    a[pos + 2 * quarter] = y2;
    a[pos + 3 * quarter] = y3;
}

__device__ __forceinline__ uint32_t add_mod_2prime(uint32_t a, uint32_t b, uint32_t p) {
    uint32_t x = a + b;
    if (x >= p) x -= p;
    return x;
}

__device__ __forceinline__ uint32_t sub_mod_2prime(uint32_t a, uint32_t b, uint32_t p) {
    return (a >= b) ? (a - b) : (a + p - b);
}

__device__ __forceinline__ void ntt_stage3_dit_apply_mont(uint32_t* a,
                                                           const uint32_t* roots_small,
                                                           const uint32_t* roots_mid,
                                                           const uint32_t* roots_big,
                                                           int len, int idx, uint32_t p) {
    const int octant = len >> 3;
    const int block = idx / octant;
    const int j = idx - block * octant;
    const int pos = block * len + j;
    uint32_t x[8], y[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) x[t] = a[pos + t * octant];

    const uint32_t ws = roots_small[j];
#pragma unroll
    for (int g = 0; g < 8; g += 2) {
        const uint32_t v = mul_mod_mont_2prime(x[g + 1], ws, p);
        y[g] = add_mod_2prime(x[g], v, p);
        y[g + 1] = sub_mod_2prime(x[g], v, p);
    }
#pragma unroll
    for (int g = 0; g < 8; g += 4) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t v = mul_mod_mont_2prime(y[g + 2 + r], roots_mid[j + r * octant], p);
            x[g + r] = add_mod_2prime(y[g + r], v, p);
            x[g + 2 + r] = sub_mod_2prime(y[g + r], v, p);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const uint32_t v = mul_mod_mont_2prime(x[4 + r], roots_big[j + r * octant], p);
        y[r] = add_mod_2prime(x[r], v, p);
        y[4 + r] = sub_mod_2prime(x[r], v, p);
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) a[pos + t * octant] = y[t];
}

__device__ __forceinline__ void ntt_stage3_dif_apply_mont(uint32_t* a,
                                                           const uint32_t* roots_big,
                                                           const uint32_t* roots_mid,
                                                           const uint32_t* roots_small,
                                                           int len, int idx, uint32_t p) {
    const int octant = len >> 3;
    const int block = idx / octant;
    const int j = idx - block * octant;
    const int pos = block * len + j;
    uint32_t x[8], y[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) x[t] = a[pos + t * octant];

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        y[r] = add_mod_2prime(x[r], x[4 + r], p);
        const uint32_t d = sub_mod_2prime(x[r], x[4 + r], p);
        y[4 + r] = mul_mod_mont_2prime(d, roots_big[j + r * octant], p);
    }
#pragma unroll
    for (int g = 0; g < 8; g += 4) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            x[g + r] = add_mod_2prime(y[g + r], y[g + 2 + r], p);
            const uint32_t d = sub_mod_2prime(y[g + r], y[g + 2 + r], p);
            x[g + 2 + r] = mul_mod_mont_2prime(d, roots_mid[j + r * octant], p);
        }
    }
    const uint32_t ws = roots_small[j];
#pragma unroll
    for (int g = 0; g < 8; g += 2) {
        y[g] = add_mod_2prime(x[g], x[g + 1], p);
        const uint32_t d = sub_mod_2prime(x[g], x[g + 1], p);
        y[g + 1] = mul_mod_mont_2prime(d, ws, p);
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) a[pos + t * octant] = y[t];
}

__device__ __forceinline__ void ntt_stage4_dit_apply_mont(
    uint32_t* a, const uint32_t* roots0, const uint32_t* roots1,
    const uint32_t* roots2, const uint32_t* roots3,
    int len, int idx, uint32_t p) {
    const int unit = len >> 4;
    const int block = idx / unit;
    const int j = idx - block * unit;
    const int pos = block * len + j;
    uint32_t x[16], y[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) x[t] = a[pos + t * unit];

#pragma unroll
    for (int g = 0; g < 16; g += 2) {
        const uint32_t v = mul_mod_mont_2prime(x[g + 1], roots0[j], p);
        y[g] = add_mod_2prime(x[g], v, p);
        y[g + 1] = sub_mod_2prime(x[g], v, p);
    }
#pragma unroll
    for (int g = 0; g < 16; g += 4) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t v = mul_mod_mont_2prime(y[g + 2 + r], roots1[j + r * unit], p);
            x[g + r] = add_mod_2prime(y[g + r], v, p);
            x[g + 2 + r] = sub_mod_2prime(y[g + r], v, p);
        }
    }
#pragma unroll
    for (int g = 0; g < 16; g += 8) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const uint32_t v = mul_mod_mont_2prime(x[g + 4 + r], roots2[j + r * unit], p);
            y[g + r] = add_mod_2prime(x[g + r], v, p);
            y[g + 4 + r] = sub_mod_2prime(x[g + r], v, p);
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const uint32_t v = mul_mod_mont_2prime(y[8 + r], roots3[j + r * unit], p);
        x[r] = add_mod_2prime(y[r], v, p);
        x[8 + r] = sub_mod_2prime(y[r], v, p);
    }
#pragma unroll
    for (int t = 0; t < 16; ++t) a[pos + t * unit] = x[t];
}

__device__ __forceinline__ void ntt_stage4_dif_apply_mont(
    uint32_t* a, const uint32_t* roots3, const uint32_t* roots2,
    const uint32_t* roots1, const uint32_t* roots0,
    int len, int idx, uint32_t p) {
    const int unit = len >> 4;
    const int block = idx / unit;
    const int j = idx - block * unit;
    const int pos = block * len + j;
    uint32_t x[16], y[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) x[t] = a[pos + t * unit];

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        y[r] = add_mod_2prime(x[r], x[8 + r], p);
        y[8 + r] = mul_mod_mont_2prime(sub_mod_2prime(x[r], x[8 + r], p),
                                       roots3[j + r * unit], p);
    }
#pragma unroll
    for (int g = 0; g < 16; g += 8) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            x[g + r] = add_mod_2prime(y[g + r], y[g + 4 + r], p);
            x[g + 4 + r] = mul_mod_mont_2prime(
                sub_mod_2prime(y[g + r], y[g + 4 + r], p), roots2[j + r * unit], p);
        }
    }
#pragma unroll
    for (int g = 0; g < 16; g += 4) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            y[g + r] = add_mod_2prime(x[g + r], x[g + 2 + r], p);
            y[g + 2 + r] = mul_mod_mont_2prime(
                sub_mod_2prime(x[g + r], x[g + 2 + r], p), roots1[j + r * unit], p);
        }
    }
#pragma unroll
    for (int g = 0; g < 16; g += 2) {
        x[g] = add_mod_2prime(y[g], y[g + 1], p);
        x[g + 1] = mul_mod_mont_2prime(sub_mod_2prime(y[g], y[g + 1], p), roots0[j], p);
    }
#pragma unroll
    for (int t = 0; t < 16; ++t) a[pos + t * unit] = x[t];
}

__global__ void ntt_stage2_2_mont_kernel(uint32_t* r0, uint32_t* r1,
                                         const uint32_t* roots10, const uint32_t* roots11,
                                         const uint32_t* roots20, const uint32_t* roots21,
                                         int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage2_apply_mont(r0, roots10, roots20, len, idx, p0);
        ntt_stage2_apply_mont(r1, roots11, roots21, len, idx, p1);
    }
}

__global__ void ntt_stage2_2_mont_scaled_kernel(uint32_t* r0, uint32_t* r1,
                                                const uint32_t* roots_small0, const uint32_t* roots_small1,
                                                const uint32_t* roots_big0, const uint32_t* roots_big1,
                                                int len, int groups, uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage2_apply_mont(r0, roots_small0, roots_big0, len, idx, p0);
        ntt_stage2_apply_mont(r1, roots_small1, roots_big1, len, idx, p1);
        const int quarter = len >> 2;
        const int block = idx / quarter;
        const int j = idx - block * quarter;
        const int pos = block * len + j;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int out = pos + t * quarter;
            r0[out] = mul_mod_mont_2prime(r0[out], inv0, p0);
            r1[out] = mul_mod_mont_2prime(r1[out], inv1, p1);
        }
    }
}

__global__ void ntt_stage_2_mont_kernel(uint32_t* r0, uint32_t* r1,
                                        const uint32_t* roots0, const uint32_t* roots1,
                                        int len, int butterflies) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        ntt_stage_one_apply_mont(r0, roots0, len, idx, p0);
        ntt_stage_one_apply_mont(r1, roots1, len, idx, p1);
    }
}
__global__ void ntt_stage_2_mont_scaled_kernel(uint32_t* r0, uint32_t* r1,
                                               const uint32_t* roots0, const uint32_t* roots1,
                                               int len, int butterflies,
                                               uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        ntt_stage_one_apply_mont(r0, roots0, len, idx, p0);
        ntt_stage_one_apply_mont(r1, roots1, len, idx, p1);
        const int half = len >> 1;
        const int block = idx / half;
        const int j = idx - block * half;
        const int pos = block * len + j;
        r0[pos] = mul_mod_mont_2prime(r0[pos], inv0, p0);
        r0[pos + half] = mul_mod_mont_2prime(r0[pos + half], inv0, p0);
        r1[pos] = mul_mod_mont_2prime(r1[pos], inv1, p1);
        r1[pos + half] = mul_mod_mont_2prime(r1[pos + half], inv1, p1);
    }
}

__global__ void ntt_stage_2_dif_mont_kernel(uint32_t* r0, uint32_t* r1,
                                            const uint32_t* roots0, const uint32_t* roots1,
                                            int len, int butterflies) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        ntt_stage_one_dif_apply_mont(r0, roots0, len, idx, p0);
        ntt_stage_one_dif_apply_mont(r1, roots1, len, idx, p1);
    }
}

__global__ void ntt_stage2_2_dif_mont_kernel(uint32_t* r0, uint32_t* r1,
                                             const uint32_t* roots_big0, const uint32_t* roots_big1,
                                             const uint32_t* roots_small0, const uint32_t* roots_small1,
                                             int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage2_dif_apply_mont(r0, roots_big0, roots_small0, len, idx, p0);
        ntt_stage2_dif_apply_mont(r1, roots_big1, roots_small1, len, idx, p1);
    }
}

__global__ void ntt_stage3_2_dit_mont_kernel(uint32_t* r0, uint32_t* r1,
                                             const uint32_t* roots_small0, const uint32_t* roots_small1,
                                             const uint32_t* roots_mid0, const uint32_t* roots_mid1,
                                             const uint32_t* roots_big0, const uint32_t* roots_big1,
                                             int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage3_dit_apply_mont(r0, roots_small0, roots_mid0, roots_big0, len, idx, p0);
        ntt_stage3_dit_apply_mont(r1, roots_small1, roots_mid1, roots_big1, len, idx, p1);
    }
}

__global__ void ntt_stage3_2_dit_mont_scaled_kernel(uint32_t* r0, uint32_t* r1,
                                                    const uint32_t* roots_small0, const uint32_t* roots_small1,
                                                    const uint32_t* roots_mid0, const uint32_t* roots_mid1,
                                                    const uint32_t* roots_big0, const uint32_t* roots_big1,
                                                    int len, int groups,
                                                    uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage3_dit_apply_mont(r0, roots_small0, roots_mid0, roots_big0, len, idx, p0);
        ntt_stage3_dit_apply_mont(r1, roots_small1, roots_mid1, roots_big1, len, idx, p1);
        const int octant = len >> 3;
        const int block = idx / octant;
        const int j = idx - block * octant;
        const int pos = block * len + j;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int out = pos + t * octant;
            r0[out] = mul_mod_mont_2prime(r0[out], inv0, p0);
            r1[out] = mul_mod_mont_2prime(r1[out], inv1, p1);
        }
    }
}

__global__ void ntt_stage3_2_dif_mont_kernel(uint32_t* r0, uint32_t* r1,
                                             const uint32_t* roots_big0, const uint32_t* roots_big1,
                                             const uint32_t* roots_mid0, const uint32_t* roots_mid1,
                                             const uint32_t* roots_small0, const uint32_t* roots_small1,
                                             int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage3_dif_apply_mont(r0, roots_big0, roots_mid0, roots_small0, len, idx, p0);
        ntt_stage3_dif_apply_mont(r1, roots_big1, roots_mid1, roots_small1, len, idx, p1);
    }
}

__global__ void ntt_stage4_2_dit_mont_kernel(
    uint32_t* r0, uint32_t* r1,
    const uint32_t* roots00, const uint32_t* roots01,
    const uint32_t* roots10, const uint32_t* roots11,
    const uint32_t* roots20, const uint32_t* roots21,
    const uint32_t* roots30, const uint32_t* roots31,
    int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage4_dit_apply_mont(r0, roots00, roots10, roots20, roots30, len, idx, p0);
        ntt_stage4_dit_apply_mont(r1, roots01, roots11, roots21, roots31, len, idx, p1);
    }
}

__global__ void ntt_stage4_2_dit_mont_scaled_kernel(
    uint32_t* r0, uint32_t* r1,
    const uint32_t* roots00, const uint32_t* roots01,
    const uint32_t* roots10, const uint32_t* roots11,
    const uint32_t* roots20, const uint32_t* roots21,
    const uint32_t* roots30, const uint32_t* roots31,
    int len, int groups, uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage4_dit_apply_mont(r0, roots00, roots10, roots20, roots30, len, idx, p0);
        ntt_stage4_dit_apply_mont(r1, roots01, roots11, roots21, roots31, len, idx, p1);
        const int unit = len >> 4;
        const int block = idx / unit;
        const int j = idx - block * unit;
        const int pos = block * len + j;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int out = pos + t * unit;
            r0[out] = mul_mod_mont_2prime(r0[out], inv0, p0);
            r1[out] = mul_mod_mont_2prime(r1[out], inv1, p1);
        }
    }
}

__global__ void ntt_stage4_2_dif_mont_kernel(
    uint32_t* r0, uint32_t* r1,
    const uint32_t* roots30, const uint32_t* roots31,
    const uint32_t* roots20, const uint32_t* roots21,
    const uint32_t* roots10, const uint32_t* roots11,
    const uint32_t* roots00, const uint32_t* roots01,
    int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage4_dif_apply_mont(r0, roots30, roots20, roots10, roots00, len, idx, p0);
        ntt_stage4_dif_apply_mont(r1, roots31, roots21, roots11, roots01, len, idx, p1);
    }
}

#ifndef GFPPS_SHARED_NTT_TILE_LOG
#define GFPPS_SHARED_NTT_TILE_LOG 11
#endif
#ifndef GFPPS_SHARED_RADIX8
#define GFPPS_SHARED_RADIX8 1
#endif
#ifndef GFPPS_SHARED_NTT_THREADS
#define GFPPS_SHARED_NTT_THREADS 256
#endif
#ifndef GFPPS_GLOBAL_RADIX4
#define GFPPS_GLOBAL_RADIX4 1
#endif
#ifndef GFPPS_GLOBAL_RADIX8
#define GFPPS_GLOBAL_RADIX8 1
#endif
#ifndef GFPPS_GLOBAL_RADIX16
#define GFPPS_GLOBAL_RADIX16 1
#endif
constexpr int kSharedNttTileLog = GFPPS_SHARED_NTT_TILE_LOG;
constexpr int kSharedNttTile = 1 << kSharedNttTileLog;
constexpr int kSharedNttThreads = GFPPS_SHARED_NTT_THREADS;

template<bool Square>
__global__ void ntt_tile_product_dit_mont_kernel(uint32_t* r0, uint32_t* r1,
                                                 const uint32_t* multiplier0,
                                                 const uint32_t* multiplier1,
                                                 const uint32_t* tw0, const uint32_t* tw1,
                                                 int total_tiles) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    __shared__ uint32_t s0[kSharedNttTile];
    __shared__ uint32_t s1[kSharedNttTile];
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int base = tile * kSharedNttTile;
        for (int i = threadIdx.x; i < kSharedNttTile; i += blockDim.x) {
            const uint32_t x0 = r0[base + i];
            const uint32_t x1 = r1[base + i];
            const uint32_t y0 = Square ? x0 : multiplier0[base + i];
            const uint32_t y1 = Square ? x1 : multiplier1[base + i];
            s0[i] = mul_mod_mont_2prime(x0, y0, p0);
            s1[i] = mul_mod_mont_2prime(x1, y1, p1);
        }
        __syncthreads();
        int stage = 1;
#if GFPPS_SHARED_RADIX8
        for (; stage + 2 <= kSharedNttTileLog; stage += 3) {
            const int len = 1 << (stage + 2);
            const int root1_offset = (1 << (stage - 1)) - 1;
            const int root2_offset = (1 << stage) - 1;
            const int root3_offset = (1 << (stage + 1)) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 8; idx += blockDim.x) {
                ntt_stage3_dit_apply_mont(s0, tw0 + root1_offset, tw0 + root2_offset,
                                           tw0 + root3_offset, len, idx, p0);
                ntt_stage3_dit_apply_mont(s1, tw1 + root1_offset, tw1 + root2_offset,
                                           tw1 + root3_offset, len, idx, p1);
            }
            __syncthreads();
        }
#endif
        for (; stage + 1 <= kSharedNttTileLog; stage += 2) {
            const int len = 1 << (stage + 1);
            const int root1_offset = (1 << (stage - 1)) - 1;
            const int root2_offset = (1 << stage) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 4; idx += blockDim.x) {
                ntt_stage2_apply_mont(s0, tw0 + root1_offset, tw0 + root2_offset, len, idx, p0);
                ntt_stage2_apply_mont(s1, tw1 + root1_offset, tw1 + root2_offset, len, idx, p1);
            }
            __syncthreads();
        }
        if (stage <= kSharedNttTileLog) {
            const int len = 1 << stage;
            const int root_offset = (1 << (stage - 1)) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 2; idx += blockDim.x) {
                ntt_stage_one_apply_mont(s0, tw0 + root_offset, len, idx, p0);
                ntt_stage_one_apply_mont(s1, tw1 + root_offset, len, idx, p1);
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < kSharedNttTile; i += blockDim.x) {
            r0[base + i] = s0[i];
            r1[base + i] = s1[i];
        }
        __syncthreads();
    }
}

__global__ void ntt_tile_dif_mont_kernel(uint32_t* r0, uint32_t* r1,
                                         const uint32_t* tw0, const uint32_t* tw1,
                                         int total_tiles) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    __shared__ uint32_t s0[kSharedNttTile];
    __shared__ uint32_t s1[kSharedNttTile];
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int base = tile * kSharedNttTile;
        for (int i = threadIdx.x; i < kSharedNttTile; i += blockDim.x) {
            s0[i] = r0[base + i];
            s1[i] = r1[base + i];
        }
        __syncthreads();
        int stage = kSharedNttTileLog;
#if GFPPS_SHARED_RADIX8
        for (; stage - 2 >= 1; stage -= 3) {
            const int len = 1 << stage;
            const int root_big_offset = (1 << (stage - 1)) - 1;
            const int root_mid_offset = (1 << (stage - 2)) - 1;
            const int root_small_offset = (1 << (stage - 3)) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 8; idx += blockDim.x) {
                ntt_stage3_dif_apply_mont(s0, tw0 + root_big_offset, tw0 + root_mid_offset,
                                           tw0 + root_small_offset, len, idx, p0);
                ntt_stage3_dif_apply_mont(s1, tw1 + root_big_offset, tw1 + root_mid_offset,
                                           tw1 + root_small_offset, len, idx, p1);
            }
            __syncthreads();
        }
#endif
        for (; stage - 1 >= 1; stage -= 2) {
            const int len = 1 << stage;
            const int root_big_offset = (1 << (stage - 1)) - 1;
            const int root_small_offset = (1 << (stage - 2)) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 4; idx += blockDim.x) {
                ntt_stage2_dif_apply_mont(s0, tw0 + root_big_offset, tw0 + root_small_offset, len, idx, p0);
                ntt_stage2_dif_apply_mont(s1, tw1 + root_big_offset, tw1 + root_small_offset, len, idx, p1);
            }
            __syncthreads();
        }
        if (stage == 1) {
            for (int idx = threadIdx.x; idx < kSharedNttTile / 2; idx += blockDim.x) {
                ntt_stage_one_dif_apply_mont(s0, tw0, 2, idx, p0);
                ntt_stage_one_dif_apply_mont(s1, tw1, 2, idx, p1);
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < kSharedNttTile; i += blockDim.x) {
            r0[base + i] = s0[i];
            r1[base + i] = s1[i];
        }
        __syncthreads();
    }
}

__global__ void pointwise_square2_mont_kernel(uint32_t* r0, uint32_t* r1, int len) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = mul_mod_mont_2prime(r0[i], r0[i], p0);
    r1[i] = mul_mod_mont_2prime(r1[i], r1[i], p1);
}

__global__ void pointwise_mul2_mont_kernel(uint32_t* r0, uint32_t* r1,
                                           const uint32_t* multiplier0,
                                           const uint32_t* multiplier1, int len) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = mul_mod_mont_2prime(r0[i], multiplier0[i], p0);
    r1[i] = mul_mod_mont_2prime(r1[i], multiplier1[i], p1);
}

// The input is Montgomery encoded. Multiplication by an ordinary inv_len both
// applies the inverse scale and leaves the result in the ordinary domain.
__global__ void scale2_from_mont_kernel(uint32_t* r0, uint32_t* r1, int len,
                                        uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = mul_mod_mont_2prime(r0[i], inv0, p0);
    r1[i] = mul_mod_mont_2prime(r1[i], inv1, p1);
}

__device__ __forceinline__ uint64_t crt2_reconstruct_shoup(uint32_t a, uint32_t b) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t inv_p0_mod_p1 = 669690699u;
    constexpr uint32_t inv_p0_mod_p1_shoup = 2863312213u;
    const uint32_t diff = (b >= a) ? (b - a) : (b + p1 - a);
    const uint32_t q = __umulhi(diff, inv_p0_mod_p1_shoup);
    uint32_t t = diff * inv_p0_mod_p1 - q * p1;
    if (t >= p1) t -= p1;
    return static_cast<uint64_t>(a) + static_cast<uint64_t>(p0) * t;
}

std::vector<int> make_twiddle_offsets(int log_len) {
    std::vector<int> offsets(log_len + 1, 0);
    int total = 0;
    for (int stage = 1; stage <= log_len; ++stage) {
        offsets[stage] = total;
        total += 1 << (stage - 1);
    }
    return offsets;
}

std::vector<uint32_t> make_twiddle_table_host(int log_len, uint32_t p, uint32_t g, bool inverse) {
    const auto offsets = make_twiddle_offsets(log_len);
    const int total_roots = (1 << log_len) - 1;
    std::vector<uint32_t> table(total_roots);
    for (int stage = 1; stage <= log_len; ++stage) {
        const int len = 1 << stage;
        const int half = len >> 1;
        uint32_t wlen = pow_mod_host(g, (p - 1ull) / static_cast<uint64_t>(len), p);
        if (inverse) wlen = pow_mod_host(wlen, p - 2ull, p);
        uint64_t w = 1;
        const int off = offsets[stage];
        for (int j = 0; j < half; ++j) {
            table[off + j] = static_cast<uint32_t>(w);
            w = (w * wlen) % p;
        }
    }
    return table;
}

void ntt2_forward_dif_mont(uint32_t* r0, uint32_t* r1, int log_len,
                           const uint32_t* tw0, const uint32_t* tw1,
                           const std::vector<int>& offsets,
                           cudaStream_t stream) {
    const int len_total = 1 << log_len;
    const int threads = 256;
    int stage = log_len;
    const int shared_tail = (log_len >= kSharedNttTileLog) ? kSharedNttTileLog : 0;
#if GFPPS_GLOBAL_RADIX16
    // Exact multiples of three are handled more efficiently by pure radix-8:
    // it uses the same number of launches without radix-16 register pressure.
    const bool prefer_global_radix16 = ((log_len - shared_tail) % 3) != 0;
    if (prefer_global_radix16) {
        for (; stage - 3 > shared_tail; stage -= 4) {
            const int len = 1 << stage;
            const int groups = len_total / 16;
            const int y = limited_ntt_grid_y(groups, threads);
            ntt_stage4_2_dif_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(
                r0, r1,
                tw0 + offsets[stage], tw1 + offsets[stage],
                tw0 + offsets[stage - 1], tw1 + offsets[stage - 1],
                tw0 + offsets[stage - 2], tw1 + offsets[stage - 2],
                tw0 + offsets[stage - 3], tw1 + offsets[stage - 3],
                len, groups);
            cuda_check(cudaGetLastError(), "ntt_stage4_2_dif_mont launch");
        }
    }
#endif
#if GFPPS_GLOBAL_RADIX8
    for (; stage - 2 > shared_tail; stage -= 3) {
        const int len = 1 << stage;
        const int groups = len_total / 8;
        const int y = limited_ntt_grid_y(groups, threads);
        ntt_stage3_2_dif_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                              tw0 + offsets[stage],
                                                              tw1 + offsets[stage],
                                                              tw0 + offsets[stage - 1],
                                                              tw1 + offsets[stage - 1],
                                                              tw0 + offsets[stage - 2],
                                                              tw1 + offsets[stage - 2],
                                                              len, groups);
        cuda_check(cudaGetLastError(), "ntt_stage3_2_dif_mont launch");
    }
#endif
#if GFPPS_GLOBAL_RADIX4
    for (; stage - 1 > shared_tail; stage -= 2) {
        const int len = 1 << stage;
        const int groups = len_total / 4;
        const int y = limited_ntt_grid_y(groups, threads);
        ntt_stage2_2_dif_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                              tw0 + offsets[stage],
                                                              tw1 + offsets[stage],
                                                              tw0 + offsets[stage - 1],
                                                              tw1 + offsets[stage - 1],
                                                              len, groups);
        cuda_check(cudaGetLastError(), "ntt_stage2_2_dif_mont launch");
    }
#endif
    while (stage > shared_tail) {
        const int len = 1 << stage;
        const int butterflies = len_total / 2;
        const int y = limited_ntt_grid_y(butterflies, threads);
        ntt_stage_2_dif_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                             tw0 + offsets[stage],
                                                             tw1 + offsets[stage],
                                                             len, butterflies);
        cuda_check(cudaGetLastError(), "ntt_stage_2_dif_mont launch");
        --stage;
    }
    if (shared_tail != 0) {
        const int total_tiles = len_total / kSharedNttTile;
        const int blocks = std::max(1, std::min(ntt_block_limit(), total_tiles));
        ntt_tile_dif_mont_kernel<<<blocks, kSharedNttThreads, 0, stream>>>(r0, r1, tw0, tw1, total_tiles);
        cuda_check(cudaGetLastError(), "ntt_tile_dif_mont launch");
    }
}

void ntt2_inverse_product_dit_mont(uint32_t* r0, uint32_t* r1, int log_len,
                                   const uint32_t* tw0, const uint32_t* tw1,
                                   const std::vector<int>& offsets,
                                   uint32_t inv0, uint32_t inv1,
                                   const uint32_t* multiplier0 = nullptr,
                                   const uint32_t* multiplier1 = nullptr,
                                   cudaStream_t stream = cudaStreamPerThread) {
    const int len_total = 1 << log_len;
    const int threads = 256;
    const int blocks = (len_total + threads - 1) / threads;
    bool scale_fused = false;
    int stage = 1;
    if (log_len >= kSharedNttTileLog) {
        const int total_tiles = len_total / kSharedNttTile;
        const int tile_blocks = std::max(1, std::min(ntt_block_limit(), total_tiles));
        if (multiplier0 == nullptr) {
            ntt_tile_product_dit_mont_kernel<true><<<tile_blocks, kSharedNttThreads, 0, stream>>>(r0, r1, nullptr, nullptr,
                                                                                      tw0, tw1, total_tiles);
        } else {
            ntt_tile_product_dit_mont_kernel<false><<<tile_blocks, kSharedNttThreads, 0, stream>>>(r0, r1, multiplier0, multiplier1,
                                                                                       tw0, tw1, total_tiles);
        }
        cuda_check(cudaGetLastError(), "ntt_tile_product_dit_mont launch");
        stage = kSharedNttTileLog + 1;
    } else {
        if (multiplier0 == nullptr) {
            pointwise_square2_mont_kernel<<<blocks, threads, 0, stream>>>(r0, r1, len_total);
        } else {
            pointwise_mul2_mont_kernel<<<blocks, threads, 0, stream>>>(r0, r1, multiplier0, multiplier1, len_total);
        }
        cuda_check(cudaGetLastError(), "pointwise product fallback launch");
    }
#if GFPPS_GLOBAL_RADIX16
    const int shared_prefix = (log_len >= kSharedNttTileLog) ? kSharedNttTileLog : 0;
    const bool prefer_global_radix16 = ((log_len - shared_prefix) % 3) != 0;
    if (prefer_global_radix16) {
        for (; stage + 3 <= log_len; stage += 4) {
            const int len = 1 << (stage + 3);
            const int groups = len_total / 16;
            const int y = limited_ntt_grid_y(groups, threads);
            if (stage + 3 == log_len) {
                ntt_stage4_2_dit_mont_scaled_kernel<<<dim3(1, y), threads, 0, stream>>>(
                    r0, r1,
                    tw0 + offsets[stage], tw1 + offsets[stage],
                    tw0 + offsets[stage + 1], tw1 + offsets[stage + 1],
                    tw0 + offsets[stage + 2], tw1 + offsets[stage + 2],
                    tw0 + offsets[stage + 3], tw1 + offsets[stage + 3],
                    len, groups, inv0, inv1);
                scale_fused = true;
            } else {
                ntt_stage4_2_dit_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(
                    r0, r1,
                    tw0 + offsets[stage], tw1 + offsets[stage],
                    tw0 + offsets[stage + 1], tw1 + offsets[stage + 1],
                    tw0 + offsets[stage + 2], tw1 + offsets[stage + 2],
                    tw0 + offsets[stage + 3], tw1 + offsets[stage + 3],
                    len, groups);
            }
            cuda_check(cudaGetLastError(), "ntt_stage4_2_inverse_dit_mont launch");
        }
    }
#endif
#if GFPPS_GLOBAL_RADIX8
    for (; stage + 2 <= log_len; stage += 3) {
        const int len = 1 << (stage + 2);
        const int groups = len_total / 8;
        const int y = limited_ntt_grid_y(groups, threads);
        if (stage + 2 == log_len) {
            ntt_stage3_2_dit_mont_scaled_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                                         tw0 + offsets[stage],
                                                                         tw1 + offsets[stage],
                                                                         tw0 + offsets[stage + 1],
                                                                         tw1 + offsets[stage + 1],
                                                                         tw0 + offsets[stage + 2],
                                                                         tw1 + offsets[stage + 2],
                                                                         len, groups, inv0, inv1);
            scale_fused = true;
        } else {
            ntt_stage3_2_dit_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                                  tw0 + offsets[stage],
                                                                  tw1 + offsets[stage],
                                                                  tw0 + offsets[stage + 1],
                                                                  tw1 + offsets[stage + 1],
                                                                  tw0 + offsets[stage + 2],
                                                                  tw1 + offsets[stage + 2],
                                                                  len, groups);
        }
        cuda_check(cudaGetLastError(), "ntt_stage3_2_inverse_dit_mont launch");
    }
#endif
#if GFPPS_GLOBAL_RADIX4
    for (; stage + 1 <= log_len; stage += 2) {
        const int len = 1 << (stage + 1);
        const int groups = len_total / 4;
        const int y = limited_ntt_grid_y(groups, threads);
        if (stage + 1 == log_len) {
            ntt_stage2_2_mont_scaled_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                                     tw0 + offsets[stage],
                                                                     tw1 + offsets[stage],
                                                                     tw0 + offsets[stage + 1],
                                                                     tw1 + offsets[stage + 1],
                                                                     len, groups, inv0, inv1);
            scale_fused = true;
        } else {
            ntt_stage2_2_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                              tw0 + offsets[stage],
                                                              tw1 + offsets[stage],
                                                              tw0 + offsets[stage + 1],
                                                              tw1 + offsets[stage + 1],
                                                              len, groups);
        }
        cuda_check(cudaGetLastError(), "ntt_stage2_2_inverse_dit_mont launch");
    }
#endif
    while (stage <= log_len) {
        const int len = 1 << stage;
        const int butterflies = len_total / 2;
        const int y = limited_ntt_grid_y(butterflies, threads);
        if (stage == log_len) {
            ntt_stage_2_mont_scaled_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                                    tw0 + offsets[stage],
                                                                    tw1 + offsets[stage],
                                                                    len, butterflies, inv0, inv1);
            scale_fused = true;
        } else {
            ntt_stage_2_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                             tw0 + offsets[stage],
                                                             tw1 + offsets[stage],
                                                             len, butterflies);
        }
        cuda_check(cudaGetLastError(), "ntt_stage_2_inverse_dit_mont launch");
        ++stage;
    }
    if (!scale_fused) {
        scale2_from_mont_kernel<<<blocks, threads, 0, stream>>>(r0, r1, len_total, inv0, inv1);
        cuda_check(cudaGetLastError(), "scale2_from_inverse_dit_mont launch");
    }
}


} // namespace gfpps_ntt
