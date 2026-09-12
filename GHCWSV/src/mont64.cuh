// SPDX-License-Identifier: GPL-2.0-or-later
// Derived from GNCWSV; A.P. 2026.
struct Mont64 {
    uint64_t mod;
    uint64_t ninv;
    uint64_t r2;
    uint64_t rmod;
};

__device__ __forceinline__ uint64_t d_add_mod(uint64_t a, uint64_t b, uint64_t m) {
    return (a >= m - b) ? (a - (m - b)) : (a + b);
}

__device__ __forceinline__ uint64_t d_mont_ninv(uint64_t n) {
    uint64_t x = 1;
#pragma unroll
    for (int i = 0; i < 6; ++i) x *= 2 - n * x;
    return 0 - x;
}

__device__ __forceinline__ uint64_t d_mont_mul(uint64_t a, uint64_t b, const Mont64& c) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    uint64_t m = lo * c.ninv;
    uint64_t mlo = m * c.mod;
    uint64_t mhi = __umul64hi(m, c.mod);
    uint64_t sumlo = lo + mlo;
    uint64_t carry = (sumlo < lo);
    uint64_t u = hi + mhi + carry;
    if (u >= c.mod) u -= c.mod;
    return u;
}

// Sparse-path Montgomery multiplier specialized for moduli below 2^62.
// It keeps the same R=2^64 representation as d_mont_mul(), but performs
// the product and REDC as two 32-bit limbs.  On NVIDIA GPUs this maps the
// multiplications to mul.wide.u32 instead of the much lower-throughput
// general 64x64 high-half path.
__device__ __forceinline__ uint64_t d_mont_mul_2x32(uint64_t a, uint64_t b, const Mont64& c) {
    const uint32_t a0 = static_cast<uint32_t>(a);
    const uint32_t a1 = static_cast<uint32_t>(a >> 32);
    const uint32_t b0 = static_cast<uint32_t>(b);
    const uint32_t b1 = static_cast<uint32_t>(b >> 32);
    const uint32_t n0 = static_cast<uint32_t>(c.mod);
    const uint32_t n1 = static_cast<uint32_t>(c.mod >> 32);
    const uint32_t ni = static_cast<uint32_t>(c.ninv);

    const uint64_t p00 = static_cast<uint64_t>(a0) * b0;
    const uint64_t p01 = static_cast<uint64_t>(a0) * b1;
    const uint64_t p10 = static_cast<uint64_t>(a1) * b0;
    const uint64_t p11 = static_cast<uint64_t>(a1) * b1;

    uint32_t t0 = static_cast<uint32_t>(p00);
    const uint64_t s1 = (p00 >> 32) + static_cast<uint32_t>(p01) + static_cast<uint32_t>(p10);
    uint32_t t1 = static_cast<uint32_t>(s1);
    const uint64_t s2 = (p01 >> 32) + (p10 >> 32) + static_cast<uint32_t>(p11) + (s1 >> 32);
    uint32_t t2 = static_cast<uint32_t>(s2);
    const uint64_t s3 = (p11 >> 32) + (s2 >> 32);
    uint32_t t3 = static_cast<uint32_t>(s3);
    uint32_t t4 = static_cast<uint32_t>(s3 >> 32);

    uint32_t m = static_cast<uint32_t>(static_cast<uint64_t>(t0) * ni);
    uint64_t u = static_cast<uint64_t>(t0) + static_cast<uint64_t>(m) * n0;
    uint64_t carry = u >> 32;
    u = static_cast<uint64_t>(t1) + static_cast<uint64_t>(m) * n1 + carry;
    t1 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t2) + carry; t2 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t3) + carry; t3 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t4) + carry; t4 = static_cast<uint32_t>(u);

    m = static_cast<uint32_t>(static_cast<uint64_t>(t1) * ni);
    u = static_cast<uint64_t>(t1) + static_cast<uint64_t>(m) * n0;
    carry = u >> 32;
    u = static_cast<uint64_t>(t2) + static_cast<uint64_t>(m) * n1 + carry;
    t2 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t3) + carry; t3 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t4) + carry; t4 = static_cast<uint32_t>(u);

    // For this program c.mod < 2^62 and a,b < c.mod, so REDC(a*b) < 2*c.mod
    // and therefore the quotient after the two 32-bit reductions fits in 64 bits.
    uint64_t r = static_cast<uint64_t>(t2) | (static_cast<uint64_t>(t3) << 32);
    if (r >= c.mod) r -= c.mod;
    return r;
}

// Sparse fast path for p < 2^44.  In this range every Montgomery residue is
// also below 2^44, so a*b < 2^88.  The generic 2x32 routine above carries
// five 32-bit limbs to remain valid up to p < 2^62; here the product has only
// three live limbs.  Two radix-2^32 REDC steps can therefore be collapsed to
// short 64-bit accumulators without t3/t4 bookkeeping.  All multiplies are
// 32x32 (or narrower) promoted to uint64_t, so nvcc maps them to the native
// wide-u32 integer path.
__device__ __forceinline__ uint64_t d_mont_mul_fast44(uint64_t a, uint64_t b, const Mont64& c) {
    const uint32_t a0 = static_cast<uint32_t>(a);
    const uint32_t a1 = static_cast<uint32_t>(a >> 32);  // <= 12 bits
    const uint32_t b0 = static_cast<uint32_t>(b);
    const uint32_t b1 = static_cast<uint32_t>(b >> 32);  // <= 12 bits
    const uint32_t n0 = static_cast<uint32_t>(c.mod);
    const uint32_t n1 = static_cast<uint32_t>(c.mod >> 32); // <= 12 bits
    const uint32_t ni = static_cast<uint32_t>(c.ninv);

    // T = a*b = t0 + t1*B + t2*B^2, B=2^32.  Since a,b<2^44, t3=0.
    const uint64_t p00 = static_cast<uint64_t>(a0) * b0;
    const uint64_t cross = (p00 >> 32)
                         + static_cast<uint64_t>(a0) * b1
                         + static_cast<uint64_t>(a1) * b0;
    const uint32_t t0 = static_cast<uint32_t>(p00);
    uint32_t t1 = static_cast<uint32_t>(cross);
    uint64_t t2 = (cross >> 32) + static_cast<uint64_t>(a1) * b1;

    // First REDC limb: t0 + m0*n0 == 0 (mod B).
    const uint32_t m0 = static_cast<uint32_t>(static_cast<uint64_t>(t0) * ni);
    const uint64_t u0 = static_cast<uint64_t>(t0) + static_cast<uint64_t>(m0) * n0;
    const uint64_t u1 = static_cast<uint64_t>(t1)
                      + static_cast<uint64_t>(m0) * n1
                      + (u0 >> 32);
    t1 = static_cast<uint32_t>(u1);
    t2 += (u1 >> 32);

    // Second REDC limb.  After this division by B the result fits in <2*p.
    const uint32_t m1 = static_cast<uint32_t>(static_cast<uint64_t>(t1) * ni);
    const uint64_t v0 = static_cast<uint64_t>(t1) + static_cast<uint64_t>(m1) * n0;
    uint64_t r = t2 + static_cast<uint64_t>(m1) * n1 + (v0 >> 32);
    if (r >= c.mod) r -= c.mod;
    return r;
}

template <bool FAST44>
__device__ __forceinline__ uint64_t d_sparse_mont_mul(uint64_t a, uint64_t b, const Mont64& c) {
    if constexpr (FAST44) return d_mont_mul_fast44(a, b, c);
    else return d_mont_mul_2x32(a, b, c);
}

template <bool FAST44>
__device__ __forceinline__ uint64_t d_sparse_mont_pow_rep(uint64_t x, uint64_t e, const Mont64& c) {
    uint64_t r = c.rmod;
    while (e) {
        if (e & 1ULL) r = d_sparse_mont_mul<FAST44>(r, x, c);
        e >>= 1;
        if (e) x = d_sparse_mont_mul<FAST44>(x, x, c);
    }
    return r;
}

__device__ __forceinline__ uint64_t d_mont_pow_rep_2x32(uint64_t x, uint64_t e, const Mont64& c) {
    uint64_t r = c.rmod;
    while (e) {
        if (e & 1ULL) r = d_mont_mul_2x32(r, x, c);
        e >>= 1;
        if (e) x = d_mont_mul_2x32(x, x, c);
    }
    return r;
}

__device__ __forceinline__ Mont64 d_make_mont(uint64_t p) {
    Mont64 c;
    c.mod = p;
    c.ninv = d_mont_ninv(p);
    c.rmod = (uint64_t(0) - p) % p; // 2^64 mod p
    uint64_t x = c.rmod;
#pragma unroll 1
    for (int i = 0; i < 64; ++i) x = d_add_mod(x, x, p);
    c.r2 = x; // 2^128 mod p
    return c;
}

// Input/output are Montgomery representations.
__device__ __forceinline__ uint64_t d_mont_pow_rep(uint64_t x, uint64_t e, const Mont64& c) {
    uint64_t r = c.rmod;
    while (e) {
        if (e & 1ULL) r = d_mont_mul(r, x, c);
        e >>= 1;
        if (e) x = d_mont_mul(x, x, c);
    }
    return r;
}

__device__ __forceinline__ uint32_t d_inverse_u32(uint32_t a, uint32_t m) {
    int64_t t = 0, nt = 1;
    int64_t r = m, nr = a;
    while (nr != 0) {
        int64_t q = r / nr;
        int64_t tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp;
    }
    if (r != 1) return 0;
    if (t < 0) t += m;
    return static_cast<uint32_t>(t);
}

// Exact inverse of a small uint32 base modulo p, avoiding a p-sized exponentiation.
__device__ __forceinline__ uint64_t d_inverse_base(uint32_t base, uint64_t p) {
    uint32_t r = static_cast<uint32_t>(p % base);
    if (r == 0) return 0;
    uint32_t inv_r = d_inverse_u32(r, base);
    if (inv_r == 0) return 0;
    uint64_t s = static_cast<uint64_t>(base - inv_r); // s*p == -1 (mod base)
    uint64_t q = p / base;
    uint64_t tail = (UINT64_C(1) + s * r) / base;
    return s * q + tail; // (1+s*p)/base
}

