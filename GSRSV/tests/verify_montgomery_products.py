"""Single-thread CPU integer audit of the proposed GSRSV raw-chunk REDC path.

No CUDA, compilation, subprocesses, source modifications or result files.
Mirrors CUDA's uint64 low/high product + carry implementation exactly; Python
big integers are the independent modular-product/inverse oracle.

Pre-seed proof (R=2^64, C chunks): pow_encoded(R^2,C)=R^(C+1).
Each Mont(acc,chunk) divides by R, so the final acc is B*R.  A standard
Montgomery Fermat power followed by one decode returns B^-1 for prime p.
Arithmetic/product invariants also work for composite odd p; Fermat inversion
is checked only for independently primality-tested p, matching the sieve API.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import random
import time

R = 1 << 64
MASK = R - 1
CAP = (1 << 62) - 1
MASK32 = (1 << 32) - 1


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    if n <= 2:
        return 2
    n |= 1
    while not is_prime(n):
        n += 2
    return n


def last_prime(n: int) -> int:
    n -= not (n & 1)
    while n >= 3 and not is_prime(n):
        n -= 2
    if n < 3:
        raise ValueError("no odd prime below bound")
    return n


@dataclass
class Mont64:
    mod: int
    ninv: int
    rmod: int
    r2: int
    rinv: int
    calls: int = 0
    noncanonical_b_calls: int = 0
    maximum_unreduced: int = 0


def make_mont(p: int) -> Mont64:
    if p < 3 or p > CAP or p % 2 == 0:
        raise ValueError("Mont64 audit requires odd 3<=p<=2^62-1; p=2 uses legacy path")
    x = 1
    for _ in range(6):
        x = (x * ((2 - p * x) & MASK)) & MASK
    ninv = (-x) & MASK
    assert (p * ninv) & MASK == MASK
    assert ninv == (-pow(p, -1, R)) & MASK
    rmod = ((-p) & MASK) % p
    r2 = rmod
    for _ in range(64):
        # Exact clone of d_add_mod(x,x,p), with canonical x.
        r2 = r2 - (p - r2) if r2 >= p - r2 else r2 + r2
        assert 0 <= r2 < p
    assert rmod == R % p and r2 == (R * R) % p
    return Mont64(p, ninv, rmod, r2, pow(R, -1, p))


def mont_mul(a: int, b: int, c: Mont64) -> int:
    p = c.mod
    assert 0 <= a < p and 0 <= b < R
    product = a * b
    lo, hi = product & MASK, product >> 64
    m = (lo * c.ninv) & MASK
    mprod = m * p
    mlo, mhi = mprod & MASK, mprod >> 64
    sumlo = (lo + mlo) & MASK
    carry = int(sumlo < lo)
    u = hi + mhi + carry
    assert sumlo == 0, ("REDC low-word cancellation", p, a, b)
    assert carry == int(lo != 0), ("low-product-free carry identity", p, a, b)
    assert product < p * R and mprod < p * R
    assert u == (product + mprod) // R
    assert 0 <= u < 2 * p < (1 << 63), ("high-word overflow/range", p, a, b, u)
    result = u - p if u >= p else u
    expected = product * c.rinv % p
    assert result == expected and 0 <= result < p, ("wrong REDC", p, a, b, result, expected)
    c.calls += 1
    c.noncanonical_b_calls += int(b >= p)
    c.maximum_unreduced = max(c.maximum_unreduced, u)
    return result


def pow_encoded(x: int, exponent: int, c: Mont64, seed: int | None = None) -> int:
    assert 0 <= x < c.mod and exponent >= 0
    result = c.rmod if seed is None else seed
    while exponent:
        if exponent & 1:
            result = mont_mul(result, x, c)
        exponent >>= 1
        if exponent:
            x = mont_mul(x, x, c)
    return result


def legacy_product(chunks: list[int], c: Mont64) -> int:
    result = c.rmod
    for chunk in chunks:
        encoded = mont_mul(chunk % c.mod, c.r2, c)
        result = mont_mul(result, encoded, c)
    return mont_mul(result, 1, c)


def preseed_product(chunks: list[int], c: Mont64) -> int:
    result = pow_encoded(c.r2, len(chunks), c)
    assert result == pow(R, len(chunks) + 1, c.mod)
    for chunk in chunks:
        result = mont_mul(result, chunk, c)
    return result  # Canonical residue representing B*R.


def raw_product(chunks: list[int], c: Mont64) -> int:
    result = c.rmod
    for chunk in chunks:
        result = mont_mul(result, chunk, c)
    return result  # Canonical residue representing B*R^(1-C).


def pack_factors(factors) -> list[int]:
    chunks, current = [], 1
    for x in factors:
        assert 1 <= x <= CAP
        if current > CAP // x:
            chunks.append(current)
            current = x
        else:
            current *= x
    if current != 1 or not chunks:
        chunks.append(current)
    assert chunks and all(1 <= x <= CAP for x in chunks)
    return chunks


def primes_up_to(n: int) -> list[int]:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[:2] = b"\0\0"
    for p in range(2, math.isqrt(n) + 1):
        if sieve[p]:
            start = p * p
            sieve[start:n + 1:p] = b"\0" * (((n - start) // p) + 1)
    return [p for p in range(2, n + 1) if sieve[p]]


def verify_products(chunks: list[int], p: int, *, exact_product=None) -> dict:
    assert all(0 <= x <= CAP for x in chunks)
    product = math.prod(chunks)
    if exact_product is not None:
        assert product == exact_product, "packed chunks disagree with independent factorial/primorial"
    normal = product % p
    c = make_mont(p)
    old = legacy_product(chunks, c)
    seeded = preseed_product(chunks, c)
    decoded = mont_mul(seeded, 1, c)
    assert old == normal == decoded
    assert seeded == normal * R % p

    raw = raw_product(chunks, c)
    assert raw == normal * pow(R, 1 - len(chunks), p) % p
    if chunks:
        # Equivalent post-compensation path, retaining the encoded factor R^C.
        compensation = pow_encoded(c.r2, len(chunks) - 1, c)
        assert compensation == pow(R, len(chunks), p)
        assert mont_mul(raw, compensation, c) == normal
    else:
        assert mont_mul(raw, 1, c) == 1 % p

    inverse = None
    if is_prime(p):
        inverse = 0 if normal == 0 else pow(normal, -1, p)
        # Proposed preseed path: all loops directly consume raw chunks.
        actual = 0 if seeded == 0 else mont_mul(pow_encoded(seeded, p - 2, c), 1, c)
        assert actual == inverse
        # Independent equivalent approach: compensate the inverse accumulator.
        if raw == 0:
            other = 0
        else:
            inverse_seed = pow_encoded(1, len(chunks), c)
            assert inverse_seed == pow(R, 1 - len(chunks), p)
            other = mont_mul(pow_encoded(raw, p - 2, c, inverse_seed), 1, c)
        assert other == inverse
        # The proposed dispatch threshold must not create a discontinuity.
        if p <= MASK32 and len(chunks) < 32:
            small = 1
            for x in chunks:
                small = small * (x % p) % p
            hybrid = pow(small, p - 2, p)
        else:
            hybrid = actual
        assert hybrid == inverse
        if inverse:
            assert normal * inverse % p == 1
    return {"p": p, "prime": is_prime(p), "chunks": len(chunks),
            "normal_product": normal, "inverse": inverse, "mont_calls": c.calls,
            "noncanonical_b_calls": c.noncanonical_b_calls, "status": "PASS"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-mul", type=int, default=10000, help="random Mont products per boundary modulus")
    parser.add_argument("--random-products", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args(argv)
    if not (0 <= args.random_mul <= 1000000 and 0 <= args.random_products <= 10000):
        parser.error("requested single-threaded test count is outside bounded audit limits")
    started = time.monotonic()
    rng = random.Random(args.seed)
    moduli = sorted({3, 5, 7, 17, 257, (1 << 31) - 1, (1 << 31) + 1,
                     (1 << 32) - 5, MASK32, (1 << 32) + 1,
                     next_prime(1 << 32), (1 << 61) - 1, last_prime(CAP), CAP})
    arithmetic = []
    for p in moduli:
        c = make_mont(p)
        av = sorted({0, 1, p // 2, p - 2, p - 1})
        bv = sorted({0, 1, p - 1, p, p + 1, MASK32, 1 << 32,
                     CAP - 1, CAP, (1 << 63) - 1, 1 << 63, MASK - 1, MASK})
        for a in av:
            for b in bv:
                mont_mul(a, b, c)
        edges = c.calls
        for i in range(args.random_mul):
            b = rng.randrange(CAP + 1) if i % 2 == 0 else rng.getrandbits(64)
            mont_mul(rng.randrange(p), b, c)
        arithmetic.append({"p": p, "boundary_cases": edges, "random_cases": args.random_mul,
                           "calls": c.calls, "noncanonical_b_cases": c.noncanonical_b_calls,
                           "maximum_unreduced": c.maximum_unreduced, "status": "PASS"})
    for even in (2, 4, 1 << 32):
        try:
            make_mont(even)
        except ValueError:
            pass
        else:
            raise AssertionError("even modulus unexpectedly accepted")

    product_count = 0
    product_mont_calls = 0
    lengths = (0, 1, 2, 3, 7, 16, 31, 32, 33, 63, 64, 65, 127)
    for p in moduli:
        for length in lengths:
            chunks = [CAP if i & 1 else 1 for i in range(length)]
            result = verify_products(chunks, p)
            product_count += 1
            product_mont_calls += result["mont_calls"]
        for chunks in ([0], [p], [1, p, CAP], [1], [CAP]):
            result = verify_products(list(chunks), p)
            product_count += 1
            product_mont_calls += result["mont_calls"]
    for _ in range(args.random_products):
        p = rng.choice(moduli)
        chunks = [rng.randrange(CAP + 1) for _ in range(rng.randrange(97))]
        result = verify_products(chunks, p)
        product_count += 1
        product_mont_calls += result["mont_calls"]

    real_results = []
    for kind, n in (("factorial", 25206), ("primorial", 104561), ("primorial", 230563)):
        if kind == "factorial":
            chunks = pack_factors(range(2, n + 1))
            exact = math.factorial(n)
        else:
            factors = primes_up_to(n)
            assert factors[-1] == n, "GSRSV requires a prime primorial endpoint"
            chunks = pack_factors(factors)
            exact = math.prod(factors)
        primes = sorted({3, next_prime(n + 1), 1000003, (1 << 32) - 5,
                         next_prime(1 << 32), (1 << 61) - 1, last_prime(CAP)})
        checks = [verify_products(chunks, p, exact_product=exact) for p in primes]
        real_results.append({"kind": kind, "n": n, "chunks": len(chunks),
                             "product_bits": exact.bit_length(),
                             "packed_chunks_sha256": hashlib.sha256(b"".join(x.to_bytes(8, "little") for x in chunks)).hexdigest(),
                             "checks": checks, "status": "PASS"})
        product_count += len(checks)
        product_mont_calls += sum(x["mont_calls"] for x in checks)

    report = {"status": "PASS", "single_threaded": True, "no_gpu": True,
                      "seed": args.seed, "seconds": round(time.monotonic() - started, 6),
                      "mul_oracle_cases": sum(x["calls"] for x in arithmetic),
                      "product_cases": product_count, "mont_calls_inside_products": product_mont_calls,
                      "p2_excluded_from_montgomery": True, "dispatch_threshold_checked": 32,
                      "arithmetic": arithmetic, "real_inputs": real_results}

    def json_safe(value):
        # Preserve exact 64-bit audit values through JavaScript JSON viewers.
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if type(value) is int and abs(value) > (1 << 53) - 1:
            return str(value)
        return value

    print(json.dumps(json_safe(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
