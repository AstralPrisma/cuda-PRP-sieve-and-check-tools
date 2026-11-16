#!/usr/bin/env python3
"""Independent reference sieve for GPRSV_N validation.

Enumerates, for every prime p <= pmax, the exponents n in [nmin, nmax] with
p | k*b^n-1 and p | k*b^n+1 using exact Python integer arithmetic.  This is a
literal one-multiplication-per-step walk, completely independent of the CUDA
program's BSGS logic.
"""
import math
import sys


def primes_upto(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def term_equals_p(k, b, n, p, c):
    """True when k*b^n+c is itself the prime p (such a term must not be removed)."""
    if n * math.log2(b) + math.log2(k) > math.log2(p + 1) + 1e-9:
        return False
    return k * b ** n + c == p


def survivors(k, b, nmin, nmax, pmax, mode, signs):
    """mode: 'paired' or 'independent'; signs: set of {1,-1} for independent."""
    minus = set(range(nmin, nmax + 1))
    plus = set(range(nmin, nmax + 1))
    for p in primes_upto(pmax):
        if p < 2:
            continue
        bm = b % p
        km = k % p
        if bm == 0 or km == 0:
            continue
        x = pow(b, nmin, p)
        for n in range(nmin, nmax + 1):
            v = x * km % p
            if v == 1 and not term_equals_p(k, b, n, p, -1):
                minus.discard(n)
            if v == p - 1 and not term_equals_p(k, b, n, p, 1):
                plus.discard(n)
            x = x * bm % p
    if mode == "paired":
        both = minus & plus
        return sorted((n, 0) for n in both)
    rows = []
    for n in sorted(minus):
        if -1 in signs:
            rows.append((n, -1))
    for n in sorted(plus):
        if 1 in signs:
            rows.append((n, 1))
    return sorted(rows)


def read_gprsvn(path):
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()
        rows = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                rows.append((int(parts[0]), 0))
            else:
                rows.append((int(parts[0]), int(parts[1])))
    return header, rows


def main():
    if len(sys.argv) < 7:
        print("usage: ref_sieve.py K B NMIN NMAX PMAX FILE [paired|independent]")
        return 2
    k, b, nmin, nmax, pmax = (int(sys.argv[i]) for i in range(1, 6))
    path = sys.argv[6]
    mode = sys.argv[7] if len(sys.argv) > 7 else "paired"
    header, rows = read_gprsvn(path)
    expected = survivors(k, b, nmin, nmax, pmax, mode, {1, -1} if mode == "independent" else set())
    if mode == "paired":
        got = sorted((n, 0) for n, c in rows)
    else:
        got = sorted(rows)
    ok = got == expected
    print(f"{'MATCH' if ok else 'MISMATCH'}: file={path} mode={mode} pmax={pmax}")
    print(f"  header: {header}")
    print(f"  reference survivors={len(expected)}, file survivors={len(got)}")
    if not ok:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        print(f"  missing ({len(missing)}): {missing[:20]}")
        print(f"  extra   ({len(extra)}): {extra[:20]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
