# GPRSV_N

GPRSV_N is a CUDA siever for `k*b^n - 1` and `k*b^n + 1` with a **fixed `k` and a
variable exponent interval** `[N1, N2]`. It is the companion of
[`GPRSV_K`](../GPRSV_K/), the original GPRSV tool, which does the opposite:
fixed `n`, variable `k`.

The two directions are not interchangeable algorithmically:

- **K direction.** `p | k*b^n±1` means `k ≡ ±(b^n)^-1 (mod p)`. One modular
  inverse per prime gives both residue classes of `k`; the cost does not grow
  with the length of the `k` interval.
- **N direction.** `p | k*b^n±1` means `b^n ≡ ±k^-1 (mod p)`, a discrete
  logarithm base `b`. A direct walk costs one modular multiplication per
  `(p, n)` pair. GPRSV_N instead solves it with a baby-step/giant-step search
  costing about `2*sqrt(length)` multiplications per prime, which is what makes
  long exponent intervals practical on a GPU.

For the same candidate count and the same sieve depth, GPRSV_N is therefore
slower than GPRSV_K. That is inherent to the problem, not an implementation
detail; measured factors are in [Performance](#performance).

Only `k*b^n±1` is supported. `k*n!±1` and `k*n#±1` remain in GPRSV_K; passing
`-t` to GPRSV_N is an explicit error.

## Current version

1.1 (2026-11-16). Changes since 1.0:

- The default sieve mode is now **independent**: `+1` and `-1` keep separate
  bitmaps. Version 1.0 defaulted to the paired/twin mode, which for a fixed-`k`
  exponent search almost always eliminates everything: the paired sieve keeps
  an `n` only while both signs are free of tested factors, and whenever `3`
  divides neither `k` nor the base, `p=3` divides one of the two signs for
  every `n` (see [Paired mode and p=3](#paired-mode-and-p3)). Use `--paired`
  for a twin search.
- Paired mode now refuses to start in that situation and explains why, instead
  of silently producing an empty survivor list.
- The progress/ETA line no longer prints the localized timezone name, which
  produced mojibake on Chinese Windows consoles (`%Z` returns a GBK string while
  the console is UTF-8).

## Usage

```text
GPRSV_N -k 12345 -b 2 -n 1000000 -N 1099999 -P 1e10
GPRSV_N -k 12345 -b 2 -n 1000000 -N 1099999 --sign minus -P 1e11
GPRSV_N -k 3387  -b 2 -n 1000000 -N 1099999 --paired -P 1e11
GPRSV_N -i k12345_b2_n1000000-1099999.txt -P 1e11 -o next.txt
```

| Option | Meaning |
| --- | --- |
| `-k, --k K` | fixed `k` |
| `-b, --base B` | base `b` |
| `-n, --nmin N1` | first exponent (inclusive) |
| `-N, --nmax N2` | last exponent (inclusive) |
| `--paired, --twin` | paired/twin sieve (both signs must survive) |
| `-s, --independent` | independent sieve, the default |
| `--sign both\|plus\|minus` | which signs to keep in independent mode (default `both`) |
| `-p, --pmin P0`, `-P, --pmax P1` | test primes `P0 < p <= P1` |
| `-i, --inputterms FILE` | resume from a GPRSV_N output file |
| `-o, --outputterms FILE` | write remaining `n` values |
| `-I, --inputfactors FILE` | apply existing `p \| term` factors |
| `-O, --outputfactors FILE` | append newly found factors |
| `-E, --expressions FILE` | also write LLR/PFGW/PRST-ready `k*b^n±1` lines |
| `-A, --applyandexit` | apply `-I` and write output without sieving |
| `--backend auto\|walk\|bsgs` | algorithm; `walk` is the naive one-multiply-per-`n` path |
| `--bsgs-m M` | baby steps, power of two; 0 = automatic by interval length |
| `-4/-5/-6` | lowest acceptable efficiency stopping (same contract as GPRSV_K) |
| `--threads`, `--blocks`, `-w`, `--prime-*` | same meaning as GPRSV_K |

`--cpu-small-prime` (default 1e5) selects the primes handled on the host; the
host path mirrors the GPU algorithm and can be forced for cross-checks with
`--cpu-small-prime 1e12`.

An existing output file is never overwritten unless it is also the `-i` input,
so a checkpoint that is still needed cannot be clobbered by accident. Output is
written to a temporary file and renamed into place.

## Output format

```text
ABC 12345*2^$a+1 & 12345*2^$a-1 // GPRSV_N v1.1 sieved_to=10000000019 nmin=1000000 nmax=1099999 terms=637 signs=both
1000103 +1
1000111 -1
...
```

Paired-mode rows carry only `n` and the header has no `signs=` field. The header
records `k`, the base, the exponent interval and the true sieve depth, and `-i`
resumes from the header rather than from the file name.

## Algorithm

1. The exponent interval is aligned to a multiple of `m`:
   `base_n = (nmin/m)*m`, so `b^(base_n+i*m+j) = b^base_n * (b^m)^i * b^j`.
2. For each prime `p`, with `f = b^-m` and `c = k^-1`, a hit requires
   `b^j = c * f^(q+i)` (and its negation for the other sign). One Fermat
   inversion of `b^m * k` yields both `f` and `c`.
3. Each warp handles one prime: lane 0 does the exponentiation/inversion, and
   the 32 lanes share the baby-step table build and the giant-step walk.
4. Table entries are 32 bits (16-bit value fingerprint plus 16-bit baby index).
   A fingerprint hit is verified by recomputing `b^j`, so truncation can never
   remove a candidate. Duplicate baby values (small multiplicative order) are
   chained through `next[]`, so every `j` remains reachable.
5. Every hit checks whether the term itself equals `p` (for example
   `2^13-1 = 8191`); such a term is not removed.
6. `--backend walk` is the straightforward one-multiplication-per-exponent path,
   used for very short intervals and as a cross-check of the BSGS kernel.

## Paired mode and p=3

In paired mode a candidate survives only while both `k*b^n+1` and `k*b^n-1`
are free of tested factors. If `3 ∤ k` and `3 ∤ b` then `b^n` runs through
`{1,2} (mod 3)` (or is constantly 1), and one of the two signs is divisible by
3 for **every** `n`. A twin search therefore needs `3 | k`. GPRSV_N rejects
this configuration instead of returning an empty file.

```text
$ GPRSV_N -k 1129 -b 2 -n 3500000 -N 4000000 --paired -P 1e10
Fatal error: paired/twin mode can never leave a candidate: 3 divides neither k
nor the base, so for every n one of k*b^n+1 and k*b^n-1 is divisible by 3. ...
```

## Performance

Measured on an RTX 4060 Laptop (Windows, sm_89), component versions as in this
repository. Same candidate count (100,000) and same sieve depth; K direction is
`n = 1e6` with `k ∈ [1000000001, 1000199999]`, N direction is `k = 12345` with
`n` starting at 1e6. Wall time includes prime generation; medians of two runs.

| p <= | GPRSV_K (1e5 k) | GPRSV_N (1e4 n) | GPRSV_N (1e5 n) | GPRSV_N (1e6 n) |
| ---: | ---: | ---: | ---: | ---: |
| 1e9  | 0.57 s | 2.9 s  | 4.5 s   | — |
| 1e10 | 1.75 s | 24.1 s | 37.4 s  | 165 s |
| 1e11 | 8.0 s  | 215 s  | —       | — |

- Same candidate count and depth: the N direction costs 5–8x the K direction at
  `p <= 1e9` and 14–27x at `p >= 1e10`; the gap grows with the interval length
  (about 95x at 1e6 exponents).
- For reference, CPU srsieve2 1.8.8 with 16 workers needs 32.8 s for the same
  `k`, 1e5 exponents and `p <= 1e10` on **one** sign; GPRSV_N needs 37.4 s for
  both signs, i.e. about 1.7x faster per sign.
- `--backend walk` needs 6.1 s where BSGS needs 2.9 s at 1e4 exponents and
  `p <= 1e9`; walk cost grows linearly with the interval while BSGS grows with
  its square root.

## Validation

See [VALIDATION.md](VALIDATION.md). Summary: 11 parameter sets match an
independent Python reference implementation exactly; `walk`/`bsgs`/different
`--bsgs-m`/host-only paths agree byte for byte up to `p <= 1e7`; resuming in two
stages equals one full run; every reported factor was re-verified; and a
`k=1129`, `n=3500000..3509999`, `p <= 1e10` slice matches srsieve2 1.8.8
`-c` (generic logic) exactly for both signs.

Sieve survivors are not primes. A survivor still needs a probable-prime test and
an independent primality proof.

## Building

```text
GPRSV_N/scripts/build-linux.sh build/linux sm_89
GPRSV_N\scripts\build-windows.bat build\windows sm_89
```

Both scripts default to `sm_86 sm_89 sm_100 sm_120`. A CUDA 13.3 toolkit with a
C++17 host compiler is required. The Windows build needs no extra libraries;
Linux links `libdl` and uses static C++/CUDA runtimes in the release build. At
run time GPRSV_N loads primesieve 12.x dynamically from the executable
directory, `bin/lib/`, or the system loader path, and falls back to its built-in
segmented/Miller-Rabin generator when the library is absent.

## License

GPL-2.0-or-later. GPRSV_N is derived from GPRSV, itself a CUDA
port/reimplementation of the mtsieve `twinsieve` application (GPLv2-or-later).
See [`LICENSE`](../LICENSE) and [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## 中文摘要

GPRSV_N 固定 `k`、筛一段指数 `n`，是 GPRSV_K（原 GPRSV，固定 `n` 筛 `k`）的逆方向。
每个素数 `p` 上要求的是 `b^n ≡ ±k^-1 (mod p)` 的离散对数，所以程序默认用大步小步
（BSGS），每个素数约 `2*sqrt(区间长度)` 次模乘；直接逐 `n` 走一遍的 `--backend walk`
只用于很短区间和交叉验证。同候选数、同筛深下比 GPRSV_K 慢 5～27 倍（见上表）。

1.1 起默认是独立模式（正负号各自保留），孪生搜索要显式 `--paired`；成对模式在
`3 ∤ k`、`3 ∤ 底数` 时会直接报错（否则 p=3 会删光全部指数）。输出带
`sieved_to/nmin/nmax/terms` 头，可 `-i` 续筛，`-E` 可导出 LLR/PFGW 表达式。
只支持 `k*b^n±1`，`n!`/`n#` 仍用 GPRSV_K。
