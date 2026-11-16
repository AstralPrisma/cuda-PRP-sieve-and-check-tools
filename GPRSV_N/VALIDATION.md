# GPRSV_N validation record

Host: RTX 4060 Laptop (sm_89), Windows 11, CUDA 13.3.33, MSVC 19.51, 2026-11-16.
All comparisons below are of final output files, not only of term counts.

## 1. Independent Python reference

`tests/ref_sieve.py` walks every `p <= pmax` and every exponent with Python
big-integer `pow()`; it implements the same "a term equal to `p` is not removed"
rule but shares no code with the CUDA program. 11 parameter sets matched
exactly:

| k | base | n range | pmax | mode | survivors |
| ---: | ---: | --- | ---: | --- | ---: |
| 12345 | 2 | 1000–1150 | 1e5 | independent | 40 |
| 12345 | 2 | 1000–1150 | 1e5 | paired | 3 |
| 98 | 3 | 500–900 | 2e5 | independent | 147 |
| 98 | 3 | 500–900 | 2e5 | paired | 4 |
| 97 | 2 | 1–300 | 1e5 | independent | 45 |
| 1 | 2 | 1–120 | 5e4 | independent | 25 |
| 12345 | 2 | 100000–102000 | 1e5 | independent | 609 |
| 12345 | 4 | 1000–1100 | 1e5 | paired | 1 |
| 2305843009213693952 | 3 | 100–300 | 5e4 | independent | 72 |
| 3 | 10 | 7–60 | 3e4 | paired | 2 |
| 6 | 5 | 40–90 | 3e4 | independent | 19 |

The set covers `k = 1` (terms can equal `p`), `k` near `2^61`, even/odd bases,
`nmin < m` and `nmin > m` window alignment, and both sieve modes.

## 2. Internal cross-checks

- `--backend walk`, `--backend bsgs`, `--bsgs-m` 16/64/128 and a host-only run
  (`--cpu-small-prime 1e7`) produce byte-identical files for `p <= 1e7`.
- Two-stage resume (sieve to 1e6, then `-i` to 1e7) equals a single run to 1e7,
  including the header's `sieved_to`.
- The 262 factors written by `-O` were re-verified with Python `pow()`; `-I`
  plus `-A` replays the same factor file to the same survivor table.
- Output replacement is atomic (temporary file plus rename).

## 3. Cross-check against srsieve2 1.8.8

`k = 12345`, `b = 2`, `n = 1000000..1099999`, minus sign only, `p <= 1e6`:

| program | survivors | factors |
| --- | ---: | ---: |
| GPRSV_N (`--sign minus`) | 9275 | 90725 |
| srsieve2 default (Legendre / single-sequence logic) | 11597 | 88403 |
| srsieve2 `-c` (generic logic) | 9275 | 90725 |

GPRSV_N and srsieve2's generic logic produce the same `n` set (empty symmetric
difference) and the same factor count. Every one of the 2322 exponents kept by
srsieve2's default path has a factor `<= 1e6` (for example `n = 1000011` has the
smallest factor 16369). This is an observation for this input only, not a
general statement about srsieve2.

## 4. Large-n slice

`k = 1129`, `b = 2`, `n = 3500000..3509999`, `p <= 1e10`:

| program | -1 survivors | +1 survivors |
| --- | ---: | ---: |
| GPRSV_N (independent) | 275 | 317 |
| srsieve2 1.8.8 `-c`, 16 workers | 275 | 317 |

Symmetric differences are empty. The full interval `n = 3500000..4000000`
(500,001 exponents, `p <= 1e10`) yields 30,889 survivors with GPRSV_N in about
2.5 minutes (-1: 14375, +1: 16514).

This case is also the origin of the 1.1 mode change: for `k = 1129` the paired
mode removes every exponent because `p = 3` divides one sign for every `n`
(even `n`: `k*b^n-1`; odd `n`: `k*b^n+1`). That result is mathematically
correct; 1.1 makes the independent mode the default and rejects the impossible
paired configuration up front.

## 5. Not covered

- Only sm_89 was executed on a GPU. The sm_86, sm_100 and sm_120 builds receive
  compile, cubin and startup checks only.
- No long-duration soak test and no dedicated Ctrl+C checkpoint test; the
  interrupt path reuses GPRSV's "drain submitted batches, then write" logic.
- Algebraic factor removal (for example when `k` is a perfect power) is not
  implemented; such composites survive the sieve and are rejected by the
  probable-prime stage.

## Reproducing

```text
cd GPRSV_N/tests
python ref_sieve.py 12345 2 1000 1150 100000 OUT.txt independent
```

`ref_sieve.py` only uses the Python standard library.
