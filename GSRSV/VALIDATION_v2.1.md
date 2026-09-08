# GSRSV 2.1 product-path validation

Validated on an NVIDIA RTX 4060 Laptop GPU (sm_89), CUDA 13.3, Windows and
Ubuntu 24.04 under WSL. Other shipped SM targets are cross-compiled; no runtime
claim is made for hardware not present in this validation environment.

## Arithmetic change

For odd p <= 2^62-1, R=2^64, reduced accumulator a<p and raw chunk b<R,
T=a*b<p*R. Montgomery REDC computes u=(T+m*p)/R<2p, so its uint64 sum does not
overflow and one subtraction remains sufficient. No per-chunk remainder is
needed. Pre-seeding the accumulator with R^(C+1) for C chunks makes the final
product equal B*R; inversion remains Montgomery encoded until one final decode.
The exact low-word cancellation condition makes its carry equal to (lo != 0).

Prime 2 and very small 32-bit products retain the old path. The power-form
arithmetic and candidate marking, factor validity checks, serialization and
stopping rules are unchanged. No additional large device tables are allocated.

## Correctness evidence

- Independent Python integer audit: 140859 standalone REDC cases and 673
  product/inverse cases, with 581238 further internal REDC comparisons.
- Direct GPU-returned inverses: 46 groups / 3316 values compared individually
  with exact CPU modular arithmetic on Linux control, Linux optimized and
  Windows optimized builds. Includes real 25206!, 104561#, 230563#, C=0/1 and
  the 31/32/33 threshold, primes on both sides of 2^32 and near the maximum p.
- Forty full sieve/oracle and continuation groups cover factorial and
  primorial, independent/twin modes, self-factor preservation, both 32-bit
  boundary sides, and near-2^62 intervals with deliberately constructed hits.
  Each factor is independently verified; racing streams may choose different
  valid factors, but complete survivor sets must match.
- Windows full factorial and primorial survivor files match the Linux control
  after normalizing platform line endings.
- An optimized run interrupted by SIGINT saved a valid frontier, was resumed
  by the original program, and matched the original uninterrupted output.
  A zero process exit code alone is not treated as proof that pmax was reached.

These tests are evidence for the tested paths, not a proof that all possible
software defects, devices or hardware faults are covered. Sieve survivors are
not primality claims. No production candidate file was modified by validation.

## Timing

Both signs independently, k=2..100000, p in (100000000000,102000000000],
libprimesieve 12, four prime-generator threads, two CUDA streams, batches of
262144 primes, four prefetched batches, two batches/region. Factor logging and
CPU factor verification disabled only for the timed runs. Two complete runs
per engine/case in alternating/reversed order; wall time includes startup and
output. Same primes and byte-identical survivors on both sides.

| Multiplier | Original wall time | Optimized wall time | Throughput |
| --- | ---: | ---: | ---: |
| 25206! | 14.439 s | 4.121 s | 3.50x |
| 230563# | 15.391 s | 4.673 s | 3.29x |

Three-pair tests below 2^32 and above it also matched complete output; measured
whole-process gains ranged approximately 2.6–3.6x. Short-run setup/generator
variation remains visible, so these numbers are not universal speed guarantees.
CUDA event times are summed across streams and must not be added to host time.
