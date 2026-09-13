# GHCWSV 1.2 validation — 2026-09-13

Published as the GHCWSV component update in suite v2026.09.12. No remote
compute-node deployment or website/client update is included.

## Change boundary

Adapted GNCWSV's 20%-smaller compact-workset rebuild trigger. GHCWSV already
compacts candidates after each completed chunk, so this change refreshes the
GPU prime-batch size instead of introducing another candidate bitmap.
Producer batches are independent, and their immutable tails are consumed using
an explicit cursor. The modular sieve kernel, sign-removal atomics, CPU factor
validation and checksummed v1/v2 candidate formats are unchanged.

`--work-batching fixed` retains the previous fixed-batch scheduling path for
diagnostic comparisons. `primes_per_s` retains its cumulative-average meaning;
recent throughput and p-boundary throughput are added explicitly. ETA is an
estimate based on the recent ~30-second p-boundary window.

## Correctness

- Host scheduler tests: exact 20% trigger rounding, fixed reference, very large
  worksets, batch limits and exhaustive producer-cursor traversal.
- Independent Python modular oracle: both signs, individual signs, auto/direct/
  transform paths, tiny primes equal to the candidate, high p near 2^62,
  malformed/truncated state rejection and conversion preservation.
- GPU selftest: 252 arithmetic / transformed-target cases on RTX4060 Laptop.
- Large workset: 2,100,001 distinct n, crossing the 2-million-pair per-launch
  limit. Independent expected counts: plus 1,400,001; minus 1,050,000.
- Original 1.1, adaptive 1.2 and fixed-reference output files are byte-identical
  for b=7, n=100001..500000, both signs, p<=2,000,000: 25,096 survivors.
- Current user-state copy (p=4,053,294,571, 16,531 survivors), extended by
  5,000,000 in p: original/adaptive results identical, 16,530 survivors.
- SIGINT in the middle of a producer batch, then resume to p=10,000,000:
  byte-identical to uninterrupted adaptive run, 22,681 survivors.

## Performance — Linux / RTX4060 Laptop

No other GPU work was running. Each program ran sequentially. These are bounded
tests, not a claim about completing a full sieve to 5e10.

| Test | Original 1.1 wall seconds | Adaptive 1.2 wall seconds |
|---|---:|---:|
| Fresh range, warm repeat 1 | 11.365 | 2.592 |
| Fresh range, warm repeat 2 | 11.621 | 2.522 |
| Already sparse input, repeat 1 | 1.822 | 1.631 |
| Already sparse input, repeat 2 | 1.747 | 1.833 |

Fresh-range speedup is about 4.5x including process overhead. Seven workset
refreshes grow the GPU batch from 11 to 80 primes. Sparse input shows no clear
speedup; the old program already starts with a suitable batch on that input.
The initial cold measurement was 15.716s vs 2.820s; it is not used for the
headline speedup. Raw measurements are in validation/adaptive-1.2/RELEASE_TEST.json.

## Builds and scope

Windows and Ubuntu22.04-compatible Linux, sm_86/sm_89/sm_100/sm_120. Other
architectures are cross-compiled and checked using CPU-reference mode only;
actual GPU testing is limited to sm_89 on the local RTX4060 Laptop.
Linux binaries use static CUDA runtime / libstdc++ / libgcc, but still require
the NVIDIA driver and glibc (highest required version: GLIBC_2.34).
Per-platform results, native architecture lists and source checksums accompany
the release packages. User candidate files are not modified by validation.

## Reproducing bounded regressions

From the repository root, the scheduler unit test needs no GPU:

```sh
c++ -std=c++17 -O2 -I GHCWSV/src GHCWSV/tests/test_batching.cpp -o /tmp/test_ghcw_batching
/tmp/test_ghcw_batching
python GHCWSV/tests/test_both.py --binary /path/to/GHCWSV_sm_89
```

After stopping competing GPU work, explicitly enable GPU tests:

```sh
python GHCWSV/tests/test_both.py --binary /path/to/GHCWSV_sm_89 --gpu
python GHCWSV/tests/test_adaptive.py --binary /path/to/GHCWSV_sm_89 --baseline /path/to/old/GHCWSV --gpu
```

The portable adaptive test compares fixed/adaptive scheduling and a split
prefix followed by resume. Actual OS signal tests are separate evidence in
the recorded validation; the split-prefix test is not advertised as a signal test.
