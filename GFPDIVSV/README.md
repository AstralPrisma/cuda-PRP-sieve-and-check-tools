# GFPDIVSV 1.1 CUDA

Generalized Factorial / Primorial Factor-Removal Siever.
Fix `A=n!` or `A=n#`, remove one allowed divisor, and sieve `A/d-1`,
`A/d+1`, or both signs together. `n#` means the product of primes not exceeding n.
Survivors are candidates, **not primality proofs**; use GFPPS for subsequent PRP checks.

This distribution contains the CUDA sieve only. A compatible NVIDIA GPU is
required for sieving. Host exact-integer construction, low-prime cancellation
checks and independent verification of every reported factor remain enabled.
There is no stand-alone host-only sieve executable or automatic fallback.

## Generate and continue

```sh
./bin/GFPDIVSV_sm_89 --n 26951 --family factorial --sign both \
  -P 1e11 -o div_26951_both.txt
./bin/GFPDIVSV_sm_89 --n 366439 --family primorial --sign both \
  -P 1e11 -o div_366439_both.txt
./bin/GFPDIVSV_sm_89 -i div_26951_both.txt -P 2e11
./bin/GFPDIVSV_sm_89 -i div_26951_both.txt -P 3e11 -o continued.txt
```

On Windows use the corresponding `.exe`. Quote paths containing spaces.
For a new search only n and family are required; the default is both signs,
target `1e8` and an automatically generated output name. Explicit `--sign +`
or `--sign -` keeps one sign. Existing unrelated output files are never overwritten.

- Factorial divisors: every integer `2..n-1`.
- Primorial divisors: odd primes from3 through the prime before the largest
  prime factor of `n#`. The largest factor and2 are excluded from this search.
- `--divisor-min` / `--divisor-max` may select a smaller domain.
- Supported n:3..10,000,000, subject to the16,000,000-bit mother-product limit.
- Maximum sieve bound: `2^62-1`; exact integral scientific notation is accepted.

The CUDA backend computes batches of exact modular product residues. Factorial
prime valuations are represented as binary-weight groups; both signs share the
same residue. Boundary primes cancelled by the removed divisor are handled
separately. A candidate equal to a sieve prime itself is retained.

## Checkpoints and diagnostics

The self-contained `GFPDIVSV 1` text snapshot binds n, family, divisor domain,
sign mask, mother-product hash, actual tested frontier and ordered `(divisor,sign)`
survivors to SHA-256. Keep this file for continuation; factor logs are optional.

- Default progress interval1second, checkpoint interval60seconds.
- Ctrl+C or creating `--stop-file FILE` requests safe interruption. An incomplete
  GPU batch is discarded and replayed after resume; it never advances the saved frontier.
- Interrupted exit code is130. GPU/setup/arithmetic failures are errors, not completed searches.
- `--max-batches N` is a bounded `PARTIAL` diagnostic, not evidence of reaching the target.
- `--verify-cuda` compares every residue with independent exact host arithmetic.
- `--selftest` runs bounded CUDA arithmetic checks without altering search files.
- `--device N`, `--batch-primes N`, `--gpu-chains 1|2|4|8` and
  `--gpu-tile-chunks N` control GPU selection and batching. See `--help` for bounds.
- Never let two processes write the same snapshot. Different candidates keep independent PRP checkpoints.

## Adaptive batching and prime supply

The default `--prime-mode auto` measures real work on the selected GPU rather
than assuming that more CPU threads or a larger batch is always faster. A cold
profile compares fixed, identical prime samples using batches32,768 /131,072 /
262,144 /524,288. Each trial checks every proposed factor using independent
exact arithmetic and operates on a disposable copy of the survivor mask.
Calibration never advances the actual checkpoint or writes trial factors.
The smallest batch within3% of the best measured throughput is preferred.

If prime generation is negligible, auto keeps a synchronous persistent iterator.
Otherwise a bounded, ordered producer queue overlaps prime generation with CUDA.
It starts with one supplier thread. During real work, sustained supplier waiting
above8% may trigger a larger supplier count, capped by CPU and memory estimates.
A change must improve throughput by at least3%; otherwise the previous count is
restored and the decision is frozen for this run. No work is skipped when a queue
is replaced: only fully verified batches advance the saved frontier.

Profiles are checksummed sidecars, `OUTPUT.tuning` by default. Their identity
includes GPU UUID/architecture/SM count, driver/runtime, arithmetic configuration,
mother-product hash, divisor/sign domain, survivor-density/depth buckets,
prime-region span, primesieve identity and CPU count. Invalid or
mismatched profiles are recalibrated. Unrelated files are never overwritten.
Removing a profile is safe; it only makes the next run calibrate again.

```sh
# Automatic, including cached calibration:
./bin/GFPDIVSV_sm_89 -i survivors.txt -P 2e12
# Recalibrate or select a separate profile:
./bin/GFPDIVSV_sm_89 -i survivors.txt -P 2e12 --retune --tuning-cache run.tuning
# Pin the resources while retaining auto mode:
./bin/GFPDIVSV_sm_89 -i survivors.txt -P 2e12 --batch-primes 262144 --prime-threads 1
# Fully manual pipeline; no calibration or profile writes:
./bin/GFPDIVSV_sm_89 -i survivors.txt -P 2e12 --prime-mode pipeline --prime-threads 1 --batch-primes 262144
```

`--prime-mode iterator` and `array` are synchronous manual alternatives; `array`
retains the legacy array-generation path for diagnosis. Multiple supplier threads
require `auto` or `pipeline`. A manual `--prime-threads` overrides the automatic
thread cap; the user is responsible for its memory budget. Calibration adds cold
startup work; cached profiles avoid repeating it. Performance is task- and
hardware-dependent, and high GPU utilization alone is not the tuning objective.

Run `tests/test_auto.py` with `--binary` and `--report` to check cache safety,
manual overrides, ordered prime streams, supplier hysteresis and interruption.

## Runtime libraries

Release packages include primesieve12.x in `bin/lib/`, with its BSD notice.
The program searches its executable directory, then its `lib/` directory,
then system library locations. An explicit `--primesieve-library FILE` or
`GFPDIVSV_PRIMESIEVE_LIBRARY` can select another compatible library.
An explicitly invalid library is an error; there is no fixed maintainer path.
Persistent/parallel suppliers require the bundled primesieve12.x iterator ABI;
an incompatible iterator version is rejected rather than assumed compatible.
Without primesieve, the bounded built-in generator supports only targets up to`1e12`.

Official Linux binaries target Ubuntu22.04+ and statically link GMP6.3.0 for
host-only exact construction/verification; no separate GMP executable or runtime
libgmp installation is required. Corresponding GMP source and license notices
are included. Windows uses Boost for these host operations and the same CUDA
residue backend. Large initial product construction can take time before sieving starts.

## Offline task conversion

```sh
python scripts/candidates.py div_26951_both.txt --expressions expressions.txt
python scripts/candidates.py div_26951_both.txt --out-dir standalone_inputs
python scripts/gfpdivsv_to_tasks.py div_26951_both.txt --out-dir server_inputs
```

The first two create expression-only GFPPS inputs. The last creates canonical
two-line task files, or `--manifest FILE.jsonl` exports JSONL. Converters verify
the snapshot, reject existing output, do not upload anything and award no credit.
Do not convert incomplete benchmark subsets into production tasks.

## Build and validation

Use `scripts/build-linux.sh OUTPUT sm_89` or `scripts/build-windows.bat OUTPUT sm_89`;
omit the target for sm_86/89/100/120. CUDA C++17 and Boost headers are required.
Set `BOOST_ROOT` to the header prefix. Linux optionally accepts a static `GMP_ROOT`
containing `include/gmp.h` and `lib/libgmp.a`; without it Boost supplies host arithmetic.
CUDA is required in either build. See [validation](VALIDATION.md) and release manifests
for actual physical-GPU coverage, separately from architecture/startup checks.

## 中文

GFPDIVSV固定`n!`或`n#`，移除单个允许的因子，再同时或单独筛正负号。
本次仅发布CUDA筛子，保留必要的主机构造和独立因子复核，不提供独立CPU筛分版本。
候选文件可自包含续筛及归档，并可用配套脚本拆成GFPPS任务；幸存不等于质数。
Windows/Linux分别提供四种SM程序，实际显卡验收范围以发布元数据为准。

1.1默认按实际计时自适应选择批量及质数供数方式，支持持续迭代器和有界并行预取。
只有供数不足且增加线程后的吞吐收益至少3%时才保留更多线程，否则回退并冻结。
调优使用独立副本，不推进真实断点、不重复记录因子；旁边的`.tuning`文件只是可删除的性能缓存。
`--batch-primes`和`--prime-threads`可手动覆盖；显式选择非auto模式则完全关闭自动调优。
