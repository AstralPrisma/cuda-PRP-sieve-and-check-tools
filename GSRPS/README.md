# GSRPS

GSRPS is a CUDA Fermat probable-prime checker for generalized Sierpinski/Riesel candidates

```text
N = k*b^n + c,  where c is +1 or -1
```

Unlike GFPS, the GSRPS command-line argument `n` is the exponent itself.

> 中文提示：GSRPS 的 `n` 就是表达式中的指数。输出 `PRP` 只代表通过指定底数的 Fermat 测试，不是确定性素数证明。

## Result status

For a selected Fermat witness `a` (2 by default), GSRPS evaluates

```text
a^(N-1) mod N
```

- `result=COMPOSITE` means the Fermat congruence failed, so the candidate is composite, assuming the computation is correct.
- `result=PRP` means the congruence passed for that witness. It does **not** prove that `N` is prime.
- `checksum=x:y` is a diagnostic checksum, not a primality certificate.
- `verification=cpp_int-PASS` means the GPU classification agreed with an independent Boost.Multiprecision computation in that run. It still does not turn a Fermat PRP into a proof.

Confirm valuable PRPs with an independent implementation and use deterministic proof software when a primality proof is required.

## Requirements

- Linux, WSL2, or 64-bit Windows with an NVIDIA GPU.
- NVIDIA CUDA Toolkit with `nvcc` (CUDA 13.x is used for the current development builds).
- A CUDA-supported C++17 host compiler.
- Boost headers, specifically Boost.Multiprecision.
- CUB, supplied by modern CUDA Toolkits through CCCL.

Optional diagnostic builds may define `GSRPS_GMP_DIAGNOSTIC` and link GMP/GMPXX. Normal builds do not require GMP.

## Build

From the `GSRPS` directory on Linux or WSL:

```bash
./scripts/build-linux.sh build sm_89
```

On Windows, install a CUDA-supported Visual C++ toolchain, set `BOOST_ROOT` to
the directory containing the `boost` folder, then run:

```bat
scripts\build-windows.bat build sm_89
```

Omit the architecture to build `sm_86`, `sm_89`, `sm_100`, and `sm_120`, or
pass a subset after the output directory. The scripts retain per-thread
default-stream semantics and enable MSVC's conforming preprocessor for CUB.

Run the built-in correctness tests after every new build:

```bash
./GSRPS --selftest
```

## Checking a candidate

An expression may be supplied directly:

```bash
./GSRPS --check '326*799^325799-1'
```

or as four numeric fields:

```bash
./GSRPS --check 326 799 325799 -1
```

Add a witness as the final argument to override the default witness 2:

```bash
./GSRPS --check '326*799^325799-1' 3
./GSRPS --check 326 799 325799 -1 3
```

Quote expressions containing `*` in a shell so that filename expansion cannot alter them.

The final machine-readable summary has this general form:

```text
gsrps-check: N=326*799^325799-1, witness=2, ..., raw_one=no,
             checksum=..., verification=not-requested, result=COMPOSITE
```

Automation should parse the final `gsrps-check:` line and its `result=PRP` or `result=COMPOSITE` field. A progress line reaching 100% is not by itself a result receipt.

## GPU tuning and throttling

Global options may appear anywhere:

```text
--force-ntt-blocks <1..4096>  Disable NTT block auto-tuning and force a cap.
--force-window-bits <1..8>    Disable automatic sliding-window selection.
--duty-percent <1..100>       GPU work-time duty cycle for a full check; default 100.
--verify-cpp-int              Repeat the complete check using Boost cpp_int on the CPU.
--tuning-cache-dir <path>     Persistent tuning cache; default .gsrps_tuning_cache.
--tuning-cache-max-age-hours N
                              Refresh entries after N hours; default 24.
--no-tuning-cache             Disable the persistent tuning cache.
```

Equivalent environment controls are available for unattended runs:

```text
GSRPS_NTT_BLOCKS=1..4096
GSRPS_WINDOW_BITS=1..8
GSRPS_DUTY_PERCENT=1..100
GSRPS_TUNING_CACHE_DIR=<path>
GSRPS_TUNING_CACHE_MAX_AGE_HOURS=1..8760
GSRPS_DISABLE_TUNING_CACHE=1
```

The automatic tuner keys entries to the GPU, build, and arithmetic shape. Delete the cache or use `--no-tuning-cache` when diagnosing performance. A tuning cache is not a computation checkpoint and contains no partial PRP result.

`--duty-percent` inserts rest time between GPU work intervals. It lowers sustained utilization at the cost of longer wall-clock time. Explicit `--force-*` values are intended for benchmarking or known hardware-specific choices; defaults are normally safer for portable runs.

## Benchmarks

```bash
./GSRPS --bench-square <k> <b> <n> <c:+1|-1> <iterations>
./GSRPS --bench-mul    <k> <b> <n> <c:+1|-1> <iterations>
```

These commands measure arithmetic primitives and do not perform a complete PRP classification. Iterations must be between 1 and 10000.

## Checkpoint behavior

Enable periodic arithmetic checkpoints for a fresh check with:

```bash
./GSRPS \
  --checkpoint work.ckpt \
  --checkpoint-every-bits 100000 \
  --check '326*799^325799-1' 2
```

Resume the same candidate and witness with:

```bash
./GSRPS \
  --checkpoint work.ckpt \
  --checkpoint-every-bits 100000 \
  --resume-checkpoint \
  --check '326*799^325799-1' 2
```

Inspect a checkpoint without resuming it:

```bash
./GSRPS --checkpoint-info work.ckpt
```

`--checkpoint-every-bits 0` disables periodic writes but still saves on a clean
completion or handled interrupt. `Ctrl+C`/`CTRL_BREAK` is handled at the next
complete sliding-window operation or at most 64 consecutive zero-bit squares;
the checker synchronizes the CUDA stream, writes a safe checkpoint, and exits
with status 130.

Checkpoint format `GSRPCK1` stores canonical radix limbs and enough metadata to
reconstruct the exact modular-exponentiation prefix. It is fixed little-endian,
SHA-256 protected, and portable between the Windows and Linux builds of the
same arithmetic layout. Candidate, witness, layout, window, progress, exact
file length, digit ranges, and residue canonicality are all validated. A
damaged or mismatched file is rejected instead of silently starting over.

The `.gsrps_tuning_cache` directory remains separate: it only caches performance
choices and contains no partial PRP result.

When the checker is used through the toolkit's Python clients, each child runs
in an isolated process group/session and a single interrupt is propagated from
outer client to wrapper to checker. Repeated interrupts are ignored while the
child is being synchronized and its checkpoint/output is being drained. The
checker flushes standard output through redirected pipes, so Windows clients
display tuning, progress, and checkpoint lines while the calculation is still
running instead of releasing them only at process exit.

## Known limits

- `k >= 1`, `b >= 2`, `n >= 1`, and `c` must be exactly `+1` or `-1`.
- The current general-base implementation requires `b <= 998244353`.
- Some shapes are rejected when `k*b^(n mod g)` does not fit the internal 32-bit fixed divisor, when the conservative two-prime CRT coefficient bound is exceeded, or when the required NTT length exceeds `2^21`.
- The witness must satisfy `2 <= witness < N`.
- `--verify-cpp-int` can require very large CPU time and memory. It is intended for validation, especially on smaller cases.
- The program uses CUDA Graphs and CUB scans. Build with `--default-stream per-thread` as shown above and retain the explicit-stream source behavior.
- The checker provides a Fermat PRP result, not Pocklington, Proth, ECPP, or another deterministic proof.
- Long GPU computations remain vulnerable to hardware faults. Recheck important PRPs independently.

Run `./GSRPS` without arguments to print the authoritative usage summary for the exact binary being used.
