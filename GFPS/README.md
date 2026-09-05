# GFPS 4.1

GFPS is a CUDA probable-prime checker for generalized Fermat numbers

```text
N = b^(2^n) + 1
```

Here `b` is the integer base and the command-line argument `n` is the power-of-two index: the exponent is `2^n`, not `n` itself.

> 中文提示：命令行中的 `n` 表示指数 `2^n`。程序执行的是 Fermat probable-prime test，不是确定性素数证明。

## Result status

GFPS evaluates the base-2 Fermat congruence

```text
2^(N-1) mod N
```

- `result_is_one=no` proves that the candidate is composite, assuming the computation is correct.
- `result_is_one=yes` means that the candidate is a base-2 probable prime (PRP). It does **not** prove that `N` is prime. Carmichael numbers and other Fermat pseudoprimes can pass.
- `hash64` is a diagnostic residue hash, not a primality certificate.
- Checkpoint SHA-256 protects checkpoint integrity; it is not a mathematical proof.

Partial runs report progress and `hash64` but cannot classify the candidate. `result_is_one` is emitted only after all exponent bits have been processed.

Independently verify every interesting PRP with a trusted implementation and, when a proof is required, use an appropriate deterministic method such as Pocklington with the known factorization of `N-1`.

## Requirements

- Linux, WSL2, or 64-bit Windows with an NVIDIA GPU.
- NVIDIA CUDA Toolkit with `nvcc` (CUDA 13.x is used for the current development builds).
- A CUDA-supported C++17 host compiler.
- Boost headers, specifically Boost.Multiprecision (`boost::multiprecision::cpp_int`).

GFPS does not require a separate GMP library.

## Build

From the `GFPS` directory on Linux or WSL:

```bash
./scripts/build-linux.sh build sm_89
```

On Windows, install a CUDA-supported Visual C++ toolchain, set `BOOST_ROOT` to
the directory containing the `boost` folder, then run:

```bat
scripts\build-windows.bat build sm_89
```

Omit the architecture to build `sm_86`, `sm_89`, `sm_100`, and `sm_120`, or
pass any subset after the output directory. A binary built for one architecture
should not be assumed to support another GPU generation. The scripts use C++17,
`-O3`, and per-thread default-stream semantics.

Run the built-in tests before using a new build:

```bash
./GFPS --selftest
```

## Global GPU controls

Global options may appear anywhere in the command line:

```text
--ntt-blocks <1..4096>   Force the maximum number of NTT blocks.
                         Omit it to use automatic selection.
--duty-percent <1..100>  GPU work-time duty cycle for resident squaring.
                         The default is 100.
--batch-bits <0..4096>   Checked carry batch size; default 512.
                         0 uses adaptive carry after every square.
--reference-mode        Original padded NTT and adaptive per-square carry.
--diagnostic-force-replay
                         Replay every carry batch for regression testing.
```

Backward-compatible environment variables are also accepted:

```text
GFPS_NTT_BLOCKS=1..4096
GFPS_DUTY_PERCENT=1..100
GFPS_DISABLE_CUDA_GRAPHS=1
```

Version 4.1 uses half-length negacyclic transforms, DIF/DIT ordering, and fused
shared-memory kernels by default. Checked carry batches reduce CPU/GPU
synchronization: an unconverged carry is latched, the batch-start residue is
restored, and the whole batch is replayed with adaptive carry before its result
is accepted. Checkpoint writes finish and validate any pending batch first.

`--duty-percent` inserts rest time between GPU work intervals. It reduces
sustained GPU use but increases wall-clock time. A duty below 100 automatically
disables carry batching. Windows accumulates approximately 10 ms of work before
a high-resolution, interruptible wait, avoiding per-square timer rounding.
The duty is a short-window approximation, not a limit on memory use.

`--reference-mode` and `--batch-bits 0` are useful for comparing optimized and
reference residues. Disabling CUDA Graphs is also supported for diagnosis.
`--diagnostic-force-replay` is for exercising the fallback, not routine runs.
Advanced environment overrides include `GFPS_BATCH_BITS`,
`GFPS_CARRY_PASSES`, `GFPS_FORCE_REPLAY`, and the presence flags
`GFPS_DISABLE_NEGACYCLIC`, `GFPS_DISABLE_DIF`, `GFPS_DISABLE_SHARED`,
and `GFPS_DISABLE_FUSED_TILE`; use the printed help for accepted values.

## Main PRP commands

For short reference tests (`n <= 12`):

```bash
./GFPS --prp-small 10 1
./GFPS --prp-small-gpu-norm 10 1
./GFPS --prp-small-resident 10 1
```

These modes compare different normalization/residency paths. The output includes `cuda_matches_ref` and `result_is_one`.

For a production-style resident run with checkpoints:

```bash
./GFPS --prp-check \
  <b> <n> \
  <target-bits-or-0-for-full> \
  <checkpoint-file> \
  <checkpoint-every-bits-or-0-for-end-only> \
  <progress-every-bits-or-0-for-auto> \
  <resume:0|1>
```

Example fresh full run:

```bash
./GFPS --prp-check 1814570323003372 16 0 work.ckpt 10000 0 0
```

Resume the same run:

```bash
./GFPS --prp-check 1814570323003372 16 0 work.ckpt 10000 0 1
```

The anchor-enabled variant is:

```bash
./GFPS --prp-check-a \
  <b> <n> \
  <target-bits-or-0-for-full> \
  <checkpoint-file> \
  <checkpoint-every-bits-or-0-for-end-only> \
  <anchor-prefix> \
  <anchor-every-bits> \
  <progress-every-bits-or-0-for-auto> \
  <resume:0|1>
```

Anchors can later be checked with:

```bash
./GFPS --verify-transition <start-checkpoint> <end-checkpoint>
./GFPS --verify-chain <anchor-prefix> <start-bits> <end-bits> <step-bits>
```

Other checkpoint utilities:

```bash
./GFPS --prp-check-ckpt <b> <n> <max-bits-or-0> <checkpoint-file> <save-every-bits-or-0>
./GFPS --prp-check-resume <checkpoint-file> <more-bits-or-0-to-end> <save-every-bits-or-0>
./GFPS --ckpt-info <checkpoint-file>
./GFPS --ckpt-cmp <left-checkpoint> <right-checkpoint>
```

Do not resume a checkpoint with different `b` or `n`. Checkpoint version 2 validates metadata, digit count/ranges, file length, and SHA-256 before allocating the resident state.

Checkpoint replacement is atomic on supported local filesystems. Windows builds durably flush the temporary file and use `MoveFileExW` with replacement semantics, so periodic saves can safely overwrite an existing checkpoint.

`Ctrl+C` on POSIX, or `Ctrl+C`/`CTRL_BREAK` on Windows, is handled after the
current resident operation reaches a consistent boundary, finishing and
validating any pending carry batch. GFPS then writes the
latest main checkpoint and exits with status 130. When the checker is used
through the toolkit's Python clients, each child is placed in its own process
group/session and one interrupt is forwarded from outer client to wrapper to
checker. Additional interrupts are ignored while shutdown and checkpoint
cleanup are in progress, so they cannot abort that cleanup path. Standard
output is flushed through pipes, including on Windows, so progress and
checkpoint messages remain visible while a run is active.

## Diagnostics and benchmarks

```text
--analyze <b> <n>
--square <b> <n>
--sparse <b> <n> <seed> <terms>
--gpu-norm <b> <n> <seed> <terms> <dup:0|1>
--sqrchk <b> <n> <seed> <terms> <dup:0|1>
--resident-mulcheck <b> <n>
--bench-resident <b> <n> <bits> <checkpoint-every-bits-or-0> <checkpoint-file> [carry-block-size]
```

Use `--analyze` before a large run to estimate transform and CRT requirements.
Its floating-point estimate can round incorrectly immediately next to a CRT
boundary; the exact integer check in production modes is authoritative.
`--resident-mulcheck` and the `--prp-small*` modes are capped at `n <= 12`
because they use expensive reference calculations.

## Known limits

- `b >= 2`. For the usual nontrivial generalized Fermat search, use an even base; an odd `b > 1` makes `N` even.
- Resident production modes support `1 <= n <= 20` and require `b < 2^63`.
- At most four configured NTT primes are available. GFPS rejects a `b,n` pair when their conservative coefficient bound exceeds the available CRT range.
- The supported NTT length is bounded by the configured primes and implementation (`2^21` in the current code).
- Large candidates require substantial GPU memory and long uninterrupted computation. Keep checkpoints on reliable storage.
- The arithmetic includes special handling for the unique missing centered residue class of even bases. Do not replace the centered tie rule without preserving that representation and its tests.
- GPU hardware faults are still possible. Use anchors, independent recomputation, and deterministic proof software for valuable results.

### Base limits by n

The `b < 2^63` requirement is an integer-representation limit, **not** the
actual maximum usable base. The CRT coefficient bound is tighter and depends
on `n`. For the current four NTT primes, their product is

```text
P4 = 998244353 * 1004535809 * 469762049 * 1224736769
   = 576929796637471070305089837581991937
```

Production modes use exact multiprecision integer arithmetic to require

```text
2 * 2^n * ceil(b / 2)^2 < P4
```

For an **even** base, the largest value satisfying this strict inequality is

```text
b_max(n) = 2 * isqrt((P4 - 1) // 2^(n + 1))
```

Here `//` is integer division and `isqrt` is the floor of the exact integer
square root. The table lists this coefficient-bound ceiling, not a performance,
memory-capacity, or hardware-validation guarantee. GFPS automatically selects
three primes when sufficient, otherwise four; the table uses all four.

| n | Exponent 2^n | Maximum even b under the four-prime CRT bound |
| ---: | ---: | ---: |
| 10 | 1,024 | 33,568,080,211,080,892 |
| 11 | 2,048 | 23,736,217,148,669,252 |
| 12 | 4,096 | 16,784,040,105,540,446 |
| 13 | 8,192 | 11,868,108,574,334,626 |
| 14 | 16,384 | 8,392,020,052,770,222 |
| 15 | 32,768 | 5,934,054,287,167,312 |
| 16 | 65,536 | 4,196,010,026,385,110 |
| 17 | 131,072 | 2,967,027,143,583,656 |
| 18 | 262,144 | 2,098,005,013,192,554 |
| 19 | 524,288 | 1,483,513,571,791,828 |
| 20 | 1,048,576 | 1,049,002,506,596,276 |
| >20 | — | Not supported |

**GFPS 4.1 currently accepts only `1 <= n <= 20`. Inputs with `n > 20` are not supported.**

For each supported row, the displayed even base passes the exact CRT inequality and
`b_max + 2` fails. For example, at `n=16`, `4196010026385110` is the ceiling;
`4196010026385112` is rejected with
`CRT dynamic range is insufficient for this base/n; refusing unsafe calculation`.
The production guard runs before GPU allocation and does not continue with an
out-of-range CRT reconstruction. Omit digit-grouping commas in CLI arguments.

> 中文说明：表中列出 n=10～20 的偶数 base 上限；**n>20 不支持**。
> 表中 n 是指数 2^n 的下标；n=16 对应 b^65536+1。
> `b<2^63` 不代表可计算到该值，实际还需满足表中的 CRT 上限。
> 超限会由正式计算路径精确检查并报错退出；`--analyze` 的浮点估计不能作为紧贴边界时的判定依据。

Run `./GFPS` without arguments to print the authoritative command summary for the exact binary being used.
