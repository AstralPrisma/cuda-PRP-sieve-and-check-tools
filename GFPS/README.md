# GFPS

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
```

Backward-compatible environment variables are also accepted:

```text
GFPS_NTT_BLOCKS=1..4096
GFPS_DUTY_PERCENT=1..100
GFPS_DISABLE_CUDA_GRAPHS=1
```

`--duty-percent` inserts rest time between GPU iterations. It reduces sustained GPU use but increases wall-clock time. Disabling CUDA Graphs is mainly useful for diagnosis and regression testing.

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

Use `--analyze` before a large run to inspect transform and CRT requirements. `--resident-mulcheck` and the `--prp-small*` modes are capped at `n <= 12` because they use expensive reference calculations.

## Known limits

- `b >= 2`. For the usual nontrivial generalized Fermat search, use an even base; an odd `b > 1` makes `N` even.
- Resident production modes support `1 <= n <= 20` and require `b < 2^63`.
- At most four configured NTT primes are available. GFPS rejects a `b,n` pair when their conservative coefficient bound exceeds the available CRT range.
- The supported NTT length is bounded by the configured primes and implementation (`2^21` in the current code).
- Large candidates require substantial GPU memory and long uninterrupted computation. Keep checkpoints on reliable storage.
- The arithmetic includes special handling for the unique missing centered residue class of even bases. Do not replace the centered tie rule without preserving that representation and its tests.
- GPU hardware faults are still possible. Use anchors, independent recomputation, and deterministic proof software for valuable results.

Run `./GFPS` without arguments to print the authoritative command summary for the exact binary being used.
