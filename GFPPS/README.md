# GFPPS 1.0

**Generalized Factorial / Primorial Prime Seeker** performs a Fermat
probable-prime check of one explicit `k*n!+1`, `k*n!-1`, `k*n#+1`, or `k*n#-1`.
GPU arithmetic uses exact integer NTT/CRT and Montgomery reduction; it does
not require FP64. A PRP result is **not a deterministic primality proof**.

`n#` is the product of all primes **not exceeding n**, not the first n primes:
`7# = 10# = 210`. The empty-product convention is `0! = 0# = 1`; the final
candidate must be at least 2. Omitting `k*` uses `k=1`. Both `k` and `n` accept
decimal or `0x` hexadecimal notation. Ranges and a literal `±` are not accepted.

## Quick start

Windows CMD / PowerShell:

```bat
.\GFPPS_sm_89.exe --check "13*4000!-1"
.\GFPPS_sm_89.exe --check "13*30000#+1"
```

Linux / WSL:

```bash
./GFPPS_sm_89 --check '13*4000!-1'
./GFPPS_sm_89 --check '13*30000#+1'
```

Always quote the expression. Use single quotes in interactive Bash to avoid
history expansion of `!`; disable delayed expansion in CMD before using `!`.
The examples illustrate syntax, not claims that these particular numbers are prime.

## Checkpoint and resume

Checkpoint saving is opt-in. **Supply `--checkpoint FILE` on the initial run**:

```bat
.\GFPPS_sm_89.exe --check "7777*2303867#+1" --checkpoint primorial.ckpt
```

Press Ctrl+C once and wait for `reason=interrupt`. The checker finishes the
current safe GPU boundary, saves the canonical arithmetic state, and exits
with status 130. Resume with the same expression and witness:

```bat
.\GFPPS_sm_89.exe --check "7777*2303867#+1" --checkpoint primorial.ckpt --resume-checkpoint
```

Without an explicit file, the program prints `checkpoint: disabled` and an
interrupted calculation cannot be resumed. The file is saved initially, every
100,000 new exponent bits by default, and at a completed/partial/interrupted
endpoint. An interruption need not wait until the periodic boundary.

Windows and Linux share the `GFPPS001` checkpoint format and can resume each
other's files. A fresh run refuses to overwrite an existing checkpoint;
explicitly resume or choose another filename. A completed state can be resumed
without repeating the exponent chain. Use one writer per checkpoint file.
An interruption during modulus construction or initial setup can occur before
the first arithmetic checkpoint exists.

For a bounded prefix, use `--max-bits N`; it limits **new bits in this
invocation**, and 0 means continue to the end:

```bash
./GFPPS_sm_89 --check '13*2300000#+1' --max-bits 1024 --checkpoint prefix.ckpt
./GFPPS_sm_89 --check '13*2300000#+1' --max-bits 16384 --checkpoint prefix.ckpt --resume-checkpoint
```

## Options

| Option | Meaning |
| --- | --- |
| `--witness 2..255` | Fermat witness; default 2. |
| `--force-ntt-blocks 1..4096` | NTT block cap; default 96, no autotuner. |
| `--no-graphs` | Disable CUDA Graphs for reference/diagnostic runs. |
| `--progress-every-bits N` | Progress interval; default 100000, positive. Endpoint progress is also printed. |
| `--checkpoint FILE` | Enable arithmetic checkpoints at this path. |
| `--resume-checkpoint` | Resume the specified checkpoint. |
| `--checkpoint-every-bits N` | Save interval; default 100000, 0 disables periodic saves only. Requires `--checkpoint FILE`. |
| `--max-bits N` | Maximum new exponent bits in this run; 0 means full remaining chain. |
| `--verify-cpp-int` | Explicit CPU repetition of the computed prefix; limited to modulus size ≤8192 bits. |
| `--print-residue` | Print the complete residue; limited to ≤8192 bits. |

Run `--help` or without arguments for the banner and command summary. Version
1.0 has no task queue, PRPNet adapter, duty-cycle throttle, or proof generation.

## Results and reliability

- `PRP`: the selected witness satisfies the Fermat congruence; independently
  recheck any hit. `170*2!+1=341` and `184*3#+1=1105` are composite base-2
  pseudoprime examples that intentionally return PRP.
- `COMPOSITE`: a trivial proper factor/evenness test or a complete Fermat
  residue different from 1, assuming correct arithmetic execution.
- `PARTIAL` / `INTERRUPTED`: no complete PRP classification; never discard a
  candidate based on these statuses.

The special candidate 2 is handled separately. A witness divisible by the
candidate is rejected rather than used to falsely reject a small prime.
Checkpoint SHA-256 covers metadata, modulus hash, progress, and all digits.
Reads validate length, parameters, and canonical residue; writes flush a
temporary file and atomically replace the destination. SHA-256 detects file
corruption; it is neither a keyed authentication mechanism nor an arithmetic
certificate. REDC cancellation and carry/range postconditions are checked,
but there is no Gerbicz error check. These checks do not detect every possible
hardware fault. Preserve evidence and use an independent implementation for
valuable results.

## Method and limits

The CPU builds the modulus with a product tree and precomputes Montgomery
constants and their NTT spectra. The GPU uses radix `B=2^15` and two NTT primes.
There are currently three large products per exponent bit; multiplication by
the small witness is fused into squaring. Choosing `R=B^m >= witness*N`
ensures `T < N*R` for canonical input, so REDC returns less than `2N` and one
conditional subtraction suffices. Carry preprocessing uses separate input
and output arrays followed by a full prefix scan. All kernels, CUB operations,
and Graph work use the same explicit stream.

The implementation checks CRT and 64-bit carry bounds. Limits include
`n <= 10,000,000`, NTT length at most `2^21`, and digit count at most `2^20`.
Large factorials can hit the arithmetic-size limits first. These limits are
not performance or GPU-memory guarantees. Display-only floating-point
estimates do not determine the modular result.

## Validation and performance

The underlying arithmetic implementation completed 112 regression cases per
platform on Windows/Linux, including both families/signs, 64-bit multipliers,
multiple witnesses, pseudoprimes, Graph on/off, corrupt-state rejection,
Unicode paths, real signal interruption, and cross-system resume. A subsequent
seven-case incremental suite per platform checked the 100,000-bit defaults
and compatibility. CUDA memcheck reported no errors or device-memory leaks
on the tested paths. These are finite tests, not a proof of implementation
correctness for every input.

The 29 known primorial samples completed on both systems with residue 1 and
byte-identical checkpoint pairs; 29 independent GMP complete repetitions also
matched. See [sample results](KNOWN_PRIMORIAL_RESULTS.md). The largest sample
has 45,259 decimal digits and took 16.025 s on Linux / 16.368 s on Windows for
exponentiation on an RTX 4060 Laptop GPU. Only `sm_89` has runtime evidence;
other release architectures are cross-compiled, not runtime-tested.

For roughly million-digit candidates, a 16,384-bit prefix took 8.685 s for
`13*210000!-1` and 8.578 s for `13*2300000#+1`. The resulting **29–30 minute
estimates are prefix extrapolations, not completed million-digit checks**.
Independent GMP checked the initial 1024-bit residues. These numbers are not
a same-condition comparison with PRST, PrimeGrid, or another implementation.

## Build

Requirements: CUDA 13.3 (or a compatible toolkit for the requested target),
a supported C++17 host compiler, Boost.Multiprecision headers, and CUDA's CUB.
Keep all three source files together in `src/`: `GFPPS.cu`, `ntt_backend.cuh`,
and `sha256.hpp`.

From the repository root:

```bash
./GFPPS/scripts/build-linux.sh build/linux sm_89
```

```bat
set BOOST_ROOT=C:\Libraries\boost
GFPPS\scripts\build-windows.bat build\windows sm_89
```

Windows requires an initialized CUDA-compatible MSVC environment. Put Boost
on the same drive as the checkout; the script converts its include directory
to a relative path and compiles from `src/` with a bare source filename.
Linux uses installed Boost headers by default; optional `BOOST_ROOT` also
becomes a relative include path. Omit the architecture list to build
`sm_86`, `sm_89`, `sm_100`, and `sm_120`. Runtime-only use needs neither nvcc nor
the Boost source headers.

> 中文提示：支持 `k*n!±1` 和 `k*n#±1`；`n#` 是所有不超过 n 的素数的乘积。
> 断点必须显式指定 `--checkpoint 文件`，再次启动加 `--resume-checkpoint`。
> 默认每 100000 bits 显示进度；PRP 不是确定性素数证明。百万位约半小时
> 目前仅为前缀估算，未完成全程验证。
