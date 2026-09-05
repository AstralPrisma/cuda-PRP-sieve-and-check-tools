# GFNSV CUDA 1.0

GFNSV is an integer-only GPU sieve for generalized Fermat candidates

```text
N = b^(2^n) + 1
```

The argument `n` is the power-of-two index, as in GFPS. With `--n 16`, the
exponent is 65536. GFNSV tests possible prime factors congruent to
`1 mod 2^(n+1)` and marks their roots across a contiguous interval of even
bases. A surviving base is a candidate for a later PRP check, not a prime.

中文：这是 GFPS 的 GPU 筛子。`Ctrl+C` 会在当前 GPU 批次完成后保存，
再次使用 `--resume` 可以从保存位置继续向更大的因子筛。版本仍为 1.0；
旧 CPU GFNSV 没有包含在本仓库中。

## Build and run

Build from this directory on Linux/WSL with a CUDA toolkit and C++17 compiler:

```bash
./scripts/build-linux.sh build sm_89
```

On Windows, use a CUDA-supported x64 MSVC toolchain and set `BOOST_ROOT` to the
directory containing `boost/` before running:

```bat
scripts\build-windows.bat build sm_89
```

Boost is used only by the Windows host-side 128-bit arithmetic. The CUDA
kernels use integer arithmetic on both platforms. There is no primesieve
runtime dependency. Keep `src/GFNSV.cu` and `src/gfnsv_state.hpp` together when
building from source.

Omit the architecture to build `sm_86`, `sm_89`, `sm_100`, and `sm_120`.
Select the binary for the target GPU; only `sm_89` has been runtime-tested on
the development RTX 4060. Examples below use `./GFNSV_sm_89`; on Windows use
`GFNSV_sm_89.exe` with the same arguments.

## A fresh sieve

```bash
./GFNSV_sm_89 --n 16 \
  --bmin 1814570322693370 --bmax 1814570323383368 \
  --pmax 1e12 --out candidates.txt --factors factors.txt
```

Both base bounds and factor bounds are inclusive. The default lower factor
bound is 3. Only even `b >= 2` is included. Decimal, hexadecimal, underscores,
and integer scientific notation such as `1e12` are accepted.

The default output format has one surviving base per non-comment line.
`--format expr` writes `b^65536+1`-style expressions instead. Lines beginning
with `#` carry the resume metadata, removed-candidate factors, and checksum.
Keep the complete file for resuming; extracting the non-comment lines creates
a candidate list but discards its resume capability.

CPU verification checks each recorded factor's divisibility and independently
tests the primality of distinct factors by default. `--no-verify` disables this
check and is intended for controlled diagnosis. A trial prime equal to the
candidate itself is not treated as a proper factor.

## Interrupt, save, and resume

Press `Ctrl+C` and wait for `checkpoint: saved ... reason=interrupt` and the
exit message. The checker finishes the already-started GPU batch, writes a
consistent checkpoint to `--out`, and exits with status 130. It advances the
saved frontier only after every root of that batch has been processed.

Continue the same search:

```bash
./GFNSV_sm_89 --resume --out candidates.txt --factors factors.txt
```

The shorter form also works:

```bash
./GFNSV_sm_89 --resume candidates.txt
```

The saved file supplies `n`, base bounds, factor bounds, format, survivors,
known factors, and the first unprocessed factor position. Explicit `--n`,
base bounds, `--pmin`, or format must agree with the saved metadata; they
cannot silently change the candidate family or skip part of the saved work.

To extend a finished or interrupted sieve to a larger factor bound:

```bash
./GFNSV_sm_89 --resume candidates.txt --pmax 2e12 --factors factors.txt
```

Changing `--pmax` cannot move the requested end below the already processed
factor interval. Batch size, GPU device, and root enumeration mode may be
changed between runs. Windows and Linux use the same state format.

Useful state controls:

```text
--state-every S          Save periodically after completed batches; default 30 seconds.
                         0 saves on handled interruption and completion only.
--checkpoint-info FILE  Validate and inspect a saved file without using a GPU.
```

An initial recoverable snapshot is also written before GPU work. A process or
power failure can require repeating work after the last saved batch; ordinary
`Ctrl+C` saves the current completed batch. State replacement uses a flushed
temporary file and platform replacement operations, preserving the preceding
state if the new write fails.

State format version 3 is independent of the program version 1.0. SHA-256
covers metadata and records; loading checks the header/footer, bounds, record
count and order, and completed frontier. Truncated, changed, or mismatched
files are rejected. An arbitrary sparse candidate file, an old CPU state, or
the earlier CUDA plain output is not accepted as a resume checkpoint.

`--factors` is an optional export of factors stored in the checkpoint. It can
be regenerated on resume; it is not required to continue. Existing factor
exports are replaced only when their metadata belongs to the same search.
Fresh searches require unused output paths.

## GPU controls and comparisons

```text
--device N       CUDA device index; default 0.
--batch N        Arithmetic-progression candidates per batch; default 65536,
                  range 1..1048576. These are possible factors, not base count.
--root-pairs     Mark r and -r together; default, faster.
--full-roots     Enumerate every root independently for comparisons.
--quiet          Suppress batch progress; keep checkpoint and final messages.
```

The two root modes must produce the same surviving bases. Factor files may
record different valid first factors because GPU workers run concurrently.
Compare candidate lines and independently validate factors rather than
requiring identical factor discovery order. A smaller batch reduces the time
between interrupt checks and checkpoint opportunities, at some launch overhead.

## Bounds and validation

- `n=1..20`, `3 <= pmin <= pmax <= 2^62-1`.
- A fresh run accepts a contiguous base interval and at most 64 million even
  candidate slots. Resuming restores that interval; arbitrary sparse input
  lists are not supported.
- Available device memory must accommodate the factor vector and batch data.
- Standard output flushes through pipes, including Windows, so progress can
  be displayed by an outer client while the sieve is running.

The initial arithmetic validation compared 14 cases in paired/full-root modes
on both Windows and Linux (56 comparisons), with the archived CPU sieve and
additional independent integer cases. It included small prime preservation,
64-bit base boundaries, factors through 61 bits, and 345,000 GFN16 candidates
through `10^12`. See [validation notes](../docs/validation-2026-09.md).

Run `./GFNSV_sm_89 --help` for the authoritative options of the binary in use.
