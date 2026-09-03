# GNCWSV

GNCWSV is a CUDA sieve for Generalized Cullen, Generalized Woodall, and four
near-Cullen/Woodall families with a fixed base `b` and variable coefficient `a`.

| Mode | Candidate | Name |
|---:|---|---|
| 1 | `a*b^a - 1` | Generalized Woodall |
| 2 | `a*b^a + 1` | Generalized Cullen |
| 3 | `a*b^(a+1) - 1` | Near Woodall, first kind; `(n-1)*b^n-1` |
| 4 | `a*b^(a-1) - 1` | Near Woodall, second kind; `(n+1)*b^n-1` |
| 5 | `a*b^(a+1) + 1` | Near Cullen, first kind; `(n-1)*b^n+1` |
| 6 | `a*b^(a-1) + 1` | Near Cullen, second kind; `(n+1)*b^n+1` |

The current source version is 1.0.

> 中文：GNCWSV 用 GPU 筛选固定底数、变化系数的广义 Cullen/Woodall 及四类 Near Cullen/Woodall 数。`--mode 1..6` 选择数型。

## Requirements

- A CUDA-capable NVIDIA GPU.
- NVIDIA CUDA Toolkit with `nvcc` and a C++17 host compiler.
- A 64-bit operating system is recommended for large ranges.
- Optional: the primesieve shared library for faster prime generation.

GNCWSV dynamically loads primesieve and does not require its headers or link
flags at build time. `--prime-generator auto` uses it when available and falls
back to the built-in generator otherwise. `--prime-generator primesieve` makes
absence of the shared library an error.

Debian/Ubuntu/WSL:

```bash
sudo apt install libprimesieve-dev
```

On Windows, place a compatible `primesieve.dll` beside the executable or on
`PATH`.

## Building

Choose the CUDA architecture for the target GPU, for example `sm_86` for many
RTX 30-series cards or `sm_89` for many RTX 40-series cards.

Linux/WSL:

```bash
cd GNCWSV
./scripts/build-linux.sh build sm_89
```

Windows with the Visual C++ host compiler:

```bat
cd GNCWSV
scripts\build-windows.bat build sm_89
```

Omit the architecture to build all four release targets, or list a subset after
the output directory.

```bash
./GNCWSV --version
./GNCWSV --help
```

## Starting a sieve

Without `--inputterms`, the lower and upper `a` bounds, fixed base, and mode are
required.

Sieve Generalized Woodall candidates `a*7^a-1`:

```bash
./GNCWSV \
  --amin 20000 --amax 100000 \
  --base 7 --mode 1 \
  --pmax 1000000000000 \
  --outputterms woodall_b7_pass1.txt \
  --outputfactors woodall_b7_factors.txt \
  --verify
```

Sieve first-kind Near Cullen candidates `a*3^(a+1)+1`:

```bash
./GNCWSV \
  -a 20000 -A 100000 -b 3 -m 5 \
  -P 1000000000000 \
  -o near_cullen_b3.txt -O near_cullen_b3_factors.txt
```

Prime bounds use the interval:

```text
pmin < p <= pmax
```

For odd `b`, odd `a` makes the resulting candidate trivially even. GNCWSV
therefore normalizes the lattice to even `a` values and discards trivially even
entries when loading a file. Modes 4 and 6 require `a >= 2` because their
exponent is `a-1`.

## Candidate and factor files

Candidate input and output use one complete expression per line:

```text
879536*3^879537-1
864194*3^864195+1
```

Spaces around operators are accepted. Each file must contain one fixed base and
one mode. Exponents must be exactly `a`, `a+1`, or `a-1`, with the sign matching
the selected family. Lines beginning with `#` or `//` are comments; inline `//`
comments are also accepted on expression lines.

The output survivor file begins with a resume header:

```text
# GNCWSV p=<largest-prime> base=<b> mode=<1..6> amin=<min-a> amax=<max-a>
```

It is followed by one full surviving expression per line. The survivor file is
rewritten. `--outputfactors` appends newly found factors as:

```text
p | expression
```

Factor storage is allocated only when `--outputfactors` is supplied. Add
`--verify` to verify every reported factor on the CPU.

Apply a separate factor file without sieving:

```bash
./GNCWSV \
  --inputterms woodall_b7_pass1.txt \
  --inputfactors additional_factors.txt \
  --applyandexit \
  --outputterms woodall_b7_cleaned.txt
```

## Resume

Resume from a survivor file and extend the upper prime bound:

```bash
./GNCWSV \
  --inputterms woodall_b7_pass1.txt \
  --pmax 10000000000000 \
  --outputterms woodall_b7_pass2.txt \
  --outputfactors woodall_b7_pass2_factors.txt \
  --verify
```

Base and mode are recovered from the expressions and checked against the resume
header. Unless explicitly restricted, the live `a` bounds shrink to the
expressions still present in the file. The lower prime bound is raised to the
header's `p` value. An explicitly supplied `--pmin` above the recorded value is
rejected because it would skip an unsieved prime interval.

Use a separate output filename when preserving the previous checkpoint is
important. When interrupting an active run, send a normal Ctrl+C and allow the
program to drain submitted CUDA work and write the survivor file. A forced kill,
power loss, or process crash cannot create a new resume file.

## Main options

### Candidate definition and files

| Option | Meaning |
|---|---|
| `-a, --amin A0` | Minimum `a`. |
| `-A, --amax A1` | Maximum `a`. |
| `-b, --base B` | Fixed base. |
| `-m, --mode 1..6` | Candidate family from the table above. |
| `-p, --pmin P0` | Exclusive lower prime bound. |
| `-P, --pmax P1` | Inclusive upper prime bound. |
| `-i, --inputterms FILE` | Load/resume an expression-per-line file. `--input` is an alias. |
| `-o, --outputterms FILE` | Write survivors. `--output` is an alias. |
| `-I, --inputfactors FILE` | Apply existing `p | expression` factors. |
| `-O, --outputfactors FILE` | Append newly found factors. |
| `--applyandexit` | Apply input factors, write survivors, and exit. |

If `--outputterms` is omitted, the default name is:

```text
gncw_b<base>_m<mode>_a<amin>_<amax>.txt
```

### CUDA and prime generation

| Option | Default | Meaning |
|---|---:|---|
| `--device D` | `0` | CUDA device index. |
| `--threads T` | `256` | Threads per block; must be a multiple of 32. |
| `--hot-gaps H` | `6` | Compact shared hot-gap specialization: `0,2,4,6,8,12,16`. |
| `--blocks B` | `0` | Grid blocks; auto uses 8/SM dense or 24/SM compact. |
| `-w, --batch-primes N` | `262144` | Primes per producer batch. |
| `--cpu-small-prime P` | `2` | CPU handles primes at or below `P`. |
| `--prime-generator auto|primesieve|segmented|mr` | `auto` | Prime generator. |
| `--prime-prefetch N` | `16` | Queued prime batches. |
| `--prime-threads N` | `0` | Prime-generator workers; `0` is automatic. |
| `--prime-region-batches N` | `12` | Batches per primesieve region. |
| `--cuda-streams N` | `1` | CUDA streams/buffers, from 1 to 4. |
| `--progress-seconds N` | `60` | Progress interval; displayed rate covers the last 60 seconds. |
| `--segment-mib M` | `8` | Built-in generator segment size per worker. |
| `--verify` | off | Verify reported factors on the CPU. |
| `--quiet` | off | Reduce progress output. |

GNCWSV intentionally has no efficiency/no-factor timeout system.

## License and provenance

The prime-generation and CUDA infrastructure is derived from the
GPL-2.0-or-later GSRSV/mtsieve-based implementation. The varying-exponent
recurrence follows the standard Generalized Cullen/Woodall sieve approach.

GNCWSV is distributed under the GNU General Public License, version 2 or later.
See [`../LICENSE`](../LICENSE). primesieve is an optional runtime dependency
distributed separately under its own BSD-2-Clause license.
