# GSRSV

GSRSV is a CUDA sieve for the paired or independent forms

```text
k*b^n - 1        k*b^n + 1
k*n#  - 1        k*n#  + 1
k*n!  - 1        k*n!  + 1
```

`n#` denotes the primorial and `n!` the factorial. In the default twin mode,
the `-1` and `+1` terms for a given `k` are treated as one pair. Use
`--independent` when the two signs must be retained and sieved separately.

The current source version is 2.0. The implementation is a single-translation-
unit CUDA port/reimplementation of the `twinsieve` application from mtsieve.

> 中文：GSRSV 用 GPU 筛选 `k*b^n±1`、`k*n#±1` 和 `k*n!±1`。默认按双子候选处理；加 `-s` 可分别筛选正负两侧。

## Requirements

- A CUDA-capable NVIDIA GPU.
- NVIDIA CUDA Toolkit with `nvcc` and a C++17 host compiler.
- A 64-bit operating system is recommended for large ranges.
- Optional: the primesieve shared library for faster prime generation.

GSRSV dynamically loads primesieve at runtime; primesieve headers and link flags
are not required to build GSRSV. With `--prime-generator auto`, GSRSV uses
primesieve when it is available and falls back to its built-in segmented/Miller-
Rabin generator otherwise. `--prime-generator primesieve` makes absence of the
library an error.

On Debian/Ubuntu/WSL, the optional library can normally be installed with:

```bash
sudo apt install libprimesieve-dev
```

On Windows, place a compatible `primesieve.dll` where the Windows loader can
find it, such as beside the executable or on `PATH`.

## Building

Choose the `sm_XX` architecture for the target GPU. For example, `sm_86` targets
many RTX 30-series cards and `sm_89` targets many RTX 40-series cards.

Linux/WSL:

```bash
cd GSRSV
./scripts/build-linux.sh build sm_89
```

Windows with the Visual C++ host compiler:

```bat
cd GSRSV
scripts\build-windows.bat build sm_89
```

Omit the architecture to build all four release targets, or list a subset after
the output directory.

Check the resulting program with:

```bash
./GSRSV --version
./GSRSV --help
```

## Starting a sieve

Without `--inputterms`, `--kmin`, `--kmax`, `--termtype`, and `--exp` are
required. `--base` is additionally required for `b^n` terms.

Sieve paired `k*1337^78647 +/- 1` candidates:

```bash
./GSRSV \
  --kmin 2 --kmax 100000 \
  --base 1337 --exp 78647 --termtype 1 \
  --pmax 1000000000000 \
  --format D \
  --outputterms b1337_pass1.pfgw \
  --outputfactors b1337_factors.txt \
  --verify
```

Sieve the two signs independently and write ABC output:

```bash
./GSRSV \
  -k 2 -K 100000 -b 1337 -n 78647 -t 1 \
  -s -f A -P 1000000000000 \
  -o b1337_independent.pfgw -O b1337_independent_factors.txt
```

Primorial and factorial runs use `--termtype 2` and `--termtype 3`
respectively. `--base` is not used for these two modes:

```bash
./GSRSV -k 1 -K 10000 -n 1000 -t 2 -P 1000000000 -o p1000.pfgw
./GSRSV -k 1 -K 10000 -n 1000 -t 3 -P 1000000000 -o f1000.pfgw
```

Prime bounds use the interval

```text
pmin < p <= pmax
```

## Candidate and factor files

`--format` accepts:

- `D`: ABCD output. This is the default and is available for twin mode.
- `A`: ABC output. Independent `+1`/`-1` output is automatically written as ABC.
- `N`: NewPGen output. This is available only for twin `k*b^n +/- 1` runs.

The output header records the largest completed prime as `Sieved to ...` (or in
the NewPGen header). Candidate data follows the selected format. The output file
is rewritten with the survivors.

`--outputfactors FILE` appends newly found factors in the form:

```text
p | term
```

Factor storage is allocated only when `--outputfactors` is supplied. Add
`--verify` to verify every factor on the CPU before it is written.

Existing factors can be applied without running the sieve:

```bash
./GSRSV \
  --inputterms b1337_pass1.pfgw \
  --inputfactors additional_factors.txt \
  --applyandexit \
  --outputterms b1337_cleaned.pfgw
```

## Resume

Resume by passing a previously written ABC, ABCD, or NewPGen survivor file to
`--inputterms` and choosing a new upper prime bound:

```bash
./GSRSV \
  --inputterms b1337_pass1.pfgw \
  --pmax 10000000000000 \
  --outputterms b1337_pass2.pfgw \
  --outputfactors b1337_pass2_factors.txt \
  --verify
```

The term type, base, exponent, sign mode, candidate range, and previous sieve
limit are recovered from the input file. The effective lower prime bound is
raised to the recorded sieve limit, so already completed prime intervals are not
repeated. Use a separate output filename when preserving the previous checkpoint
is important.

When interrupting an active run, send a normal Ctrl+C and allow GSRSV to drain
already submitted CUDA batches and write its final survivor file. A forced kill,
power loss, or process crash cannot create a new resume file.

## Main options

### Candidate definition

| Option | Meaning |
|---|---|
| `-k, --kmin K` | Minimum `k`. |
| `-K, --kmax K` | Maximum `k`. |
| `-b, --base B` | Base for `b^n`. |
| `-n, --exp N` | Exponent, primorial index, or factorial index. |
| `-t, --termtype 1|2|3` | `1=b^n`, `2=n#`, `3=n!`. |
| `-s, --independent` | Sieve `+1` and `-1` independently. |
| `-r, --remove` | Remove `k` divisible by the base in `b^n` mode. |
| `-p, --pmin P0` | Exclusive lower prime bound. |
| `-P, --pmax P1` | Inclusive upper prime bound. |

### Files

| Option | Meaning |
|---|---|
| `-f, --format A|D|N` | ABC, ABCD, or NewPGen. |
| `-i, --inputterms FILE` | Load/resume a survivor file. |
| `-o, --outputterms FILE` | Write remaining candidates. |
| `-I, --inputfactors FILE` | Apply existing `p | term` factors. |
| `-O, --outputfactors FILE` | Append newly found factors. |
| `-A, --applyandexit` | Apply `-I`, write survivors, and exit. |

### CUDA and prime generation

| Option | Default | Meaning |
|---|---:|---|
| `--device D` | `0` | CUDA device index. |
| `--threads T` | `256` | Threads per block; must be a multiple of 32. |
| `--blocks B` | `0` | Grid/resident blocks; `0` selects 8 per SM. |
| `-w, --batch-primes N` | `1048576` | Primes copied per launch. |
| `--cpu-small-prime P` | `100000` | CPU handles primes at or below `P`. |
| `--prime-generator auto|primesieve|segmented|mr` | `auto` | Prime generator. |
| `--prime-prefetch N` | `32` | Queued pinned prime-batch views. |
| `--prime-threads N` | `0` | Prime-generator workers; `0` is automatic. |
| `--prime-region-batches N` | `12` | Batches per primesieve initialization/region. |
| `--cuda-streams N` | `2` | Pinned CUDA submission slots, from 1 to 4. |
| `--progress-seconds N` | `60` | Full progress/ETA report interval. |
| `--segment-mib M` | `8` | Built-in generator segment size per worker. |
| `--verify` | off | Verify reported factors on the CPU. |
| `--quiet` | off | Reduce progress output. |

## Optional efficiency stop

The efficiency stop is enabled only when all three values are supplied:

```bash
./GSRSV ... \
  --max-factor-seconds 600 \
  --max-average-factor-seconds 30 \
  --efficiency-window-minutes 60
```

Short aliases are `-4`, `-5`, and `-6`. `--spftarget` aliases
`--max-average-factor-seconds`, and `--minutesforspf` aliases
`--efficiency-window-minutes`.

- The first limit stops after the requested number of seconds without a newly
  removed candidate.
- The second limit stops when rolling seconds per newly removed candidate exceed
  the threshold after a complete rolling window.
- Limits are evaluated after completed batches; submitted CUDA batches are
  drained first.
- Enabling this system replaces the user `pmax` with `2^62-1`; the efficiency
  conditions determine the stopping point.

## License and provenance

GSRSV is derived from the GPL-2.0-or-later mtsieve `twinsieve` code. The source
records the original CPU `TwinApp`/`TwinWorker` copyright as:

```text
Copyright (C) Mark Rodenkirch, 2018
```

This CUDA reimplementation is distributed under the GNU General Public License,
version 2 or later. See [`../LICENSE`](../LICENSE). primesieve is an optional
runtime dependency distributed separately under its own BSD-2-Clause license.
