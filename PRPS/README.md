# PRPS — Proth/Riesel Prime Seeker (CUDA)

Checks candidates of the fixed form **`k*2^n + 1`** (Proth) and
**`k*2^n - 1`** (Riesel) with a Fermat probable-prime test on the GPU, and can
emit a **prst-verifiable certificate** for every check.

PRPS is the base-2 specialisation of [GPRPS](../GPRPS/): the base is fixed at 2
and the exponentiation chain is reduced to `x0 = a^k` followed by `n`
squarings. That is what makes certificates cheap enough to produce on every
run, and it is why one binary covers the whole base-2 Proth/Riesel workload.

## One unified binary

Unlike the sieve tools, PRPS ships as a **single executable containing every
supported SM target** (`sm_86`, `sm_89`, `sm_100`, `sm_120`). The CUDA driver
picks the matching code at run time, so the same file runs on an RTX 30, 40 or
50 series card and on a data-centre part without a per-architecture download.

Both build scripts therefore produce exactly one artifact:

| Platform | Artifact | Runtime dependency |
| --- | --- | --- |
| Windows x64 | `PRPS.exe` | `libgmp-gw1.dll` next to it |
| Linux x86-64 | `PRPS` | system `libgmp` / `libgmpxx` |

No PTX is embedded, so support for a future SM architecture requires a rebuild
(or adding `-gencode arch=compute_120,code=compute_120` for driver-side JIT).

## Build

Additional requirement compared with GPRPS: **GMP**, because PRPS serialises
proof points and certificates with GMP.

| Variable | Meaning |
| --- | --- |
| `BOOST_ROOT` | directory containing `boost/multiprecision/cpp_int.hpp` |
| `GMP_ROOT` | Windows only: directory containing `gmp.h`, `win64/libgmp-gw1.lib` and `win64/libgmp-gw1.dll` |
| `NVCC` | optional, full path to `nvcc` |

```powershell
./scripts/build-windows.ps1                 # -> build\PRPS.exe + libgmp-gw1.dll
./scripts/build-windows.ps1 -OutputDirectory out
```

```bash
./scripts/build-linux.sh                    # -> build/PRPS
./scripts/build-linux.sh /tmp/out
```

On Ubuntu the multiarch `gmp.h` lives in `/usr/include/x86_64-linux-gnu/`,
which is already on the default include path, so only `libgmp-dev` is needed.

## Usage

```
PRPS --check <k*2^n+/-1> [Fermat-witness, default 3]
PRPS --check <k> 2 <n> <c:+1|-1> [Fermat-witness, default 3]
PRPS --checkpoint-info <checkpoint-file>
PRPS --selftest
PRPS --bench-square <k> <b> <n> <c:+1|-1> <iterations>
PRPS --bench-mul <k> <b> <n> <c:+1|-1> <iterations>
```

The base is fixed at 2, so the second form accepts `2` only.

Global options:

| Option | Meaning |
| --- | --- |
| `--force-ntt-blocks <1..4096>` | disable auto-tuning and force the NTT block cap |
| `--force-window-bits <1..8>` | disable automatic sliding-window selection |
| `--duty-percent <1..100>` | GPU duty cycle for a full check; default 100 |
| `--verify-cpp-int` | repeat `--check` independently with Boost cpp_int on the CPU |
| `--tuning-cache-dir <path>` | persistent NTT tuning cache; default `.gprps_tuning_cache` |
| `--tuning-cache-max-age-hours N` | refresh cached tuning after N hours; default 24 |
| `--no-tuning-cache` | disable the persistent tuning cache |
| `--checkpoint <path>` | save a portable SHA-256-protected check state |
| `--checkpoint-every-bits <N>` | periodic save interval; 0 = end/interrupt only |
| `--resume-checkpoint` | require and resume the checkpoint file |

### Progress

A live progress bar with elapsed time and ETA:

```
progress: [#######                     ] 25.39%, bits=50782/200000, elapsed_s=11.4, eta_s=33.6
```

### Checkpoint / resume

`--checkpoint` makes a run interruptible. The checkpoint carries the processed
square count and the current residue; a companion file `«checkpoint».points`
records how much of the proof-point dump is already on disk, so a resumed run
appends the remaining points instead of starting over.

* interrupt with Ctrl+C → the current state is saved before exiting
* kill the process → at most `--checkpoint-every-bits` of work is lost
  (50000 bits ≈ 30 s at n = 3.5M)
* resume with the same `--checkpoint` plus `--resume-checkpoint`
* a completed run deletes the checkpoint automatically, after the certificate
  has been written

A resumed run produces a bit-identical residue **and** a byte-identical
certificate compared with an uninterrupted run.

## Certificates

The certificate is written to the path in `PRPS_CERT`; the proof points are
dumped to `PRPS_POINTS` and the number of points is taken from `PRPS_COUNT`
(512 matches the PrimeCrunch/proof convention). Both are environment variables
so that the format is unchanged from GPRPS.

The resulting file is byte-identical to what `prst` produces for the same
candidate, and `prst -proof cert` verifies it. Certificate sizes are roughly
437 KB at n = 3.5M; checking one takes a few seconds of CPU (it expands the
points into X and performs M squarings, not the full chain).

## Relationship to GPRPS

`PRPS/src/PRPS.cu` is a **standalone translation unit** — the build does not
read, include or generate anything from `GPRPS/`. It was derived from GPRPS
once, by replacing the sliding-window exponentiation with the fixed
`a^k` + `n` squarings chain and adding the certificate writer; after that it
evolves on its own.

## Verified

Byte-identical certificates against `prst` at n = 1024, 4096, 4097, 100000,
1000000 and 3500000, for both `c = -1` and `c = +1`, and across `sm_89` and
`sm_120` hardware. Residues also match GPRPS's independent chain on the same
candidates. The Linux and Windows binaries produce identical residues and
identical certificate files.
