# CUDA PRP: sieve and check tools

This repository collects six CUDA programs for experimental large-integer
searches. Source code is tracked in Git. Prebuilt Linux and Windows executables
are published as GitHub Release assets rather than committed to the repository.

The [v2026.09.7 release](docs/releases/v2026.09.7.md) updates GFNSV to 1.1 with
self-contained single-file recovery, optional factor logs, efficiency stopping,
and offline candidate-task conversion. The other five tools retain their
component versions and byte-identical v2026.09.6 executables.

> [!IMPORTANT]
> A probable-prime (PRP) result is not a deterministic primality proof. Treat a
> PRP hit as a candidate that requires independent verification. A sieve output
> is only a list of survivors; it does not assert that those survivors are
> prime.

## Included tools

| Directory | Version | Purpose |
| --- | ---: | --- |
| [`GFPS/`](GFPS/) | 4.4 | Checks generalized Fermat candidates `b^(2^n)+1` with CUDA NTT arithmetic. |
| [`GSRPS/`](GSRPS/) | 2.3 | Checks generalized Sierpinski/Riesel candidates `k*b^n+1` and `k*b^n-1`. |
| [`GFPPS/`](GFPPS/) | 1.0 | Checks generalized factorial/primorial candidates `k*n!+/-1` and `k*n#+/-1`. |
| [`GFNSV/`](GFNSV/) | 1.1 | GPU-sieves even bases for `b^(2^n)+1`, with single-file continuation and offline task conversion. |
| [`GSRSV/`](GSRSV/) | 2.0 | Sieves `k*b^n+/-1`, `k*n#+/-1`, and `k*n!+/-1` candidate families. |
| [`GNCWSV/`](GNCWSV/) | 1.0 | Sieves generalized Cullen/Woodall and near-Cullen/near-Woodall families. |

## 中文简介

本仓库统一发布六个 CUDA 大整数搜索工具：GFPS 用于广义费马数 PRP
检查，GFNSV 用于广义费马数 GPU 筛选，GSRPS 用于广义 Sierpinski/Riesel
数 PRP 检查，GFPPS 用于广义阶乘/素数阶乘数 PRP 检查，GSRSV 用于固定
`b,n` 的 `k*b^n±1` 等数型筛选，GNCWSV 用于广义 Cullen/Woodall 与
Near Cullen/Woodall 数型筛选。源码位于六个同名目录；预编译程序只在
Releases 中发布。`PRP` 不是确定性素数证明，命中后仍须由独立程序复核。

这里的 GFPS 指本仓库中的 generalized-Fermat CUDA checker；其他项目或
历史素数记录中可能存在同缩写但不同的程序。

The checkers use exact integer NTT/CRT arithmetic within their accepted input
ranges. They reject parameters when the available CRT range is insufficient.
This arithmetic guarantee does not change the mathematical meaning of a PRP
test. See [Correctness and result semantics](docs/correctness.md).

For GFPS, consult the [base-limit table](GFPS/README.md#base-limits-by-n)
before selecting a search range. Its actual base ceiling depends on `n`, not
just the `b < 2^63` storage limit. **Current production modes support only
`1 <= n <= 20`; inputs with `n > 20` are not supported.**

GFPS 4.4 retains half-length negacyclic NTT and checked carry batches, and
fuses carry rotation, convergence checking, and RNS export for 100%-duty
batches. GSRPS 2.3 uses condition-checked weighted scans and compact carry
reduction, retaining the exact fallback where required. GFNSV 1.1 provides GPU
sieving with portable single-file recovery, progress/ETA, and optional
efficiency limits; the older CPU GFNSV is not bundled.
GFPPS 1.0 adds general integer Montgomery arithmetic and portable checkpoints
for factorial and primorial PRP checks. Its `n#` means the product of primes
not exceeding `n`; arithmetic checkpoints require an explicit `--checkpoint FILE`.
The component READMEs describe the controls and resume interfaces. Recorded
timings and the limits of validation are in
[September 2026 validation](docs/validation-2026-09.md).

## Prebuilt targets

Release packages are separated by operating system. Each package contains the
native CUDA architectures shown below; select the executable matching the
target GPU.

| CUDA target | GFPS | GSRPS | GFPPS | GFNSV | GSRSV | GNCWSV |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sm_86` | yes | yes | yes | yes | yes | yes |
| `sm_89` | yes | yes | yes | yes | yes | yes |
| `sm_100` | yes | yes | yes | yes | yes | yes |
| `sm_120` | yes | yes | yes | yes | yes | yes |

Both Linux x86-64 ELF and Windows x64 PE executables are provided. Each file
contains native cubins for the SM target in its filename. CUDA may also embed
PTX for that same target; these packages do not use one universal executable
for all GPU generations. Architectures that have only been cross-compiled are
marked as such in the Release notes; only `sm_89` was runtime-tested for the
initial public release.

The table describes the build targets for this source tree. Check the version
and validation notes of a downloaded Release: published binaries can be older
than the current source.

After downloading an archive, verify its checksum and, on Linux, make the
executable runnable if necessary:

```bash
sha256sum -c SHA256SUMS
chmod 755 GFPS_sm_89
```

Run a program without a command, or use `-h` for a siever, to display its full
command reference:

```bash
./GFPS_sm_89
./GSRPS_sm_89
./GFPPS_sm_89 --help
./GFNSV_sm_89 --help
./GSRSV_sm_89 -h
./GNCWSV_sm_89 -h
```

Example checker invocations:

```bash
./GFPS_sm_89 --prp-check-pref 1814570322684094 16 1000
./GSRPS_sm_89 --check '328*799^325799-1'
./GFPPS_sm_89 --check '13*4000!-1' --checkpoint factorial.ckpt
```

The examples illustrate syntax only. They are not claims about the primality of
the displayed numbers.

## Building from source

The programs require a CUDA toolkit supporting the selected architecture and a
C++17 host compiler. GFPS, GSRPS, and GFPPS use Boost.Multiprecision headers; the
Windows GFNSV build uses Boost for host-side 128-bit arithmetic. GSRPS and GFPPS use CUB
through the CUDA toolkit.

Every tool includes Linux/WSL and Windows build scripts. From the repository
root, for example:

```bash
./GFPS/scripts/build-linux.sh build/linux sm_89
./GSRPS/scripts/build-linux.sh build/linux sm_89
./GFPPS/scripts/build-linux.sh build/linux sm_89
./GFNSV/scripts/build-linux.sh build/linux sm_89
./GSRSV/scripts/build-linux.sh build/linux sm_89
./GNCWSV/scripts/build-linux.sh build/linux sm_89
```

```bat
GFPS\scripts\build-windows.bat build\windows sm_89
GSRPS\scripts\build-windows.bat build\windows sm_89
GFPPS\scripts\build-windows.bat build\windows sm_89
GFNSV\scripts\build-windows.bat build\windows sm_89
GSRSV\scripts\build-windows.bat build\windows sm_89
GNCWSV\scripts\build-windows.bat build\windows sm_89
```

Omit the architecture arguments to build all four release targets. GFPS,
GSRPS, GFPPS, and Windows GFNSV require Boost headers; set `BOOST_ROOT` to the directory that
contains the `boost` folder. The `.bat` launchers call the checked PowerShell
build scripts and use CUDA's MSVC host compiler integration.

See [Building on Linux and Windows](docs/building.md) for prerequisites,
argument conventions, and architecture notes.

To build the full matrix in one command:

```bash
./scripts/build-all-linux.sh release-assets sm_86 sm_89 sm_100 sm_120
```

```bat
scripts\build-all-windows.bat release-assets sm_86 sm_89 sm_100 sm_120
```

GSRSV and GNCWSV can load `primesieve` dynamically when it is installed. They
retain a built-in prime generator for systems where that optional library is
unavailable.

Builds intended for publication must use the pinned process in
[Releasing](docs/releasing.md). In particular, the build environment, CUDA
version, native architecture, source commit, runtime test status, and generated
SHA-256 digest must be recorded.

## Verification before long runs

At minimum:

```bash
./GFPS_sm_89 --selftest
./GSRPS_sm_89 --selftest
./GFPPS_sm_89 --check '3*5!+1' --verify-cpp-int
./GFNSV_sm_89 --help
./GSRSV_sm_89 --version
./GNCWSV_sm_89 --version
```

For expensive searches, preserve checkpoints and logs, independently recheck
PRP hits, and compare known prefix/final residues after changing CUDA code,
compiler versions, or GPU architecture. The project-specific validation matrix
is described in [docs/correctness.md](docs/correctness.md).

## License and attribution

The repository is distributed under the GNU General Public License, version 2
or later (`GPL-2.0-or-later`). See [`LICENSE`](LICENSE). GSRSV and GNCWSV contain
work derived from the GPLv2-or-later mtsieve/twinsieve implementation. Other
libraries and toolchain components retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
