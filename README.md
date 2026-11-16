# CUDA PRP: sieve and check tools

This repository collects twelve CUDA programs for experimental large-integer
searches. Source code is tracked in Git. Prebuilt Linux and Windows executables
are published as GitHub Release assets rather than committed to the repository.

The [v2026.11.16 release](docs/releases/v2026.11.16.md) adds GPRSV_N 1.1, a CUDA
sieve for `k*b^n+/-1` with a fixed `k` and a variable exponent interval, and
renames GPRSV to GPRSV_K to make the direction explicit (fixed `n`, variable
`k`). GPRSV_K 2.3 itself is unchanged; its v2026.10.29 assets were renamed
accordingly.

The [v2026.10.29.2 release](docs/releases/v2026.10.29.2.md) updates GFPDIVSV to1.1
with adaptive batching, persistent prime iterators and bounded producer pipelines.
The [v2026.10.29.1 release](docs/releases/v2026.10.29.1.md) introduced GFPDIVSV1.0,
a CUDA sieve for factorial/primorial single-factor removal, with checked
continuation, dual signs and offline GFPPS task conversion.

The [v2026.10.29 release](docs/releases/v2026.10.29.md) adds CSPS1.0 and CSSV1.3,
adds exact single-factor-removal inputs to GFPPS1.1, and updates GPRSV_K 2.3 and
GHCWSV1.3. Updated CUDA components retain separate Windows/Linux executables
for sm_86, sm_89, sm_100 and sm_120. GFPS remains the generic variable-index
checker; no fixed-GFN16 specialization is included.

The [v2026.09.13 release](docs/releases/v2026.09.13.md) updates GFPS to4.5, GHCWPS to1.2 and GNCWSV to1.2.0. It adds general division simplification, safe adaptive PRP radix selection and multi-mode/sparse-workset sieve scheduling. Other components are unchanged. Performance and physical-GPU coverage are documented per workload, not claimed universally.

The [v2026.09.11 release](docs/releases/v2026.09.11.md) adds `--sign both` to GHCWSV 1.1, including sign-preserving checkpoints and conversion. Other seven tools are unchanged.

The [v2026.09.10 release](docs/releases/v2026.09.10.md) adds GHCWSV and GHCWPS for `b^n*n^b卤1`. Four native GPU targets per platform are provided. The other six tools remain unchanged; their packages are linked from v2026.09.9.

The [v2026.09.9 release](docs/releases/v2026.09.9.md) promotes GFPPS's parallel
NTT and exact fused-carry path while retaining component version 1.0. For the
complete 100,001-digit `2*25206!+1` check, controlled before/after measurements
showed 1.64x throughput on RTX 5090/Linux and 1.22x on RTX 4060 Laptop/Windows.
These are within-platform comparisons, not cross-platform speed claims.
The other five tools retain their versions and byte-identical v2026.09.8
executables; GFPS's separate experimental optimization is not promoted.

> [!IMPORTANT]
> A probable-prime (PRP) result is not a deterministic primality proof. Treat a
> PRP hit as a candidate that requires independent verification. A sieve output
> is only a list of survivors; it does not assert that those survivors are
> prime.

## Included tools

| Directory | Version | Purpose |
| --- | ---: | --- |
| [`GFPS/`](GFPS/) | 4.5 | Checks generalized Fermat candidates `b^(2^n)+1` with CUDA NTT arithmetic. |
| [`GPRPS/`](GPRPS/) | 2.4 | Checks generalized Proth/Riesel candidates `k*b^n+1` and `k*b^n-1`. |
| [`PRPS/`](PRPS/) | 1.0 | Checks base-2 Proth/Riesel candidates `k*2^n+1` and `k*2^n-1`, with checkpoint/resume and prst-verifiable certificates. Ships as one unified binary covering every supported SM target. |
| [`GFPPS/`](GFPPS/) | 1.1 | Checks factorial/primorial multiplication and exact single-factor-removal candidates. |
| [`GFPDIVSV/`](GFPDIVSV/) | 1.1 | Adaptive CUDA sieve for `n!/d卤1` and `n#/q卤1`, with checked recovery and offline task conversion. |
| [`CSPS/`](CSPS/) | 1.0 | Checks consecutive/alternate Catena Siderum power-product candidates. |
| [`CSSV/`](CSSV/) | 1.3 | CPU/CUDA sieve for Catena Siderum candidates, with independent signs and checked continuation. |
| [`GFNSV/`](GFNSV/) | 1.1 | GPU-sieves even bases for `b^(2^n)+1`, with single-file continuation and offline task conversion. |
| [`GPRSV_K/`](GPRSV_K/) | 2.3 | Sieves `k*b^n+/-1`, `k*n#+/-1`, and `k*n!+/-1` for a fixed `n` and a `k` interval. |
| [`GPRSV_N/`](GPRSV_N/) | 1.1 | Sieves `k*b^n+/-1` for a fixed `k` over an exponent interval (discrete-log BSGS). |
| [`GNCWSV/`](GNCWSV/) | 1.2.0 | Sieves generalized Cullen/Woodall and near-Cullen/near-Woodall families. |
| [`GHCWSV/`](GHCWSV/) | 1.3 | Sieves generalized hyper-Cullen/Woodall candidates `b^n*n^b卤1`. |
| [`GHCWPS/`](GHCWPS/) | 1.2 | Checks the same family with NTT arithmetic and safe checkpoints. |

## 涓枃绠€浠?
鏈粨搴撶粺涓€鍙戝竷鍗佷簩涓?CUDA 澶ф暣鏁版悳绱㈠伐鍏凤細GFPS 鐢ㄤ簬骞夸箟璐归┈鏁?PRP
妫€鏌ワ紝GFNSV 鐢ㄤ簬骞夸箟璐归┈鏁?GPU 绛涢€夛紝GPRPS 鐢ㄤ簬骞夸箟 Proth/Riesel
鏁?PRP 妫€鏌ワ紝GFPPS 鐢ㄤ簬骞夸箟闃朵箻/绱犳暟闃朵箻鏁?PRP 妫€鏌ワ紝GPRSV_K 鐢ㄤ簬鍥哄畾
`n`銆佺瓫涓€娈?`k` 鐨?`k*b^n卤1` 绛夋暟鍨嬬瓫閫夛紝GPRSV_N 鐢ㄤ簬鍥哄畾 `k`銆佺瓫涓€娈垫寚鏁?`n` 鐨勫悓绫荤瓫閫夛紝GNCWSV 鐢ㄤ簬骞夸箟 Cullen/Woodall 涓?Near Cullen/Woodall 鏁板瀷绛涢€夛紱CSPS/CSSV鐢ㄤ簬Catena Siderum妫€楠屽拰绛涘垎銆?GFPPS鏀寔涔樻硶鍜岀簿纭Щ闄ゅ崟涓洜瀛愮殑闄ゆ硶褰㈠紡锛汫FPDIVSV鐢ㄤ簬鍗曞洜瀛愮Щ闄ゅ舰寮忕殑CUDA绛涘垎銆?婧愮爜浣嶄簬鍗佷簩涓悓鍚嶇洰褰曪紱棰勭紪璇戠▼搴忓彧鍦?Releases 涓彂甯冦€俙PRP` 涓嶆槸纭畾鎬х礌鏁拌瘉鏄庯紝鍛戒腑鍚庝粛椤荤敱鐙珛绋嬪簭澶嶆牳銆?
杩欓噷鐨?GFPS 鎸囨湰浠撳簱涓殑 generalized-Fermat CUDA checker锛涘叾浠栭」鐩垨
鍘嗗彶绱犳暟璁板綍涓彲鑳藉瓨鍦ㄥ悓缂╁啓浣嗕笉鍚岀殑绋嬪簭銆?
The checkers use exact integer NTT/CRT arithmetic within their accepted input
ranges. They reject parameters when the available CRT range is insufficient.
This arithmetic guarantee does not change the mathematical meaning of a PRP
test. See [Correctness and result semantics](docs/correctness.md).

For GFPS, consult the [base-limit table](GFPS/README.md#base-limits-by-n)
before selecting a search range. Its actual base ceiling depends on `n`, not
just the `b < 2^63` storage limit. **Current production modes support only
`1 <= n <= 20`; inputs with `n > 20` are not supported.**

GFPS 4.5 retains half-length negacyclic NTT and checked carry batches, and
fuses carry rotation, convergence checking, and RNS export for 100%-duty
batches. GPRPS 2.4 uses condition-checked weighted scans and compact carry
reduction, retaining the exact fallback where required. GFNSV 1.1 provides GPU
sieving with portable single-file recovery, progress/ETA, and optional
efficiency limits; the older CPU GFNSV is not bundled.
GFPPS 1.0 provides general integer Montgomery arithmetic and portable checkpoints
for factorial and primorial PRP checks. Its optimized path splits the two NTT
prime planes and fuses exact carry preprocessing while retaining the complete
carry scan. Its `n#` means the product of primes not exceeding `n`; arithmetic
checkpoints still require an explicit `--checkpoint FILE`. Existing `GFPPS001`
checkpoints remain compatible. See [optimization validation](GFPPS/VALIDATION_optimized_20260909.md).
The component READMEs describe the controls and resume interfaces. Recorded
timings and the limits of validation are in
[September 2026 validation](docs/validation-2026-09.md).

## Prebuilt targets

| Program | Linux x86-64 (Ubuntu 22.04+) | Windows x64 |
| --- | --- | --- |
| GFPDIVSV 1.1 | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29.2/gfpdivsv-1.1-linux-x86_64-cuda13.3.tar.xz) | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29.2/gfpdivsv-1.1-windows-x86_64-cuda13.3.zip) |
| GFPPS 1.1 | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/gfpps-1.1-linux-x86_64-cuda13.3.tar.xz) | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/gfpps-1.1-windows-x86_64-cuda13.3.zip) |
| CSPS 1.0 | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/csps-1.0-linux-x86_64-cuda13.3.tar.xz) | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/csps-1.0-windows-x86_64-cuda13.3.zip) |
| CSSV 1.3 | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/cssv-1.3-linux-x86_64-cuda13.3.tar.xz) | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/cssv-1.3-windows-x86_64-cuda13.3.zip) |
| GPRSV_K 2.3 | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/gprsv_k-2.3-linux-x86_64-cuda13.3.tar.xz) | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/gprsv_k-2.3-windows-x86_64-cuda13.3.zip) |
| GPRSV_N 1.1 | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.11.16/gprsv_n-1.1-linux-x86_64-cuda13.3.tar.xz) | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.11.16/gprsv_n-1.1-windows-x86_64-cuda13.3.zip) |
| GHCWSV 1.3 | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/ghcwsv-1.3-linux-x86_64-cuda13.3.tar.xz) | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.10.29/ghcwsv-1.3-windows-x86_64-cuda13.3.zip) |
| GPRPS 2.4 | [CUDA 12.8](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.09.12/gprps-2.4-linux-x86_64-cuda12.8.tar.xz) | [CUDA 13.3](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/releases/download/v2026.09.12/gprps-2.4-windows-x86_64-cuda13.3.zip) |

Each archive includes `SHA256SUMS` and build metadata. The GPRSV_K and GPRSV_N
Linux archives include primesieve 12.x and its BSD license.

Release packages are separated by operating system. Each package contains the
native CUDA architectures shown below; select the executable matching the
target GPU.

| CUDA target | GFPS | GPRPS | GFPPS | GFNSV | GPRSV_K | GPRSV_N | GNCWSV | GHCWSV | GHCWPS |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sm_86` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `sm_89` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `sm_100` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `sm_120` | yes | yes | yes | yes | yes | yes | yes | yes | yes |

Both Linux x86-64 ELF and Windows x64 PE executables are provided. Each file
contains native cubins for the SM target in its filename. CUDA may also embed
PTX for that same target; these packages do not use one universal executable
for all GPU generations. Architectures that have only been cross-compiled are
marked as such in the Release notes; only `sm_89` was runtime-tested for the
initial public release.

The GFPPS optimization additionally has RTX 5090/Linux (`sm_120`) runtime
evidence, alongside RTX 4060 Laptop/Windows (`sm_89`). This does not extend
runtime coverage to every component, target, or operating-system combination;
consult each artifact's `BUILDINFO.txt` and the release validation notes.

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
./GPRPS_sm_89
./GFPPS_sm_89 --help
./GFNSV_sm_89 --help
./GPRSV_K_sm_89 -h
./GPRSV_N_sm_89 -h
./GNCWSV_sm_89 -h
```

Example checker invocations:

```bash
./GFPS_sm_89 --prp-check-pref 1814570322684094 16 1000
./GPRPS_sm_89 --check '328*799^325799-1'
./GFPPS_sm_89 --check '13*4000!-1' --checkpoint factorial.ckpt
```

The examples illustrate syntax only. They are not claims about the primality of
the displayed numbers.

## Building from source

The programs require a CUDA toolkit supporting the selected architecture and a
C++17 host compiler. GFPS, GPRPS, and GFPPS use Boost.Multiprecision headers; the
Windows GFNSV build uses Boost for host-side 128-bit arithmetic. GPRPS and GFPPS use CUB
through the CUDA toolkit.

Every tool includes Linux/WSL and Windows build scripts. From the repository
root, for example:

```bash
./GFPS/scripts/build-linux.sh build/linux sm_89
./GPRPS/scripts/build-linux.sh build/linux sm_89
./GFPPS/scripts/build-linux.sh build/linux sm_89
./GFNSV/scripts/build-linux.sh build/linux sm_89
./GPRSV_K/scripts/build-linux.sh build/linux sm_89
./GPRSV_N/scripts/build-linux.sh build/linux sm_89
./GNCWSV/scripts/build-linux.sh build/linux sm_89
```

```bat
GFPS\scripts\build-windows.bat build\windows sm_89
GPRPS\scripts\build-windows.bat build\windows sm_89
GFPPS\scripts\build-windows.bat build\windows sm_89
GFNSV\scripts\build-windows.bat build\windows sm_89
GPRSV_K\scripts\build-windows.bat build\windows sm_89
GPRSV_N\scripts\build-windows.bat build\windows sm_89
GNCWSV\scripts\build-windows.bat build\windows sm_89
```

Omit the architecture arguments to build all four release targets. GFPS,
GPRPS, GFPPS, and Windows GFNSV require Boost headers; set `BOOST_ROOT` to the directory that
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

GPRSV_K, GPRSV_N and GNCWSV can load `primesieve` dynamically when it is installed. They
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
./GPRPS_sm_89 --selftest
./GFPPS_sm_89 --check '3*5!+1' --verify-cpp-int
./GFNSV_sm_89 --help
./GPRSV_K_sm_89 --version
./GPRSV_N_sm_89 --version
./GNCWSV_sm_89 --version
```

For expensive searches, preserve checkpoints and logs, independently recheck
PRP hits, and compare known prefix/final residues after changing CUDA code,
compiler versions, or GPU architecture. The project-specific validation matrix
is described in [docs/correctness.md](docs/correctness.md).

## License and attribution

The repository is distributed under the GNU General Public License, version 2
or later (`GPL-2.0-or-later`). See [`LICENSE`](LICENSE). GPRSV_K, GPRSV_N and GNCWSV
contain work derived from the GPLv2-or-later mtsieve/twinsieve implementation. Other
libraries and toolchain components retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

