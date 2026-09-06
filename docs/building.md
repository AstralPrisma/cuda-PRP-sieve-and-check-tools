# Building on Linux and Windows

All six programs are single-translation-unit CUDA C++17 applications. The
build scripts intentionally create one executable per CUDA SM target. Each
contains native cubins for that target; CUDA may also emit target-matched PTX.
The release therefore uses separate files instead of one universal executable
for all GPU generations.

## Common prerequisites

- An NVIDIA CUDA Toolkit new enough to support the requested `sm_XX` target.
- A 64-bit CUDA-supported host compiler.
- Boost headers for GFPS, GSRPS, and GFPPS, and for GFNSV on Windows.
- CUB/CCCL from the CUDA Toolkit for GSRPS and GFPPS.

GSRSV and GNCWSV optionally load primesieve at runtime. It is not a build-time
dependency because no primesieve header or import library is linked.

## Linux and WSL

Each tool accepts an output directory followed by zero or more architectures:

```bash
./GFPS/scripts/build-linux.sh build/linux sm_89
./GSRPS/scripts/build-linux.sh build/linux sm_86 sm_89
./GFPPS/scripts/build-linux.sh build/linux sm_89
./GFNSV/scripts/build-linux.sh build/linux sm_89
```

With no architecture arguments, a script builds `sm_86`, `sm_89`, `sm_100`,
and `sm_120`. Set `NVCC=/path/to/nvcc` when `nvcc` is not on `PATH`; the scripts
also recognize `/usr/local/cuda/bin/nvcc` automatically.

Build all components with:

```bash
./scripts/build-all-linux.sh release-assets sm_86 sm_89 sm_100 sm_120
```

GSRSV and GNCWSV link `libdl` and enable pthread support for prime-generation
workers. GFNSV has its own integer prime/root generation and does not load
primesieve. GFPS, GSRPS, GFPPS, and GFNSV use per-thread default-stream semantics.
Keep GFPPS's `ntt_backend.cuh` and `sha256.hpp` next to `src/GFPPS.cu`.
Its build script uses system Boost headers or an optional `BOOST_ROOT`; explicit
include paths are made relative to `src` before invoking nvcc.

## Windows

Install the CUDA Toolkit and a supported 64-bit Microsoft Visual C++ compiler.
GFPS, GSRPS, GFPPS, and GFNSV additionally require Boost headers. Point `BOOST_ROOT` at the
directory immediately above the `boost` folder:

```bat
set BOOST_ROOT=C:\Libraries\boost_1_83_0
GFPS\scripts\build-windows.bat build\windows sm_89
GSRPS\scripts\build-windows.bat build\windows sm_89
GFPPS\scripts\build-windows.bat build\windows sm_89
GFNSV\scripts\build-windows.bat build\windows sm_89
GSRSV\scripts\build-windows.bat build\windows sm_89
GNCWSV\scripts\build-windows.bat build\windows sm_89
```

The `.bat` files are small launchers for the adjacent PowerShell scripts. They
work with Windows PowerShell and use MSVC's conforming preprocessor, UTF-8
source handling, C++17, optimization, and per-thread default-stream semantics.
Run them from a CUDA-compatible x64 MSVC developer environment. GFPPS compiles
from its `src` directory using `GFPPS.cu` as a bare filename and a relative
Boost include directory to avoid embedding maintainer-local paths. Its Boost
headers must be on the same drive as the source tree on Windows.

Build every component and target with:

```bat
scripts\build-all-windows.bat release-assets sm_86 sm_89 sm_100 sm_120
```

Set `NVCC` to the full compiler path if `nvcc.exe` is not on `PATH`.

## Validation

On hardware matching the built target, run:

```bash
./scripts/smoke-test-linux.sh release-assets sm_89
```

or:

```bat
scripts\smoke-test-windows.bat release-assets sm_89
```

The smoke suite runs GFPS/GSRPS self-tests, GFPPS small prime/composite and
checkpoint/resume cases with independent `cpp_int` verification, and small GPU
sieve jobs with CPU factor verification. It does not replace the longer release matrix in
[correctness.md](correctness.md).

Use `cuobjdump --list-elf <binary>` to confirm that the embedded cubin matches
the architecture in the filename. A successful cross-compile is not a runtime
test on that GPU generation.
