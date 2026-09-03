# Building on Linux and Windows

All four programs are single-translation-unit CUDA C++17 applications. The
build scripts intentionally create one native-cubin executable per CUDA SM
target; they do not add a universal PTX fallback.

## Common prerequisites

- An NVIDIA CUDA Toolkit new enough to support the requested `sm_XX` target.
- A 64-bit CUDA-supported host compiler.
- Boost headers for GFPS and GSRPS.
- CUB/CCCL from the CUDA Toolkit for GSRPS.

GSRSV and GNCWSV optionally load primesieve at runtime. It is not a build-time
dependency because no primesieve header or import library is linked.

## Linux and WSL

Each tool accepts an output directory followed by zero or more architectures:

```bash
./GFPS/scripts/build-linux.sh build/linux sm_89
./GSRPS/scripts/build-linux.sh build/linux sm_86 sm_89
```

With no architecture arguments, a script builds `sm_86`, `sm_89`, `sm_100`,
and `sm_120`. Set `NVCC=/path/to/nvcc` when `nvcc` is not on `PATH`; the scripts
also recognize `/usr/local/cuda/bin/nvcc` automatically.

Build all components with:

```bash
./scripts/build-all-linux.sh release-assets sm_86 sm_89 sm_100 sm_120
```

The sievers link `libdl` and enable pthread support for prime-generation
workers. GFPS and GSRPS build with per-thread default-stream semantics.

## Windows

Install the CUDA Toolkit and a supported 64-bit Microsoft Visual C++ compiler.
GFPS and GSRPS additionally require Boost headers. Point `BOOST_ROOT` at the
directory immediately above the `boost` folder:

```bat
set BOOST_ROOT=C:\Libraries\boost_1_83_0
GFPS\scripts\build-windows.bat build\windows sm_89
GSRPS\scripts\build-windows.bat build\windows sm_89
GSRSV\scripts\build-windows.bat build\windows sm_89
GNCWSV\scripts\build-windows.bat build\windows sm_89
```

The `.bat` files are small launchers for the adjacent PowerShell scripts. They
work with Windows PowerShell and use MSVC's conforming preprocessor, UTF-8
source handling, C++17, optimization, and per-thread default-stream semantics.

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

The smoke suite runs both checker self-tests and small GPU sieve jobs with CPU
factor verification. It does not replace the longer release matrix in
[correctness.md](correctness.md).

Use `cuobjdump --list-elf <binary>` to confirm that the embedded cubin matches
the architecture in the filename. A successful cross-compile is not a runtime
test on that GPU generation.
