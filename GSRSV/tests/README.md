# Product-path regression utilities

These checks are offline and do not modify existing candidate files. Do not run
GPU checks alongside a production computation. Redirect stdout to a new log if
you want to retain the detailed results.

CPU-only exact integer and format/oracle checks (from `GSRSV/`):

```bash
python3 tests/verify_montgomery_products.py
python3 -m unittest discover -s tests -p 'test_oracle.py'
python3 tests/gsrsv_oracle.py plan
```

The plan is bounded and does not itself run CUDA. It covers both forms and sign
modes, self-factor exceptions and narrow intervals crossing 2^32 or near 2^62.
Actual sieve outputs can be checked with the oracle's `check`/`compare`/`factors`
commands; use `--help` for arguments. A factor log can contain different winning
primes under racing streams, so only its divisor correctness and removal
coverage, not byte equality, are required.

Direct GPU inverse check against CPU uint128 arithmetic, with one small kernel
per group (Linux, from `GSRSV/`, target adjusted to the actual GPU):

```bash
mkdir -p build
nvcc -O3 -std=c++17 --threads 1 -arch=sm_89 -Xcompiler=-pthread -ldl tests/inverse_audit.cu -o build/inverse_audit
./build/inverse_audit
```

Windows x64 from an initialized compatible MSVC environment:

```bat
nvcc -O3 -std=c++17 --threads 1 -arch=sm_89 -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor tests\inverse_audit.cu -o build\inverse_audit.exe
build\inverse_audit.exe
```

Create the Windows `build` directory first. This requires the CUDA toolkit only
for compilation. The main prebuilt GSRSV program does not require nvcc to run.

The inverse check includes 46 groups and 3316 prime/input combinations in the
recorded matrix. It checks every GPU-returned value; sparse or zero-hit sieve
output alone is not accepted as an arithmetic oracle. The CPU arithmetic audit
also covers composite odd moduli for REDC/product bounds, but only prime moduli
are used to assert Fermat inversion.

`--version` in GSRSV includes its banner and the final version line. Normal
Ctrl+C may still return zero after writing a partial frontier, so use the saved
frontier and expected prime range to distinguish interruption from completion.
