# GFNSV CPU-only regression tests

These tests generate small mock files. They require no GPU, production sieve
input, server, or administrator token. Run them from the repository root.

Python 3.10 or later (standard library only):

```sh
python -B -m unittest discover -s GFNSV/tests -p "test_*.py" -v
```

The suite covers legacy v3 and compact v4 readers, single-file integrity,
truncation, stale-sidecar rejection, read-only conversion, n-bound task
directories, no-clobber publication, dry-run and repeat runs. Platform-specific
filesystem tests can be skipped when the required feature is unavailable.

C++17 (no CUDA or Boost required for these two host-only tests):

```sh
mkdir -p build
g++ -std=c++17 -O2 GFNSV/tests/test_efficiency.cpp -o build/test_efficiency
./build/test_efficiency
g++ -std=c++17 -O2 GFNSV/tests/test_compact.cpp -o build/test_compact
./build/test_compact build/compact-test-output
```

The compact test output directory must not already exist; generated fixtures
are retained for inspection. Windows can build the same tests with a C++17
MSVC toolchain. Real GPU and interrupt validation is documented separately in
the repository's validation notes and exact-binary release manifest.
