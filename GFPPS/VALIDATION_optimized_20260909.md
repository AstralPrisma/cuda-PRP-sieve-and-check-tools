# GFPPS 1.0: parallel NTT and exact carry optimization

This record accompanies suite release **v2026.09.9**. GFPPS's component version
remains **1.0**. It describes finite arithmetic and compatibility tests, not a
proof that every accepted input or hardware execution is correct. A Fermat
PRP result is not a deterministic primality proof.

## Promoted implementation

- Split the two NTT prime planes into independent grid work, in both global
  and shared-memory transform kernels.
- Use 1024-point shared tiles with 128 threads; the default NTT block cap is
  256 per prime plane, still overridable with `--force-ntt-blocks`.
- Fuse the three carry relaxation passes with carry-map generation. Preserve
  the complete CUB carry scan, REDC cancellation checks, range checks, and
  latched arithmetic error reporting.
- Use the equivalent low-word carry predicate in Montgomery cancellation.
- Retain exact radix-2^15 NTT/CRT arithmetic, the accepted parameter bounds,
  Fermat witness semantics, explicit checkpoint opt-in, and `GFPPS001` format.

This promotion does not include the unbounded carry-backtracking experiment,
carry truncation, floating-point modular arithmetic, or changes to GFPS/GPRPS.

### Why the carry fusion preserves the original operation

Write one synchronous relaxation as
`r[i] = (x[i] mod B) + floor(x[i-1]/B)`, where `B=2^15` and negative indices
are zero. Evaluating that recurrence three times depends on the original
`x[i]` through `x[i-3]`. The fused kernel computes those three levels directly
from an immutable input and writes a separate output. This is exactly the
three old relaxations, not an assumption that a carry can travel only three
digits. The subsequent full prefix scan propagates the remaining carry over
the entire array. The same coefficient/range guards remain enabled.

The split NTT grids change scheduling, not the modulus, roots of unity,
transform order, or CRT reconstruction. Kernel work continues on the same
explicit CUDA stream, including CUB operations and CUDA Graphs.

## Controlled performance measurements

The baseline is the previous GFPPS 1.0 arithmetic path with block cap 96.
The optimized path uses its new default cap 256. Each full comparison used
two runs per variant in alternating/reversed order, after warm-up. There was
one test computation at a time. The Windows tests left ordinary desktop
applications running; no clock or power-limit changes were made.

These timings come from the preceding isolated optimization builds of the
same arithmetic path now enabled by default in the release. The final
release binaries were independently smoke-tested below; they did not repeat
the complete benchmark or the entire 60-case optimization suite.

### Complete 100,001-digit case

Input: `2*25206!+1`, witness 2, **332,195 exponent bits**, Graphs enabled.
Benchmarks save initial and final checkpoints with periodic saves disabled.
Times below are the checker's **exponentiation time**, not process wall time.

| Platform and matching GPU | Baseline median | Optimized median | Throughput gain | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Ubuntu 22.04, GCC 11.4, CUDA 13.3.33; RTX 5090 (`sm_120`) | 31.326306 s | 19.082555 s | 64.2% | 39.1% |
| Windows x64, MSVC 14.51, CUDA 13.3.33; RTX 4060 Laptop (`sm_89`) | 42.770754 s | 35.198393 s | 21.5% | 17.7% |

Compute throughput gain as `baseline/optimized - 1`; compute time reduction
as `1 - optimized/baseline`. These percentages are different quantities.
Each row is a same-platform comparison; the table does not isolate the
performance difference between Linux and Windows or between the two GPUs.

All compared complete runs returned `COMPOSITE` with
`checksum=912290683:910219393` and byte-identical complete checkpoints:

```text
SHA-256 282e31a871d852d612f3cf501f94ebc7e89a74096b0e956a9353c56efc556471
```

For the optimized path on the 4060, this case's transform grids do not reach either cap 96 or 256;
the gain cannot be attributed simply to the larger block-cap number.

### Million-digit prefix only

Input: `7777*2303867#+1`, witness 2, 1,000,005 estimated decimal digits and
3,321,942 exponent bits. The controlled Windows comparison advanced only
**4096 bits**, with three interleaved runs per variant:

| RTX 4060 Laptop / Windows | Baseline median | Optimized median | Throughput gain |
| --- | ---: | ---: | ---: |
| 4096-bit prefix | 2.240692 s | 2.096653 s | 6.9% |

Same-progress checkpoint bytes matched. This is not a complete million-digit
PRP test, a full-run stability result, or a promise of the same percentage
gain at other sizes. Setup and modulus construction can dominate process wall
time for short prefixes; the table reports exponentiation time only.

## Optimization validation evidence

The optimized paths were tested on **Linux/RTX 5090 (`sm_120`)** and
**Windows/RTX 4060 Laptop (`sm_89`)**. Each platform recorded:

- **60 independent integer-residue checks**, covering factorial and primorial
  inputs, both signs, Graph enabled/disabled, proper composite and prime
  examples, a base-2 pseudoprime, and witnesses 2, 3, 17, and 255. Small
  results were compared with independent Python/Boost integer arithmetic.
- **Six large-layout prefix comparisons** against the baseline, covering
  both families/signs and approximately 100,000-digit and million-digit
  layouts. Checkpoint files at the same progress matched byte-for-byte.
- **Two complete known primorial PRPs**: `4599280*104561#+1` and
  `2874926*74719#-1`, returning the expected Fermat result and matching the
  baseline checkpoints. These checks repeat the PRP test, not a proof.
- **Cross-build checkpoint continuation**, plus corrupted-checkpoint and
  wrong-parameter rejection.
- **Complete 100,001-digit comparison** described above, including the full
  checkpoint rather than only its short displayed checksum.

The Linux optimized build also passed a real SIGINT save/resume test. The
Windows transfer study did not repeat real console Ctrl+C testing; prior 1.0
interrupt evidence is historical, not a new optimized-build test. Likewise,
the original 112-case/platform suites and 29 full known-primorial checks in
[the README](README.md#original-10-validation-historical-timings) describe the
earlier 1.0 implementation and are not counted again as fresh optimization
tests.

## Release binaries and deployment boundaries

The final release executables passed independent smoke checks on **Windows
`sm_89`**, **Linux `sm_89`**, and **Linux `sm_120`**. Each exact binary passed
six Python/Boost integer-reference cases (`1*3!+1`, `1*4!+1`, `1*7#+1`,
`1*7#-1`, `170*2!+1`, and `13*257!+1`), checkpoint continuation across
Graph-enabled/disabled execution with complete-state equality, and a
512-bit prefix of the large-NTT `2*25206!+1` case. All three large-prefix
checkpoint files have the same SHA-256:

```text
33722814289224773b70eb0d11370e29cb1e778f5304cd736a43819939a86387
```

All eight final Windows/Linux binaries had their native cubin and PTX targets
inspected. The five other OS/SM combinations were cross-built and inspected,
not executed on matching GPUs. The full timings and larger test counts above
remain optimization-build evidence, not claims of those tests being rerun on
every final executable.

The release rebuilds GFPPS for Linux/Windows and `sm_86`, `sm_89`, `sm_100`,
and `sm_120`. Compilation and embedded target inspection alone are not
matching-hardware runtime tests. The records above establish optimization-path
coverage; the per-binary `BUILDINFO.txt` and manifest specify the validation
actually performed on each final release artifact. Do not infer Windows
`sm_120` coverage from Linux `sm_120`, or Linux `sm_89` coverage from Windows
`sm_89`. The other forty executables are reused from v2026.09.8 with their
original source and runtime provenance.

Linux GFPPS uses an Ubuntu 22.04-compatible build with static C++/GCC/CUDA
runtimes. glibc, the Linux loader, and the compatible NVIDIA driver are still
required; this is not a fully static executable. Dependency inspection and
actual symbol-version floors belong to each artifact's build record.
The rebuilt Linux artifacts require at most GLIBC 2.34 and no dynamic
GLIBCXX/libstdc++ dependency; the other components retain their own requirements.
Their ELF dynamic dependencies are `libm.so.6`, `libc.so.6`, and the Linux
loader; access to a compatible NVIDIA driver is still required at runtime.

Version 1.0 and the CLI/checkpoint format remain unchanged. Existing checkpoints
can be continued after replacing a stopped executable. Use one writer per
checkpoint, preserve important logs and checkpoint evidence, and independently
verify PRP hits. A checksum protects file integrity; it does not prove the
calculation or authenticate an untrusted checkpoint producer.

GFPS remains 4.4 without its separate 5090 layout experiment: its 4060 prefix
study showed only a 1.8% median difference, smaller than the observed run
variation. No GFPS speedup is claimed or promoted in this release.
