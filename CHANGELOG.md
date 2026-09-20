# 2026.09.13

- GFPS4.5: remove the provably zero highest quotient digit in large-base reciprocal division; retain general input support and exact correction.
- GHCWPS1.2: checkpoint-compatible15/18/21-bit adaptive arithmetic with exact safety bounds and measured device/transform policies.
- GNCWSV1.2.0: shared multi-mode prime generation, safe per-chunk fast44 selection and sparse-workset coordinator tuning.
- Native Windows/Linux sm86/89/100/120 builds; runtime evidence is recorded separately for physically tested targets. Other components unchanged.

# 2026.09.12

- GHCWSV 1.2 adapts GNCWSV's 20%-smaller workset trigger to refresh GPU prime batches, while a separate producer batch preserves every unprocessed tail.
- Keeps the arithmetic kernel and v1/v2 snapshot formats unchanged; adds a fixed scheduling reference, recent throughput, p-boundary throughput and recent-window ETA.
- Bounded fresh-range RTX4060 Laptop measurements improve by about 4.5x including process overhead. Already sparse continuation shows no clear speedup.
- Eight native builds checked; GPU arithmetic, independent oracle comparisons, multi-launch worksets and real interrupt/resume checked on sm_89. Other seven components unchanged.

# 2026.09.11

- GHCWSV 1.1: shared-work `--sign both`, independent sign masks/factors, v2 sign-tagged snapshots, v1 compatibility, per-sign counts, and converter sign filtering.
- Windows and WSL sm_89 correctness/interrupt tests passed; all four native targets built on both platforms.
- Other seven components unchanged.

# 2026.09.10

- Add GHCWSV 1.0 and GHCWPS 1.0 for b^n*n^b±1.
- Wire user display banners with Windows UTF-8 support; sieve always prints, checker prints help/usage only.
- Add Windows/Linux native sm_86/89/100/120 artifacts and offline/network task converter.
- Other six tools and component versions unchanged.

# Changelog

## v2026.09.9 - GFPPS 1.0 parallel NTT and exact carry optimization

- Split both shared and global NTT work across the two prime planes; use
  1024-point shared tiles with 128 threads and default NTT block cap 256.
- Fuse three exact carry relaxation passes with carry-map generation, retaining
  the full CUB scan and arithmetic error checks. No bounded/truncated carry
  shortcut or unbounded carry-backtracking experiment is enabled.
- Recorded complete `2*25206!+1` checks improve throughput by 64.2% on RTX
  5090/Linux and 21.5% on RTX 4060 Laptop/Windows. Million-digit measurements
  cover prefixes only, not full completed checks.
- Keep GFPPS component version 1.0, CLI/result semantics, and `GFPPS001`
  checkpoint compatibility. Old/new residues, checkpoints, independent integer
  checks, known PRPs, and resume/error rejection cases were compared.
- Rebuild only GFPPS's eight target/platform binaries. Linux GFPPS uses an
  Ubuntu 22.04-compatible build with static C++/GCC/CUDA runtimes; glibc and
  the GPU driver remain external dependencies. See artifact build metadata.
- Reuse the other 40 binaries from v2026.09.8 with their original provenance.
  GFPS's experimental 5090 optimization is not included in this promotion.

## v2026.09.8 - GPRSV 2.1 factorial/primorial acceleration

- Replaced repeated packed-chunk remainders and Montgomery conversions with
  exact pre-seeded raw-chunk multiplication; simplified product-only REDC carry.
- Tested fixed-range RTX 4060 workloads ran about 3.3–3.5x faster, with identical
  survivors. The power-form arithmetic and sieve/resume/file interfaces remain
  unchanged. Prime 2 and small 32-bit products retain their previous path.
- Added independent integer and direct GPU inverse regression utilities;
  corrected the primorial example to use a prime endpoint.
- Rebuilt eight GPRSV binaries for Windows/Linux and four SM targets. Other
  forty executables are reused from v2026.09.7 with original build provenance.
- Sieve distribution remains in GitHub; separate PRPNet client ZIPs omit sieves.

## v2026.09.7 - GFNSV CUDA 1.1 single-file recovery

- GFNSV 1.1 writes self-contained v4 GFN/ABC/base/expression snapshots with
  bounded resume metadata, ordered survivors, and count/SHA-256 validation.
  The main file alone resumes across Windows/Linux; factor logs are optional.
- Added GPRSV-style short options, periodic survivor/progress/ETA output, and
  the combined `-4`/`-5`/`-6` efficiency limits. Efficiency-controlled runs
  override `pmax` and save their completed prefix when stopping normally.
- Legacy CUDA v3 checkpoints and validated old GFN/ABC plus v3 companion pairs
  migrate on save. Damaged v4 snapshots never fall back to an older companion.
- Both GFNSV platform packages include standard-library Python tools for
  validated offline conversion to `cand_<b>.txt`, with dry-run, per-n manifests,
  conflict checks, and safe repeat conversion. No upload or PRP testing occurs.
- Only GFNSV's eight platform/SM binaries are updated. The other 40 binaries
  are reused byte-for-byte from v2026.09.6 after source and binary hash checks;
  their prior runtime-test evidence is retained, not reported as a new test run.
- The 1.1 validation covers 47 CLI cases, compact codecs, cross-language and
  converter tests, real interruptions, and single-file cross-platform resume.
  Only `sm_89` has matching-GPU runtime evidence; other targets are cross-built.

## v2026.09.6 - Windows console UTF-8 rebuild

- All six programs initialize UTF-8 output for real Windows consoles, then
  restore the invoking console's original output codepage on normal exit.
  Input codepage is unchanged; redirected files and pipes remain UTF-8.
- Added the required local `console_utf8.hpp` helper to every component.
  Linux initialization is a no-op. Arithmetic, sieving, checkpoints, and
  component versions are unchanged.
- The exact Windows `sm_89` binaries passed 54 real-console/redirection help
  tests plus 18 complete screen-to-file comparisons. These output tests do
  not constitute another full-length PRP validation round.

## v2026.09.5 - GFPS 4.4 and GFPPS 1.0

- GFPS 4.4 fuses checked-batch carry propagation, fixed-point validation, and
  RNS export, retaining latched failures and independent adaptive replay.
  Reduced-duty, non-batched, and reference runs retain the existing path.
  CRT limits and the checkpoint format are unchanged.
- Added GFPPS 1.0 for `k*n!+/-1` and `k*n#+/-1` Fermat PRP checks, using
  exact integer NTT/Montgomery arithmetic without GPU FP64.
- GFPPS supports explicit SHA-256 checkpoints, Windows/Linux cross-resume,
  safe Ctrl+C handling, and default 100,000-bit progress. There is no implicit
  checkpoint file, PRPNet integration, or deterministic proof generation.
- Build, smoke, and release packaging scripts now cover six tools and all
  source/header hashes. GPRPS 2.3, GFNSV 1.0, GPRSV 2.0, and GNCWSV 1.0
  retain their component versions.
- Published validation distinguishes complete PRP checks from million-digit
  GFPPS prefix timing extrapolations and cross-compiled GPU architectures.

## v2026.09.4 - GFPS 4.1, GPRPS 2.3, and GFNSV CUDA 1.0

- GFPS 4.1 enables half-length negacyclic NTT, DIF/DIT and shared-memory kernel
  fusion, plus checked carry batches with whole-batch adaptive replay.
  `--reference-mode`, `--batch-bits`, and diagnostic replay controls support
  direct result comparisons. Windows duty throttling now waits after short
  work windows instead of rounding a delay after every square.
- GPRPS 2.3 enables condition-checked weighted division scans and compact carry
  reduction, preserving exact fallback behavior for other arithmetic shapes.
- Added GFNSV CUDA 1.0 for generalized Fermat interval sieving, with paired-root
  enumeration, default CPU factor verification, and saved-state continuation.
  The separate CPU GFNSV archive is not part of this repository.
- The local source tree, build matrix, component documentation, smoke tests,
  and packaging configuration now cover five tools.
- Added the exact GFPS even-base CRT ceilings for n=10..20; larger n is marked
  `Not supported`. Approximate `--analyze` output is not authoritative at the boundary.
- Release packaging validates explicit compiler, flags, source/header hashes,
  binary hashes and runtime-test evidence instead of inferring tests from SM names.

## 2026-09-04 - Cross-platform interrupt and live-output fix

- Updated the toolkit's matching Python clients so each child has an isolated
  process group/session and interrupt delivery flows once from outer client to
  wrapper to CUDA checker. Repeated interrupts are ignored during shutdown so
  cleanup and final output draining can finish.
- Replaced blocking Windows pipe reads in those clients with background
  readers, keeping the main thread responsive to `Ctrl+C`/`CTRL_BREAK`.
- GFPS now saves the latest safe main checkpoint when interrupted and exits
  with status 130; interrupted tasks remain available for resume and are not
  submitted or released as completed.
- GFPS and GPRPS now flush redirected output in real time on Windows. This fixes
  delayed GPRPS progress/checkpoint output and makes GFPS interruption visible
  throughout the three-layer client process tree.

## 2026-09-04 - GPRPS arithmetic checkpoints

- Added portable, SHA-256-protected `GSRPCK1` checkpoints for CUDA `--check`
  runs, including periodic saves, completion saves, checkpoint inspection, and
  strict resume validation.
- Added safe `Ctrl+C`/`CTRL_BREAK` handling. Long zero-bit runs are split into
  at most 64-square chunks so an interrupt can synchronize and save promptly.
- Checkpoints contain only canonical radix limbs and progress metadata; NTT
  tables, tuning data, and CUDA Graphs are rebuilt on resume.
- Validated Windows/Linux cross-resume, completed-state replay, known PRP and
  composite results, corrupted-file rejection, and real process-tree interrupts.

## 2026-09-04 - Windows GFPS checkpoint fix

- Fixed periodic GFPS checkpoint updates on Windows. The Microsoft C runtime
  `rename()` does not replace an existing destination, so the second save used
  to stop a run with `cannot move checkpoint temp file into place`.
- Windows now flushes the completed temporary checkpoint and atomically replaces
  the previous file with `MoveFileExW`.
- Revalidated fresh writes, repeated overwrites, restart/resume, SHA-256 parsing,
  and the GFPS CUDA self-test on `sm_89`.

## 2026-09-03 - Initial public source release

- GFPS 4.0: CUDA probable-prime testing for generalized Fermat numbers.
- GPRPS 2.0: CUDA probable-prime testing for generalized Proth/Riesel forms.
- GPRSV 2.0: CUDA sieve for generalized Proth/Riesel, primorial, and factorial forms.
- GNCWSV 1.0: CUDA sieve for generalized Cullen/Woodall and near Cullen/Woodall forms.
- Added reproducible Linux/WSL and Windows build scripts for `sm_86`, `sm_89`,
  `sm_100`, and `sm_120`.
- Added MSVC portability for host-side 128-bit arithmetic and Windows SDK macro
  compatibility without changing CUDA kernels.

This is the first release of the four tools in a single source repository. Each
tool keeps its own program version; repository release tags use calendar dates.
