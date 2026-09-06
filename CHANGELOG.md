# Changelog

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
  source/header hashes. GSRPS 2.3, GFNSV 1.0, GSRSV 2.0, and GNCWSV 1.0
  retain their component versions.
- Published validation distinguishes complete PRP checks from million-digit
  GFPPS prefix timing extrapolations and cross-compiled GPU architectures.

## v2026.09.4 - GFPS 4.1, GSRPS 2.3, and GFNSV CUDA 1.0

- GFPS 4.1 enables half-length negacyclic NTT, DIF/DIT and shared-memory kernel
  fusion, plus checked carry batches with whole-batch adaptive replay.
  `--reference-mode`, `--batch-bits`, and diagnostic replay controls support
  direct result comparisons. Windows duty throttling now waits after short
  work windows instead of rounding a delay after every square.
- GSRPS 2.3 enables condition-checked weighted division scans and compact carry
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
- GFPS and GSRPS now flush redirected output in real time on Windows. This fixes
  delayed GSRPS progress/checkpoint output and makes GFPS interruption visible
  throughout the three-layer client process tree.

## 2026-09-04 - GSRPS arithmetic checkpoints

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
- GSRPS 2.0: CUDA probable-prime testing for generalized Sierpinski/Riesel forms.
- GSRSV 2.0: CUDA sieve for generalized Sierpinski/Riesel, primorial, and factorial forms.
- GNCWSV 1.0: CUDA sieve for generalized Cullen/Woodall and near Cullen/Woodall forms.
- Added reproducible Linux/WSL and Windows build scripts for `sm_86`, `sm_89`,
  `sm_100`, and `sm_120`.
- Added MSVC portability for host-side 128-bit arithmetic and Windows SDK macro
  compatibility without changing CUDA kernels.

This is the first release of the four tools in a single source repository. Each
tool keeps its own program version; repository release tags use calendar dates.
