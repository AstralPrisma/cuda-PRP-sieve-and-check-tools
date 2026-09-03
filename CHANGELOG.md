# Changelog

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
