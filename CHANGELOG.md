# Changelog

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
