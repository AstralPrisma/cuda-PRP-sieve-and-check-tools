# GFPPS 1.1: exact single-factor removal

In addition to the legacy `k*n!+/-1` and `k*n#+/-1`, the checker accepts
`n!/d+/-1` and `n#/q+/-1`. Division happens exactly before the final offset;
floating-point estimates never construct the candidate.

```sh
./GFPPS_sm_89 --check '26951!/2-1' --checkpoint progress.ckpt
./GFPPS_sm_89 --check '26951!/2-1' --checkpoint progress.ckpt --resume-checkpoint
./GFPPS_sm_89 --check '366439#/3+1'
./GFPPS_sm_89 --capabilities
```

Factorial divisors are2..n-1. Primorial divisors must be prime and below the
largest prime factor. The engine also permits divisor2 for diagnostics, but
those primorial quotients normally produce even composites; server-generated
search jobs use odd prime divisors starting at3.

The shared parser permits a positive leading multiplier as well, but Prime
Seeker's factor-removal task protocol uses the simple `n!/d`/`n#/q` forms.
The engine does not establish that the mother `n!+/-1` or `n#+/-1` is prime.

Legacy jobs retain the `GFPPS001` checkpoint format and were tested in both
old-to-new and new-to-old directions. Quotient checkpoints use `GFPPS002` and
bind the divisor, modulus, expression parameters and witness with SHA-256.
Never reuse them with the old engine or a different candidate. `--stop-file`
requests a safe checkpoint/exit. `--max-bits` is diagnostic: `PARTIAL` is not
a final mathematical verdict.

Linux builds can use GNU GMP for CPU-only candidate and Montgomery-parameter
construction. This host library is **not a CUDA program**. Official Linux
1.1 binaries statically link GMP6.3.0, so there is no separate GMP executable,
fixed `bin/linux64` dependency, or runtime libgmp installation requirement.
The matching upstream source archive and license accompany the release.
For a source build set `GMP_ROOT` to a built static prefix; without it, the
portable Boost construction path remains available. Windows uses that Boost
path while retaining the same CUDA arithmetic.

The device NTT, carry and modular-square kernels are unchanged from the
validated previous arithmetic path. Validation covers21 representative CUDA
expressions and Graph/resume variants, small exact constructor comparisons,
and independent GMP construction/prefix comparisons at larger sizes. This
is not a formal proof of all inputs or hardware. No4x acceleration is claimed
from adding division syntax.
