# Correctness and result semantics

This document defines what the five tools establish, what they do not
establish, and the minimum evidence expected before a build is used for a long
search.

## PRP is not a primality proof

GFPS and GSRPS perform probable-prime checks. Passing such a check means that a
candidate satisfies the implemented congruence for the selected witness. It
does not constitute a deterministic proof that the candidate is prime.

- A failed PRP check identifies a composite, assuming the arithmetic execution
  was correct.
- A passed PRP check is a PRP hit and must be independently rechecked.
- Publication as a prime requires a deterministic proof or certificate from an
  appropriate independent program.

Do not label a GFPS or GSRPS `PRP` result as `prime` merely because the final
residue is one.

## Checker arithmetic

### GFPS

GFPS represents residues modulo `b^(2^n)+1` and uses CUDA NTT convolution,
centered CRT reconstruction, and balanced carry normalization. Depending on the
coefficient bound, the resident path selects three or four NTT primes. It must
abort rather than continue when the centered CRT range is insufficient.

GFPS 4.1 batches carry work only with a persistent convergence-failure flag.
Every accepted batch has either passed those checks or been restored and
replayed from its saved start using adaptive carry. Validate both paths,
including forced replay, before changing batching or carry bounds.

For even `b`, the ordinary balanced alphabet has exactly one missing residue
class. GFPS includes the additional canonical representation

```text
d[0] = -b/2
d[i] = 1-b/2, i > 0
```

for that class. Checkpoints authenticate their metadata and digits and validate
the ordinary or special digit range before resuming.

### GSRPS

GSRPS checks `N = k*b^n+c`, where `c` is `+1` or `-1`. Its accepted production
path uses integer NTT/CRT arithmetic and exact carry/reduction logic. Runtime
tuning parameters such as NTT block count and window size change scheduling and
performance, not the intended modular result.

`--verify-cpp-int` requests an independent Boost.Multiprecision repetition for
supported checks. It is useful for validation, but its use does not turn a
Fermat PRP result into a deterministic primality proof.

## Siever semantics

GFNSV, GSRSV, and GNCWSV remove candidates for which they find a factor in the requested
prime interval. A surviving term has only survived that sieve range.

- A reported factor should exactly divide the displayed candidate.
- Use `--verify` for GSRSV/GNCWSV when validating a build or changing kernels;
  GFNSV verifies factors on the CPU by default unless `--no-verify` is supplied.
- Compare uninterrupted and resumed runs over small ranges.
- Preserve the input, output header, prime bounds, factors file, and command
  line needed to reproduce a sieve run.

An empty factor file is not proof that all candidates are prime, and a survivor
file is not a PRP result.

## Required validation matrix

Every release candidate should pass the following checks before publication.

### Source-level checks

1. Compile all advertised `(tool, sm)` targets from the tagged source.
2. Confirm each artifact contains the advertised native target with
   `cuobjdump --list-elf`.
3. Record compiler warnings and reject builds with unresolved errors.
4. Confirm that no checkpoint, candidate, result, token, private URL, or local
   absolute path is included in the source archive.

### Checker tests

1. Run the built-in GFPS and GSRPS self-tests.
2. Compare small exact cases against an independent big-integer calculation.
3. Test both signs in GSRPS and both square/multiply branches used by each
   checker.
4. Compare known prefix checksums before and after every arithmetic
   optimization.
5. For GFPS, exercise three-prime and four-prime paths, both exponent-bit
   variants, the even-base special residue, checkpoint save/resume, corrupted
   checkpoint rejection, and insufficient-CRT rejection.
6. For GSRPS, exercise automatic and forced NTT blocks, automatic and forced
   windows, CUDA Graph enabled/disabled paths, and any persistent tuning cache.
7. Complete at least one end-to-end composite check with an unchanged expected
   final residue. Independently repeat any PRP hit.

### Siever tests

1. Compare small output ranges with direct divisibility enumeration.
2. Verify every emitted factor on the CPU.
3. Compare single-thread and multi-thread survivor sets.
4. Interrupt and resume a controlled run and compare it with an uninterrupted
   run.
5. Exercise built-in and optional primesieve prime-generation modes when both
   are available.
6. For GFNSV, compare paired-root and full-root survivor sets, keep a small
   prime candidate when a trial factor equals the candidate itself, and reject
   corrupted or mismatched resume state. Test interruption during root/batch
   work so no unprocessed prime interval is skipped.

When comparing two sieve builds, compare survivor files and the set of removed
expressions. If one expression has several factors in the requested interval,
parallel workers may record different valid first factors or a different factor
line order. This is acceptable only when every reported factor verifies and the
survivor and removed-expression sets are identical.

### Architecture status

Cross-compilation is not runtime validation. Release notes must distinguish:

- `built`: nvcc produced an artifact and its cubin target was inspected;
- `self-tested`: built-in GPU tests passed on that architecture;
- `end-to-end tested`: a recorded real check or sieve completed;
- `not runtime-tested`: no matching GPU was available.

Never copy runtime-test status from one SM target to another.

## Long-run safeguards

- Keep checkpoint and result files until the server or downstream workflow has
  confirmed receipt.
- Retain the exact executable hash and command line with every long result.
- Record GPU model, driver, CUDA build, architecture target, and source commit.
- Treat CUDA errors, failed postconditions, CRT-bound rejection, malformed
  checkpoints, and incomplete output as failed work rather than as a composite.
- Re-run important hits with an independent implementation and, when practical,
  different hardware.

Silent hardware faults and implementation defects cannot be excluded solely by
a successful process exit. Reproducible checksums and independent verification
are part of the result, not optional metadata.
