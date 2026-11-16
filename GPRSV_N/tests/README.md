# GPRSV_N tests

`ref_sieve.py` is an independent reference sieve for validation. It shares no
code with the CUDA program: for every prime `p <= pmax` it walks the exponent
interval with Python big-integer arithmetic and records which `n` have
`p | k*b^n-1` or `p | k*b^n+1`. A term that is itself equal to `p` (for example
`2^13-1 = 8191`) is kept, matching GPRSV_N's rule.

Usage:

```text
python ref_sieve.py K BASE NMIN NMAX PMAX FILE [paired|independent]
```

`FILE` is a GPRSV_N output file. Paired mode expects rows of `n`; independent
mode expects `n +1` / `n -1` rows. The script exits non-zero and prints the
missing/extra survivors when the sets differ.

The oracle in this directory is intentionally slow (`O(primes * exponents)`) and
is meant for small ranges (`pmax` up to about `1e6`, interval up to a few
thousand exponents). Larger checks should compare GPRSV_N's `--backend walk`
against `--backend bsgs`, or compare against `srsieve2 -c`.
