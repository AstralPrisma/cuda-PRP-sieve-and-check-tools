# September 2026 local validation

This record covers the GFPS 4.1/4.4 and GSRPS 2.3 arithmetic promotions,
GFPPS 1.0, and the GFNSV CUDA 1.0 sieve. It describes tested cases, not a guarantee that every
parameter, device, or hardware execution is free of defects.

## Hardware and build coverage

The runtime test device was an NVIDIA GeForce RTX 4060 Laptop GPU (`sm_89`).
Windows x64 and Linux/WSL x86-64 builds use CUDA 13.3, with MSVC and GCC host
compilers respectively. `sm_86`, `sm_100`, and `sm_120` binaries were
cross-compiled and their cubin targets checked; they were not executed on
matching hardware.

## GFPS 4.4 carry fusion

The optimized batch path fuses carry rotation, convergence checking, and RNS
export. It uses three carry kernels for the large-base case and retains four
passes for small bases. The NTT, CRT capacity, rounding rule, special residue,
and checkpoint format remain unchanged. Each accepted step satisfies the same
fixed-point predicate as before; a failure is latched across the complete
batch, restored, and independently replayed with adaptive normalization.

The first WSL `sm_89` regression completed 27 runs: a self-test, eight small
complete `cpp_int` comparisons including 101 and the pseudoprime 46657, twelve
alternating benchmarks, and forced replay/Graph-disabled/negacyclic-disabled
compatibility pairs. All six benchmark and three compatibility checkpoint
pairs were byte-identical. An independent integer model checked 19,600 old/new
fixed-point predicates; all matched, with 15,226 accepted congruent results
and 4,374 properly rejected unconverged states.

For three alternating 30,000-bit prefixes per mode, with NTT blocks fixed at
96 on an RTX 4060 Laptop GPU and unrelated CPU PRST load:

| b, n | GFPS 4.1 median | Fused carry median | Time reduction |
| --- | ---: | ---: | ---: |
| 1814570322984178, 16 (four primes) | 3.360 s | 2.631 s | 21.7% |
| 100000000, 16 (three primes) | 3.115 s | 2.029 s | 34.9% |

The complete million-digit `1814570322984178^65536+1` finished all 3,321,925
exponent bits in 265.811 s (266.435 s process wall time), returning base-2 PRP.
Five anchors at 0, 1,000,000, 2,000,000, 3,000,000, and the final bit were
byte-identical to the validated GFPS 4.1 run. Final checkpoint SHA-256:

`61a52137498e8729dba4fce2a5aec0b4e5fabead91a8c1e671edcd1980da8ce8`

The earlier full 4.1 time was 356.581 s on another date; the apparent 25.5%
time reduction is not a controlled same-session speed comparison. The
alternating prefix measurements above provide the closer same-condition
comparison. The fusion applies only to checked batches at 100% duty; these
speedups should not be assumed for duty-throttled or reference execution.

## GFPPS 1.0

The arithmetic implementation was validated before its 1.0 banner promotion
on both Windows and Linux. Each platform passed 112 regression cases covering
factorial/primorial inputs, both signs, 64-bit k, witnesses, pseudoprimes,
Graph/reference modes, Unicode paths, checkpoint integrity, and cross-resume.
Actual signal interruption saved a consistent state whose complete Montgomery
residue was checked independently with GMP. CUDA memcheck reported zero
errors and zero device-memory leaks on the tested path.

After changing the default progress interval to 100,000 bits, seven additional
cases per platform checked the new interval, custom intervals, the user's
100,000-digit examples, older checkpoints, and corrupt-state rejection. The
arithmetic and checkpoint format were not changed by that logging update.

All 29 known primorial samples returned residue 1 on both Windows and Linux:
58 completed GPU runs, 29 byte-identical cross-platform checkpoint pairs,
and 29 independent full GMP repetitions. The largest was
`4599280*104561#+1`, 45,259 digits: 16.025 s Linux / 16.368 s Windows for
exponentiation. [Per-sample results](../GFPPS/KNOWN_PRIMORIAL_RESULTS.md) include
the verified interpretation of the expressions. These results are not newly
generated deterministic primality certificates.

Million-digit tests of `13*210000!-1` and `13*2300000#+1` ran 16,384-bit
prefixes in 8.685 s and 8.578 s respectively. The initial 1024-bit residues
were independently checked with GMP. **The roughly 29–30 minute full-chain
estimates are extrapolations; neither million-digit case completed the full
PRP test in that validation round.** No claim of superiority to another
prime-search program is made without a same-input, same-condition comparison.

## GFPS 4.1

- An initial checker regression contained 108 subprocess runs, 33 complete
  checkpoint comparisons, and four real interruptions across the two checkers.
- A full Linux check of the known PRP `1814570322984178^65536+1` finished in
  356.581 seconds. The final checkpoint SHA-256 was
  `61a52137498e8729dba4fce2a5aec0b4e5fabead91a8c1e671edcd1980da8ce8`.
- Windows resumed that calculation at 2,000,000 of 3,321,925 exponent bits and
  completed in 149.201 seconds. Its final checkpoint was byte-identical to the
  uninterrupted Linux result. This is a cross-platform resume measurement,
  not a complete Windows runtime measurement.
- After the Windows duty-wait adjustment, an 8,192-bit prefix took
  3.546 seconds at 100% duty and 7.421 seconds at 50%; the checkpoints matched.
  The final build's GPU SASS matched the preceding checked build, and 1% duty
  interruption/resume also passed.
- Tests exercised three/four-prime arithmetic, the even-base special residue,
  checkpoint corruption rejection, optimized/reference comparisons, and
  checked-batch replay.

The faster path is the default. Use `--reference-mode` or `--batch-bits 0` for
additional comparisons. These carry-convergence checks are not a Gerbicz proof
or a general detector of all possible hardware faults.

## GSRPS 2.3

Prefixes and bidirectional Windows/Linux checkpoint resumes matched for bases
32500, 1337, and 799. Small independent integer models, boundary cases,
interrupts, corrupted checkpoints, and complete known PRPs were exercised.
A complete Windows check of `11127*32500^32500+1` returned the expected
`result=PRP` and `checksum=1:1`.

Weighted scans and compact carry are conditional optimizations; unsupported
shapes retain the exact general reduction. Timing improvements depend on the
radix, transform length, fixed divisor, and background workload.

## GFNSV CUDA 1.0 arithmetic

Before resume support was added, 56 comparisons covered 14 cases with both
paired-root and full-root kernels on Windows and Linux. Every survivor list
matched the archived CPU implementation. Eight cases additionally matched
independent Python integer enumeration. Coverage included `n=1..6`, keeping
the prime 65537, bases near `UINT64_MAX`, constructed 32/40/50/61-bit factors,
and the GFN16 interval through `10^12`. Default CPU factor verification was
enabled in each run.

The 345,000-candidate GFN16 test retained 157,019 bases. All four GPU survivor
files and the CPU reference had SHA-256
`023df3c83f72d939a09c9f53bb246371c139d41dde29360c2a52e72c74870c33`.

## GFNSV saved-state continuation

The resume/security matrix, including the final atomic-signal rebuild,
passed 95 CLI executions and 42 independent full-state validations. It covered:

- 24 small cases (`n=1..6`, paired/full roots, both operating systems), checked
  against independent Python integer calculations.
- Real repeated signals: 24 Windows `CTRL_BREAK` events produced a saved
  checkpoint and exit in 0.75 seconds; 30 Linux `SIGINT` events took
  0.988 seconds. No traceback or skipped work was observed.
- Windows-to-Linux and Linux-to-Windows resumes through `10^13`, each retaining
  the same 144,864 candidates as the uninterrupted baseline.
- Periodic checkpoint creation followed by a forced process kill and resume.
- A complete 345,000-candidate GFN16 interval through `10^12`, retaining the
  expected 157,019 bases.
- Unicode paths, rejection of an unrelated existing factor log, corrupted or
  truncated states, and incompatible parameters. Rejected resumes preserved
  the original input file.

After the initial 80-run matrix, the host stop flag was changed from volatile storage to a
lock-free `std::atomic<int>` with a compile-time lock-free assertion. Windows
console handlers run on another thread, so the atomic removes a data race
between that handler and the main loop. CUDA kernels and checkpoint format
were unchanged. The final rebuild then passed another 15 CLI executions and
seven independent full-state validations: 26 Windows `CTRL_BREAK` events
saved and exited with status 130 in 0.797 seconds, and 30 Linux `SIGINT` events
did so in 1.150 seconds. Both cross-platform resume directions and the
periodic-save/forced-kill/resume test retained the expected 144,864 survivors
through `10^13`. A completed no-work cross-platform resume preserved the file
SHA-256 exactly.

State files preserve the completed prime frontier, survivors, and previously
found factors together. Tests compare candidate rows separately from factor
discovery order, and exercise completed-state extension to a larger `pmax`.
The repository smoke scripts include a small full-root/paired-root comparison
with completed-state extension. Release validation also includes the longer
interruption and large-interval tests above.
