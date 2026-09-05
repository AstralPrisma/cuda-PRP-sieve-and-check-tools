# September 2026 local validation

This record covers the GFPS 4.1 and GSRPS 2.3 arithmetic promotion and the
GFNSV CUDA 1.0 sieve. It describes tested cases, not a guarantee that every
parameter, device, or hardware execution is free of defects.

## Hardware and build coverage

The runtime test device was an NVIDIA GeForce RTX 4060 Laptop GPU (`sm_89`).
Windows x64 and Linux/WSL x86-64 builds use CUDA 13.3, with MSVC and GCC host
compilers respectively. `sm_86`, `sm_100`, and `sm_120` binaries were
cross-compiled and their cubin targets checked; they were not executed on
matching hardware.

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
