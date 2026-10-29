# Correctness and release coverage

The factor-removal backend uses integer Montgomery products with an explicit
`p<=2^62-1` bound. Exact host reconstruction checks the product encoding before
upload, and every proposed factor is independently checked before candidate removal.
No floating-point residue approximation or unchecked deletion path is used.

Release validation distinguishes native architecture compilation/CPU-only help
startup from real CUDA execution. RTX4060 Laptop (sm_89) is the physical test
device for Windows and WSL; no RTX30/50-series or Blackwell hardware claim is made.

Checks include independent integer modular residues (including full64-bit
operands and odd moduli near2^62), complete small factorial/primorial searches,
both signs and single signs, packed-product variants, batch tails, scientific
notation, checksummed continuation, malformed-state rejection, cancellation,
safe interruption and offline conversion. Detailed release evidence records
the exact executable hashes and test counts.

Run `python tests/test_cuda.py --binary /path/to/GFPDIVSV_sm_89` on a matching
GPU after pausing other GPU workloads. This is a correctness test, not a benchmark.
No tests upload tasks, use account credentials or alter production data.

Version1.1 also validates32 prime-sequence comparisons across legacy arrays,
persistent iterators and ordered producer queues, plus12 deterministic supplier
policy checks (starvation, minimum3% gain, rollback, freeze and resource cap).
`tests/test_auto.py` checks cold/warm profiles, resource-key invalidation,
corrupt or mismatched profile rejection, unrelated-file preservation, manual
overrides, prime endpoints and safe interruption before/during calibration.
These tests run on both Windows and WSL sm_89 release binaries; their exact
hashes are recorded in`validation/AUTO_WINDOWS.json` and`AUTO_LINUX.json`.

Calibration operates on disposable states and compares complete residue streams
and survivor masks across batch choices. Every proposed factor remains subject
to independent exact verification. Real task progress is committed only after
a full batch completes; prefetched/generated ranges do not constitute progress.
