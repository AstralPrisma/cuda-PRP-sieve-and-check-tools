# GFNSV CUDA 1.1

GFNSV is an integer-only GPU sieve for generalized Fermat candidates
`N = b^(2^n) + 1`. The argument `n` is the power-of-two index: `-n16`
means exponent 65536. It searches possible prime factors congruent to
`1 mod 2^(n+1)` across an interval of even bases. Survivors still require PRP
testing and independent verification; sieving is not a primality proof.

Version 1.1 saves candidates and continuation state in **one self-contained
file**. Default GFN output uses format version 4; no `.sieve-state` companion
or factor log is required. Program and file-format versions are independent.

中文：单个 `.gfn` 文件即可中断续筛和跨 Windows/Linux 恢复。`-O` 因子日志
完全可选；不要删除或修改文件中的元数据、候选行和 SHA-256 校验末行。

## Quick start

From an extracted Linux release directory:

```bash
./bin/GFNSV_sm_89 -n16 -b1000000 -B1100000 -P1e12 -o result.gfn
./bin/GFNSV_sm_89 -i result.gfn
./bin/GFNSV_sm_89 -i result.gfn -P1e13
./bin/GFNSV_sm_89 --checkpoint-info result.gfn
```

On Windows, replace `./bin/GFNSV_sm_89` with `bin\GFNSV_sm_89.exe`. Select the
binary for your GPU. Only `sm_89` has matching-device runtime validation on
the development RTX 4060; `sm_86`, `sm_100`, and `sm_120` are cross-compiled.

Both base bounds and factor bounds are inclusive. Only even `b >= 2` are
considered. Default `n` is 16 and default lower factor bound is 3. Decimal,
hexadecimal, underscores, and integer scientific notation such as `1e12` are
accepted. Fresh output paths must not already exist.

Press Ctrl+C once, then wait for `reason=interrupt` and exit status 130. GFNSV
finishes the current GPU batch and atomically saves its completed frontier.
Resume an interrupted file with `-i`; a completed file needs a larger `-P`
for new work. Use `-i result.gfn -P1e13 -o deeper.gfn` to preserve the original
and save separately. `--resume result.gfn` and `--resume --out result.gfn`
remain supported. `--checkpoint-info` validates without GPU work.

The main short/long option pairs are:

```text
-n / --n               GFN index
-b / --bmin            Inclusive minimum base
-B / --bmax            Inclusive maximum base
-p / --pmin            Inclusive minimum factor
-P / --pmax            Inclusive maximum factor
-i / --inputterms      Saved file to resume
-o / --outputterms     Output file; --out is also supported
-f / --format          G, A, base, or expr
-O / --outputfactors   Optional factor log; --factors is also supported
-w / --batch           AP candidates per GPU batch
```

## Self-contained output and recovery

The default `-f G` output has this structure:

```text
GFN n=16 // format=4 Sieved to P
#GFNSV_COMPLETE version=4 engine=cuda n=16 ... format=G
<one surviving even base per line>
#GFNSV_END count=<survivor count> sha256=<digest>
```

This schematic shows a completed file and omits fields; actual files contain
full metadata. Partial checkpoints use `#GFNSV_STATE` instead. The marker must
agree with the stored frontier and target bound; inconsistent states are
rejected.

Metadata preserves the original base interval, requested factor bounds, and
first unprocessed factor position. The footer covers the header, metadata,
and ordered survivors using canonical LF text. Counts, ranges, order, bounds,
and SHA-256 are checked before accepting a file. Malformed or truncated v4
data is rejected even if an older companion exists alongside it.

| Option | Header and survivor representation |
| --- | --- |
| `-f G` (default) | `GFN n=16 // format=4 Sieved to P`, then base values. |
| `-f A` | `ABC $a^65536+1 // format=4 Sieved to P`, then base values. |
| `--format base` | Metadata first, then base values. |
| `--format expr` | Metadata first, then `b^65536+1` expressions. |

All four formats are self-contained. Preserve the whole file for recovery;
removing metadata discards trustworthy continuation state. SHA-256 detects
damage and inconsistent edits; it is not a signature or proof that an
untrusted producer performed the sieve correctly. Allow only one writer per
output path.

On resume, explicit `n`, base bounds, and `pmin` must match the stored search;
they cannot silently skip unfinished work. `pmax` cannot move below the
already processed factor interval. GPU device, batch size, and root mode may
change between runs. Windows and Linux use the same file format.

An initial snapshot is saved before GPU work. `--state-every S` saves at
completed-batch boundaries, defaulting to 30 seconds; `0` disables periodic
saves but retains initial, interrupt, and final saves. Process or power
failure may require repeating work since the last successful save. Publication
uses a flushed temporary file and atomic replacement; a failed replacement
preserves the previous checkpoint. Wait for normal interruption to finish.

## Optional factor audit log

```bash
./bin/GFNSV_sm_89 -n16 -b1000000 -B1100000 -P1e12 -o audit.gfn -O factors.txt
./bin/GFNSV_sm_89 -i audit.gfn -P1e13 -O factors.txt
```

`-O`, `--outputfactors`, and `--factors` name the same optional log. Without
them, no log is created. Losing the log does not prevent continuation from
the main file. Default CPU verification checks newly retained factor values
for divisibility and primality before saving. A prime equal to the candidate
itself is not a proper factor. Historical removals in v4 are represented by
the validated survivor set, not a mandatory factor table. `--no-verify`
explicitly disables CPU mathematical verification for controlled diagnosis.

An existing log's checksum, search identity, and factor records are validated
before merging. A log cannot remove a current survivor. Omitted historical
factors cannot be reconstructed from the survivor list: enabling logging later
records available factors and subsequent discoveries. A failed log update
reports an error after preserving the already-saved main checkpoint.

## Progress and efficiency stopping

Progress normally appears about once per second with `survivors`, `elapsed_s`,
and `eta_s`. `--progress-seconds S` changes the interval (minimum 0.1 seconds).
Updates occur at complete batch boundaries, so long batches can delay output.
Fixed-`-P` ETA estimates use this run's completed arithmetic-progression
candidates, excluding downtime before resume. `-q` / `--quiet` hides progress
while keeping checkpoint and final messages.

Supply all three positive finite limits to stop on low removal efficiency:

```bash
./bin/GFNSV_sm_89 -n16 -b1000000 -B1100000 -o efficiency.gfn -4 60 -5 5 -6 10
```

- `-4 S` / `--max-factor-seconds S`: stop after more than S seconds without
  a new removal, including before the full window is ready.
- `-5 S` / `--max-average-factor-seconds S` / `--spftarget S`: after the full
  rolling window is ready, stop if average seconds per new removal exceed S.
- `-6 M` / `--efficiency-window-minutes M` / `--minutesforspf M`: rolling
  window length in minutes. Removals, not repeated factors, are counted.

Efficiency limits force `pmax=2^62-1`, overriding `-P`; ETA becomes
`n/a(efficiency-stop)`. A normal efficiency stop saves the completed prefix as
the file's finished sieve depth. Continue with a higher `-P`, or pass all three
limits again. Limits are not stored in the file; timers reset on resume.

## Offline conversion to GFPS task files

Both platform archives include `scripts/gfnsv_to_cands.py` and its sibling
`scripts/gfnsv_queue.py`. Keep both files together. They need Python 3.10+
and only the standard library; no GPU, CUDA, network, or credentials.

```bash
python3 scripts/gfnsv_to_cands.py result.gfn --out-dir cands_new --dry-run
python3 scripts/gfnsv_to_cands.py result.gfn --out-dir cands_new
```

On Windows use `python`. Conversion requires a complete, valid snapshot and
defaults to `--expected-n 16`. For another n, pass `--expected-n N` and use a
separate, matching project and output directory. The script reads one validated
snapshot without modifying it and never uploads tasks. Dry-run creates no
output directory, files, or import caches.

Each survivor becomes `cand_<b>.txt` containing one base and an LF newline.
`.gfn-tasks.json` binds the directory to one n. Byte-identical existing tasks
are skipped; conflicting content, unrelated old `cand_*.txt`, missing or
incompatible manifests, and symlink/reparse targets are rejected before task
writes. Use a new empty directory after deepening a sieve. No user files are
deleted. Partial failure can leave complete installed tasks that a retry skips.
Allow only one writer per output directory, including on filesystems whose
atomic-install fallback lacks hard-link support.

The JSON summary reports counts, n, sieve depth, and state digest; it is not a
candidate task. Conversion performs no PRP checks. Some PRP clients separately
back up a full snapshot before consuming a plain queue. That consumption backup
is unrelated to GFNSV single-file recovery and is not made by this converter.

## Legacy migration

Full CUDA v3 checkpoints remain readable. Old bare GFN/ABC exports require
their original checksum-validated v3 `<file>.sieve-state` companion for
migration; the export must contain all survivors of that state. Plain sparse
lists, consumed plain PRP queues, and legacy CPU checkpoints are not safe
recovery inputs.

```bash
./bin/GFNSV_sm_89 -i old.gfn -o upgraded.gfn
./bin/GFNSV_sm_89 -i old_v3.txt -f G -o upgraded.gfn
./bin/GFNSV_sm_89 --checkpoint-info upgraded.gfn
```

Once a saved `format=4` file has passed validation, its old companion is no
longer read and may be archived. Keep migration inputs until the new standalone
file is checked. New-format failures never fall back to legacy state. The
converter also accepts complete v3 files and validated old GFN/ABC-plus-v3
pairs solely for migration.

## Build and limits

From the source repository's `GFNSV` directory:

```bash
./scripts/build-linux.sh build sm_89
```

```bat
scripts\build-windows.bat build sm_89
```

Use a CUDA toolkit supporting the target and a C++17 host compiler. Windows
requires a CUDA-supported x64 MSVC toolchain and `BOOST_ROOT` pointing to the
directory containing `boost/` for host-side 128-bit arithmetic. There is no
primesieve runtime dependency. Keep all six source files together in `src/`:
`GFNSV.cu`, `gfnsv_state.hpp`, `gfnsv_compact.hpp`, `gfnsv_output.hpp`,
`gfnsv_efficiency.hpp`, and `console_utf8.hpp`. Omit the architecture argument
to build all four release targets.

- Supported `n=1..20`, `3 <= pmin <= pmax <= 2^62-1`.
- At most `64 * 1024 * 1024` original even-base slots in a contiguous interval;
  the CUDA sieve still needs memory for the original interval on resume.
- `-w` / `--batch` selects 1..1048576 possible AP factors per batch, default
  65536; it is neither the number of primes nor the number of bases.
- `--device N` selects a CUDA device, default 0. `--root-pairs` is the default;
  `--full-roots` is the independent enumeration mode for comparisons.

Root modes must agree on survivors; concurrent discovery can choose different
valid first factors. Compare survivors and independently validate factors
instead of requiring identical factor discovery order.

The 1.1 validation includes 47 CLI cases, 141 C++ compact-codec checks,
cross-language fixtures, Python tests, and real Windows/Linux interruptions
followed by standalone cross-platform resume. Only `sm_89` GPU execution is
claimed. See the repository's
[validation notes](https://github.com/AstralPrisma/cuda-PRP-sieve-and-check-tools/blob/v2026.09.7/docs/validation-2026-09.md).
Portable CPU test instructions are in [TESTING.md](TESTING.md).
Run the binary's `--help` for its authoritative command reference.
