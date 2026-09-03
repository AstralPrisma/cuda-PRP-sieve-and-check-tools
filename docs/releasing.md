# Releasing

The repository stores source code and documentation. Linux binaries belong in
GitHub Releases and must not be committed to the main Git history.

## Versioning

The four tools have independent component versions:

| Tool | Current component version |
| --- | ---: |
| GFPS | 4.0 |
| GSRPS | 2.0 |
| GSRSV | 2.0 |
| GNCWSV | 1.0 |

Use a suite tag such as `v2026.09.0` for a coordinated repository release and
list all four component versions in its notes. Increment the final field for a
rebuild or packaging correction that does not change every component.

## Pre-release checklist

1. Start from a clean tagged commit.
2. Confirm the component version printed by each executable matches the release
   notes.
3. Confirm all source files carry the intended `GPL-2.0-or-later` notice and
   preserve upstream attribution.
4. Search for credentials, private service URLs, checkpoints, candidate files,
   logs, tuning caches, and absolute local paths.
5. Run the full matrix in [correctness.md](correctness.md).
6. Build in a pinned environment and record:
   - source commit and source SHA-256;
   - CUDA toolkit and nvcc versions;
   - host compiler and operating-system image;
   - target SM;
   - complete compiler flags;
   - whether the artifact was runtime-tested on matching hardware.
7. Inspect dynamic dependencies and required `GLIBC`, `GLIBCXX`, and `CXXABI`
   symbol versions. Build in the oldest supported pinned environment rather
   than silently requiring the maintainer's current distribution.
8. Verify the embedded cubin target with `cuobjdump --list-elf`.
9. Generate a manifest and SHA-256 checksums after packaging.

## Build matrix

The intended release architecture names are:

```text
sm_86
sm_89
sm_100
sm_120
```

Current prepared coverage is:

| Tool | `sm_86` | `sm_89` | `sm_100` | `sm_120` |
| --- | :---: | :---: | :---: | :---: |
| GFPS | built | built | built | built |
| GSRPS | built | built | built | built |
| GSRSV | built | built | built | built |
| GNCWSV | built | built | built | built |

Do not fill a missing cell by renaming an artifact from another target. Either
cross-compile and validate the requested target or leave it absent and document
the limitation.

GitHub-hosted runners can compile CUDA code without a GPU, but that proves only
the `built` status. A self-hosted matching GPU or a recorded external tester is
required for runtime status.

## Artifact naming

Use lowercase, explicit, sortable names. A component package contains all four
native SM variants for one operating system:

```text
gfps-4.0-linux-x86_64-cuda13.3.tar.xz
gfps-4.0-windows-x86_64-cuda13.3.zip
gsrps-2.0-linux-x86_64-cuda13.3.tar.xz
gsrps-2.0-windows-x86_64-cuda13.3.zip
```

Replace `cuda13.3` with the actual toolkit used by that release. Do not use a
marketing GPU-family label in place of the SM target.

Each archive should contain a single top-level directory and include:

```text
<tool>-<version>/
  bin/<tool>_sm_86[.exe]
  bin/<tool>_sm_89[.exe]
  bin/<tool>_sm_100[.exe]
  bin/<tool>_sm_120[.exe]
  README.md
  LICENSE
  THIRD_PARTY_NOTICES.md
  BUILDINFO.txt
```

Set executable mode to `0755` before creating a tar archive. Packaging directly
from a Windows-mounted directory can otherwise preserve unsuitable permission
bits. Public builds should omit path-bearing debug/line information and must be
scanned for maintainer-local absolute paths before upload.

The Release should also contain:

```text
SHA256SUMS
manifest.json
cuda-prp-sieve-and-check-tools-v2026.09.0-source.tar.xz
```

GitHub's generated source archives point to the tag, but an explicit source
archive built from the same commit makes binary/source correspondence and GPL
compliance easier to audit.

## Manifest fields

At minimum, `manifest.json` should record for every artifact:

```json
{
  "file": "gfps-4.0-linux-x86_64-cuda13.3.tar.xz",
  "tool": "GFPS",
  "component_version": "4.0",
  "source_commit": "<full commit id>",
  "source_sha256": "<sha256>",
  "artifact_sha256": "<sha256>",
  "cuda_toolkit": "13.3",
  "targets": ["sm_86", "sm_89", "sm_100", "sm_120"],
  "host": "linux-x86_64",
  "runtime_tested_targets": ["sm_89"],
  "test_hardware": "<GPU model>"
}
```

Populate test fields from evidence; never turn `runtime_tested` on merely
because compilation succeeded.

## Release verification

Before uploading, unpack every archive into a fresh directory and run:

```bash
sha256sum -c SHA256SUMS
file <executable>
ldd <executable>
cuobjdump --list-elf <executable>
```

Then run the applicable `--version`, help, self-test, known-checksum, and small
sieve tests. Confirm that the archive contains no generated checkpoint, factor,
candidate, result, cache, or log file.

## Publishing with GitHub CLI

After review, signing, and checksum verification, a maintainer may create the
release with a command shaped like:

```bash
gh auth status
gh release create v2026.09.0 \
  --title 'CUDA PRP and sieve tools 2026.09.0' \
  --notes-file release-notes.md \
  release-assets/*
```

This is an example, not an instruction to publish an unreviewed working tree.
After upload, download the assets again and repeat checksum and archive-content
verification.
