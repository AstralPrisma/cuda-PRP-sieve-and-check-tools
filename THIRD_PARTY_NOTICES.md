# Third-party notices

This file summarizes third-party components and upstream work referenced or
used by this repository. It does not replace the corresponding upstream license
texts.

## mtsieve / twinsieve

GSRSV is a CUDA port/reimplementation derived from the GPLv2-or-later mtsieve
twinsieve code. Its source records the original CPU TwinApp/TwinWorker copyright
as:

```text
Copyright (C) Mark Rodenkirch, 2018
```

GNCWSV derives prime-generation and CUDA infrastructure from the same
mtsieve-based GSRSV implementation. These portions are distributed under the
GNU General Public License, version 2 or later.

Upstream project: <https://github.com/primesearch/mtsieve>

## primesieve

GSRSV and GNCWSV can optionally load the primesieve iterator API at runtime for
prime generation. The programs also provide a built-in generator and do not
require primesieve to start.

primesieve is distributed under the BSD 2-Clause license. A copy is included at
[`LICENSES/primesieve-BSD-2-Clause.txt`](LICENSES/primesieve-BSD-2-Clause.txt).

Upstream project: <https://github.com/kimwalisch/primesieve>

## Boost.Multiprecision

GFPS and GSRPS use Boost.Multiprecision headers for host-side exact-integer
operations and verification paths. Boost is distributed under the Boost
Software License, version 1.0.

License: <https://www.boost.org/LICENSE_1_0.txt>

A copy is included at [`LICENSES/BSL-1.0.txt`](LICENSES/BSL-1.0.txt).

## CUB / NVIDIA CCCL

GSRPS uses CUB headers supplied with the CUDA toolkit. CUB is part of NVIDIA's
CUDA Core Compute Libraries (CCCL) and is distributed under the BSD 3-Clause
license. Consult the exact CUDA/CCCL version used for a release for its bundled
copyright and license notices.

A copy of the CUB notice used for this distribution is included at
[`LICENSES/CUB-BSD-3-Clause.txt`](LICENSES/CUB-BSD-3-Clause.txt).

Upstream project: <https://github.com/NVIDIA/cccl>

## NVIDIA CUDA Toolkit and runtime

All four tools require NVIDIA CUDA to build and an appropriate NVIDIA driver
and GPU to execute. The CUDA toolkit, headers, compiler, driver, and any
redistributable runtime components remain subject to NVIDIA's licenses and EULA.
They are not relicensed by this repository's GPL license.

Release maintainers are responsible for verifying that packaged CUDA runtime
components, if any, are permitted redistributables under the toolkit version
used to build that release.

## Related prior art and validation tools

Development and validation of the checker family was informed by existing
prime-search software, including PRST and llrCUDA. Those projects are not
vendored in this repository; their copyrights and licenses remain with their
authors.

- PRST: <https://github.com/AenBleidd/rebirther-prst>
- llrCUDA mirror: <https://github.com/primesearch/llrCUDA>

GFPS contains a self-contained SHA-256 implementation used to authenticate
checkpoint metadata and digits. It implements the standardized SHA-256
algorithm; a cryptographic standard is not itself a substitute for a primality
certificate.

## License relationship

The repository as a whole is distributed under `GPL-2.0-or-later`. Third-party
copyright notices and license obligations remain in effect. When upstream
license text or attribution requirements differ from this summary, the
upstream license text controls.
