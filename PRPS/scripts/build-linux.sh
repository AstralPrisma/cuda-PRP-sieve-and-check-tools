#!/usr/bin/env bash
# PRPS is one unified binary: all four SM targets are embedded in a single
# executable and the CUDA driver picks the right one at run time.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tool_dir="$(cd -- "${script_dir}/.." && pwd)"
src="${tool_dir}/src/PRPS.cu"
out_dir="${1:-${tool_dir}/build}"

if [[ -n "${NVCC:-}" ]]; then
  nvcc_bin="${NVCC}"
elif command -v nvcc >/dev/null 2>&1; then
  nvcc_bin="nvcc"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  nvcc_bin="/usr/local/cuda/bin/nvcc"
else
  printf 'nvcc not found; set NVCC to its full path\n' >&2
  exit 127
fi
if [[ -z "${BOOST_ROOT:-}" ]]; then
  printf 'set BOOST_ROOT to the directory containing boost/multiprecision/cpp_int.hpp\n' >&2
  exit 1
fi
source_hash="$(sha256sum "${src}" | awk '{print $1}')"

mkdir -p "${out_dir}"
out_dir="$(cd -- "${out_dir}" && pwd)"
cd "${tool_dir}"
"${nvcc_bin}" -O3 -std=c++17 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120 \
  --default-stream per-thread \
  -I"${BOOST_ROOT}" \
  -DPRPS_BUILD_ID="\"${source_hash:0:20}-O3-cxx17-ptds-unified\"" \
  -o "${out_dir}/PRPS" src/PRPS.cu -lgmp -lgmpxx