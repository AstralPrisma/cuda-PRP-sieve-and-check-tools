#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tool_dir="$(cd -- "${script_dir}/.." && pwd)"
src_dir="${tool_dir}/src"
out_dir="${1:-${tool_dir}/build}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then architectures=("$@"); else architectures=(sm_86 sm_89 sm_100 sm_120); fi
if [[ -n "${NVCC:-}" ]]; then
  nvcc_bin="${NVCC}"
elif command -v nvcc >/dev/null 2>&1; then
  nvcc_bin="$(command -v nvcc)"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  nvcc_bin="/usr/local/cuda/bin/nvcc"
else
  printf 'nvcc not found; set NVCC to its full path\n' >&2
  exit 127
fi
include_args=()
if [[ -n "${BOOST_ROOT:-}" ]]; then
  [[ -f "${BOOST_ROOT}/boost/multiprecision/cpp_int.hpp" ]] || {
    printf 'BOOST_ROOT must contain boost/multiprecision/cpp_int.hpp\n' >&2
    exit 1
  }
  boost_relative="$(realpath --relative-to="${src_dir}" "${BOOST_ROOT}")"
  include_args=("-I${boost_relative}")
fi
mkdir -p -- "${out_dir}"
out_dir="$(cd -- "${out_dir}" && pwd)"
# Compile from src with a bare input filename; do not embed maintainer paths.
cd -- "${src_dir}"
for arch in "${architectures[@]}"; do
  [[ "${arch}" =~ ^sm_[0-9]+$ ]] || { printf 'Invalid architecture: %s\n' "${arch}" >&2; exit 1; }
  "${nvcc_bin}" -O3 -std=c++17 --threads 1 -arch="${arch}" \
    --default-stream per-thread "${include_args[@]}" \
    GFPPS.cu -o "${out_dir}/GFPPS_${arch}"
done
