#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tool_dir="$(cd -- "${script_dir}/.." && pwd)"
src="${tool_dir}/src/GNCWSV.cu"
out_dir="${1:-${tool_dir}/build}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then
  architectures=("$@")
else
  architectures=(sm_86 sm_89 sm_100 sm_120)
fi
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

mkdir -p "${out_dir}"
out_dir="$(cd -- "${out_dir}" && pwd)"
cd "${tool_dir}"
for arch in "${architectures[@]}"; do
  "${nvcc_bin}" -O3 -std=c++17 -arch="${arch}" \
    -Xcompiler=-pthread -ldl \
    -o "${out_dir}/GNCWSV_${arch}" src/GNCWSV.cu
done
