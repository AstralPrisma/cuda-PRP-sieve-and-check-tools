#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
out_dir="${1:-${repo_dir}/release-assets}"
if (( $# > 0 )); then shift; fi
architectures=("$@")

mkdir -p "${out_dir}"
for tool in GFPS GSRPS GFNSV GSRSV GNCWSV; do
  if (( ${#architectures[@]} > 0 )); then
    "${repo_dir}/${tool}/scripts/build-linux.sh" "${out_dir}" "${architectures[@]}"
  else
    "${repo_dir}/${tool}/scripts/build-linux.sh" "${out_dir}"
  fi
done
