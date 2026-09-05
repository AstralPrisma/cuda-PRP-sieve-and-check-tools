#!/usr/bin/env bash
set -euo pipefail

bin_dir="${1:-release-assets}"
arch="${2:-sm_89}"
bin_dir="$(cd -- "${bin_dir}" && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT

"${bin_dir}/GFPS_${arch}" --selftest
"${bin_dir}/GSRPS_${arch}" --selftest

"${bin_dir}/GFNSV_${arch}" \
  --n 3 --bmin 2 --bmax 100 --pmax 100 --batch 16 \
  --out "${scratch}/gfnsv-resume.txt" --quiet
"${bin_dir}/GFNSV_${arch}" \
  --resume --out "${scratch}/gfnsv-resume.txt" --pmax 1000 --full-roots --quiet
"${bin_dir}/GFNSV_${arch}" \
  --n 3 --bmin 2 --bmax 100 --pmax 1000 \
  --out "${scratch}/gfnsv-direct.txt" --quiet
awk 'NF && $0 !~ /^#/' "${scratch}/gfnsv-resume.txt" > "${scratch}/gfnsv-resume-bases.txt"
awk 'NF && $0 !~ /^#/' "${scratch}/gfnsv-direct.txt" > "${scratch}/gfnsv-direct-bases.txt"
cmp "${scratch}/gfnsv-resume-bases.txt" "${scratch}/gfnsv-direct-bases.txt"
"${bin_dir}/GFNSV_${arch}" --checkpoint-info "${scratch}/gfnsv-resume.txt"

"${bin_dir}/GSRSV_${arch}" \
  --kmin 1 --kmax 100 --base 2 --exp 100 --termtype 1 \
  --pmin 2 --pmax 10000 --cpu-small-prime 2 --prime-generator segmented \
  --outputterms "${scratch}/gsrsv.txt" --verify --quiet
test -f "${scratch}/gsrsv.txt"

"${bin_dir}/GNCWSV_${arch}" \
  --amin 2 --amax 40 --base 3 --mode 1 \
  --pmin 2 --pmax 1000 --outputterms "${scratch}/gncwsv.txt" --verify --quiet
test -s "${scratch}/gncwsv.txt"

printf 'smoke tests passed for %s\n' "${arch}"
