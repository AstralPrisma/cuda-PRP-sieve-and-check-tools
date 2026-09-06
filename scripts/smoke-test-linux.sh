#!/usr/bin/env bash
set -euo pipefail

bin_dir="${1:-release-assets}"
arch="${2:-sm_89}"
bin_dir="$(cd -- "${bin_dir}" && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT

"${bin_dir}/GFPS_${arch}" --selftest
"${bin_dir}/GSRPS_${arch}" --selftest

# GFPPS has explicit cpp_int reference checks rather than a --selftest mode.
"${bin_dir}/GFPPS_${arch}" --check '1*3!+1' --verify-cpp-int | tee "${scratch}/gfpps-prime.log"
grep -q 'verification=cpp_int-match, result=PRP' "${scratch}/gfpps-prime.log"
"${bin_dir}/GFPPS_${arch}" --check '3*5!+1' --no-graphs --verify-cpp-int | tee "${scratch}/gfpps-composite.log"
grep -q 'verification=cpp_int-match, result=COMPOSITE' "${scratch}/gfpps-composite.log"
"${bin_dir}/GFPPS_${arch}" --check '17*31#+1' --max-bits 16 \
  --checkpoint "${scratch}/gfpps.ckpt" --verify-cpp-int | tee "${scratch}/gfpps-partial.log"
grep -q 'verification=cpp_int-match, result=PARTIAL' "${scratch}/gfpps-partial.log"
"${bin_dir}/GFPPS_${arch}" --check '17*31#+1' --checkpoint "${scratch}/gfpps.ckpt" \
  --resume-checkpoint --verify-cpp-int | tee "${scratch}/gfpps-resumed.log"
grep -Eq 'verification=cpp_int-match, result=(PRP|COMPOSITE)' "${scratch}/gfpps-resumed.log"

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
