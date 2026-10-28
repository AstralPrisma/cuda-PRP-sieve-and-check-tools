// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <vector>
namespace cssv {
using u64=uint64_t;
constexpr u64 MAX_P=(u64(1)<<62)-1, NONE=UINT64_MAX;
struct Config {u64 start=0,end=0,terms=0,pmin=2,pmax=100000; uint32_t step=0; int sign=0; bool both=false,reference=false;};
struct Snapshot {Config cfg; u64 frontier=1; std::vector<u64> factors;};
}
