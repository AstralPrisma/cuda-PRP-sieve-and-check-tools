// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace gfpdiv_cuda {
// Groups are in descending binary-weight order. For every group:
// A <- A^2 * product(group); this is an exact product representation.
using Groups = std::vector<std::vector<uint64_t>>;
class Backend {
public:
    Backend(const Groups&, int device, int chains, uint32_t tile_chunks);
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    // False means an interrupted, uncommitted batch; output must not be used.
    bool remainders(const std::vector<uint64_t>& primes,
                    std::vector<uint64_t>& output,
                    const std::function<bool()>& should_stop);
    double stream_seconds() const;
    std::string tuning_identity() const;
    void reset_timing();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
int selftest();
}
