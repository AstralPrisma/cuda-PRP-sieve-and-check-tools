// SPDX-License-Identifier: GPL-2.0-or-later
// GNCWSV's 20%-smaller workset trigger, adapted to GHCWSV's already compact n list.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ghcw_work {
class Batching {
public:
    static constexpr std::size_t term_budget = 2000000;
    Batching(std::uint64_t limit, std::size_t n_values, bool adaptive)
        : limit_(limit), basis_(n_values), adaptive_(adaptive) {
        if (!limit || !n_values) throw std::invalid_argument("nonempty batch/workset required");
        selected_ = choose(n_values);
    }
    // Call only after all launches and CPU factor validation for a chunk finish.
    bool refresh(std::size_t n_values) {
        if (!adaptive_ || !n_values || n_values > basis_ - (basis_ + 4) / 5) return false;
        basis_ = n_values;
        selected_ = choose(n_values);
        return true;
    }
    std::uint64_t selected() const { return selected_; }
    std::uint64_t producer_batch() const { return adaptive_ ? limit_ : selected_; }
    std::size_t basis() const { return basis_; }
private:
    std::uint64_t choose(std::size_t n_values) const {
        return std::min(limit_, std::max<std::uint64_t>(1, term_budget / n_values));
    }
    std::uint64_t limit_, selected_;
    std::size_t basis_;
    bool adaptive_;
};

// Split one immutable producer batch without discarding its unused tail.
inline std::size_t next_chunk(std::size_t total, std::size_t offset, std::uint64_t limit) {
    if (offset > total || !limit) throw std::invalid_argument("invalid producer cursor");
    return static_cast<std::size_t>(std::min<std::uint64_t>(total - offset, limit));
}
}
