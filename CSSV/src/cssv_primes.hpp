// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "cssv_common.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif
#else
#include <dlfcn.h>
#if defined(__linux__)
#include <unistd.h>
#endif
#endif

namespace cssv {

// Independent host arithmetic, also used when validating emitted factors.
// The fallback is overflow-safe for the complete uint64_t domain.
inline u64 cpu_addmod(u64 a, u64 b, u64 m) {
    return a >= m - b ? a - (m - b) : a + b;
}

inline u64 cpu_mulmod(u64 a, u64 b, u64 m) {
    if (m == 0) throw std::invalid_argument("modular multiplication by modulus zero");
    a %= m;
    b %= m;
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned __int64 high = 0, remainder = 0;
    const unsigned __int64 low = _umul128(a, b, &high);
    // a,b < m ensures high < m, so the quotient fits in 64 bits.
    (void)_udiv128(high, low, m, &remainder);
    return remainder;
#elif defined(__SIZEOF_INT128__)
    return static_cast<u64>((static_cast<unsigned __int128>(a) * b) % m);
#else
    u64 result = 0;
    while (b != 0) {
        if (b & 1) result = cpu_addmod(result, a, m);
        b >>= 1;
        if (b != 0) a = cpu_addmod(a, a, m);
    }
    return result;
#endif
}

inline u64 cpu_powmod(u64 a, u64 e, u64 m) {
    if (m == 0) throw std::invalid_argument("modular exponentiation by modulus zero");
    u64 result = 1 % m;
    a %= m;
    while (e != 0) {
        if (e & 1) result = cpu_mulmod(result, a, m);
        e >>= 1;
        if (e != 0) a = cpu_mulmod(a, a, m);
    }
    return result;
}

// These seven strong probable-prime bases are deterministic for n < 2^64.
inline bool prime64(u64 n) {
    if (n < 2) return false;
    const uint32_t trial[] = {2,3,5,7,11,13,17,19,23,29,31,37};
    for (uint32_t p : trial) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    u64 d = n - 1;
    unsigned s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    const u64 witnesses[] = {2,325,9375,28178,450775,9780504,1795265022};
    for (u64 a : witnesses) {
        if (a % n == 0) continue;
        u64 x = cpu_powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool passed = false;
        for (unsigned r = 1; r < s; ++r) {
            x = cpu_mulmod(x, x, n);
            if (x == n - 1) { passed = true; break; }
        }
        if (!passed) return false;
    }
    return true;
}

namespace prime_detail {

inline u64 integer_sqrt(u64 n) {
    u64 r = static_cast<u64>(std::sqrt(static_cast<long double>(n)));
    while (r != 0 && r > n / r) --r;
    while (r < std::numeric_limits<u64>::max() && r + 1 <= n / (r + 1)) ++r;
    return r;
}

// Public C iterator layout from primesieve 12.x. Never use an unverified ABI.
struct Iterator12 {
    size_t i;
    size_t size;
    uint64_t start;
    uint64_t stop_hint;
    uint64_t* primes;
    void* memory;
    int is_error;
};

class Runtime12 {
public:
    using Init = void (*)(Iterator12*);
    using Free = void (*)(Iterator12*);
    using Jump = void (*)(Iterator12*, uint64_t, uint64_t);
    using Generate = void (*)(Iterator12*);
    Runtime12() = default;
    Runtime12(const Runtime12&) = delete;
    Runtime12& operator=(const Runtime12&) = delete;
    ~Runtime12() { close(); }

    bool load() {
        if (available()) return true;
#if defined(_WIN32)
        wchar_t executable[32768];
        const DWORD size = GetModuleFileNameW(nullptr, executable, 32768);
        if (size > 0 && size < 32768) {
            std::wstring path(executable, size);
            const size_t sep = path.find_last_of(L"\\/");
            if (sep != std::wstring::npos) {
                path.resize(sep + 1);
                path += L"lib\\primesieve.dll";
                if (GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES) {
                    if (!try_load(path, "bundled lib/primesieve.dll", true))
                        throw std::runtime_error("Bundled primesieve: " + error_);
                    return true;
                }
            }
        }
        if (try_load(L"primesieve.dll", "primesieve.dll", false)) return true;
        if (try_load(L"libprimesieve.dll", "libprimesieve.dll", false)) return true;
#else
#if defined(__linux__)
        char executable[4096];
        const auto size = readlink("/proc/self/exe", executable, sizeof(executable));
        if (size > 0 && static_cast<size_t>(size) < sizeof(executable)) {
            std::string path(executable, static_cast<size_t>(size));
            const size_t sep = path.find_last_of('/');
            if (sep != std::string::npos) {
                path.resize(sep + 1);
                path += "lib/libprimesieve.so.12";
                if (access(path.c_str(), F_OK) == 0) {
                    if (!try_load(path)) throw std::runtime_error("Bundled primesieve: " + error_);
                    return true;
                }
            }
        }
#endif
#if defined(__APPLE__)
        if (try_load("libprimesieve.12.dylib")) return true;
        if (try_load("libprimesieve.dylib")) return true;
#else
        if (try_load("libprimesieve.so.12")) return true;
        if (try_load("libprimesieve.so")) return true;
#endif
#endif
        return false;
    }

    bool available() const { return handle_ && init_ && free_ && jump_ && generate_; }
    const std::string& description() const { return description_; }
    const std::string& error() const { return error_; }
    void initialize(Iterator12& it, u64 lo, u64 hi) const {
        std::memset(&it, 0, sizeof(it));
        init_(&it);
        jump_(&it, lo, hi); // Inclusive start, unlike legacy skipto().
    }
    void release(Iterator12& it) const { free_(&it); }
    void refill(Iterator12& it) const {
        generate_(&it);
        if (it.is_error || !it.primes || it.i >= it.size)
            throw std::runtime_error("libprimesieve failed while generating primes");
    }

private:
#if defined(_WIN32)
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
    Init init_ = nullptr;
    Free free_ = nullptr;
    Jump jump_ = nullptr;
    Generate generate_ = nullptr;
    std::string description_, error_;

    template<class T> T symbol(const char* name) {
#if defined(_WIN32)
        return reinterpret_cast<T>(GetProcAddress(handle_, name));
#else
        return reinterpret_cast<T>(dlsym(handle_, name));
#endif
    }
    void close() {
        if (handle_) {
#if defined(_WIN32)
            FreeLibrary(handle_);
#else
            dlclose(handle_);
#endif
        }
        handle_ = nullptr;
        init_ = nullptr; free_ = nullptr; jump_ = nullptr; generate_ = nullptr;
    }
#if defined(_WIN32)
    bool try_load(const std::wstring& path, const std::string& display, bool absolute) {
        // Do not search the current directory for a DLL. The absolute bundled
        // path also makes its directory available for its own dependencies.
        DWORD flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        if (absolute) flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
        handle_ = LoadLibraryExW(path.c_str(), nullptr, flags);
#else
    bool try_load(const std::string& display) {
        handle_ = dlopen(display.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle_) {
            if (error_.empty()) error_ = display + " could not be loaded";
            return false;
        }
        using Version = const char* (*)();
        const auto version_function = symbol<Version>("primesieve_version");
        const char* version = version_function ? version_function() : nullptr;
        if (!version || std::string(version).compare(0, 3, "12.") != 0) {
            error_ = display + " has incompatible iterator ABI (expected 12.x; got " +
                     (version ? version : "unknown") + ")";
            close();
            return false;
        }
        init_ = symbol<Init>("primesieve_init");
        free_ = symbol<Free>("primesieve_free_iterator");
        jump_ = symbol<Jump>("primesieve_jump_to");
        generate_ = symbol<Generate>("primesieve_generate_next_primes");
        if (!available()) {
            error_ = display + " is missing the primesieve 12.x iterator API";
            close();
            return false;
        }
        description_ = display + " (v" + version + ", one iterator)";
        error_.clear();
        return true;
    }
};
} // namespace prime_detail

// Inclusive [lo,hi] prime stream. The caller must commit frontier() only after
// it has completely processed the returned batch. next() may return fewer than
// max_count (even zero) while !done(): bounded fallback scans keep cancellation
// and progress responsive in long prime-free intervals. Such an empty batch
// can advance frontier() over an independently checked composite-only prefix.
class PrimeGenerator {
public:
    PrimeGenerator(u64 lo, u64 hi, const std::string& mode = "auto",
                   const std::atomic<bool>* stop = nullptr)
        : lo_(lo), hi_(hi), frontier_(lo ? lo - 1 : 0), cursor_(std::max<u64>(2, lo)), stop_(stop) {
        if (hi > MAX_P) throw std::invalid_argument("prime bound exceeds 2^62-1");
        if (mode != "auto" && mode != "primesieve" && mode != "segmented" && mode != "mr")
            throw std::invalid_argument("prime-generator must be auto|primesieve|segmented|mr");
        if (lo > hi || hi < 2) { done_ = true; frontier_ = hi; description_ = "empty prime interval"; return; }
        if (mode == "auto" || mode == "primesieve") {
            if (runtime_.load()) {
                method_ = Method::Library;
                runtime_.initialize(iterator_, cursor_, hi_);
                iterator_live_ = true;
                description_ = "libprimesieve: " + runtime_.description();
                return;
            }
            if (mode == "primesieve")
                throw std::runtime_error("primesieve 12.x unavailable: " + runtime_.error());
        }
        if (mode == "segmented" || (mode == "auto" && hi <= SEGMENTED_LIMIT)) {
            if (hi > SEGMENTED_LIMIT)
                throw std::invalid_argument("segmented generator supports pmax<=1e12; install primesieve 12.x or use --prime-generator mr for a narrow interval");
            method_ = Method::Segmented;
            make_base_primes();
            description_ = "bounded segmented Eratosthenes (one thread, 256 KiB segment)";
            if (mode == "auto")
                std::cerr << "WARNING: primesieve 12.x unavailable; using slower bounded segmented generator.\n";
        } else {
            method_ = Method::MillerRabin;
            description_ = "deterministic uint64 Miller-Rabin (one thread, bounded scans)";
            if (mode == "auto")
                std::cerr << "WARNING: primesieve 12.x unavailable and pmax>1e12; using slow bounded Miller-Rabin generation. Install primesieve for deep long-running sieves.\n";
        }
    }
    PrimeGenerator(const PrimeGenerator&) = delete;
    PrimeGenerator& operator=(const PrimeGenerator&) = delete;
    ~PrimeGenerator() { if (iterator_live_) runtime_.release(iterator_); }

    std::vector<u64> next(size_t max_count) {
        if (max_count == 0) throw std::invalid_argument("prime batch size must be positive");
        std::vector<u64> out;
        if (done_ || stopped()) return out;
        // Bound allocation even if the requested batch size is unreasonable.
        const size_t limit = std::min<size_t>(max_count, size_t(1) << 20);
        out.reserve(limit);
        if (method_ == Method::Library) next_library(limit, out);
        else if (method_ == Method::Segmented) next_segmented(limit, out);
        else next_mr(limit, out);
        return out;
    }
    u64 frontier() const { return frontier_; }
    bool done() const { return done_; }
    const std::string& description() const { return description_; }

private:
    enum class Method { Library, Segmented, MillerRabin };
    static constexpr u64 SEGMENTED_LIMIT = UINT64_C(1000000000000);
    static constexpr size_t SEGMENT_ODDS = size_t(1) << 18;
    static constexpr size_t MR_SCAN_BUDGET = 4096;
    u64 lo_, hi_, frontier_, cursor_;
    const std::atomic<bool>* stop_;
    bool done_ = false, iterator_live_ = false;
    Method method_ = Method::MillerRabin;
    std::string description_;
    prime_detail::Runtime12 runtime_;
    prime_detail::Iterator12 iterator_{};
    std::vector<uint32_t> base_primes_;
    std::vector<uint8_t> segment_;
    u64 segment_lo_ = 0, segment_hi_ = 0;
    size_t segment_pos_ = 0;

    bool stopped() const { return stop_ && stop_->load(std::memory_order_relaxed); }
    void finish() { done_ = true; frontier_ = hi_; }

    void next_library(size_t max_count, std::vector<u64>& out) {
        // Match primesieve_next_prime: increment i before consuming. Copy from
        // contiguous library buffers rather than making one indirect call/prime.
        while (out.size() < max_count && !stopped()) {
            ++iterator_.i;
            if (iterator_.i >= iterator_.size) runtime_.refill(iterator_);
            if (iterator_.is_error || !iterator_.primes || iterator_.i >= iterator_.size)
                throw std::runtime_error("libprimesieve returned an invalid prime buffer");
            const u64* begin = iterator_.primes + iterator_.i;
            size_t take = std::min(max_count - out.size(), iterator_.size - iterator_.i);
            const u64* valid_end = std::upper_bound(begin, begin + take, hi_);
            const size_t valid = static_cast<size_t>(valid_end - begin);
            if (valid != 0) {
                if (*begin < lo_ || *begin <= frontier_)
                    throw std::runtime_error("libprimesieve returned nonmonotonic or out-of-range primes");
                out.insert(out.end(), begin, valid_end);
                frontier_ = out.back();
                iterator_.i += valid - 1;
            } else {
                // Nothing consumed; no further library accesses after finish().
                finish();
                return;
            }
            if (valid < take || frontier_ == hi_) { finish(); return; }
        }
    }

    void make_base_primes() {
        const uint32_t root = static_cast<uint32_t>(prime_detail::integer_sqrt(hi_));
        std::vector<bool> composite(static_cast<size_t>(root) + 1, false);
        for (u64 p = 2; p <= root / p; ++p)
            if (!composite[static_cast<size_t>(p)])
                for (u64 j = p * p; j <= root; j += p)
                    composite[static_cast<size_t>(j)] = true;
        for (uint32_t p = 3; p <= root; p += 2)
            if (!composite[p]) base_primes_.push_back(p);
    }

    bool build_segment() {
        if (cursor_ > hi_) { finish(); return false; }
        if ((cursor_ & 1) == 0) ++cursor_;
        if (cursor_ > hi_) { finish(); return false; }
        segment_lo_ = cursor_;
        const u64 width = static_cast<u64>(SEGMENT_ODDS - 1) * 2;
        segment_hi_ = cursor_ + std::min(width, hi_ - cursor_);
        if ((segment_hi_ & 1) == 0) --segment_hi_;
        segment_.assign(static_cast<size_t>((segment_hi_ - segment_lo_) / 2 + 1), 0);
        segment_pos_ = 0;
        for (uint32_t p32 : base_primes_) {
            if (stopped()) {
                // This partial segment is uncommitted and must be rebuilt if
                // the same generator is used again after clearing stop.
                segment_.clear();
                return false;
            }
            const u64 p = p32;
            if (p > segment_hi_ / p) break;
            const u64 square = p * p;
            const u64 remainder = segment_lo_ % p;
            u64 first = segment_lo_ + (remainder ? p - remainder : 0);
            first = std::max(first, square);
            if ((first & 1) == 0) first += p;
            for (u64 j = first; j <= segment_hi_; j += 2 * p)
                segment_[static_cast<size_t>((j - segment_lo_) / 2)] = 1;
        }
        return true;
    }

    void next_segmented(size_t max_count, std::vector<u64>& out) {
        if (cursor_ == 2) {
            out.push_back(2); frontier_ = 2; cursor_ = 3;
            if (hi_ == 2) { finish(); return; }
            if (out.size() == max_count) return;
        }
        if (segment_.empty() || segment_pos_ == segment_.size()) {
            if (!build_segment()) return;
        }
        u64 scanned = frontier_;
        while (segment_pos_ < segment_.size() && out.size() < max_count && !stopped()) {
            const size_t at = segment_pos_++;
            scanned = segment_lo_ + 2 * static_cast<u64>(at);
            if (!segment_[at]) { out.push_back(scanned); frontier_ = scanned; }
            cursor_ = scanned + 2;
        }
        if (cursor_ > hi_) { finish(); return; }
        if (out.empty()) frontier_ = scanned;
        // One segment per call bounds cancellation / progress latency and peak
        // memory. cursor_ may lead frontier_ by a composite-only suffix.
    }

    void next_mr(size_t max_count, std::vector<u64>& out) {
        if (cursor_ == 2) {
            out.push_back(2); frontier_ = 2; cursor_ = 3;
            if (hi_ == 2) { finish(); return; }
            if (out.size() == max_count) return;
        }
        if ((cursor_ & 1) == 0) ++cursor_;
        size_t tested = 0;
        u64 scanned = frontier_;
        while (cursor_ <= hi_ && out.size() < max_count && tested < MR_SCAN_BUDGET && !stopped()) {
            scanned = cursor_;
            if (prime64(cursor_)) { out.push_back(cursor_); frontier_ = cursor_; }
            cursor_ += 2; // hi <= 2^62-1 keeps this safely below UINT64_MAX.
            ++tested;
        }
        if (cursor_ > hi_) finish();
        else if (out.empty()) frontier_ = scanned;
    }
};
} // namespace cssv
