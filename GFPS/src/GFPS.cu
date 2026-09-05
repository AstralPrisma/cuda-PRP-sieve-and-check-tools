/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace {

class GfpsInterrupted : public std::runtime_error {
public:
    explicit GfpsInterrupted(const std::string& message)
        : std::runtime_error(message) {}
};

volatile std::sig_atomic_t g_gfps_stop_requested = 0;

void gfps_signal_handler(int) {
    g_gfps_stop_requested = 1;
}

void install_gfps_signal_handlers() {
    g_gfps_stop_requested = 0;
    std::signal(SIGINT, gfps_signal_handler);
#ifdef _WIN32
    std::signal(SIGBREAK, gfps_signal_handler);
#endif
}

struct ModInfo {
    uint32_t p;
    uint32_t g;
};

constexpr ModInfo kMods[] = {
    {998244353u, 3u},
    {1004535809u, 3u},
    {469762049u, 3u},
    {1224736769u, 3u},
};

// Runtime throttling controls. NTT blocks are selected from transform size and
// SM count unless explicitly forced.
//
// Environment variables (backward compatible):
//   GFPS_NTT_BLOCKS=1..4096
//   GFPS_DUTY_PERCENT=1..100
//
// Command-line options (override environment variables and may appear anywhere):
//   --ntt-blocks <1..4096>      or --ntt-blocks=<1..4096>
//   --duty-percent <1..100>     or --duty-percent=<1..100>
constexpr int kDefaultNttBlocks = 64;
constexpr int kMaxNttBlocks = 4096;  // n <= 20: at most 2^20 butterflies / 256 threads.
constexpr int kDefaultDutyPercent = 100;

int parse_bounded_int(const std::string& text, const char* name, int min_value, int max_value) {
    if (text.empty()) {
        throw std::runtime_error(std::string(name) + " requires a value");
    }
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || value < min_value || value > max_value) {
        std::ostringstream msg;
        msg << name << " must be an integer in [" << min_value << ", " << max_value << "]";
        throw std::runtime_error(msg.str());
    }
    return static_cast<int>(value);
}

int read_env_int(const char* name, int default_value, int min_value, int max_value) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') return default_value;
    return parse_bounded_int(text, name, min_value, max_value);
}

struct GpuThrottleConfig {
    int ntt_blocks;
    bool force_ntt_blocks;
    int duty_percent;
};

GpuThrottleConfig& gpu_throttle_config() {
    static GpuThrottleConfig config = [] {
        const char* ntt_text = std::getenv("GFPS_NTT_BLOCKS");
        return GpuThrottleConfig{
            ntt_text != nullptr && *ntt_text != '\0'
                ? parse_bounded_int(ntt_text, "GFPS_NTT_BLOCKS", 1, kMaxNttBlocks)
                : kDefaultNttBlocks,
            ntt_text != nullptr && *ntt_text != '\0',
            read_env_int("GFPS_DUTY_PERCENT", kDefaultDutyPercent, 1, 100),
        };
    }();
    return config;
}

int ntt_block_limit() {
    return gpu_throttle_config().ntt_blocks;
}

bool ntt_blocks_are_forced() {
    return gpu_throttle_config().force_ntt_blocks;
}

int gpu_duty_percent() {
    return gpu_throttle_config().duty_percent;
}

// The optimized resident path is the default. The padded NTT / adaptive-carry
// implementation remains available for independent checkpoint comparisons.
struct ResidentAlgorithmConfig {
    bool reference_mode = std::getenv("GFPS_REFERENCE_MODE") != nullptr;
    int batch_bits = read_env_int("GFPS_BATCH_BITS", 512, 0, 4096);
    bool force_replay = std::getenv("GFPS_FORCE_REPLAY") != nullptr;
};

ResidentAlgorithmConfig& resident_algorithm_config() {
    static ResidentAlgorithmConfig config;
    return config;
}

// Remove global throttle options from argv so the existing mode-specific
// positional argument parser remains unchanged. Command-line values override
// environment variables.
std::vector<char*> consume_gpu_throttle_args(int argc, char** argv) {
    std::vector<char*> filtered;
    filtered.reserve(static_cast<size_t>(argc));
    filtered.push_back(argv[0]);

    auto require_next = [&](int& i, const char* option) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string(option) + " requires a value");
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--ntt-blocks") {
            gpu_throttle_config().ntt_blocks =
                parse_bounded_int(require_next(i, "--ntt-blocks"), "--ntt-blocks", 1, kMaxNttBlocks);
            gpu_throttle_config().force_ntt_blocks = true;
        } else if (arg.rfind("--ntt-blocks=", 0) == 0) {
            gpu_throttle_config().ntt_blocks =
                parse_bounded_int(arg.substr(std::strlen("--ntt-blocks=")), "--ntt-blocks", 1, kMaxNttBlocks);
            gpu_throttle_config().force_ntt_blocks = true;
        } else if (arg == "--duty-percent") {
            gpu_throttle_config().duty_percent =
                parse_bounded_int(require_next(i, "--duty-percent"), "--duty-percent", 1, 100);
        } else if (arg.rfind("--duty-percent=", 0) == 0) {
            gpu_throttle_config().duty_percent =
                parse_bounded_int(arg.substr(std::strlen("--duty-percent=")), "--duty-percent", 1, 100);
        } else if (arg == "--reference-mode") {
            resident_algorithm_config().reference_mode = true;
        } else if (arg == "--batch-bits") {
            resident_algorithm_config().batch_bits =
                parse_bounded_int(require_next(i, "--batch-bits"), "--batch-bits", 0, 4096);
        } else if (arg.rfind("--batch-bits=", 0) == 0) {
            resident_algorithm_config().batch_bits =
                parse_bounded_int(arg.substr(std::strlen("--batch-bits=")), "--batch-bits", 0, 4096);
        } else if (arg == "--diagnostic-force-replay") {
            resident_algorithm_config().force_replay = true;
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

int limited_ntt_grid_y(int work_items, int threads) {
    const int needed = std::max(1, (work_items + threads - 1) / threads);
    return std::min(ntt_block_limit(), needed);
}

#ifdef _WIN32
// The Windows standard-library sleep can round each sub-millisecond duty pause
// up to a full scheduler tick. A private high-resolution one-shot timer avoids
// that slowdown without changing the timer resolution for other processes.
// Windows versions before 10 1803 may reject the flag; keep the existing sleep
// as a compatibility fallback in that case or if a timer operation fails.
class DutySleepTimer {
public:
    DutySleepTimer() noexcept
        : _handle(CreateWaitableTimerExW(nullptr, nullptr,
              kHighResolutionFlag,
              TIMER_MODIFY_STATE | SYNCHRONIZE)) {}

    ~DutySleepTimer() noexcept {
        if (_handle != nullptr) CloseHandle(_handle);
    }

    DutySleepTimer(const DutySleepTimer&) = delete;
    DutySleepTimer& operator=(const DutySleepTimer&) = delete;

    bool sleep_for(std::chrono::nanoseconds duration) noexcept {
        if (_handle == nullptr) return false;
        // Negative due times are relative, in units of 100 ns. Round upward
        // without adding 99 to a potentially very large nanosecond count.
        const int64_t ns = duration.count();
        if (ns <= 0) return true;
        LARGE_INTEGER due{};
        due.QuadPart = -(ns / 100 + (ns % 100 != 0 ? 1 : 0));
        if (SetWaitableTimer(_handle, &due, 0, nullptr, nullptr, FALSE) &&
            WaitForSingleObject(_handle, INFINITE) == WAIT_OBJECT_0) {
            return true;
        }
        // Do not reuse an uncertain timer after an API failure.
        CancelWaitableTimer(_handle);
        CloseHandle(_handle);
        _handle = nullptr;
        return false;
    }

private:
    // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION; using its documented value also
    // compiles when the SDK target macro predates Windows 10 1803. Availability
    // is checked by CreateWaitableTimerExW at runtime, with fallback above.
    static constexpr DWORD kHighResolutionFlag = 0x00000002;
    HANDLE _handle = nullptr;
};
#endif

void sleep_for_gpu_duty(std::chrono::nanoseconds duration) {
    if (duration.count() <= 0) return;
#ifdef _WIN32
    static thread_local DutySleepTimer timer;
    if (timer.sleep_for(duration)) return;
#endif
    std::this_thread::sleep_for(duration);
}

struct GpuDutyBudget {
    int64_t pending_work_ns = 0;
};

void throttle_after_gpu_iteration(std::chrono::steady_clock::time_point started,
                                  GpuDutyBudget& budget) {
    const int duty = gpu_duty_percent();
    if (duty >= 100) return;

    const auto finished = std::chrono::steady_clock::now();
    const auto work_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started);
    if (work_ns.count() <= 0) return;

    // duty = work / (work + sleep), so sleep = work * (100-duty) / duty.
#ifdef _WIN32
    // Even high-resolution waits have significant overhead compared with one
    // fast GPU iteration. Accumulate only the measured iteration durations,
    // not checkpoint/printing time or the previous sleep, then rest once per
    // 10 ms of work. The budget belongs to this residue, not a process global.
    constexpr int64_t work_quantum_ns = 10000000;
    budget.pending_work_ns += work_ns.count();
    if (budget.pending_work_ns < work_quantum_ns) return;
    auto remaining = std::chrono::nanoseconds(
        budget.pending_work_ns * static_cast<int64_t>(100 - duty) / duty);
    budget.pending_work_ns = 0;
    // A 1% duty cycle can require a long rest. Check the interrupt flag between
    // short timer waits so checkpoint-on-Ctrl+C remains responsive.
    constexpr auto max_pause = std::chrono::milliseconds(25);
    while (remaining.count() > 0 && g_gfps_stop_requested == 0) {
        const auto pause = std::min(remaining,
            std::chrono::duration_cast<std::chrono::nanoseconds>(max_pause));
        sleep_for_gpu_duty(pause);
        remaining -= pause;
    }
#else
    (void)budget;
    const auto sleep_ns = std::chrono::nanoseconds(
        work_ns.count() * static_cast<int64_t>(100 - duty) / static_cast<int64_t>(duty));
    sleep_for_gpu_duty(sleep_ns);
#endif
}

void print_gpu_throttle_config() {
    std::cout << "GPU throttle: ntt_blocks="
              << (ntt_blocks_are_forced() ? "forced:" : "auto(initial):")
              << ntt_block_limit()
              << ", duty_percent=" << gpu_duty_percent() << "\n";
}

struct U128 {
    uint64_t lo = 0;
    uint64_t hi = 0;
};

struct S128 {
    bool neg = false;
    U128 mag{};
};

bool is_zero(U128 a) {
    return a.lo == 0 && a.hi == 0;
}

int cmp_u128(U128 a, U128 b) {
    if (a.hi != b.hi) return a.hi < b.hi ? -1 : 1;
    if (a.lo != b.lo) return a.lo < b.lo ? -1 : 1;
    return 0;
}

U128 add_u128(U128 a, U128 b) {
    U128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1ull : 0ull);
    return r;
}

U128 sub_u128(U128 a, U128 b) {
    U128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1ull : 0ull);
    return r;
}

U128 shr1_u128(U128 a) {
    return U128{(a.lo >> 1) | (a.hi << 63), a.hi >> 1};
}

U128 from_u64(uint64_t x) {
    return U128{x, 0};
}

U128 mul64_u128(uint64_t a, uint64_t b) {
#if defined(_MSC_VER) && defined(_M_X64)
    uint64_t hi = 0;
    const uint64_t lo = _umul128(a, b, &hi);
    return U128{lo, hi};
#else
    unsigned __int128 x = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return U128{static_cast<uint64_t>(x), static_cast<uint64_t>(x >> 64)};
#endif
}

U128 mul_u128_u32(U128 a, uint32_t b) {
    uint32_t limbs[4] = {
        static_cast<uint32_t>(a.lo),
        static_cast<uint32_t>(a.lo >> 32),
        static_cast<uint32_t>(a.hi),
        static_cast<uint32_t>(a.hi >> 32),
    };
    uint64_t carry = 0;
    for (uint32_t& limb : limbs) {
        const uint64_t prod = static_cast<uint64_t>(limb) * b + carry;
        limb = static_cast<uint32_t>(prod);
        carry = prod >> 32;
    }
    if (carry != 0) throw std::runtime_error("u128 overflow");
    return U128{
        (static_cast<uint64_t>(limbs[1]) << 32) | limbs[0],
        (static_cast<uint64_t>(limbs[3]) << 32) | limbs[2],
    };
}

bool bit_u128(U128 a, int bit) {
    return bit < 64 ? ((a.lo >> bit) & 1ull) != 0 : ((a.hi >> (bit - 64)) & 1ull) != 0;
}

void set_bit_u128(U128& a, int bit) {
    if (bit < 64) a.lo |= 1ull << bit;
    else a.hi |= 1ull << (bit - 64);
}

U128 divmod_u128_u64(U128 a, uint64_t d, uint64_t& rem) {
    if (d == 0) throw std::runtime_error("division by zero");
    U128 q{};
    rem = 0;
    for (int i = 127; i >= 0; --i) {
        if (rem >= (1ull << 63)) throw std::runtime_error("division remainder overflow");
        rem = (rem << 1) | (bit_u128(a, i) ? 1ull : 0ull);
        if (rem >= d) {
            rem -= d;
            set_bit_u128(q, i);
        }
    }
    return q;
}

S128 make_s128(int64_t x) {
    if (x < 0) return S128{true, from_u64(static_cast<uint64_t>(-(x + 1)) + 1ull)};
    return S128{false, from_u64(static_cast<uint64_t>(x))};
}

S128 neg_s128(S128 x) {
    if (!is_zero(x.mag)) x.neg = !x.neg;
    return x;
}

S128 add_s128(S128 a, S128 b) {
    if (a.neg == b.neg) return S128{a.neg, add_u128(a.mag, b.mag)};
    const int c = cmp_u128(a.mag, b.mag);
    if (c == 0) return S128{};
    if (c > 0) return S128{a.neg, sub_u128(a.mag, b.mag)};
    return S128{b.neg, sub_u128(b.mag, a.mag)};
}

S128 sub_s128(S128 a, S128 b) {
    return add_s128(a, neg_s128(b));
}

S128 mul_s128_u32(S128 a, uint32_t b) {
    return S128{a.neg, mul_u128_u32(a.mag, b)};
}

int64_t to_i64_checked(S128 x) {
    if (x.mag.hi != 0 || x.mag.lo > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("s128 does not fit int64");
    }
    const int64_t v = static_cast<int64_t>(x.mag.lo);
    return x.neg ? -v : v;
}

uint64_t parse_u64(const char* s) {
    char* end = nullptr;
    const uint64_t v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') throw std::runtime_error("invalid uint64 argument");
    return v;
}

uint64_t rotl64_host(uint64_t x, unsigned n) {
    return (x << n) | (x >> ((64 - n) & 63));
}

[[maybe_unused]] uint64_t hash_digits64(const std::vector<int64_t>& digits) {
    uint64_t hash = 0;
    bool is_zero_value = true;
    for (const int64_t d : digits) {
        const uint64_t x = static_cast<uint64_t>(d);
        hash += x;
        hash ^= rotl64_host(x + 0xc39d8a0552b073e8ull, static_cast<unsigned>((17 * x + 5) & 63));
        is_zero_value &= (d == 0);
    }
    if (is_zero_value) throw std::runtime_error("hash of zero residue requested");
    return hash;
}

// Self-contained SHA-256 for checkpoint integrity. Checkpoints are exchanged
// between Linux and Windows builds, so authenticated integers use explicit
// little-endian encoding rather than native structs.
class Sha256 {
public:
    Sha256()
        : _state{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

    void update(const uint8_t* data, size_t size) {
        if (_finalized) throw std::runtime_error("SHA-256 update after finalization");
        if (size > (std::numeric_limits<uint64_t>::max() - _total_bytes)) {
            throw std::runtime_error("SHA-256 input is too large");
        }
        _total_bytes += static_cast<uint64_t>(size);
        while (size != 0) {
            const size_t take = std::min(size, _buffer.size() - _buffer_size);
            std::memcpy(_buffer.data() + _buffer_size, data, take);
            _buffer_size += take;
            data += take;
            size -= take;
            if (_buffer_size == _buffer.size()) {
                transform(_buffer.data());
                _buffer_size = 0;
            }
        }
    }

    std::array<uint8_t, 32> final() {
        if (_finalized) throw std::runtime_error("SHA-256 finalized twice");
        const uint64_t bit_length = _total_bytes * 8;
        _buffer[_buffer_size++] = 0x80;
        if (_buffer_size > 56) {
            std::fill(_buffer.begin() + static_cast<std::ptrdiff_t>(_buffer_size), _buffer.end(), 0);
            transform(_buffer.data());
            _buffer_size = 0;
        }
        std::fill(_buffer.begin() + static_cast<std::ptrdiff_t>(_buffer_size), _buffer.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) _buffer[63 - i] = static_cast<uint8_t>(bit_length >> (8 * i));
        transform(_buffer.data());
        _finalized = true;

        std::array<uint8_t, 32> out{};
        for (size_t i = 0; i < _state.size(); ++i) {
            out[4 * i] = static_cast<uint8_t>(_state[i] >> 24);
            out[4 * i + 1] = static_cast<uint8_t>(_state[i] >> 16);
            out[4 * i + 2] = static_cast<uint8_t>(_state[i] >> 8);
            out[4 * i + 3] = static_cast<uint8_t>(_state[i]);
        }
        return out;
    }

private:
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

    void transform(const uint8_t block[64]) {
        static constexpr uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
                   (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
                   static_cast<uint32_t>(block[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = _state[0], b = _state[1], c = _state[2], d = _state[3];
        uint32_t e = _state[4], f = _state[5], g = _state[6], h = _state[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        _state[0] += a; _state[1] += b; _state[2] += c; _state[3] += d;
        _state[4] += e; _state[5] += f; _state[6] += g; _state[7] += h;
    }

    std::array<uint32_t, 8> _state;
    std::array<uint8_t, 64> _buffer{};
    size_t _buffer_size = 0;
    uint64_t _total_bytes = 0;
    bool _finalized = false;
};

void sha256_update_u32_le(Sha256& sha, uint32_t value) {
    uint8_t bytes[4];
    for (int i = 0; i < 4; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    sha.update(bytes, sizeof(bytes));
}

void sha256_update_u64_le(Sha256& sha, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    sha.update(bytes, sizeof(bytes));
}

std::array<uint8_t, 32> checkpoint_sha256(uint32_t n, uint64_t base, uint64_t total_bits,
                                          uint64_t processed_bits, const std::vector<int64_t>& digits) {
    static constexpr uint8_t domain[] = "GFPS checkpoint v2";
    static constexpr uint8_t magic[8] = {'B', 'G', 'F', 'N', 'C', 'K', '2', 0};
    Sha256 sha;
    sha.update(domain, sizeof(domain) - 1);
    sha.update(magic, sizeof(magic));
    sha256_update_u32_le(sha, 2);
    sha256_update_u32_le(sha, n);
    sha256_update_u64_le(sha, base);
    sha256_update_u64_le(sha, total_bits);
    sha256_update_u64_le(sha, processed_bits);
    sha256_update_u64_le(sha, static_cast<uint64_t>(digits.size()));
    for (const int64_t digit : digits) sha256_update_u64_le(sha, static_cast<uint64_t>(digit));
    return sha.final();
}

uint32_t pow_mod_host(uint32_t a, uint64_t e, uint32_t p) {
    uint64_t r = 1, x = a;
    while (e != 0) {
        if ((e & 1) != 0) r = (r * x) % p;
        x = (x * x) % p;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

__device__ uint32_t pow_mod_dev(uint32_t a, uint64_t e, uint32_t p) {
    uint64_t r = 1, x = a;
    while (e != 0) {
        if ((e & 1) != 0) r = (r * x) % p;
        x = (x * x) % p;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

__global__ void load_mod_kernel(uint32_t* dst, const int64_t* src, int m, int len, uint32_t p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    if (i < m) {
        int64_t v = src[i];
        int64_t r = v % static_cast<int64_t>(p);
        if (r < 0) r += p;
        dst[i] = static_cast<uint32_t>(r);
    } else {
        dst[i] = 0;
    }
}

__global__ void bit_reverse_kernel(uint32_t* out, const uint32_t* in, int log_len) {
    const int n = 1 << log_len;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = static_cast<unsigned>(i);
    unsigned r = 0;
    for (int b = 0; b < log_len; ++b) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    out[r] = in[i];
}

__global__ void bit_reverse_table_kernel(uint32_t* out, const uint32_t* in, const uint32_t* rev, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[rev[i]] = in[i];
}

__global__ void make_twiddle_kernel(uint32_t* roots, int half, uint32_t wlen, uint32_t p) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    roots[j] = pow_mod_dev(wlen, static_cast<uint64_t>(j), p);
}

__global__ void ntt_stage_kernel(uint32_t* a, const uint32_t* roots, int len, int butterflies, uint32_t p) {
    const int half = len >> 1;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        const int block = idx / half;
        const int j = idx - block * half;
        const int pos = block * len + j;
        const uint32_t w = roots[j];
        const uint32_t u = a[pos];
        const uint32_t v = static_cast<uint32_t>((static_cast<uint64_t>(a[pos + half]) * w) % p);
        uint32_t x = u + v;
        if (x >= p) x -= p;
        uint32_t y = (u >= v) ? (u - v) : (u + p - v);
        a[pos] = x;
        a[pos + half] = y;
    }
}

__global__ void ntt_stage2_kernel(uint32_t* a, const uint32_t* roots1, const uint32_t* roots2,
                                  int len, int groups, uint32_t p) {
    const int quarter = len >> 2;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        const int block = idx / quarter;
        const int j = idx - block * quarter;
        const int pos = block * len + j;

        const uint32_t w1 = roots1[j];
        const uint32_t w20 = roots2[j];
        const uint32_t w21 = roots2[j + quarter];

        const uint32_t a0 = a[pos];
        const uint32_t a1 = static_cast<uint32_t>((static_cast<uint64_t>(a[pos + quarter]) * w1) % p);
        const uint32_t a2 = a[pos + 2 * quarter];
        const uint32_t a3 = static_cast<uint32_t>((static_cast<uint64_t>(a[pos + 3 * quarter]) * w1) % p);

        uint32_t b0 = a0 + a1;
        if (b0 >= p) b0 -= p;
        const uint32_t b1 = (a0 >= a1) ? (a0 - a1) : (a0 + p - a1);
        uint32_t b2 = a2 + a3;
        if (b2 >= p) b2 -= p;
        const uint32_t b3 = (a2 >= a3) ? (a2 - a3) : (a2 + p - a3);

        const uint32_t t0 = static_cast<uint32_t>((static_cast<uint64_t>(b2) * w20) % p);
        const uint32_t t1 = static_cast<uint32_t>((static_cast<uint64_t>(b3) * w21) % p);

        uint32_t c0 = b0 + t0;
        if (c0 >= p) c0 -= p;
        const uint32_t c2 = (b0 >= t0) ? (b0 - t0) : (b0 + p - t0);
        uint32_t c1 = b1 + t1;
        if (c1 >= p) c1 -= p;
        const uint32_t c3 = (b1 >= t1) ? (b1 - t1) : (b1 + p - t1);

        a[pos] = c0;
        a[pos + quarter] = c1;
        a[pos + 2 * quarter] = c2;
        a[pos + 3 * quarter] = c3;
    }
}

__device__ void ntt_stage_one_apply(uint32_t* a, const uint32_t* roots, int len, int idx, uint32_t p) {
    const int half = len >> 1;
    const int block = idx / half;
    const int j = idx - block * half;
    const int pos = block * len + j;
    const uint32_t w = roots[j];
    const uint32_t u = a[pos];
    const uint32_t v = static_cast<uint32_t>((static_cast<uint64_t>(a[pos + half]) * w) % p);
    uint32_t x = u + v;
    if (x >= p) x -= p;
    const uint32_t y = (u >= v) ? (u - v) : (u + p - v);
    a[pos] = x;
    a[pos + half] = y;
}

__device__ void ntt_stage2_apply(uint32_t* a, const uint32_t* roots1, const uint32_t* roots2,
                                 int len, int idx, uint32_t p) {
    const int quarter = len >> 2;
    const int block = idx / quarter;
    const int j = idx - block * quarter;
    const int pos = block * len + j;

    const uint32_t w1 = roots1[j];
    const uint32_t w20 = roots2[j];
    const uint32_t w21 = roots2[j + quarter];

    const uint32_t a0 = a[pos];
    const uint32_t a1 = static_cast<uint32_t>((static_cast<uint64_t>(a[pos + quarter]) * w1) % p);
    const uint32_t a2 = a[pos + 2 * quarter];
    const uint32_t a3 = static_cast<uint32_t>((static_cast<uint64_t>(a[pos + 3 * quarter]) * w1) % p);

    uint32_t b0 = a0 + a1;
    if (b0 >= p) b0 -= p;
    const uint32_t b1 = (a0 >= a1) ? (a0 - a1) : (a0 + p - a1);
    uint32_t b2 = a2 + a3;
    if (b2 >= p) b2 -= p;
    const uint32_t b3 = (a2 >= a3) ? (a2 - a3) : (a2 + p - a3);

    const uint32_t t0 = static_cast<uint32_t>((static_cast<uint64_t>(b2) * w20) % p);
    const uint32_t t1 = static_cast<uint32_t>((static_cast<uint64_t>(b3) * w21) % p);

    uint32_t c0 = b0 + t0;
    if (c0 >= p) c0 -= p;
    const uint32_t c2 = (b0 >= t0) ? (b0 - t0) : (b0 + p - t0);
    uint32_t c1 = b1 + t1;
    if (c1 >= p) c1 -= p;
    const uint32_t c3 = (b1 >= t1) ? (b1 - t1) : (b1 + p - t1);

    a[pos] = c0;
    a[pos + quarter] = c1;
    a[pos + 2 * quarter] = c2;
    a[pos + 3 * quarter] = c3;
}

__global__ void bit_reverse_table3_kernel(uint32_t* out0, uint32_t* out1, uint32_t* out2,
                                          const uint32_t* in0, const uint32_t* in1, const uint32_t* in2,
                                          const uint32_t* rev, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t j = rev[i];
    out0[j] = in0[i];
    out1[j] = in1[i];
    out2[j] = in2[i];
}

__global__ void copy3_kernel(uint32_t* dst0, uint32_t* dst1, uint32_t* dst2,
                             const uint32_t* src0, const uint32_t* src1, const uint32_t* src2, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst0[i] = src0[i];
    dst1[i] = src1[i];
    dst2[i] = src2[i];
}

__global__ void ntt_stage_3_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                                   const uint32_t* roots0, const uint32_t* roots1, const uint32_t* roots2,
                                   int len, int butterflies) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        ntt_stage_one_apply(r0, roots0, len, idx, p0);
        ntt_stage_one_apply(r1, roots1, len, idx, p1);
        ntt_stage_one_apply(r2, roots2, len, idx, p2);
    }
}

__global__ void ntt_stage2_3_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                                    const uint32_t* roots10, const uint32_t* roots11, const uint32_t* roots12,
                                    const uint32_t* roots20, const uint32_t* roots21, const uint32_t* roots22,
                                    int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage2_apply(r0, roots10, roots20, len, idx, p0);
        ntt_stage2_apply(r1, roots11, roots21, len, idx, p1);
        ntt_stage2_apply(r2, roots12, roots22, len, idx, p2);
    }
}

__global__ void pointwise_square3_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2, int len) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = static_cast<uint32_t>((static_cast<uint64_t>(r0[i]) * r0[i]) % p0);
    r1[i] = static_cast<uint32_t>((static_cast<uint64_t>(r1[i]) * r1[i]) % p1);
    r2[i] = static_cast<uint32_t>((static_cast<uint64_t>(r2[i]) * r2[i]) % p2);
}

__global__ void scale3_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2, int len,
                              uint32_t inv0, uint32_t inv1, uint32_t inv2) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = static_cast<uint32_t>((static_cast<uint64_t>(r0[i]) * inv0) % p0);
    r1[i] = static_cast<uint32_t>((static_cast<uint64_t>(r1[i]) * inv1) % p1);
    r2[i] = static_cast<uint32_t>((static_cast<uint64_t>(r2[i]) * inv2) % p2);
}

__global__ void bit_reverse_table4_kernel(uint32_t* out0, uint32_t* out1, uint32_t* out2, uint32_t* out3,
                                          const uint32_t* in0, const uint32_t* in1,
                                          const uint32_t* in2, const uint32_t* in3,
                                          const uint32_t* rev, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t j = rev[i];
    out0[j] = in0[i];
    out1[j] = in1[i];
    out2[j] = in2[i];
    out3[j] = in3[i];
}

__global__ void copy4_kernel(uint32_t* dst0, uint32_t* dst1, uint32_t* dst2, uint32_t* dst3,
                             const uint32_t* src0, const uint32_t* src1,
                             const uint32_t* src2, const uint32_t* src3, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst0[i] = src0[i];
    dst1[i] = src1[i];
    dst2[i] = src2[i];
    dst3[i] = src3[i];
}

__global__ void ntt_stage_4_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                                   const uint32_t* roots0, const uint32_t* roots1,
                                   const uint32_t* roots2, const uint32_t* roots3,
                                   int len, int butterflies) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    constexpr uint32_t p3 = 1224736769u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        ntt_stage_one_apply(r0, roots0, len, idx, p0);
        ntt_stage_one_apply(r1, roots1, len, idx, p1);
        ntt_stage_one_apply(r2, roots2, len, idx, p2);
        ntt_stage_one_apply(r3, roots3, len, idx, p3);
    }
}

__global__ void ntt_stage2_4_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                                    const uint32_t* roots10, const uint32_t* roots11,
                                    const uint32_t* roots12, const uint32_t* roots13,
                                    const uint32_t* roots20, const uint32_t* roots21,
                                    const uint32_t* roots22, const uint32_t* roots23,
                                    int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    constexpr uint32_t p3 = 1224736769u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage2_apply(r0, roots10, roots20, len, idx, p0);
        ntt_stage2_apply(r1, roots11, roots21, len, idx, p1);
        ntt_stage2_apply(r2, roots12, roots22, len, idx, p2);
        ntt_stage2_apply(r3, roots13, roots23, len, idx, p3);
    }
}

__global__ void pointwise_square4_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, int len) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    constexpr uint32_t p3 = 1224736769u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = static_cast<uint32_t>((static_cast<uint64_t>(r0[i]) * r0[i]) % p0);
    r1[i] = static_cast<uint32_t>((static_cast<uint64_t>(r1[i]) * r1[i]) % p1);
    r2[i] = static_cast<uint32_t>((static_cast<uint64_t>(r2[i]) * r2[i]) % p2);
    r3[i] = static_cast<uint32_t>((static_cast<uint64_t>(r3[i]) * r3[i]) % p3);
}

// psi has order 2*m. Twisting by psi^i converts x^m+1 convolution
// to an m-point cyclic NTT; inverse twisting restores integer coefficients.
__global__ void twist_rns_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                                 uint32_t* r3, const uint32_t* twist,
                                 int m, int mod_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    r0[i] = static_cast<uint32_t>((static_cast<uint64_t>(r0[i]) * twist[i]) % 998244353u);
    r1[i] = static_cast<uint32_t>((static_cast<uint64_t>(r1[i]) * twist[m + i]) % 1004535809u);
    r2[i] = static_cast<uint32_t>((static_cast<uint64_t>(r2[i]) * twist[2*m + i]) % 469762049u);
    if (mod_count == 4)
        r3[i] = static_cast<uint32_t>((static_cast<uint64_t>(r3[i]) * twist[3*m + i]) % 1224736769u);
}

__global__ void scale4_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, int len,
                              uint32_t inv0, uint32_t inv1, uint32_t inv2, uint32_t inv3) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    constexpr uint32_t p3 = 1224736769u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = static_cast<uint32_t>((static_cast<uint64_t>(r0[i]) * inv0) % p0);
    r1[i] = static_cast<uint32_t>((static_cast<uint64_t>(r1[i]) * inv1) % p1);
    r2[i] = static_cast<uint32_t>((static_cast<uint64_t>(r2[i]) * inv2) % p2);
    r3[i] = static_cast<uint32_t>((static_cast<uint64_t>(r3[i]) * inv3) % p3);
}

template<uint32_t P>
__device__ void dif_stage2_apply(uint32_t* a, const uint32_t* roots_big,
                                  const uint32_t* roots_small, int len, int idx) {
    const int quarter = len >> 2;
    const int group = idx / quarter;
    const int j = idx - group * quarter;
    const int pos = group * len + j;
    const uint32_t a0 = a[pos], a1 = a[pos+quarter];
    const uint32_t a2 = a[pos+2*quarter], a3 = a[pos+3*quarter];
    uint32_t b0 = a0+a2; if (b0>=P) b0-=P;
    uint32_t b1 = a1+a3; if (b1>=P) b1-=P;
    const uint32_t b2 = static_cast<uint32_t>(uint64_t(a0>=a2 ? a0-a2 : a0+P-a2)*roots_big[j]%P);
    const uint32_t b3 = static_cast<uint32_t>(uint64_t(a1>=a3 ? a1-a3 : a1+P-a3)*roots_big[j+quarter]%P);
    uint32_t c0=b0+b1; if(c0>=P)c0-=P;
    uint32_t c2=b2+b3; if(c2>=P)c2-=P;
    a[pos]=c0;
    a[pos+quarter]=static_cast<uint32_t>(uint64_t(b0>=b1 ? b0-b1 : b0+P-b1)*roots_small[j]%P);
    a[pos+2*quarter]=c2;
    a[pos+3*quarter]=static_cast<uint32_t>(uint64_t(b2>=b3 ? b2-b3 : b2+P-b3)*roots_small[j]%P);
}

__global__ void ntt_dif_stage2_rns_kernel(uint32_t* r0, uint32_t* r1,
    uint32_t* r2, uint32_t* r3, const uint32_t* tw0, const uint32_t* tw1,
    const uint32_t* tw2, const uint32_t* tw3,
    int off_big, int off_small, int len, int groups, int mod_count) {
    for (int idx=blockIdx.x*blockDim.x+threadIdx.x; idx<groups;
         idx+=blockDim.x*gridDim.x) {
        dif_stage2_apply<998244353u>(r0,tw0+off_big,tw0+off_small,len,idx);
        dif_stage2_apply<1004535809u>(r1,tw1+off_big,tw1+off_small,len,idx);
        dif_stage2_apply<469762049u>(r2,tw2+off_big,tw2+off_small,len,idx);
        if(mod_count==4) dif_stage2_apply<1224736769u>(r3,tw3+off_big,tw3+off_small,len,idx);
    }
}

template<int Mode>
__global__ void ntt_shared1024_rns_kernel(uint32_t* r0, uint32_t* r1,
    uint32_t* r2, uint32_t* r3, const uint32_t* tw0, const uint32_t* tw1,
    const uint32_t* tw2, const uint32_t* tw3, int mod_count,
    const uint32_t* inv0, const uint32_t* inv1,
    const uint32_t* inv2, const uint32_t* inv3) {
    __shared__ uint32_t a0[1024], a1[1024], a2[1024], a3[1024];
    const int lane=threadIdx.x, offset=blockIdx.x*1024;
    for(int i=lane;i<1024;i+=256) {
        a0[i]=r0[offset+i]; a1[i]=r1[offset+i]; a2[i]=r2[offset+i];
        if(mod_count==4) a3[i]=r3[offset+i];
    }
    __syncthreads();
    #pragma unroll
    for(int phase=0;phase<(Mode==2 ? 2 : 1);++phase) {
    const uint32_t* tr0=Mode==2 && phase==1 ? inv0 : tw0;
    const uint32_t* tr1=Mode==2 && phase==1 ? inv1 : tw1;
    const uint32_t* tr2=Mode==2 && phase==1 ? inv2 : tw2;
    const uint32_t* tr3=Mode==2 && phase==1 ? inv3 : tw3;
    #pragma unroll
    for(int pair=0;pair<5;++pair) {
        const bool forward = Mode==1 || (Mode==2 && phase==0);
        const int high=forward ? 10-2*pair : 2+2*pair;
        const int off_big=(1<<(high-1))-1;
        const int off_small=(1<<(high-2))-1;
        if(forward) {
            dif_stage2_apply<998244353u>(a0,tr0+off_big,tr0+off_small,1<<high,lane);
            dif_stage2_apply<1004535809u>(a1,tr1+off_big,tr1+off_small,1<<high,lane);
            dif_stage2_apply<469762049u>(a2,tr2+off_big,tr2+off_small,1<<high,lane);
            if(mod_count==4)
                dif_stage2_apply<1224736769u>(a3,tr3+off_big,tr3+off_small,1<<high,lane);
        } else {
            ntt_stage2_apply(a0,tr0+off_small,tr0+off_big,1<<high,lane,998244353u);
            ntt_stage2_apply(a1,tr1+off_small,tr1+off_big,1<<high,lane,1004535809u);
            ntt_stage2_apply(a2,tr2+off_small,tr2+off_big,1<<high,lane,469762049u);
            if(mod_count==4)
                ntt_stage2_apply(a3,tr3+off_small,tr3+off_big,1<<high,lane,1224736769u);
        }
        __syncthreads();
    }
    if(Mode==2 && phase==0) {
        for(int i=lane;i<1024;i+=256) {
            a0[i]=static_cast<uint32_t>(uint64_t(a0[i])*a0[i]%998244353u);
            a1[i]=static_cast<uint32_t>(uint64_t(a1[i])*a1[i]%1004535809u);
            a2[i]=static_cast<uint32_t>(uint64_t(a2[i])*a2[i]%469762049u);
            if(mod_count==4) a3[i]=static_cast<uint32_t>(uint64_t(a3[i])*a3[i]%1224736769u);
        }
        __syncthreads();
    }
    }
    for(int i=lane;i<1024;i+=256) {
        r0[offset+i]=a0[i]; r1[offset+i]=a1[i]; r2[offset+i]=a2[i];
        if(mod_count==4) r3[offset+i]=a3[i];
    }
}

template<uint32_t P>
__device__ void dif_stage_apply(uint32_t* a, const uint32_t* roots, int len, int idx) {
    const int half=len/2, group=idx/half, j=idx-group*half;
    const int pos=group*len+j;
    const uint32_t u=a[pos], v=a[pos+half];
    uint32_t sum=u+v; if(sum>=P) sum-=P;
    a[pos]=sum;
    a[pos+half]=static_cast<uint32_t>(uint64_t(u>=v ? u-v : u+P-v)*roots[j]%P);
}

__global__ void ntt_dif_stage_rns_kernel(uint32_t* r0, uint32_t* r1,
    uint32_t* r2, uint32_t* r3, const uint32_t* tw0, const uint32_t* tw1,
    const uint32_t* tw2, const uint32_t* tw3,
    int offset, int len, int butterflies, int mod_count) {
    for(int idx=blockIdx.x*blockDim.x+threadIdx.x; idx<butterflies;idx+=gridDim.x*blockDim.x) {
        dif_stage_apply<998244353u>(r0,tw0+offset,len,idx);
        dif_stage_apply<1004535809u>(r1,tw1+offset,len,idx);
        dif_stage_apply<469762049u>(r2,tw2+offset,len,idx);
        if(mod_count==4) dif_stage_apply<1224736769u>(r3,tw3+offset,len,idx);
    }
}

__global__ void pointwise_square_kernel(uint32_t* a, int len, uint32_t p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    a[i] = static_cast<uint32_t>((static_cast<uint64_t>(a[i]) * a[i]) % p);
}

__global__ void pointwise_mul_kernel(uint32_t* a, const uint32_t* b, int len, uint32_t p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    a[i] = static_cast<uint32_t>((static_cast<uint64_t>(a[i]) * b[i]) % p);
}

__global__ void scale_kernel(uint32_t* a, int len, uint32_t inv_len, uint32_t p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    a[i] = static_cast<uint32_t>((static_cast<uint64_t>(a[i]) * inv_len) % p);
}

__device__ bool d_is_zero(U128 a) {
    return a.lo == 0 && a.hi == 0;
}

__device__ int d_cmp_u128(U128 a, U128 b) {
    if (a.hi != b.hi) return a.hi < b.hi ? -1 : 1;
    if (a.lo != b.lo) return a.lo < b.lo ? -1 : 1;
    return 0;
}

__device__ U128 d_add_u128(U128 a, U128 b) {
    U128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1ull : 0ull);
    return r;
}

__device__ U128 d_sub_u128(U128 a, U128 b) {
    U128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1ull : 0ull);
    return r;
}

__device__ U128 d_from_u64(uint64_t x) {
    return U128{x, 0};
}

__device__ U128 d_mul64_u128(uint64_t a, uint64_t b) {
    return U128{a * b, __umul64hi(a, b)};
}

__device__ U128 d_mul_u128_u32(U128 a, uint32_t b) {
    uint32_t limbs[4] = {
        static_cast<uint32_t>(a.lo),
        static_cast<uint32_t>(a.lo >> 32),
        static_cast<uint32_t>(a.hi),
        static_cast<uint32_t>(a.hi >> 32),
    };
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        const uint64_t prod = static_cast<uint64_t>(limbs[i]) * b + carry;
        limbs[i] = static_cast<uint32_t>(prod);
        carry = prod >> 32;
    }
    return U128{
        (static_cast<uint64_t>(limbs[1]) << 32) | limbs[0],
        (static_cast<uint64_t>(limbs[3]) << 32) | limbs[2],
    };
}

__device__ bool d_bit_u128(U128 a, int bit) {
    return bit < 64 ? ((a.lo >> bit) & 1ull) != 0 : ((a.hi >> (bit - 64)) & 1ull) != 0;
}

__device__ void d_set_bit_u128(U128& a, int bit) {
    if (bit < 64) a.lo |= 1ull << bit;
    else a.hi |= 1ull << (bit - 64);
}

#ifndef GFPS_USE_RECIPROCAL_DIV
#define GFPS_USE_RECIPROCAL_DIV 1
#endif
#ifndef GFPS_USE_CUDA_GRAPHS
#define GFPS_USE_CUDA_GRAPHS 1
#endif

__device__ U128 d_divmod_u128_u64_bitwise(U128 a, uint64_t d,
                                           uint64_t& rem) {
    U128 q{};
    rem = 0;
    for (int i = 127; i >= 0; --i) {
        rem = (rem << 1) | (d_bit_u128(a, i) ? 1ull : 0ull);
        if (rem >= d) {
            rem -= d;
            d_set_bit_u128(q, i);
        }
    }
    return q;
}

__device__ U128 d_divmod_u128_u64(U128 a, uint64_t d, double inv_d, uint64_t& rem) {
    if (d <= 0xffffffffull) {
        const uint32_t limbs[4] = {
            static_cast<uint32_t>(a.lo),
            static_cast<uint32_t>(a.lo >> 32),
            static_cast<uint32_t>(a.hi),
            static_cast<uint32_t>(a.hi >> 32),
        };
        uint32_t q[4] = {};
        uint64_t r = 0;
        for (int i = 3; i >= 0; --i) {
            const uint64_t cur = (r << 32) | limbs[i];
            q[i] = static_cast<uint32_t>(cur / d);
            r = cur - static_cast<uint64_t>(q[i]) * d;
        }
        rem = r;
        return U128{
            (static_cast<uint64_t>(q[1]) << 32) | q[0],
            (static_cast<uint64_t>(q[3]) << 32) | q[2],
        };
    }

    if (d <= (1ull << 53)) {
        const uint32_t limbs[4] = {
            static_cast<uint32_t>(a.lo),
            static_cast<uint32_t>(a.lo >> 32),
            static_cast<uint32_t>(a.hi),
            static_cast<uint32_t>(a.hi >> 32),
        };
        uint32_t q[4] = {};
        uint64_t r = 0;
        for (int i = 3; i >= 0; --i) {
            const U128 cur{(r << 32) | limbs[i], r >> 32};
#if GFPS_USE_RECIPROCAL_DIV
            uint64_t qi = static_cast<uint64_t>(
                ((static_cast<double>(r) * 4294967296.0) +
                 static_cast<double>(limbs[i])) * inv_d);
#else
            uint64_t qi = static_cast<uint64_t>(
                ((static_cast<double>(r) * 4294967296.0) +
                 static_cast<double>(limbs[i])) / static_cast<double>(d));
#endif
            if (qi > 0xffffffffull) qi = 0xffffffffull;

            U128 prod = d_mul64_u128(d, static_cast<uint32_t>(qi));
            while (d_cmp_u128(prod, cur) > 0) {
                --qi;
                prod = d_sub_u128(prod, d_from_u64(d));
            }
            U128 rem_u = d_sub_u128(cur, prod);
            while (d_cmp_u128(rem_u, d_from_u64(d)) >= 0 &&
                   qi < 0xffffffffull) {
                rem_u = d_sub_u128(rem_u, d_from_u64(d));
                ++qi;
            }
            if (rem_u.hi != 0 || rem_u.lo >= d) {
                return d_divmod_u128_u64_bitwise(a, d, rem);
            }
            q[i] = static_cast<uint32_t>(qi);
            r = rem_u.lo;
        }
        rem = r;
        return U128{
            (static_cast<uint64_t>(q[1]) << 32) | q[0],
            (static_cast<uint64_t>(q[3]) << 32) | q[2],
        };
    }

    return d_divmod_u128_u64_bitwise(a, d, rem);
}

__device__ S128 d_make_s128_i64(int64_t x) {
    if (x < 0) return S128{true, d_from_u64(static_cast<uint64_t>(-(x + 1)) + 1ull)};
    return S128{false, d_from_u64(static_cast<uint64_t>(x))};
}

__device__ S128 d_neg_s128(S128 x) {
    if (!d_is_zero(x.mag)) x.neg = !x.neg;
    return x;
}

__device__ S128 d_add_s128(S128 a, S128 b) {
    if (a.neg == b.neg) return S128{a.neg, d_add_u128(a.mag, b.mag)};
    const int c = d_cmp_u128(a.mag, b.mag);
    if (c == 0) return S128{};
    if (c > 0) return S128{a.neg, d_sub_u128(a.mag, b.mag)};
    return S128{b.neg, d_sub_u128(b.mag, a.mag)};
}

__device__ S128 d_sub_s128(S128 a, S128 b) {
    return d_add_s128(a, d_neg_s128(b));
}

__device__ bool d_eq_s128(S128 a, S128 b) {
    return a.neg == b.neg && a.mag.lo == b.mag.lo && a.mag.hi == b.mag.hi;
}

__device__ S128 d_mul_s128_u32(S128 a, uint32_t b) {
    return S128{a.neg, d_mul_u128_u32(a.mag, b)};
}

__device__ uint32_t d_sub_mod_u32(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? a - b : a + p - b;
}

__device__ uint32_t d_i64_to_mod(int64_t v, uint32_t p) {
    int64_t r = v % static_cast<int64_t>(p);
    if (r < 0) r += p;
    return static_cast<uint32_t>(r);
}

__device__ S128 d_crt3_signed(uint32_t r0, uint32_t r1, uint32_t r2) {
    constexpr uint64_t p0 = 998244353ull;
    constexpr uint64_t p1 = 1004535809ull;
    constexpr uint64_t p2 = 469762049ull;
    constexpr uint64_t inv_p0_p1 = 669690699ull;
    constexpr uint64_t inv_p01_p2 = 354521948ull;
    constexpr uint64_t p01 = 1002772198720536577ull;
    constexpr U128 mod = U128{1943602308845666305ull, 25536448ull};
    constexpr U128 half = U128{971801154422833152ull, 12768224ull};

    uint64_t t1 = (r1 + p1 - (r0 % p1)) % p1;
    t1 = (t1 * inv_p0_p1) % p1;

    uint64_t x_mod_p2 = (r0 + (p0 % p2) * (t1 % p2)) % p2;
    uint64_t t2 = (r2 + p2 - x_mod_p2) % p2;
    t2 = (t2 * inv_p01_p2) % p2;

    U128 x = d_from_u64(r0);
    x = d_add_u128(x, d_mul64_u128(p0, t1));
    x = d_add_u128(x, d_mul64_u128(p01, t2));
    if (d_cmp_u128(x, half) > 0) return S128{true, d_sub_u128(mod, x)};
    return S128{false, x};
}

__device__ S128 d_crt4_signed(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    constexpr uint64_t p0 = 998244353ull;
    constexpr uint64_t p1 = 1004535809ull;
    constexpr uint64_t p2 = 469762049ull;
    constexpr uint64_t p3 = 1224736769ull;
    constexpr uint64_t inv_p0_p1 = 669690699ull;
    constexpr uint64_t inv_p01_p2 = 354521948ull;
    constexpr uint64_t p01_mod_p3 = 378708305ull;
    constexpr uint64_t inv_p012_p3 = 125636969ull;
    constexpr uint64_t p01 = 1002772198720536577ull;
    constexpr U128 p012 = U128{1943602308845666305ull, 25536448ull};
    constexpr U128 mod = U128{4971815662639906817ull, 31275426944298320ull};
    constexpr U128 half = U128{2485907831319953408ull, 15637713472149160ull};

    uint64_t t1 = (r1 + p1 - (r0 % p1)) % p1;
    t1 = (t1 * inv_p0_p1) % p1;

    uint64_t x_mod_p2 = (r0 + (p0 % p2) * (t1 % p2)) % p2;
    uint64_t t2 = (r2 + p2 - x_mod_p2) % p2;
    t2 = (t2 * inv_p01_p2) % p2;

    uint64_t x_mod_p3 = r0 % p3;
    x_mod_p3 = (x_mod_p3 + (p0 % p3) * (t1 % p3)) % p3;
    x_mod_p3 = (x_mod_p3 + p01_mod_p3 * (t2 % p3)) % p3;
    uint64_t t3 = (r3 + p3 - x_mod_p3) % p3;
    t3 = (t3 * inv_p012_p3) % p3;

    U128 x = d_from_u64(r0);
    x = d_add_u128(x, d_mul64_u128(p0, t1));
    x = d_add_u128(x, d_mul64_u128(p01, t2));
    x = d_add_u128(x, d_mul_u128_u32(p012, static_cast<uint32_t>(t3)));
    if (d_cmp_u128(x, half) > 0) return S128{true, d_sub_u128(mod, x)};
    return S128{false, x};
}

__device__ S128 d_centered_divrem(S128 v, uint64_t base, double inv_base,
                                  int64_t& digit);
__device__ S128 d_centered_divrem_lower_tie(
    S128 v, uint64_t base, double inv_base, int64_t& digit);

__global__ void crt3_fold_kernel(const uint32_t* r0, const uint32_t* r1, const uint32_t* r2,
                                 int m, uint32_t scale, S128* coeff) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    S128 c = d_crt3_signed(
        d_sub_mod_u32(r0[i], r0[i + m], p0),
        d_sub_mod_u32(r1[i], r1[i + m], p1),
        d_sub_mod_u32(r2[i], r2[i + m], p2));
    if (scale != 1) c = d_mul_s128_u32(c, scale);
    coeff[i] = c;
}

__global__ void crt3_fold_carry_kernel(const uint32_t* r0, const uint32_t* r1, const uint32_t* r2,
                                       int m, uint64_t base, double inv_base, uint32_t scale,
                                       S128* coeff, int64_t* digits, S128* carry_out) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    S128 c = d_crt3_signed(
        d_sub_mod_u32(r0[i], r0[i + m], p0),
        d_sub_mod_u32(r1[i], r1[i + m], p1),
        d_sub_mod_u32(r2[i], r2[i + m], p2));
    if (scale != 1) c = d_mul_s128_u32(c, scale);
    coeff[i] = c;
    int64_t digit = 0;
    carry_out[i] = d_centered_divrem(c, base, inv_base, digit);
    digits[i] = digit;
}

__global__ void crt4_fold_kernel(const uint32_t* r0, const uint32_t* r1, const uint32_t* r2, const uint32_t* r3,
                                 int m, uint32_t scale, S128* coeff) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    constexpr uint32_t p3 = 1224736769u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    S128 c = d_crt4_signed(
        d_sub_mod_u32(r0[i], r0[i + m], p0),
        d_sub_mod_u32(r1[i], r1[i + m], p1),
        d_sub_mod_u32(r2[i], r2[i + m], p2),
        d_sub_mod_u32(r3[i], r3[i + m], p3));
    if (scale != 1) c = d_mul_s128_u32(c, scale);
    coeff[i] = c;
}

__global__ void crt4_fold_carry_kernel(const uint32_t* r0, const uint32_t* r1, const uint32_t* r2, const uint32_t* r3,
                                       int m, uint64_t base, double inv_base, uint32_t scale,
                                       S128* coeff, int64_t* digits, S128* carry_out) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    constexpr uint32_t p3 = 1224736769u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    S128 c = d_crt4_signed(
        d_sub_mod_u32(r0[i], r0[i + m], p0),
        d_sub_mod_u32(r1[i], r1[i + m], p1),
        d_sub_mod_u32(r2[i], r2[i + m], p2),
        d_sub_mod_u32(r3[i], r3[i + m], p3));
    if (scale != 1) c = d_mul_s128_u32(c, scale);
    coeff[i] = c;
    int64_t digit = 0;
    carry_out[i] = d_centered_divrem(c, base, inv_base, digit);
    digits[i] = digit;
}

__global__ void carry_normalize_serial_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                                              int m, uint64_t base, double inv_base, uint32_t scale,
                                              S128* coeff, int64_t* out, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *status = 0;
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    constexpr uint32_t p3 = 1224736769u;

    S128 carry{};
    const uint64_t half_base = base / 2;
    for (int pass = 0; pass < 8; ++pass) {
        carry = S128{};
        for (int i = 0; i < m; ++i) {
            S128 v = d_add_s128(coeff[i], carry);
            uint64_t rem = 0;
            U128 qmag = d_divmod_u128_u64(v.mag, base, inv_base, rem);
            S128 q{};
            int64_t digit = 0;
            if (!v.neg) {
                q = S128{false, qmag};
                digit = static_cast<int64_t>(rem);
            } else if (rem == 0) {
                q = S128{true, qmag};
                digit = 0;
            } else {
                q = S128{true, d_add_u128(qmag, d_from_u64(1))};
                digit = static_cast<int64_t>(base - rem);
            }
            if (static_cast<uint64_t>(digit) > half_base) {
                digit -= static_cast<int64_t>(base);
                q = d_add_s128(q, d_make_s128_i64(1));
            }
            coeff[i] = d_make_s128_i64(digit);
            carry = q;
        }
        if (d_is_zero(carry.mag)) break;
        coeff[0] = d_sub_s128(coeff[0], carry);
    }
    if (!d_is_zero(carry.mag)) {
        if ((base & 1ull) != 0) {
            *status = 1;
            return;
        }
        for (int i = 0; i < m; ++i) {
            S128 value = d_crt4_signed(
                d_sub_mod_u32(r0[i], r0[i + m], p0),
                d_sub_mod_u32(r1[i], r1[i + m], p1),
                d_sub_mod_u32(r2[i], r2[i + m], p2),
                d_sub_mod_u32(r3[i], r3[i + m], p3));
            if (scale != 1) value = d_mul_s128_u32(value, scale);
            coeff[i] = value;
        }
        bool lower_converged = false;
        for (int pass = 0; pass < 32; ++pass) {
            carry = S128{};
            for (int i = 0; i < m; ++i) {
                const S128 value = d_add_s128(coeff[i], carry);
                int64_t digit = 0;
                carry = d_centered_divrem_lower_tie(
                    value, base, inv_base, digit);
                coeff[i] = d_make_s128_i64(digit);
            }
            if (d_is_zero(carry.mag)) {
                lower_converged = true;
                break;
            }
            coeff[0] = d_sub_s128(coeff[0], carry);
        }
        if (!lower_converged) {
            *status = 1;
            return;
        }
        const int64_t half = static_cast<int64_t>(base / 2);
        for (int i = 0; i < m; ++i) {
            if (coeff[i].mag.hi != 0 ||
                coeff[i].mag.lo > static_cast<uint64_t>(INT64_MAX)) {
                *status = 1;
                return;
            }
            const int64_t magnitude = static_cast<int64_t>(coeff[i].mag.lo);
            const int64_t digit = coeff[i].neg ? -magnitude : magnitude;
            const int64_t expected = i == 0 ? -half : 1 - half;
            if (digit != expected) {
                *status = 1;
                return;
            }
        }
    }

    for (int i = 0; i < m; ++i) {
        if (coeff[i].mag.hi != 0 || coeff[i].mag.lo > static_cast<uint64_t>(0x7fffffffffffffffull)) {
            *status = 2;
            return;
        }
        const int64_t v = static_cast<int64_t>(coeff[i].mag.lo);
        const int64_t digit = coeff[i].neg ? -v : v;
        if (out != nullptr) out[i] = digit;
        r0[i] = d_i64_to_mod(digit, p0);
        r1[i] = d_i64_to_mod(digit, p1);
        r2[i] = d_i64_to_mod(digit, p2);
        r3[i] = d_i64_to_mod(digit, p3);
        r0[i + m] = 0;
        r1[i + m] = 0;
        r2[i + m] = 0;
        r3[i + m] = 0;
    }
}

__device__ S128 d_centered_divrem(S128 v, uint64_t base, double inv_base,
                                  int64_t& digit) {
    uint64_t rem = 0;
    U128 qmag = d_divmod_u128_u64(v.mag, base, inv_base, rem);
    S128 q{};
    if (!v.neg) {
        q = S128{false, qmag};
        digit = static_cast<int64_t>(rem);
    } else if (rem == 0) {
        q = S128{true, qmag};
        digit = 0;
    } else {
        q = S128{true, d_add_u128(qmag, d_from_u64(1))};
        digit = static_cast<int64_t>(base - rem);
    }
    if (static_cast<uint64_t>(digit) > base / 2) {
        digit -= static_cast<int64_t>(base);
        q = d_add_s128(q, d_make_s128_i64(1));
    }
    return q;
}

// The normal even-base alphabet [1-b/2, b/2] has b values and therefore
// misses exactly one residue modulo b^m+1.  This lower-tie variant uses
// [-b/2, b/2-1] and is only used by the verified rare-residue fallback.
__device__ S128 d_centered_divrem_lower_tie(
    S128 v, uint64_t base, double inv_base, int64_t& digit) {
    uint64_t rem = 0;
    U128 qmag = d_divmod_u128_u64(v.mag, base, inv_base, rem);
    S128 q{};
    if (!v.neg) {
        q = S128{false, qmag};
        digit = static_cast<int64_t>(rem);
    } else if (rem == 0) {
        q = S128{true, qmag};
        digit = 0;
    } else {
        q = S128{true, d_add_u128(qmag, d_from_u64(1))};
        digit = static_cast<int64_t>(base - rem);
    }
    const uint64_t half = base / 2;
    if (static_cast<uint64_t>(digit) >= half) {
        digit -= static_cast<int64_t>(base);
        q = d_add_s128(q, d_make_s128_i64(1));
    }
    return q;
}

__global__ void carry_blocks_kernel(const S128* raw_coeff, int64_t* digits, S128* carry_out,
                                    const S128* carry_in, int m, int block_size,
                                    uint64_t base, double inv_base) {
    const int block = blockIdx.x;
    const int start = block * block_size;
    if (start >= m) return;
    const int end = min(start + block_size, m);
    S128 carry = carry_in[block];
    for (int i = start; i < end; ++i) {
        S128 v = d_add_s128(raw_coeff[i], carry);
        int64_t digit = 0;
        S128 q = d_centered_divrem(v, base, inv_base, digit);
        digits[i] = digit;
        carry = q;
    }
    carry_out[block] = carry;
}

__global__ void carry_all_kernel(const S128* raw_coeff, int64_t* digits, S128* carry_out,
                                 const S128* carry_in, int m, uint64_t base,
                                 double inv_base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    S128 v = d_add_s128(raw_coeff[i], carry_in[i]);
    int64_t digit = 0;
    S128 q = d_centered_divrem(v, base, inv_base, digit);
    digits[i] = digit;
    carry_out[i] = q;
}

__global__ void update_block_carries_kernel(S128* carry_in, const S128* carry_out, int num_blocks, int* changed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_blocks) return;
    S128 next = (i == 0) ? d_neg_s128(carry_out[num_blocks - 1]) : carry_out[i - 1];
    if (!d_eq_s128(carry_in[i], next)) atomicExch(changed, 1);
    carry_in[i] = next;
}

__global__ void update_block_carries_plain_kernel(S128* carry_in, const S128* carry_out, int num_blocks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_blocks) return;
    carry_in[i] = (i == 0) ? d_neg_s128(carry_out[num_blocks - 1]) : carry_out[i - 1];
}

// A nonconverged speculative step is never accepted: its whole batch is
// restored and replayed through the original adaptive normalizer on the host.
__global__ void latch_carry_failure_kernel(const int* changed, int* sticky) {
    if (*changed != 0) *sticky = 1;
}

__global__ void digits_to_rns3_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                                      const int64_t* digits, int m, int64_t* out) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const int64_t digit = digits[i];
    if (out != nullptr) out[i] = digit;
    r0[i] = d_i64_to_mod(digit, p0);
    r1[i] = d_i64_to_mod(digit, p1);
    r2[i] = d_i64_to_mod(digit, p2);
    r0[i + m] = 0;
    r1[i + m] = 0;
    r2[i + m] = 0;
}

__global__ void digits_to_rns_kernel(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                                     const int64_t* digits, int m, int64_t* out) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t p2 = 469762049u;
    constexpr uint32_t p3 = 1224736769u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const int64_t digit = digits[i];
    if (out != nullptr) out[i] = digit;
    r0[i] = d_i64_to_mod(digit, p0);
    r1[i] = d_i64_to_mod(digit, p1);
    r2[i] = d_i64_to_mod(digit, p2);
    r3[i] = d_i64_to_mod(digit, p3);
    r0[i + m] = 0;
    r1[i + m] = 0;
    r2[i + m] = 0;
    r3[i + m] = 0;
}

__global__ void try_even_special_residue_kernel(
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    int mod_count, int m, uint64_t base, double inv_base,
    S128* coeff, int64_t* digits, int64_t* out, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *status = 1;
    if ((base & 1ull) != 0 || m <= 0) return;

    bool converged = false;
    S128 carry{};
    for (int pass = 0; pass < 32; ++pass) {
        carry = S128{};
        for (int i = 0; i < m; ++i) {
            const S128 value = d_add_s128(coeff[i], carry);
            int64_t digit = 0;
            carry = d_centered_divrem_lower_tie(
                value, base, inv_base, digit);
            coeff[i] = d_make_s128_i64(digit);
        }
        if (d_is_zero(carry.mag)) {
            converged = true;
            break;
        }
        coeff[0] = d_sub_s128(coeff[0], carry);
    }
    if (!converged) return;

    const int64_t half = static_cast<int64_t>(base / 2);
    for (int i = 0; i < m; ++i) {
        if (coeff[i].mag.hi != 0 ||
            coeff[i].mag.lo > static_cast<uint64_t>(INT64_MAX)) {
            return;
        }
        const int64_t magnitude = static_cast<int64_t>(coeff[i].mag.lo);
        const int64_t digit = coeff[i].neg ? -magnitude : magnitude;
        const int64_t expected = i == 0 ? -half : 1 - half;
        if (digit != expected) return;
        digits[i] = digit;
    }
    for (int i = 0; i < m; ++i) {
        const int64_t digit = digits[i];
        if (out != nullptr) out[i] = digit;
        r0[i] = d_i64_to_mod(digit, kMods[0].p);
        r1[i] = d_i64_to_mod(digit, kMods[1].p);
        r2[i] = d_i64_to_mod(digit, kMods[2].p);
        if (mod_count == 4) r3[i] = d_i64_to_mod(digit, kMods[3].p);
        r0[i + m] = 0;
        r1[i + m] = 0;
        r2[i + m] = 0;
        if (mod_count == 4) r3[i + m] = 0;
    }
    *status = 0;
}

void cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::string msg = where;
        msg += ": ";
        msg += cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

void normalize4_split(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                      int m, uint64_t base, uint32_t scale, S128* coeff, int64_t* out, int* status,
                      const char* tag) {
    const int threads = 256;
    const int blocks = (m + threads - 1) / threads;
    const double inv_base = 1.0 / static_cast<double>(base);
    crt4_fold_kernel<<<blocks, threads>>>(r0, r1, r2, r3, m, scale, coeff);
    cuda_check(cudaGetLastError(), (std::string(tag) + " crt4_fold launch").c_str());
    carry_normalize_serial_kernel<<<1, 1>>>(r0, r1, r2, r3, m, base, inv_base, scale,
                                             coeff, out, status);
    cuda_check(cudaGetLastError(), (std::string(tag) + " carry_normalize launch").c_str());
    cuda_check(cudaDeviceSynchronize(), (std::string(tag) + " normalize synchronize").c_str());
}

int normalize3_blocked(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                       int m, uint64_t base, double inv_base, uint32_t scale,
                       S128* raw_coeff, int64_t* digits, int64_t* out,
                       S128* carry_in, S128* carry_out, int num_carry_blocks, int carry_block_size,
                       int* changed, const char* tag) {
    const int threads = 256;
    const int coeff_blocks = (m + threads - 1) / threads;
    crt3_fold_kernel<<<coeff_blocks, threads>>>(r0, r1, r2, m, scale, raw_coeff);
    cuda_check(cudaGetLastError(), (std::string(tag) + " crt3_fold launch").c_str());
    cuda_check(cudaMemset(carry_in, 0, sizeof(S128) * num_carry_blocks), (std::string(tag) + " clear carry_in").c_str());
    cuda_check(cudaMemset(carry_out, 0, sizeof(S128) * num_carry_blocks), (std::string(tag) + " clear carry_out").c_str());

    bool converged = false;
    int iterations = 0;
    const int max_iters = num_carry_blocks + 8;
    const int carry_update_blocks = (num_carry_blocks + threads - 1) / threads;

    auto update_and_check = [&]() {
        cuda_check(cudaMemset(changed, 0, sizeof(int)), (std::string(tag) + " clear changed").c_str());
        update_block_carries_kernel<<<carry_update_blocks, threads>>>(carry_in, carry_out, num_carry_blocks, changed);
        cuda_check(cudaGetLastError(), (std::string(tag) + " update carries launch").c_str());
        int h_changed = 0;
        cuda_check(cudaMemcpy(&h_changed, changed, sizeof(int), cudaMemcpyDeviceToHost),
                   (std::string(tag) + " copy changed").c_str());
        return h_changed == 0;
    };

    carry_blocks_kernel<<<num_carry_blocks, 1>>>(raw_coeff, digits, carry_out, carry_in, m, carry_block_size, base, inv_base);
    cuda_check(cudaGetLastError(), (std::string(tag) + " carry_blocks launch").c_str());
    update_block_carries_plain_kernel<<<carry_update_blocks, threads>>>(carry_in, carry_out, num_carry_blocks);
    cuda_check(cudaGetLastError(), (std::string(tag) + " update carries plain launch").c_str());
    iterations = 1;

    if (max_iters > 1) {
        carry_blocks_kernel<<<num_carry_blocks, 1>>>(raw_coeff, digits, carry_out, carry_in, m, carry_block_size, base, inv_base);
        cuda_check(cudaGetLastError(), (std::string(tag) + " carry_blocks launch").c_str());
        iterations = 2;
        converged = update_and_check();
    }

    for (int iter = 2; !converged && iter < max_iters; ++iter) {
        carry_blocks_kernel<<<num_carry_blocks, 1>>>(raw_coeff, digits, carry_out, carry_in, m, carry_block_size, base, inv_base);
        cuda_check(cudaGetLastError(), (std::string(tag) + " carry_blocks launch").c_str());
        iterations = iter + 1;
        if (update_and_check()) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        try_even_special_residue_kernel<<<1, 1>>>(
            r0, r1, r2, nullptr, 3, m, base, inv_base,
            raw_coeff, digits, out, changed);
        cuda_check(cudaGetLastError(),
                   (std::string(tag) + " special residue launch").c_str());
        int special_status = 1;
        cuda_check(cudaMemcpy(&special_status, changed, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   (std::string(tag) + " special residue status").c_str());
        if (special_status == 0) {
            std::cout << tag
                      << ": recovered unique even-base centered residue\n";
            return iterations + 1;
        }
        throw std::runtime_error(std::string(tag) + " blocked carry did not converge");
    }

    digits_to_rns3_kernel<<<coeff_blocks, threads>>>(r0, r1, r2, digits, m, out);
    cuda_check(cudaGetLastError(), (std::string(tag) + " digits_to_rns3 launch").c_str());
    cuda_check(cudaDeviceSynchronize(), (std::string(tag) + " normalize synchronize").c_str());
    return iterations;
}

void launch_normalize3_relaxed_prefix(
    uint32_t* r0, uint32_t* r1, uint32_t* r2,
    int m, uint64_t base, double inv_base, uint32_t scale,
    S128* raw_coeff, int64_t* digits, S128* carry_in, S128* carry_out,
    int* changed, cudaStream_t stream) {
    const int threads = 256;
    const int coeff_blocks = (m + threads - 1) / threads;
    crt3_fold_carry_kernel<<<coeff_blocks, threads, 0, stream>>>(
        r0, r1, r2, m, base, inv_base, scale,
        raw_coeff, digits, carry_out);
    cuda_check(cudaGetLastError(), "normalize3 prefix crt3_fold_carry launch");
    update_block_carries_plain_kernel<<<coeff_blocks, threads, 0, stream>>>(
        carry_in, carry_out, m);
    cuda_check(cudaGetLastError(), "normalize3 prefix update1 launch");
    carry_all_kernel<<<coeff_blocks, threads, 0, stream>>>(
        raw_coeff, digits, carry_out, carry_in, m, base, inv_base);
    cuda_check(cudaGetLastError(), "normalize3 prefix carry2 launch");
    update_block_carries_plain_kernel<<<coeff_blocks, threads, 0, stream>>>(
        carry_in, carry_out, m);
    cuda_check(cudaGetLastError(), "normalize3 prefix update2 launch");
    carry_all_kernel<<<coeff_blocks, threads, 0, stream>>>(
        raw_coeff, digits, carry_out, carry_in, m, base, inv_base);
    cuda_check(cudaGetLastError(), "normalize3 prefix carry3 launch");
    cuda_check(cudaMemsetAsync(changed, 0, sizeof(int), stream),
               "normalize3 prefix clear changed");
    update_block_carries_kernel<<<coeff_blocks, threads, 0, stream>>>(
        carry_in, carry_out, m, changed);
    cuda_check(cudaGetLastError(), "normalize3 prefix convergence launch");
}

int normalize3_relaxed(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                       int m, uint64_t base, double inv_base, uint32_t scale,
                       S128* raw_coeff, int64_t* digits, int64_t* out,
                       S128* carry_in, S128* carry_out, int num_carry_blocks, int carry_block_size,
                       int* changed, const char* tag,
                       bool prefix_already_launched = false,
                       cudaStream_t stream = 0,
                       int* pinned_host_changed = nullptr) {
    const int threads = 256;
    const int coeff_blocks = (m + threads - 1) / threads;
    if (!prefix_already_launched) {
        launch_normalize3_relaxed_prefix(
            r0, r1, r2, m, base, inv_base, scale,
            raw_coeff, digits, carry_in, carry_out, changed, stream);
    }

    int local_changed = 0;
    int* host_changed = pinned_host_changed != nullptr
        ? pinned_host_changed : &local_changed;
    cuda_check(cudaMemcpyAsync(host_changed, changed, sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               (std::string(tag) + " copy changed").c_str());
    cuda_check(cudaStreamSynchronize(stream),
               (std::string(tag) + " prefix synchronize").c_str());
    bool converged = *host_changed == 0;
    int iterations = 3;
    constexpr int max_iters = 32;
#ifdef GFPS_DIAGNOSTIC_FORCE_EXTRA_CARRY
    converged = false;
#endif

    auto update_and_check = [&]() {
        cuda_check(cudaMemsetAsync(changed, 0, sizeof(int), stream),
                   (std::string(tag) + " clear changed").c_str());
        update_block_carries_kernel<<<coeff_blocks, threads, 0, stream>>>(
            carry_in, carry_out, m, changed);
        cuda_check(cudaGetLastError(), (std::string(tag) + " update relaxed carries launch").c_str());
        *host_changed = 0;
        cuda_check(cudaMemcpyAsync(host_changed, changed, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   (std::string(tag) + " copy changed").c_str());
        cuda_check(cudaStreamSynchronize(stream),
                   (std::string(tag) + " carry synchronize").c_str());
        return *host_changed == 0;
    };

    for (int iter = 3; !converged && iter < max_iters; ++iter) {
        carry_all_kernel<<<coeff_blocks, threads, 0, stream>>>(
            raw_coeff, digits, carry_out, carry_in, m, base, inv_base);
        cuda_check(cudaGetLastError(), (std::string(tag) + " carry_all launch").c_str());
        iterations = iter + 1;
        if (update_and_check()) {
            converged = true;
            break;
        }
    }
#ifdef GFPS_DIAGNOSTIC_FORCE_BLOCKED_FALLBACK
    converged = false;
#endif
    if (!converged) {
        cuda_check(cudaStreamSynchronize(stream),
                   (std::string(tag) + " before blocked fallback").c_str());
        try_even_special_residue_kernel<<<1, 1>>>(
            r0, r1, r2, nullptr, 3, m, base, inv_base,
            raw_coeff, digits, out, changed);
        cuda_check(cudaGetLastError(),
                   (std::string(tag) + " special residue launch").c_str());
        int special_status = 1;
        cuda_check(cudaMemcpy(&special_status, changed, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   (std::string(tag) + " special residue status").c_str());
        if (special_status == 0) {
            std::cout << tag
                      << ": recovered unique even-base centered residue\n";
            return iterations + 1;
        }
        return normalize3_blocked(r0, r1, r2, m, base, inv_base, scale, raw_coeff, digits, out,
                                  carry_in, carry_out, num_carry_blocks, carry_block_size,
                                  changed, tag);
    }

    digits_to_rns3_kernel<<<coeff_blocks, threads, 0, stream>>>(
        r0, r1, r2, digits, m, out);
    cuda_check(cudaGetLastError(), (std::string(tag) + " digits_to_rns3 launch").c_str());
    cuda_check(cudaStreamSynchronize(stream),
               (std::string(tag) + " normalize synchronize").c_str());
    return iterations;
}

int normalize4_blocked(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                       int m, uint64_t base, double inv_base, uint32_t scale,
                       S128* raw_coeff, int64_t* digits, int64_t* out,
                       S128* carry_in, S128* carry_out, int num_carry_blocks, int carry_block_size,
                       int* changed, const char* tag) {
    const int threads = 256;
    const int coeff_blocks = (m + threads - 1) / threads;
    crt4_fold_kernel<<<coeff_blocks, threads>>>(r0, r1, r2, r3, m, scale, raw_coeff);
    cuda_check(cudaGetLastError(), (std::string(tag) + " crt4_fold launch").c_str());
    cuda_check(cudaMemset(carry_in, 0, sizeof(S128) * num_carry_blocks), (std::string(tag) + " clear carry_in").c_str());
    cuda_check(cudaMemset(carry_out, 0, sizeof(S128) * num_carry_blocks), (std::string(tag) + " clear carry_out").c_str());

    bool converged = false;
    int iterations = 0;
    const int max_iters = num_carry_blocks + 8;
    const int carry_update_blocks = (num_carry_blocks + threads - 1) / threads;

    auto update_and_check = [&]() {
        cuda_check(cudaMemset(changed, 0, sizeof(int)), (std::string(tag) + " clear changed").c_str());
        update_block_carries_kernel<<<carry_update_blocks, threads>>>(carry_in, carry_out, num_carry_blocks, changed);
        cuda_check(cudaGetLastError(), (std::string(tag) + " update carries launch").c_str());
        int h_changed = 0;
        cuda_check(cudaMemcpy(&h_changed, changed, sizeof(int), cudaMemcpyDeviceToHost),
                   (std::string(tag) + " copy changed").c_str());
        return h_changed == 0;
    };

    carry_blocks_kernel<<<num_carry_blocks, 1>>>(raw_coeff, digits, carry_out, carry_in, m, carry_block_size, base, inv_base);
    cuda_check(cudaGetLastError(), (std::string(tag) + " carry_blocks launch").c_str());
    update_block_carries_plain_kernel<<<carry_update_blocks, threads>>>(carry_in, carry_out, num_carry_blocks);
    cuda_check(cudaGetLastError(), (std::string(tag) + " update carries plain launch").c_str());
    iterations = 1;

    if (max_iters > 1) {
        carry_blocks_kernel<<<num_carry_blocks, 1>>>(raw_coeff, digits, carry_out, carry_in, m, carry_block_size, base, inv_base);
        cuda_check(cudaGetLastError(), (std::string(tag) + " carry_blocks launch").c_str());
        iterations = 2;
        converged = update_and_check();
    }

    for (int iter = 2; !converged && iter < max_iters; ++iter) {
        carry_blocks_kernel<<<num_carry_blocks, 1>>>(raw_coeff, digits, carry_out, carry_in, m, carry_block_size, base, inv_base);
        cuda_check(cudaGetLastError(), (std::string(tag) + " carry_blocks launch").c_str());
        iterations = iter + 1;
        if (update_and_check()) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        try_even_special_residue_kernel<<<1, 1>>>(
            r0, r1, r2, r3, 4, m, base, inv_base,
            raw_coeff, digits, out, changed);
        cuda_check(cudaGetLastError(),
                   (std::string(tag) + " special residue launch").c_str());
        int special_status = 1;
        cuda_check(cudaMemcpy(&special_status, changed, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   (std::string(tag) + " special residue status").c_str());
        if (special_status == 0) {
            std::cout << tag
                      << ": recovered unique even-base centered residue\n";
            return iterations + 1;
        }
        throw std::runtime_error(std::string(tag) + " blocked carry did not converge");
    }

    digits_to_rns_kernel<<<coeff_blocks, threads>>>(r0, r1, r2, r3, digits, m, out);
    cuda_check(cudaGetLastError(), (std::string(tag) + " digits_to_rns launch").c_str());
    cuda_check(cudaDeviceSynchronize(), (std::string(tag) + " normalize synchronize").c_str());
    return iterations;
}

void launch_normalize4_relaxed_prefix(
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    int m, uint64_t base, double inv_base, uint32_t scale,
    S128* raw_coeff, int64_t* digits, S128* carry_in, S128* carry_out,
    int* changed, cudaStream_t stream) {
    const int threads = 256;
    const int coeff_blocks = (m + threads - 1) / threads;
    crt4_fold_carry_kernel<<<coeff_blocks, threads, 0, stream>>>(
        r0, r1, r2, r3, m, base, inv_base, scale,
        raw_coeff, digits, carry_out);
    cuda_check(cudaGetLastError(), "normalize4 prefix crt4_fold_carry launch");
    update_block_carries_plain_kernel<<<coeff_blocks, threads, 0, stream>>>(
        carry_in, carry_out, m);
    cuda_check(cudaGetLastError(), "normalize4 prefix update1 launch");
    carry_all_kernel<<<coeff_blocks, threads, 0, stream>>>(
        raw_coeff, digits, carry_out, carry_in, m, base, inv_base);
    cuda_check(cudaGetLastError(), "normalize4 prefix carry2 launch");
    update_block_carries_plain_kernel<<<coeff_blocks, threads, 0, stream>>>(
        carry_in, carry_out, m);
    cuda_check(cudaGetLastError(), "normalize4 prefix update2 launch");
    carry_all_kernel<<<coeff_blocks, threads, 0, stream>>>(
        raw_coeff, digits, carry_out, carry_in, m, base, inv_base);
    cuda_check(cudaGetLastError(), "normalize4 prefix carry3 launch");
    cuda_check(cudaMemsetAsync(changed, 0, sizeof(int), stream),
               "normalize4 prefix clear changed");
    update_block_carries_kernel<<<coeff_blocks, threads, 0, stream>>>(
        carry_in, carry_out, m, changed);
    cuda_check(cudaGetLastError(), "normalize4 prefix convergence launch");
}

int normalize4_relaxed(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                       int m, uint64_t base, double inv_base, uint32_t scale,
                       S128* raw_coeff, int64_t* digits, int64_t* out,
                       S128* carry_in, S128* carry_out, int num_carry_blocks, int carry_block_size,
                       int* changed, const char* tag,
                       bool prefix_already_launched = false,
                       cudaStream_t stream = 0,
                       int* pinned_host_changed = nullptr) {
    const int threads = 256;
    const int coeff_blocks = (m + threads - 1) / threads;
    if (!prefix_already_launched) {
        launch_normalize4_relaxed_prefix(
            r0, r1, r2, r3, m, base, inv_base, scale,
            raw_coeff, digits, carry_in, carry_out, changed, stream);
    }

    int local_changed = 0;
    int* host_changed = pinned_host_changed != nullptr
        ? pinned_host_changed : &local_changed;
    cuda_check(cudaMemcpyAsync(host_changed, changed, sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               (std::string(tag) + " copy changed").c_str());
    cuda_check(cudaStreamSynchronize(stream),
               (std::string(tag) + " prefix synchronize").c_str());
    bool converged = *host_changed == 0;
    int iterations = 3;
    constexpr int max_iters = 32;
#ifdef GFPS_DIAGNOSTIC_FORCE_EXTRA_CARRY
    converged = false;
#endif

    auto update_and_check = [&]() {
        cuda_check(cudaMemsetAsync(changed, 0, sizeof(int), stream),
                   (std::string(tag) + " clear changed").c_str());
        update_block_carries_kernel<<<coeff_blocks, threads, 0, stream>>>(
            carry_in, carry_out, m, changed);
        cuda_check(cudaGetLastError(), (std::string(tag) + " update relaxed carries launch").c_str());
        *host_changed = 0;
        cuda_check(cudaMemcpyAsync(host_changed, changed, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   (std::string(tag) + " copy changed").c_str());
        cuda_check(cudaStreamSynchronize(stream),
                   (std::string(tag) + " carry synchronize").c_str());
        return *host_changed == 0;
    };

    for (int iter = 3; !converged && iter < max_iters; ++iter) {
        carry_all_kernel<<<coeff_blocks, threads, 0, stream>>>(
            raw_coeff, digits, carry_out, carry_in, m, base, inv_base);
        cuda_check(cudaGetLastError(), (std::string(tag) + " carry_all launch").c_str());
        iterations = iter + 1;
        if (update_and_check()) {
            converged = true;
            break;
        }
    }
#ifdef GFPS_DIAGNOSTIC_FORCE_BLOCKED_FALLBACK
    converged = false;
#endif
    if (!converged) {
        cuda_check(cudaStreamSynchronize(stream),
                   (std::string(tag) + " before blocked fallback").c_str());
        try_even_special_residue_kernel<<<1, 1>>>(
            r0, r1, r2, r3, 4, m, base, inv_base,
            raw_coeff, digits, out, changed);
        cuda_check(cudaGetLastError(),
                   (std::string(tag) + " special residue launch").c_str());
        int special_status = 1;
        cuda_check(cudaMemcpy(&special_status, changed, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   (std::string(tag) + " special residue status").c_str());
        if (special_status == 0) {
            std::cout << tag
                      << ": recovered unique even-base centered residue\n";
            return iterations + 1;
        }
        return normalize4_blocked(r0, r1, r2, r3, m, base, inv_base, scale, raw_coeff, digits, out,
                                  carry_in, carry_out, num_carry_blocks, carry_block_size,
                                  changed, tag);
    }

    digits_to_rns_kernel<<<coeff_blocks, threads, 0, stream>>>(
        r0, r1, r2, r3, digits, m, out);
    cuda_check(cudaGetLastError(), (std::string(tag) + " digits_to_rns launch").c_str());
    cuda_check(cudaStreamSynchronize(stream),
               (std::string(tag) + " normalize synchronize").c_str());
    return iterations;
}

void ntt_inplace_workspace(uint32_t* d_a, uint32_t* d_tmp, uint32_t* d_roots, int log_len, uint32_t p, uint32_t g, bool inverse) {
    const int len_total = 1 << log_len;
    const int threads = 256;
    const int blocks = (len_total + threads - 1) / threads;
    bit_reverse_kernel<<<blocks, threads>>>(d_tmp, d_a, log_len);
    cuda_check(cudaGetLastError(), "bit_reverse launch");
    cuda_check(cudaMemcpy(d_a, d_tmp, sizeof(uint32_t) * len_total, cudaMemcpyDeviceToDevice), "bit_reverse copy");

    for (int len = 2; len <= len_total; len <<= 1) {
        const int half = len >> 1;
        uint32_t wlen = pow_mod_host(g, (p - 1ull) / static_cast<uint64_t>(len), p);
        if (inverse) wlen = pow_mod_host(wlen, p - 2ull, p);
        const int root_blocks = (half + threads - 1) / threads;
        make_twiddle_kernel<<<root_blocks, threads>>>(d_roots, half, wlen, p);
        cuda_check(cudaGetLastError(), "make_twiddle launch");
        const int butterflies = len_total / 2;
        const int y = limited_ntt_grid_y(butterflies, threads);
        ntt_stage_kernel<<<dim3(1, y), threads>>>(d_a, d_roots, len, butterflies, p);
        cuda_check(cudaGetLastError(), "ntt_stage launch");
    }

    if (inverse) {
        const uint32_t inv_len = pow_mod_host(static_cast<uint32_t>(len_total), p - 2ull, p);
        scale_kernel<<<blocks, threads>>>(d_a, len_total, inv_len, p);
        cuda_check(cudaGetLastError(), "scale launch");
    }
    cuda_check(cudaDeviceSynchronize(), "ntt synchronize");
}

std::vector<int> make_twiddle_offsets(int log_len) {
    std::vector<int> offsets(log_len + 1, 0);
    int total = 0;
    for (int stage = 1; stage <= log_len; ++stage) {
        offsets[stage] = total;
        total += 1 << (stage - 1);
    }
    return offsets;
}

std::vector<uint32_t> make_bit_reverse_table_host(int log_len) {
    const int n = 1 << log_len;
    std::vector<uint32_t> rev(n);
    for (int i = 0; i < n; ++i) {
        uint32_t x = static_cast<uint32_t>(i);
        uint32_t r = 0;
        for (int b = 0; b < log_len; ++b) {
            r = (r << 1) | (x & 1u);
            x >>= 1;
        }
        rev[i] = r;
    }
    return rev;
}

std::vector<uint32_t> make_twiddle_table_host(int log_len, uint32_t p, uint32_t g, bool inverse) {
    const auto offsets = make_twiddle_offsets(log_len);
    const int total_roots = (1 << log_len) - 1;
    std::vector<uint32_t> table(total_roots);
    for (int stage = 1; stage <= log_len; ++stage) {
        const int len = 1 << stage;
        const int half = len >> 1;
        uint32_t wlen = pow_mod_host(g, (p - 1ull) / static_cast<uint64_t>(len), p);
        if (inverse) wlen = pow_mod_host(wlen, p - 2ull, p);
        uint64_t w = 1;
        const int off = offsets[stage];
        for (int j = 0; j < half; ++j) {
            table[off + j] = static_cast<uint32_t>(w);
            w = (w * wlen) % p;
        }
    }
    return table;
}

void ntt_inplace_precomputed(uint32_t* d_a, uint32_t* d_tmp, int log_len, uint32_t p, const uint32_t* d_rev,
                             bool inverse, const uint32_t* d_twiddles, const std::vector<int>& offsets) {
    const int len_total = 1 << log_len;
    const int threads = 256;
    const int blocks = (len_total + threads - 1) / threads;
    bit_reverse_table_kernel<<<blocks, threads>>>(d_tmp, d_a, d_rev, len_total);
    cuda_check(cudaGetLastError(), "bit_reverse launch");
    cuda_check(cudaMemcpy(d_a, d_tmp, sizeof(uint32_t) * len_total, cudaMemcpyDeviceToDevice), "bit_reverse copy");

    int stage = 1;
    for (; stage + 1 <= log_len; stage += 2) {
        const int len = 1 << (stage + 1);
        const int groups = len_total / 4;
        const int y = limited_ntt_grid_y(groups, threads);
        ntt_stage2_kernel<<<dim3(1, y), threads>>>(d_a,
                                                   d_twiddles + offsets[stage],
                                                   d_twiddles + offsets[stage + 1],
                                                   len, groups, p);
        cuda_check(cudaGetLastError(), "ntt_stage2 launch");
    }
    if (stage <= log_len) {
        const int len = 1 << stage;
        const int butterflies = len_total / 2;
        const int y = limited_ntt_grid_y(butterflies, threads);
        ntt_stage_kernel<<<dim3(1, y), threads>>>(d_a, d_twiddles + offsets[stage], len, butterflies, p);
        cuda_check(cudaGetLastError(), "ntt_stage launch");
    }

    if (inverse) {
        const uint32_t inv_len = pow_mod_host(static_cast<uint32_t>(len_total), p - 2ull, p);
        scale_kernel<<<blocks, threads>>>(d_a, len_total, inv_len, p);
        cuda_check(cudaGetLastError(), "scale launch");
    }
    cuda_check(cudaDeviceSynchronize(), "ntt synchronize");
}

void ntt3_inplace_precomputed(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                              uint32_t* tmp0, uint32_t* tmp1, uint32_t* tmp2,
                              int log_len, const uint32_t* d_rev, bool inverse,
                              const uint32_t* tw0, const uint32_t* tw1, const uint32_t* tw2,
                              const std::vector<int>& offsets, bool synchronize,
                              const uint32_t* inverse_lengths,
                              cudaStream_t stream = 0, bool reverse_input = true,
                              bool shared_low = false, bool shared_already_done = false) {
    const int len_total = 1 << log_len;
    const int threads = 256;
    const int blocks = (len_total + threads - 1) / threads;
    if (reverse_input) {
    bit_reverse_table3_kernel<<<blocks, threads, 0, stream>>>(
        tmp0, tmp1, tmp2, r0, r1, r2, d_rev, len_total);
    cuda_check(cudaGetLastError(), "bit_reverse3 launch");
    copy3_kernel<<<blocks, threads, 0, stream>>>(
        r0, r1, r2, tmp0, tmp1, tmp2, len_total);
    cuda_check(cudaGetLastError(), "bit_reverse3 copy launch");
    }

    int stage = 1;
    if (shared_low && log_len >= 10) {
        if (!shared_already_done) {
        ntt_shared1024_rns_kernel<false><<<len_total/1024,256,0,stream>>>(
            r0,r1,r2,nullptr,tw0,tw1,tw2,nullptr,3,nullptr,nullptr,nullptr,nullptr);
        cuda_check(cudaGetLastError(), "shared inverse low stages3");
        }
        stage=11;
    }
    for (; stage + 1 <= log_len; stage += 2) {
        const int len = 1 << (stage + 1);
        const int groups = len_total / 4;
        const int y = limited_ntt_grid_y(groups, threads);
        ntt_stage2_3_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1, r2,
                                                     tw0 + offsets[stage],
                                                     tw1 + offsets[stage],
                                                     tw2 + offsets[stage],
                                                     tw0 + offsets[stage + 1],
                                                     tw1 + offsets[stage + 1],
                                                     tw2 + offsets[stage + 1],
                                                     len, groups);
        cuda_check(cudaGetLastError(), "ntt_stage2_3 launch");
    }
    if (stage <= log_len) {
        const int len = 1 << stage;
        const int butterflies = len_total / 2;
        const int y = limited_ntt_grid_y(butterflies, threads);
        ntt_stage_3_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1, r2,
                                                    tw0 + offsets[stage],
                                                    tw1 + offsets[stage],
                                                    tw2 + offsets[stage],
                                                    len, butterflies);
        cuda_check(cudaGetLastError(), "ntt_stage_3 launch");
    }

    if (inverse) {
        scale3_kernel<<<blocks, threads, 0, stream>>>(r0, r1, r2, len_total,
                                           inverse_lengths[0], inverse_lengths[1],
                                           inverse_lengths[2]);
        cuda_check(cudaGetLastError(), "scale3 launch");
    }
    if (synchronize) cuda_check(cudaStreamSynchronize(stream), "ntt3 synchronize");
}

void ntt4_inplace_precomputed(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                              uint32_t* tmp0, uint32_t* tmp1, uint32_t* tmp2, uint32_t* tmp3,
                              int log_len, const uint32_t* d_rev, bool inverse,
                              const uint32_t* tw0, const uint32_t* tw1,
                              const uint32_t* tw2, const uint32_t* tw3,
                              const std::vector<int>& offsets, bool synchronize,
                              const uint32_t* inverse_lengths,
                              cudaStream_t stream = 0, bool reverse_input = true,
                              bool shared_low = false, bool shared_already_done = false) {
    const int len_total = 1 << log_len;
    const int threads = 256;
    const int blocks = (len_total + threads - 1) / threads;
    if (reverse_input) {
    bit_reverse_table4_kernel<<<blocks, threads, 0, stream>>>(
        tmp0, tmp1, tmp2, tmp3, r0, r1, r2, r3, d_rev, len_total);
    cuda_check(cudaGetLastError(), "bit_reverse4 launch");
    copy4_kernel<<<blocks, threads, 0, stream>>>(
        r0, r1, r2, r3, tmp0, tmp1, tmp2, tmp3, len_total);
    cuda_check(cudaGetLastError(), "bit_reverse4 copy launch");
    }

    int stage = 1;
    if (shared_low && log_len >= 10) {
        if (!shared_already_done) {
        ntt_shared1024_rns_kernel<false><<<len_total/1024,256,0,stream>>>(
            r0,r1,r2,r3,tw0,tw1,tw2,tw3,4,nullptr,nullptr,nullptr,nullptr);
        cuda_check(cudaGetLastError(), "shared inverse low stages4");
        }
        stage=11;
    }
    for (; stage + 1 <= log_len; stage += 2) {
        const int len = 1 << (stage + 1);
        const int groups = len_total / 4;
        const int y = limited_ntt_grid_y(groups, threads);
        ntt_stage2_4_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1, r2, r3,
                                                     tw0 + offsets[stage],
                                                     tw1 + offsets[stage],
                                                     tw2 + offsets[stage],
                                                     tw3 + offsets[stage],
                                                     tw0 + offsets[stage + 1],
                                                     tw1 + offsets[stage + 1],
                                                     tw2 + offsets[stage + 1],
                                                     tw3 + offsets[stage + 1],
                                                     len, groups);
        cuda_check(cudaGetLastError(), "ntt_stage2_4 launch");
    }
    if (stage <= log_len) {
        const int len = 1 << stage;
        const int butterflies = len_total / 2;
        const int y = limited_ntt_grid_y(butterflies, threads);
        ntt_stage_4_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1, r2, r3,
                                                    tw0 + offsets[stage],
                                                    tw1 + offsets[stage],
                                                    tw2 + offsets[stage],
                                                    tw3 + offsets[stage],
                                                    len, butterflies);
        cuda_check(cudaGetLastError(), "ntt_stage_4 launch");
    }

    if (inverse) {
        scale4_kernel<<<blocks, threads, 0, stream>>>(r0, r1, r2, r3, len_total,
                                           inverse_lengths[0], inverse_lengths[1],
                                           inverse_lengths[2], inverse_lengths[3]);
        cuda_check(cudaGetLastError(), "scale4 launch");
    }
    if (synchronize) cuda_check(cudaStreamSynchronize(stream), "ntt4 synchronize");
}

void ntt_inplace(uint32_t* d_a, uint32_t* d_tmp, int log_len, uint32_t p, uint32_t g, bool inverse) {
    const int len_total = 1 << log_len;
    uint32_t* d_roots = nullptr;
    cuda_check(cudaMalloc(&d_roots, sizeof(uint32_t) * (len_total / 2)), "malloc d_roots");
    try {
        ntt_inplace_workspace(d_a, d_tmp, d_roots, log_len, p, g, inverse);
        cudaFree(d_roots);
    } catch (...) {
        cudaFree(d_roots);
        throw;
    }
}

std::vector<std::vector<uint32_t>> cuda_linear_square_residues(const std::vector<int64_t>& a) {
    const int m = static_cast<int>(a.size());
    int log_m = 0;
    while ((1 << log_m) < m) ++log_m;
    if ((1 << log_m) != m) throw std::runtime_error("input length must be a power of two");
    const int log_len = log_m + 1;
    const int len = 1 << log_len;
    const int threads = 256;
    const int blocks = (len + threads - 1) / threads;

    int64_t* d_src = nullptr;
    uint32_t* d_a = nullptr;
    uint32_t* d_tmp = nullptr;
    cuda_check(cudaMalloc(&d_src, sizeof(int64_t) * m), "malloc d_src");
    cuda_check(cudaMalloc(&d_a, sizeof(uint32_t) * len), "malloc d_a");
    cuda_check(cudaMalloc(&d_tmp, sizeof(uint32_t) * len), "malloc d_tmp");
    cuda_check(cudaMemcpy(d_src, a.data(), sizeof(int64_t) * m, cudaMemcpyHostToDevice), "copy input");

    std::vector<std::vector<uint32_t>> out;
    for (const auto mod : kMods) {
        load_mod_kernel<<<blocks, threads>>>(d_a, d_src, m, len, mod.p);
        cuda_check(cudaGetLastError(), "load_mod launch");
        ntt_inplace(d_a, d_tmp, log_len, mod.p, mod.g, false);
        pointwise_square_kernel<<<blocks, threads>>>(d_a, len, mod.p);
        cuda_check(cudaGetLastError(), "pointwise_square launch");
        ntt_inplace(d_a, d_tmp, log_len, mod.p, mod.g, true);

        std::vector<uint32_t> lin(len);
        cuda_check(cudaMemcpy(lin.data(), d_a, sizeof(uint32_t) * len, cudaMemcpyDeviceToHost), "copy residues");
        out.push_back(std::move(lin));
    }

    cudaFree(d_src);
    cudaFree(d_a);
    cudaFree(d_tmp);
    return out;
}

std::vector<int64_t> cuda_square_dup_gpu_norm(const std::vector<int64_t>& a, uint64_t base, bool dup) {
    const int m = static_cast<int>(a.size());
    int log_m = 0;
    while ((1 << log_m) < m) ++log_m;
    if ((1 << log_m) != m) throw std::runtime_error("input length must be a power of two");
    if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("current GPU normalize path requires base < 2^63");
    }
    const int log_len = log_m + 1;
    const int len = 1 << log_len;
    const int threads = 256;
    const int blocks = (len + threads - 1) / threads;

    int64_t* d_src = nullptr;
    uint32_t* d_r[4] = {};
    uint32_t* d_tmp = nullptr;
    S128* d_coeff = nullptr;
    int64_t* d_out = nullptr;
    int* d_status = nullptr;

    cuda_check(cudaMalloc(&d_src, sizeof(int64_t) * m), "malloc d_src");
    cuda_check(cudaMemcpy(d_src, a.data(), sizeof(int64_t) * m, cudaMemcpyHostToDevice), "copy input");
    cuda_check(cudaMalloc(&d_tmp, sizeof(uint32_t) * len), "malloc d_tmp");
    for (int i = 0; i < 4; ++i) cuda_check(cudaMalloc(&d_r[i], sizeof(uint32_t) * len), "malloc d_r");
    cuda_check(cudaMalloc(&d_coeff, sizeof(S128) * m), "malloc d_coeff");
    cuda_check(cudaMalloc(&d_out, sizeof(int64_t) * m), "malloc d_out");
    cuda_check(cudaMalloc(&d_status, sizeof(int)), "malloc d_status");

    try {
        for (int mi = 0; mi < 4; ++mi) {
            const auto mod = kMods[mi];
            load_mod_kernel<<<blocks, threads>>>(d_r[mi], d_src, m, len, mod.p);
            cuda_check(cudaGetLastError(), "load_mod launch");
            ntt_inplace(d_r[mi], d_tmp, log_len, mod.p, mod.g, false);
            pointwise_square_kernel<<<blocks, threads>>>(d_r[mi], len, mod.p);
            cuda_check(cudaGetLastError(), "pointwise_square launch");
            ntt_inplace(d_r[mi], d_tmp, log_len, mod.p, mod.g, true);
        }

        normalize4_split(d_r[0], d_r[1], d_r[2], d_r[3], m, base, dup ? 2u : 1u,
                         d_coeff, d_out, d_status, "gpu_norm");

        int status = 0;
        cuda_check(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), "copy normalize status");
        if (status != 0) throw std::runtime_error("GPU normalization failed with status " + std::to_string(status));

        std::vector<int64_t> out(m);
        cuda_check(cudaMemcpy(out.data(), d_out, sizeof(int64_t) * m, cudaMemcpyDeviceToHost), "copy normalized output");

        cudaFree(d_status);
        cudaFree(d_out);
        cudaFree(d_coeff);
        for (auto& p : d_r) cudaFree(p);
        cudaFree(d_tmp);
        cudaFree(d_src);
        return out;
    } catch (...) {
        cudaFree(d_status);
        cudaFree(d_out);
        cudaFree(d_coeff);
        for (auto& p : d_r) cudaFree(p);
        cudaFree(d_tmp);
        cudaFree(d_src);
        throw;
    }
}

int checked_power_of_two_size(int n) {
    if (n < 1 || n > 20) throw std::runtime_error("n out of prototype range");
    return 1 << n;
}

int select_resident_mod_count(uint64_t base, int n) {
    using boost::multiprecision::cpp_int;
    if (base < 2) throw std::runtime_error("base must be at least 2");
    const int m = checked_power_of_two_size(n);
    const cpp_int max_digit = (cpp_int(base) + 1) / 2;
    const cpp_int twice_bound = cpp_int(2) * m * max_digit * max_digit;
    cpp_int product = 1;
    for (int count = 1; count <= 4; ++count) {
        product *= kMods[count - 1].p;
        if (count >= 3 && twice_bound < product) return count;
    }
    throw std::runtime_error(
        "CRT dynamic range is insufficient for this base/n; refusing unsafe calculation");
}

int checked_carry_block_size(int requested, int m) {
    if (requested < 1) throw std::runtime_error("invalid carry block size");
    return std::min(requested, m);
}

class GpuResidue {
public:
    explicit GpuResidue(uint64_t base, int n, int carry_block_size = 256)
        : _base(base), _m(checked_power_of_two_size(n)), _log_len(n + 1),
          _len(_m * 2), _mod_count(select_resident_mod_count(base, n)),
          _carry_block_size(checked_carry_block_size(carry_block_size, _m)),
          _num_carry_blocks((_m + _carry_block_size - 1) / _carry_block_size),
          _twiddle_offsets(make_twiddle_offsets(n + 1)) {
        if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("resident GPU path requires base < 2^63");
        }
        if (base < 2) throw std::runtime_error("base must be at least 2");
        _inv_base = 1.0 / static_cast<double>(base);
        const auto& algorithm = resident_algorithm_config();
        _negacyclic = !algorithm.reference_mode &&
                      std::getenv("GFPS_DISABLE_NEGACYCLIC") == nullptr;
        _dif = !algorithm.reference_mode && std::getenv("GFPS_DISABLE_DIF") == nullptr;
        _shared_low = _dif && std::getenv("GFPS_DISABLE_SHARED") == nullptr;
        _square_log_len = _log_len - (_negacyclic ? 1 : 0);
        _square_len = 1 << _square_log_len;
        _fused_tile = _shared_low && _square_log_len >= 10 &&
                      std::getenv("GFPS_DISABLE_FUSED_TILE") == nullptr;
        _batch_bits = algorithm.reference_mode ? 0 : algorithm.batch_bits;
        _fast_carry_passes = read_env_int("GFPS_CARRY_PASSES",
                                        _base < 1000000000000ull ? 4 : 3, 3, 8);
        if (gpu_duty_percent() != 100) _batch_bits = 0;
        _force_replay = algorithm.force_replay;
        std::cout << "resident-algorithm: ntt=" << (_negacyclic ? "negacyclic" : "padded")
                  << ", ntt_length=" << _square_len
                  << ", dif=" << (_dif ? "yes" : "no")
                  << ", shared=" << (_shared_low && _square_log_len >= 10 ? "yes" : "no")
                  << ", fused=" << (_fused_tile ? "yes" : "no")
                  << ", batch_bits=" << _batch_bits
                  << ", force_replay=" << (_force_replay ? "yes" : "no") << "\n";
        if (!ntt_blocks_are_forced()) {
            int device = 0;
            cudaDeviceProp properties{};
            cuda_check(cudaGetDevice(&device), "resident auto-blocks get device");
            cuda_check(cudaGetDeviceProperties(&properties, device),
                       "resident auto-blocks get properties");
            const int stage2_useful_blocks = std::max(
                1, ((_len / 4) + 255) / 256);
            gpu_throttle_config().ntt_blocks = std::min(
                stage2_useful_blocks,
                std::max(kDefaultNttBlocks,
                         properties.multiProcessorCount * 4));
            std::cout << "ntt-auto: sm_count="
                      << properties.multiProcessorCount
                      << ", ntt_length=" << _len
                      << ", selected=" << ntt_block_limit() << "\n";
        }
        try {
            cuda_check(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking),
                       "resident create stream");
            for (int i = 0; i < _mod_count; ++i) {
                _inverse_lengths[i] = pow_mod_host(
                    static_cast<uint32_t>(_len), kMods[i].p - 2ull, kMods[i].p);
                _square_inverse_lengths[i] = pow_mod_host(
                    static_cast<uint32_t>(_square_len), kMods[i].p - 2ull, kMods[i].p);
            }
            for (int i = 0; i < _mod_count; ++i) {
                cuda_check(cudaMalloc(&_d_tmp_r[i], sizeof(uint32_t) * _len), "resident malloc tmp");
            }
            cuda_check(cudaMalloc(&_d_rev, sizeof(uint32_t) * _len), "resident malloc bit reverse table");
            const auto rev = make_bit_reverse_table_host(_log_len);
            cuda_check(cudaMemcpy(_d_rev, rev.data(), sizeof(uint32_t) * _len, cudaMemcpyHostToDevice),
                       "resident copy bit reverse table");
            _square_rev = _d_rev;
            if (_negacyclic) {
                cuda_check(cudaMalloc(&_d_neg_rev, sizeof(uint32_t) * _m), "negacyclic malloc reverse table");
                const auto neg_rev = make_bit_reverse_table_host(_square_log_len);
                cuda_check(cudaMemcpy(_d_neg_rev, neg_rev.data(), sizeof(uint32_t) * _m,
                    cudaMemcpyHostToDevice), "negacyclic copy reverse table");
                _square_rev = _d_neg_rev;
                std::vector<uint32_t> twist(_m * _mod_count), untwist(_m * _mod_count);
                for (int j = 0; j < _mod_count; ++j) {
                    const uint32_t p = kMods[j].p;
                    const uint32_t psi = pow_mod_host(kMods[j].g, (p-1ull)/(_m*2ull), p);
                    const uint32_t invpsi = pow_mod_host(psi, p-2ull, p);
                    if (pow_mod_host(psi, _m, p) != p-1)
                        throw std::runtime_error("negacyclic twist order mismatch");
                    uint64_t u = 1, v = 1;
                    for (int i = 0; i < _m; ++i) {
                        twist[j*_m+i] = static_cast<uint32_t>(u);
                        untwist[j*_m+i] = static_cast<uint32_t>(v);
                        u = u*psi%p; v = v*invpsi%p;
                    }
                }
                cuda_check(cudaMalloc(&_d_twist, sizeof(uint32_t)*twist.size()), "negacyclic malloc twist");
                cuda_check(cudaMalloc(&_d_untwist, sizeof(uint32_t)*untwist.size()), "negacyclic malloc untwist");
                cuda_check(cudaMemcpy(_d_twist, twist.data(), sizeof(uint32_t)*twist.size(),
                    cudaMemcpyHostToDevice), "negacyclic copy twist");
                cuda_check(cudaMemcpy(_d_untwist, untwist.data(), sizeof(uint32_t)*untwist.size(),
                    cudaMemcpyHostToDevice), "negacyclic copy untwist");
            }
            const int total_roots = _len - 1;
            for (int i = 0; i < _mod_count; ++i) {
                cuda_check(cudaMalloc(&_d_twiddle_fwd[i], sizeof(uint32_t) * total_roots), "resident malloc fwd twiddles");
                cuda_check(cudaMalloc(&_d_twiddle_inv[i], sizeof(uint32_t) * total_roots), "resident malloc inv twiddles");
                const auto fwd = make_twiddle_table_host(_log_len, kMods[i].p, kMods[i].g, false);
                const auto inv = make_twiddle_table_host(_log_len, kMods[i].p, kMods[i].g, true);
                cuda_check(cudaMemcpy(_d_twiddle_fwd[i], fwd.data(), sizeof(uint32_t) * total_roots, cudaMemcpyHostToDevice),
                           "resident copy fwd twiddles");
                cuda_check(cudaMemcpy(_d_twiddle_inv[i], inv.data(), sizeof(uint32_t) * total_roots, cudaMemcpyHostToDevice),
                           "resident copy inv twiddles");
            }
            for (int i = 0; i < _mod_count; ++i) {
                cuda_check(cudaMalloc(&_d_r[i], sizeof(uint32_t) * _len), "resident malloc rns");
            }
            cuda_check(cudaMalloc(&_d_mul_rhs, sizeof(uint32_t) * _len), "resident malloc mul rhs");
            cuda_check(cudaMalloc(&_d_coeff, sizeof(S128) * _m), "resident malloc coeff");
            cuda_check(cudaMalloc(&_d_digits, sizeof(int64_t) * _m), "resident malloc normalized digits");
            cuda_check(cudaMalloc(&_d_carry_in, sizeof(S128) * _m), "resident malloc carry_in");
            cuda_check(cudaMalloc(&_d_carry_out, sizeof(S128) * _m), "resident malloc carry_out");
            cuda_check(cudaMalloc(&_d_changed, sizeof(int)), "resident malloc carry changed");
            cuda_check(cudaMallocHost(&_h_changed, sizeof(int)),
                       "resident malloc pinned changed flag");
            cuda_check(cudaMalloc(&_d_out, sizeof(int64_t) * _m), "resident malloc out");
            cuda_check(cudaMalloc(&_d_status, sizeof(int)), "resident malloc status");
            if (_batch_bits > 0) {
                for (int i = 0; i < _mod_count; ++i)
                    cuda_check(cudaMalloc(&_d_batch_snapshot[i], sizeof(uint32_t) * _len),
                               "batch malloc snapshot");
                cuda_check(cudaMalloc(&_d_batch_failed, sizeof(int)), "batch malloc sticky flag");
                _pending_dups.reserve(static_cast<size_t>(_batch_bits));
                std::cout << "carry-batching: batch_bits=" << _batch_bits
                          << ", fast_carry_passes=" << _fast_carry_passes
                          << ", exact adaptive replay on carry failure\n";
            }
            _graphs_enabled = GFPS_USE_CUDA_GRAPHS != 0 &&
                              std::getenv("GFPS_DISABLE_CUDA_GRAPHS") == nullptr;
            if (_graphs_enabled) {
                capture_square_graphs();
                std::cout << "cuda-graphs: enabled, rns_primes="
                          << _mod_count << ", variants=2\n";
            } else {
                std::cout << "cuda-graphs: disabled, rns_primes="
                          << _mod_count << "\n";
            }
        } catch (...) {
            release_resources_noexcept();
            throw;
        }
    }

    ~GpuResidue() {
        release_resources_noexcept();
    }

    GpuResidue(const GpuResidue&) = delete;
    GpuResidue& operator=(const GpuResidue&) = delete;

    void set_coeffs(const std::vector<int64_t>& a) {
        finish_pending();
        if (static_cast<int>(a.size()) != _m) throw std::runtime_error("bad coefficient vector size");
        int64_t* d_src = nullptr;
        cuda_check(cudaMalloc(&d_src, sizeof(int64_t) * _m), "resident malloc src");
        try {
            cuda_check(cudaMemcpy(d_src, a.data(), sizeof(int64_t) * _m, cudaMemcpyHostToDevice), "resident copy input");
            const int threads = 256;
            const int blocks = (_len + threads - 1) / threads;
            for (int i = 0; i < _mod_count; ++i) {
                load_mod_kernel<<<blocks, threads>>>(_d_r[i], d_src, _m, _len, kMods[i].p);
                cuda_check(cudaGetLastError(), "resident load_mod launch");
            }
            cuda_check(cudaDeviceSynchronize(), "resident load synchronize");
            cudaFree(d_src);
        } catch (...) {
            cudaFree(d_src);
            throw;
        }
    }

    [[maybe_unused]] void set_one() {
        std::vector<int64_t> one(_m, 0);
        one[0] = 1;
        set_coeffs(one);
    }

    [[maybe_unused]] void square_dup(bool dup) {
        if (_batch_bits > 0) {
            if (_pending_dups.empty()) {
                if (_timing_enabled) _batch_started = std::chrono::steady_clock::now();
                for (int i = 0; i < _mod_count; ++i)
                    cuda_check(cudaMemcpyAsync(_d_batch_snapshot[i], _d_r[i],
                        sizeof(uint32_t) * _len, cudaMemcpyDeviceToDevice, _stream),
                        "batch snapshot RNS");
                cuda_check(cudaMemsetAsync(_d_batch_failed, 0, sizeof(int), _stream),
                           "batch reset sticky flag");
            }
            if (_graphs_enabled) {
                cuda_check(cudaGraphLaunch(_square_graph_exec[dup ? 1 : 0], _stream),
                           "batch launch complete square graph");
            } else {
                if (_mod_count == 3) launch_square3_prefix(dup ? 2u : 1u);
                else launch_square4_prefix(dup ? 2u : 1u);
                launch_speculative_tail();
            }
            _pending_dups.push_back(dup ? 1 : 0);
            if (static_cast<int>(_pending_dups.size()) >= _batch_bits) finish_pending();
            return;
        }
        const auto throttle_started = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t_transform0;
        std::chrono::steady_clock::time_point t_norm0;
        int carry_iterations = 0;
        if (_timing_enabled) t_transform0 = std::chrono::steady_clock::now();
        if (_mod_count == 3) {
            const uint32_t scale = dup ? 2u : 1u;
            if (_graphs_enabled) {
                cuda_check(cudaGraphLaunch(_square_graph_exec[dup ? 1 : 0], _stream),
                           "resident launch square graph");
            } else {
                launch_square3_prefix(scale);
            }
            if (_timing_enabled) t_norm0 = std::chrono::steady_clock::now();
            carry_iterations = normalize3_relaxed(
                _d_r[0], _d_r[1], _d_r[2], _m, _base, _inv_base,
                scale, _d_coeff, _d_digits, nullptr, _d_carry_in,
                _d_carry_out, _num_carry_blocks, _carry_block_size,
                _d_changed, "resident square", true, _stream, _h_changed);
        } else if (_mod_count == 4) {
            const uint32_t scale = dup ? 2u : 1u;
            if (_graphs_enabled) {
                cuda_check(cudaGraphLaunch(_square_graph_exec[dup ? 1 : 0], _stream),
                           "resident launch square graph");
            } else {
                launch_square4_prefix(scale);
            }
            if (_timing_enabled) t_norm0 = std::chrono::steady_clock::now();
            carry_iterations = normalize4_relaxed(
                _d_r[0], _d_r[1], _d_r[2], _d_r[3],
                _m, _base, _inv_base, scale, _d_coeff, _d_digits, nullptr,
                _d_carry_in, _d_carry_out, _num_carry_blocks,
                _carry_block_size, _d_changed, "resident square", true,
                _stream, _h_changed);
        }
        record_carry_iterations(carry_iterations);
        if (_timing_enabled) {
            const auto t1 = std::chrono::steady_clock::now();
            _transform_seconds += std::chrono::duration<double>(t_norm0 - t_transform0).count();
            _normalize_seconds += std::chrono::duration<double>(t1 - t_norm0).count();
        }
        throttle_after_gpu_iteration(throttle_started, _duty_budget);
    }

    [[maybe_unused]] void mul_small(uint32_t scale) {
        finish_pending();
        if (scale == 1) return;
        record_carry_iterations(normalize_current_mods(scale, nullptr, "resident scalar"));
    }

    [[maybe_unused]] void mul_assign(GpuResidue& rhs) {
        finish_pending();
        rhs.finish_pending();
        if (_base != rhs._base || _m != rhs._m) throw std::runtime_error("resident multiply shape mismatch");
        if (this == &rhs) {
            square_dup(false);
            return;
        }
        const int threads = 256;
        const int blocks = (_len + threads - 1) / threads;
        for (int i = 0; i < _mod_count; ++i) {
            cuda_check(cudaMemcpy(_d_mul_rhs, rhs._d_r[i], sizeof(uint32_t) * _len, cudaMemcpyDeviceToDevice),
                       "resident copy multiply rhs");
            ntt_inplace_precomputed(_d_r[i], _d_tmp_r[0], _log_len, kMods[i].p, _d_rev, false,
                                    _d_twiddle_fwd[i], _twiddle_offsets);
            ntt_inplace_precomputed(_d_mul_rhs, _d_tmp_r[0], _log_len, kMods[i].p, _d_rev, false,
                                    _d_twiddle_fwd[i], _twiddle_offsets);
            pointwise_mul_kernel<<<blocks, threads>>>(_d_r[i], _d_mul_rhs, _len, kMods[i].p);
            cuda_check(cudaGetLastError(), "resident pointwise_mul launch");
            ntt_inplace_precomputed(_d_r[i], _d_tmp_r[0], _log_len, kMods[i].p, _d_rev, true,
                                    _d_twiddle_inv[i], _twiddle_offsets);
        }
        record_carry_iterations(normalize_current_mods(1u, nullptr, "resident multiply"));
    }

    [[maybe_unused]] std::vector<int64_t> get_digits(bool count_stats = true) {
        finish_pending();
        const int iterations = normalize_current_mods(1u, _d_out, "resident get_digits");
        if (count_stats) record_carry_iterations(iterations);
        std::vector<int64_t> out(_m);
        cuda_check(cudaMemcpy(out.data(), _d_out, sizeof(int64_t) * _m, cudaMemcpyDeviceToHost), "resident copy digits");
        return out;
    }

    [[maybe_unused]] void reset_stats() {
        finish_pending();
        _duty_budget.pending_work_ns = 0;
        _carry_iterations = 0;
        _normalize_calls = 0;
        _transform_seconds = 0.0;
        _normalize_seconds = 0.0;
    }

    [[maybe_unused]] uint64_t carry_iterations() const {
        return _carry_iterations;
    }

    [[maybe_unused]] uint64_t normalize_calls() const {
        return _normalize_calls;
    }

    [[maybe_unused]] void set_timing(bool enabled) {
        _timing_enabled = enabled;
    }

    [[maybe_unused]] double transform_seconds() const {
        return _transform_seconds;
    }

    [[maybe_unused]] double normalize_seconds() const {
        return _normalize_seconds;
    }

    [[maybe_unused]] int mod_count() const {
        return _mod_count;
    }

    [[maybe_unused]] int carry_block_size() const {
        return _carry_block_size;
    }

    void finish_pending() {
        if (_pending_dups.empty()) return;
        cuda_check(cudaMemcpyAsync(_h_changed, _d_batch_failed, sizeof(int),
                                   cudaMemcpyDeviceToHost, _stream), "batch read sticky flag");
        cuda_check(cudaStreamSynchronize(_stream), "batch completion barrier");
        const bool replay = *_h_changed != 0 || _force_replay;
        if (replay) {
            for (int i = 0; i < _mod_count; ++i)
                cuda_check(cudaMemcpyAsync(_d_r[i], _d_batch_snapshot[i],
                    sizeof(uint32_t) * _len, cudaMemcpyDeviceToDevice, _stream),
                    "batch restore last validated RNS");
            for (unsigned char bit : _pending_dups) {
                const uint32_t scale = bit ? 2u : 1u;
                if (_mod_count == 3) launch_square3_prefix(scale);
                else launch_square4_prefix(scale);
                int iterations = _mod_count == 3
                    ? normalize3_relaxed(_d_r[0], _d_r[1], _d_r[2], _m, _base,
                        _inv_base, scale, _d_coeff, _d_digits, nullptr, _d_carry_in,
                        _d_carry_out, _num_carry_blocks, _carry_block_size,
                        _d_changed, "batch replay", true, _stream, _h_changed)
                    : normalize4_relaxed(_d_r[0], _d_r[1], _d_r[2], _d_r[3], _m,
                        _base, _inv_base, scale, _d_coeff, _d_digits, nullptr,
                        _d_carry_in, _d_carry_out, _num_carry_blocks,
                        _carry_block_size, _d_changed, "batch replay", true,
                        _stream, _h_changed);
                record_carry_iterations(iterations);
            }
            ++_replayed_batches;
            std::cout << "carry-batching: replayed batch=" << _replayed_batches
                      << ", bits=" << _pending_dups.size() << "\n";
        } else {
            _carry_iterations += _fast_carry_passes * _pending_dups.size();
            _normalize_calls += _pending_dups.size();
        }
        // Asynchronous batches cannot be meaningfully split into transform and
        // carry host timings. Account the complete validated/replayed batch in
        // the combined counter reported by --bench-resident instead.
        if (_timing_enabled) {
            _transform_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - _batch_started).count();
        }
        _pending_dups.clear();
    }

private:
    void release_resources_noexcept() noexcept {
        if (_stream != nullptr) cudaStreamSynchronize(_stream);
        for (int i = 0; i < 2; ++i) {
            if (_square_graph_exec[i] != nullptr) {
                cudaGraphExecDestroy(_square_graph_exec[i]);
                _square_graph_exec[i] = nullptr;
            }
            if (_square_graph[i] != nullptr) {
                cudaGraphDestroy(_square_graph[i]);
                _square_graph[i] = nullptr;
            }
        }
        if (_h_changed != nullptr) {
            cudaFreeHost(_h_changed);
            _h_changed = nullptr;
        }
        cudaFree(_d_status); _d_status = nullptr;
        cudaFree(_d_neg_rev); _d_neg_rev = nullptr;
        cudaFree(_d_twist); _d_twist = nullptr;
        cudaFree(_d_untwist); _d_untwist = nullptr;
        cudaFree(_d_batch_failed); _d_batch_failed = nullptr;
        for (auto& pointer : _d_batch_snapshot) {
            cudaFree(pointer); pointer = nullptr;
        }
        cudaFree(_d_out); _d_out = nullptr;
        cudaFree(_d_changed); _d_changed = nullptr;
        cudaFree(_d_carry_out); _d_carry_out = nullptr;
        cudaFree(_d_carry_in); _d_carry_in = nullptr;
        cudaFree(_d_digits); _d_digits = nullptr;
        cudaFree(_d_coeff); _d_coeff = nullptr;
        cudaFree(_d_mul_rhs); _d_mul_rhs = nullptr;
        for (auto& pointer : _d_r) {
            cudaFree(pointer);
            pointer = nullptr;
        }
        for (auto& pointer : _d_twiddle_inv) {
            cudaFree(pointer);
            pointer = nullptr;
        }
        for (auto& pointer : _d_twiddle_fwd) {
            cudaFree(pointer);
            pointer = nullptr;
        }
        cudaFree(_d_rev); _d_rev = nullptr;
        for (auto& pointer : _d_tmp_r) {
            cudaFree(pointer);
            pointer = nullptr;
        }
        if (_stream != nullptr) {
            cudaStreamDestroy(_stream);
            _stream = nullptr;
        }
    }

    void launch_square3_prefix(uint32_t scale) {
        const int threads = 256;
        const int blocks = (_square_len + threads - 1) / threads;
        if (_negacyclic) launch_twist(false);
        if (_dif) launch_dif_forward(_fused_tile);
        else ntt3_inplace_precomputed(
            _d_r[0], _d_r[1], _d_r[2],
            _d_tmp_r[0], _d_tmp_r[1], _d_tmp_r[2],
            _square_log_len, _square_rev, false,
            _d_twiddle_fwd[0], _d_twiddle_fwd[1], _d_twiddle_fwd[2],
            _twiddle_offsets, false, _square_inverse_lengths.data(), _stream);
        if (_fused_tile) launch_fused_tile();
        else pointwise_square3_kernel<<<blocks, threads, 0, _stream>>>(
            _d_r[0], _d_r[1], _d_r[2], _square_len);
        cuda_check(cudaGetLastError(), "resident pointwise_square3 launch");
        ntt3_inplace_precomputed(
            _d_r[0], _d_r[1], _d_r[2],
            _d_tmp_r[0], _d_tmp_r[1], _d_tmp_r[2],
            _square_log_len, _square_rev, true,
            _d_twiddle_inv[0], _d_twiddle_inv[1], _d_twiddle_inv[2],
            _twiddle_offsets, false, _square_inverse_lengths.data(), _stream, !_dif, _shared_low, _fused_tile);
        if (_negacyclic) launch_twist(true);
        launch_normalize3_relaxed_prefix(
            _d_r[0], _d_r[1], _d_r[2], _m, _base, _inv_base, scale,
            _d_coeff, _d_digits, _d_carry_in, _d_carry_out,
            _d_changed, _stream);
    }

    void launch_square4_prefix(uint32_t scale) {
        const int threads = 256;
        const int blocks = (_square_len + threads - 1) / threads;
        if (_negacyclic) launch_twist(false);
        if (_dif) launch_dif_forward(_fused_tile);
        else ntt4_inplace_precomputed(
            _d_r[0], _d_r[1], _d_r[2], _d_r[3],
            _d_tmp_r[0], _d_tmp_r[1], _d_tmp_r[2], _d_tmp_r[3],
            _square_log_len, _square_rev, false,
            _d_twiddle_fwd[0], _d_twiddle_fwd[1],
            _d_twiddle_fwd[2], _d_twiddle_fwd[3],
            _twiddle_offsets, false, _square_inverse_lengths.data(), _stream);
        if (_fused_tile) launch_fused_tile();
        else pointwise_square4_kernel<<<blocks, threads, 0, _stream>>>(
            _d_r[0], _d_r[1], _d_r[2], _d_r[3], _square_len);
        cuda_check(cudaGetLastError(), "resident pointwise_square4 launch");
        ntt4_inplace_precomputed(
            _d_r[0], _d_r[1], _d_r[2], _d_r[3],
            _d_tmp_r[0], _d_tmp_r[1], _d_tmp_r[2], _d_tmp_r[3],
            _square_log_len, _square_rev, true,
            _d_twiddle_inv[0], _d_twiddle_inv[1],
            _d_twiddle_inv[2], _d_twiddle_inv[3],
            _twiddle_offsets, false, _square_inverse_lengths.data(), _stream, !_dif, _shared_low, _fused_tile);
        if (_negacyclic) launch_twist(true);
        launch_normalize4_relaxed_prefix(
            _d_r[0], _d_r[1], _d_r[2], _d_r[3],
            _m, _base, _inv_base, scale, _d_coeff, _d_digits,
            _d_carry_in, _d_carry_out, _d_changed, _stream);
    }

    void launch_dif_forward(bool skip_low = false) {
        int stage = _square_log_len;
        const bool tiled = _shared_low && stage >= 10;
        if(tiled && (stage % 2)!=0) {
            ntt_dif_stage_rns_kernel<<<limited_ntt_grid_y(_square_len/2,256),256,0,_stream>>>(
                _d_r[0],_d_r[1],_d_r[2],_d_r[3],
                _d_twiddle_fwd[0],_d_twiddle_fwd[1],_d_twiddle_fwd[2],_d_twiddle_fwd[3],
                _twiddle_offsets[stage],1<<stage,_square_len/2,_mod_count);
            cuda_check(cudaGetLastError(), "DIF odd top stage");
            --stage;
        }
        for (; stage >= (tiled ? 12 : 2); stage -= 2) {
            const int groups = _square_len / 4;
            ntt_dif_stage2_rns_kernel<<<limited_ntt_grid_y(groups,256),256,0,_stream>>>(
                _d_r[0],_d_r[1],_d_r[2],_d_r[3],
                _d_twiddle_fwd[0],_d_twiddle_fwd[1],_d_twiddle_fwd[2],_d_twiddle_fwd[3],
                _twiddle_offsets[stage],_twiddle_offsets[stage-1],1<<stage,groups,_mod_count);
            cuda_check(cudaGetLastError(), "DIF two-stage launch");
        }
        if(tiled) {
            if (!skip_low) {
            ntt_shared1024_rns_kernel<true><<<_square_len/1024,256,0,_stream>>>(
                _d_r[0],_d_r[1],_d_r[2],_d_r[3],
                _d_twiddle_fwd[0],_d_twiddle_fwd[1],_d_twiddle_fwd[2],_d_twiddle_fwd[3],_mod_count,
                nullptr,nullptr,nullptr,nullptr);
            cuda_check(cudaGetLastError(), "shared forward low stages");
            }
            stage=0;
        }
        if(stage==1) {
            const int y=limited_ntt_grid_y(_square_len/2,256);
            if(_mod_count==3)
                ntt_stage_3_kernel<<<dim3(1,y),256,0,_stream>>>(_d_r[0],_d_r[1],_d_r[2],
                    _d_twiddle_fwd[0],_d_twiddle_fwd[1],_d_twiddle_fwd[2],2,_square_len/2);
            else
                ntt_stage_4_kernel<<<dim3(1,y),256,0,_stream>>>(_d_r[0],_d_r[1],_d_r[2],_d_r[3],
                    _d_twiddle_fwd[0],_d_twiddle_fwd[1],_d_twiddle_fwd[2],_d_twiddle_fwd[3],2,_square_len/2);
            cuda_check(cudaGetLastError(), "DIF final stage launch");
        }
    }

    void launch_fused_tile() {
        ntt_shared1024_rns_kernel<2><<<_square_len/1024,256,0,_stream>>>(
            _d_r[0],_d_r[1],_d_r[2],_d_r[3],
            _d_twiddle_fwd[0],_d_twiddle_fwd[1],_d_twiddle_fwd[2],_d_twiddle_fwd[3],_mod_count,
            _d_twiddle_inv[0],_d_twiddle_inv[1],_d_twiddle_inv[2],_d_twiddle_inv[3]);
        cuda_check(cudaGetLastError(), "fused forward-square-inverse tile");
    }

    void launch_twist(bool inverse) {
        twist_rns_kernel<<<(_m+255)/256, 256, 0, _stream>>>(
            _d_r[0], _d_r[1], _d_r[2], _d_r[3],
            inverse ? _d_untwist : _d_twist, _m, _mod_count);
        cuda_check(cudaGetLastError(), "negacyclic twist launch");
    }

    void capture_square_graphs() {
        cuda_check(cudaDeviceSynchronize(), "resident before graph capture");
        for (int dup = 0; dup < 2; ++dup) {
            cuda_check(cudaStreamBeginCapture(
                           _stream, cudaStreamCaptureModeThreadLocal),
                       "resident begin square graph capture");
            if (_mod_count == 3) {
                launch_square3_prefix(dup ? 2u : 1u);
            } else {
                launch_square4_prefix(dup ? 2u : 1u);
            }
            if (_batch_bits > 0) launch_speculative_tail();
            cuda_check(cudaStreamEndCapture(_stream, &_square_graph[dup]),
                       "resident end square graph capture");
            cuda_check(cudaGraphInstantiate(
                           &_square_graph_exec[dup], _square_graph[dup],
                           nullptr, nullptr, 0),
                       "resident instantiate square graph");
        }
    }

    void launch_speculative_tail() {
        const int threads = 256;
        const int blocks = (_m + threads - 1) / threads;
        if (_fast_carry_passes == 3) {
            latch_carry_failure_kernel<<<1, 1, 0, _stream>>>(_d_changed, _d_batch_failed);
            cuda_check(cudaGetLastError(), "batch latch convergence");
        } else {
            // Small bases propagate carries farther. Only the last pass must
            // be a fixed point; intermediate nonconvergence is expected.
            for (int pass=3; pass<_fast_carry_passes; ++pass) {
                carry_all_kernel<<<blocks,threads,0,_stream>>>(
                    _d_coeff,_d_digits,_d_carry_out,_d_carry_in,_m,_base,_inv_base);
                cuda_check(cudaGetLastError(), "batch extra carry pass");
                if (pass+1 == _fast_carry_passes) {
                    // atomicExch(...,1) only: failures from earlier bits
                    // remain latched until the batch is validated/replayed.
                    update_block_carries_kernel<<<blocks,threads,0,_stream>>>(
                        _d_carry_in,_d_carry_out,_m,_d_batch_failed);
                } else {
                    update_block_carries_plain_kernel<<<blocks,threads,0,_stream>>>(
                        _d_carry_in,_d_carry_out,_m);
                }
                cuda_check(cudaGetLastError(), "batch extra carry convergence");
            }
        }
        if (_mod_count == 3)
            digits_to_rns3_kernel<<<blocks, threads, 0, _stream>>>(
                _d_r[0], _d_r[1], _d_r[2], _d_digits, _m, nullptr);
        else
            digits_to_rns_kernel<<<blocks, threads, 0, _stream>>>(
                _d_r[0], _d_r[1], _d_r[2], _d_r[3], _d_digits, _m, nullptr);
        cuda_check(cudaGetLastError(), "batch export normalized digits");
    }

    int normalize_current_mods(uint32_t scale, int64_t* out, const char* tag) {
        if (_mod_count == 3) {
            return normalize3_relaxed(_d_r[0], _d_r[1], _d_r[2],
                                      _m, _base, _inv_base, scale, _d_coeff, _d_digits, out,
                                      _d_carry_in, _d_carry_out, _num_carry_blocks, _carry_block_size,
                                      _d_changed, tag, false, _stream, _h_changed);
        }
        return normalize4_relaxed(_d_r[0], _d_r[1], _d_r[2], _d_r[3],
                                  _m, _base, _inv_base, scale, _d_coeff, _d_digits, out,
                                  _d_carry_in, _d_carry_out, _num_carry_blocks, _carry_block_size,
                                  _d_changed, tag, false, _stream, _h_changed);
    }

    void record_carry_iterations(int iterations) {
        _carry_iterations += static_cast<uint64_t>(iterations);
        ++_normalize_calls;
    }

    uint64_t _base;
    double _inv_base = 0.0;
    int _m;
    int _log_len;
    int _len;
    int _mod_count;
    bool _negacyclic = false;
    bool _dif = false;
    bool _shared_low = false;
    bool _fused_tile = false;
    int _square_log_len = 0;
    int _square_len = 0;
    uint32_t* _d_neg_rev = nullptr;
    uint32_t* _square_rev = nullptr;
    uint32_t* _d_twist = nullptr;
    uint32_t* _d_untwist = nullptr;
    std::array<uint32_t, 4> _square_inverse_lengths{};
    int _carry_block_size;
    int _num_carry_blocks;
    uint32_t* _d_r[4] = {};
    uint32_t* _d_tmp_r[4] = {};
    uint32_t* _d_rev = nullptr;
    uint32_t* _d_twiddle_fwd[4] = {};
    uint32_t* _d_twiddle_inv[4] = {};
    std::array<uint32_t, 4> _inverse_lengths{};
    uint32_t* _d_mul_rhs = nullptr;
    std::vector<int> _twiddle_offsets;
    S128* _d_coeff = nullptr;
    int64_t* _d_digits = nullptr;
    S128* _d_carry_in = nullptr;
    S128* _d_carry_out = nullptr;
    int* _d_changed = nullptr;
    int* _h_changed = nullptr;
    int64_t* _d_out = nullptr;
    int* _d_status = nullptr;
    int _batch_bits = 0;
    int _fast_carry_passes = 3;
    bool _force_replay = false;
    uint64_t _replayed_batches = 0;
    uint32_t* _d_batch_snapshot[4] = {};
    int* _d_batch_failed = nullptr;
    std::vector<unsigned char> _pending_dups;
    cudaStream_t _stream = nullptr;
    bool _graphs_enabled = false;
    cudaGraph_t _square_graph[2] = {};
    cudaGraphExec_t _square_graph_exec[2] = {};
    uint64_t _carry_iterations = 0;
    uint64_t _normalize_calls = 0;
    GpuDutyBudget _duty_budget;
    bool _timing_enabled = false;
    std::chrono::steady_clock::time_point _batch_started;
    double _transform_seconds = 0.0;
    double _normalize_seconds = 0.0;
};

// Host reference CRT must retain the full roughly 89-bit centered result.
// The former int64_t implementation silently overflowed and has been removed.
S128 crt3_signed(uint32_t r0, uint32_t r1, uint32_t r2) {
    const uint64_t p0 = kMods[0].p;
    const uint64_t p1 = kMods[1].p;
    const uint64_t p2 = kMods[2].p;
    const uint64_t inv_p0_p1 = pow_mod_host(static_cast<uint32_t>(p0 % p1), p1 - 2, static_cast<uint32_t>(p1));
    const uint64_t p01_mod_p2 = ((p0 % p2) * (p1 % p2)) % p2;
    const uint64_t inv_p01_p2 = pow_mod_host(static_cast<uint32_t>(p01_mod_p2), p2 - 2, static_cast<uint32_t>(p2));

    uint64_t t1 = (r1 + p1 - (r0 % p1)) % p1;
    t1 = (t1 * inv_p0_p1) % p1;
    uint64_t x_mod_p2 = (r0 + (p0 % p2) * (t1 % p2)) % p2;
    uint64_t t2 = (r2 + p2 - x_mod_p2) % p2;
    t2 = (t2 * inv_p01_p2) % p2;

    const uint64_t p01 = p0 * p1;
    const U128 mod = mul64_u128(p01, p2);
    U128 x = from_u64(r0);
    x = add_u128(x, mul64_u128(p0, t1));
    x = add_u128(x, mul64_u128(p01, t2));
    const U128 half = shr1_u128(mod);
    if (cmp_u128(x, half) > 0) return S128{true, sub_u128(mod, x)};
    return S128{false, x};
}

S128 crt4_signed(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    const uint64_t p0 = kMods[0].p;
    const uint64_t p1 = kMods[1].p;
    const uint64_t p2 = kMods[2].p;
    const uint64_t p3 = kMods[3].p;

    const uint64_t inv_p0_p1 = pow_mod_host(static_cast<uint32_t>(p0 % p1), p1 - 2, static_cast<uint32_t>(p1));
    const uint64_t p01_mod_p2 = ((p0 % p2) * (p1 % p2)) % p2;
    const uint64_t inv_p01_p2 = pow_mod_host(static_cast<uint32_t>(p01_mod_p2), p2 - 2, static_cast<uint32_t>(p2));
    const uint64_t p01_mod_p3 = ((p0 % p3) * (p1 % p3)) % p3;
    const uint64_t p012_mod_p3 = (p01_mod_p3 * (p2 % p3)) % p3;
    const uint64_t inv_p012_p3 = pow_mod_host(static_cast<uint32_t>(p012_mod_p3), p3 - 2, static_cast<uint32_t>(p3));

    uint64_t t1 = (r1 + p1 - (r0 % p1)) % p1;
    t1 = (t1 * inv_p0_p1) % p1;

    uint64_t x_mod_p2 = (r0 + (p0 % p2) * (t1 % p2)) % p2;
    uint64_t t2 = (r2 + p2 - x_mod_p2) % p2;
    t2 = (t2 * inv_p01_p2) % p2;

    uint64_t x_mod_p3 = r0 % p3;
    x_mod_p3 = (x_mod_p3 + (p0 % p3) * (t1 % p3)) % p3;
    x_mod_p3 = (x_mod_p3 + p01_mod_p3 * (t2 % p3)) % p3;
    uint64_t t3 = (r3 + p3 - x_mod_p3) % p3;
    t3 = (t3 * inv_p012_p3) % p3;

    const uint64_t p01 = p0 * p1;
    const U128 p012 = mul64_u128(p01, p2);
    const U128 mod = mul_u128_u32(p012, static_cast<uint32_t>(p3));

    U128 x = from_u64(r0);
    x = add_u128(x, mul64_u128(p0, t1));
    x = add_u128(x, mul64_u128(p01, t2));
    x = add_u128(x, mul_u128_u32(p012, static_cast<uint32_t>(t3)));

    const U128 half = shr1_u128(mod);
    if (cmp_u128(x, half) > 0) {
        return S128{true, sub_u128(mod, x)};
    }
    return S128{false, x};
}

std::vector<int64_t> normalize_s128_coeffs(std::vector<S128> coeff, uint64_t base) {
    const int m = static_cast<int>(coeff.size());
    const uint64_t b = base;
    const uint64_t half = b / 2;
    const std::vector<S128> original = coeff;
    const auto settle = [&](std::vector<S128>& values, bool lower_tie,
                            int max_passes) {
        S128 carry{};
        for (int pass = 0; pass < max_passes; ++pass) {
            carry = S128{};
            for (int i = 0; i < m; ++i) {
                S128 v = add_s128(values[i], carry);
                uint64_t rem = 0;
                U128 qmag = divmod_u128_u64(v.mag, b, rem);
                S128 q{};
                int64_t digit = 0;
                if (!v.neg) {
                    q = S128{false, qmag};
                    digit = static_cast<int64_t>(rem);
                } else if (rem == 0) {
                    q = S128{true, qmag};
                    digit = 0;
                } else {
                    q = S128{true, add_u128(qmag, from_u64(1))};
                    digit = static_cast<int64_t>(b - rem);
                }
                const bool reduce = lower_tie
                    ? static_cast<uint64_t>(digit) >= half
                    : static_cast<uint64_t>(digit) > half;
                if (reduce) {
                    digit -= static_cast<int64_t>(b);
                    q = add_s128(q, make_s128(1));
                }
                values[i] = make_s128(digit);
                carry = q;
            }
            if (is_zero(carry.mag)) return true;
            values[0] = sub_s128(values[0], carry);
        }
        return false;
    };

    if (!settle(coeff, false, 8)) {
        if ((base & 1ull) != 0) {
            throw std::runtime_error("normalization did not settle");
        }
        coeff = original;
        if (!settle(coeff, true, 32)) {
            throw std::runtime_error("special normalization did not settle");
        }
        const int64_t signed_half = static_cast<int64_t>(half);
        for (int i = 0; i < m; ++i) {
            const int64_t digit = to_i64_checked(coeff[i]);
            const int64_t expected = i == 0 ? -signed_half : 1 - signed_half;
            if (digit != expected) {
                throw std::runtime_error(
                    "lower-tie normalization was not the unique special residue");
            }
        }
    }

    std::vector<int64_t> out(m);
    for (int i = 0; i < m; ++i) out[i] = to_i64_checked(coeff[i]);
    return out;
}

std::vector<int64_t> normalize_from_residues(const std::vector<std::vector<uint32_t>>& lin, uint64_t base, uint32_t scale = 1) {
    const int len = static_cast<int>(lin[0].size());
    const int m = len / 2;
    std::vector<S128> coeff(m);
    for (int i = 0; i < m; ++i) {
        const uint32_t r0 = (lin[0][i] + kMods[0].p - lin[0][i + m]) % kMods[0].p;
        const uint32_t r1 = (lin[1][i] + kMods[1].p - lin[1][i + m]) % kMods[1].p;
        const uint32_t r2 = (lin[2][i] + kMods[2].p - lin[2][i + m]) % kMods[2].p;
        if (lin.size() >= 4) {
            const uint32_t r3 = (lin[3][i] + kMods[3].p - lin[3][i + m]) % kMods[3].p;
            coeff[i] = crt4_signed(r0, r1, r2, r3);
        } else {
            coeff[i] = crt3_signed(r0, r1, r2);
        }
        if (scale != 1) coeff[i] = mul_s128_u32(coeff[i], scale);
    }
    return normalize_s128_coeffs(std::move(coeff), base);
}

std::vector<int64_t> cpu_schoolbook_square_norm(const std::vector<int64_t>& a, uint64_t base) {
    const int m = static_cast<int>(a.size());
    std::vector<int64_t> c(m, 0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            const int k = i + j;
            const int64_t v = a[i] * a[j];
            if (k < m) c[k] += v;
            else c[k - m] -= v;
        }
    }

    std::vector<S128> wide(m);
    for (int i = 0; i < m; ++i) wide[i] = make_s128(c[i]);
    return normalize_s128_coeffs(std::move(wide), base);
}

void run_special_residue_atomicity_selftest() {
    constexpr int m = 4;
    constexpr int len = 2 * m;
    constexpr uint64_t base = 10;
    const std::array<int64_t, m> raw = {-95, -100, 0, 0};
    std::array<S128, m> host_coeff{};
    for (int i = 0; i < m; ++i) host_coeff[i] = make_s128(raw[i]);
    std::array<int64_t, m> host_digits{};
    std::array<int64_t, m> host_out = {91, 92, 93, 94};
    std::array<std::array<uint32_t, len>, 4> host_r{};
    for (int mod = 0; mod < 4; ++mod) {
        for (int i = 0; i < len; ++i) {
            host_r[mod][i] = static_cast<uint32_t>(1000 + mod * 100 + i);
        }
    }
    const auto original_r = host_r;
    const auto original_out = host_out;

    std::array<uint32_t*, 4> device_r{};
    S128* device_coeff = nullptr;
    int64_t* device_digits = nullptr;
    int64_t* device_out = nullptr;
    int* device_status = nullptr;
    try {
        for (int mod = 0; mod < 4; ++mod) {
            cuda_check(cudaMalloc(&device_r[mod], sizeof(uint32_t) * len),
                       "special atomicity malloc rns");
            cuda_check(cudaMemcpy(device_r[mod], host_r[mod].data(),
                                  sizeof(uint32_t) * len,
                                  cudaMemcpyHostToDevice),
                       "special atomicity copy rns");
        }
        cuda_check(cudaMalloc(&device_coeff, sizeof(S128) * m),
                   "special atomicity malloc coeff");
        cuda_check(cudaMalloc(&device_digits, sizeof(int64_t) * m),
                   "special atomicity malloc digits");
        cuda_check(cudaMalloc(&device_out, sizeof(int64_t) * m),
                   "special atomicity malloc out");
        cuda_check(cudaMalloc(&device_status, sizeof(int)),
                   "special atomicity malloc status");
        cuda_check(cudaMemcpy(device_coeff, host_coeff.data(),
                              sizeof(S128) * m, cudaMemcpyHostToDevice),
                   "special atomicity copy coeff");
        cuda_check(cudaMemcpy(device_digits, host_digits.data(),
                              sizeof(int64_t) * m, cudaMemcpyHostToDevice),
                   "special atomicity copy digits");
        cuda_check(cudaMemcpy(device_out, host_out.data(),
                              sizeof(int64_t) * m, cudaMemcpyHostToDevice),
                   "special atomicity copy out");
        try_even_special_residue_kernel<<<1, 1>>>(
            device_r[0], device_r[1], device_r[2], device_r[3],
            4, m, base, 0.1, device_coeff, device_digits,
            device_out, device_status);
        cuda_check(cudaGetLastError(), "special atomicity kernel launch");
        int status = 0;
        cuda_check(cudaMemcpy(&status, device_status, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "special atomicity copy status");
        if (status == 0) {
            throw std::runtime_error(
                "non-special vector was accepted by special fallback");
        }
        for (int mod = 0; mod < 4; ++mod) {
            cuda_check(cudaMemcpy(host_r[mod].data(), device_r[mod],
                                  sizeof(uint32_t) * len,
                                  cudaMemcpyDeviceToHost),
                       "special atomicity read rns");
        }
        cuda_check(cudaMemcpy(host_out.data(), device_out,
                              sizeof(int64_t) * m, cudaMemcpyDeviceToHost),
                   "special atomicity read out");
        if (host_r != original_r || host_out != original_out) {
            throw std::runtime_error(
                "failed special fallback modified RNS/output state");
        }
    } catch (...) {
        cudaFree(device_status);
        cudaFree(device_out);
        cudaFree(device_digits);
        cudaFree(device_coeff);
        for (auto pointer : device_r) cudaFree(pointer);
        throw;
    }
    cudaFree(device_status);
    cudaFree(device_out);
    cudaFree(device_digits);
    cudaFree(device_coeff);
    for (auto pointer : device_r) cudaFree(pointer);
}

void run_selftest() {
    std::mt19937_64 rng(0x47464e43554441ull);
    for (int log_m = 2; log_m <= 8; ++log_m) {
        const int m = 1 << log_m;
        const uint64_t base = 1000003;
        std::uniform_int_distribution<int64_t> dist(-2000, 2000);
        for (int round = 0; round < 20; ++round) {
            std::vector<int64_t> a(m);
            for (auto& x : a) x = dist(rng);
            const auto lin = cuda_linear_square_residues(a);
            const auto got = normalize_from_residues(lin, base);
            const auto want = cpu_schoolbook_square_norm(a, base);
            if (got != want) {
                std::cerr << "selftest failed at log_m=" << log_m << " round=" << round << "\n";
                std::exit(1);
            }
        }
    }

    for (uint64_t base : {2ull, 4ull, 6ull, 10ull}) {
        for (int m : {2, 4}) {
            uint64_t modulus = 1;
            for (int i = 0; i < m; ++i) modulus *= base;
            ++modulus;
            std::set<std::vector<int64_t>> encodings;
            uint64_t special_count = 0;
            for (uint64_t residue = 0; residue < modulus; ++residue) {
                std::vector<S128> coeff(static_cast<size_t>(m));
                coeff[0] = S128{false, from_u64(residue)};
                const auto digits = normalize_s128_coeffs(coeff, base);
                int64_t value = 0;
                int64_t power = 1;
                for (int i = 0; i < m; ++i) {
                    value += digits[i] * power;
                    power *= static_cast<int64_t>(base);
                }
                int64_t reduced = value % static_cast<int64_t>(modulus);
                if (reduced < 0) reduced += static_cast<int64_t>(modulus);
                if (reduced != static_cast<int64_t>(residue)) {
                    throw std::runtime_error(
                        "even-base representation selftest residue mismatch");
                }
                const int64_t half = static_cast<int64_t>(base / 2);
                bool special = digits[0] == -half;
                for (int i = 1; i < m && special; ++i) {
                    special = digits[i] == 1 - half;
                }
                if (special) {
                    ++special_count;
                } else {
                    for (int64_t digit : digits) {
                        if (digit < 1 - half || digit > half) {
                            throw std::runtime_error(
                                "even-base representation selftest digit range mismatch");
                        }
                    }
                }
                encodings.insert(digits);
            }
            if (encodings.size() != modulus || special_count != 1) {
                throw std::runtime_error(
                    "even-base representation selftest uniqueness mismatch");
            }
        }
    }

    for (const auto& shape : std::vector<std::pair<uint64_t, int>>{
             {2ull, 4}, {4ull, 4}, {10ull, 4}, {10000000000000ull, 4}}) {
        const uint64_t base = shape.first;
        const int n = shape.second;
        const int m = 1 << n;
        const int64_t half = static_cast<int64_t>(base / 2);
        std::vector<int64_t> special(static_cast<size_t>(m), 1 - half);
        special[0] = -half;
        GpuResidue residue(base, n);
        residue.set_coeffs(special);
        if (residue.get_digits(false) != special) {
            throw std::runtime_error(
                "device special-residue roundtrip selftest mismatch");
        }
    }
    run_special_residue_atomicity_selftest();
    std::cout << "selftest ok: CUDA NTT square, even-base canonical residues, and special fallback\n";
}

void run_square(uint64_t base, int n) {
    if (n < 1 || n > 20) throw std::runtime_error("n out of prototype range");
    if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("current normalization backend requires base < 2^63");
    }
    const int m = 1 << n;
    std::vector<int64_t> a(m, 0);
    a[0] = 3;
    const auto lin = cuda_linear_square_residues(a);
    const auto got = normalize_from_residues(lin, base);
    std::cout << "square(3) mod (b^(2^n)+1) gives coeff[0]=" << got[0]
              << ", nonzero coeffs:";
    int shown = 0;
    for (int i = 0; i < m && shown < 8; ++i) {
        if (got[i] != 0) {
            std::cout << " [" << i << "]=" << got[i];
            ++shown;
        }
    }
    std::cout << "\n";
}

void run_sparse(uint64_t base, int n, uint64_t seed, int terms) {
    if (n < 1 || n > 20) throw std::runtime_error("n out of prototype range");
    if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("current digit output requires base < 2^63");
    }
    if (terms <= 0) throw std::runtime_error("terms must be positive");
    const int m = 1 << n;
    std::vector<int64_t> a(m, 0);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> pos_dist(0, m - 1);
    const int64_t amp = static_cast<int64_t>(base / 4);
    std::uniform_int_distribution<int64_t> val_dist(-amp, amp);
    for (int i = 0; i < terms; ++i) {
        int64_t v = 0;
        while (v == 0) v = val_dist(rng);
        a[pos_dist(rng)] = v;
    }
    const auto lin = cuda_linear_square_residues(a);
    const auto got = normalize_from_residues(lin, base);
    int nonzero = 0;
    int64_t max_abs = 0;
    for (const int64_t x : got) {
        if (x != 0) ++nonzero;
        const int64_t ax = x < 0 ? -x : x;
        if (ax > max_abs) max_abs = ax;
        if (static_cast<uint64_t>(ax) > base / 2 + 1) {
            throw std::runtime_error("normalized digit outside balanced radix range");
        }
    }
    std::cout << "sparse square ok: b=" << base << ", n=" << n
              << ", terms=" << terms << ", nonzero result digits=" << nonzero
              << ", max_abs_digit=" << max_abs << "\n";
}

std::vector<int64_t> make_sparse_input(uint64_t base, int n, uint64_t seed, int terms) {
    if (n < 1 || n > 20) throw std::runtime_error("n out of prototype range");
    if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("current digit output requires base < 2^63");
    }
    if (terms <= 0) throw std::runtime_error("terms must be positive");
    const int m = 1 << n;
    std::vector<int64_t> a(m, 0);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> pos_dist(0, m - 1);
    const int64_t amp = static_cast<int64_t>(base / 4);
    std::uniform_int_distribution<int64_t> val_dist(-amp, amp);
    for (int i = 0; i < terms; ++i) {
        int64_t v = 0;
        while (v == 0) v = val_dist(rng);
        a[pos_dist(rng)] = v;
    }
    return a;
}

void run_gpu_norm(uint64_t base, int n, uint64_t seed, int terms, bool dup) {
    const auto a = make_sparse_input(base, n, seed, terms);
    const auto lin = cuda_linear_square_residues(a);
    const auto host = normalize_from_residues(lin, base, dup ? 2u : 1u);
    const auto gpu = cuda_square_dup_gpu_norm(a, base, dup);
    if (host != gpu) {
        for (size_t i = 0; i < host.size(); ++i) {
            if (host[i] != gpu[i]) {
                std::cerr << "first mismatch at " << i << ": host=" << host[i] << ", gpu=" << gpu[i] << "\n";
                break;
            }
        }
        throw std::runtime_error("GPU normalize output mismatch");
    }
    int nonzero = 0;
    int64_t max_abs = 0;
    for (const int64_t x : gpu) {
        if (x != 0) ++nonzero;
        const int64_t ax = x < 0 ? -x : x;
        if (ax > max_abs) max_abs = ax;
    }
    std::cout << "gpu-normalize ok: b=" << base << ", n=" << n
              << ", terms=" << terms << ", dup=" << (dup ? 1 : 0)
              << ", nonzero result digits=" << nonzero
              << ", max_abs_digit=" << max_abs << "\n";
}

void run_resident_squarecheck(uint64_t base, int n, uint64_t seed, int terms, bool dup) {
    const auto a = make_sparse_input(base, n, seed, terms);
    const auto ref = cuda_square_dup_gpu_norm(a, base, dup);
    GpuResidue r(base, n);
    r.set_coeffs(a);
    r.square_dup(dup);
    const auto got = r.get_digits();
    if (got != ref) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != ref[i]) {
                std::cerr << "first mismatch at " << i << ": ref=" << ref[i] << ", resident=" << got[i] << "\n";
                break;
            }
        }
        throw std::runtime_error("resident square output mismatch");
    }
    int nonzero = 0;
    for (const int64_t x : got) {
        if (x != 0) ++nonzero;
    }
    std::cout << "resident-squarecheck ok: b=" << base << ", n=" << n
              << ", terms=" << terms << ", dup=" << (dup ? 1 : 0)
              << ", nonzero result digits=" << nonzero
              << ", hash64=0x" << std::hex << std::setw(16) << std::setfill('0') << hash_digits64(got)
              << std::dec << "\n";
}

[[maybe_unused]] std::vector<int64_t> cuda_square_dup(const std::vector<int64_t>& a, uint64_t base, bool dup) {
    const auto lin = cuda_linear_square_residues(a);
    return normalize_from_residues(lin, base, dup ? 2u : 1u);
}

boost::multiprecision::cpp_int coeffs_to_cpp_int(const std::vector<int64_t>& a, uint64_t base) {
    using boost::multiprecision::cpp_int;
    cpp_int x = 0;
    for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i) {
        x *= base;
        x += a[i];
    }
    return x;
}

boost::multiprecision::cpp_int pow_cpp(uint64_t base, uint64_t exp) {
    using boost::multiprecision::cpp_int;
    cpp_int r = 1;
    cpp_int x = base;
    while (exp != 0) {
        if ((exp & 1) != 0) r *= x;
        x *= x;
        exp >>= 1;
    }
    return r;
}

boost::multiprecision::cpp_int powmod_cpp(boost::multiprecision::cpp_int a, boost::multiprecision::cpp_int e,
                                           const boost::multiprecision::cpp_int& mod) {
    using boost::multiprecision::cpp_int;
    cpp_int r = 1;
    a %= mod;
    while (e != 0) {
        if ((e & 1) != 0) r = (r * a) % mod;
        a = (a * a) % mod;
        e >>= 1;
    }
    return r;
}

void run_prp_small_impl(uint64_t base, int n, bool gpu_norm, bool resident) {
    using boost::multiprecision::cpp_int;
    if (n < 1 || n > 12) throw std::runtime_error("--prp-small is capped at n <= 12 for reference testing");
    if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("current digit output requires base < 2^63");
    }
    const int m = 1 << n;
    const cpp_int exponent = pow_cpp(base, static_cast<uint64_t>(m));
    const cpp_int modulus = exponent + 1;

    const int top = static_cast<int>(boost::multiprecision::msb(exponent));
    std::vector<int64_t> z;
    if (resident) {
        GpuResidue r(base, n);
        r.set_one();
        for (int i = top; i >= 0; --i) r.square_dup(bit_test(exponent, i));
        z = r.get_digits();
    } else {
        z.assign(m, 0);
        z[0] = 1;
        for (int i = top; i >= 0; --i) {
            const bool bit = bit_test(exponent, i);
            z = gpu_norm ? cuda_square_dup_gpu_norm(z, base, bit) : cuda_square_dup(z, base, bit);
        }
    }

    cpp_int cuda_res = coeffs_to_cpp_int(z, base) % modulus;
    if (cuda_res < 0) cuda_res += modulus;
    const cpp_int ref = powmod_cpp(2, exponent, modulus);
    const bool prp = (cuda_res == 1);
    std::cout << (resident ? "prp-small-resident" : (gpu_norm ? "prp-small-gpu-norm" : "prp-small"))
              << ": b=" << base << ", n=" << n
              << ", exponent_bits=" << (top + 1)
              << ", cuda_matches_ref=" << (cuda_res == ref ? "yes" : "no")
              << ", result_is_one=" << (prp ? "yes" : "no") << "\n";
    if (cuda_res != ref) throw std::runtime_error("CUDA PRP loop does not match cpp_int reference");
}

void run_prp_small(uint64_t base, int n) {
    run_prp_small_impl(base, n, false, false);
}

void run_prp_small_gpu_norm(uint64_t base, int n) {
    run_prp_small_impl(base, n, true, false);
}

void run_prp_small_resident(uint64_t base, int n) {
    run_prp_small_impl(base, n, true, true);
}

void run_resident_mulcheck(uint64_t base, int n) {
    using boost::multiprecision::cpp_int;
    if (n < 1 || n > 12) throw std::runtime_error("--resident-mulcheck is capped at n <= 12 for reference testing");
    if (base < 3) throw std::runtime_error("--resident-mulcheck needs base >= 3");
    const int m = 1 << n;
    const int64_t span = static_cast<int64_t>(std::max<uint64_t>(1, std::min<uint64_t>(base / 3, 1000)));
    std::vector<int64_t> a(m), b(m);
    for (int i = 0; i < m; ++i) {
        a[i] = static_cast<int64_t>((static_cast<uint64_t>(i) * 17u + 5u) % static_cast<uint64_t>(2 * span + 1)) - span;
        b[i] = static_cast<int64_t>((static_cast<uint64_t>(i) * 29u + 11u) % static_cast<uint64_t>(2 * span + 1)) - span;
    }
    a[0] += 1;
    b[1 % m] += 1;

    GpuResidue ra(base, n);
    GpuResidue rb(base, n);
    ra.set_coeffs(a);
    rb.set_coeffs(b);
    ra.mul_assign(rb);
    const auto got_digits = ra.get_digits();

    const cpp_int modulus = pow_cpp(base, static_cast<uint64_t>(m)) + 1;
    cpp_int got = coeffs_to_cpp_int(got_digits, base) % modulus;
    if (got < 0) got += modulus;
    cpp_int ref = (coeffs_to_cpp_int(a, base) * coeffs_to_cpp_int(b, base)) % modulus;
    if (ref < 0) ref += modulus;
    std::cout << "resident-mulcheck: b=" << base << ", n=" << n
              << ", cuda_matches_ref=" << (got == ref ? "yes" : "no")
              << ", hash64=0x" << std::hex << std::setw(16) << std::setfill('0') << hash_digits64(got_digits)
              << std::dec << "\n";
    if (got != ref) throw std::runtime_error("resident multiply does not match cpp_int reference");
}

struct CheckpointData {
    uint64_t base = 0;
    int n = 0;
    uint64_t total_bits = 0;
    uint64_t processed_bits = 0;
    std::vector<int64_t> digits;
};

bool digits_are_one(const std::vector<int64_t>& digits) {
    if (digits.empty() || digits[0] != 1) return false;
    for (size_t i = 1; i < digits.size(); ++i) {
        if (digits[i] != 0) return false;
    }
    return true;
}

std::string format_duration(double seconds) {
    if (seconds < 0 || !std::isfinite(seconds)) return "unknown";
    const uint64_t total = static_cast<uint64_t>(seconds + 0.5);
    const uint64_t h = total / 3600;
    const uint64_t m = (total / 60) % 60;
    const uint64_t s = total % 60;
    std::ostringstream os;
    if (h != 0) os << h << "h";
    if (h != 0 || m != 0) os << m << "m";
    os << s << "s";
    return os.str();
}

bool file_exists(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    return static_cast<bool>(is);
}

std::string make_anchor_path(const std::string& prefix, uint64_t processed_bits) {
    std::ostringstream os;
    os << prefix << "." << processed_bits << ".ckpt";
    return os.str();
}

void write_u32_le(std::ostream& os, uint32_t value) {
    uint8_t bytes[4];
    for (int i = 0; i < 4; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    os.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void write_u64_le(std::ostream& os, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    os.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

uint32_t read_u32_le(std::istream& is, const std::string& path) {
    uint8_t bytes[4];
    is.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!is) throw std::runtime_error("checkpoint header is truncated: " + path);
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= static_cast<uint32_t>(bytes[i]) << (8 * i);
    return value;
}

uint64_t read_u64_le(std::istream& is, const std::string& path) {
    uint8_t bytes[8];
    is.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!is) throw std::runtime_error("checkpoint header is truncated: " + path);
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    return value;
}

void validate_checkpoint_fields(uint64_t base, uint32_t n, uint64_t total_bits,
                                uint64_t processed_bits, uint64_t digit_count) {
    if (base < 2 || base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("checkpoint base is out of range");
    }
    if (n < 1 || n > 20 || digit_count != (uint64_t(1) << n)) {
        throw std::runtime_error("checkpoint has inconsistent n/digit_count");
    }
    if (total_bits == 0 || processed_bits > total_bits) {
        throw std::runtime_error("checkpoint has inconsistent bit counts");
    }
}

void validate_checkpoint_digits(uint64_t base, const std::vector<int64_t>& digits) {
    const int64_t lower = -static_cast<int64_t>((base - 1) / 2);
    const int64_t upper = static_cast<int64_t>(base / 2);
    bool is_even_base_special_residue = (base % 2 == 0 && !digits.empty() && digits[0] == -upper);
    if (is_even_base_special_residue) {
        for (size_t i = 1; i < digits.size(); ++i) {
            if (digits[i] != lower) {
                is_even_base_special_residue = false;
                break;
            }
        }
    }
    for (size_t i = 0; i < digits.size(); ++i) {
        const int64_t digit = digits[i];
        if (digit < lower || digit > upper) {
            if (!(is_even_base_special_residue && i == 0 && digit == -upper)) {
                throw std::runtime_error("checkpoint contains a digit outside the balanced range");
            }
        }
    }
}

void write_checkpoint_file(const std::string& path, uint64_t base, int n, uint64_t total_bits,
                           uint64_t processed_bits, const std::vector<int64_t>& digits) {
    if (n < 0) throw std::runtime_error("checkpoint n is negative");
    validate_checkpoint_fields(base, static_cast<uint32_t>(n), total_bits, processed_bits,
                               static_cast<uint64_t>(digits.size()));
    validate_checkpoint_digits(base, digits);
    static constexpr uint8_t magic[8] = {'B', 'G', 'F', 'N', 'C', 'K', '2', 0};
    const auto digest = checkpoint_sha256(static_cast<uint32_t>(n), base, total_bits, processed_bits, digits);

    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream os(tmp_path, std::ios::binary);
        if (!os) throw std::runtime_error("cannot open checkpoint temp file for writing: " + tmp_path);
        os.write(reinterpret_cast<const char*>(magic), sizeof(magic));
        write_u32_le(os, 2);
        write_u32_le(os, static_cast<uint32_t>(n));
        write_u64_le(os, base);
        write_u64_le(os, total_bits);
        write_u64_le(os, processed_bits);
        write_u64_le(os, static_cast<uint64_t>(digits.size()));
        os.write(reinterpret_cast<const char*>(digest.data()), static_cast<std::streamsize>(digest.size()));
        for (const int64_t digit : digits) write_u64_le(os, static_cast<uint64_t>(digit));
        os.flush();
        if (!os) throw std::runtime_error("failed while writing checkpoint: " + tmp_path);
    }
#ifdef _WIN32
    const std::filesystem::path tmp_fs(tmp_path);
    const std::filesystem::path target_fs(path);
    const std::wstring tmp_w = tmp_fs.wstring();
    const std::wstring target_w = target_fs.wstring();
    HANDLE tmp_handle = CreateFileW(
        tmp_w.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (tmp_handle == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        throw std::runtime_error(
            "cannot open checkpoint temp file for durable flush: " + tmp_path +
            " (Windows error " + std::to_string(static_cast<unsigned long>(error)) + ")");
    }
    if (!FlushFileBuffers(tmp_handle)) {
        const DWORD error = GetLastError();
        CloseHandle(tmp_handle);
        throw std::runtime_error(
            "cannot durably flush checkpoint temp file: " + tmp_path +
            " (Windows error " + std::to_string(static_cast<unsigned long>(error)) + ")");
    }
    CloseHandle(tmp_handle);
    if (!MoveFileExW(tmp_w.c_str(), target_w.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        throw std::runtime_error(
            "cannot move checkpoint temp file into place: " + path +
            " (Windows error " + std::to_string(static_cast<unsigned long>(error)) + ")");
    }
#else
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("cannot move checkpoint temp file into place: " + path);
    }
#endif
}

CheckpointData read_checkpoint_file(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("cannot open checkpoint file: " + path);

    uint8_t magic[8];
    is.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!is) throw std::runtime_error("checkpoint header is truncated: " + path);
    static constexpr uint8_t expected_magic[8] = {'B', 'G', 'F', 'N', 'C', 'K', '2', 0};
    if (std::memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        if (std::memcmp(magic, "BGFNCK1", 7) == 0) {
            throw std::runtime_error("legacy checkpoint v1 is not trusted; restart this candidate: " + path);
        }
        throw std::runtime_error("unsupported checkpoint format: " + path);
    }
    const uint32_t version = read_u32_le(is, path);
    const uint32_t n = read_u32_le(is, path);
    const uint64_t base = read_u64_le(is, path);
    const uint64_t total_bits = read_u64_le(is, path);
    const uint64_t processed_bits = read_u64_le(is, path);
    const uint64_t digit_count = read_u64_le(is, path);
    if (version != 2) throw std::runtime_error("unsupported checkpoint version: " + path);
    validate_checkpoint_fields(base, n, total_bits, processed_bits, digit_count);
    std::array<uint8_t, 32> expected_digest{};
    is.read(reinterpret_cast<char*>(expected_digest.data()), static_cast<std::streamsize>(expected_digest.size()));
    if (!is) throw std::runtime_error("checkpoint digest is truncated: " + path);

    CheckpointData c;
    c.base = base;
    c.n = static_cast<int>(n);
    c.total_bits = total_bits;
    c.processed_bits = processed_bits;
    c.digits.resize(static_cast<size_t>(digit_count));
    for (int64_t& digit : c.digits) digit = static_cast<int64_t>(read_u64_le(is, path));
    char trailing = 0;
    if (is.read(&trailing, 1)) throw std::runtime_error("checkpoint has trailing data: " + path);
    if (!is.eof()) throw std::runtime_error("failed while checking checkpoint length: " + path);
    validate_checkpoint_digits(c.base, c.digits);
    const auto got_digest = checkpoint_sha256(n, base, total_bits, processed_bits, c.digits);
    if (got_digest != expected_digest) throw std::runtime_error("checkpoint SHA-256 mismatch: " + path);
    return c;
}

void save_resident_checkpoint(const std::string& path, GpuResidue& r, uint64_t base, int n,
                              uint64_t total_bits, uint64_t processed_bits) {
    const auto digits = r.get_digits(false);
    write_checkpoint_file(path, base, n, total_bits, processed_bits, digits);
}

void print_final_residue_summary(const std::vector<int64_t>& digits, uint64_t processed_bits, uint64_t total_bits) {
    const uint64_t h = hash_digits64(digits);
    std::cout << ", hash64=0x" << std::hex << std::setw(16) << std::setfill('0') << h << std::dec;
    if (processed_bits == total_bits) {
        std::cout << ", result_is_one=" << (digits_are_one(digits) ? "yes" : "no");
    }
}

void run_prp_resident_prefix(uint64_t base, int n, uint64_t max_bits) {
    using boost::multiprecision::cpp_int;
    if (n < 1 || n > 20) throw std::runtime_error("n out of prototype range");
    if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("resident GPU path requires base < 2^63");
    }
    const uint64_t m = uint64_t(1) << n;
    const cpp_int exponent = pow_cpp(base, m);
    const int top = static_cast<int>(boost::multiprecision::msb(exponent));
    const uint64_t total_bits = static_cast<uint64_t>(top) + 1;
    const uint64_t todo = (max_bits == 0 || max_bits > total_bits) ? total_bits : max_bits;

    GpuResidue r(base, n);
    r.set_one();
    r.reset_stats();
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t k = 0; k < todo; ++k) {
        const int bit_index = top - static_cast<int>(k);
        r.square_dup(bit_test(exponent, bit_index));
    }
    r.finish_pending();
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t carry_iterations = r.carry_iterations();
    const uint64_t normalize_calls = r.normalize_calls();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const auto digits = r.get_digits(false);

    std::cout << "prp-resident-prefix: b=" << base << ", n=" << n
              << ", processed_bits=" << todo << "/" << total_bits
              << ", seconds=" << std::fixed << std::setprecision(3) << seconds
              << ", ms_per_bit=" << (seconds * 1000.0 / static_cast<double>(todo))
              << ", avg_carry_iters=" << (normalize_calls == 0 ? 0.0 : static_cast<double>(carry_iterations) / static_cast<double>(normalize_calls));
    print_final_residue_summary(digits, todo, total_bits);
    std::cout << "\n";
}

void run_prp_resident_checkpoint(uint64_t base, int n, uint64_t max_bits,
                                 const std::string& checkpoint_path, uint64_t save_every_bits) {
    using boost::multiprecision::cpp_int;
    if (n < 1 || n > 20) throw std::runtime_error("n out of prototype range");
    if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("resident GPU path requires base < 2^63");
    }
    const uint64_t m = uint64_t(1) << n;
    const cpp_int exponent = pow_cpp(base, m);
    const int top = static_cast<int>(boost::multiprecision::msb(exponent));
    const uint64_t total_bits = static_cast<uint64_t>(top) + 1;
    uint64_t target = (max_bits == 0 || max_bits > total_bits) ? total_bits : max_bits;

    GpuResidue r(base, n);
    r.set_one();
    r.reset_stats();
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t k = 0; k < target; ++k) {
        const int bit_index = top - static_cast<int>(k);
        r.square_dup(bit_test(exponent, bit_index));
        if (save_every_bits != 0 && ((k + 1) % save_every_bits) == 0) {
            save_resident_checkpoint(checkpoint_path, r, base, n, total_bits, k + 1);
        }
    }
    save_resident_checkpoint(checkpoint_path, r, base, n, total_bits, target);
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t carry_iterations = r.carry_iterations();
    const uint64_t normalize_calls = r.normalize_calls();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const auto digits = r.get_digits(false);
    std::cout << "prp-resident-checkpoint: b=" << base << ", n=" << n
              << ", processed_bits=" << target << "/" << total_bits
              << ", seconds=" << std::fixed << std::setprecision(3) << seconds
              << ", ms_per_bit=" << (target == 0 ? 0.0 : seconds * 1000.0 / static_cast<double>(target))
              << ", avg_carry_iters=" << (normalize_calls == 0 ? 0.0 : static_cast<double>(carry_iterations) / static_cast<double>(normalize_calls))
              << ", checkpoint=" << checkpoint_path;
    print_final_residue_summary(digits, target, total_bits);
    std::cout << "\n";
}

void run_prp_resident_resume(const std::string& checkpoint_path, uint64_t more_bits, uint64_t save_every_bits) {
    using boost::multiprecision::cpp_int;
    CheckpointData c = read_checkpoint_file(checkpoint_path);
    if (c.n < 1 || c.n > 20) throw std::runtime_error("checkpoint n out of prototype range");
    if (c.base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("resident GPU path requires base < 2^63");
    }
    const uint64_t m = uint64_t(1) << c.n;
    const cpp_int exponent = pow_cpp(c.base, m);
    const int top = static_cast<int>(boost::multiprecision::msb(exponent));
    const uint64_t total_bits = static_cast<uint64_t>(top) + 1;
    if (total_bits != c.total_bits) throw std::runtime_error("checkpoint total_bits does not match recomputed exponent");
    if (c.processed_bits > total_bits) throw std::runtime_error("checkpoint processed_bits is past the end");
    const uint64_t target = (more_bits == 0 || more_bits > total_bits - c.processed_bits)
        ? total_bits
        : c.processed_bits + more_bits;

    GpuResidue r(c.base, c.n);
    r.set_coeffs(c.digits);
    r.reset_stats();
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t k = c.processed_bits; k < target; ++k) {
        const int bit_index = top - static_cast<int>(k);
        r.square_dup(bit_test(exponent, bit_index));
        if (save_every_bits != 0 && ((k + 1) % save_every_bits) == 0) {
            save_resident_checkpoint(checkpoint_path, r, c.base, c.n, total_bits, k + 1);
        }
    }
    save_resident_checkpoint(checkpoint_path, r, c.base, c.n, total_bits, target);
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t carry_iterations = r.carry_iterations();
    const uint64_t normalize_calls = r.normalize_calls();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const auto digits = r.get_digits(false);
    const uint64_t done_now = target - c.processed_bits;
    std::cout << "prp-resident-resume: b=" << c.base << ", n=" << c.n
              << ", processed_bits=" << target << "/" << total_bits
              << ", advanced_bits=" << done_now
              << ", seconds=" << std::fixed << std::setprecision(3) << seconds
              << ", ms_per_bit=" << (done_now == 0 ? 0.0 : seconds * 1000.0 / static_cast<double>(done_now))
              << ", avg_carry_iters=" << (normalize_calls == 0 ? 0.0 : static_cast<double>(carry_iterations) / static_cast<double>(normalize_calls))
              << ", checkpoint=" << checkpoint_path;
    print_final_residue_summary(digits, target, total_bits);
    std::cout << "\n";
}

void run_checkpoint_info(const std::string& checkpoint_path) {
    const CheckpointData c = read_checkpoint_file(checkpoint_path);
    const uint64_t h = hash_digits64(c.digits);
    std::cout << "checkpoint-info: path=" << checkpoint_path
              << ", b=" << c.base
              << ", n=" << c.n
              << ", processed_bits=" << c.processed_bits << "/" << c.total_bits
              << ", digit_count=" << c.digits.size()
              << ", hash64=0x" << std::hex << std::setw(16) << std::setfill('0') << h << std::dec;
    if (c.processed_bits == c.total_bits) {
        std::cout << ", result_is_one=" << (digits_are_one(c.digits) ? "yes" : "no");
    }
    std::cout << "\n";
}

void run_checkpoint_compare(const std::string& left_path, const std::string& right_path) {
    const CheckpointData a = read_checkpoint_file(left_path);
    const CheckpointData b = read_checkpoint_file(right_path);
    const bool same_meta = (a.base == b.base && a.n == b.n && a.total_bits == b.total_bits &&
                            a.processed_bits == b.processed_bits && a.digits.size() == b.digits.size());
    bool same_digits = same_meta;
    size_t mismatch = 0;
    if (same_meta) {
        for (; mismatch < a.digits.size(); ++mismatch) {
            if (a.digits[mismatch] != b.digits[mismatch]) {
                same_digits = false;
                break;
            }
        }
    }
    std::cout << "checkpoint-compare: left=" << left_path
              << ", right=" << right_path
              << ", same_meta=" << (same_meta ? "yes" : "no")
              << ", same_digits=" << (same_digits ? "yes" : "no");
    if (!same_meta) {
        std::cout << ", left_bits=" << a.processed_bits << "/" << a.total_bits
                  << ", right_bits=" << b.processed_bits << "/" << b.total_bits;
    } else if (!same_digits) {
        std::cout << ", first_mismatch_digit=" << mismatch
                  << ", left_digit=" << a.digits[mismatch]
                  << ", right_digit=" << b.digits[mismatch];
    }
    std::cout << "\n";
    if (!same_meta || !same_digits) throw std::runtime_error("checkpoint comparison failed");
}

void run_prp_resident_run(uint64_t base, int n, uint64_t max_bits, const std::string& checkpoint_path,
                          uint64_t checkpoint_every_bits, uint64_t progress_every_bits, bool resume,
                          const std::string& anchor_prefix = std::string(), uint64_t anchor_every_bits = 0) {
    using boost::multiprecision::cpp_int;
    if (n < 1 || n > 20) throw std::runtime_error("n out of prototype range");
    if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("resident GPU path requires base < 2^63");
    }
    install_gfps_signal_handlers();

    const uint64_t m = uint64_t(1) << n;
    const cpp_int exponent = pow_cpp(base, m);
    const int top = static_cast<int>(boost::multiprecision::msb(exponent));
    const uint64_t total_bits = static_cast<uint64_t>(top) + 1;
    uint64_t target = (max_bits == 0 || max_bits > total_bits) ? total_bits : max_bits;
    uint64_t start = 0;

    GpuResidue r(base, n);
    if (resume) {
        if (!file_exists(checkpoint_path)) throw std::runtime_error("resume requested but checkpoint file does not exist");
        CheckpointData c = read_checkpoint_file(checkpoint_path);
        if (c.base != base || c.n != n) throw std::runtime_error("checkpoint base/n does not match run arguments");
        if (c.total_bits != total_bits) throw std::runtime_error("checkpoint total_bits does not match recomputed exponent");
        if (c.processed_bits > total_bits) throw std::runtime_error("checkpoint processed_bits is past the end");
        r.set_coeffs(c.digits);
        start = c.processed_bits;
        std::cout << "prp-resident-run: resumed checkpoint=" << checkpoint_path
                  << ", processed_bits=" << start << "/" << total_bits << "\n";
    } else {
        r.set_one();
        std::cout << "prp-resident-run: fresh start, checkpoint=" << checkpoint_path << "\n";
    }
    if (anchor_every_bits != 0 && !anchor_prefix.empty() && (start % anchor_every_bits) == 0) {
        const std::string anchor_path = make_anchor_path(anchor_prefix, start);
        save_resident_checkpoint(anchor_path, r, base, n, total_bits, start);
        std::cout << "anchor: processed_bits=" << start << ", path=" << anchor_path << "\n";
    }
    if (start > target) {
        std::cout << "prp-resident-run: checkpoint is already beyond requested target, no square steps run\n";
        target = start;
    }
    if (g_gfps_stop_requested != 0) {
        save_resident_checkpoint(checkpoint_path, r, base, n, total_bits, start);
        throw GfpsInterrupted("interrupted before the first resident exponent bit");
    }

    const uint64_t run_start = start;
    const uint64_t todo = target > start ? target - start : 0;
    if (progress_every_bits == 0) {
        progress_every_bits = std::max<uint64_t>(1, std::min<uint64_t>(1000, std::max<uint64_t>(todo / 100, 1)));
    }

    r.reset_stats();
    const auto t0 = std::chrono::steady_clock::now();
    auto last_progress = t0;
    uint64_t last_progress_bit = run_start;

    for (uint64_t k = run_start; k < target; ++k) {
        const int bit_index = top - static_cast<int>(k);
        r.square_dup(bit_test(exponent, bit_index));
        const uint64_t done = k + 1;

        if (g_gfps_stop_requested != 0) {
            save_resident_checkpoint(checkpoint_path, r, base, n, total_bits, done);
            std::cout << "checkpoint: interrupt saved, processed_bits=" << done
                      << "/" << total_bits << ", path=" << checkpoint_path << "\n";
            throw GfpsInterrupted("interrupted after saving a safe resident checkpoint");
        }

        if (checkpoint_every_bits != 0 && (done % checkpoint_every_bits) == 0) {
            save_resident_checkpoint(checkpoint_path, r, base, n, total_bits, done);
        }
        if (anchor_every_bits != 0 && !anchor_prefix.empty() && (done % anchor_every_bits) == 0) {
            const std::string anchor_path = make_anchor_path(anchor_prefix, done);
            save_resident_checkpoint(anchor_path, r, base, n, total_bits, done);
            std::cout << "anchor: processed_bits=" << done << ", path=" << anchor_path << "\n";
        }

        const bool should_report = (done == target) || ((done - run_start) % progress_every_bits) == 0;
        if (should_report) {
            r.finish_pending();
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            const double interval = std::chrono::duration<double>(now - last_progress).count();
            const uint64_t advanced = done - run_start;
            const uint64_t interval_bits = done - last_progress_bit;
            const double avg_ms = advanced == 0 ? 0.0 : elapsed * 1000.0 / static_cast<double>(advanced);
            const double interval_ms = interval_bits == 0 ? 0.0 : interval * 1000.0 / static_cast<double>(interval_bits);
            const double eta = (avg_ms <= 0.0) ? -1.0 : (static_cast<double>(target - done) * avg_ms / 1000.0);
            const uint64_t carry_iterations = r.carry_iterations();
            const uint64_t normalize_calls = r.normalize_calls();
            std::cout << "progress: processed_bits=" << done << "/" << total_bits
                      << ", target=" << target
                      << ", interval_ms_per_bit=" << std::fixed << std::setprecision(3) << interval_ms
                      << ", avg_ms_per_bit=" << avg_ms
                      << ", avg_carry_iters=" << (normalize_calls == 0 ? 0.0 : static_cast<double>(carry_iterations) / static_cast<double>(normalize_calls))
                      << ", elapsed=" << format_duration(elapsed)
                      << ", eta_to_target=" << format_duration(eta)
                      << "\n";
            last_progress = now;
            last_progress_bit = done;
        }
    }

    save_resident_checkpoint(checkpoint_path, r, base, n, total_bits, target);
    if (anchor_every_bits != 0 && !anchor_prefix.empty() && (target % anchor_every_bits) != 0) {
        const std::string anchor_path = make_anchor_path(anchor_prefix, target);
        save_resident_checkpoint(anchor_path, r, base, n, total_bits, target);
        std::cout << "anchor: processed_bits=" << target << ", path=" << anchor_path << "\n";
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const uint64_t carry_iterations = r.carry_iterations();
    const uint64_t normalize_calls = r.normalize_calls();
    const auto digits = r.get_digits(false);

    std::cout << "prp-resident-run: b=" << base << ", n=" << n
              << ", processed_bits=" << target << "/" << total_bits
              << ", advanced_bits=" << todo
              << ", seconds=" << std::fixed << std::setprecision(3) << seconds
              << ", ms_per_bit=" << (todo == 0 ? 0.0 : seconds * 1000.0 / static_cast<double>(todo))
              << ", avg_carry_iters=" << (normalize_calls == 0 ? 0.0 : static_cast<double>(carry_iterations) / static_cast<double>(normalize_calls))
              << ", checkpoint=" << checkpoint_path;
    print_final_residue_summary(digits, target, total_bits);
    std::cout << "\n";
}

void run_verify_transition(const std::string& start_path, const std::string& end_path) {
    using boost::multiprecision::cpp_int;
    CheckpointData a = read_checkpoint_file(start_path);
    CheckpointData b = read_checkpoint_file(end_path);
    if (a.base != b.base || a.n != b.n || a.total_bits != b.total_bits) {
        throw std::runtime_error("transition checkpoints do not describe the same GFN");
    }
    if (a.processed_bits > b.processed_bits) {
        throw std::runtime_error("transition start is after transition end");
    }
    if (a.n < 1 || a.n > 20) throw std::runtime_error("checkpoint n out of prototype range");

    const uint64_t m = uint64_t(1) << a.n;
    const cpp_int exponent = pow_cpp(a.base, m);
    const int top = static_cast<int>(boost::multiprecision::msb(exponent));
    const uint64_t total_bits = static_cast<uint64_t>(top) + 1;
    if (total_bits != a.total_bits) throw std::runtime_error("checkpoint total_bits does not match recomputed exponent");

    GpuResidue r(a.base, a.n);
    r.set_coeffs(a.digits);
    r.reset_stats();
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t k = a.processed_bits; k < b.processed_bits; ++k) {
        const int bit_index = top - static_cast<int>(k);
        r.square_dup(bit_test(exponent, bit_index));
    }
    r.finish_pending();
    const auto t1 = std::chrono::steady_clock::now();
    const auto got = r.get_digits(false);
    if (got != b.digits) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != b.digits[i]) {
                std::cerr << "first mismatch at digit " << i
                          << ": recomputed=" << got[i] << ", checkpoint=" << b.digits[i] << "\n";
                break;
            }
        }
        throw std::runtime_error("transition verification failed");
    }
    const uint64_t advanced = b.processed_bits - a.processed_bits;
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const uint64_t carry_iterations = r.carry_iterations();
    const uint64_t normalize_calls = r.normalize_calls();
    std::cout << "verify-transition ok: b=" << a.base << ", n=" << a.n
              << ", start_bits=" << a.processed_bits
              << ", end_bits=" << b.processed_bits
              << ", advanced_bits=" << advanced
              << ", seconds=" << std::fixed << std::setprecision(3) << seconds
              << ", ms_per_bit=" << (advanced == 0 ? 0.0 : seconds * 1000.0 / static_cast<double>(advanced))
              << ", avg_carry_iters=" << (normalize_calls == 0 ? 0.0 : static_cast<double>(carry_iterations) / static_cast<double>(normalize_calls));
    print_final_residue_summary(got, b.processed_bits, b.total_bits);
    std::cout << "\n";
}

void run_verify_chain(const std::string& anchor_prefix, uint64_t start_bits, uint64_t end_bits, uint64_t step_bits) {
    if (step_bits == 0) throw std::runtime_error("--verify-chain step-bits must be nonzero");
    if (start_bits > end_bits) throw std::runtime_error("--verify-chain start-bits is after end-bits");
    uint64_t segments = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t a = start_bits; a < end_bits; ) {
        const uint64_t b = std::min<uint64_t>(a + step_bits, end_bits);
        run_verify_transition(make_anchor_path(anchor_prefix, a), make_anchor_path(anchor_prefix, b));
        a = b;
        ++segments;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "verify-chain ok: prefix=" << anchor_prefix
              << ", start_bits=" << start_bits
              << ", end_bits=" << end_bits
              << ", step_bits=" << step_bits
              << ", segments=" << segments
              << ", seconds=" << std::fixed << std::setprecision(3) << seconds
              << "\n";
}

void run_bench_resident(uint64_t base, int n, uint64_t bits, uint64_t checkpoint_every_bits,
                        const std::string& checkpoint_path, int carry_block_size = 256) {
    using boost::multiprecision::cpp_int;
    if (n < 1 || n > 20) throw std::runtime_error("n out of prototype range");
    if (bits == 0) throw std::runtime_error("--bench-resident bits must be nonzero");
    const uint64_t m = uint64_t(1) << n;
    const cpp_int exponent = pow_cpp(base, m);
    const int top = static_cast<int>(boost::multiprecision::msb(exponent));
    const uint64_t total_bits = static_cast<uint64_t>(top) + 1;
    const uint64_t todo = std::min(bits, total_bits);

    GpuResidue r(base, n, carry_block_size);
    r.set_one();
    r.reset_stats();
    r.set_timing(true);

    double checkpoint_seconds = 0.0;
    uint64_t checkpoint_count = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t k = 0; k < todo; ++k) {
        const int bit_index = top - static_cast<int>(k);
        r.square_dup(bit_test(exponent, bit_index));
        if (checkpoint_every_bits != 0 && ((k + 1) % checkpoint_every_bits) == 0) {
            // Finish GPU work before timing checkpoint serialization to avoid
            // counting pending batch execution in both benchmark phases.
            r.finish_pending();
            const auto c0 = std::chrono::steady_clock::now();
            save_resident_checkpoint(checkpoint_path, r, base, n, total_bits, k + 1);
            const auto c1 = std::chrono::steady_clock::now();
            checkpoint_seconds += std::chrono::duration<double>(c1 - c0).count();
            ++checkpoint_count;
        }
    }
    r.finish_pending();
    const auto t1 = std::chrono::steady_clock::now();
    const auto digits = r.get_digits(false);
    const double total_seconds = std::chrono::duration<double>(t1 - t0).count();
    const uint64_t carry_iterations = r.carry_iterations();
    const uint64_t normalize_calls = r.normalize_calls();
    const double transform_seconds = r.transform_seconds();
    const double normalize_seconds = r.normalize_seconds();
    const double accounted = transform_seconds + normalize_seconds + checkpoint_seconds;

    std::cout << "bench-resident: b=" << base << ", n=" << n
              << ", rns_primes=" << r.mod_count()
              << ", carry_block_size=" << r.carry_block_size()
              << ", bits=" << todo << "/" << total_bits
              << ", total_seconds=" << std::fixed << std::setprecision(3) << total_seconds
              << ", total_ms_per_bit=" << (total_seconds * 1000.0 / static_cast<double>(todo));
    std::cout << ", transform_normalize_combined_ms_per_bit="
              << ((transform_seconds + normalize_seconds) * 1000.0 /
                  static_cast<double>(todo));
    std::cout
              << ", checkpoint_count=" << checkpoint_count
              << ", checkpoint_ms_each=" << (checkpoint_count == 0 ? 0.0 : checkpoint_seconds * 1000.0 / static_cast<double>(checkpoint_count))
              << ", unaccounted_ms_per_bit=" << ((total_seconds - accounted) * 1000.0 / static_cast<double>(todo))
              << ", avg_carry_iters=" << (normalize_calls == 0 ? 0.0 : static_cast<double>(carry_iterations) / static_cast<double>(normalize_calls));
    print_final_residue_summary(digits, todo, total_bits);
    std::cout << "\n";
}
void run_analyze(uint64_t base, int n) {
    const long double log2b = std::log2(static_cast<long double>(base));
    const long double m = std::ldexp(1.0L, n);
    const long double exp_bits = m * log2b;
    const long double balanced_coeff_bits = n + 2.0L * (log2b - 1.0L);
    const long double three_prime_bits =
        std::log2(static_cast<long double>(kMods[0].p)) +
        std::log2(static_cast<long double>(kMods[1].p)) +
        std::log2(static_cast<long double>(kMods[2].p));
    const long double four_prime_bits = three_prime_bits + std::log2(1224736769.0L);

    std::cout << "N = b^(2^n)+1 analysis\n";
    std::cout << "b = " << base << ", n = " << n << ", M = 2^n = " << static_cast<uint64_t>(m) << "\n";
    std::cout << "decimal digits ~= " << static_cast<uint64_t>(exp_bits / std::log2(10.0L)) + 1 << "\n";
    std::cout << "Fermat exponent bits ~= " << static_cast<uint64_t>(exp_bits) + 1 << "\n";
    std::cout << "one balanced square coefficient bound ~= 2^" << static_cast<double>(balanced_coeff_bits) << "\n";
    std::cout << "3-prime NTT product bits ~= " << static_cast<double>(three_prime_bits) << "\n";
    std::cout << "4-prime NTT product bits ~= " << static_cast<double>(four_prime_bits) << "\n";
    if (balanced_coeff_bits + 1.0L < three_prime_bits) {
        std::cout << "prototype 3-prime CRT range is enough for one balanced square coefficient\n";
    } else if (balanced_coeff_bits + 1.0L < four_prime_bits) {
        std::cout << "4-prime/128-bit prototype range is enough for one balanced square coefficient\n";
    } else {
        std::cout << "this base/n needs more than four 31-bit NTT primes or a smaller digit split\n";
    }
}

void display_banner() {
    printf("%s\n","════════════════════════════════════════════════════════════════════════════════════════");
    printf("%s\n","            .oooooo.        oooooooooooo     ooooooooo.        .oooooo..o               ");
    printf("%s\n","           d8P'  `Y8b       `888'     `8     `888   `Y88.     d8P'    `Y8               ");
    printf("%s\n","          888                888              888   .d88'     Y88bo.                    ");
    printf("%s\n","          888                888oooo8         888ooo88P'       `'Y8888o.                ");
    printf("%s\n","          888     ooooo      888    '         888                  `'Y88b               ");
    printf("%s\n","          `88.    .88'  .o.  888         .o.  888         .o. oo     .d8P .o.           ");
    printf("%s\n","           `Y8bood8P'   Y8P o888o        Y8P o888o        Y8P 8''88888P'  Y8P           ");
    printf("%s\n","════════════════════════════════════════════════════════════════════════════════════════");
    printf("%s\n","                            Generalized-Fermat-Primes-Seeker                            ");
    printf("%s\n","                           Version 4.1 CUDA by A.P. Sept 2026                           ");
}

void usage(const char* argv0) {
    display_banner();
    std::cout
        << "global GPU options (may appear anywhere):\n"
        << "  --ntt-blocks <1..4096>    force maximum NTT blocks; omit for automatic selection\n"
        << "  --duty-percent <1..100>   resident-square time duty cycle; default 100\n"
        << "  --batch-bits <0..4096>    checked carry batch size; default 512, 0=adaptive per square\n"
        << "                           batching is disabled whenever duty-percent is below 100\n"
        << "                           Windows duty pauses follow approximately 10 ms work windows\n"
        << "  --reference-mode         original padded NTT and adaptive per-square carry\n"
        << "  --diagnostic-force-replay replay every carry batch for correctness testing\n"
        << "diagnostic environment controls (presence disables the named optimization):\n"
        << "  GFPS_DISABLE_NEGACYCLIC, GFPS_DISABLE_DIF, GFPS_DISABLE_SHARED,\n"
        << "  GFPS_DISABLE_FUSED_TILE, GFPS_DISABLE_CUDA_GRAPHS\n"
        << "  GFPS_BATCH_BITS=0..4096; GFPS_CARRY_PASSES=3..8; GFPS_FORCE_REPLAY=1\n"
        << "usage:\n"
        << "  " << argv0 << " --selftest\n"
        << "  " << argv0 << " --analyze <b> <n>\n"
        << "  " << argv0 << " --square <b> <n>\n"
        << "  " << argv0 << " --sparse <b> <n> <seed> <terms>\n"
        << "  " << argv0 << " --gpu-norm <b> <n> <seed> <terms> <dup:0|1>\n"
        << "  " << argv0 << " --sqrchk <b> <n> <seed> <terms> <dup:0|1>\n"
        << "  " << argv0 << " --prp-small <b> <n>\n"
        << "  " << argv0 << " --prp-small-gpu-norm <b> <n>\n"
        << "  " << argv0 << " --prp-small-resident <b> <n>\n"
        << "  " << argv0 << " --resident-mulcheck <b> <n>\n"
        << "  " << argv0 << " --prp-check-pref <b> <n> <max-bits, 0=full>\n"
        << "  " << argv0 << " --prp-check-ckpt <b> <n> <max-bits, 0=full> <checkpoint-file> <save-every-bits, 0=end-only>\n"
        << "  " << argv0 << " --prp-check-resume <checkpoint-file> <more-bits, 0=to-end> <save-every-bits, 0=end-only>\n"
        << "  " << argv0 << " --prp-check <b> <n> <target-bits, 0=full> <checkpoint-file> <checkpoint-every-bits, 0=end-only> <progress-every-bits, 0=auto> <resume:0|1>\n"
        << "  " << argv0 << " --prp-check-a <b> <n> <target-bits, 0=full> <checkpoint-file> <checkpoint-every-bits, 0=end-only> <anchor-prefix> <anchor-every-bits> <progress-every-bits, 0=auto> <resume:0|1>\n"
        << "  " << argv0 << " --ckpt-info <checkpoint-file>\n"
        << "  " << argv0 << " --ckpt-cmp <left-checkpoint> <right-checkpoint>\n"
        << "  " << argv0 << " --verify-transition <start-checkpoint> <end-checkpoint>\n"
        << "  " << argv0 << " --verify-chain <anchor-prefix> <start-bits> <end-bits> <step-bits>\n"
        << "  " << argv0 << " --bench-resident <b> <n> <bits> <checkpoint-every-bits, 0=none> <checkpoint-file> [carry-block-size]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    try {
        std::vector<char*> filtered_argv = consume_gpu_throttle_args(argc, argv);
        argc = static_cast<int>(filtered_argv.size());
        argv = filtered_argv.data();
        print_gpu_throttle_config();
        if (argc == 2 && std::string(argv[1]) == "--selftest") {
            run_selftest();
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--analyze") {
            run_analyze(parse_u64(argv[2]), std::stoi(argv[3]));
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--square") {
            run_square(parse_u64(argv[2]), std::stoi(argv[3]));
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--sparse") {
            run_sparse(parse_u64(argv[2]), std::stoi(argv[3]), parse_u64(argv[4]), std::stoi(argv[5]));
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--gpu-norm") {
            run_gpu_norm(parse_u64(argv[2]), std::stoi(argv[3]), parse_u64(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]) != 0);
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--sqrchk") {
            run_resident_squarecheck(parse_u64(argv[2]), std::stoi(argv[3]), parse_u64(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]) != 0);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--prp-small") {
            run_prp_small(parse_u64(argv[2]), std::stoi(argv[3]));
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--prp-small-gpu-norm") {
            run_prp_small_gpu_norm(parse_u64(argv[2]), std::stoi(argv[3]));
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--prp-small-resident") {
            run_prp_small_resident(parse_u64(argv[2]), std::stoi(argv[3]));
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--resident-mulcheck") {
            run_resident_mulcheck(parse_u64(argv[2]), std::stoi(argv[3]));
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--prp-check-pref") {
            run_prp_resident_prefix(parse_u64(argv[2]), std::stoi(argv[3]), parse_u64(argv[4]));
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--prp-check-ckpt") {
            run_prp_resident_checkpoint(parse_u64(argv[2]), std::stoi(argv[3]), parse_u64(argv[4]),
                                        argv[5], parse_u64(argv[6]));
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--prp-check-resume") {
            run_prp_resident_resume(argv[2], parse_u64(argv[3]), parse_u64(argv[4]));
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "--prp-check") {
            run_prp_resident_run(parse_u64(argv[2]), std::stoi(argv[3]), parse_u64(argv[4]),
                                 argv[5], parse_u64(argv[6]), parse_u64(argv[7]), std::stoi(argv[8]) != 0);
            return 0;
        }
        if (argc == 11 && std::string(argv[1]) == "--prp-check-a") {
            run_prp_resident_run(parse_u64(argv[2]), std::stoi(argv[3]), parse_u64(argv[4]),
                                 argv[5], parse_u64(argv[6]), parse_u64(argv[9]), std::stoi(argv[10]) != 0,
                                 argv[7], parse_u64(argv[8]));
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--ckpt-info") {
            run_checkpoint_info(argv[2]);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--ckpt-cmp") {
            run_checkpoint_compare(argv[2], argv[3]);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--verify-transition") {
            run_verify_transition(argv[2], argv[3]);
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--verify-chain") {
            run_verify_chain(argv[2], parse_u64(argv[3]), parse_u64(argv[4]), parse_u64(argv[5]));
            return 0;
        }
        if ((argc == 7 || argc == 8) && std::string(argv[1]) == "--bench-resident") {
            run_bench_resident(parse_u64(argv[2]), std::stoi(argv[3]), parse_u64(argv[4]), parse_u64(argv[5]), argv[6],
                               argc == 8 ? std::stoi(argv[7]) : 256);
            return 0;
        }
        usage(argv[0]);
        return 1;
    } catch (const GfpsInterrupted& e) {
        std::cerr << "interrupted: " << e.what() << "\n";
        return 130;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
