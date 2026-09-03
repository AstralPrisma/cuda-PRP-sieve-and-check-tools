/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <boost/multiprecision/cpp_int.hpp>
#ifdef GSRPS_GMP_DIAGNOSTIC
#include <gmpxx.h>
#endif
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

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

// Runtime tuning and throttling controls.
//
// Environment variables (backward compatible):
//   GSRPS_NTT_BLOCKS=1..4096       (forces a value and disables auto-tuning)
//   GSRPS_WINDOW_BITS=1..8         (forces a value and disables window selection)
//   GSRPS_DUTY_PERCENT=1..100
//   GSRPS_TUNING_CACHE_DIR=<path>   (default .gsrps_tuning_cache)
//   GSRPS_TUNING_CACHE_MAX_AGE_HOURS=1..8760 (default 24)
//   GSRPS_DISABLE_TUNING_CACHE=1
//
// Command-line options (override environment variables and may appear anywhere):
//   --force-ntt-blocks <1..4096> or --force-ntt-blocks=<1..4096>
//   --force-window-bits <1..8>   or --force-window-bits=<1..8>
//   --duty-percent <1..100>     or --duty-percent=<1..100>
//   --verify-cpp-int            independently repeat --check on the CPU
//   --tuning-cache-dir <path>
//   --tuning-cache-max-age-hours <1..8760>
//   --no-tuning-cache
constexpr int kDefaultNttBlocks = 64;
constexpr int kMaxNttBlocks = 4096;  // n <= 20: at most 2^20 butterflies / 256 threads.
constexpr int kDefaultDutyPercent = 100;
#ifndef GSRPS_BUILD_ID
#define GSRPS_BUILD_ID __DATE__ "_" __TIME__
#endif

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
    int window_bits;
    bool force_window_bits;
    int duty_percent;
    bool verify_cpp_int;
    std::string tuning_cache_dir;
    bool tuning_cache_enabled;
    int tuning_cache_max_age_hours;
};

GpuThrottleConfig& gpu_throttle_config() {
    static GpuThrottleConfig config = [] {
        const char* ntt_text = std::getenv("GSRPS_NTT_BLOCKS");
        const char* window_text = std::getenv("GSRPS_WINDOW_BITS");
        const char* cache_dir_text = std::getenv("GSRPS_TUNING_CACHE_DIR");
        return GpuThrottleConfig{
            ntt_text != nullptr && *ntt_text != '\0'
                ? parse_bounded_int(ntt_text, "GSRPS_NTT_BLOCKS", 1, kMaxNttBlocks)
                : kDefaultNttBlocks,
            ntt_text != nullptr && *ntt_text != '\0',
            window_text != nullptr && *window_text != '\0'
                ? parse_bounded_int(window_text, "GSRPS_WINDOW_BITS", 1, 8)
                : 0,
            window_text != nullptr && *window_text != '\0',
            read_env_int("GSRPS_DUTY_PERCENT", kDefaultDutyPercent, 1, 100),
            false,
            cache_dir_text != nullptr && *cache_dir_text != '\0'
                ? std::string(cache_dir_text)
                : std::string(".gsrps_tuning_cache"),
            std::getenv("GSRPS_DISABLE_TUNING_CACHE") == nullptr,
            read_env_int("GSRPS_TUNING_CACHE_MAX_AGE_HOURS", 24, 1, 8760),
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

bool cpp_int_verification_enabled() {
    return gpu_throttle_config().verify_cpp_int;
}

bool tuning_cache_enabled() {
    return gpu_throttle_config().tuning_cache_enabled;
}

const std::string& tuning_cache_dir() {
    return gpu_throttle_config().tuning_cache_dir;
}

int tuning_cache_max_age_hours() {
    return gpu_throttle_config().tuning_cache_max_age_hours;
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
        if (arg == "--force-ntt-blocks") {
            gpu_throttle_config().ntt_blocks =
                parse_bounded_int(require_next(i, "--force-ntt-blocks"),
                                  "--force-ntt-blocks", 1, kMaxNttBlocks);
            gpu_throttle_config().force_ntt_blocks = true;
        } else if (arg.rfind("--force-ntt-blocks=", 0) == 0) {
            gpu_throttle_config().ntt_blocks =
                parse_bounded_int(arg.substr(std::strlen("--force-ntt-blocks=")),
                                  "--force-ntt-blocks", 1, kMaxNttBlocks);
            gpu_throttle_config().force_ntt_blocks = true;
        } else if (arg == "--ntt-blocks" || arg.rfind("--ntt-blocks=", 0) == 0) {
            throw std::runtime_error(
                "--ntt-blocks was renamed to --force-ntt-blocks; omit it to use automatic tuning");
        } else if (arg == "--force-window-bits") {
            gpu_throttle_config().window_bits =
                parse_bounded_int(require_next(i, "--force-window-bits"),
                                  "--force-window-bits", 1, 8);
            gpu_throttle_config().force_window_bits = true;
        } else if (arg.rfind("--force-window-bits=", 0) == 0) {
            gpu_throttle_config().window_bits =
                parse_bounded_int(arg.substr(std::strlen("--force-window-bits=")),
                                  "--force-window-bits", 1, 8);
            gpu_throttle_config().force_window_bits = true;
        } else if (arg == "--duty-percent") {
            gpu_throttle_config().duty_percent =
                parse_bounded_int(require_next(i, "--duty-percent"), "--duty-percent", 1, 100);
        } else if (arg.rfind("--duty-percent=", 0) == 0) {
            gpu_throttle_config().duty_percent =
                parse_bounded_int(arg.substr(std::strlen("--duty-percent=")), "--duty-percent", 1, 100);
        } else if (arg == "--verify-cpp-int") {
            gpu_throttle_config().verify_cpp_int = true;
        } else if (arg.rfind("--verify-cpp-int=", 0) == 0) {
            throw std::runtime_error("--verify-cpp-int is a flag and does not take a value");
        } else if (arg == "--tuning-cache-dir") {
            gpu_throttle_config().tuning_cache_dir =
                require_next(i, "--tuning-cache-dir");
            if (gpu_throttle_config().tuning_cache_dir.empty()) {
                throw std::runtime_error("--tuning-cache-dir must not be empty");
            }
            gpu_throttle_config().tuning_cache_enabled = true;
        } else if (arg.rfind("--tuning-cache-dir=", 0) == 0) {
            gpu_throttle_config().tuning_cache_dir =
                arg.substr(std::strlen("--tuning-cache-dir="));
            if (gpu_throttle_config().tuning_cache_dir.empty()) {
                throw std::runtime_error("--tuning-cache-dir must not be empty");
            }
            gpu_throttle_config().tuning_cache_enabled = true;
        } else if (arg == "--tuning-cache-max-age-hours") {
            gpu_throttle_config().tuning_cache_max_age_hours =
                parse_bounded_int(require_next(i, "--tuning-cache-max-age-hours"),
                                  "--tuning-cache-max-age-hours", 1, 8760);
        } else if (arg.rfind("--tuning-cache-max-age-hours=", 0) == 0) {
            gpu_throttle_config().tuning_cache_max_age_hours =
                parse_bounded_int(
                    arg.substr(std::strlen("--tuning-cache-max-age-hours=")),
                    "--tuning-cache-max-age-hours", 1, 8760);
        } else if (arg == "--no-tuning-cache") {
            gpu_throttle_config().tuning_cache_enabled = false;
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

void throttle_after_gpu_iteration(std::chrono::steady_clock::time_point started) {
    const int duty = gpu_duty_percent();
    if (duty >= 100) return;

    const auto finished = std::chrono::steady_clock::now();
    const auto work_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started);
    if (work_ns.count() <= 0) return;

    // duty = work / (work + sleep), so sleep = work * (100-duty) / duty.
    const auto sleep_ns = std::chrono::nanoseconds(
        work_ns.count() * static_cast<int64_t>(100 - duty) / static_cast<int64_t>(duty));
    if (sleep_ns.count() > 0) std::this_thread::sleep_for(sleep_ns);
}

void print_gpu_throttle_config() {
    std::cout << "GPU tuning: ntt_blocks="
              << (ntt_blocks_are_forced() ? "forced:" : "auto(initial):")
              << ntt_block_limit()
              << ", window_bits="
              << (gpu_throttle_config().force_window_bits
                      ? "forced:" + std::to_string(gpu_throttle_config().window_bits)
                      : "auto")
              << ", duty_percent=" << gpu_duty_percent()
              << ", cpp_int_verify="
              << (cpp_int_verification_enabled() ? "enabled" : "disabled")
              << ", tuning_cache="
              << (tuning_cache_enabled()
                      ? tuning_cache_dir() + ":" +
                            std::to_string(tuning_cache_max_age_hours()) + "h"
                      : "disabled")
              << "\n";
}

uint64_t tuning_cache_hash(const std::string& text) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char value : text) {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string tuning_cache_hash_text(const std::string& text) {
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0')
           << tuning_cache_hash(text);
    return output.str();
}

std::filesystem::path tuning_cache_file(const std::string& key) {
    std::ostringstream name;
    name << std::hex << std::setw(16) << std::setfill('0')
         << tuning_cache_hash(key) << ".txt";
    return std::filesystem::path(tuning_cache_dir()) / name.str();
}

struct CachedTuning {
    int blocks = 0;
    double age_hours = 0.0;
};

CachedTuning load_cached_tuning(const std::string& key) {
    CachedTuning result;
    if (!tuning_cache_enabled()) return result;
    try {
        const std::filesystem::path path = tuning_cache_file(key);
        std::error_code size_error;
        const uintmax_t size = std::filesystem::file_size(path, size_error);
        if (size_error || size == 0 || size > 16384) return result;
        std::ifstream input(path, std::ios::binary);
        std::string magic, timestamp_text, blocks_text, stored_key, checksum;
        if (!std::getline(input, magic) ||
            !std::getline(input, timestamp_text) ||
            !std::getline(input, blocks_text) ||
            !std::getline(input, stored_key) ||
            !std::getline(input, checksum) ||
            magic != "GSRPS_TUNING_CACHE_V1" || stored_key != key) {
            return result;
        }
        std::string extra;
        if (std::getline(input, extra) && !extra.empty()) return result;
        const std::string payload =
            timestamp_text + "\n" + blocks_text + "\n" + stored_key + "\n";
        if (checksum != tuning_cache_hash_text(payload)) return result;
        size_t parsed = 0;
        const long long timestamp = std::stoll(timestamp_text, &parsed, 10);
        if (parsed != timestamp_text.size()) return result;
        parsed = 0;
        const long blocks = std::stol(blocks_text, &parsed, 10);
        if (parsed != blocks_text.size() || blocks < 1 || blocks > kMaxNttBlocks) {
            return result;
        }
        const std::time_t now = std::time(nullptr);
        if (now < 0 || timestamp < 0 || timestamp > static_cast<long long>(now)) {
            return result;
        }
        const double age_hours =
            static_cast<double>(static_cast<long long>(now) - timestamp) / 3600.0;
        if (age_hours >= tuning_cache_max_age_hours()) return result;
        result.blocks = static_cast<int>(blocks);
        result.age_hours = age_hours;
    } catch (...) {
        return CachedTuning{};
    }
    return result;
}

bool save_cached_tuning(const std::string& key, int blocks) {
    if (!tuning_cache_enabled()) return false;
    try {
        const std::filesystem::path directory(tuning_cache_dir());
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) return false;
        const std::filesystem::path path = tuning_cache_file(key);
        const uint64_t nonce = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
            static_cast<uint64_t>(std::hash<std::thread::id>{}(
                std::this_thread::get_id())) ^
            (static_cast<uint64_t>(std::random_device{}()) << 32) ^
            static_cast<uint64_t>(std::random_device{}());
        std::ostringstream suffix;
        suffix << ".tmp." << std::hex << nonce;
        const std::filesystem::path temporary(path.string() + suffix.str());
        const auto remove_temporary = [&]() {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
        };
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                remove_temporary();
                return false;
            }
            const std::string timestamp =
                std::to_string(static_cast<long long>(std::time(nullptr)));
            const std::string blocks_text = std::to_string(blocks);
            const std::string payload =
                timestamp + "\n" + blocks_text + "\n" + key + "\n";
            output << "GSRPS_TUNING_CACHE_V1\n"
                   << payload
                   << tuning_cache_hash_text(payload) << "\n";
            output.flush();
            if (!output) {
                remove_temporary();
                return false;
            }
        }
#ifdef _WIN32
        if (!MoveFileExW(
                temporary.wstring().c_str(), path.wstring().c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            remove_temporary();
            return false;
        }
#else
        std::filesystem::rename(temporary, path, error);
        if (error) {
            remove_temporary();
            return false;
        }
#endif
        return true;
    } catch (...) {
        return false;
    }
}

uint64_t parse_u64(const char* s) {
    char* end = nullptr;
    const uint64_t v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') throw std::runtime_error("invalid uint64 argument");
    return v;
}

struct GsrpsExpression {
    uint64_t k;
    uint64_t b;
    uint64_t n;
    int c;
};

GsrpsExpression parse_gsrps_expression(std::string expression) {
    expression.erase(std::remove_if(expression.begin(), expression.end(),
                                    [](unsigned char ch) { return std::isspace(ch) != 0; }),
                     expression.end());
    if (expression.size() < 5) throw std::runtime_error("invalid GSRPS expression");
    int c = 0;
    if (expression.size() >= 2 && expression.compare(expression.size() - 2, 2, "+1") == 0) c = 1;
    if (expression.size() >= 2 && expression.compare(expression.size() - 2, 2, "-1") == 0) c = -1;
    if (c == 0) throw std::runtime_error("expression must end in +1 or -1");
    expression.resize(expression.size() - 2);
    const size_t caret = expression.find('^');
    if (caret == std::string::npos || expression.find('^', caret + 1) != std::string::npos) {
        throw std::runtime_error("expression must have the form k*b^n+/-1");
    }
    const std::string left = expression.substr(0, caret);
    const std::string exponent = expression.substr(caret + 1);
    const size_t star = left.find('*');
    const std::string k_text = (star == std::string::npos) ? "1" : left.substr(0, star);
    const std::string b_text = (star == std::string::npos) ? left : left.substr(star + 1);
    if (k_text.empty() || b_text.empty() || exponent.empty() ||
        (star != std::string::npos && left.find('*', star + 1) != std::string::npos)) {
        throw std::runtime_error("expression must have the form k*b^n+/-1");
    }
    return GsrpsExpression{parse_u64(k_text.c_str()), parse_u64(b_text.c_str()),
                           parse_u64(exponent.c_str()), c};
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

uint32_t pow_mod_host(uint32_t a, uint64_t e, uint32_t p) {
    uint64_t r = 1, x = a;
    while (e != 0) {
        if ((e & 1) != 0) r = (r * x) % p;
        x = (x * x) % p;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

__device__ __forceinline__ uint32_t mul_mod_mont_2prime(uint32_t a, uint32_t b, uint32_t p) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t nprime0 = 998244351u;   // -p0^-1 mod 2^32
    constexpr uint32_t nprime1 = 1004535807u;  // -p1^-1 mod 2^32
    const uint32_t nprime = (p == p0) ? nprime0 : nprime1;
    const uint32_t lo = a * b;
    const uint32_t hi = __umulhi(a, b);
    const uint32_t m = lo * nprime;
    const uint32_t mp_lo = m * p;
    const uint32_t mp_hi = __umulhi(m, p);
    const uint32_t sum_lo = lo + mp_lo;
    uint32_t u = hi + mp_hi + static_cast<uint32_t>(sum_lo < lo);
    if (u >= p) u -= p;
    return u;
}

__device__ __forceinline__ void ntt_stage_one_apply_mont(uint32_t* a, const uint32_t* roots,
                                                          int len, int idx, uint32_t p) {
    const int half = len >> 1;
    const int block = idx / half;
    const int j = idx - block * half;
    const int pos = block * len + j;
    const uint32_t u = a[pos];
    const uint32_t v = mul_mod_mont_2prime(a[pos + half], roots[j], p);
    uint32_t x = u + v;
    if (x >= p) x -= p;
    const uint32_t y = (u >= v) ? (u - v) : (u + p - v);
    a[pos] = x;
    a[pos + half] = y;
}

__device__ __forceinline__ void ntt_stage2_apply_mont(uint32_t* a,
                                                       const uint32_t* roots1,
                                                       const uint32_t* roots2,
                                                       int len, int idx, uint32_t p) {
    const int quarter = len >> 2;
    const int block = idx / quarter;
    const int j = idx - block * quarter;
    const int pos = block * len + j;
    const uint32_t w1 = roots1[j];
    const uint32_t a0 = a[pos];
    const uint32_t a1 = mul_mod_mont_2prime(a[pos + quarter], w1, p);
    const uint32_t a2 = a[pos + 2 * quarter];
    const uint32_t a3 = mul_mod_mont_2prime(a[pos + 3 * quarter], w1, p);
    uint32_t b0 = a0 + a1;
    if (b0 >= p) b0 -= p;
    const uint32_t b1 = (a0 >= a1) ? (a0 - a1) : (a0 + p - a1);
    uint32_t b2 = a2 + a3;
    if (b2 >= p) b2 -= p;
    const uint32_t b3 = (a2 >= a3) ? (a2 - a3) : (a2 + p - a3);
    const uint32_t t0 = mul_mod_mont_2prime(b2, roots2[j], p);
    const uint32_t t1 = mul_mod_mont_2prime(b3, roots2[j + quarter], p);
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

__device__ __forceinline__ void ntt_stage_one_dif_apply_mont(uint32_t* a, const uint32_t* roots,
                                                              int len, int idx, uint32_t p) {
    const int half = len >> 1;
    const int block = idx / half;
    const int j = idx - block * half;
    const int pos = block * len + j;
    const uint32_t u = a[pos];
    const uint32_t v = a[pos + half];
    uint32_t sum = u + v;
    if (sum >= p) sum -= p;
    const uint32_t diff = (u >= v) ? (u - v) : (u + p - v);
    a[pos] = sum;
    a[pos + half] = mul_mod_mont_2prime(diff, roots[j], p);
}

// Two descending DIF stages: len followed by len/2. The output remains in the
// same array and a complete forward transform ends in bit-reversed order.
__device__ __forceinline__ void ntt_stage2_dif_apply_mont(uint32_t* a,
                                                           const uint32_t* roots_big,
                                                           const uint32_t* roots_small,
                                                           int len, int idx, uint32_t p) {
    const int quarter = len >> 2;
    const int block = idx / quarter;
    const int j = idx - block * quarter;
    const int pos = block * len + j;
    const uint32_t x0 = a[pos];
    const uint32_t x1 = a[pos + quarter];
    const uint32_t x2 = a[pos + 2 * quarter];
    const uint32_t x3 = a[pos + 3 * quarter];

    uint32_t a0 = x0 + x2;
    if (a0 >= p) a0 -= p;
    uint32_t a1 = x1 + x3;
    if (a1 >= p) a1 -= p;
    const uint32_t d0 = (x0 >= x2) ? (x0 - x2) : (x0 + p - x2);
    const uint32_t d1 = (x1 >= x3) ? (x1 - x3) : (x1 + p - x3);
    const uint32_t a2 = mul_mod_mont_2prime(d0, roots_big[j], p);
    const uint32_t a3 = mul_mod_mont_2prime(d1, roots_big[j + quarter], p);
    const uint32_t ws = roots_small[j];

    uint32_t y0 = a0 + a1;
    if (y0 >= p) y0 -= p;
    const uint32_t d2 = (a0 >= a1) ? (a0 - a1) : (a0 + p - a1);
    const uint32_t y1 = mul_mod_mont_2prime(d2, ws, p);
    uint32_t y2 = a2 + a3;
    if (y2 >= p) y2 -= p;
    const uint32_t d3 = (a2 >= a3) ? (a2 - a3) : (a2 + p - a3);
    const uint32_t y3 = mul_mod_mont_2prime(d3, ws, p);
    a[pos] = y0;
    a[pos + quarter] = y1;
    a[pos + 2 * quarter] = y2;
    a[pos + 3 * quarter] = y3;
}

__device__ __forceinline__ uint32_t add_mod_2prime(uint32_t a, uint32_t b, uint32_t p) {
    uint32_t x = a + b;
    if (x >= p) x -= p;
    return x;
}

__device__ __forceinline__ uint32_t sub_mod_2prime(uint32_t a, uint32_t b, uint32_t p) {
    return (a >= b) ? (a - b) : (a + p - b);
}

__device__ __forceinline__ void ntt_stage3_dit_apply_mont(uint32_t* a,
                                                           const uint32_t* roots_small,
                                                           const uint32_t* roots_mid,
                                                           const uint32_t* roots_big,
                                                           int len, int idx, uint32_t p) {
    const int octant = len >> 3;
    const int block = idx / octant;
    const int j = idx - block * octant;
    const int pos = block * len + j;
    uint32_t x[8], y[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) x[t] = a[pos + t * octant];

    const uint32_t ws = roots_small[j];
#pragma unroll
    for (int g = 0; g < 8; g += 2) {
        const uint32_t v = mul_mod_mont_2prime(x[g + 1], ws, p);
        y[g] = add_mod_2prime(x[g], v, p);
        y[g + 1] = sub_mod_2prime(x[g], v, p);
    }
#pragma unroll
    for (int g = 0; g < 8; g += 4) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t v = mul_mod_mont_2prime(y[g + 2 + r], roots_mid[j + r * octant], p);
            x[g + r] = add_mod_2prime(y[g + r], v, p);
            x[g + 2 + r] = sub_mod_2prime(y[g + r], v, p);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const uint32_t v = mul_mod_mont_2prime(x[4 + r], roots_big[j + r * octant], p);
        y[r] = add_mod_2prime(x[r], v, p);
        y[4 + r] = sub_mod_2prime(x[r], v, p);
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) a[pos + t * octant] = y[t];
}

__device__ __forceinline__ void ntt_stage3_dif_apply_mont(uint32_t* a,
                                                           const uint32_t* roots_big,
                                                           const uint32_t* roots_mid,
                                                           const uint32_t* roots_small,
                                                           int len, int idx, uint32_t p) {
    const int octant = len >> 3;
    const int block = idx / octant;
    const int j = idx - block * octant;
    const int pos = block * len + j;
    uint32_t x[8], y[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) x[t] = a[pos + t * octant];

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        y[r] = add_mod_2prime(x[r], x[4 + r], p);
        const uint32_t d = sub_mod_2prime(x[r], x[4 + r], p);
        y[4 + r] = mul_mod_mont_2prime(d, roots_big[j + r * octant], p);
    }
#pragma unroll
    for (int g = 0; g < 8; g += 4) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            x[g + r] = add_mod_2prime(y[g + r], y[g + 2 + r], p);
            const uint32_t d = sub_mod_2prime(y[g + r], y[g + 2 + r], p);
            x[g + 2 + r] = mul_mod_mont_2prime(d, roots_mid[j + r * octant], p);
        }
    }
    const uint32_t ws = roots_small[j];
#pragma unroll
    for (int g = 0; g < 8; g += 2) {
        y[g] = add_mod_2prime(x[g], x[g + 1], p);
        const uint32_t d = sub_mod_2prime(x[g], x[g + 1], p);
        y[g + 1] = mul_mod_mont_2prime(d, ws, p);
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) a[pos + t * octant] = y[t];
}

__device__ __forceinline__ void ntt_stage4_dit_apply_mont(
    uint32_t* a, const uint32_t* roots0, const uint32_t* roots1,
    const uint32_t* roots2, const uint32_t* roots3,
    int len, int idx, uint32_t p) {
    const int unit = len >> 4;
    const int block = idx / unit;
    const int j = idx - block * unit;
    const int pos = block * len + j;
    uint32_t x[16], y[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) x[t] = a[pos + t * unit];

#pragma unroll
    for (int g = 0; g < 16; g += 2) {
        const uint32_t v = mul_mod_mont_2prime(x[g + 1], roots0[j], p);
        y[g] = add_mod_2prime(x[g], v, p);
        y[g + 1] = sub_mod_2prime(x[g], v, p);
    }
#pragma unroll
    for (int g = 0; g < 16; g += 4) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t v = mul_mod_mont_2prime(y[g + 2 + r], roots1[j + r * unit], p);
            x[g + r] = add_mod_2prime(y[g + r], v, p);
            x[g + 2 + r] = sub_mod_2prime(y[g + r], v, p);
        }
    }
#pragma unroll
    for (int g = 0; g < 16; g += 8) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const uint32_t v = mul_mod_mont_2prime(x[g + 4 + r], roots2[j + r * unit], p);
            y[g + r] = add_mod_2prime(x[g + r], v, p);
            y[g + 4 + r] = sub_mod_2prime(x[g + r], v, p);
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const uint32_t v = mul_mod_mont_2prime(y[8 + r], roots3[j + r * unit], p);
        x[r] = add_mod_2prime(y[r], v, p);
        x[8 + r] = sub_mod_2prime(y[r], v, p);
    }
#pragma unroll
    for (int t = 0; t < 16; ++t) a[pos + t * unit] = x[t];
}

__device__ __forceinline__ void ntt_stage4_dif_apply_mont(
    uint32_t* a, const uint32_t* roots3, const uint32_t* roots2,
    const uint32_t* roots1, const uint32_t* roots0,
    int len, int idx, uint32_t p) {
    const int unit = len >> 4;
    const int block = idx / unit;
    const int j = idx - block * unit;
    const int pos = block * len + j;
    uint32_t x[16], y[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) x[t] = a[pos + t * unit];

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        y[r] = add_mod_2prime(x[r], x[8 + r], p);
        y[8 + r] = mul_mod_mont_2prime(sub_mod_2prime(x[r], x[8 + r], p),
                                       roots3[j + r * unit], p);
    }
#pragma unroll
    for (int g = 0; g < 16; g += 8) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            x[g + r] = add_mod_2prime(y[g + r], y[g + 4 + r], p);
            x[g + 4 + r] = mul_mod_mont_2prime(
                sub_mod_2prime(y[g + r], y[g + 4 + r], p), roots2[j + r * unit], p);
        }
    }
#pragma unroll
    for (int g = 0; g < 16; g += 4) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            y[g + r] = add_mod_2prime(x[g + r], x[g + 2 + r], p);
            y[g + 2 + r] = mul_mod_mont_2prime(
                sub_mod_2prime(x[g + r], x[g + 2 + r], p), roots1[j + r * unit], p);
        }
    }
#pragma unroll
    for (int g = 0; g < 16; g += 2) {
        x[g] = add_mod_2prime(y[g], y[g + 1], p);
        x[g + 1] = mul_mod_mont_2prime(sub_mod_2prime(y[g], y[g + 1], p), roots0[j], p);
    }
#pragma unroll
    for (int t = 0; t < 16; ++t) a[pos + t * unit] = x[t];
}

__global__ void ntt_stage2_2_mont_kernel(uint32_t* r0, uint32_t* r1,
                                         const uint32_t* roots10, const uint32_t* roots11,
                                         const uint32_t* roots20, const uint32_t* roots21,
                                         int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage2_apply_mont(r0, roots10, roots20, len, idx, p0);
        ntt_stage2_apply_mont(r1, roots11, roots21, len, idx, p1);
    }
}

__global__ void ntt_stage2_2_mont_scaled_kernel(uint32_t* r0, uint32_t* r1,
                                                const uint32_t* roots_small0, const uint32_t* roots_small1,
                                                const uint32_t* roots_big0, const uint32_t* roots_big1,
                                                int len, int groups, uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage2_apply_mont(r0, roots_small0, roots_big0, len, idx, p0);
        ntt_stage2_apply_mont(r1, roots_small1, roots_big1, len, idx, p1);
        const int quarter = len >> 2;
        const int block = idx / quarter;
        const int j = idx - block * quarter;
        const int pos = block * len + j;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int out = pos + t * quarter;
            r0[out] = mul_mod_mont_2prime(r0[out], inv0, p0);
            r1[out] = mul_mod_mont_2prime(r1[out], inv1, p1);
        }
    }
}

__global__ void ntt_stage_2_mont_kernel(uint32_t* r0, uint32_t* r1,
                                        const uint32_t* roots0, const uint32_t* roots1,
                                        int len, int butterflies) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        ntt_stage_one_apply_mont(r0, roots0, len, idx, p0);
        ntt_stage_one_apply_mont(r1, roots1, len, idx, p1);
    }
}
__global__ void ntt_stage_2_mont_scaled_kernel(uint32_t* r0, uint32_t* r1,
                                               const uint32_t* roots0, const uint32_t* roots1,
                                               int len, int butterflies,
                                               uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        ntt_stage_one_apply_mont(r0, roots0, len, idx, p0);
        ntt_stage_one_apply_mont(r1, roots1, len, idx, p1);
        const int half = len >> 1;
        const int block = idx / half;
        const int j = idx - block * half;
        const int pos = block * len + j;
        r0[pos] = mul_mod_mont_2prime(r0[pos], inv0, p0);
        r0[pos + half] = mul_mod_mont_2prime(r0[pos + half], inv0, p0);
        r1[pos] = mul_mod_mont_2prime(r1[pos], inv1, p1);
        r1[pos + half] = mul_mod_mont_2prime(r1[pos + half], inv1, p1);
    }
}

__global__ void ntt_stage_2_dif_mont_kernel(uint32_t* r0, uint32_t* r1,
                                            const uint32_t* roots0, const uint32_t* roots1,
                                            int len, int butterflies) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < butterflies; idx += total * gridDim.y) {
        ntt_stage_one_dif_apply_mont(r0, roots0, len, idx, p0);
        ntt_stage_one_dif_apply_mont(r1, roots1, len, idx, p1);
    }
}

__global__ void ntt_stage2_2_dif_mont_kernel(uint32_t* r0, uint32_t* r1,
                                             const uint32_t* roots_big0, const uint32_t* roots_big1,
                                             const uint32_t* roots_small0, const uint32_t* roots_small1,
                                             int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage2_dif_apply_mont(r0, roots_big0, roots_small0, len, idx, p0);
        ntt_stage2_dif_apply_mont(r1, roots_big1, roots_small1, len, idx, p1);
    }
}

__global__ void ntt_stage3_2_dit_mont_kernel(uint32_t* r0, uint32_t* r1,
                                             const uint32_t* roots_small0, const uint32_t* roots_small1,
                                             const uint32_t* roots_mid0, const uint32_t* roots_mid1,
                                             const uint32_t* roots_big0, const uint32_t* roots_big1,
                                             int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage3_dit_apply_mont(r0, roots_small0, roots_mid0, roots_big0, len, idx, p0);
        ntt_stage3_dit_apply_mont(r1, roots_small1, roots_mid1, roots_big1, len, idx, p1);
    }
}

__global__ void ntt_stage3_2_dit_mont_scaled_kernel(uint32_t* r0, uint32_t* r1,
                                                    const uint32_t* roots_small0, const uint32_t* roots_small1,
                                                    const uint32_t* roots_mid0, const uint32_t* roots_mid1,
                                                    const uint32_t* roots_big0, const uint32_t* roots_big1,
                                                    int len, int groups,
                                                    uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage3_dit_apply_mont(r0, roots_small0, roots_mid0, roots_big0, len, idx, p0);
        ntt_stage3_dit_apply_mont(r1, roots_small1, roots_mid1, roots_big1, len, idx, p1);
        const int octant = len >> 3;
        const int block = idx / octant;
        const int j = idx - block * octant;
        const int pos = block * len + j;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int out = pos + t * octant;
            r0[out] = mul_mod_mont_2prime(r0[out], inv0, p0);
            r1[out] = mul_mod_mont_2prime(r1[out], inv1, p1);
        }
    }
}

__global__ void ntt_stage3_2_dif_mont_kernel(uint32_t* r0, uint32_t* r1,
                                             const uint32_t* roots_big0, const uint32_t* roots_big1,
                                             const uint32_t* roots_mid0, const uint32_t* roots_mid1,
                                             const uint32_t* roots_small0, const uint32_t* roots_small1,
                                             int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage3_dif_apply_mont(r0, roots_big0, roots_mid0, roots_small0, len, idx, p0);
        ntt_stage3_dif_apply_mont(r1, roots_big1, roots_mid1, roots_small1, len, idx, p1);
    }
}

__global__ void ntt_stage4_2_dit_mont_kernel(
    uint32_t* r0, uint32_t* r1,
    const uint32_t* roots00, const uint32_t* roots01,
    const uint32_t* roots10, const uint32_t* roots11,
    const uint32_t* roots20, const uint32_t* roots21,
    const uint32_t* roots30, const uint32_t* roots31,
    int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage4_dit_apply_mont(r0, roots00, roots10, roots20, roots30, len, idx, p0);
        ntt_stage4_dit_apply_mont(r1, roots01, roots11, roots21, roots31, len, idx, p1);
    }
}

__global__ void ntt_stage4_2_dit_mont_scaled_kernel(
    uint32_t* r0, uint32_t* r1,
    const uint32_t* roots00, const uint32_t* roots01,
    const uint32_t* roots10, const uint32_t* roots11,
    const uint32_t* roots20, const uint32_t* roots21,
    const uint32_t* roots30, const uint32_t* roots31,
    int len, int groups, uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage4_dit_apply_mont(r0, roots00, roots10, roots20, roots30, len, idx, p0);
        ntt_stage4_dit_apply_mont(r1, roots01, roots11, roots21, roots31, len, idx, p1);
        const int unit = len >> 4;
        const int block = idx / unit;
        const int j = idx - block * unit;
        const int pos = block * len + j;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int out = pos + t * unit;
            r0[out] = mul_mod_mont_2prime(r0[out], inv0, p0);
            r1[out] = mul_mod_mont_2prime(r1[out], inv1, p1);
        }
    }
}

__global__ void ntt_stage4_2_dif_mont_kernel(
    uint32_t* r0, uint32_t* r1,
    const uint32_t* roots30, const uint32_t* roots31,
    const uint32_t* roots20, const uint32_t* roots21,
    const uint32_t* roots10, const uint32_t* roots11,
    const uint32_t* roots00, const uint32_t* roots01,
    int len, int groups) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    for (int idx = id + blockIdx.y * total; idx < groups; idx += total * gridDim.y) {
        ntt_stage4_dif_apply_mont(r0, roots30, roots20, roots10, roots00, len, idx, p0);
        ntt_stage4_dif_apply_mont(r1, roots31, roots21, roots11, roots01, len, idx, p1);
    }
}

#ifndef GSRPS_SHARED_NTT_TILE_LOG
#define GSRPS_SHARED_NTT_TILE_LOG 11
#endif
#ifndef GSRPS_SHARED_RADIX8
#define GSRPS_SHARED_RADIX8 1
#endif
#ifndef GSRPS_SHARED_NTT_THREADS
#define GSRPS_SHARED_NTT_THREADS 256
#endif
#ifndef GSRPS_GLOBAL_RADIX4
#define GSRPS_GLOBAL_RADIX4 1
#endif
#ifndef GSRPS_GLOBAL_RADIX8
#define GSRPS_GLOBAL_RADIX8 1
#endif
#ifndef GSRPS_GLOBAL_RADIX16
#define GSRPS_GLOBAL_RADIX16 1
#endif
constexpr int kSharedNttTileLog = GSRPS_SHARED_NTT_TILE_LOG;
constexpr int kSharedNttTile = 1 << kSharedNttTileLog;
constexpr int kSharedNttThreads = GSRPS_SHARED_NTT_THREADS;

template<bool Square>
__global__ void ntt_tile_product_dit_mont_kernel(uint32_t* r0, uint32_t* r1,
                                                 const uint32_t* multiplier0,
                                                 const uint32_t* multiplier1,
                                                 const uint32_t* tw0, const uint32_t* tw1,
                                                 int total_tiles) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    __shared__ uint32_t s0[kSharedNttTile];
    __shared__ uint32_t s1[kSharedNttTile];
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int base = tile * kSharedNttTile;
        for (int i = threadIdx.x; i < kSharedNttTile; i += blockDim.x) {
            const uint32_t x0 = r0[base + i];
            const uint32_t x1 = r1[base + i];
            const uint32_t y0 = Square ? x0 : multiplier0[base + i];
            const uint32_t y1 = Square ? x1 : multiplier1[base + i];
            s0[i] = mul_mod_mont_2prime(x0, y0, p0);
            s1[i] = mul_mod_mont_2prime(x1, y1, p1);
        }
        __syncthreads();
        int stage = 1;
#if GSRPS_SHARED_RADIX8
        for (; stage + 2 <= kSharedNttTileLog; stage += 3) {
            const int len = 1 << (stage + 2);
            const int root1_offset = (1 << (stage - 1)) - 1;
            const int root2_offset = (1 << stage) - 1;
            const int root3_offset = (1 << (stage + 1)) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 8; idx += blockDim.x) {
                ntt_stage3_dit_apply_mont(s0, tw0 + root1_offset, tw0 + root2_offset,
                                           tw0 + root3_offset, len, idx, p0);
                ntt_stage3_dit_apply_mont(s1, tw1 + root1_offset, tw1 + root2_offset,
                                           tw1 + root3_offset, len, idx, p1);
            }
            __syncthreads();
        }
#endif
        for (; stage + 1 <= kSharedNttTileLog; stage += 2) {
            const int len = 1 << (stage + 1);
            const int root1_offset = (1 << (stage - 1)) - 1;
            const int root2_offset = (1 << stage) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 4; idx += blockDim.x) {
                ntt_stage2_apply_mont(s0, tw0 + root1_offset, tw0 + root2_offset, len, idx, p0);
                ntt_stage2_apply_mont(s1, tw1 + root1_offset, tw1 + root2_offset, len, idx, p1);
            }
            __syncthreads();
        }
        if (stage <= kSharedNttTileLog) {
            const int len = 1 << stage;
            const int root_offset = (1 << (stage - 1)) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 2; idx += blockDim.x) {
                ntt_stage_one_apply_mont(s0, tw0 + root_offset, len, idx, p0);
                ntt_stage_one_apply_mont(s1, tw1 + root_offset, len, idx, p1);
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < kSharedNttTile; i += blockDim.x) {
            r0[base + i] = s0[i];
            r1[base + i] = s1[i];
        }
        __syncthreads();
    }
}

__global__ void ntt_tile_dif_mont_kernel(uint32_t* r0, uint32_t* r1,
                                         const uint32_t* tw0, const uint32_t* tw1,
                                         int total_tiles) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    __shared__ uint32_t s0[kSharedNttTile];
    __shared__ uint32_t s1[kSharedNttTile];
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int base = tile * kSharedNttTile;
        for (int i = threadIdx.x; i < kSharedNttTile; i += blockDim.x) {
            s0[i] = r0[base + i];
            s1[i] = r1[base + i];
        }
        __syncthreads();
        int stage = kSharedNttTileLog;
#if GSRPS_SHARED_RADIX8
        for (; stage - 2 >= 1; stage -= 3) {
            const int len = 1 << stage;
            const int root_big_offset = (1 << (stage - 1)) - 1;
            const int root_mid_offset = (1 << (stage - 2)) - 1;
            const int root_small_offset = (1 << (stage - 3)) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 8; idx += blockDim.x) {
                ntt_stage3_dif_apply_mont(s0, tw0 + root_big_offset, tw0 + root_mid_offset,
                                           tw0 + root_small_offset, len, idx, p0);
                ntt_stage3_dif_apply_mont(s1, tw1 + root_big_offset, tw1 + root_mid_offset,
                                           tw1 + root_small_offset, len, idx, p1);
            }
            __syncthreads();
        }
#endif
        for (; stage - 1 >= 1; stage -= 2) {
            const int len = 1 << stage;
            const int root_big_offset = (1 << (stage - 1)) - 1;
            const int root_small_offset = (1 << (stage - 2)) - 1;
            for (int idx = threadIdx.x; idx < kSharedNttTile / 4; idx += blockDim.x) {
                ntt_stage2_dif_apply_mont(s0, tw0 + root_big_offset, tw0 + root_small_offset, len, idx, p0);
                ntt_stage2_dif_apply_mont(s1, tw1 + root_big_offset, tw1 + root_small_offset, len, idx, p1);
            }
            __syncthreads();
        }
        if (stage == 1) {
            for (int idx = threadIdx.x; idx < kSharedNttTile / 2; idx += blockDim.x) {
                ntt_stage_one_dif_apply_mont(s0, tw0, 2, idx, p0);
                ntt_stage_one_dif_apply_mont(s1, tw1, 2, idx, p1);
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < kSharedNttTile; i += blockDim.x) {
            r0[base + i] = s0[i];
            r1[base + i] = s1[i];
        }
        __syncthreads();
    }
}

__global__ void pointwise_square2_mont_kernel(uint32_t* r0, uint32_t* r1, int len) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = mul_mod_mont_2prime(r0[i], r0[i], p0);
    r1[i] = mul_mod_mont_2prime(r1[i], r1[i], p1);
}

__global__ void pointwise_mul2_mont_kernel(uint32_t* r0, uint32_t* r1,
                                           const uint32_t* multiplier0,
                                           const uint32_t* multiplier1, int len) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = mul_mod_mont_2prime(r0[i], multiplier0[i], p0);
    r1[i] = mul_mod_mont_2prime(r1[i], multiplier1[i], p1);
}

// The input is Montgomery encoded. Multiplication by an ordinary inv_len both
// applies the inverse scale and leaves the result in the ordinary domain.
__global__ void scale2_from_mont_kernel(uint32_t* r0, uint32_t* r1, int len,
                                        uint32_t inv0, uint32_t inv1) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    r0[i] = mul_mod_mont_2prime(r0[i], inv0, p0);
    r1[i] = mul_mod_mont_2prime(r1[i], inv1, p1);
}

__device__ __forceinline__ uint64_t crt2_reconstruct_shoup(uint32_t a, uint32_t b) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t inv_p0_mod_p1 = 669690699u;
    constexpr uint32_t inv_p0_mod_p1_shoup = 2863312213u;
    const uint32_t diff = (b >= a) ? (b - a) : (b + p1 - a);
    const uint32_t q = __umulhi(diff, inv_p0_mod_p1_shoup);
    uint32_t t = diff * inv_p0_mod_p1 - q * p1;
    if (t >= p1) t -= p1;
    return static_cast<uint64_t>(a) + static_cast<uint64_t>(p0) * t;
}

__global__ void crt2_raw_kernel(const uint32_t* r0, const uint32_t* r1,
                                uint64_t* coeff, int coeff_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < coeff_count) coeff[i] = crt2_reconstruct_shoup(r0[i], r1[i]);
}
#ifndef GSRPS_EXACT_CARRY_SEGMENT
#define GSRPS_EXACT_CARRY_SEGMENT 8
#endif
constexpr int kExactCarrySegment = GSRPS_EXACT_CARRY_SEGMENT;
static_assert(kExactCarrySegment >= 4 && kExactCarrySegment <= 256,
              "GSRPS_EXACT_CARRY_SEGMENT must be in [4, 256]");

// A finite dependency halo is not an exact carry algorithm: a one-unit carry
// can propagate through an arbitrarily long run of radix-1 digits.  The exact
// path below normalizes short independent segments with zero incoming carry,
// records each segment's transfer function, resolves the segment carries in
// order, and then applies them in parallel.
//
// If zero-incoming normalization gives
//
//     V = base_carry * R^B + low,
//
// then adding an incoming carry x changes the outgoing carry only when
// x >= R^B-low.  R^B-low is stored as a saturated uint64 threshold.
__device__ __forceinline__ void saturated_weighted_add(uint64_t term, uint64_t place,
                                                        uint64_t& sum) {
    if (sum == UINT64_MAX || term == 0) return;
    if (place == UINT64_MAX || term > (UINT64_MAX - sum) / place) {
        sum = UINT64_MAX;
    } else {
        sum += term * place;
    }
}

__device__ __forceinline__ uint64_t saturated_mul_radix(uint64_t place, uint64_t radix) {
    if (place == UINT64_MAX || place > UINT64_MAX / radix) return UINT64_MAX;
    return place * radix;
}

__device__ __forceinline__ uint64_t divrem_radix_u64(uint64_t value, uint64_t radix,
                                                      uint64_t reciprocal, uint64_t& digit) {
    uint64_t q = __umul64hi(value, reciprocal);
    digit = value - q * radix;
    if (digit >= radix) {
        digit -= radix;
        ++q;
    }
    return q;
}

int radix_u64_exact_positions(uint64_t radix) {
    int positions = 1;  // radix^0
    uint64_t place = 1;
    while (place <= UINT64_MAX / radix) {
        place *= radix;
        ++positions;
    }
    return positions;
}

struct UnsignedCarryMap {
    uint64_t base;
    uint64_t threshold;
};

struct UnsignedCarryCompose {
    __host__ __device__ __forceinline__ UnsignedCarryMap operator()(
        const UnsignedCarryMap& left, const UnsignedCarryMap& right) const {
        // right o left, where F(x)=base+[x>=threshold].
        if (right.threshold <= left.base) {
            return UnsignedCarryMap{right.base + 1, UINT64_MAX};
        }
        if (right.threshold > left.base + 1) {
            return UnsignedCarryMap{right.base, UINT64_MAX};
        }
        return UnsignedCarryMap{right.base, left.threshold};
    }
};

__global__ void crt2_unsigned_segment_prepare_kernel(
    const uint32_t* r0, const uint32_t* r1, int64_t* digits,
    UnsignedCarryMap* maps,
    int coeff_count, int len, uint64_t radix, uint64_t radix_reciprocal,
    int exact_positions) {
    const int segment = blockIdx.x * blockDim.x + threadIdx.x;
    const int begin = segment * kExactCarrySegment;
    if (begin >= len) return;
    const int end = min(len, begin + kExactCarrySegment);

    uint64_t carry = 0;
    bool high_is_max = true;
    for (int i = begin; i < end; ++i) {
        const uint64_t coeff = (i < coeff_count) ? crt2_reconstruct_shoup(r0[i], r1[i]) : 0;
        uint64_t digit = 0;
        carry = divrem_radix_u64(coeff + carry, radix, radix_reciprocal, digit);
        digits[i] = static_cast<int64_t>(digit);
        if (i - begin >= exact_positions && digit != radix - 1) {
            high_is_max = false;
        }
    }
    uint64_t threshold = high_is_max ? 1 : UINT64_MAX;
    if (high_is_max) {
        uint64_t place = 1;
        const int exact_end = min(end, begin + exact_positions);
        for (int i = begin; i < exact_end; ++i) {
            const uint64_t digit = static_cast<uint64_t>(digits[i]);
            saturated_weighted_add(
                radix - 1 - digit, place, threshold);
            place = saturated_mul_radix(place, radix);
        }
    }
    maps[segment] = UnsignedCarryMap{carry, threshold};
}

__global__ void unsigned_segment_apply_kernel(
    int64_t* digits, const UnsignedCarryMap* inclusive_prefix,
    int len, uint64_t radix, uint64_t radix_reciprocal) {
    const int segment = blockIdx.x * blockDim.x + threadIdx.x;
    const int begin = segment * kExactCarrySegment;
    if (begin >= len) return;
    const int end = min(len, begin + kExactCarrySegment);
    const uint64_t carry_in =
        (segment == 0) ? 0 : inclusive_prefix[segment - 1].base;
    uint64_t carry = carry_in;
    if (carry == 0) return;
    for (int i = begin; i < end; ++i) {
        uint64_t digit = 0;
        carry = divrem_radix_u64(static_cast<uint64_t>(digits[i]) + carry,
                                radix, radix_reciprocal, digit);
        digits[i] = static_cast<int64_t>(digit);
        if (carry == 0) break;
    }
}

struct AffineModMap {
    uint32_t a;
    uint32_t b;
};

__device__ __forceinline__ uint32_t mod_runtime_barrett(uint64_t value, uint32_t divisor, uint64_t reciprocal) {
    if (divisor == 1) return 0;
    uint64_t q = __umul64hi(value, reciprocal);
    uint64_t r = value - q * divisor;
    if (r >= divisor) r -= divisor;
    return static_cast<uint32_t>(r);
}

struct AffineCompose {
    uint32_t divisor;
    uint64_t reciprocal;

    __device__ __forceinline__ AffineModMap operator()(const AffineModMap& left,
                                                       const AffineModMap& right) const {
        // Scan order is most-significant to least-significant.  Combining a
        // prefix F with the next digit map G must return G o F.
        AffineModMap out;
        out.a = mod_runtime_barrett(static_cast<uint64_t>(right.a) * left.a, divisor, reciprocal);
        out.b = mod_runtime_barrett(static_cast<uint64_t>(right.a) * left.b + right.b, divisor, reciprocal);
        return out;
    }
};

__global__ void make_div_affine_maps_kernel(const int64_t* product_digits, int high_offset,
                                             int high_count, uint32_t radix_mod,
                                             uint32_t divisor, AffineModMap* maps) {
    const int scan_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (scan_index >= high_count) return;
    const int digit_index = high_count - 1 - scan_index;
    maps[scan_index] = AffineModMap{radix_mod,
        static_cast<uint32_t>(static_cast<uint64_t>(product_digits[high_offset + digit_index]) % divisor)};
}

__global__ void emit_div_from_prefix_kernel(const int64_t* product_digits, int high_offset,
                                             int high_count, uint64_t radix,
                                             uint32_t divisor, uint64_t reciprocal,
                                             const AffineModMap* prefix,
                                             uint64_t* quotient, uint64_t* final_remainder) {
    const int scan_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (scan_index >= high_count) return;
    const int digit_index = high_count - 1 - scan_index;
    const uint64_t incoming = prefix[scan_index].b;
    const uint64_t current = incoming * radix + static_cast<uint64_t>(product_digits[high_offset + digit_index]);
    uint64_t q = (divisor == 1) ? current : __umul64hi(current, reciprocal);
    uint64_t rem = current - q * divisor;
    if (rem >= divisor) { rem -= divisor; ++q; }
    quotient[digit_index] = q;
    if (digit_index == 0) *final_remainder = rem;
}

__global__ void make_div_sum_values_kernel(const int64_t* product_digits, int high_offset,
                                           int high_count, uint64_t* values) {
    const int scan_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (scan_index >= high_count) return;
    const int digit_index = high_count - 1 - scan_index;
    values[scan_index] = static_cast<uint64_t>(product_digits[high_offset + digit_index]);
}

__global__ void emit_div_from_sum_prefix_kernel(const int64_t* product_digits, int high_offset,
                                                 int high_count, uint64_t radix,
                                                 uint32_t divisor, uint64_t reciprocal,
                                                 const uint64_t* prefix_sum,
                                                 uint64_t* quotient, uint64_t* final_remainder) {
    const int scan_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (scan_index >= high_count) return;
    const int digit_index = high_count - 1 - scan_index;
    const uint64_t incoming = mod_runtime_barrett(prefix_sum[scan_index], divisor, reciprocal);
    const uint64_t current = incoming * radix + static_cast<uint64_t>(product_digits[high_offset + digit_index]);
    uint64_t q = (divisor == 1) ? current : __umul64hi(current, reciprocal);
    uint64_t rem = current - q * divisor;
    if (rem >= divisor) { rem -= divisor; ++q; }
    quotient[digit_index] = q;
    if (digit_index == 0) *final_remainder = rem;
}

// If divisor divides the digit radix, long division has no long dependency.
// For R = multiplier * divisor and H = sum(h_i R^i), the base-R quotient digit is
// floor(h_i/divisor) + multiplier * (h_{i+1} mod divisor).  Thus every quotient
// digit and the final remainder can be emitted independently without a scan.
__global__ void emit_div_radix_multiple_kernel(const int64_t* product_digits, int high_offset,
                                                int high_count, uint64_t radix_multiplier,
                                                uint32_t divisor, uint64_t reciprocal,
                                                uint64_t* quotient, uint64_t* final_remainder) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= high_count) return;
    const uint64_t digit = static_cast<uint64_t>(product_digits[high_offset + i]);
    uint64_t q = (divisor == 1) ? digit : __umul64hi(digit, reciprocal);
    uint64_t rem = digit - q * divisor;
    if (rem >= divisor) { rem -= divisor; ++q; }
    uint64_t incoming = 0;
    if (i + 1 < high_count) {
        const uint64_t next_digit = static_cast<uint64_t>(product_digits[high_offset + i + 1]);
        incoming = mod_runtime_barrett(next_digit, divisor, reciprocal);
    }
    quotient[i] = q + incoming * radix_multiplier;
    if (i == 0) *final_remainder = rem;
}

__device__ __forceinline__ int64_t floor_div_radix_recip(int64_t value, uint64_t radix,
                                                          uint64_t reciprocal, int64_t& digit) {
    const bool negative = value < 0;
    const uint64_t magnitude = negative
        ? static_cast<uint64_t>(-(value + 1)) + 1ull
        : static_cast<uint64_t>(value);
    uint64_t q = __umul64hi(magnitude, reciprocal);
    uint64_t r = magnitude - q * radix;
    if (r >= radix) { r -= radix; ++q; }
    if (!negative) {
        digit = static_cast<int64_t>(r);
        return static_cast<int64_t>(q);
    }
    if (r == 0) {
        digit = 0;
        return -static_cast<int64_t>(q);
    }
    digit = static_cast<int64_t>(radix - r);
    return -static_cast<int64_t>(q) - 1;
}

__device__ __forceinline__ int64_t build_folded_value(const int64_t* product_digits,
                                                       const uint64_t* quotient,
                                                       const uint64_t* remainder,
                                                       const int64_t* modulus_digits,
                                                       int i, int low_count, int quotient_count,
                                                       int active, int len, int c) {
    if (i < 0 || i >= len) return 0;
    int64_t value = 0;
    if (i < low_count) value += product_digits[i];
    if (i < quotient_count) {
        const int64_t qdigit = static_cast<int64_t>(quotient[i]);
        value += (c == 1) ? -qdigit : qdigit;
    }
    if (i == low_count) value += static_cast<int64_t>(*remainder);
    // For c=+1 the raw folded value lies in (-N,N); adding N makes it
    // non-negative.  For c=-1 it is already non-negative.
    if (c == 1 && i < active) value += modulus_digits[i];
    return value;
}

struct SignedCarryMap {
    int64_t low;
    int64_t threshold1;
    int64_t threshold2;
};

__host__ __device__ __forceinline__ int64_t evaluate_signed_carry_map(
    const SignedCarryMap& map, int64_t value) {
    return map.low + static_cast<int64_t>(value >= map.threshold1) +
           static_cast<int64_t>(value >= map.threshold2);
}

struct SignedCarryCompose {
    __host__ __device__ __forceinline__ SignedCarryMap operator()(
        const SignedCarryMap& left, const SignedCarryMap& right) const {
        // right o left.  left has at most three consecutive output values;
        // evaluate right on those values and inherit the corresponding
        // transition locations.
        const int64_t y0 = evaluate_signed_carry_map(right, left.low);
        const int64_t y1 = evaluate_signed_carry_map(right, left.low + 1);
        const int64_t y2 = evaluate_signed_carry_map(right, left.low + 2);
        SignedCarryMap out{y0, INT64_MAX, INT64_MAX};
        int count = 0;
        for (int64_t i = 0; i < y1 - y0 && count < 2; ++i) {
            if (count++ == 0) out.threshold1 = left.threshold1;
            else out.threshold2 = left.threshold1;
        }
        for (int64_t i = 0; i < y2 - y1 && count < 2; ++i) {
            if (count++ == 0) out.threshold1 = left.threshold2;
            else out.threshold2 = left.threshold2;
        }
        return out;
    }
};

__global__ void folded_signed_segment_prepare_kernel(
    const int64_t* product_digits, const uint64_t* quotient,
    const uint64_t* remainder, const int64_t* modulus_digits,
    int64_t* digits, SignedCarryMap* maps,
    int low_count, int quotient_count, int active, int len, int c,
    uint64_t radix, uint64_t radix_reciprocal, int exact_positions) {
    const int segment = blockIdx.x * blockDim.x + threadIdx.x;
    const int begin = segment * kExactCarrySegment;
    if (begin >= len) return;
    const int end = min(len, begin + kExactCarrySegment);

    int64_t carry = 0;
    bool high_is_max = true;
    bool high_is_zero = true;
    for (int i = begin; i < end; ++i) {
        const int64_t coeff = build_folded_value(
            product_digits, quotient, remainder, modulus_digits,
            i, low_count, quotient_count, active, len, c);
        int64_t digit = 0;
        carry = floor_div_radix_recip(coeff + carry, radix, radix_reciprocal, digit);
        digits[i] = digit;
        if (i - begin >= exact_positions) {
            const uint64_t unsigned_digit =
                static_cast<uint64_t>(digit);
            if (unsigned_digit != radix - 1) high_is_max = false;
            if (unsigned_digit != 0) high_is_zero = false;
        }
    }
    // For V = base*R^B + low and signed incoming x:
    //   x > 0  overflows when x >= R^B-low;
    //   x < 0 underflows when -x >= low+1.
    uint64_t positive = high_is_max ? 1 : UINT64_MAX;
    uint64_t negative = high_is_zero ? 1 : UINT64_MAX;
    if (high_is_max || high_is_zero) {
        uint64_t place = 1;
        const int exact_end = min(end, begin + exact_positions);
        for (int i = begin; i < exact_end; ++i) {
            const uint64_t digit = static_cast<uint64_t>(digits[i]);
            if (high_is_max) {
                saturated_weighted_add(
                    radix - 1 - digit, place, positive);
            }
            if (high_is_zero) {
                saturated_weighted_add(digit, place, negative);
            }
            place = saturated_mul_radix(place, radix);
        }
    }
    SignedCarryMap map{carry, INT64_MAX, INT64_MAX};
    int threshold_count = 0;
    if (negative <= (uint64_t(1) << 63)) {
        map.low = carry - 1;
        map.threshold1 =
            (negative == (uint64_t(1) << 63))
                ? INT64_MIN + 1
                : 1 - static_cast<int64_t>(negative);
        threshold_count = 1;
    }
    if (positive <= static_cast<uint64_t>(INT64_MAX)) {
        if (threshold_count++ == 0) {
            map.threshold1 = static_cast<int64_t>(positive);
        } else {
            map.threshold2 = static_cast<int64_t>(positive);
        }
    }
    maps[segment] = map;
}

__global__ void signed_segment_apply_kernel(
    int64_t* digits, const SignedCarryMap* inclusive_prefix,
    int len, uint64_t radix, uint64_t radix_reciprocal) {
    const int segment = blockIdx.x * blockDim.x + threadIdx.x;
    const int begin = segment * kExactCarrySegment;
    if (begin >= len) return;
    const int end = min(len, begin + kExactCarrySegment);
    const int64_t carry_in =
        (segment == 0)
            ? 0
            : evaluate_signed_carry_map(inclusive_prefix[segment - 1], 0);
    int64_t carry = carry_in;
    if (carry == 0) return;
    for (int i = begin; i < end; ++i) {
        int64_t digit = 0;
        carry = floor_div_radix_recip(
            digits[i] + carry, radix, radix_reciprocal, digit);
        digits[i] = digit;
        if (carry == 0) break;
    }
}

struct BorrowMap {
    uint32_t f0;
    uint32_t f1;
};

struct BorrowCompose {
    __device__ __forceinline__ BorrowMap operator()(const BorrowMap& left,
                                                     const BorrowMap& right) const {
        // right o left
        BorrowMap out;
        out.f0 = left.f0 ? right.f1 : right.f0;
        out.f1 = left.f1 ? right.f1 : right.f0;
        return out;
    }
};

__global__ void make_borrow_maps_kernel(const int64_t* digits, const int64_t* modulus,
                                        BorrowMap* maps, int active) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= active) return;
    const int64_t a = digits[i];
    const int64_t b = modulus[i];
    maps[i].f0 = static_cast<uint32_t>(a < b);
    maps[i].f1 = static_cast<uint32_t>(a <= b);
}

__global__ void emit_conditional_subtract_and_rns2_mont_kernel(
    int64_t* digits, const int64_t* modulus,
    const BorrowMap* inclusive_prefix,
    uint32_t* r0, uint32_t* r1,
    int canonical_count, int rns_active, int len, int64_t radix) {
    constexpr uint32_t p0 = 998244353u;
    constexpr uint32_t p1 = 1004535809u;
    constexpr uint32_t r2_0 = 932051910u;
    constexpr uint32_t r2_1 = 542374313u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    int64_t value = 0;
    if (i < canonical_count) {
        value = digits[i];
        // A final borrow means digits < modulus, so the original residue is
        // already canonical.  Otherwise subtract the modulus in-place.
        if (inclusive_prefix[canonical_count - 1].f0 == 0) {
            const int64_t borrow_in =
                (i == 0) ? 0 : static_cast<int64_t>(inclusive_prefix[i - 1].f0);
            value -= modulus[i] + borrow_in;
            if (value < 0) value += radix;
            digits[i] = value;
        }
    }

    const uint32_t d =
        (i < rns_active) ? static_cast<uint32_t>(value) : 0u;
    r0[i] = mul_mod_mont_2prime(d, r2_0, p0);
    r1[i] = mul_mod_mont_2prime(d, r2_1, p1);
}

void cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::string msg = where;
        msg += ": ";
        msg += cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
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

void ntt2_forward_dif_mont(uint32_t* r0, uint32_t* r1, int log_len,
                           const uint32_t* tw0, const uint32_t* tw1,
                           const std::vector<int>& offsets,
                           cudaStream_t stream) {
    const int len_total = 1 << log_len;
    const int threads = 256;
    int stage = log_len;
    const int shared_tail = (log_len >= kSharedNttTileLog) ? kSharedNttTileLog : 0;
#if GSRPS_GLOBAL_RADIX16
    // Exact multiples of three are handled more efficiently by pure radix-8:
    // it uses the same number of launches without radix-16 register pressure.
    const bool prefer_global_radix16 = ((log_len - shared_tail) % 3) != 0;
    if (prefer_global_radix16) {
        for (; stage - 3 > shared_tail; stage -= 4) {
            const int len = 1 << stage;
            const int groups = len_total / 16;
            const int y = limited_ntt_grid_y(groups, threads);
            ntt_stage4_2_dif_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(
                r0, r1,
                tw0 + offsets[stage], tw1 + offsets[stage],
                tw0 + offsets[stage - 1], tw1 + offsets[stage - 1],
                tw0 + offsets[stage - 2], tw1 + offsets[stage - 2],
                tw0 + offsets[stage - 3], tw1 + offsets[stage - 3],
                len, groups);
            cuda_check(cudaGetLastError(), "ntt_stage4_2_dif_mont launch");
        }
    }
#endif
#if GSRPS_GLOBAL_RADIX8
    for (; stage - 2 > shared_tail; stage -= 3) {
        const int len = 1 << stage;
        const int groups = len_total / 8;
        const int y = limited_ntt_grid_y(groups, threads);
        ntt_stage3_2_dif_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                              tw0 + offsets[stage],
                                                              tw1 + offsets[stage],
                                                              tw0 + offsets[stage - 1],
                                                              tw1 + offsets[stage - 1],
                                                              tw0 + offsets[stage - 2],
                                                              tw1 + offsets[stage - 2],
                                                              len, groups);
        cuda_check(cudaGetLastError(), "ntt_stage3_2_dif_mont launch");
    }
#endif
#if GSRPS_GLOBAL_RADIX4
    for (; stage - 1 > shared_tail; stage -= 2) {
        const int len = 1 << stage;
        const int groups = len_total / 4;
        const int y = limited_ntt_grid_y(groups, threads);
        ntt_stage2_2_dif_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                              tw0 + offsets[stage],
                                                              tw1 + offsets[stage],
                                                              tw0 + offsets[stage - 1],
                                                              tw1 + offsets[stage - 1],
                                                              len, groups);
        cuda_check(cudaGetLastError(), "ntt_stage2_2_dif_mont launch");
    }
#endif
    while (stage > shared_tail) {
        const int len = 1 << stage;
        const int butterflies = len_total / 2;
        const int y = limited_ntt_grid_y(butterflies, threads);
        ntt_stage_2_dif_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                             tw0 + offsets[stage],
                                                             tw1 + offsets[stage],
                                                             len, butterflies);
        cuda_check(cudaGetLastError(), "ntt_stage_2_dif_mont launch");
        --stage;
    }
    if (shared_tail != 0) {
        const int total_tiles = len_total / kSharedNttTile;
        const int blocks = std::max(1, std::min(ntt_block_limit(), total_tiles));
        ntt_tile_dif_mont_kernel<<<blocks, kSharedNttThreads, 0, stream>>>(r0, r1, tw0, tw1, total_tiles);
        cuda_check(cudaGetLastError(), "ntt_tile_dif_mont launch");
    }
}

void ntt2_inverse_product_dit_mont(uint32_t* r0, uint32_t* r1, int log_len,
                                   const uint32_t* tw0, const uint32_t* tw1,
                                   const std::vector<int>& offsets,
                                   uint32_t inv0, uint32_t inv1,
                                   const uint32_t* multiplier0 = nullptr,
                                   const uint32_t* multiplier1 = nullptr,
                                   cudaStream_t stream = cudaStreamPerThread) {
    const int len_total = 1 << log_len;
    const int threads = 256;
    const int blocks = (len_total + threads - 1) / threads;
    bool scale_fused = false;
    int stage = 1;
    if (log_len >= kSharedNttTileLog) {
        const int total_tiles = len_total / kSharedNttTile;
        const int tile_blocks = std::max(1, std::min(ntt_block_limit(), total_tiles));
        if (multiplier0 == nullptr) {
            ntt_tile_product_dit_mont_kernel<true><<<tile_blocks, kSharedNttThreads, 0, stream>>>(r0, r1, nullptr, nullptr,
                                                                                      tw0, tw1, total_tiles);
        } else {
            ntt_tile_product_dit_mont_kernel<false><<<tile_blocks, kSharedNttThreads, 0, stream>>>(r0, r1, multiplier0, multiplier1,
                                                                                       tw0, tw1, total_tiles);
        }
        cuda_check(cudaGetLastError(), "ntt_tile_product_dit_mont launch");
        stage = kSharedNttTileLog + 1;
    } else {
        if (multiplier0 == nullptr) {
            pointwise_square2_mont_kernel<<<blocks, threads, 0, stream>>>(r0, r1, len_total);
        } else {
            pointwise_mul2_mont_kernel<<<blocks, threads, 0, stream>>>(r0, r1, multiplier0, multiplier1, len_total);
        }
        cuda_check(cudaGetLastError(), "pointwise product fallback launch");
    }
#if GSRPS_GLOBAL_RADIX16
    const int shared_prefix = (log_len >= kSharedNttTileLog) ? kSharedNttTileLog : 0;
    const bool prefer_global_radix16 = ((log_len - shared_prefix) % 3) != 0;
    if (prefer_global_radix16) {
        for (; stage + 3 <= log_len; stage += 4) {
            const int len = 1 << (stage + 3);
            const int groups = len_total / 16;
            const int y = limited_ntt_grid_y(groups, threads);
            if (stage + 3 == log_len) {
                ntt_stage4_2_dit_mont_scaled_kernel<<<dim3(1, y), threads, 0, stream>>>(
                    r0, r1,
                    tw0 + offsets[stage], tw1 + offsets[stage],
                    tw0 + offsets[stage + 1], tw1 + offsets[stage + 1],
                    tw0 + offsets[stage + 2], tw1 + offsets[stage + 2],
                    tw0 + offsets[stage + 3], tw1 + offsets[stage + 3],
                    len, groups, inv0, inv1);
                scale_fused = true;
            } else {
                ntt_stage4_2_dit_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(
                    r0, r1,
                    tw0 + offsets[stage], tw1 + offsets[stage],
                    tw0 + offsets[stage + 1], tw1 + offsets[stage + 1],
                    tw0 + offsets[stage + 2], tw1 + offsets[stage + 2],
                    tw0 + offsets[stage + 3], tw1 + offsets[stage + 3],
                    len, groups);
            }
            cuda_check(cudaGetLastError(), "ntt_stage4_2_inverse_dit_mont launch");
        }
    }
#endif
#if GSRPS_GLOBAL_RADIX8
    for (; stage + 2 <= log_len; stage += 3) {
        const int len = 1 << (stage + 2);
        const int groups = len_total / 8;
        const int y = limited_ntt_grid_y(groups, threads);
        if (stage + 2 == log_len) {
            ntt_stage3_2_dit_mont_scaled_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                                         tw0 + offsets[stage],
                                                                         tw1 + offsets[stage],
                                                                         tw0 + offsets[stage + 1],
                                                                         tw1 + offsets[stage + 1],
                                                                         tw0 + offsets[stage + 2],
                                                                         tw1 + offsets[stage + 2],
                                                                         len, groups, inv0, inv1);
            scale_fused = true;
        } else {
            ntt_stage3_2_dit_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                                  tw0 + offsets[stage],
                                                                  tw1 + offsets[stage],
                                                                  tw0 + offsets[stage + 1],
                                                                  tw1 + offsets[stage + 1],
                                                                  tw0 + offsets[stage + 2],
                                                                  tw1 + offsets[stage + 2],
                                                                  len, groups);
        }
        cuda_check(cudaGetLastError(), "ntt_stage3_2_inverse_dit_mont launch");
    }
#endif
#if GSRPS_GLOBAL_RADIX4
    for (; stage + 1 <= log_len; stage += 2) {
        const int len = 1 << (stage + 1);
        const int groups = len_total / 4;
        const int y = limited_ntt_grid_y(groups, threads);
        if (stage + 1 == log_len) {
            ntt_stage2_2_mont_scaled_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                                     tw0 + offsets[stage],
                                                                     tw1 + offsets[stage],
                                                                     tw0 + offsets[stage + 1],
                                                                     tw1 + offsets[stage + 1],
                                                                     len, groups, inv0, inv1);
            scale_fused = true;
        } else {
            ntt_stage2_2_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                              tw0 + offsets[stage],
                                                              tw1 + offsets[stage],
                                                              tw0 + offsets[stage + 1],
                                                              tw1 + offsets[stage + 1],
                                                              len, groups);
        }
        cuda_check(cudaGetLastError(), "ntt_stage2_2_inverse_dit_mont launch");
    }
#endif
    while (stage <= log_len) {
        const int len = 1 << stage;
        const int butterflies = len_total / 2;
        const int y = limited_ntt_grid_y(butterflies, threads);
        if (stage == log_len) {
            ntt_stage_2_mont_scaled_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                                    tw0 + offsets[stage],
                                                                    tw1 + offsets[stage],
                                                                    len, butterflies, inv0, inv1);
            scale_fused = true;
        } else {
            ntt_stage_2_mont_kernel<<<dim3(1, y), threads, 0, stream>>>(r0, r1,
                                                             tw0 + offsets[stage],
                                                             tw1 + offsets[stage],
                                                             len, butterflies);
        }
        cuda_check(cudaGetLastError(), "ntt_stage_2_inverse_dit_mont launch");
        ++stage;
    }
    if (!scale_fused) {
        scale2_from_mont_kernel<<<blocks, threads, 0, stream>>>(r0, r1, len_total, inv0, inv1);
        cuda_check(cudaGetLastError(), "scale2_from_inverse_dit_mont launch");
    }
}

void run_bench_gsrps_full(uint64_t k, uint64_t b, uint64_t n, int c, int iterations,
                          bool multiply_mode = false, bool check_mode = false,
                          uint64_t check_witness = 2) {
    const auto complete_started = std::chrono::steady_clock::now();
    const cudaStream_t stream = cudaStreamPerThread;
    if (k == 0 || b < 2) throw std::runtime_error("full path requires k >= 1 and b >= 2");
    if (n == 0) throw std::runtime_error("full path requires positive n");
    if (c != -1 && c != 1) throw std::runtime_error("c must be -1 or +1");
    if (!check_mode && (iterations < 1 || iterations > 10000)) {
        throw std::runtime_error("iterations must be in 1..10000");
    }
    if (check_mode && check_witness < 2) throw std::runtime_error("Fermat witness must be at least 2");
#ifndef GSRPS_PREFERRED_RADIX
#define GSRPS_PREFERRED_RADIX 100000
#endif
    constexpr uint64_t preferred_radix = GSRPS_PREFERRED_RADIX;
    constexpr uint64_t maximum_single_limb_base = 998244353ull;
    if (b > maximum_single_limb_base) {
        throw std::runtime_error("current general-base path requires b <= 998244353");
    }

    int g = 1;
    uint64_t radix = b;
    while (radix <= preferred_radix / b && static_cast<uint64_t>(g) < n) {
        radix *= b;
        ++g;
    }

    // Keep the conservative ~1e5 limb radix unless increasing g actually
    // crosses an NTT power-of-two boundary.  A larger radix makes carry and
    // reduction arithmetic more expensive, so equal-length transforms do not
    // justify it.  This matters at sizes such as 1.5M decimal digits, where
    // 10^6 limbs halve the transform length relative to 10^5 limbs.
    const auto candidate_log_len = [=](int candidate_g, uint64_t candidate_radix,
                                       int& out_log_len) -> bool {
        const uint64_t candidate_low = n / static_cast<uint64_t>(candidate_g);
        if (candidate_low == 0 || candidate_low > (uint64_t(1) << 20)) return false;

        const int candidate_rem = static_cast<int>(n % static_cast<uint64_t>(candidate_g));
        uint64_t candidate_d = 1;
        for (int i = 0; i < candidate_rem; ++i) candidate_d *= b;
        if (candidate_d == 0 || k > UINT32_MAX / candidate_d) return false;
        const uint64_t candidate_divisor = k * candidate_d;
        if (candidate_divisor == 0 || candidate_divisor > UINT32_MAX) return false;

        uint64_t top = (c == 1) ? candidate_divisor : candidate_divisor - 1;
        uint64_t top_digits = 0;
        while (top != 0) {
            ++top_digits;
            top /= candidate_radix;
        }
        const uint64_t candidate_active = candidate_low + top_digits;
        if (candidate_active == 0 || candidate_active > (uint64_t(1) << 20) + 32) return false;

        int candidate_log = 1;
        while ((uint64_t(1) << candidate_log) < 2 * candidate_active) ++candidate_log;
        if (candidate_log > 21) return false;

        const boost::multiprecision::uint128_t coefficient_bound =
            boost::multiprecision::uint128_t(candidate_active) *
            (candidate_radix - 1) * (candidate_radix - 1);
        const boost::multiprecision::uint128_t crt_product =
            boost::multiprecision::uint128_t(kMods[0].p) * kMods[1].p;
        if (coefficient_bound >= crt_product) return false;
        out_log_len = candidate_log;
        return true;
    };

    int selected_log_len = 0;
    if (!candidate_log_len(g, radix, selected_log_len)) {
        throw std::runtime_error("conservative radix is incompatible with the current fixed-divisor/CRT path");
    }
    bool radix_tier_upgrade = false;
#ifndef GSRPS_DISABLE_RADIX_UPGRADE
    int candidate_g = g;
    uint64_t candidate_radix = radix;
    while (candidate_radix <= maximum_single_limb_base / b &&
           static_cast<uint64_t>(candidate_g) < n) {
        candidate_radix *= b;
        ++candidate_g;
        int candidate_log = 0;
        if (candidate_log_len(candidate_g, candidate_radix, candidate_log) &&
            candidate_log < selected_log_len) {
            g = candidate_g;
            radix = candidate_radix;
            selected_log_len = candidate_log;
            radix_tier_upgrade = true;
        }
    }
#endif
    const uint64_t low64 = n / static_cast<uint64_t>(g);
    const int rem = static_cast<int>(n % static_cast<uint64_t>(g));
    uint64_t d = 1;
    for (int i = 0; i < rem; ++i) d *= b;
    if (d != 0 && k > UINT32_MAX / d) {
        throw std::runtime_error("current fixed-divisor path requires k*b^(n mod g) < 2^32");
    }
    const uint64_t divisor = k * d;
    if (divisor == 0 || divisor > UINT32_MAX) throw std::runtime_error("fold divisor out of 32-bit range");
    if (low64 == 0) throw std::runtime_error("selected radix leaves no low limbs; exponent is too small for full benchmark");
    if (low64 > (uint64_t(1) << 20)) throw std::runtime_error("active limb count exceeds prototype range");
    const int low_count = static_cast<int>(low64);

    std::vector<uint64_t> divisor_digits;
    for (uint64_t x = divisor; x != 0; x /= radix) divisor_digits.push_back(x % radix);
    std::vector<int64_t> modulus_digits(static_cast<size_t>(low_count) + divisor_digits.size() + 1, 0);
    if (c == 1) {
        uint64_t carry = 1;
        for (size_t i = 0; i < modulus_digits.size() && carry != 0; ++i) {
            const uint64_t sum = static_cast<uint64_t>(modulus_digits[i]) + carry;
            modulus_digits[i] = static_cast<int64_t>(sum % radix);
            carry = sum / radix;
        }
        carry = 0;
        for (size_t j = 0; j < divisor_digits.size(); ++j) {
            const size_t i = static_cast<size_t>(low_count) + j;
            const uint64_t sum = static_cast<uint64_t>(modulus_digits[i]) + divisor_digits[j] + carry;
            modulus_digits[i] = static_cast<int64_t>(sum % radix);
            carry = sum / radix;
        }
        size_t i = static_cast<size_t>(low_count) + divisor_digits.size();
        while (carry != 0) {
            const uint64_t sum = static_cast<uint64_t>(modulus_digits[i]) + carry;
            modulus_digits[i] = static_cast<int64_t>(sum % radix);
            carry = sum / radix;
            ++i;
        }
    } else {
        std::fill(modulus_digits.begin(), modulus_digits.begin() + low_count,
                  static_cast<int64_t>(radix - 1));
        uint64_t top = divisor - 1;
        size_t i = static_cast<size_t>(low_count);
        do {
            modulus_digits[i++] = static_cast<int64_t>(top % radix);
            top /= radix;
        } while (top != 0);
    }
    while (modulus_digits.size() > 1 && modulus_digits.back() == 0) modulus_digits.pop_back();
    const int active = static_cast<int>(modulus_digits.size());
    const int canonical_count = active + 1;
    modulus_digits.push_back(0);
    int log_len = 1;
    while ((1 << log_len) < 2 * active) ++log_len;
    if (log_len > 21) throw std::runtime_error("required NTT length exceeds 2^21");
    const int len = 1 << log_len;
    const int coeff_count = 2 * active - 1;
    const int product_count = 2 * active;
    const int high_count = product_count - low_count;
    const boost::multiprecision::uint128_t coefficient_bound =
        boost::multiprecision::uint128_t(active) * (radix - 1) * (radix - 1);
    const boost::multiprecision::uint128_t crt_product =
        boost::multiprecision::uint128_t(kMods[0].p) * kMods[1].p;
    if (coefficient_bound >= crt_product) {
        throw std::runtime_error("two-prime CRT coefficient bound exceeded for selected radix");
    }
    const int total_roots = len - 1;
    const int threads = 256;
    const int div_blocks = (high_count + threads - 1) / threads;
    const int blocks = (len + threads - 1) / threads;
    const int carry_segment_count =
        (len + kExactCarrySegment - 1) / kExactCarrySegment;
    const int carry_segment_blocks =
        (carry_segment_count + threads - 1) / threads;
    const auto offsets = make_twiddle_offsets(log_len);
    const uint32_t ntt_inv0 = pow_mod_host(
        static_cast<uint32_t>(len), kMods[0].p - 2ull, kMods[0].p);
    const uint32_t ntt_inv1 = pow_mod_host(
        static_cast<uint32_t>(len), kMods[1].p - 2ull, kMods[1].p);

    boost::multiprecision::cpp_int check_modulus = 0;
    boost::multiprecision::cpp_int check_exponent = 0;
    if (check_mode) {
        boost::multiprecision::cpp_int power = 1;
        boost::multiprecision::cpp_int factor = b;
        uint64_t exponent = n;
        while (exponent != 0) {
            if (exponent & 1u) power *= factor;
            exponent >>= 1;
            if (exponent != 0) factor *= factor;
        }
        check_modulus = boost::multiprecision::cpp_int(k) * power + c;
        if (check_modulus <= check_witness) {
            throw std::runtime_error("Fermat witness must be smaller than the candidate");
        }
        check_exponent = check_modulus - 1;
    }

    int check_window_bits = 1;
    uint64_t planned_check_multiplications = 0;
    if (check_mode) {
        const uint64_t exponent_bits =
            static_cast<uint64_t>(boost::multiprecision::msb(check_exponent)) + 1;
        const auto count_window_multiplications =
            [&](int window_bits) -> uint64_t {
                uint64_t count = 0;
                int64_t bit = static_cast<int64_t>(exponent_bits) - 1;
                while (bit >= 0) {
                    if (!boost::multiprecision::bit_test(
                            check_exponent, static_cast<unsigned>(bit))) {
                        --bit;
                        continue;
                    }
                    int64_t low = std::max<int64_t>(0, bit - window_bits + 1);
                    while (low < bit &&
                           !boost::multiprecision::bit_test(
                               check_exponent, static_cast<unsigned>(low))) {
                        ++low;
                    }
                    ++count;
                    bit = low - 1;
                }
                return count;
            };

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
                   "query memory for window selection");
        const size_t memory_reserve = size_t(768) << 20;
        uint64_t best_operation_count = std::numeric_limits<uint64_t>::max();
        int best_window = 1;
        uint64_t best_multiplications = 0;
        for (int candidate = 1; candidate <= 8; ++candidate) {
            const size_t entries = size_t(1) << (candidate - 1);
            const size_t table_bytes =
                static_cast<size_t>(len) * entries * 2 * sizeof(uint32_t);
            const bool memory_safe =
                table_bytes <= free_bytes / 3 &&
                table_bytes + memory_reserve <= free_bytes;
            if (!memory_safe && candidate != 1) continue;
            const uint64_t multiplications =
                count_window_multiplications(candidate);
            const uint64_t setup_multiplications =
                static_cast<uint64_t>(entries - 1);
            const uint64_t operation_count =
                multiplications + setup_multiplications;
            if (operation_count < best_operation_count) {
                best_operation_count = operation_count;
                best_window = candidate;
                best_multiplications = multiplications;
            }
        }
        if (gpu_throttle_config().force_window_bits) {
            const int forced = gpu_throttle_config().window_bits;
            const size_t entries = size_t(1) << (forced - 1);
            const size_t table_bytes =
                static_cast<size_t>(len) * entries * 2 * sizeof(uint32_t);
            if (table_bytes + memory_reserve > free_bytes) {
                throw std::runtime_error(
                    "forced window table leaves less than 768 MiB of free GPU memory");
            }
            check_window_bits = forced;
            planned_check_multiplications =
                count_window_multiplications(forced);
        } else {
            check_window_bits = best_window;
            planned_check_multiplications = best_multiplications;
        }
        const size_t selected_entries =
            size_t(1) << (check_window_bits - 1);
        const size_t selected_table_bytes =
            static_cast<size_t>(len) * selected_entries * 2 * sizeof(uint32_t);
        std::cout << "window-selection: mode="
                  << (gpu_throttle_config().force_window_bits ? "forced" : "auto")
                  << ", bits=" << check_window_bits
                  << ", exponent_bits=" << exponent_bits
                  << ", planned_multiplies=" << planned_check_multiplications
                  << ", setup_multiplies=" << (selected_entries - 1)
                  << ", table_mib=" << std::fixed << std::setprecision(1)
                  << static_cast<double>(selected_table_bytes) / (1024.0 * 1024.0)
                  << "\n";
    }

    auto cpp_int_to_digits = [radix, active](boost::multiprecision::cpp_int value) {
        std::vector<int64_t> out(active, 0);
        for (int i = 0; i < active && value != 0; ++i) {
            const boost::multiprecision::cpp_int q = value / radix;
            out[i] = static_cast<int64_t>(value - q * radix);
            value = q;
        }
        if (value != 0) throw std::runtime_error("value does not fit selected radix representation");
        return out;
    };

    std::vector<int64_t> initial_digits(active, 0);
    uint64_t state = 0x46554c4c5f475352ull ^ k ^ b ^ n ^ static_cast<uint64_t>(c + 1);
    if (check_mode) {
        initial_digits[0] = 1;
    } else {
        for (int i = 0; i < low_count; ++i) {
            state ^= state << 7; state ^= state >> 9; state ^= state << 8;
            initial_digits[i] = static_cast<int64_t>(state % radix);
        }
    }
    std::vector<int64_t> multiplier_digits(active, 0);
    std::vector<int64_t> witness_square_digits(active, 0);
    if (check_mode) {
        const boost::multiprecision::cpp_int witness = check_witness;
        multiplier_digits = cpp_int_to_digits(witness % check_modulus);
        witness_square_digits = cpp_int_to_digits((witness * witness) % check_modulus);
    } else {
        uint64_t multiplier_state = state ^ 0x9e3779b97f4a7c15ull;
        for (int i = 0; i < low_count; ++i) {
            multiplier_state ^= multiplier_state << 7;
            multiplier_state ^= multiplier_state >> 9;
            multiplier_state ^= multiplier_state << 8;
            multiplier_digits[i] = static_cast<int64_t>(multiplier_state % radix);
        }
    }
    std::vector<uint32_t> h0(len, 0), h1(len, 0), hmul0(len, 0), hmul1(len, 0);
    std::vector<uint32_t> hpower0(len, 0), hpower1(len, 0);
    constexpr uint64_t mont_r0 = 301989884ull;
    constexpr uint64_t mont_r1 = 276824060ull;
    for (int i = 0; i < active; ++i) {
        const uint32_t d = static_cast<uint32_t>(initial_digits[i]);
        h0[i] = static_cast<uint32_t>((static_cast<uint64_t>(d) * mont_r0) % kMods[0].p);
        h1[i] = static_cast<uint32_t>((static_cast<uint64_t>(d) * mont_r1) % kMods[1].p);
        const uint32_t md = static_cast<uint32_t>(multiplier_digits[i]);
        hmul0[i] = static_cast<uint32_t>((static_cast<uint64_t>(md) * mont_r0) % kMods[0].p);
        hmul1[i] = static_cast<uint32_t>((static_cast<uint64_t>(md) * mont_r1) % kMods[1].p);
        if (check_mode) {
            const uint32_t pd = static_cast<uint32_t>(witness_square_digits[i]);
            hpower0[i] = static_cast<uint32_t>((static_cast<uint64_t>(pd) * mont_r0) % kMods[0].p);
            hpower1[i] = static_cast<uint32_t>((static_cast<uint64_t>(pd) * mont_r1) % kMods[1].p);
        }
    }

    uint32_t *r0 = nullptr, *r1 = nullptr;
    uint32_t *mul0 = nullptr, *mul1 = nullptr;
    uint32_t *power0 = nullptr, *power1 = nullptr;
    uint32_t *fwd0 = nullptr, *fwd1 = nullptr, *inv0 = nullptr, *inv1 = nullptr;
    uint64_t *raw_coeff = nullptr;
    int64_t *digits = nullptr, *modulus_d = nullptr, *folded = nullptr;
    UnsignedCarryMap *unsigned_carry_maps = nullptr, *unsigned_carry_prefix = nullptr;
    void* unsigned_carry_scan_temp = nullptr;
    size_t unsigned_carry_scan_temp_bytes = 0;
    SignedCarryMap *signed_carry_maps = nullptr, *signed_carry_prefix = nullptr;
    void* signed_carry_scan_temp = nullptr;
    size_t signed_carry_scan_temp_bytes = 0;
    AffineModMap *div_maps = nullptr, *div_prefix = nullptr;
    void* div_scan_temp = nullptr;
    size_t div_scan_temp_bytes = 0;
    uint64_t *div_sum_values = nullptr, *div_sum_prefix = nullptr;
    void* div_sum_temp = nullptr;
    size_t div_sum_temp_bytes = 0;
    BorrowMap *borrow_maps = nullptr, *borrow_prefix = nullptr;
    void* borrow_scan_temp = nullptr;
    size_t borrow_scan_temp_bytes = 0;
    uint64_t *quotient = nullptr, *remainder = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    std::vector<cudaGraph_t> iteration_graphs;
    std::vector<cudaGraphExec_t> iteration_graph_execs;
    auto free_all = [&]() {
        if (stop) cudaEventDestroy(stop); if (start) cudaEventDestroy(start);
        for (cudaGraphExec_t exec : iteration_graph_execs) {
            if (exec) cudaGraphExecDestroy(exec);
        }
        for (cudaGraph_t graph : iteration_graphs) {
            if (graph) cudaGraphDestroy(graph);
        }
        cudaFree(borrow_scan_temp);
        cudaFree(borrow_prefix); cudaFree(borrow_maps);
        cudaFree(remainder); cudaFree(quotient);
        cudaFree(signed_carry_scan_temp);
        cudaFree(signed_carry_prefix); cudaFree(signed_carry_maps);
        cudaFree(unsigned_carry_scan_temp);
        cudaFree(unsigned_carry_prefix); cudaFree(unsigned_carry_maps);
        cudaFree(div_sum_temp); cudaFree(div_sum_prefix); cudaFree(div_sum_values);
        cudaFree(div_scan_temp); cudaFree(div_prefix); cudaFree(div_maps);
        cudaFree(folded); cudaFree(modulus_d); cudaFree(digits); cudaFree(raw_coeff);
        cudaFree(inv1); cudaFree(inv0); cudaFree(fwd1); cudaFree(fwd0);
        cudaFree(power1); cudaFree(power0);
        cudaFree(mul1); cudaFree(mul0); cudaFree(r1); cudaFree(r0);
    };
    try {
        cuda_check(cudaMalloc(&r0, sizeof(uint32_t) * len), "full malloc r0");
        cuda_check(cudaMalloc(&r1, sizeof(uint32_t) * len), "full malloc r1");
        const int check_table_entries = 1 << (check_window_bits - 1);
        const int multiplier_entries = check_mode ? check_table_entries : (multiply_mode ? 1 : 0);
        if (multiplier_entries != 0) {
            cuda_check(cudaMalloc(&mul0, sizeof(uint32_t) * len * multiplier_entries), "full malloc mul0");
            cuda_check(cudaMalloc(&mul1, sizeof(uint32_t) * len * multiplier_entries), "full malloc mul1");
        }
        if (check_mode) {
            cuda_check(cudaMalloc(&power0, sizeof(uint32_t) * len), "check malloc witness square0");
            cuda_check(cudaMalloc(&power1, sizeof(uint32_t) * len), "check malloc witness square1");
        }
        cuda_check(cudaMalloc(&fwd0, sizeof(uint32_t) * total_roots), "full malloc fwd0");
        cuda_check(cudaMalloc(&fwd1, sizeof(uint32_t) * total_roots), "full malloc fwd1");
        cuda_check(cudaMalloc(&inv0, sizeof(uint32_t) * total_roots), "full malloc inv0");
        cuda_check(cudaMalloc(&inv1, sizeof(uint32_t) * total_roots), "full malloc inv1");
        cuda_check(cudaMalloc(&raw_coeff, sizeof(uint64_t) * len), "full malloc coeff");
        cuda_check(cudaMalloc(&digits, sizeof(int64_t) * len), "full malloc digits");
        cuda_check(cudaMalloc(&modulus_d, sizeof(int64_t) * canonical_count), "full malloc modulus");
        cuda_check(cudaMalloc(&folded, sizeof(int64_t) * len), "full malloc folded");
        cuda_check(cudaMalloc(&quotient, sizeof(uint64_t) * high_count), "full malloc quotient");
        cuda_check(cudaMalloc(&remainder, sizeof(uint64_t)), "full malloc remainder");
        cuda_check(cudaMalloc(&unsigned_carry_maps,
                              sizeof(UnsignedCarryMap) * carry_segment_count),
                   "full malloc unsigned carry maps");
        cuda_check(cudaMalloc(&unsigned_carry_prefix,
                              sizeof(UnsignedCarryMap) * carry_segment_count),
                   "full malloc unsigned carry prefix");
        const UnsignedCarryCompose unsigned_carry_compose{};
        cuda_check(cub::DeviceScan::InclusiveScan(
                       nullptr, unsigned_carry_scan_temp_bytes,
                       unsigned_carry_maps, unsigned_carry_prefix,
                       unsigned_carry_compose, carry_segment_count),
                   "full query unsigned carry scan temp");
        cuda_check(cudaMalloc(&unsigned_carry_scan_temp,
                              unsigned_carry_scan_temp_bytes),
                   "full malloc unsigned carry scan temp");
        cuda_check(cudaMalloc(&signed_carry_maps,
                              sizeof(SignedCarryMap) * carry_segment_count),
                   "full malloc signed carry maps");
        cuda_check(cudaMalloc(&signed_carry_prefix,
                              sizeof(SignedCarryMap) * carry_segment_count),
                   "full malloc signed carry prefix");
        const SignedCarryCompose signed_carry_compose{};
        cuda_check(cub::DeviceScan::InclusiveScan(
                       nullptr, signed_carry_scan_temp_bytes,
                       signed_carry_maps, signed_carry_prefix,
                       signed_carry_compose, carry_segment_count),
                   "full query signed carry scan temp");
        cuda_check(cudaMalloc(&signed_carry_scan_temp,
                              signed_carry_scan_temp_bytes),
                   "full malloc signed carry scan temp");
        const uint32_t divisor32 = static_cast<uint32_t>(divisor);
        const uint64_t divisor_reciprocal = (divisor == 1) ? 0 : (UINT64_MAX / divisor);
        const uint64_t radix_reciprocal = UINT64_MAX / radix;
        const int exact_positions = radix_u64_exact_positions(radix);
        const bool use_radix_multiple = (radix % divisor) == 0;
        const bool use_sum_scan = !use_radix_multiple && (radix % divisor) == 1;
        const AffineCompose compose{divisor32, divisor_reciprocal};
        const AffineModMap identity{static_cast<uint32_t>(1 % divisor), 0};
        if (!use_radix_multiple && !use_sum_scan) {
            cuda_check(cudaMalloc(&div_maps, sizeof(AffineModMap) * high_count), "full malloc div maps");
            cuda_check(cudaMalloc(&div_prefix, sizeof(AffineModMap) * high_count), "full malloc div prefix");
            cuda_check(cub::DeviceScan::ExclusiveScan(nullptr, div_scan_temp_bytes,
                                                       div_maps, div_prefix, compose, identity, high_count),
                       "full query div scan temp");
            cuda_check(cudaMalloc(&div_scan_temp, div_scan_temp_bytes), "full malloc div scan temp");
        } else if (use_sum_scan) {
            cuda_check(cudaMalloc(&div_sum_values, sizeof(uint64_t) * high_count), "full malloc div sum values");
            cuda_check(cudaMalloc(&div_sum_prefix, sizeof(uint64_t) * high_count), "full malloc div sum prefix");
            cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, div_sum_temp_bytes,
                                                      div_sum_values, div_sum_prefix, high_count),
                       "full query div sum temp");
            cuda_check(cudaMalloc(&div_sum_temp, div_sum_temp_bytes), "full malloc div sum temp");
        }
        cuda_check(cudaMalloc(&borrow_maps, sizeof(BorrowMap) * canonical_count), "full malloc borrow maps");
        cuda_check(cudaMalloc(&borrow_prefix, sizeof(BorrowMap) * canonical_count), "full malloc borrow prefix");
        const BorrowCompose borrow_compose{};
        cuda_check(cub::DeviceScan::InclusiveScan(nullptr, borrow_scan_temp_bytes,
                                                   borrow_maps, borrow_prefix,
                                                   borrow_compose, canonical_count),
                   "full query borrow scan temp");
        cuda_check(cudaMalloc(&borrow_scan_temp, borrow_scan_temp_bytes), "full malloc borrow scan temp");

        auto hfwd0 = make_twiddle_table_host(log_len, kMods[0].p, kMods[0].g, false);
        auto hfwd1 = make_twiddle_table_host(log_len, kMods[1].p, kMods[1].g, false);
        auto hinv0 = make_twiddle_table_host(log_len, kMods[0].p, kMods[0].g, true);
        auto hinv1 = make_twiddle_table_host(log_len, kMods[1].p, kMods[1].g, true);
        for (auto& w : hfwd0) w = static_cast<uint32_t>((static_cast<uint64_t>(w) * mont_r0) % kMods[0].p);
        for (auto& w : hfwd1) w = static_cast<uint32_t>((static_cast<uint64_t>(w) * mont_r1) % kMods[1].p);
        for (auto& w : hinv0) w = static_cast<uint32_t>((static_cast<uint64_t>(w) * mont_r0) % kMods[0].p);
        for (auto& w : hinv1) w = static_cast<uint32_t>((static_cast<uint64_t>(w) * mont_r1) % kMods[1].p);
        cuda_check(cudaMemcpy(r0, h0.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice), "full copy r0");
        cuda_check(cudaMemcpy(r1, h1.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice), "full copy r1");
        if (multiply_mode || check_mode) {
            cuda_check(cudaMemcpy(mul0, hmul0.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice), "full copy mul0");
            cuda_check(cudaMemcpy(mul1, hmul1.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice), "full copy mul1");
        }
        if (check_mode) {
            cuda_check(cudaMemcpy(power0, hpower0.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice),
                       "check copy witness square0");
            cuda_check(cudaMemcpy(power1, hpower1.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice),
                       "check copy witness square1");
        }
        cuda_check(cudaMemcpy(fwd0, hfwd0.data(), sizeof(uint32_t) * total_roots, cudaMemcpyHostToDevice), "full copy fwd0");
        cuda_check(cudaMemcpy(fwd1, hfwd1.data(), sizeof(uint32_t) * total_roots, cudaMemcpyHostToDevice), "full copy fwd1");
        cuda_check(cudaMemcpy(inv0, hinv0.data(), sizeof(uint32_t) * total_roots, cudaMemcpyHostToDevice), "full copy inv0");
        cuda_check(cudaMemcpy(inv1, hinv1.data(), sizeof(uint32_t) * total_roots, cudaMemcpyHostToDevice), "full copy inv1");
        cuda_check(cudaMemcpy(modulus_d, modulus_digits.data(), sizeof(int64_t) * canonical_count, cudaMemcpyHostToDevice), "full copy modulus");
        if (multiply_mode || check_mode) ntt2_forward_dif_mont(mul0, mul1, log_len, fwd0, fwd1, offsets, stream);
        if (check_mode) ntt2_forward_dif_mont(power0, power1, log_len, fwd0, fwd1, offsets, stream);

        const bool validate_small = !check_mode &&
            (n <= 200000 && iterations <= 1000);
        const bool debug_intermediate = !check_mode && n <= 200000 && iterations == 1;
        auto iteration = [&](const uint32_t* cached0, const uint32_t* cached1) {
            ntt2_forward_dif_mont(r0, r1, log_len, fwd0, fwd1, offsets, stream);
            ntt2_inverse_product_dit_mont(r0, r1, log_len, inv0, inv1, offsets,
                                          ntt_inv0, ntt_inv1, cached0, cached1, stream);
            if (debug_intermediate) {
                crt2_raw_kernel<<<blocks, threads, 0, stream>>>(r0, r1, raw_coeff, coeff_count);
                cuda_check(cudaGetLastError(), "full debug CRT launch");
            }
            crt2_unsigned_segment_prepare_kernel<<<carry_segment_blocks, threads, 0, stream>>>(
                r0, r1, digits, unsigned_carry_maps,
                coeff_count, len, radix, radix_reciprocal, exact_positions);
            cuda_check(cub::DeviceScan::InclusiveScan(
                           unsigned_carry_scan_temp, unsigned_carry_scan_temp_bytes,
                           unsigned_carry_maps, unsigned_carry_prefix,
                           unsigned_carry_compose, carry_segment_count, stream),
                       "full unsigned carry scan");
            unsigned_segment_apply_kernel<<<carry_segment_blocks, threads, 0, stream>>>(
                digits, unsigned_carry_prefix, len, radix, radix_reciprocal);
            std::vector<int64_t> debug_product_digits;
            boost::multiprecision::cpp_int debug_product = 0;
            if (debug_intermediate) {
                std::vector<uint64_t> debug_raw_coeff(coeff_count);
                cuda_check(cudaMemcpy(debug_raw_coeff.data(), raw_coeff,
                                      sizeof(uint64_t) * coeff_count, cudaMemcpyDeviceToHost),
                           "full debug copy raw CRT coefficients");
                std::vector<int64_t> debug_raw_digits(product_count, 0);
                boost::multiprecision::uint128_t debug_carry = 0;
                for (int i = 0; i < product_count; ++i) {
                    if (i < coeff_count) debug_carry += debug_raw_coeff[i];
                    debug_raw_digits[i] = static_cast<int64_t>(debug_carry % radix);
                    debug_carry /= radix;
                }
                if (debug_carry != 0) throw std::runtime_error("debug raw CRT carry overflow");
                debug_product_digits.resize(product_count);
                cuda_check(cudaMemcpy(debug_product_digits.data(), digits,
                                      sizeof(int64_t) * product_count, cudaMemcpyDeviceToHost),
                           "full debug copy product digits");
                boost::multiprecision::cpp_int debug_initial = 0;
                boost::multiprecision::cpp_int debug_multiplier = 0;
                for (int i = active - 1; i >= 0; --i) debug_initial = debug_initial * radix + initial_digits[i];
                for (int i = active - 1; i >= 0; --i) {
                    debug_multiplier = debug_multiplier * radix + multiplier_digits[i];
                }
                boost::multiprecision::cpp_int debug_raw_product = 0;
                for (int i = product_count - 1; i >= 0; --i) {
                    debug_raw_product = debug_raw_product * radix + debug_raw_digits[i];
                    debug_product = debug_product * radix + debug_product_digits[i];
                }
                const boost::multiprecision::cpp_int debug_expected = cached0 != nullptr
                    ? debug_initial * debug_multiplier
                    : debug_initial * debug_initial;
                if (debug_raw_product != debug_expected) {
                    throw std::runtime_error("debug stage mismatch: NTT/CRT");
                }
                if (debug_product != debug_expected) {
                    throw std::runtime_error("debug stage mismatch: unsigned carry");
                }
            }
            if (use_radix_multiple) {
                emit_div_radix_multiple_kernel<<<div_blocks, threads, 0, stream>>>(digits, low_count, high_count,
                                                                         radix / divisor, divisor32,
                                                                         divisor_reciprocal, quotient, remainder);
            } else if (use_sum_scan) {
                make_div_sum_values_kernel<<<div_blocks, threads, 0, stream>>>(digits, low_count, high_count, div_sum_values);
                cuda_check(cub::DeviceScan::ExclusiveSum(div_sum_temp, div_sum_temp_bytes,
                                                          div_sum_values, div_sum_prefix, high_count, stream),
                           "full div sum scan");
                emit_div_from_sum_prefix_kernel<<<div_blocks, threads, 0, stream>>>(digits, low_count, high_count, radix,
                                                                          divisor32, divisor_reciprocal,
                                                                          div_sum_prefix, quotient, remainder);
            } else {
                make_div_affine_maps_kernel<<<div_blocks, threads, 0, stream>>>(digits, low_count, high_count,
                                                                      static_cast<uint32_t>(radix % divisor),
                                                                      divisor32, div_maps);
                cuda_check(cub::DeviceScan::ExclusiveScan(div_scan_temp, div_scan_temp_bytes,
                                                           div_maps, div_prefix, compose, identity, high_count, stream),
                           "full div affine scan");
                emit_div_from_prefix_kernel<<<div_blocks, threads, 0, stream>>>(digits, low_count, high_count, radix,
                                                                      divisor32, divisor_reciprocal,
                                                                      div_prefix, quotient, remainder);
            }
            if (debug_intermediate) {
                std::vector<uint64_t> debug_quotient(high_count);
                uint64_t debug_remainder = 0;
                cuda_check(cudaMemcpy(debug_quotient.data(), quotient,
                                      sizeof(uint64_t) * high_count, cudaMemcpyDeviceToHost),
                           "full debug copy quotient");
                cuda_check(cudaMemcpy(&debug_remainder, remainder, sizeof(uint64_t), cudaMemcpyDeviceToHost),
                           "full debug copy remainder");
                boost::multiprecision::cpp_int high = 0, qvalue = 0;
                for (int i = product_count - 1; i >= low_count; --i) high = high * radix + debug_product_digits[i];
                for (int i = high_count - 1; i >= 0; --i) qvalue = qvalue * radix + debug_quotient[i];
                if (high != qvalue * divisor + debug_remainder) {
                    throw std::runtime_error("debug stage mismatch: fixed-divisor scan");
                }
            }
            folded_signed_segment_prepare_kernel<<<carry_segment_blocks, threads, 0, stream>>>(
                digits, quotient, remainder, modulus_d, folded,
                signed_carry_maps,
                low_count, high_count, active, len, c,
                radix, radix_reciprocal, exact_positions);
            cuda_check(cub::DeviceScan::InclusiveScan(
                           signed_carry_scan_temp, signed_carry_scan_temp_bytes,
                           signed_carry_maps, signed_carry_prefix,
                           signed_carry_compose, carry_segment_count, stream),
                       "full signed carry scan");
            signed_segment_apply_kernel<<<carry_segment_blocks, threads, 0, stream>>>(
                folded, signed_carry_prefix, len, radix, radix_reciprocal);
            const int canonical_blocks = (canonical_count + threads - 1) / threads;
            make_borrow_maps_kernel<<<canonical_blocks, threads, 0, stream>>>(folded, modulus_d, borrow_maps, canonical_count);
            cuda_check(cub::DeviceScan::InclusiveScan(borrow_scan_temp, borrow_scan_temp_bytes,
                                                       borrow_maps, borrow_prefix,
                                                       borrow_compose, canonical_count, stream),
                       "full borrow scan");
            emit_conditional_subtract_and_rns2_mont_kernel<<<blocks, threads, 0, stream>>>(
                folded, modulus_d, borrow_prefix, r0, r1,
                canonical_count, active, len, radix);
        };

        if (check_mode && !ntt_blocks_are_forced()) {
            const uint64_t total_bits =
                static_cast<uint64_t>(boost::multiprecision::msb(check_exponent)) + 1;
            if (total_bits >= 100000) {
                int device = 0;
                cudaDeviceProp properties{};
                cuda_check(cudaGetDevice(&device), "autotune get device");
                cuda_check(cudaGetDeviceProperties(&properties, device),
                           "autotune get device properties");
                const int maximum_useful_blocks =
                    std::max(1, (len / 2 + threads - 1) / threads);
                std::vector<int> candidates;
                const auto add_candidate = [&](int value) {
                    value = std::max(1, std::min(value, maximum_useful_blocks));
                    if (std::find(candidates.begin(), candidates.end(), value) ==
                        candidates.end()) {
                        candidates.push_back(value);
                    }
                };
                add_candidate(kDefaultNttBlocks);
                add_candidate(128);
                add_candidate(256);
                for (int multiplier : {1, 2, 3, 4, 6, 8, 10}) {
                    add_candidate(properties.multiProcessorCount * multiplier);
                }
                std::sort(candidates.begin(), candidates.end());

                int driver_version = 0;
                int runtime_version = 0;
                cuda_check(cudaDriverGetVersion(&driver_version),
                           "autotune get driver version");
                cuda_check(cudaRuntimeGetVersion(&runtime_version),
                           "autotune get runtime version");
                std::ostringstream device_uuid;
                device_uuid << std::hex << std::setfill('0');
                for (unsigned char value : properties.uuid.bytes) {
                    device_uuid << std::setw(2) << static_cast<unsigned>(value);
                }
                const bool autotune_uses_graphs =
                    std::getenv("GSRPS_DISABLE_CUDA_GRAPHS") == nullptr;
                const uint64_t multiply_ratio_per_mille =
                    (planned_check_multiplications * 1000 + total_bits / 2) /
                    total_bits;
                std::ostringstream tuning_key_builder;
                tuning_key_builder
                    << "schema=1|build=" << GSRPS_BUILD_ID
                    << "|uuid=" << device_uuid.str()
                    << "|cc=" << properties.major << "." << properties.minor
                    << "|sm=" << properties.multiProcessorCount
                    << "|driver=" << driver_version
                    << "|runtime=" << runtime_version
                    << "|len=" << len
                    << "|radix=" << radix
                    << "|active=" << active
                    << "|low=" << low_count
                    << "|high=" << high_count
                    << "|window=" << check_window_bits
                    << "|total_bits=" << total_bits
                    << "|mul_ratio_permille=" << multiply_ratio_per_mille
                    << "|cuda_graphs=" << (autotune_uses_graphs ? 1 : 0)
                    << "|b=" << b << "|n=" << n << "|c=" << c
                    << "|carry_segment=" << kExactCarrySegment
                    << "|global_radix16=" << GSRPS_GLOBAL_RADIX16;
                const std::string tuning_key = tuning_key_builder.str();
                const CachedTuning cached_tuning =
                    load_cached_tuning(tuning_key);
                const bool cache_hit =
                    cached_tuning.blocks != 0 &&
                    std::find(candidates.begin(), candidates.end(),
                              cached_tuning.blocks) != candidates.end();
                if (cache_hit) {
                    gpu_throttle_config().ntt_blocks = cached_tuning.blocks;
                    std::cout << "ntt-autotune: cache-hit, sm_count="
                              << properties.multiProcessorCount
                              << ", ntt_length=" << len
                              << ", selected=" << cached_tuning.blocks
                              << ", age_h=" << std::fixed
                              << std::setprecision(3)
                              << cached_tuning.age_hours << "\n";
                }

                if (!cache_hit) {

                struct TuneResult {
                    int blocks;
                    std::vector<double> square_ms;
                    std::vector<double> multiply_ms;
                };
                std::vector<TuneResult> results;
                results.reserve(candidates.size());
                for (int candidate : candidates) {
                    results.push_back(TuneResult{candidate, {}, {}});
                }

                cuda_check(cudaEventCreate(&start), "autotune create start event");
                cuda_check(cudaEventCreate(&stop), "autotune create stop event");
                const auto autotune_started = std::chrono::steady_clock::now();
                const auto reset_tuning_residue = [&]() {
                    cuda_check(cudaMemcpy(r0, hmul0.data(), sizeof(uint32_t) * len,
                                          cudaMemcpyHostToDevice),
                               "autotune reset residue0");
                    cuda_check(cudaMemcpy(r1, hmul1.data(), sizeof(uint32_t) * len,
                                          cudaMemcpyHostToDevice),
                               "autotune reset residue1");
                };
                struct TuningGraph {
                    int blocks;
                    const uint32_t* cached0;
                    const uint32_t* cached1;
                    int iterations;
                    cudaGraph_t graph;
                    cudaGraphExec_t exec;
                };
                std::vector<TuningGraph> tuning_graphs;
                struct TuningGraphCleanup {
                    std::vector<TuningGraph>* graphs;
                    ~TuningGraphCleanup() {
                        for (TuningGraph& graph : *graphs) {
                            if (graph.exec != nullptr) {
                                cudaGraphExecDestroy(graph.exec);
                                graph.exec = nullptr;
                            }
                            if (graph.graph != nullptr) {
                                cudaGraphDestroy(graph.graph);
                                graph.graph = nullptr;
                            }
                        }
                    }
                } tuning_graph_cleanup{&tuning_graphs};
                const auto get_tuning_graph =
                    [&](const uint32_t* cached0, const uint32_t* cached1,
                        int iterations) -> cudaGraphExec_t {
                        const int blocks = ntt_block_limit();
                        for (const TuningGraph& existing : tuning_graphs) {
                            if (existing.blocks == blocks &&
                                existing.cached0 == cached0 &&
                                existing.cached1 == cached1 &&
                                existing.iterations == iterations) {
                                return existing.exec;
                            }
                        }
                        cudaGraph_t graph = nullptr;
                        cudaGraphExec_t exec = nullptr;
                        cuda_check(cudaStreamBeginCapture(
                                       stream, cudaStreamCaptureModeThreadLocal),
                                   "autotune begin graph capture");
                        for (int i = 0; i < iterations; ++i) {
                            iteration(cached0, cached1);
                        }
                        cuda_check(cudaStreamEndCapture(stream, &graph),
                                   "autotune end graph capture");
                        try {
                            cuda_check(cudaGraphInstantiate(
                                           &exec, graph, nullptr, nullptr, 0),
                                       "autotune instantiate graph");
                        } catch (...) {
                            cudaGraphDestroy(graph);
                            throw;
                        }
                        tuning_graphs.push_back(TuningGraph{
                            blocks, cached0, cached1, iterations, graph, exec});
                        return exec;
                    };
                const auto time_batch =
                    [&](const uint32_t* cached0, const uint32_t* cached1,
                        int count) -> double {
                        const int graph_iterations =
                            cached0 == nullptr ? 64 : 16;
                        if (autotune_uses_graphs &&
                            count % graph_iterations != 0) {
                            throw std::runtime_error(
                                "autotune batch is not divisible by graph size");
                        }
                        const cudaGraphExec_t exec = autotune_uses_graphs
                            ? get_tuning_graph(cached0, cached1,
                                               graph_iterations)
                            : nullptr;
                        reset_tuning_residue();
                        cuda_check(cudaEventRecord(start, stream),
                                   "autotune record start");
                        if (autotune_uses_graphs) {
                            for (int i = 0; i < count / graph_iterations; ++i) {
                                cuda_check(cudaGraphLaunch(exec, stream),
                                           "autotune launch graph");
                            }
                        } else {
                            for (int i = 0; i < count; ++i) {
                                iteration(cached0, cached1);
                            }
                        }
                        cuda_check(cudaEventRecord(stop, stream),
                                   "autotune record stop");
                        cuda_check(cudaEventSynchronize(stop), "autotune synchronize");
                        float elapsed_ms = 0;
                        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
                                   "autotune elapsed time");
                        return static_cast<double>(elapsed_ms) / count;
                    };

                // Raise clocks and populate instruction/data caches before collecting
                // interleaved samples.  The residue starts at the witness rather than
                // one, so carry/reduction behavior is representative.
                gpu_throttle_config().ntt_blocks =
                    candidates[candidates.size() / 2];
                reset_tuning_residue();
                if (autotune_uses_graphs) {
                    const cudaGraphExec_t warmup_graph =
                        get_tuning_graph(nullptr, nullptr, 64);
                    for (int i = 0; i < 16; ++i) {
                        cuda_check(cudaGraphLaunch(warmup_graph, stream),
                                   "autotune launch warmup graph");
                    }
                } else {
                    for (int i = 0; i < 1024; ++i) {
                        iteration(nullptr, nullptr);
                    }
                }
                cuda_check(cudaStreamSynchronize(stream), "autotune warmup synchronize");

                constexpr int coarse_rounds = 3;
                constexpr int coarse_square_iterations = 128;
                constexpr int coarse_multiply_iterations = 32;
                for (int round = 0; round < coarse_rounds; ++round) {
                    for (size_t order_index = 0; order_index < results.size();
                         ++order_index) {
                        const size_t index = (round & 1)
                            ? results.size() - 1 - order_index
                            : order_index;
                        TuneResult& result = results[index];
                        gpu_throttle_config().ntt_blocks = result.blocks;
                        result.square_ms.push_back(
                            time_batch(nullptr, nullptr, coarse_square_iterations));
                        result.multiply_ms.push_back(
                            time_batch(mul0, mul1, coarse_multiply_iterations));
                    }
                }

                const auto median = [](std::vector<double> values) {
                    std::sort(values.begin(), values.end());
                    return values[values.size() / 2];
                };
                const uint64_t setup_multiplications =
                    (uint64_t(1) << (check_window_bits - 1)) - 1;
                const auto projected_score = [&](const TuneResult& result) {
                    return median(result.square_ms) * static_cast<double>(total_bits) +
                           median(result.multiply_ms) * static_cast<double>(
                               planned_check_multiplications + setup_multiplications);
                };
                std::vector<size_t> finalists(results.size());
                for (size_t i = 0; i < finalists.size(); ++i) finalists[i] = i;
                std::sort(finalists.begin(), finalists.end(),
                          [&](size_t left, size_t right) {
                              return projected_score(results[left]) <
                                     projected_score(results[right]);
                          });
                if (finalists.size() > 3) finalists.resize(3);
                for (size_t index : finalists) {
                    results[index].square_ms.clear();
                    results[index].multiply_ms.clear();
                }

                constexpr int final_rounds = 5;
                constexpr int final_square_iterations = 256;
                constexpr int final_multiply_iterations = 64;
                for (int round = 0; round < final_rounds; ++round) {
                    for (size_t order_index = 0; order_index < finalists.size();
                         ++order_index) {
                        const size_t finalist_position = (round & 1)
                            ? finalists.size() - 1 - order_index
                            : order_index;
                        TuneResult& result =
                            results[finalists[finalist_position]];
                        gpu_throttle_config().ntt_blocks = result.blocks;
                        result.square_ms.push_back(
                            time_batch(nullptr, nullptr, final_square_iterations));
                        result.multiply_ms.push_back(
                            time_batch(mul0, mul1, final_multiply_iterations));
                    }
                }

                double best_score = std::numeric_limits<double>::infinity();
                int selected_blocks = kDefaultNttBlocks;
                std::cout << "ntt-autotune: sm_count="
                          << properties.multiProcessorCount
                          << ", ntt_length=" << len << ", candidates=";
                for (size_t i = 0; i < results.size(); ++i) {
                    TuneResult& result = results[i];
                    const double square_ms = median(result.square_ms);
                    const double multiply_ms = median(result.multiply_ms);
                    const bool is_finalist =
                        std::find(finalists.begin(), finalists.end(), i) !=
                        finalists.end();
                    const double score = projected_score(result);
                    if (is_finalist && score < best_score) {
                        best_score = score;
                        selected_blocks = result.blocks;
                    }
                    if (i != 0) std::cout << ";";
                    std::cout << result.blocks << ":" << std::fixed
                              << std::setprecision(3) << square_ms
                              << "/" << multiply_ms
                              << (is_finalist ? "*" : "");
                }
                gpu_throttle_config().ntt_blocks = selected_blocks;
                const double autotune_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - autotune_started).count();
                std::cout << ", selected=" << selected_blocks
                          << ", calibration_s=" << std::setprecision(3)
                          << autotune_seconds << "\n";
                cuda_check(cudaMemcpy(r0, h0.data(), sizeof(uint32_t) * len,
                                      cudaMemcpyHostToDevice),
                           "autotune restore residue0");
                cuda_check(cudaMemcpy(r1, h1.data(), sizeof(uint32_t) * len,
                                      cudaMemcpyHostToDevice),
                           "autotune restore residue1");
                cuda_check(cudaStreamSynchronize(stream), "autotune restore synchronize");
                for (TuningGraph& graph : tuning_graphs) {
                    cuda_check(cudaGraphExecDestroy(graph.exec),
                               "autotune destroy graph exec");
                    graph.exec = nullptr;
                    cuda_check(cudaGraphDestroy(graph.graph),
                               "autotune destroy graph");
                    graph.graph = nullptr;
                }
                cuda_check(cudaEventDestroy(stop), "autotune destroy stop event");
                stop = nullptr;
                cuda_check(cudaEventDestroy(start), "autotune destroy start event");
                start = nullptr;
                const double autotune_total_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        autotune_started).count();
                const bool cache_saved =
                    save_cached_tuning(tuning_key, selected_blocks);
                std::cout << "ntt-autotune-total: seconds=" << std::fixed
                          << std::setprecision(3) << autotune_total_seconds
                          << ", cache="
                          << (cache_saved ? "stored" :
                              (tuning_cache_enabled() ? "write-failed" :
                                                       "disabled"))
                          << "\n";
                }
            } else {
                std::cout << "ntt-autotune: skipped for exponent shorter than 100000 bits"
                          << ", selected=" << ntt_block_limit() << "\n";
            }
        } else if (check_mode) {
            std::cout << "ntt-autotune: disabled by --force-ntt-blocks"
                      << ", selected=" << ntt_block_limit() << "\n";
        }

        if (check_mode) {
            // Build the transformed odd-power table w^1,w^3,...,w^31.  The
            // current residue stays in coefficient form between operations;
            // table entries remain in DIF frequency order for cached multiply.
            cuda_check(cudaMemcpy(r0, hmul0.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice),
                       "check seed odd powers0");
            cuda_check(cudaMemcpy(r1, hmul1.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice),
                       "check seed odd powers1");
            for (int entry = 1; entry < check_table_entries; ++entry) {
                iteration(power0, power1);
                uint32_t* table0 = mul0 + static_cast<size_t>(entry) * len;
                uint32_t* table1 = mul1 + static_cast<size_t>(entry) * len;
                cuda_check(cudaMemcpy(table0, r0, sizeof(uint32_t) * len, cudaMemcpyDeviceToDevice),
                           "check copy odd power0");
                cuda_check(cudaMemcpy(table1, r1, sizeof(uint32_t) * len, cudaMemcpyDeviceToDevice),
                           "check copy odd power1");
                ntt2_forward_dif_mont(table0, table1, log_len, fwd0, fwd1, offsets, stream);
            }
            cuda_check(cudaMemcpy(r0, h0.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice),
                       "check reset residue0");
            cuda_check(cudaMemcpy(r1, h1.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice),
                       "check reset residue1");
            cuda_check(cudaStreamSynchronize(stream), "check table setup synchronize");

            const uint64_t total_bits = static_cast<uint64_t>(
                boost::multiprecision::msb(check_exponent)) + 1;
            const bool use_cuda_graphs =
                std::getenv("GSRPS_DISABLE_CUDA_GRAPHS") == nullptr;
            constexpr int max_square_run_graph = 64;
            const int square_run_graph_count = static_cast<int>(
                std::min<uint64_t>(max_square_run_graph, total_bits));
            std::vector<cudaGraphExec_t> square_run_graph_execs(
                static_cast<size_t>(square_run_graph_count) + 1, nullptr);
            std::vector<cudaGraphExec_t> window_graph_execs(
                check_table_entries, nullptr);
            if (use_cuda_graphs) {
                std::vector<unsigned char> needed_square_graphs(
                    static_cast<size_t>(square_run_graph_count) + 1, 0);
                std::vector<unsigned char> needed_window_graphs(
                    static_cast<size_t>(check_table_entries), 0);
                int64_t scan_bit = static_cast<int64_t>(total_bits) - 1;
                while (scan_bit >= 0) {
                    if (!boost::multiprecision::bit_test(
                            check_exponent, static_cast<unsigned>(scan_bit))) {
                        int zero_run = 1;
                        while (scan_bit - zero_run >= 0 &&
                               !boost::multiprecision::bit_test(
                                   check_exponent,
                                   static_cast<unsigned>(scan_bit - zero_run))) {
                            ++zero_run;
                        }
                        int remaining = zero_run;
                        while (remaining > 0) {
                            const int chunk =
                                std::min(remaining, square_run_graph_count);
                            needed_square_graphs[static_cast<size_t>(chunk)] = 1;
                            remaining -= chunk;
                        }
                        scan_bit -= zero_run;
                    } else {
                        int64_t low = std::max<int64_t>(
                            0, scan_bit - check_window_bits + 1);
                        while (low < scan_bit &&
                               !boost::multiprecision::bit_test(
                                   check_exponent, static_cast<unsigned>(low))) {
                            ++low;
                        }
                        unsigned window_value = 0;
                        for (int64_t j = scan_bit; j >= low; --j) {
                            window_value = (window_value << 1) |
                                static_cast<unsigned>(
                                    boost::multiprecision::bit_test(
                                        check_exponent,
                                        static_cast<unsigned>(j)));
                        }
                        needed_window_graphs[
                            static_cast<size_t>((window_value - 1u) >> 1)] = 1;
                        scan_bit = low - 1;
                    }
                }
                iteration_graphs.reserve(
                    static_cast<size_t>(
                        check_table_entries + square_run_graph_count));
                iteration_graph_execs.reserve(
                    static_cast<size_t>(
                        check_table_entries + square_run_graph_count));
                const auto capture_iteration_graph =
                    [&](int square_iterations,
                        const uint32_t* cached0,
                        const uint32_t* cached1) {
                        cudaGraph_t graph = nullptr;
                        cudaGraphExec_t exec = nullptr;
                        cuda_check(cudaStreamBeginCapture(
                                       stream,
                                       cudaStreamCaptureModeThreadLocal),
                                   "check begin CUDA Graph capture");
                        for (int i = 0; i < square_iterations; ++i) {
                            iteration(nullptr, nullptr);
                        }
                        if (cached0 != nullptr) {
                            iteration(cached0, cached1);
                        }
                        cuda_check(cudaStreamEndCapture(
                                       stream, &graph),
                                   "check end CUDA Graph capture");
                        try {
                            cuda_check(cudaGraphInstantiate(
                                           &exec, graph, nullptr, nullptr, 0),
                                       "check instantiate CUDA Graph");
                        } catch (...) {
                            cudaGraphDestroy(graph);
                            throw;
                        }
                        iteration_graphs.push_back(graph);
                        iteration_graph_execs.push_back(exec);
                        return exec;
                    };
                for (int count = 1; count <= square_run_graph_count;
                     ++count) {
                    if (needed_square_graphs[static_cast<size_t>(count)]) {
                        square_run_graph_execs[count] =
                            capture_iteration_graph(count, nullptr, nullptr);
                    }
                }
                for (int entry = 0; entry < check_table_entries; ++entry) {
                    if (!needed_window_graphs[static_cast<size_t>(entry)]) {
                        continue;
                    }
                    unsigned window_value =
                        static_cast<unsigned>(entry * 2 + 1);
                    int window_length = 0;
                    for (unsigned value = window_value; value != 0;
                         value >>= 1) {
                        ++window_length;
                    }
                    window_graph_execs[entry] = capture_iteration_graph(
                        window_length,
                        mul0 + static_cast<size_t>(entry) * len,
                        mul1 + static_cast<size_t>(entry) * len);
                }
                std::cout << "cuda-graphs: enabled, batched_graphs="
                          << iteration_graph_execs.size()
                          << ", max_square_run=" << square_run_graph_count
                          << ", square_graph_types="
                          << std::count(needed_square_graphs.begin(),
                                        needed_square_graphs.end(), 1)
                          << ", window_graph_types="
                          << std::count(needed_window_graphs.begin(),
                                        needed_window_graphs.end(), 1)
                          << "\n";
            } else {
                std::cout << "cuda-graphs: disabled by "
                             "GSRPS_DISABLE_CUDA_GRAPHS\n";
            }
            const auto run_square_graph = [&](int count) {
                while (count > 0) {
                    const int chunk =
                        std::min(count, square_run_graph_count);
                    if (square_run_graph_execs[chunk] == nullptr) {
                        throw std::runtime_error(
                            "missing precomputed square-run CUDA Graph");
                    }
                    cuda_check(cudaGraphLaunch(
                                   square_run_graph_execs[chunk],
                                   stream),
                               "check launch square-run CUDA Graph");
                    count -= chunk;
                }
            };
            const auto run_window_graph = [&](size_t table_index) {
                if (table_index >= window_graph_execs.size() ||
                    window_graph_execs[table_index] == nullptr) {
                    throw std::runtime_error(
                        "missing precomputed window CUDA Graph");
                }
                cuda_check(cudaGraphLaunch(
                               window_graph_execs[table_index],
                               stream),
                           "check launch window CUDA Graph");
            };

            uint64_t diagnostic_stop_bits = 0;
            if (const char* stop_text = std::getenv("GSRPS_DIAGNOSTIC_STOP_BITS")) {
                diagnostic_stop_bits = parse_u64(stop_text);
                if (diagnostic_stop_bits == 0 || diagnostic_stop_bits > total_bits) {
                    throw std::runtime_error("GSRPS_DIAGNOSTIC_STOP_BITS is out of range");
                }
            }
#ifdef GSRPS_GMP_PREFIX_BITS
            const uint64_t progress_step = GSRPS_GMP_PREFIX_BITS;
#else
            const uint64_t progress_step = diagnostic_stop_bits != 0 ? diagnostic_stop_bits :
                std::max<uint64_t>(100000, (total_bits + 99) / 100);
#endif
            uint64_t next_progress = progress_step;
            uint64_t square_count = 0;
            uint64_t multiply_count = 0;
            int64_t bit = static_cast<int64_t>(total_bits) - 1;
#ifdef GSRPS_GMP_DIAGNOSTIC
            const mpz_class diagnostic_modulus(check_modulus.convert_to<std::string>());
            const mpz_class diagnostic_base(check_witness);
#endif
            const auto wall_started = std::chrono::steady_clock::now();
            auto duty_batch_started = wall_started;
            while (bit >= 0) {
                if (!boost::multiprecision::bit_test(check_exponent, static_cast<unsigned>(bit))) {
                    if (use_cuda_graphs) {
                        int zero_run = 1;
                        while (bit - zero_run >= 0 &&
                               !boost::multiprecision::bit_test(
                                   check_exponent,
                                   static_cast<unsigned>(bit - zero_run))) {
                            ++zero_run;
                        }
                        run_square_graph(zero_run);
                        square_count += static_cast<uint64_t>(zero_run);
                        bit -= zero_run;
                    } else {
                        iteration(nullptr, nullptr);
                        ++square_count;
                        --bit;
                    }
                } else {
                    int64_t low = std::max<int64_t>(0, bit - check_window_bits + 1);
                    while (low < bit &&
                           !boost::multiprecision::bit_test(check_exponent, static_cast<unsigned>(low))) {
                        ++low;
                    }
                    unsigned window_value = 0;
                    for (int64_t j = bit; j >= low; --j) {
                        window_value = (window_value << 1) |
                            static_cast<unsigned>(boost::multiprecision::bit_test(
                                check_exponent, static_cast<unsigned>(j)));
                    }
                    const int window_length = static_cast<int>(bit - low + 1);
                    const size_t table_index = (window_value - 1u) >> 1;
                    if (use_cuda_graphs) {
                        run_window_graph(table_index);
                        square_count +=
                            static_cast<uint64_t>(window_length);
                    } else {
                        for (int j = 0; j < window_length; ++j) {
                            iteration(nullptr, nullptr);
                            ++square_count;
                        }
                        iteration(mul0 + table_index * len,
                                  mul1 + table_index * len);
                    }
                    ++multiply_count;
                    bit = low - 1;
                }

                if (square_count >= next_progress || bit < 0) {
                    cuda_check(cudaStreamSynchronize(stream), "check progress synchronize");
                    throttle_after_gpu_iteration(duty_batch_started);
                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed = std::chrono::duration<double>(now - wall_started).count();
                    const double fraction = static_cast<double>(square_count) / static_cast<double>(total_bits);
                    const double eta = fraction > 0.0 ? elapsed * (1.0 - fraction) / fraction : 0.0;
                    std::cout << "progress: " << std::fixed << std::setprecision(2)
                              << fraction * 100.0 << "%, bits=" << square_count << "/" << total_bits
                              << ", multiplies=" << multiply_count
                              << ", elapsed_s=" << std::setprecision(1) << elapsed
                              << ", eta_s=" << eta << "\n";
#ifdef GSRPS_GMP_DIAGNOSTIC
                    const auto reference_started = std::chrono::steady_clock::now();
                    std::vector<int64_t> diagnostic_digits(canonical_count, 0);
                    cuda_check(cudaMemcpy(diagnostic_digits.data(), folded,
                                          sizeof(int64_t) * canonical_count,
                                          cudaMemcpyDeviceToHost),
                               "GMP diagnostic copy residue");
                    const auto gpu_mod = [&](uint64_t modulus) {
                        uint64_t value = 0;
                        for (int i = canonical_count - 1; i >= 0; --i) {
                            value = static_cast<uint64_t>((
                                boost::multiprecision::uint128_t(value) * (radix % modulus) +
                                static_cast<uint64_t>(diagnostic_digits[i])) % modulus);
                        }
                        return value;
                    };
                    const uint64_t remaining_bits = total_bits - square_count;
                    const boost::multiprecision::cpp_int prefix_exponent =
                        check_exponent >> remaining_bits;
                    const mpz_class diagnostic_exponent(prefix_exponent.convert_to<std::string>());
                    mpz_class diagnostic_reference;
                    mpz_powm(diagnostic_reference.get_mpz_t(), diagnostic_base.get_mpz_t(),
                             diagnostic_exponent.get_mpz_t(), diagnostic_modulus.get_mpz_t());
                    constexpr uint64_t diagnostic_q0 = 1000000007ull;
                    constexpr uint64_t diagnostic_q1 = 1000000009ull;
                    const uint64_t gpu_q0 = gpu_mod(diagnostic_q0);
                    const uint64_t gpu_q1 = gpu_mod(diagnostic_q1);
                    const uint64_t cpu_q0 = mpz_fdiv_ui(diagnostic_reference.get_mpz_t(), diagnostic_q0);
                    const uint64_t cpu_q1 = mpz_fdiv_ui(diagnostic_reference.get_mpz_t(), diagnostic_q1);
                    const double reference_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - reference_started).count();
                    std::cout << "gmp-prefix-check: bits=" << square_count
                              << ", reference_s=" << std::fixed << std::setprecision(3)
                              << reference_seconds << ", status="
                              << ((gpu_q0 == cpu_q0 && gpu_q1 == cpu_q1) ? "PASS" : "FAIL")
                              << "\n";
                    if (gpu_q0 != cpu_q0 || gpu_q1 != cpu_q1) {
                        throw std::runtime_error("GMP prefix mismatch at processed bit " +
                                                 std::to_string(square_count));
                    }
#ifdef GSRPS_GMP_STOP_AFTER_PREFIX
                    free_all();
                    return;
#endif
#endif
                    if (diagnostic_stop_bits != 0 && square_count >= diagnostic_stop_bits) {
                        std::vector<int64_t> prefix_digits(canonical_count, 0);
                        cuda_check(cudaMemcpy(prefix_digits.data(), folded,
                                              sizeof(int64_t) * canonical_count,
                                              cudaMemcpyDeviceToHost),
                                   "diagnostic prefix copy residue");
                        const auto prefix_mod = [&](uint64_t modulus) {
                            uint64_t value = 0;
                            for (int i = canonical_count - 1; i >= 0; --i) {
                                value = static_cast<uint64_t>((
                                    boost::multiprecision::uint128_t(value) * (radix % modulus) +
                                    static_cast<uint64_t>(prefix_digits[i])) % modulus);
                            }
                            return value;
                        };
                        std::cout << "gpu-prefix-checksum: bits=" << square_count
                                  << ", checksum=" << prefix_mod(1000000007ull)
                                  << ":" << prefix_mod(1000000009ull) << "\n";
                        free_all();
                        return;
                    }
                    next_progress = square_count + progress_step;
                    duty_batch_started = std::chrono::steady_clock::now();
                }
            }

            std::vector<int64_t> result(canonical_count, 0);
            cuda_check(cudaMemcpy(result.data(), folded, sizeof(int64_t) * canonical_count,
                                  cudaMemcpyDeviceToHost), "check copy final residue");
            const auto is_one = [&]() {
                if (result[0] != 1) return false;
                for (int i = 1; i < canonical_count; ++i) {
                    if (result[i] != 0) return false;
                }
                return true;
            };
            const auto at_least_modulus = [&]() {
                for (int i = canonical_count - 1; i >= 0; --i) {
                    if (result[i] != modulus_digits[i]) return result[i] > modulus_digits[i];
                }
                return true;
            };
            const bool raw_result_is_one = is_one();
            int host_canonical_subtractions = 0;
            while (at_least_modulus() && host_canonical_subtractions < 4) {
                int64_t borrow = 0;
                for (int i = 0; i < canonical_count; ++i) {
                    int64_t value = result[i] - modulus_digits[i] - borrow;
                    if (value < 0) {
                        value += static_cast<int64_t>(radix);
                        borrow = 1;
                    } else {
                        borrow = 0;
                    }
                    result[i] = value;
                }
                ++host_canonical_subtractions;
            }
            const bool result_is_one = is_one();
            const auto final_residue_mod = [&](uint64_t modulus) {
                uint64_t value = 0;
                for (int i = canonical_count - 1; i >= 0; --i) {
                    value = static_cast<uint64_t>((
                        boost::multiprecision::uint128_t(value) * (radix % modulus) +
                        static_cast<uint64_t>(result[i])) % modulus);
                }
                return value;
            };
            const uint64_t final_checksum0 = final_residue_mod(1000000007ull);
            const uint64_t final_checksum1 = final_residue_mod(1000000009ull);

            const auto exponentiation_finished = std::chrono::steady_clock::now();
            const bool verify_cpp_int = cpp_int_verification_enabled();
            double verification_seconds = 0.0;
            if (verify_cpp_int) {
                std::cout << "cpp-int-verification: started, exponent_bits="
                          << total_bits << " (this CPU verification may be slow)\n"
                          << std::flush;
                const auto verification_started = std::chrono::steady_clock::now();
                const boost::multiprecision::cpp_int reference = boost::multiprecision::powm(
                    boost::multiprecision::cpp_int(check_witness), check_exponent, check_modulus);
                const bool reference_matches = (reference == 1) == result_is_one;
                verification_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - verification_started).count();
                if (!reference_matches) throw std::runtime_error("complete check disagrees with cpp_int powm");
                std::cout << "cpp-int-verification: PASS, elapsed_s=" << std::fixed
                          << std::setprecision(3) << verification_seconds << "\n";
            }
            const auto check_finished = std::chrono::steady_clock::now();
            const double exponentiation_seconds = std::chrono::duration<double>(
                exponentiation_finished - wall_started).count();
            const double total_seconds = std::chrono::duration<double>(
                check_finished - complete_started).count();
            std::cout << "gsrps-check: N=" << k << "*" << b << "^" << n
                      << (c == 1 ? "+1" : "-1")
                      << ", witness=" << check_witness
                      << ", exponent_bits=" << total_bits
                      << ", squares=" << square_count
                      << ", cached_multiplies=" << multiply_count
                      << ", window_bits=" << check_window_bits
                      << ", exponentiation_seconds=" << std::fixed << std::setprecision(3)
                      << exponentiation_seconds
                      << ", total_seconds=" << total_seconds
                      << ", host_canonical_subtractions=" << host_canonical_subtractions
                      << ", raw_one=" << (raw_result_is_one ? "yes" : "no")
                      << ", checksum=" << final_checksum0 << ":" << final_checksum1
                      << ", verification=" << (verify_cpp_int ? "cpp_int-PASS" : "not-requested");
            if (verify_cpp_int) {
                std::cout << ", verification_seconds=" << verification_seconds;
            }
            std::cout
                      << ", result=" << (result_is_one ? "PRP" : "COMPOSITE") << "\n";
            free_all();
            return;
        }

        for (int i = 0; i < (validate_small ? 0 : 2); ++i) {
            iteration(multiply_mode ? mul0 : nullptr, multiply_mode ? mul1 : nullptr);
        }
        cuda_check(cudaStreamSynchronize(stream), "full warmup synchronize");
        cuda_check(cudaEventCreate(&start), "full create start");
        cuda_check(cudaEventCreate(&stop), "full create stop");
        cuda_check(cudaEventRecord(start), "full record start");
        for (int i = 0; i < iterations; ++i) {
            iteration(multiply_mode ? mul0 : nullptr, multiply_mode ? mul1 : nullptr);
        }
        cuda_check(cudaEventRecord(stop), "full record stop");
        cuda_check(cudaEventSynchronize(stop), "full stop synchronize");
        float elapsed_ms = 0;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "full elapsed");
        if (validate_small) {
            std::vector<int64_t> result(canonical_count);
            cuda_check(cudaMemcpy(result.data(), folded, sizeof(int64_t) * canonical_count, cudaMemcpyDeviceToHost), "full validation copy");
            if (result[active] != 0) throw std::runtime_error("canonical reduction left a nonzero guard limb");
            boost::multiprecision::cpp_int initial = 0, multiplier = 0, modulus = 0, got = 0;
            for (int i = active - 1; i >= 0; --i) {
                initial = initial * radix + initial_digits[i];
                multiplier = multiplier * radix + multiplier_digits[i];
                modulus = modulus * radix + modulus_digits[i];
                got = got * radix + result[i];
            }
            boost::multiprecision::cpp_int expected = initial;
            for (int i = 0; i < iterations; ++i) {
                expected = multiply_mode ? (expected * multiplier) % modulus : (expected * expected) % modulus;
            }
            if (got != expected) throw std::runtime_error("full modular product validation mismatch");
        }
        const double ms_per_square = static_cast<double>(elapsed_ms) / iterations;
        const double exponent_bits = static_cast<double>(n) * std::log2(static_cast<double>(b)) +
                                     std::log2(static_cast<double>(k));
        std::cout << "gsrps-full-bench: operation=" << (multiply_mode ? "mul-cached" : "square")
                  << ", k=" << k << ", b=" << b << ", n=" << n << ", c=" << c
                  << ", g=" << g << ", radix=" << radix << ", fold_divisor=" << divisor
                  << ", radix_policy=" << (radix_tier_upgrade ? "ntt-tier-upgrade" : "conservative")
                  << ", carry=exact-segment-" << kExactCarrySegment
                  << ", active_limbs=" << active << ", ntt_length=" << len
                  << ", ntt_blocks=" << ntt_block_limit() << ", iterations=" << iterations
                  << ", div_scan=" << (use_radix_multiple ? "none-radix-multiple" :
                                         (use_sum_scan ? "sum" : "affine"))
                  << ", total_ms_per_square=" << std::fixed << std::setprecision(6) << ms_per_square
                  << ", projected_seconds=" << std::setprecision(3) << ms_per_square * exponent_bits / 1000.0
                  << ", validation=" << (validate_small ? "PASS" : "not-requested") << "\n";
        free_all();
    } catch (...) {
        free_all();
        throw;
    }
}

void run_gsrps_check(uint64_t k, uint64_t b, uint64_t n, int c, uint64_t witness) {
    run_bench_gsrps_full(k, b, n, c, 1, false, true, witness);
}

void run_exact_carry_regression() {
    constexpr int len = 768;
    constexpr uint64_t radix = 10000;
    constexpr int segment_count =
        (len + kExactCarrySegment - 1) / kExactCarrySegment;
    const uint64_t reciprocal = UINT64_MAX / radix;
    const int exact_positions = radix_u64_exact_positions(radix);

    uint64_t map_state = 0x475352505343414eull;
    auto next_map_random = [&]() {
        map_state ^= map_state << 7;
        map_state ^= map_state >> 9;
        map_state ^= map_state << 8;
        return map_state;
    };
    for (int trial = 0; trial < 10000; ++trial) {
        const UnsignedCarryMap uf{
            next_map_random() % 100,
            1 + next_map_random() % 101};
        const UnsignedCarryMap ug{
            next_map_random() % 100,
            1 + next_map_random() % 101};
        const UnsignedCarryMap uh = UnsignedCarryCompose{}(uf, ug);
        for (uint64_t x : {uint64_t(0), uint64_t(1), uint64_t(50), uint64_t(100)}) {
            const auto eval_unsigned = [](const UnsignedCarryMap& map, uint64_t input) {
                return map.base + static_cast<uint64_t>(input >= map.threshold);
            };
            if (eval_unsigned(uh, x) != eval_unsigned(ug, eval_unsigned(uf, x))) {
                throw std::runtime_error("unsigned carry-map composition regression failed");
            }
        }

        int64_t ft1 = static_cast<int64_t>(next_map_random() % 21) - 10;
        int64_t ft2 = static_cast<int64_t>(next_map_random() % 21) - 10;
        if (ft1 > ft2) std::swap(ft1, ft2);
        int64_t gt1 = static_cast<int64_t>(next_map_random() % 21) - 10;
        int64_t gt2 = static_cast<int64_t>(next_map_random() % 21) - 10;
        if (gt1 > gt2) std::swap(gt1, gt2);
        const SignedCarryMap sf{
            static_cast<int64_t>(next_map_random() % 21) - 10, ft1, ft2};
        const SignedCarryMap sg{
            static_cast<int64_t>(next_map_random() % 21) - 10, gt1, gt2};
        const SignedCarryMap sh = SignedCarryCompose{}(sf, sg);
        for (int64_t x : {int64_t(-20), int64_t(-5), int64_t(0), int64_t(5), int64_t(20)}) {
            if (evaluate_signed_carry_map(sh, x) !=
                evaluate_signed_carry_map(sg, evaluate_signed_carry_map(sf, x))) {
                throw std::runtime_error("signed carry-map composition regression failed");
            }
        }
    }

    std::vector<uint64_t> unsigned_coeff(len, 0);
    unsigned_coeff[0] = radix;
    std::fill(unsigned_coeff.begin() + 1, unsigned_coeff.begin() + 600, radix - 1);
    std::vector<uint32_t> h0(len), h1(len);
    for (int i = 0; i < len; ++i) {
        h0[i] = static_cast<uint32_t>(unsigned_coeff[i] % kMods[0].p);
        h1[i] = static_cast<uint32_t>(unsigned_coeff[i] % kMods[1].p);
    }

    uint32_t *r0 = nullptr, *r1 = nullptr;
    int64_t *digits = nullptr;
    UnsignedCarryMap *unsigned_maps = nullptr, *unsigned_prefix = nullptr;
    void* unsigned_temp = nullptr;
    size_t unsigned_temp_bytes = 0;
    int64_t *product = nullptr, *modulus = nullptr;
    SignedCarryMap *signed_maps = nullptr, *signed_prefix = nullptr;
    void* signed_temp = nullptr;
    size_t signed_temp_bytes = 0;
    uint64_t *quotient = nullptr, *remainder = nullptr;
    auto cleanup = [&]() {
        cudaFree(signed_temp); cudaFree(signed_prefix); cudaFree(signed_maps);
        cudaFree(remainder); cudaFree(quotient);
        cudaFree(modulus); cudaFree(product);
        cudaFree(unsigned_temp);
        cudaFree(unsigned_prefix); cudaFree(unsigned_maps);
        cudaFree(digits); cudaFree(r1); cudaFree(r0);
    };

    try {
        cuda_check(cudaMalloc(&r0, sizeof(uint32_t) * len), "carry regression malloc r0");
        cuda_check(cudaMalloc(&r1, sizeof(uint32_t) * len), "carry regression malloc r1");
        cuda_check(cudaMalloc(&digits, sizeof(int64_t) * len), "carry regression malloc digits");
        cuda_check(cudaMalloc(&unsigned_maps, sizeof(UnsignedCarryMap) * segment_count),
                   "carry regression malloc unsigned maps");
        cuda_check(cudaMalloc(&unsigned_prefix, sizeof(UnsignedCarryMap) * segment_count),
                   "carry regression malloc unsigned prefix");
        const UnsignedCarryCompose unsigned_compose{};
        cuda_check(cub::DeviceScan::InclusiveScan(
                       nullptr, unsigned_temp_bytes, unsigned_maps, unsigned_prefix,
                       unsigned_compose, segment_count),
                   "carry regression query unsigned scan");
        cuda_check(cudaMalloc(&unsigned_temp, unsigned_temp_bytes),
                   "carry regression malloc unsigned scan");
        cuda_check(cudaMemcpy(r0, h0.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice),
                   "carry regression copy r0");
        cuda_check(cudaMemcpy(r1, h1.data(), sizeof(uint32_t) * len, cudaMemcpyHostToDevice),
                   "carry regression copy r1");

        crt2_unsigned_segment_prepare_kernel<<<1, segment_count>>>(
            r0, r1, digits, unsigned_maps,
            len, len, radix, reciprocal, exact_positions);
        cuda_check(cub::DeviceScan::InclusiveScan(
                       unsigned_temp, unsigned_temp_bytes, unsigned_maps, unsigned_prefix,
                       unsigned_compose, segment_count),
                   "carry regression unsigned scan");
        unsigned_segment_apply_kernel<<<1, segment_count>>>(
            digits, unsigned_prefix, len, radix, reciprocal);
        std::vector<int64_t> got_unsigned(len);
        cuda_check(cudaMemcpy(got_unsigned.data(), digits, sizeof(int64_t) * len,
                              cudaMemcpyDeviceToHost),
                   "carry regression copy unsigned result");

        std::vector<int64_t> expected_unsigned(len);
        uint64_t carry_unsigned = 0;
        for (int i = 0; i < len; ++i) {
            const uint64_t value = unsigned_coeff[i] + carry_unsigned;
            expected_unsigned[i] = static_cast<int64_t>(value % radix);
            carry_unsigned = value / radix;
        }
        if (got_unsigned != expected_unsigned || carry_unsigned != 0) {
            throw std::runtime_error("exact unsigned carry propagation regression failed");
        }

        std::vector<int64_t> hproduct(len, 0), hmodulus(len, 0);
        std::vector<uint64_t> hquotient(len, 0);
        hproduct[0] = -1;
        hproduct[600] = 1;
        const uint64_t zero_remainder = 0;
        cuda_check(cudaMalloc(&product, sizeof(int64_t) * len), "carry regression malloc product");
        cuda_check(cudaMalloc(&modulus, sizeof(int64_t) * len), "carry regression malloc modulus");
        cuda_check(cudaMalloc(&quotient, sizeof(uint64_t) * len), "carry regression malloc quotient");
        cuda_check(cudaMalloc(&remainder, sizeof(uint64_t)), "carry regression malloc remainder");
        cuda_check(cudaMalloc(&signed_maps, sizeof(SignedCarryMap) * segment_count),
                   "carry regression malloc signed maps");
        cuda_check(cudaMalloc(&signed_prefix, sizeof(SignedCarryMap) * segment_count),
                   "carry regression malloc signed prefix");
        const SignedCarryCompose signed_compose{};
        cuda_check(cub::DeviceScan::InclusiveScan(
                       nullptr, signed_temp_bytes, signed_maps, signed_prefix,
                       signed_compose, segment_count),
                   "carry regression query signed scan");
        cuda_check(cudaMalloc(&signed_temp, signed_temp_bytes),
                   "carry regression malloc signed scan");
        cuda_check(cudaMemcpy(product, hproduct.data(), sizeof(int64_t) * len,
                              cudaMemcpyHostToDevice),
                   "carry regression copy product");
        cuda_check(cudaMemcpy(modulus, hmodulus.data(), sizeof(int64_t) * len,
                              cudaMemcpyHostToDevice),
                   "carry regression copy modulus");
        cuda_check(cudaMemcpy(quotient, hquotient.data(), sizeof(uint64_t) * len,
                              cudaMemcpyHostToDevice),
                   "carry regression copy quotient");
        cuda_check(cudaMemcpy(remainder, &zero_remainder, sizeof(uint64_t),
                              cudaMemcpyHostToDevice),
                   "carry regression copy remainder");

        folded_signed_segment_prepare_kernel<<<1, segment_count>>>(
            product, quotient, remainder, modulus, digits,
            signed_maps,
            len, len, len, len, -1, radix, reciprocal, exact_positions);
        cuda_check(cub::DeviceScan::InclusiveScan(
                       signed_temp, signed_temp_bytes, signed_maps, signed_prefix,
                       signed_compose, segment_count),
                   "carry regression signed scan");
        signed_segment_apply_kernel<<<1, segment_count>>>(
            digits, signed_prefix, len, radix, reciprocal);
        std::vector<int64_t> got_signed(len);
        cuda_check(cudaMemcpy(got_signed.data(), digits, sizeof(int64_t) * len,
                              cudaMemcpyDeviceToHost),
                   "carry regression copy signed result");

        std::vector<int64_t> expected_signed(len);
        int64_t carry_signed = 0;
        for (int i = 0; i < len; ++i) {
            const int64_t value = hproduct[i] + carry_signed;
            int64_t q = value / static_cast<int64_t>(radix);
            int64_t rem = value % static_cast<int64_t>(radix);
            if (rem < 0) {
                rem += static_cast<int64_t>(radix);
                --q;
            }
            expected_signed[i] = rem;
            carry_signed = q;
        }
        if (got_signed != expected_signed || carry_signed != 0) {
            throw std::runtime_error("exact signed carry propagation regression failed");
        }
        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

void run_gsrps_selftest() {
    run_exact_carry_regression();
    run_bench_gsrps_full(4, 10, 2560, 1, 5, false);
    run_bench_gsrps_full(4, 10, 2560, -1, 5, true);
    // --selftest is itself an explicit request for validation. Keep the
    // independent CPU oracle for these tiny regression cases only.
    const bool saved_verify_cpp_int = gpu_throttle_config().verify_cpp_int;
    gpu_throttle_config().verify_cpp_int = true;
    run_gsrps_check(1, 2, 5, -1, 2);   // 31, prime
    run_gsrps_check(3, 10, 2, 1, 2);   // 301, composite
    run_gsrps_check(2, 3, 2, 1, 2);    // 19, prime; non-decimal base
    run_gsrps_check(4, 3, 4, 1, 2);    // 325, composite; non-decimal base
    gpu_throttle_config().verify_cpp_int = saved_verify_cpp_int;
    std::cout << "GSRPS selftest: PASS\n";
}

void display_banner() {
    printf("%s\n","════════════════════════════════════════════════════════════════════════════════════════");
    printf("%s\n","     .oooooo.         .oooooo..o     ooooooooo.       ooooooooo.        .oooooo..o      ");
    printf("%s\n","    d8P'  `Y8b       d8P'    `Y8     `888   `Y88.     `888   `Y88.     d8P'    `Y8      ");
    printf("%s\n","   888               Y88bo.           888   .d88'      888   .d88'     Y88bo.           ");
    printf("%s\n","   888                `'Y8888o.       888ooo88P'       888ooo88P'       `'Y8888o.       ");
    printf("%s\n","   888     ooooo          `'Y88b      888`88b.         888                  `'Y88b      ");
    printf("%s\n","   `88.    .88'  .o. oo     .d8P .o.  888  `88b.  .o.  888         .o. oo     .d8P .o.  ");
    printf("%s\n","    `Y8bood8P'   Y8P 8''88888P'  Y8P o888o  o888o Y8P o888o        Y8P 8''88888P'  Y8P  ");
    printf("%s\n","════════════════════════════════════════════════════════════════════════════════════════");
    printf("%s\n","                       Generalized-Sierpinski/Riesel-Prime-Seeker                       ");
    printf("%s\n","                           Version 2.0 CUDA by A.P. Sept 2026                           ");
}

void usage(const char* argv0) {
    display_banner();
    std::cout
        << "GSRPS commands:\n"
        << "  " << argv0 << " --check <k*b^n+/-1> [Fermat-witness, default 2]\n"
        << "  " << argv0 << " --check <k> <b> <n> <c:+1|-1> [Fermat-witness, default 2]\n"
        << "  " << argv0 << " --selftest\n"
        << "  " << argv0 << " --bench-square <k> <b> <n> <c:+1|-1> <iterations>\n"
        << "  " << argv0 << " --bench-mul <k> <b> <n> <c:+1|-1> <iterations>\n"
        << "global options (may appear anywhere):\n"
        << "  --force-ntt-blocks <1..4096>  disable auto-tuning and force the NTT block cap\n"
        << "  --force-window-bits <1..8>    disable automatic sliding-window selection\n"
        << "  --duty-percent <1..100>       full-check GPU duty cycle; default 100\n"
        << "  --verify-cpp-int               repeat --check independently with Boost cpp_int on CPU\n"
        << "  --tuning-cache-dir <path>      persistent NTT tuning cache; default .gsrps_tuning_cache\n"
        << "  --tuning-cache-max-age-hours N refresh cached tuning after N hours; default 24\n"
        << "  --no-tuning-cache              disable persistent NTT tuning cache\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<char*> filtered_argv = consume_gpu_throttle_args(argc, argv);
        argc = static_cast<int>(filtered_argv.size());
        argv = filtered_argv.data();
        print_gpu_throttle_config();
        if ((argc == 3 || argc == 4) && std::string(argv[1]) == "--check") {
            const GsrpsExpression expression = parse_gsrps_expression(argv[2]);
            run_gsrps_check(expression.k, expression.b, expression.n, expression.c,
                            argc == 4 ? parse_u64(argv[3]) : 2);
            return 0;
        }
        if ((argc == 6 || argc == 7) && std::string(argv[1]) == "--check") {
            run_gsrps_check(parse_u64(argv[2]), parse_u64(argv[3]), parse_u64(argv[4]),
                            std::stoi(argv[5]), argc == 7 ? parse_u64(argv[6]) : 2);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--selftest") {
            run_gsrps_selftest();
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--bench-square") {
            run_bench_gsrps_full(parse_u64(argv[2]), parse_u64(argv[3]), parse_u64(argv[4]),
                                 std::stoi(argv[5]), std::stoi(argv[6]));
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--bench-mul") {
            run_bench_gsrps_full(parse_u64(argv[2]), parse_u64(argv[3]), parse_u64(argv[4]),
                                 std::stoi(argv[5]), std::stoi(argv[6]), true);
            return 0;
        }
        usage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
