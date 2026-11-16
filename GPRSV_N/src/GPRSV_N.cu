/*
   GPRSV_N.cu

   SPDX-License-Identifier: GPL-2.0-or-later
   Copyright (C) 2026 AstralPrisma (A.P.).

   Fixed-k / variable-n CUDA siever for k*b^n +/- 1.

   GPRSV_K sieves a k range for one fixed exponent: for every prime p it
   computes one modular inverse of b^n and marks the residue classes of k.
   This program is the symmetric case: k is fixed and the exponent runs over
   an interval, so the residue of n for a given p is a discrete logarithm
   base b.  A direct n-walk costs one modular multiplication per (p, n) pair;
   the production path instead uses a baby-step/giant-step search costing
   about 2*sqrt(length) multiplications per prime, which is what makes long
   exponent intervals practical on a GPU.

   Derived from GPRSV (a CUDA port/reimplementation of mtsieve twinsieve),
   which is derived from GPLv2-or-later mtsieve code.  Distributed under the
   GNU General Public License, version 2 or later.
*/

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#if defined(__linux__)
#include <unistd.h>
#endif
#endif
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace gprn_cuda {
#ifdef _WIN32
static bool bulk_prime_copy = true;
#else
static bool bulk_prime_copy = false;
#endif
constexpr uint64_t KMAX_MAX = (UINT64_C(1) << 62);
constexpr uint64_t PMAX_MAX = (UINT64_C(1) << 62) - 1;
constexpr uint32_t BMAX_MAX = (UINT32_C(1) << 31);
constexpr uint64_t NMIN_MIN = 1;
constexpr uint64_t NMAX_MAX = (UINT64_C(1) << 31);
constexpr uint64_t LENGTH_MAX = (UINT64_C(1) << 31);
constexpr uint64_t SIGN_BIT = UINT64_C(1) << 63;
constexpr const char* APP_VERSION = "1.1";
enum class TermType : int { Unknown = 0, BN = 1 };
enum class FileFormat : int { Unknown = 0, Native, ABC };
enum class PrimeMode : int { Auto = 0, PrimeSieve, Segmented, MillerRabin };

struct Options {
    // Fixed coefficient and base; the exponent interval is [nmin, nmax].
    uint64_t k = 0;
    uint32_t base = 0;
    uint64_t nmin = 0;
    uint64_t nmax = 0;

    // Default is the independent sieve: +1 and -1 keep separate bitmaps, which
    // is what a fixed-k exponent search normally needs.  --paired selects the
    // twin sieve, where a candidate survives only while both signs are free of
    // tested factors.  For a twin search 3 must divide k when 3 does not divide
    // the base, otherwise p=3 eliminates every exponent (checked at startup).
    bool only_twins = false;
    bool plus_enabled = true;
    bool minus_enabled = true;

    uint64_t min_prime = 1;          // primes p satisfy min_prime < p
    uint64_t max_prime = PMAX_MAX;   // and p <= max_prime
    bool min_prime_explicit = false;
    bool max_prime_explicit = false;

    std::string input_terms;
    std::string output_terms;
    std::string input_factors;
    std::string output_factors;
    std::string expression_file;
    bool apply_and_exit = false;

    int device = 0;
    int threads = 256;
    int blocks = 0;                  // 0 = SM count * 8
    uint64_t batch_primes = UINT64_C(1) << 20;
    uint64_t cpu_small_prime = 100000;
    uint32_t segment_mib = 8;
    PrimeMode prime_mode = PrimeMode::Auto;
    uint64_t mr_switch_sqrt = 100000000;
    uint32_t prime_prefetch = 32;     // queued batch views generated ahead of the GPU
    uint32_t prime_threads = 0;       // 0 = auto (half logical CPUs, capped at 16)
    uint32_t prime_region_batches = 12;
    uint32_t cuda_streams = 2;
    uint32_t progress_seconds = 60;

    // 0 = walk, 1 = bsgs, 2 = auto (walk for very short intervals)
    int backend = 2;
    uint32_t bsgs_m = 0;              // 0 = automatic (power of two)

    bool verify_factors = false;
    bool quiet = false;

    // Lowest acceptable efficiency stopping (same contract as GPRSV_K).
    double max_factor_seconds = 0.0;
    double max_average_factor_seconds = 0.0;
    double efficiency_window_minutes = 0.0;
    bool max_factor_seconds_explicit = false;
    bool max_average_factor_seconds_explicit = false;
    bool efficiency_window_minutes_explicit = false;
    bool efficiency_enabled = false;

    bool help = false;
    bool version = false;
};

struct Problem {
    Options opt;
    uint64_t length = 0;        // nmax - nmin + 1
    uint64_t base_n = 0;        // window anchor: (nmin / m) * m
    uint64_t giants = 0;        // ceil((nmax - base_n + 1) / m)
    uint32_t m = 0;             // baby-step count (power of two)
    uint64_t q = 0;             // base_n / m
    uint64_t input_sieved_to = 1;

    std::vector<uint32_t> twin_bits;
    std::vector<uint32_t> minus_bits;
    std::vector<uint32_t> plus_bits;

    // Allocated only when -O/--outputfactors is used.  In twin mode bit 63
    // stores the sign (+1); a clear bit means -1.
    std::vector<uint64_t> twin_factor;
    std::vector<uint64_t> minus_factor;
    std::vector<uint64_t> plus_factor;
};

static volatile std::sig_atomic_t g_interrupted = 0;

static void handle_interrupt(int) {
    g_interrupted = 1;
}

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": " \
             << cudaGetErrorString(_e); \
        ::gprn_cuda::fail(_oss.str()); \
    } \
} while (0)
static std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static uint64_t parse_u64(const std::string& text, uint64_t lo, uint64_t hi, const char* what) {
    std::string s = trim(text);
    if (s.empty()) fail(std::string("Missing value for ") + what);

    auto invalid = [&]() -> void {
        fail(std::string("Invalid integer for ") + what + ": " + text);
    };
    auto overflow = [&]() -> void {
        fail(std::string("Integer overflow for ") + what + ": " + text);
    };

    uint64_t mult = 1;
    bool has_suffix = false;
    char last = static_cast<char>(std::tolower(static_cast<unsigned char>(s.back())));
    if (last == 'k' || last == 'm' || last == 'g' || last == 't') {
        has_suffix = true;
        s.pop_back();
        if (s.empty()) invalid();
        if (last == 'k') mult = UINT64_C(1000);
        if (last == 'm') mult = UINT64_C(1000000);
        if (last == 'g') mult = UINT64_C(1000000000);
        if (last == 't') mult = UINT64_C(1000000000000);
    }

    uint64_t raw = 0;
    const bool is_hex = s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
    const size_t epos = is_hex ? std::string::npos : s.find_first_of("eE");

    if (epos != std::string::npos) {
        // Scientific notation is intentionally integer-only: 1e15 and 10E+12
        // are accepted, while 1.5e15, 1e-3, and suffix combinations are not.
        if (has_suffix || epos == 0 || s.find_first_of("eE", epos + 1) != std::string::npos)
            invalid();

        const std::string mantissa = s.substr(0, epos);
        std::string exponent = s.substr(epos + 1);
        if (exponent.empty()) invalid();
        if (exponent[0] == '+') exponent.erase(exponent.begin());
        if (exponent.empty() || exponent[0] == '-') invalid();

        for (unsigned char c : mantissa) {
            if (!std::isdigit(c)) invalid();
            const uint32_t digit = static_cast<uint32_t>(c - '0');
            if (raw > (std::numeric_limits<uint64_t>::max() - digit) / 10)
                overflow();
            raw = raw * 10 + digit;
        }

        uint64_t exp10 = 0;
        for (unsigned char c : exponent) {
            if (!std::isdigit(c)) invalid();
            const uint32_t digit = static_cast<uint32_t>(c - '0');
            // Any exponent above 19 overflows every nonzero uint64 mantissa.
            // Keep parsing exact, but cap early to avoid exponent overflow.
            if (exp10 > 19 || (exp10 == 19 && digit > 9)) overflow();
            exp10 = exp10 * 10 + digit;
        }

        if (raw != 0) {
            for (uint64_t i = 0; i < exp10; ++i) {
                if (raw > std::numeric_limits<uint64_t>::max() / 10)
                    overflow();
                raw *= 10;
            }
        }
    } else {
        size_t pos = 0;
        unsigned long long parsed = 0;
        try {
            parsed = std::stoull(s, &pos, 0);
        } catch (...) {
            invalid();
        }
        if (pos != s.size()) invalid();
        raw = static_cast<uint64_t>(parsed);
    }

    if (raw > std::numeric_limits<uint64_t>::max() / mult)
        overflow();
    uint64_t value = raw * mult;
    if (value < lo || value > hi) {
        std::ostringstream oss;
        oss << what << " out of range [" << lo << ", " << hi << "]: " << value;
        fail(oss.str());
    }
    return value;
}

static int parse_int(const std::string& s, int lo, int hi, const char* what) {
    uint64_t v = parse_u64(s, static_cast<uint64_t>(lo), static_cast<uint64_t>(hi), what);
    return static_cast<int>(v);
}

static double parse_positive_double(const std::string& text, const char* what) {
    std::string s = trim(text);
    if (s.empty()) fail(std::string("Missing value for ") + what);
    size_t pos = 0;
    double value = 0.0;
    try {
        value = std::stod(s, &pos);
    } catch (...) {
        fail(std::string("Invalid number for ") + what + ": " + text);
    }
    if (pos != s.size() || !std::isfinite(value) || value <= 0.0)
        fail(std::string("Invalid positive number for ") + what + ": " + text);
    return value;
}

static std::string option_value(int& i, int argc, char** argv, const std::string& arg) {
    size_t eq = arg.find('=');
    if (eq != std::string::npos) return arg.substr(eq + 1);
    if (i + 1 >= argc) fail("Missing value after " + arg);
    return argv[++i];
}
static void parse_options(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { o.help = true; continue; }
        if (a == "--version") { o.version = true; continue; }
        if (a == "-s" || a == "--independent") { o.only_twins = false; continue; }
        if (a == "--paired" || a == "--twin") { o.only_twins = true; continue; }
        if (a == "-A" || a == "--applyandexit") { o.apply_and_exit = true; continue; }
        if (a == "--verify") { o.verify_factors = true; continue; }
        if (a == "--quiet") { o.quiet = true; continue; }

        auto is_long = [&](const char* name) {
            std::string n(name);
            return a == n || a.rfind(n + "=", 0) == 0;
        };
        auto short_value = [&](char c) -> std::string {
            if (a.size() > 2) return a.substr(2);
            if (i + 1 >= argc) fail(std::string("Missing value after -") + c);
            return argv[++i];
        };

        if (a.rfind("-k", 0) == 0 && a.rfind("--", 0) != 0) {
            o.k = parse_u64(short_value('k'), 1, KMAX_MAX, "k");
        } else if (is_long("--k")) {
            o.k = parse_u64(option_value(i, argc, argv, a), 1, KMAX_MAX, "k");
        } else if (a.rfind("-K", 0) == 0 && a.rfind("--", 0) != 0) {
            fail("-K is the k range end of GPRSV_K; GPRSV_N takes one fixed -k "
                 "(the exponent interval is -n N1 -N N2)");
        } else if (is_long("--kmax") || is_long("--kmin")) {
            fail("GPRSV_N has no k range; use -k K (the exponent interval is -n N1 -N N2)");
        } else if (a.rfind("-b", 0) == 0 && a.rfind("--", 0) != 0) {
            o.base = static_cast<uint32_t>(parse_u64(short_value('b'), 2, BMAX_MAX, "base"));
        } else if (is_long("--base")) {
            o.base = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, a), 2, BMAX_MAX, "base"));
        } else if (a.rfind("-n", 0) == 0 && a.rfind("--", 0) != 0) {
            o.nmin = parse_u64(short_value('n'), NMIN_MIN, NMAX_MAX, "nmin");
        } else if (is_long("--nmin")) {
            o.nmin = parse_u64(option_value(i, argc, argv, a), NMIN_MIN, NMAX_MAX, "nmin");
        } else if (a.rfind("-N", 0) == 0 && a.rfind("--", 0) != 0) {
            o.nmax = parse_u64(short_value('N'), NMIN_MIN, NMAX_MAX, "nmax");
        } else if (is_long("--nmax")) {
            o.nmax = parse_u64(option_value(i, argc, argv, a), NMIN_MIN, NMAX_MAX, "nmax");
        } else if (is_long("--sign")) {
            std::string v = option_value(i, argc, argv, a);
            if (v == "both") { o.plus_enabled = true; o.minus_enabled = true; }
            else if (v == "plus" || v == "+" || v == "+1") { o.plus_enabled = true; o.minus_enabled = false; }
            else if (v == "minus" || v == "-" || v == "-1") { o.plus_enabled = false; o.minus_enabled = true; }
            else fail("--sign must be both, plus, or minus");
        } else if (a.rfind("-t", 0) == 0 && a.rfind("--", 0) != 0) {
            fail("GPRSV_N supports only k*b^n+/-1; use GPRSV_K for k*n!+/-1 and k*n#+/-1");
        } else if (is_long("--termtype")) {
            fail("GPRSV_N supports only k*b^n+/-1; use GPRSV_K for k*n!+/-1 and k*n#+/-1");
        } else if (a.rfind("-p", 0) == 0 && a.rfind("--", 0) != 0) {
            o.min_prime = parse_u64(short_value('p'), 1, PMAX_MAX, "pmin");
            o.min_prime_explicit = true;
        } else if (is_long("--pmin")) {
            o.min_prime = parse_u64(option_value(i, argc, argv, a), 1, PMAX_MAX, "pmin");
            o.min_prime_explicit = true;
        } else if (a.rfind("-P", 0) == 0 && a.rfind("--", 0) != 0) {
            o.max_prime = parse_u64(short_value('P'), 2, PMAX_MAX, "pmax");
            o.max_prime_explicit = true;
        } else if (is_long("--pmax")) {
            o.max_prime = parse_u64(option_value(i, argc, argv, a), 2, PMAX_MAX, "pmax");
            o.max_prime_explicit = true;
        } else if (a.rfind("-w", 0) == 0 && a.rfind("--", 0) != 0) {
            o.batch_primes = parse_u64(short_value('w'), 1024, UINT64_C(1) << 30, "worksize");
        } else if (is_long("--worksize") || is_long("--batch-primes")) {
            o.batch_primes = parse_u64(option_value(i, argc, argv, a), 1024, UINT64_C(1) << 30, "batch-primes");
        } else if (a.rfind("-i", 0) == 0 && a.rfind("--", 0) != 0) {
            o.input_terms = short_value('i');
        } else if (is_long("--inputterms")) {
            o.input_terms = option_value(i, argc, argv, a);
        } else if (a.rfind("-o", 0) == 0 && a.rfind("--", 0) != 0) {
            o.output_terms = short_value('o');
        } else if (is_long("--outputterms")) {
            o.output_terms = option_value(i, argc, argv, a);
        } else if (a.rfind("-I", 0) == 0 && a.rfind("--", 0) != 0) {
            o.input_factors = short_value('I');
        } else if (is_long("--inputfactors")) {
            o.input_factors = option_value(i, argc, argv, a);
        } else if (a.rfind("-O", 0) == 0 && a.rfind("--", 0) != 0) {
            o.output_factors = short_value('O');
        } else if (is_long("--outputfactors")) {
            o.output_factors = option_value(i, argc, argv, a);
        } else if (a.rfind("-E", 0) == 0 && a.rfind("--", 0) != 0) {
            o.expression_file = short_value('E');
        } else if (is_long("--expressions") || is_long("--pfgw")) {
            o.expression_file = option_value(i, argc, argv, a);
        } else if (is_long("--device")) {
            o.device = parse_int(option_value(i, argc, argv, a), 0, 255, "device");
        } else if (is_long("--threads")) {
            o.threads = parse_int(option_value(i, argc, argv, a), 32, 1024, "threads");
        } else if (is_long("--blocks") || is_long("--gpuworkgroups")) {
            o.blocks = parse_int(option_value(i, argc, argv, a), 0, 1000000, "blocks");
        } else if (is_long("--cpu-small-prime")) {
            o.cpu_small_prime = parse_u64(option_value(i, argc, argv, a), 0, PMAX_MAX, "cpu-small-prime");
        } else if (is_long("--segment-mib")) {
            o.segment_mib = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, a), 1, 2048, "segment-mib"));
        } else if (is_long("--mr-switch-sqrt")) {
            o.mr_switch_sqrt = parse_u64(option_value(i, argc, argv, a), 1000, UINT32_MAX, "mr-switch-sqrt");
        } else if (is_long("--prime-generator")) {
            std::string v = option_value(i, argc, argv, a);
            if (v == "auto") o.prime_mode = PrimeMode::Auto;
            else if (v == "primesieve" || v == "ps") o.prime_mode = PrimeMode::PrimeSieve;
            else if (v == "segmented") o.prime_mode = PrimeMode::Segmented;
            else if (v == "mr") o.prime_mode = PrimeMode::MillerRabin;
            else fail("prime-generator must be auto, primesieve, segmented, or mr");
        } else if (is_long("--prime-copy")) {
            const auto value = option_value(i, argc, argv, a);
            if (value != "bulk" && value != "scalar") fail("prime-copy must be bulk or scalar");
            bulk_prime_copy = value == "bulk";
        } else if (is_long("--prime-prefetch")) {
            o.prime_prefetch = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, a), 1, 64, "prime-prefetch"));
        } else if (is_long("--prime-region-batches")) {
            o.prime_region_batches = static_cast<uint32_t>(parse_u64(
                option_value(i, argc, argv, a), 1, 256, "prime-region-batches"));
        } else if (is_long("--cuda-streams")) {
            o.cuda_streams = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, a), 1, 4, "cuda-streams"));
        } else if (is_long("--prime-threads")) {
            o.prime_threads = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, a), 0, 64, "prime-threads"));
        } else if (a == "--workers" || a.rfind("--workers=", 0) == 0) {
            o.prime_threads = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, a), 0, 64, "workers"));
        } else if (is_long("--progress-seconds") || is_long("--report-seconds")) {
            o.progress_seconds = static_cast<uint32_t>(parse_u64(
                option_value(i, argc, argv, a), 1, 86400, "progress-seconds"));
        } else if (is_long("--backend")) {
            std::string v = option_value(i, argc, argv, a);
            if (v == "auto") o.backend = 2;
            else if (v == "walk" || v == "direct") o.backend = 0;
            else if (v == "bsgs" || v == "baby-step") o.backend = 1;
            else fail("--backend must be auto, walk, or bsgs");
        } else if (is_long("--bsgs-m")) {
            o.bsgs_m = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, a), 16, 4096, "bsgs-m"));
        } else if (is_long("--max-factor-seconds") || is_long("--max-no-factor-seconds")) {
            o.max_factor_seconds = parse_positive_double(
                option_value(i, argc, argv, a), "max-factor-seconds");
            o.max_factor_seconds_explicit = true;
        } else if (is_long("--max-average-factor-seconds") ||
                   is_long("--max-average-seconds-per-factor") ||
                   is_long("--spftarget")) {
            o.max_average_factor_seconds = parse_positive_double(
                option_value(i, argc, argv, a), "max-average-factor-seconds");
            o.max_average_factor_seconds_explicit = true;
        } else if (a.rfind("-5", 0) == 0 && a.rfind("--", 0) != 0) {
            o.max_average_factor_seconds = parse_positive_double(
                short_value('5'), "max-average-factor-seconds");
            o.max_average_factor_seconds_explicit = true;
        } else if (is_long("--efficiency-window-minutes") ||
                   is_long("--minutesforspf")) {
            o.efficiency_window_minutes = parse_positive_double(
                option_value(i, argc, argv, a), "efficiency-window-minutes");
            o.efficiency_window_minutes_explicit = true;
        } else if (a.rfind("-6", 0) == 0 && a.rfind("--", 0) != 0) {
            o.efficiency_window_minutes = parse_positive_double(
                short_value('6'), "efficiency-window-minutes");
            o.efficiency_window_minutes_explicit = true;
        } else if (a.rfind("-W", 0) == 0 && a.rfind("--", 0) != 0) {
            (void)short_value('W'); // accepted for command-line compatibility
        } else if (a.rfind("-4", 0) == 0 && a.rfind("--", 0) != 0) {
            o.max_factor_seconds = parse_positive_double(
                short_value('4'), "max-factor-seconds");
            o.max_factor_seconds_explicit = true;
        } else if (is_long("--fpstarget")) {
            (void)option_value(i, argc, argv, a);
        } else {
            fail("Unknown option: " + a);
        }
    }

    if (o.threads % 32 != 0) fail("--threads must be a multiple of 32");

    const int efficiency_values =
        static_cast<int>(o.max_factor_seconds_explicit) +
        static_cast<int>(o.max_average_factor_seconds_explicit) +
        static_cast<int>(o.efficiency_window_minutes_explicit);
    if (efficiency_values != 0 && efficiency_values != 3) {
        fail("Lowest acceptable efficiency requires all three options: "
             "--max-factor-seconds, --max-average-factor-seconds, and "
             "--efficiency-window-minutes");
    }
    if (efficiency_values == 3) {
        o.efficiency_enabled = true;
        o.max_prime = PMAX_MAX;
        o.max_prime_explicit = true;
    }
    if (!o.only_twins && !o.plus_enabled && !o.minus_enabled)
        fail("--sign leaves no sign to sieve");
    if (o.only_twins && !o.plus_enabled)
        fail("--sign is only valid in independent mode (the default); --paired always tests both signs");
    if (o.only_twins && !o.minus_enabled)
        fail("--sign is only valid in independent mode (the default); --paired always tests both signs");
}

static void print_help() {
    std::cout
        << "GPRSV_N v" << APP_VERSION << "\n"
        << "GPU sieve for k*b^n+/-1 with fixed k and a variable exponent interval.\n\n"
        << "Core options:\n"
        << "  -k, --k K                  fixed k\n"
        << "  -b, --base B               base b\n"
        << "  -n, --nmin N1              first exponent (inclusive)\n"
        << "  -N, --nmax N2              last exponent (inclusive)\n"
        << "  -s, --independent          sieve +1 and -1 independently (default)\n"
        << "      --paired, --twin       paired/twin sieve: keep n only while both signs survive\n"
        << "      --sign both|plus|minus with -s: which signs to keep (default both)\n"
        << "  -p, --pmin P0              test primes p > P0\n"
        << "  -P, --pmax P1              test primes p <= P1 (e.g. 1e11)\n"
        << "  -i, --inputterms FILE      resume from a GPRSV_N output file\n"
        << "  -o, --outputterms FILE     write remaining n values\n"
        << "  -I, --inputfactors FILE    apply existing 'p | term' factors\n"
        << "  -O, --outputfactors FILE   append newly found factors\n"
        << "  -E, --expressions FILE     also write LLR/PFGW-ready k*b^n+/-1 lines\n"
        << "  -A, --applyandexit         apply -I and write output without sieving\n\n"
        << "Algorithm:\n"
        << "      --backend M            auto, walk (one multiply per n), or bsgs\n"
        << "      --bsgs-m M             baby steps, power of two 16-4096 (0=auto)\n\n"
        << "CUDA options:\n"
        << "      --device D             CUDA device index (default 0)\n"
        << "      --threads T            threads per block (default 256)\n"
        << "      --blocks B             grid blocks; 0=8 per SM\n"
        << "  -w, --batch-primes N       primes copied per launch (default 1048576)\n"
        << "      --cpu-small-prime P    CPU handles p <= P (default 100000)\n"
        << "      --prime-generator M    auto, primesieve, segmented, or mr\n"
        << "      --prime-copy M         bulk or scalar (Windows default bulk, Linux default scalar)\n"
        << "      --prime-prefetch N     queued pinned batch views (default 32)\n"
        << "      --prime-threads N      libprimesieve iterators / built-in workers; 0=auto\n"
        << "      --prime-region-batches N  batches per libprimesieve initialization (default 12)\n"
        << "      --cuda-streams N       pinned CUDA streams/buffers, 1-4 (default 2)\n"
        << "      --progress-seconds N   full progress/ETA line interval (default 60)\n"
        << "      --segment-mib M        MiB per built-in worker segment (default 8)\n"
        << "      --verify               verify every reported factor on CPU\n"
        << "      --quiet                reduce progress output\n\n"
        << "Lowest acceptable efficiency (supply all three):\n"
        << "  -4, --max-factor-seconds S stop after S seconds with no newly removed term\n"
        << "  -5, --max-average-factor-seconds S\n"
        << "  -6, --efficiency-window-minutes M\n"
        << "  Limits are evaluated after completed batches; already submitted CUDA batches are drained.\n"
        << "  Enabling this system forces pmax to " << PMAX_MAX << " (2^62-1).\n\n"
        << "Examples:\n"
        << "  GPRSV_N -k 12345 -b 2 -n 100000 -N 200000 -P 1e11\n"
        << "  GPRSV_N -k 12345 -b 2 -n 100000 -N 200000 -s --sign minus -P 1e11\n"
        << "  GPRSV_N -i k12345_b2_n100000-200000.txt -P 1e12 -o next.txt\n";
}
// ------------------------- bit set helpers -------------------------

static uint64_t words_for_bits(uint64_t bits) { return (bits + 31) >> 5; }

static bool bit_test(const std::vector<uint32_t>& bits, uint64_t idx) {
    return (bits[static_cast<size_t>(idx >> 5)] >> (idx & 31)) & 1U;
}

static bool bit_clear(std::vector<uint32_t>& bits, uint64_t idx) {
    uint32_t& w = bits[static_cast<size_t>(idx >> 5)];
    uint32_t mask = UINT32_C(1) << (idx & 31);
    bool was = (w & mask) != 0;
    w &= ~mask;
    return was;
}

static void bit_set(std::vector<uint32_t>& bits, uint64_t idx) {
    bits[static_cast<size_t>(idx >> 5)] |= UINT32_C(1) << (idx & 31);
}

static void fill_bits(std::vector<uint32_t>& bits, uint64_t count, bool value) {
    bits.assign(static_cast<size_t>(words_for_bits(count)), value ? UINT32_MAX : 0U);
    if (value && (count & 31)) {
        bits.back() &= (UINT32_C(1) << (count & 31)) - 1U;
    }
}

static uint64_t count_bits(const std::vector<uint32_t>& bits) {
    uint64_t n = 0;
    for (uint32_t x : bits) {
#if defined(_MSC_VER)
        n += static_cast<uint64_t>(__popcnt(x));
#else
        n += static_cast<uint64_t>(__builtin_popcount(x));
#endif
    }
    return n;
}
// ------------------------- exact host modular arithmetic -------------------------

[[maybe_unused]] static uint64_t add_mod_host(uint64_t a, uint64_t b, uint64_t m) {
    return (a >= m - b) ? (a - (m - b)) : (a + b);
}

static uint64_t mul_mod_host(uint64_t a, uint64_t b, uint64_t m) {
#if defined(_MSC_VER) && defined(_M_X64)
    uint64_t hi = 0;
    uint64_t lo = _umul128(a, b, &hi);
    uint64_t rem = 0;
    (void)_udiv128(hi, lo, m, &rem);
    return rem;
#elif defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) % m);
#else
    uint64_t r = 0;
    a %= m;
    while (b) {
        if (b & 1) r = add_mod_host(r, a, m);
        b >>= 1;
        if (b) a = add_mod_host(a, a, m);
    }
    return r;
#endif
}

static uint64_t pow_mod_host(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = mul_mod_host(r, a, m);
        e >>= 1;
        if (e) a = mul_mod_host(a, a, m);
    }
    return r;
}

static uint64_t inverse_mod_host(uint64_t a, uint64_t p) {
    // p is prime in all sieve uses. Fermat avoids signed-coefficient overflow.
    if (a % p == 0) return 0;
    return pow_mod_host(a, p - 2, p);
}
// ------------------------- problem model helpers -------------------------

static std::string term_text(const Problem& p, uint64_t n, int c) {
    std::ostringstream oss;
    oss << p.opt.k << '*' << p.opt.base << '^' << n << (c > 0 ? "+1" : "-1");
    return oss.str();
}

static std::string multiplier_text(const Problem& p) {
    return std::to_string(p.opt.k) + '*' + std::to_string(p.opt.base) + "^$a";
}

static uint64_t active_terms(const Problem& p) {
    if (p.opt.only_twins) return count_bits(p.twin_bits);
    uint64_t total = 0;
    if (p.opt.minus_enabled) total += count_bits(p.minus_bits);
    if (p.opt.plus_enabled) total += count_bits(p.plus_bits);
    return total;
}

static bool n_to_index(const Problem& p, uint64_t n, uint64_t& idx) {
    if (n < p.opt.nmin || n > p.opt.nmax) return false;
    idx = n - p.opt.nmin;
    return true;
}

static uint64_t index_to_n(const Problem& p, uint64_t idx) {
    return p.opt.nmin + idx;
}

// True when the tested term k*b^n+/-1 is itself equal to the prime p.  Such a
// term must not be removed: p divides it only because it is the term.
static bool term_equals_prime_host(const Problem& p, uint64_t prime, uint64_t n, int c) {
    const uint64_t limit = (c > 0) ? prime - 1 : prime + 1; // k*b^n must equal this
    if (limit < p.opt.k) return false;
    uint64_t v = p.opt.k;
    for (uint64_t i = 0; i < n; ++i) {
        if (v > limit / p.opt.base) return false;
        v *= p.opt.base;
        if (v > limit) return false;
    }
    return v == limit;
}

static bool factor_is_valid(const Problem& pr, uint64_t factor, uint64_t n, int c) {
    if (factor < 2 || factor > PMAX_MAX) return false;
    const uint64_t base = pr.opt.base;
    const uint64_t k = pr.opt.k;
    if (base % factor == 0 || (factor <= k && k % factor == 0)) return false;
    uint64_t r = mul_mod_host(k % factor, pow_mod_host(base % factor, n, factor), factor);
    if (c > 0) return r == factor - 1;
    return r == 1;
}
static uint64_t isqrt_u64(uint64_t x) {
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<long double>(x)));
    while (r > 0 && r > x / r) --r;
    while (r + 1 <= x / (r + 1)) ++r;
    return r;
}

static bool is_prime_mr(uint64_t n) {
    if (n < 2) return false;
    static constexpr uint32_t small_primes[] = {2,3,5,7,11,13,17,19,23,29,31,37};
    for (uint32_t p : small_primes) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    static constexpr uint64_t bases[] = {
        2, 325, 9375, 28178, 450775, 9780504, 1795265022
    };
    for (uint64_t a : bases) {
        if (a % n == 0) continue;
        uint64_t x = pow_mod_host(a % n, d, n);
        if (x == 1 || x == n - 1) continue;
        bool witness = true;
        for (unsigned r = 1; r < s; ++r) {
            x = mul_mod_host(x, x, n);
            if (x == n - 1) { witness = false; break; }
        }
        if (witness) return false;
    }
    return true;
}

static std::vector<uint32_t> simple_primes_upto(uint32_t limit) {
    if (limit < 2) return {};
    std::vector<bool> composite(static_cast<size_t>(limit) + 1, false);
    for (uint64_t p = 2; p * p <= limit; ++p)
        if (!composite[static_cast<size_t>(p)])
            for (uint64_t j = p * p; j <= limit; j += p)
                composite[static_cast<size_t>(j)] = true;
    std::vector<uint32_t> primes;
    for (uint32_t i = 2; i <= limit; ++i)
        if (!composite[i]) primes.push_back(i);
    return primes;
}

static std::vector<uint32_t> segmented_base_primes(uint32_t limit, uint32_t segment_mib) {
    if (limit <= 10000000U) return simple_primes_upto(limit);
    uint32_t root = static_cast<uint32_t>(isqrt_u64(limit));
    std::vector<uint32_t> small_primes = simple_primes_upto(root);
    std::vector<uint32_t> out;
    if (limit >= 2) out.push_back(2);

    uint64_t odd_slots = std::max<uint64_t>(UINT64_C(1) << 20,
        static_cast<uint64_t>(segment_mib) * 1024 * 1024);
    uint64_t low = 3;
    while (low <= limit) {
        if ((low & 1) == 0) ++low;
        uint64_t high = std::min<uint64_t>(limit, low + 2 * odd_slots - 2);
        if ((high & 1) == 0) --high;
        size_t count = static_cast<size_t>((high - low) / 2 + 1);
        std::vector<uint8_t> composite(count, 0);
        for (uint32_t p : small_primes) {
            if (p == 2) continue;
            uint64_t pp = static_cast<uint64_t>(p) * p;
            if (pp > high) break;
            uint64_t start = std::max<uint64_t>(pp, ((low + p - 1) / p) * p);
            if ((start & 1) == 0) start += p;
            for (uint64_t j = start; j <= high; j += 2ULL * p)
                composite[static_cast<size_t>((j - low) / 2)] = 1;
        }
        for (size_t i = 0; i < count; ++i)
            if (!composite[i]) out.push_back(static_cast<uint32_t>(low + 2ULL * i));
        low = high + 2;
    }
    return out;
}
// Optional runtime bridge to the same highly optimized primesieve iterator used
// by mtsieve.  It is loaded dynamically so GPRSV remains one .cu source file
// and still compiles without primesieve headers or link flags.
struct PrimeSieveIteratorAbi {
    size_t i;
    size_t size;
    uint64_t start;
    uint64_t stop_hint;
    uint64_t* primes;
    void* memory;
    int is_error;
};

class PrimeSieveRuntime {
public:
    using InitFn = void (*)(PrimeSieveIteratorAbi*);
    using FreeFn = void (*)(PrimeSieveIteratorAbi*);
    using JumpFn = void (*)(PrimeSieveIteratorAbi*, uint64_t, uint64_t);
    using GenerateFn = void (*)(PrimeSieveIteratorAbi*);

    PrimeSieveRuntime() { load(); }
    PrimeSieveRuntime(const PrimeSieveRuntime&) = delete;
    PrimeSieveRuntime& operator=(const PrimeSieveRuntime&) = delete;
    ~PrimeSieveRuntime() { close(); }

    bool available() const {
        return handle_ && init_ && free_ && generate_ && (jump_ || skip_);
    }

    const std::string& loaded_name() const { return loaded_name_; }
    const std::string& error_text() const { return error_text_; }

    void init_empty_iterator(PrimeSieveIteratorAbi* it) const {
        std::memset(it, 0, sizeof(*it));
        init_(it);
    }

    void jump_iterator(PrimeSieveIteratorAbi* it, uint64_t pmin, uint64_t pmax) const {
        if (jump_) {
            // jump_to() includes its start value; pmin is exclusive.
            jump_(it, pmin + 1, pmax);
        } else {
            // Older primesieve versions provide skipto(), which excludes start.
            skip_(it, pmin, pmax);
        }
    }

    void init_iterator(PrimeSieveIteratorAbi* it, uint64_t pmin, uint64_t pmax) const {
        init_empty_iterator(it);
        jump_iterator(it, pmin, pmax);
    }

    void free_iterator(PrimeSieveIteratorAbi* it) const { free_(it); }

    uint64_t next_prime(PrimeSieveIteratorAbi* it) const {
        ++it->i;
        if (it->i >= it->size) generate_(it);
        if (it->is_error || !it->primes || it->i >= it->size)
            fail("libprimesieve iterator failed while generating primes");
        return it->primes[it->i];
    }

    std::pair<const uint64_t*, size_t> next_span(PrimeSieveIteratorAbi* it) const {
        ++it->i;
        if (it->i >= it->size) generate_(it);
        if (it->is_error || !it->primes || it->i >= it->size)
            fail("libprimesieve span generation failed");
        const auto* begin = it->primes + it->i;
        const size_t count = it->size - it->i;
        it->i = it->size - 1;
        return {begin, count};
    }

private:
#if defined(_WIN32)
    using Handle = HMODULE;
#else
    using Handle = void*;
#endif

    template <typename T>
    T symbol(const char* name) {
#if defined(_WIN32)
        return reinterpret_cast<T>(GetProcAddress(handle_, name));
#else
        return reinterpret_cast<T>(dlsym(handle_, name));
#endif
    }

    bool try_load(const std::string& name) {
#if defined(_WIN32)
        handle_ = LoadLibraryA(name.c_str());
#else
        dlerror();
        handle_ = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle_) {
            // Preserve a useful ABI rejection rather than replacing it with
            // an error for an optional library name that simply is absent.
            if (error_text_.empty()) error_text_ = name + " could not be loaded";
            return false;
        }
        using VersionFn = const char* (*)();
        auto version_fn = symbol<VersionFn>("primesieve_version");
        const char* version = version_fn ? version_fn() : nullptr;
        if (!version || std::string(version).rfind("12.", 0) != 0) {
            error_text_ = name + " has incompatible iterator ABI (version " +
                          (version ? version : "unknown") + "); expected primesieve 12.x";
            close();
            return false;
        }
        init_ = symbol<InitFn>("primesieve_init");
        free_ = symbol<FreeFn>("primesieve_free_iterator");
        jump_ = symbol<JumpFn>("primesieve_jump_to");
        skip_ = symbol<JumpFn>("primesieve_skipto");
        generate_ = symbol<GenerateFn>("primesieve_generate_next_primes");
        if (!available()) {
            error_text_ = name + " is missing the primesieve 12.x iterator API";
            close();
            return false;
        }
        loaded_name_ = name + " (v" + version + ")";
        error_text_.clear();
        return true;
    }

    void load() {
#if defined(__linux__)
        // Resolve relative to the ELF, not the current working directory or
        // the login shell's environment. Prefer our verified compatibility
        // library even when Conda or system packages provide another version.
        char executable[4096];
        const auto length = readlink("/proc/self/exe", executable, sizeof(executable));
        if (length > 0 && static_cast<size_t>(length) < sizeof(executable)) {
            const std::string path(executable, static_cast<size_t>(length));
            const auto slash = path.rfind('/');
            if (slash != std::string::npos) {
                const auto bundled = path.substr(0, slash) + "/lib/libprimesieve.so.12";
                if (access(bundled.c_str(), F_OK) == 0) {
                    if (!try_load(bundled)) fail("Bundled primesieve library: " + error_text_);
                    return;
                }
            }
        }
#endif
#if defined(_WIN32)
        const char* names[] = {"primesieve.dll", "libprimesieve.dll"};
#elif defined(__APPLE__)
        const char* names[] = {"libprimesieve.12.dylib", "libprimesieve.dylib"};
#else
        const char* names[] = {"libprimesieve.so.12", "libprimesieve.so"};
#endif
        for (const char* name : names) if (try_load(name)) return;
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
        init_ = nullptr;
        free_ = nullptr;
        jump_ = nullptr;
        skip_ = nullptr;
        generate_ = nullptr;
    }

    Handle handle_ = nullptr;
    InitFn init_ = nullptr;
    FreeFn free_ = nullptr;
    JumpFn jump_ = nullptr;
    JumpFn skip_ = nullptr;
    GenerateFn generate_ = nullptr;
    std::string loaded_name_;
    std::string error_text_;
};


struct PrimeBatch {
    std::shared_ptr<void> owner;
    const uint64_t* data = nullptr;
    size_t count = 0;
    bool pinned = false;

    void clear() {
        owner.reset();
        data = nullptr;
        count = 0;
        pinned = false;
    }
    bool empty() const { return count == 0; }
    uint64_t first() const {
        if (empty()) fail("Internal empty prime batch");
        return data[0];
    }
    uint64_t last() const {
        if (empty()) fail("Internal empty prime batch");
        return data[count - 1];
    }
};

struct PinnedPrimeRegion {
    uint64_t* ptr = nullptr;
    size_t capacity = 0;
    size_t count = 0;

    PinnedPrimeRegion() = default;
    PinnedPrimeRegion(const PinnedPrimeRegion&) = delete;
    PinnedPrimeRegion& operator=(const PinnedPrimeRegion&) = delete;
    ~PinnedPrimeRegion() {
        if (ptr) cudaFreeHost(ptr);
    }
};

class PinnedPrimeRegionPool : public std::enable_shared_from_this<PinnedPrimeRegionPool> {
public:
    PinnedPrimeRegionPool(size_t max_buffers, size_t initial_capacity)
        : max_buffers_(std::max<size_t>(1, max_buffers)),
          initial_capacity_(std::max<size_t>(1024, initial_capacity)) {}

    PinnedPrimeRegionPool(const PinnedPrimeRegionPool&) = delete;
    PinnedPrimeRegionPool& operator=(const PinnedPrimeRegionPool&) = delete;

    std::shared_ptr<PinnedPrimeRegion> acquire() {
        std::unique_ptr<PinnedPrimeRegion> region;
        bool create_new = false;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] {
                return stopped_ || !free_.empty() || created_ < max_buffers_;
            });
            if (stopped_) return {};
            if (!free_.empty()) {
                region = std::move(free_.back());
                free_.pop_back();
            } else {
                ++created_;
                create_new = true;
            }
        }

        if (create_new) {
            try {
                region = std::make_unique<PinnedPrimeRegion>();
                allocate_initial(*region);
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    --created_;
                }
                cv_.notify_one();
                throw;
            }
        }

        region->count = 0;
        PinnedPrimeRegion* raw = region.release();
        auto self = shared_from_this();
        return std::shared_ptr<PinnedPrimeRegion>(
            raw, [self](PinnedPrimeRegion* p) {
                self->release(std::unique_ptr<PinnedPrimeRegion>(p));
            });
    }

    void grow(PinnedPrimeRegion& region, size_t needed) {
        if (needed <= region.capacity) return;
        size_t grown = region.capacity + region.capacity / 2 + 4096;
        if (grown < needed) grown = needed;
        if (grown > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
            fail("Pinned prime-region capacity overflow");

        uint64_t* replacement = nullptr;
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&replacement),
                                 grown * sizeof(uint64_t),
                                 cudaHostAllocDefault));
        std::memcpy(replacement, region.ptr, region.count * sizeof(uint64_t));
        CUDA_CHECK(cudaFreeHost(region.ptr));
        size_t old_capacity = region.capacity;
        region.ptr = replacement;
        region.capacity = grown;

        {
            std::lock_guard<std::mutex> lock(mu_);
            allocated_bytes_ += (grown - old_capacity) * sizeof(uint64_t);
            ++growths_;
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

    size_t created() const {
        std::lock_guard<std::mutex> lock(mu_);
        return created_;
    }
    size_t allocated_bytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return allocated_bytes_;
    }
    size_t growths() const {
        std::lock_guard<std::mutex> lock(mu_);
        return growths_;
    }
    size_t max_buffers() const { return max_buffers_; }
    size_t initial_capacity() const { return initial_capacity_; }

private:
    void allocate_initial(PinnedPrimeRegion& region) {
        if (initial_capacity_ > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
            fail("Pinned prime-region allocation overflow");
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&region.ptr),
                                 initial_capacity_ * sizeof(uint64_t),
                                 cudaHostAllocDefault));
        region.capacity = initial_capacity_;
        {
            std::lock_guard<std::mutex> lock(mu_);
            allocated_bytes_ += initial_capacity_ * sizeof(uint64_t);
        }
    }

    void release(std::unique_ptr<PinnedPrimeRegion> region) {
        region->count = 0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!stopped_) {
                free_.push_back(std::move(region));
                cv_.notify_one();
                return;
            }
        }
        // If the pool is stopping, destroy outside the mutex.
    }

    const size_t max_buffers_;
    const size_t initial_capacity_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<PinnedPrimeRegion>> free_;
    size_t created_ = 0;
    size_t allocated_bytes_ = 0;
    size_t growths_ = 0;
    bool stopped_ = false;
};

class PrimeStream {
public:
    PrimeStream(uint64_t pmin, uint64_t pmax, uint64_t batch_size,
                PrimeMode requested, uint64_t mr_switch_sqrt, uint32_t segment_mib,
                uint32_t requested_prime_threads, uint32_t prime_region_batches,
                uint32_t prime_prefetch, uint32_t cuda_streams,
                bool quiet)
        : pmin_(pmin), pmax_(pmax), batch_size_(batch_size),
          segment_mib_(segment_mib),
          ps_region_batches_(std::max<uint32_t>(1, prime_region_batches)),
          prime_prefetch_(std::max<uint32_t>(1, prime_prefetch)),
          cuda_streams_(std::max<uint32_t>(1, cuda_streams)),
          quiet_(quiet) {
        uint64_t root = isqrt_u64(pmax_);
        unsigned hardware_threads = std::thread::hardware_concurrency();
        if (hardware_threads == 0) hardware_threads = 1;

        if (requested == PrimeMode::Auto || requested == PrimeMode::PrimeSieve) {
            ps_ = std::make_unique<PrimeSieveRuntime>();
            if (ps_->available()) {
                mode_ = PrimeMode::PrimeSieve;
                if (requested_prime_threads == 0) {
                    uint32_t half = hardware_threads >= 4
                        ? static_cast<uint32_t>(hardware_threads / 2)
                        : static_cast<uint32_t>(hardware_threads);
                    prime_threads_ = std::max<uint32_t>(1, std::min<uint32_t>(16, half));
                } else {
                    prime_threads_ = requested_prime_threads;
                }
                ps_assign_low_ = pmin_ + 1;

                // Each iterator owns one coarse region.  Regions are filled
                // directly into page-locked memory, so libprimesieve output
                // can be copied to the device without the old vector->pinned
                // staging memcpy.  Additional pool entries cover batches
                // already queued for the consumer and CUDA streams whose H2D
                // copies are still in flight.
                ps_max_outstanding_regions_ = std::max<size_t>(1, prime_threads_);
                size_t queued_regions =
                    (static_cast<size_t>(prime_prefetch_) + ps_region_batches_ - 1) /
                    ps_region_batches_;
                size_t max_pinned_regions = ps_max_outstanding_regions_ +
                    queued_regions + static_cast<size_t>(cuda_streams_) + 1;

                long double cap_ld =
                    static_cast<long double>(batch_size_) *
                    static_cast<long double>(ps_region_batches_) +
                    static_cast<long double>(batch_size_) +
                    65536.0L;
                if (cap_ld > static_cast<long double>(std::numeric_limits<size_t>::max()))
                    fail("Pinned prime-region capacity is too large");
                size_t initial_capacity = static_cast<size_t>(cap_ld);
                ps_pinned_pool_ = std::make_shared<PinnedPrimeRegionPool>(
                    max_pinned_regions, initial_capacity);
                start_primesieve_workers();
                return;
            }
            if (requested == PrimeMode::PrimeSieve) {
                fail("--prime-generator primesieve requested, but libprimesieve could not be loaded: " +
                     ps_->error_text() +
                     ". On Ubuntu/WSL install it with: sudo apt install libprimesieve-dev");
            }
            if (!quiet_) {
                std::cout << "libprimesieve not found; using the slower built-in generator. "
                          << "Install libprimesieve-dev to match mtsieve prime generation speed.\n";
            }
            ps_.reset();
        }

        mode_ = requested;
        if (mode_ == PrimeMode::Auto)
            mode_ = (root <= mr_switch_sqrt) ? PrimeMode::Segmented : PrimeMode::MillerRabin;

        if (requested_prime_threads == 0) {
            prime_threads_ = std::max<uint32_t>(1, std::min<uint32_t>(16, hardware_threads));
        } else {
            prime_threads_ = requested_prime_threads;
        }

        if (mode_ == PrimeMode::Segmented) {
            auto t0 = std::chrono::steady_clock::now();
            if (!quiet_) std::cout << "Generating base primes through " << root << "..." << std::flush;
            base_ = segmented_base_primes(static_cast<uint32_t>(root), segment_mib_);
            if (!quiet_) {
                double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::cout << " done (" << base_.size() << " primes, "
                          << std::fixed << std::setprecision(2) << sec << "s)\n";
            }
            next_segment_low_ = pmin_ + 1;
        } else {
            presieve_ = simple_primes_upto(1000000);
            next_segment_low_ = pmin_ + 1;
        }
    }

    ~PrimeStream() {
        stop_primesieve_workers();
    }

    PrimeMode mode() const { return mode_; }

    size_t pinned_region_buffers() const {
        return ps_pinned_pool_ ? ps_pinned_pool_->created() : 0;
    }
    size_t pinned_region_pool_limit() const {
        return ps_pinned_pool_ ? ps_pinned_pool_->max_buffers() : 0;
    }
    size_t pinned_region_bytes() const {
        return ps_pinned_pool_ ? ps_pinned_pool_->allocated_bytes() : 0;
    }
    size_t pinned_region_growths() const {
        return ps_pinned_pool_ ? ps_pinned_pool_->growths() : 0;
    }

    std::string description() const {
        if (mode_ == PrimeMode::PrimeSieve) {
            std::ostringstream oss;
            oss << "parallel libprimesieve (" << prime_threads_ << " iterator"
                << (prime_threads_ == 1 ? "" : "s") << ", "
                << ps_region_batches_ << " batches/region, direct pinned regions, "
                << ps_->loaded_name() << ")";
            return oss.str();
        }
        std::ostringstream oss;
        if (mode_ == PrimeMode::Segmented)
            oss << "built-in parallel segmented Eratosthenes";
        else
            oss << "built-in parallel pre-sieve + deterministic Miller-Rabin";
        oss << " (" << prime_threads_ << " thread" << (prime_threads_ == 1 ? "" : "s")
            << ", " << segment_mib_ << " MiB/segment)";
        return oss.str();
    }

    bool next(PrimeBatch& batch) {
        batch.clear();
        if (mode_ == PrimeMode::PrimeSieve)
            return next_primesieve_batch(batch);

        auto storage = std::make_shared<std::vector<uint64_t>>();
        storage->reserve(static_cast<size_t>(batch_size_));
        while (storage->size() < batch_size_) {
            if (segment_pos_ < segment_primes_.size()) {
                size_t n = std::min<size_t>(segment_primes_.size() - segment_pos_,
                                            static_cast<size_t>(batch_size_ - storage->size()));
                storage->insert(storage->end(),
                                segment_primes_.begin() + static_cast<std::ptrdiff_t>(segment_pos_),
                                segment_primes_.begin() + static_cast<std::ptrdiff_t>(segment_pos_ + n));
                segment_pos_ += n;
                continue;
            }
            if (next_segment_low_ > pmax_) break;
            generate_segment_wave();
        }
        if (storage->empty()) return false;
        batch.owner = storage;
        batch.data = storage->data();
        batch.count = storage->size();
        batch.pinned = false;
        return true;
    }

private:
    struct PsRegion {
        uint64_t id = 0;
        uint64_t low = 0;
        uint64_t high = 0;
    };

    uint64_t estimated_region_span(uint64_t low) const {
        long double batches = static_cast<long double>(ps_region_batches_);
        long double target_primes = static_cast<long double>(
            std::max<uint64_t>(UINT64_C(65536), batch_size_)) * batches;
        long double logx = std::log(static_cast<long double>(std::max<uint64_t>(low, 3)));
        long double span_ld = target_primes * std::max<long double>(3.0L, logx - 1.0L) * 1.03L;
        long double max_u64 = static_cast<long double>(std::numeric_limits<uint64_t>::max());
        if (span_ld >= max_u64) return std::numeric_limits<uint64_t>::max();
        uint64_t span = static_cast<uint64_t>(std::ceil(span_ld));
        return std::max<uint64_t>(span, UINT64_C(1) << 20);
    }

    bool assign_primesieve_region_locked(PsRegion& region) {
        if (ps_assign_low_ > pmax_) return false;
        region.id = ps_next_assign_id_++;
        region.low = ps_assign_low_;
        uint64_t span = estimated_region_span(region.low);
        uint64_t remaining = pmax_ - region.low + 1;
        if (span > remaining) span = remaining;
        region.high = region.low + span - 1;
        ps_assign_low_ = (region.high == std::numeric_limits<uint64_t>::max())
            ? region.high : region.high + 1;
        ++ps_outstanding_regions_;
        return true;
    }

    void start_primesieve_workers() {
        ps_workers_.reserve(prime_threads_);
        for (uint32_t i = 0; i < prime_threads_; ++i)
            ps_workers_.emplace_back(&PrimeStream::primesieve_worker, this);
    }

    void stop_primesieve_workers() {
        if (ps_workers_.empty()) return;
        {
            std::lock_guard<std::mutex> lock(ps_mu_);
            ps_stop_ = true;
        }
        if (ps_pinned_pool_) ps_pinned_pool_->stop();
        ps_cv_work_.notify_all();
        ps_cv_ready_.notify_all();
        for (std::thread& worker : ps_workers_)
            if (worker.joinable()) worker.join();
        ps_workers_.clear();
    }

    void primesieve_worker() {
        PrimeSieveIteratorAbi it{};
        bool initialized = false;
        try {
            ps_->init_empty_iterator(&it);
            initialized = true;
            for (;;) {
                PsRegion region;
                {
                    std::unique_lock<std::mutex> lock(ps_mu_);
                    ps_cv_work_.wait(lock, [&] {
                        return ps_stop_ || ps_error_ ||
                               (ps_assign_low_ <= pmax_ &&
                                ps_outstanding_regions_ < ps_max_outstanding_regions_);
                    });
                    if (ps_stop_ || ps_error_) break;
                    if (!assign_primesieve_region_locked(region)) continue;
                }

                std::shared_ptr<PinnedPrimeRegion> result =
                    ps_pinned_pool_->acquire();
                if (!result) break;

                // One jump per coarse region, not one jump per CUDA batch.
                // libprimesieve now writes directly into page-locked memory.
                // The main thread later submits spans of this same allocation
                // to cudaMemcpyAsync, eliminating vector->pinned staging.
                ps_->jump_iterator(&it, region.low - 1, region.high);
                if (bulk_prime_copy) {
                    for (;;) {
                        const auto span = ps_->next_span(&it);
                        const auto* end = span.first + span.second;
                        if (end[-1] == std::numeric_limits<uint64_t>::max())
                            fail("libprimesieve returned PRIMESIEVE_ERROR");
                        const auto* first = std::lower_bound(span.first, end, region.low);
                        const auto* stop = std::upper_bound(first, end, region.high);
                        const size_t count = static_cast<size_t>(stop-first);
                        if (count > std::numeric_limits<size_t>::max()-result->count)
                            fail("prime span overflow");
                        if (result->count+count > result->capacity)
                            ps_pinned_pool_->grow(*result, result->count+count);
                        if (count) std::memcpy(result->ptr+result->count, first, count*sizeof(uint64_t));
                        result->count += count;
                        if (stop != end) break;
                    }
                } else {
                for (;;) {
                    uint64_t q = ps_->next_prime(&it);
                    if (q == std::numeric_limits<uint64_t>::max())
                        fail("libprimesieve returned PRIMESIEVE_ERROR");
                    if (q > region.high) break;
                    if (q < region.low) continue;
                    if (result->count == result->capacity)
                        ps_pinned_pool_->grow(*result, result->count + 1);
                    result->ptr[result->count++] = q;
                }
                }

                {
                    std::lock_guard<std::mutex> lock(ps_mu_);
                    ps_completed_regions_.emplace(region.id, std::move(result));
                }
                ps_cv_ready_.notify_all();
                // A completed region still counts as outstanding until every
                // batch in that region has been handed to the consumer.  This
                // bounds memory while keeping one region of look-ahead.
                ps_cv_work_.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(ps_mu_);
            if (!ps_error_) ps_error_ = std::current_exception();
            ps_cv_ready_.notify_all();
            ps_cv_work_.notify_all();
        }
        if (initialized) ps_->free_iterator(&it);
    }

    void finish_current_primesieve_region() {
        if (!ps_current_region_active_) return;
        {
            std::lock_guard<std::mutex> lock(ps_mu_);
            if (ps_outstanding_regions_ == 0)
                fail("Internal libprimesieve region accounting underflow");
            --ps_outstanding_regions_;
            ps_current_region_active_ = false;
        }
        ps_current_region_.reset();
        ps_current_region_offset_ = 0;
        ps_cv_work_.notify_all();
    }

    bool next_primesieve_batch(PrimeBatch& batch) {
        batch.clear();
        for (;;) {
            if (ps_current_region_) {
                if (ps_current_region_offset_ < ps_current_region_->count) {
                    size_t remaining =
                        ps_current_region_->count - ps_current_region_offset_;
                    size_t n = std::min<size_t>(
                        remaining, static_cast<size_t>(batch_size_));
                    batch.owner = ps_current_region_;
                    batch.data = ps_current_region_->ptr + ps_current_region_offset_;
                    batch.count = n;
                    batch.pinned = true;
                    ps_current_region_offset_ += n;
                    if (ps_current_region_offset_ == ps_current_region_->count)
                        finish_current_primesieve_region();
                    return true;
                }
                finish_current_primesieve_region();
                continue;
            }

            std::unique_lock<std::mutex> lock(ps_mu_);
            ps_cv_ready_.wait(lock, [&] {
                return ps_error_ || ps_stop_ ||
                       ps_completed_regions_.find(ps_next_consume_id_) != ps_completed_regions_.end() ||
                       (ps_assign_low_ > pmax_ && ps_outstanding_regions_ == 0);
            });
            if (ps_error_) std::rethrow_exception(ps_error_);

            auto it = ps_completed_regions_.find(ps_next_consume_id_);
            if (it == ps_completed_regions_.end()) {
                if (ps_assign_low_ > pmax_ && ps_outstanding_regions_ == 0)
                    return false;
                continue;
            }

            ps_current_region_ = std::move(it->second);
            ps_completed_regions_.erase(it);
            ++ps_next_consume_id_;
            ps_current_region_offset_ = 0;
            ps_current_region_active_ = true;
            lock.unlock();

            if (!ps_current_region_ || ps_current_region_->count == 0) {
                finish_current_primesieve_region();
                continue;
            }
        }
    }

    void generate_one_segment(uint64_t raw_low,
                              uint64_t raw_high,
                              std::vector<uint64_t>& out) const {
        out.clear();
        uint64_t low = raw_low;
        uint64_t high = raw_high;

        if (low <= 2 && 2 <= high && pmin_ < 2) out.push_back(2);
        if (low < 3) low = 3;
        if ((low & 1) == 0) ++low;
        if ((high & 1) == 0) --high;
        if (low > high) return;

        size_t count = static_cast<size_t>((high - low) / 2 + 1);
        std::vector<uint8_t> composite(count, 0);
        const std::vector<uint32_t>& marking =
            (mode_ == PrimeMode::Segmented) ? base_ : presieve_;

        for (uint32_t p : marking) {
            if (p == 2) continue;
            uint64_t pp = static_cast<uint64_t>(p) * p;
            if (mode_ == PrimeMode::Segmented && pp > high) break;

            uint64_t rem = low % p;
            uint64_t first = rem == 0 ? low : low + (p - rem);
            if (first < pp) first = pp;
            if ((first & 1) == 0) first += p;
            if (first > high) continue;

            const uint64_t step = 2ULL * p;
            for (uint64_t j = first; j <= high; j += step)
                composite[static_cast<size_t>((j - low) / 2)] = 1;
        }

        long double estimate = static_cast<long double>(high - low + 1) /
            std::max<long double>(2.0L, std::log(static_cast<long double>(std::max<uint64_t>(low, 3))));
        out.reserve(static_cast<size_t>(estimate * 1.15L) + 32);
        for (size_t i = 0; i < count; ++i) {
            if (composite[i]) continue;
            uint64_t q = low + 2ULL * i;
            if (q <= pmin_ || q > pmax_) continue;
            if (mode_ == PrimeMode::Segmented || is_prime_mr(q))
                out.push_back(q);
        }
    }

    void generate_segment_wave() {
        segment_primes_.clear();
        segment_pos_ = 0;
        if (next_segment_low_ > pmax_) return;

        const uint64_t odd_slots = std::max<uint64_t>(UINT64_C(1) << 18,
            static_cast<uint64_t>(segment_mib_) * 1024 * 1024);
        const uint64_t span = 2 * odd_slots;
        const uint64_t remaining_span = pmax_ - next_segment_low_;
        const uint64_t segments_left = remaining_span / span + 1;
        const uint32_t wave_segments = static_cast<uint32_t>(
            std::min<uint64_t>(prime_threads_, segments_left));

        struct Range { uint64_t low, high; };
        std::vector<Range> ranges;
        ranges.reserve(wave_segments);
        for (uint32_t i = 0; i < wave_segments; ++i) {
            uint64_t low = next_segment_low_;
            uint64_t high = low + std::min<uint64_t>(span - 1, pmax_ - low);
            ranges.push_back({low, high});
            next_segment_low_ = (high == std::numeric_limits<uint64_t>::max())
                ? high : high + 1;
        }

        std::vector<std::vector<uint64_t>> results(wave_segments);
        if (wave_segments == 1) {
            generate_one_segment(ranges[0].low, ranges[0].high, results[0]);
        } else {
            std::atomic<uint32_t> next_job{0};
            const uint32_t worker_count = std::min<uint32_t>(prime_threads_, wave_segments);
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (uint32_t t = 0; t < worker_count; ++t) {
                workers.emplace_back([&, this] {
                    for (;;) {
                        uint32_t job = next_job.fetch_add(1, std::memory_order_relaxed);
                        if (job >= wave_segments) break;
                        generate_one_segment(ranges[job].low, ranges[job].high, results[job]);
                    }
                });
            }
            for (std::thread& worker : workers) worker.join();
        }

        size_t total = 0;
        for (const auto& result : results) total += result.size();
        segment_primes_.reserve(total);
        for (auto& result : results) {
            segment_primes_.insert(segment_primes_.end(),
                                   std::make_move_iterator(result.begin()),
                                   std::make_move_iterator(result.end()));
        }
    }


    uint64_t pmin_, pmax_, batch_size_;
    uint32_t segment_mib_;
    uint32_t prime_threads_ = 1;
    uint32_t ps_region_batches_ = 12;
    uint32_t prime_prefetch_ = 32;
    uint32_t cuda_streams_ = 2;
    bool quiet_;
    PrimeMode mode_ = PrimeMode::Auto;

    std::unique_ptr<PrimeSieveRuntime> ps_;
    std::shared_ptr<PinnedPrimeRegionPool> ps_pinned_pool_;
    uint64_t ps_assign_low_ = 0;
    uint64_t ps_next_assign_id_ = 0;
    uint64_t ps_next_consume_id_ = 0;
    size_t ps_outstanding_regions_ = 0;
    size_t ps_max_outstanding_regions_ = 0;
    bool ps_current_region_active_ = false;
    std::shared_ptr<PinnedPrimeRegion> ps_current_region_;
    size_t ps_current_region_offset_ = 0;
    std::map<uint64_t, std::shared_ptr<PinnedPrimeRegion>> ps_completed_regions_;
    std::mutex ps_mu_;
    std::condition_variable ps_cv_work_, ps_cv_ready_;
    bool ps_stop_ = false;
    std::exception_ptr ps_error_;
    std::vector<std::thread> ps_workers_;

    std::vector<uint32_t> base_, presieve_;
    uint64_t next_segment_low_ = 0;
    std::vector<uint64_t> segment_primes_;
    size_t segment_pos_ = 0;
};

// A bounded producer queue keeps prime generation one or more batches ahead
// while the main thread runs the CUDA kernel.  This fixes the old strict
// CPU-generate -> GPU-synchronize -> CPU-generate serialization.
class PrimeBatchPipeline {
public:
    PrimeBatchPipeline(PrimeStream& stream, uint32_t depth)
        : stream_(stream), depth_(std::max<uint32_t>(1, depth)), worker_(&PrimeBatchPipeline::produce, this) {}

    PrimeBatchPipeline(const PrimeBatchPipeline&) = delete;
    PrimeBatchPipeline& operator=(const PrimeBatchPipeline&) = delete;

    ~PrimeBatchPipeline() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_not_full_.notify_all();
        cv_not_empty_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    bool next(PrimeBatch& out, double& wait_seconds) {
        auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mu_);
        cv_not_empty_.wait(lock, [&] { return !queue_.empty() || done_ || error_; });
        wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (error_) std::rethrow_exception(error_);
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop_front();
        lock.unlock();
        cv_not_full_.notify_one();
        return true;
    }

private:
    void produce() {
        try {
            while (!g_interrupted) {
                PrimeBatch batch;
                if (!stream_.next(batch)) break;
                std::unique_lock<std::mutex> lock(mu_);
                cv_not_full_.wait(lock, [&] { return queue_.size() < depth_ || stop_; });
                if (stop_) break;
                queue_.push_back(std::move(batch));
                lock.unlock();
                cv_not_empty_.notify_one();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mu_);
            error_ = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            done_ = true;
        }
        cv_not_empty_.notify_all();
    }

    PrimeStream& stream_;
    size_t depth_;
    std::mutex mu_;
    std::condition_variable cv_not_empty_, cv_not_full_;
    std::deque<PrimeBatch> queue_;
    bool done_ = false;
    bool stop_ = false;
    std::exception_ptr error_;
    std::thread worker_;
};
// ------------------------- window and parameter normalization -------------------------

static bool is_power_of_two(uint32_t v) { return v && ((v & (v - 1)) == 0); }

static uint32_t choose_bsgs_m(uint64_t length) {
    // Measured on the local RTX 4060 (Windows, sm_89) for pmax=2e8:
    //   L=1e3   m=64/128/256 tie, m=512 slower
    //   L=1e4   m=128 best
    //   L=1e5   m=256 best
    //   L=1e6   m=512 best
    // Each warp owns 2*m table slots plus m chain links, so a larger m costs
    // occupancy long before it saves giant steps.  m=1024 needs --threads 128
    // (its shared-memory footprint exceeds the default per-block limit).
    if (length <= 4096) return 64;
    if (length <= 32768) return 128;
    if (length <= 262144) return 256;
    return 512;
}

static void configure_window(Problem& p) {
    p.length = p.opt.nmax - p.opt.nmin + 1;
    p.m = p.opt.bsgs_m ? p.opt.bsgs_m : choose_bsgs_m(p.length);
    if (!is_power_of_two(p.m)) fail("--bsgs-m must be a power of two");
    p.base_n = (p.opt.nmin / p.m) * p.m;                 // <= nmin, a multiple of m
    p.q = p.base_n / p.m;
    p.giants = (p.opt.nmax - p.base_n) / p.m + 1;        // covers nmax
}

static void validate_problem(Problem& p) {
    if (p.opt.k == 0) fail("-k K is required");
    if (p.opt.base < 2) fail("-b BASE is required");
    if (p.opt.nmin == 0 || p.opt.nmax == 0) fail("-n N1 and -N N2 are required");
    if (p.opt.nmax < p.opt.nmin) fail("nmax must not be smaller than nmin");
    if (p.length > LENGTH_MAX) fail("exponent interval is too long");
    if (p.opt.k % 2 == 1 && p.opt.base % 2 == 1)
        fail("k and base are both odd, so every k*b^n+/-1 is an even number greater than 2");
    if (p.opt.only_twins && (p.opt.base % 3) != 0 && (p.opt.k % 3) != 0)
        // p=3 then divides one of the two signs for every exponent: if b = 1 (mod 3)
        // the terms are k+/-1, and if b = 2 (mod 3) the powers b^n cover both
        // nonzero residues.  A twin search needs 3 | k in this case.
        fail("paired/twin mode can never leave a candidate: 3 divides neither k nor the base, "
             "so for every n one of k*b^n+1 and k*b^n-1 is divisible by 3. "
             "Use the default independent mode (or --sign), or choose k divisible by 3");
    if (p.opt.max_prime <= p.opt.min_prime) fail("pmax must be greater than pmin");
    if (p.opt.only_twins && (!p.opt.plus_enabled || !p.opt.minus_enabled))
        fail("paired sieving always covers both signs");
    if (!p.opt.only_twins && !p.opt.plus_enabled && !p.opt.minus_enabled)
        fail("no sign selected");
    if (p.opt.input_factors.empty() && p.opt.apply_and_exit)
        fail("-A requires -I factors");
    if (p.opt.output_terms.empty()) {
        std::ostringstream oss;
        oss << "k" << p.opt.k << "_b" << p.opt.base << "_n" << p.opt.nmin
            << "-" << p.opt.nmax << ".txt";
        p.opt.output_terms = oss.str();
    }
}

static void allocate_bitmaps(Problem& p, bool value) {
    if (p.opt.only_twins) fill_bits(p.twin_bits, p.length, value);
    else {
        fill_bits(p.minus_bits, p.length, value && p.opt.minus_enabled);
        fill_bits(p.plus_bits, p.length, value && p.opt.plus_enabled);
    }
}

static void allocate_factor_storage(Problem& p) {
    if (p.opt.output_factors.empty()) return;
    size_t n = static_cast<size_t>(p.length);
    if (p.opt.only_twins) p.twin_factor.assign(n, 0);
    else { p.minus_factor.assign(n, 0); p.plus_factor.assign(n, 0); }
}

// ------------------------- term file input / output -------------------------

struct TermsHeader {
    uint64_t k = 0;
    uint32_t base = 0;
    bool has_plus = false;
    bool has_minus = false;
    bool native = false;
    bool independent = false;   // rows carry their sign
    bool sign_plus = true;      // native independent: enabled signs
    bool sign_minus = true;
    uint64_t nmin = 0;
    uint64_t nmax = 0;
    uint64_t sieved_to = 1;
    uint64_t declared_terms = 0;
};

static bool regex_match_groups(const std::string& s, const std::string& pat, std::smatch& m) {
    return std::regex_match(s, m, std::regex(pat, std::regex::icase));
}

static TermsHeader parse_terms_header(const std::string& raw_line) {
    TermsHeader h;
    const std::string line = trim(raw_line);
    const size_t comment_at = line.find("//");
    const std::string expr = trim(line.substr(0, comment_at));
    const std::string comment = comment_at == std::string::npos ? std::string() : line.substr(comment_at);
    std::smatch m;

    // "ABC 12345*2^$a+1", "ABC 12345*2^$a-1", "ABC 12345*2^$a+1 & 12345*2^$a-1"
    if (regex_match_groups(expr, R"(ABC\s+(\d+)\*(\d+)\^\$a\+1\s*&\s*(\d+)\*(\d+)\^\$a-1)", m)) {
        h.k = std::stoull(m[1]); h.base = static_cast<uint32_t>(std::stoul(m[2]));
        if (m[1] != m[3] || m[2] != m[4]) fail("twin header coefficient/base mismatch");
        h.has_plus = h.has_minus = true;
    } else if (regex_match_groups(expr, R"(ABC\s+(\d+)\*(\d+)\^\$a\+1)", m)) {
        h.k = std::stoull(m[1]); h.base = static_cast<uint32_t>(std::stoul(m[2]));
        h.has_plus = true;
    } else if (regex_match_groups(expr, R"(ABC\s+(\d+)\*(\d+)\^\$a-1)", m)) {
        h.k = std::stoull(m[1]); h.base = static_cast<uint32_t>(std::stoul(m[2]));
        h.has_minus = true;
    } else {
        fail("unrecognized input header: " + raw_line);
    }

    if (comment.find("GPRSV_N") != std::string::npos) {
        h.native = true;
        auto field = [&](const std::string& name, uint64_t& out) {
            std::smatch f;
            std::regex pat(std::string("(?:^|[^A-Za-z_])") + name + "=([0-9]+)");
            std::string c = " " + comment;
            if (std::regex_search(c, f, pat)) out = std::stoull(f[1]);
        };
        field("nmin", h.nmin);
        field("nmax", h.nmax);
        field("sieved_to", h.sieved_to);
        field("terms", h.declared_terms);
        std::smatch s;
        if (std::regex_search(comment, s, std::regex(R"(signs=([a-z]+))"))) {
            const std::string signs = s[1];
            h.independent = true;
            h.sign_plus = signs != "minus";
            h.sign_minus = signs != "plus";
        } else {
            h.independent = false;
        }
    } else {
        // Plain ABC form, e.g. srsieve2 output: "ABC 12345*2^$a-1 // Sieved to 1e6"
        std::smatch s;
        if (std::regex_search(comment, s, std::regex(R"(Sieved\s+to\s+([0-9]+))", std::regex::icase)))
            h.sieved_to = std::stoull(s[1]);
        h.independent = !(h.has_plus && h.has_minus);
    }
    return h;
}

static void scan_term_rows(const std::string& filename, const TermsHeader& h,
                           const std::function<void(uint64_t,int)>& fn) {
    std::ifstream in(filename);
    if (!in) fail("Unable to open input terms file: " + filename);
    std::string line;
    if (!std::getline(in, line)) fail("Empty input terms file: " + filename);
    uint64_t rows = 0;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        uint64_t n = 0;
        if (!(iss >> n)) fail("Malformed terms line: " + line);
        int c = 0;
        if (h.independent) {
            std::string sign;
            if (!(iss >> sign) || (sign != "+1" && sign != "-1"))
                fail("Independent terms rows require n and +1/-1: " + line);
            c = sign == "+1" ? 1 : -1;
        }
        fn(n, c);
        ++rows;
    }
    if (rows == 0) fail("input terms file contains no candidates");
    if (h.declared_terms != 0 && rows != h.declared_terms)
        fail("input terms count does not match the header");
}

static void load_input_terms(Problem& p) {
    std::ifstream in(p.opt.input_terms);
    if (!in) fail("Unable to open input terms file: " + p.opt.input_terms);
    std::string header_line;
    if (!std::getline(in, header_line)) fail("Empty input terms file");
    in.close();
    const TermsHeader h = parse_terms_header(header_line);

    if (p.opt.k != 0 && p.opt.k != h.k) fail("input header k does not match -k");
    if (p.opt.base != 0 && p.opt.base != h.base) fail("input header base does not match -b");
    p.opt.k = h.k;
    p.opt.base = h.base;
    if (h.native) {
        if (p.opt.nmin != 0 && p.opt.nmin != h.nmin) fail("input header nmin does not match -n");
        if (p.opt.nmax != 0 && p.opt.nmax != h.nmax) fail("input header nmax does not match -N");
        p.opt.nmin = h.nmin;
        p.opt.nmax = h.nmax;
    }
    if (p.opt.nmin == 0 || p.opt.nmax == 0)
        fail("input header lacks the exponent interval; pass -n N1 -N N2");
    if (p.opt.nmax < p.opt.nmin) fail("input header has an empty exponent interval");

    p.opt.only_twins = !h.independent;
    if (h.independent) {
        p.opt.plus_enabled = h.sign_plus;
        p.opt.minus_enabled = h.sign_minus;
    }
    p.input_sieved_to = h.sieved_to;
    configure_window(p);
    validate_problem(p);

    allocate_bitmaps(p, false);
    scan_term_rows(p.opt.input_terms, h, [&](uint64_t n, int c) {
        uint64_t idx = 0;
        if (!n_to_index(p, n, idx)) fail("input term outside the declared exponent interval: " + std::to_string(n));
        if (p.opt.only_twins) bit_set(p.twin_bits, idx);
        else if (c < 0) { if (p.opt.minus_enabled) bit_set(p.minus_bits, idx); }
        else { if (p.opt.plus_enabled) bit_set(p.plus_bits, idx); }
    });
    p.opt.min_prime = std::max<uint64_t>(p.opt.min_prime, p.input_sieved_to);
    if (p.opt.max_prime <= p.opt.min_prime)
        fail("pmax must be greater than the already sieved depth");
}

// Write the whole file next to its destination and replace it atomically, so a
// crash mid-write cannot destroy an existing checkpoint.
static void atomic_write_text(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc | std::ios::binary);
        if (!out) fail("Unable to create temporary output file: " + tmp);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.flush();
        if (!out) { out.close(); std::remove(tmp.c_str()); fail("output write failed: " + tmp); }
    }
    const std::filesystem::path temp_path = std::filesystem::u8path(tmp);
    const std::filesystem::path target_path = std::filesystem::u8path(path);
#if defined(_WIN32)
    if (!MoveFileExW(temp_path.c_str(), target_path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
        fail("output replace failed: " + path);
    }
#else
    std::error_code ec;
    std::filesystem::rename(temp_path, target_path, ec);
    if (ec) {
        std::filesystem::remove(temp_path, ec);
        fail("output replace failed: " + path);
    }
#endif
}

static void write_terms(const Problem& p, uint64_t largest_prime) {
    std::ostringstream out;
    const std::string mult = multiplier_text(p);
    out << "ABC " << mult << "+1 & " << mult << "-1 // GPRSV_N v" << APP_VERSION
        << " sieved_to=" << largest_prime
        << " nmin=" << p.opt.nmin << " nmax=" << p.opt.nmax
        << " terms=" << active_terms(p);
    if (!p.opt.only_twins) {
        const char* signs = (p.opt.plus_enabled && p.opt.minus_enabled) ? "both"
                          : p.opt.plus_enabled ? "plus" : "minus";
        out << " signs=" << signs;
    }
    out << "\n";

    for (uint64_t i = 0; i < p.length; ++i) {
        const uint64_t n = index_to_n(p, i);
        if (p.opt.only_twins) {
            if (bit_test(p.twin_bits, i)) out << n << '\n';
        } else {
            if (bit_test(p.plus_bits, i)) out << n << " +1\n";
            if (bit_test(p.minus_bits, i)) out << n << " -1\n";
        }
    }
    atomic_write_text(p.opt.output_terms, out.str());
}

static void write_expressions(const Problem& p) {
    if (p.opt.expression_file.empty()) return;
    std::ostringstream out;
    for (uint64_t i = 0; i < p.length; ++i) {
        const uint64_t n = index_to_n(p, i);
        if (p.opt.only_twins) {
            if (!bit_test(p.twin_bits, i)) continue;
            out << term_text(p, n, +1) << '\n' << term_text(p, n, -1) << '\n';
        } else {
            if (bit_test(p.plus_bits, i)) out << term_text(p, n, +1) << '\n';
            if (bit_test(p.minus_bits, i)) out << term_text(p, n, -1) << '\n';
        }
    }
    atomic_write_text(p.opt.expression_file, out.str());
}

static uint64_t write_factors(const Problem& p) {
    if (p.opt.output_factors.empty()) return 0;
    std::ofstream out(p.opt.output_factors, std::ios::app);
    if (!out) fail("Unable to open output factor file: " + p.opt.output_factors);
    uint64_t count = 0;
    for (uint64_t i = 0; i < p.length; ++i) {
        const uint64_t n = index_to_n(p, i);
        if (p.opt.only_twins) {
            uint64_t enc = p.twin_factor[static_cast<size_t>(i)];
            if (!enc) continue;
            const int c = (enc & SIGN_BIT) ? 1 : -1;
            const uint64_t q = enc & ~SIGN_BIT;
            if (p.opt.verify_factors && !factor_is_valid(p, q, n, c))
                fail("factor verification failed: " + std::to_string(q) + " | " + term_text(p, n, c));
            out << q << " | " << term_text(p, n, c) << '\n'; ++count;
        } else {
            uint64_t qm = p.minus_factor[static_cast<size_t>(i)];
            uint64_t qp = p.plus_factor[static_cast<size_t>(i)];
            if (qm) {
                if (p.opt.verify_factors && !factor_is_valid(p, qm, n, -1)) fail("factor verification failed");
                out << qm << " | " << term_text(p, n, -1) << '\n'; ++count;
            }
            if (qp) {
                if (p.opt.verify_factors && !factor_is_valid(p, qp, n, +1)) fail("factor verification failed");
                out << qp << " | " << term_text(p, n, +1) << '\n'; ++count;
            }
        }
    }
    return count;
}

static void apply_factor_file(Problem& p) {
    if (p.opt.input_factors.empty()) return;
    std::ifstream in(p.opt.input_factors);
    if (!in) fail("Unable to open factor file: " + p.opt.input_factors);
    std::string line;
    uint64_t read = 0, applied = 0;
    std::smatch m;
    while (std::getline(in, line)) {
        line = trim(line); if (line.empty()) continue;
        if (!regex_match_groups(line, R"((\d+)\s*\|\s*(\d+)\*(\d+)\^(\d+)([+-])1)", m))
            fail("Malformed factor line: " + line);
        const uint64_t factor = std::stoull(m[1]);
        const uint64_t k = std::stoull(m[2]);
        const uint64_t base = std::stoull(m[3]);
        const uint64_t n = std::stoull(m[4]);
        const int c = m[5] == "+" ? 1 : -1;
        ++read;
        if (k != p.opt.k || base != p.opt.base) fail("Factor line coefficient/base mismatch: " + line);
        uint64_t idx = 0;
        if (!n_to_index(p, n, idx)) continue;
        if (p.opt.only_twins) {
            if (!bit_clear(p.twin_bits, idx)) continue;
            if (!p.twin_factor.empty()) p.twin_factor[static_cast<size_t>(idx)] = factor | (c > 0 ? SIGN_BIT : 0);
        } else if (c < 0) {
            if (!p.opt.minus_enabled || !bit_clear(p.minus_bits, idx)) continue;
            if (!p.minus_factor.empty()) p.minus_factor[static_cast<size_t>(idx)] = factor;
        } else {
            if (!p.opt.plus_enabled || !bit_clear(p.plus_bits, idx)) continue;
            if (!p.plus_factor.empty()) p.plus_factor[static_cast<size_t>(idx)] = factor;
        }
        if (!factor_is_valid(p, factor, n, c)) fail("Invalid factor: " + line);
        ++applied;
    }
    if (!p.opt.quiet) std::cout << "Applied " << applied << " of " << read << " input factors.\n";
}
// ------------------------- baby-step/giant-step table -------------------------

// A table entry packs the low 48 bits of a baby-step value with its index.
// The full value is 62 bits, so a probe hit is verified by recomputing b^j;
// a 48-bit match that is not a real hit is astronomically unlikely and is
// rejected by that check.  Duplicate baby values (small multiplicative order)
// are chained through next[], so every j is still reachable.
constexpr uint64_t TAB_EMPTY = ~UINT64_C(0);
constexpr uint64_t TAB_VAL_MASK = (UINT64_C(1) << 48) - 1;
constexpr uint16_t TAB_NONE = 0xFFFFu;
// GPU table: 16-bit value fingerprint in the low half, 16-bit baby index above.
constexpr uint32_t TAB_EMPTY32 = ~UINT32_C(0);
constexpr uint32_t TAB_KEY_MASK = 0xFFFFU;

static inline uint32_t table_slot(uint64_t v, uint32_t mask) {
    return static_cast<uint32_t>(((v * UINT64_C(0x9E3779B97F4A7C15)) >> 32) & mask);
}

// ------------------------- CPU small-prime sieve -------------------------

class HostSieve {
public:
    explicit HostSieve(Problem& pr) : p_(pr) {
        use_bsgs_ = (pr.opt.backend == 1) || (pr.opt.backend == 2 && pr.length > 256);
        m_ = pr.m;
        mask_ = 2 * m_ - 1;
        if (use_bsgs_) {
            table_.assign(static_cast<size_t>(2) * m_, TAB_EMPTY);
            next_.assign(m_, TAB_NONE);
        }
    }

    uint64_t sieve_prime(uint64_t prime) {
        return use_bsgs_ ? bsgs_prime(prime) : walk_prime(prime);
    }

private:
    uint64_t clear_candidate(uint64_t prime, uint64_t n, int c) {
        if (term_equals_prime_host(p_, prime, n, c)) return 0;
        const uint64_t idx = n - p_.opt.nmin;
        bool removed = false;
        if (p_.opt.only_twins) {
            removed = bit_clear(p_.twin_bits, idx);
            if (removed && !p_.twin_factor.empty())
                p_.twin_factor[static_cast<size_t>(idx)] = prime | (c > 0 ? SIGN_BIT : 0);
        } else if (c < 0) {
            if (!p_.opt.minus_enabled) return 0;
            removed = bit_clear(p_.minus_bits, idx);
            if (removed && !p_.minus_factor.empty()) p_.minus_factor[static_cast<size_t>(idx)] = prime;
        } else {
            if (!p_.opt.plus_enabled) return 0;
            removed = bit_clear(p_.plus_bits, idx);
            if (removed && !p_.plus_factor.empty()) p_.plus_factor[static_cast<size_t>(idx)] = prime;
        }
        return removed ? 1 : 0;
    }

    uint64_t walk_prime(uint64_t prime) {
        const uint64_t p = prime;
        const uint64_t base = p_.opt.base % p;
        const uint64_t kk = p_.opt.k % p;
        if (base == 0 || kk == 0) return 0;
        const uint64_t inv_k = inverse_mod_host(kk, p);
        if (inv_k == 0) return 0;
        const uint64_t plus_target = p - inv_k;
        uint64_t x = pow_mod_host(base, p_.opt.nmin, p);
        uint64_t removed = 0;
        for (uint64_t idx = 0; idx < p_.length; ++idx) {
            if (x == inv_k) removed += clear_candidate(p, p_.opt.nmin + idx, -1);
            else if (x == plus_target) removed += clear_candidate(p, p_.opt.nmin + idx, +1);
            x = mul_mod_host(x, base, p);
        }
        return removed;
    }

    uint64_t apply_value(uint64_t prime, uint64_t value, uint64_t i, int sign) {
        const uint64_t key = value & TAB_VAL_MASK;
        uint32_t slot = table_slot(value, mask_);
        for (uint32_t probes = 0; probes <= mask_; ++probes) {
            const uint64_t e = table_[slot];
            if (e == TAB_EMPTY) return 0;
            if ((e >> 16) == key) {
                uint64_t removed = 0;
                for (uint32_t j = static_cast<uint32_t>(e & 0xFFFFu); j != TAB_NONE; j = next_[j]) {
                    if (pow_mod_host(p_.opt.base, j, prime) != value) continue;
                    const uint64_t n = p_.base_n + i * static_cast<uint64_t>(m_) + j;
                    if (n < p_.opt.nmin || n > p_.opt.nmax) continue;
                    removed += clear_candidate(prime, n, sign);
                }
                return removed;
            }
            slot = (slot + 1) & mask_;
        }
        return 0;
    }

    uint64_t bsgs_prime(uint64_t prime) {
        const uint64_t p = prime;
        const uint64_t base = p_.opt.base % p;
        const uint64_t kk = p_.opt.k % p;
        if (base == 0 || kk == 0) return 0;
        const uint64_t bm = pow_mod_host(base, m_, p);
        const uint64_t inv = inverse_mod_host(mul_mod_host(bm, kk, p), p);
        if (inv == 0) return 0;
        const uint64_t f = mul_mod_host(inv, kk, p);   // b^-m
        const uint64_t c = mul_mod_host(inv, bm, p);   // k^-1
        uint64_t g = mul_mod_host(c, pow_mod_host(f, p_.q, p), p);

        std::fill(table_.begin(), table_.end(), TAB_EMPTY);
        uint64_t v = 1 % p;
        for (uint32_t j = 0; j < m_; ++j) {
            const uint64_t key = v & TAB_VAL_MASK;
            const uint64_t entry = (key << 16) | j;
            uint32_t slot = table_slot(v, mask_);
            for (uint32_t probes = 0; probes <= mask_; ++probes) {
                const uint64_t old = table_[slot];
                if (old == TAB_EMPTY) { table_[slot] = entry; next_[j] = TAB_NONE; break; }
                if ((old >> 16) == key) { next_[j] = static_cast<uint16_t>(old & 0xFFFFu); table_[slot] = entry; break; }
                slot = (slot + 1) & mask_;
            }
            v = mul_mod_host(v, base, p);
        }

        uint64_t removed = 0;
        for (uint64_t i = 0; i < p_.giants; ++i) {
            removed += apply_value(p, g, i, -1);
            removed += apply_value(p, p - g, i, +1);
            g = mul_mod_host(g, f, p);
        }
        return removed;
    }

    Problem& p_;
    bool use_bsgs_ = false;
    uint32_t m_ = 0, mask_ = 0;
    std::vector<uint64_t> table_;
    std::vector<uint16_t> next_;
};
struct Mont64 {
    uint64_t mod;
    uint64_t ninv;
    uint64_t r2;
    uint64_t rmod;
};

__device__ __forceinline__ uint64_t d_add_mod(uint64_t a, uint64_t b, uint64_t m) {
    return (a >= m - b) ? (a - (m - b)) : (a + b);
}

__device__ __forceinline__ uint64_t d_mont_ninv(uint64_t n) {
    uint64_t x = 1;
#pragma unroll
    for (int i = 0; i < 6; ++i) x *= 2 - n * x;
    return 0 - x;
}

__device__ __forceinline__ uint64_t d_mont_mul(uint64_t a, uint64_t b, const Mont64& c) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    uint64_t m = lo * c.ninv;
    uint64_t mlo = m * c.mod;
    uint64_t mhi = __umul64hi(m, c.mod);
    uint64_t sumlo = lo + mlo;
    uint64_t carry = (sumlo < lo);
    uint64_t u = hi + mhi + carry;
    if (u >= c.mod) u -= c.mod;
    return u;
}

__device__ __forceinline__ Mont64 d_make_mont(uint64_t p) {
    Mont64 c;
    c.mod = p;
    c.ninv = d_mont_ninv(p);
    c.rmod = (uint64_t(0) - p) % p; // 2^64 mod p
    uint64_t x = c.rmod;
#pragma unroll 1
    for (int i = 0; i < 64; ++i) x = d_add_mod(x, x, p);
    c.r2 = x; // 2^128 mod p
    return c;
}

__device__ __forceinline__ uint64_t d_mont_pow_normal(uint64_t a, uint64_t e, const Mont64& c) {
    uint64_t x = d_mont_mul(a % c.mod, c.r2, c);
    uint64_t r = c.rmod;
    while (e) {
        if (e & 1) r = d_mont_mul(r, x, c);
        e >>= 1;
        if (e) x = d_mont_mul(x, x, c);
    }
    return d_mont_mul(r, 1, c);
}

__device__ __forceinline__ uint32_t d_inverse_u32(uint32_t a, uint32_t m) {
    int64_t t = 0, nt = 1;
    int64_t r = m, nr = a;
    while (nr != 0) {
        int64_t q = r / nr;
        int64_t tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp;
    }
    if (r != 1) return 0;
    if (t < 0) t += m;
    return static_cast<uint32_t>(t);
}

__device__ __forceinline__ uint64_t d_pow_small(uint64_t a, uint32_t e, uint64_t p) {
    uint64_t r = 1 % p;
    a %= p;
    while (e) {
        if (e & 1U) r = (r * a) % p;
        e >>= 1;
        if (e) a = (a * a) % p;
    }
    return r;
}

__device__ __forceinline__ uint64_t d_inverse_base_large(uint32_t base, uint64_t p) {
    uint32_t r = static_cast<uint32_t>(p % base);
    if (r == 0) return 0;
    uint32_t inv_r = d_inverse_u32(r, base);
    if (inv_r == 0) return 0;
    uint64_t s = static_cast<uint64_t>(base - inv_r); // s*p == -1 (mod base)
    uint64_t q = p / base;
    uint64_t tail = (UINT64_C(1) + s * r) / base;
    return s * q + tail; // (1+s*p)/base, exact and < p
}

// Product-only REDC: low(T + m*p) is zero, so its carry is exactly (lo != 0).
// Keep the old helper for BN and small fallback paths to isolate this change.
__device__ __forceinline__ uint64_t d_product_mont_mul(uint64_t a, uint64_t b, const Mont64& c) {
    const uint64_t lo = a * b;
    const uint64_t hi = __umul64hi(a, b);
    const uint64_t m = lo * c.ninv;
    uint64_t u = hi + __umul64hi(m, c.mod) + static_cast<uint64_t>(lo != 0);
    if (u >= c.mod) u -= c.mod;
    return u;
}

// Input/output remain Montgomery encoded. No variable division or conversion.
__device__ __forceinline__ uint64_t d_pow_mont_encoded(
    uint64_t x, uint64_t e, const Mont64& c) {
    uint64_t r = c.rmod;
    while (e) {
        if (e & 1ULL) r = d_product_mont_mul(r, x, c);
        e >>= 1;
        if (e) x = d_product_mont_mul(x, x, c);
    }
    return r;
}
// ------------------------- CUDA exact-arithmetic helpers -------------------------

struct GprnArgs {
    uint32_t base;
    uint64_t k;
    uint64_t nmin;
    uint64_t nmax;
    uint64_t base_n;
    uint64_t q;
    uint64_t giants;
    uint32_t m;
    uint32_t mask;
    int mode;                 // 0 = paired, 1 = independent
    uint32_t sign_mask;       // independent: bit0 = +1, bit1 = -1
    uint32_t* twin_bits;
    uint32_t* minus_bits;
    uint32_t* plus_bits;
    uint64_t* twin_factor;
    uint64_t* minus_factor;
    uint64_t* plus_factor;
    unsigned long long* removed;
};

__device__ __forceinline__ uint32_t d_table_slot(uint64_t v, uint32_t mask) {
    return static_cast<uint32_t>(((v * UINT64_C(0x9E3779B97F4A7C15)) >> 32) & mask);
}

__device__ __forceinline__ uint64_t d_to_mont(uint64_t a, const Mont64& c) {
    return d_product_mont_mul(a % c.mod, c.r2, c);
}

// True when the tested term k*b^n+/-1 is itself equal to the prime p.
__device__ __forceinline__ bool d_term_equals_prime(
    uint64_t prime, uint64_t k, uint32_t base, uint64_t n, int c) {
    const uint64_t limit = (c > 0) ? prime - 1 : prime + 1;
    if (limit < k) return false;
    uint64_t v = k;
    for (uint64_t i = 0; i < n; ++i) {
        if (v > limit / base) return false;
        v *= base;
        if (v > limit) return false;
    }
    return v == limit;
}

__device__ __forceinline__ void d_clear_candidate(
    const GprnArgs& a, uint64_t prime, uint64_t n, int c) {
    const uint64_t idx = n - a.nmin;
    const uint32_t word = static_cast<uint32_t>(idx >> 5);
    const uint32_t bit = UINT32_C(1) << (idx & 31);
    if (a.mode == 0) {
        const uint32_t old = atomicAnd(a.twin_bits + word, ~bit);
        if (old & bit) {
            atomicAdd(a.removed, 1ULL);
            if (a.twin_factor) a.twin_factor[idx] = prime | (c > 0 ? SIGN_BIT : 0);
        }
        return;
    }
    if (c < 0) {
        if (!(a.sign_mask & 1U)) return;
        const uint32_t old = atomicAnd(a.minus_bits + word, ~bit);
        if (old & bit) {
            atomicAdd(a.removed, 1ULL);
            if (a.minus_factor) a.minus_factor[idx] = prime;
        }
    } else {
        if (!(a.sign_mask & 2U)) return;
        const uint32_t old = atomicAnd(a.plus_bits + word, ~bit);
        if (old & bit) {
            atomicAdd(a.removed, 1ULL);
            if (a.plus_factor) a.plus_factor[idx] = prime;
        }
    }
}

// ------------------------- GPU kernels -------------------------

// One warp per prime.  Lane 0 performs the modular setup (three
// exponentiations and one inversion); all lanes then build the baby-step
// table and walk the giant steps.
//
// Table entries are 32-bit: 16 bits of value fingerprint and 16 bits of baby
// index.  A fingerprint hit is verified by recomputing b^j, so a truncated
// match can never remove a candidate.  Duplicate baby values (small
// multiplicative order) are chained through next[], so every j stays
// reachable.  The 32-bit layout halves shared memory versus 64-bit entries.
__global__ void gprn_bsgs_kernel(
    const uint64_t* __restrict__ primes, uint64_t prime_count, GprnArgs a) {
    extern __shared__ uint32_t smem[];
    const uint32_t warps_per_block = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t m = a.m;
    const uint32_t mask = a.mask;
    // Per-warp slab: 2*m table slots (uint32) followed by m chain links
    // (uint16).  In uint32 units that is 2*m + m/2.
    const uint32_t slab = 2 * m + (m >> 1);
    uint32_t* tab = smem + static_cast<size_t>(warp) * slab;
    uint16_t* nxt = reinterpret_cast<uint16_t*>(tab + 2 * static_cast<size_t>(m));
    const uint64_t warp_stride = static_cast<uint64_t>(gridDim.x) * warps_per_block;

    for (uint64_t pi = static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp;
         pi < prime_count; pi += warp_stride) {
        const uint64_t p = primes[pi];
        if (p < 3 || (a.base % p) == 0 || (a.k % p) == 0) continue;

        const Mont64 mc = d_make_mont(p);
        const uint64_t base_m = d_to_mont(a.base, mc);
        uint64_t f = 0, g0 = 0;
        if (lane == 0) {
            const uint64_t bm = d_pow_mont_encoded(base_m, m, mc);          // b^m
            const uint64_t km = d_to_mont(a.k % p, mc);
            const uint64_t inv = d_pow_mont_encoded(
                d_product_mont_mul(bm, km, mc), p - 2, mc);                 // 1/(b^m*k)
            f = d_product_mont_mul(inv, km, mc);                            // b^-m
            const uint64_t c = d_product_mont_mul(inv, bm, mc);             // k^-1
            const uint64_t fq = (a.q == 0) ? mc.rmod : d_pow_mont_encoded(f, a.q, mc);
            g0 = d_product_mont_mul(c, fq, mc);                             // k^-1 * b^-base_n
        }
        f = __shfl_sync(0xffffffffU, f, 0);
        g0 = __shfl_sync(0xffffffffU, g0, 0);

        for (uint32_t s = lane; s < 2 * m; s += 32) tab[s] = TAB_EMPTY32;
        __syncwarp();

        const uint64_t step32 = d_pow_mont_encoded(base_m, 32, mc);         // b^32
        uint64_t v = d_pow_mont_encoded(base_m, lane, mc);                  // b^lane
        for (uint32_t j = lane; j < m; j += 32) {
            const uint32_t key = static_cast<uint32_t>(v) & TAB_KEY_MASK;
            const uint32_t entry = (j << 16) | key;
            uint32_t slot = d_table_slot(v, mask);
            for (uint32_t probes = 0; probes <= mask; ++probes) {
                const uint32_t old = tab[slot];
                if (old == TAB_EMPTY32) {
                    if (atomicCAS(tab + slot, TAB_EMPTY32, entry) == TAB_EMPTY32) {
                        nxt[j] = TAB_NONE;
                        break;
                    }
                    continue;
                }
                if ((old & TAB_KEY_MASK) == key) {
                    nxt[j] = static_cast<uint16_t>(old >> 16);
                    if (atomicCAS(tab + slot, old, entry) == old) break;
                    continue;
                }
                slot = (slot + 1) & mask;
            }
            v = d_product_mont_mul(v, step32, mc);
        }
        __syncwarp();

        const uint64_t fl = d_pow_mont_encoded(f, lane, mc);                // f^lane
        const uint64_t f32 = d_pow_mont_encoded(f, 32, mc);                 // f^32
        uint64_t g = d_product_mont_mul(g0, fl, mc);
        for (uint64_t i = lane; i < a.giants; i += 32) {
            for (int sign = -1; sign <= 1; sign += 2) {
                const uint64_t value = sign < 0 ? g : p - g;
                const uint32_t key = static_cast<uint32_t>(value) & TAB_KEY_MASK;
                uint32_t slot = d_table_slot(value, mask);
                for (uint32_t probes = 0; probes <= mask; ++probes) {
                    const uint32_t e = tab[slot];
                    if (e == TAB_EMPTY32) break;
                    if ((e & TAB_KEY_MASK) == key) {
                        for (uint32_t j = static_cast<uint32_t>(e >> 16);
                             j != TAB_NONE; j = nxt[j]) {
                            if (d_pow_mont_encoded(base_m, j, mc) != value) continue;
                            const uint64_t n = a.base_n + i * static_cast<uint64_t>(m) + j;
                            if (n < a.nmin || n > a.nmax) continue;
                            if (d_term_equals_prime(p, a.k, a.base, n, sign)) continue;
                            d_clear_candidate(a, p, n, sign);
                        }
                        break;
                    }
                    slot = (slot + 1) & mask;
                }
            }
            g = d_product_mont_mul(g, f32, mc);
        }
        __syncwarp();
    }
}

// One thread per prime, one modular multiplication per exponent step.  Used for
// very short intervals and as the cross-check backend.
__global__ void gprn_walk_kernel(
    const uint64_t* __restrict__ primes, uint64_t prime_count, GprnArgs a) {
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t pi = tid; pi < prime_count; pi += stride) {
        const uint64_t p = primes[pi];
        if (p < 3 || (a.base % p) == 0 || (a.k % p) == 0) continue;
        const Mont64 mc = d_make_mont(p);
        const uint64_t base_m = d_to_mont(a.base, mc);
        const uint64_t inv_k = d_pow_mont_encoded(d_to_mont(a.k % p, mc), p - 2, mc);
        const uint64_t plus_target = p - inv_k;
        uint64_t x = d_pow_mont_encoded(base_m, a.nmin, mc);
        for (uint64_t idx = 0; idx < a.nmax - a.nmin + 1; ++idx) {
            const uint64_t n = a.nmin + idx;
            if (x == inv_k) {
                if (!d_term_equals_prime(p, a.k, a.base, n, -1)) d_clear_candidate(a, p, n, -1);
            } else if (x == plus_target) {
                if (!d_term_equals_prime(p, a.k, a.base, n, +1)) d_clear_candidate(a, p, n, +1);
            }
            x = d_product_mont_mul(x, base_m, mc);
        }
    }
}
// ------------------------- GPU storage and execution -------------------------

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { allocate(n); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr; count = n;
        if (n) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
};

template <typename T>
struct PinnedBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    void allocate(size_t n) {
        if (ptr) CUDA_CHECK(cudaFreeHost(ptr));
        ptr = nullptr; count = n;
        if (n) CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr), n * sizeof(T), cudaHostAllocDefault));
    }
};

struct GpuCompletion {
    uint64_t prime_count = 0;
    uint64_t last_prime = 0;
    uint64_t removed = 0;
    double h2d_ms = 0.0;
    double reset_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_device_ms = 0.0;
};

class GpuSieve {
public:
    explicit GpuSieve(Problem& pr) : p_(pr) {
        CUDA_CHECK(cudaSetDevice(p_.opt.device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop_, p_.opt.device));
        if (p_.opt.blocks == 0) p_.opt.blocks = prop_.multiProcessorCount * 8;
        if (p_.opt.threads > prop_.maxThreadsPerBlock)
            fail("--threads exceeds device maxThreadsPerBlock");

        use_walk_ = (p_.opt.backend == 0) || (p_.opt.backend == 2 && p_.length <= 256);
        warps_per_block_ = static_cast<uint32_t>(p_.opt.threads / 32);
        if (!use_walk_) {
            shared_bytes_ = static_cast<size_t>(warps_per_block_) *
                (2 * static_cast<size_t>(p_.m) * sizeof(uint32_t) +
                 static_cast<size_t>(p_.m) * sizeof(uint16_t));
            if (shared_bytes_ > prop_.sharedMemPerBlockOptin) {
                std::ostringstream oss;
                oss << "BSGS table needs " << shared_bytes_ << " bytes of shared memory per block but the device allows "
                    << prop_.sharedMemPerBlockOptin << "; use --bsgs-m smaller or --threads smaller";
                fail(oss.str());
            }
            if (shared_bytes_ > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(gprn_bsgs_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(shared_bytes_)));
            }
        }

        size_t words = static_cast<size_t>(words_for_bits(p_.length));
        if (p_.opt.only_twins) {
            d_twin_.allocate(words);
            CUDA_CHECK(cudaMemcpy(d_twin_.ptr, p_.twin_bits.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice));
            if (!p_.twin_factor.empty()) {
                d_twin_factor_.allocate(static_cast<size_t>(p_.length));
                CUDA_CHECK(cudaMemcpy(d_twin_factor_.ptr, p_.twin_factor.data(),
                                      p_.twin_factor.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
            }
        } else {
            d_minus_.allocate(words); d_plus_.allocate(words);
            CUDA_CHECK(cudaMemcpy(d_minus_.ptr, p_.minus_bits.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_plus_.ptr, p_.plus_bits.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice));
            if (!p_.minus_factor.empty()) {
                d_minus_factor_.allocate(static_cast<size_t>(p_.length));
                d_plus_factor_.allocate(static_cast<size_t>(p_.length));
                CUDA_CHECK(cudaMemcpy(d_minus_factor_.ptr, p_.minus_factor.data(),
                                      p_.minus_factor.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_plus_factor_.ptr, p_.plus_factor.data(),
                                      p_.plus_factor.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
            }
        }

        args_.base = p_.opt.base;
        args_.k = p_.opt.k;
        args_.nmin = p_.opt.nmin;
        args_.nmax = p_.opt.nmax;
        args_.base_n = p_.base_n;
        args_.q = p_.q;
        args_.giants = p_.giants;
        args_.m = p_.m;
        args_.mask = 2 * p_.m - 1;
        args_.mode = p_.opt.only_twins ? 0 : 1;
        args_.sign_mask = (p_.opt.plus_enabled ? 2U : 0U) | (p_.opt.minus_enabled ? 1U : 0U);
        args_.twin_bits = d_twin_.ptr;
        args_.minus_bits = d_minus_.ptr;
        args_.plus_bits = d_plus_.ptr;
        args_.twin_factor = d_twin_factor_.ptr;
        args_.minus_factor = d_minus_factor_.ptr;
        args_.plus_factor = d_plus_factor_.ptr;

        slots_.reserve(p_.opt.cuda_streams);
        for (uint32_t i = 0; i < p_.opt.cuda_streams; ++i) {
            auto slot = std::make_unique<Slot>();
            CUDA_CHECK(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreate(&slot->start));
            CUDA_CHECK(cudaEventCreate(&slot->h2d_done));
            CUDA_CHECK(cudaEventCreate(&slot->reset_done));
            CUDA_CHECK(cudaEventCreate(&slot->kernel_done));
            CUDA_CHECK(cudaEventCreate(&slot->done));
            slot->d_removed.allocate(1);
            slot->h_removed.allocate(1);
            slots_.push_back(std::move(slot));
        }

        if (!p_.opt.quiet) {
            std::cout << "CUDA device " << p_.opt.device << ": " << prop_.name
                      << ", SMs=" << prop_.multiProcessorCount
                      << ", launch=" << p_.opt.blocks << "x" << p_.opt.threads << "\n"
                      << "CUDA pipeline: " << slots_.size()
                      << " pinned buffer/stream" << (slots_.size() == 1 ? "" : "s")
                      << " (asynchronous H2D + kernel)";
            if (use_walk_) {
                std::cout << ", backend=walk\n";
            } else {
                std::cout << ", backend=bsgs (m=" << p_.m
                          << ", shared/block=" << shared_bytes_ << " B)\n";
            }
        }
    }

    ~GpuSieve() {
        for (auto& slot_ptr : slots_) {
            Slot& s = *slot_ptr;
            if (s.stream) cudaStreamSynchronize(s.stream);
            if (s.start) cudaEventDestroy(s.start);
            if (s.h2d_done) cudaEventDestroy(s.h2d_done);
            if (s.reset_done) cudaEventDestroy(s.reset_done);
            if (s.kernel_done) cudaEventDestroy(s.kernel_done);
            if (s.done) cudaEventDestroy(s.done);
            if (s.stream) cudaStreamDestroy(s.stream);
            s.stream = nullptr;
        }
    }

    size_t capacity() const { return slots_.size(); }
    size_t in_flight() const { return order_.size(); }
    double staging_seconds() const { return staging_seconds_; }
    uint64_t direct_pinned_batches() const { return direct_pinned_batches_; }
    uint64_t fallback_staged_batches() const { return fallback_staged_batches_; }
    bool uses_walk() const { return use_walk_; }

    void submit(const PrimeBatch& batch, size_t offset, size_t prime_count) {
        if (prime_count == 0) return;
        if (!batch.data || offset > batch.count || prime_count > batch.count - offset)
            fail("Internal invalid prime batch span");
        const uint64_t* primes = batch.data + offset;
        submit(primes, prime_count, batch.owner, batch.pinned);
    }

    void submit(const uint64_t* primes, size_t prime_count,
                std::shared_ptr<void> owner, bool source_is_pinned) {
        if (prime_count == 0) return;
        if (!primes) fail("Internal null prime batch");
        if (order_.size() >= slots_.size())
            fail("Internal CUDA pipeline overflow: retire a slot before submit");

        size_t slot_index = next_slot_;
        for (size_t tries = 0; tries < slots_.size(); ++tries) {
            if (!slots_[slot_index]->in_flight) break;
            slot_index = (slot_index + 1) % slots_.size();
        }
        Slot& s = *slots_[slot_index];
        if (s.in_flight) fail("No free CUDA submission slot");
        next_slot_ = (slot_index + 1) % slots_.size();
        ensure_device_capacity(s, prime_count);

        const uint64_t* h_source = primes;
        if (source_is_pinned) {
            if (!owner) fail("Pinned prime batch is missing its lifetime owner");
            ++direct_pinned_batches_;
        } else {
            ensure_staging_capacity(s, prime_count);
            auto stage_start = std::chrono::steady_clock::now();
            std::memcpy(s.h_primes.ptr, primes, prime_count * sizeof(uint64_t));
            staging_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stage_start).count();
            h_source = s.h_primes.ptr;
            ++fallback_staged_batches_;
        }
        *s.h_removed.ptr = 0;

        CUDA_CHECK(cudaEventRecord(s.start, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_primes.ptr, h_source,
                                   prime_count * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaEventRecord(s.h2d_done, s.stream));
        CUDA_CHECK(cudaMemsetAsync(s.d_removed.ptr, 0, sizeof(unsigned long long), s.stream));
        CUDA_CHECK(cudaEventRecord(s.reset_done, s.stream));

        GprnArgs args = args_;
        args.removed = s.d_removed.ptr;
        if (use_walk_) {
            gprn_walk_kernel<<<p_.opt.blocks, p_.opt.threads, 0, s.stream>>>(
                s.d_primes.ptr, prime_count, args);
        } else {
            gprn_bsgs_kernel<<<p_.opt.blocks, p_.opt.threads, shared_bytes_, s.stream>>>(
                s.d_primes.ptr, prime_count, args);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(s.kernel_done, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.h_removed.ptr, s.d_removed.ptr,
                                   sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaEventRecord(s.done, s.stream));

        s.host_owner = std::move(owner);
        s.prime_count = prime_count;
        s.last_prime = primes[prime_count - 1];
        s.in_flight = true;
        order_.push_back(slot_index);
    }

    GpuCompletion retire_oldest() {
        if (order_.empty()) fail("Internal CUDA pipeline underflow");
        size_t slot_index = order_.front();
        order_.pop_front();
        Slot& s = *slots_[slot_index];
        CUDA_CHECK(cudaEventSynchronize(s.done));

        auto elapsed = [](cudaEvent_t a, cudaEvent_t b) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            return static_cast<double>(ms);
        };

        GpuCompletion c;
        c.prime_count = s.prime_count;
        c.last_prime = s.last_prime;
        c.removed = static_cast<uint64_t>(*s.h_removed.ptr);
        c.h2d_ms = elapsed(s.start, s.h2d_done);
        c.reset_ms = elapsed(s.h2d_done, s.reset_done);
        c.kernel_ms = elapsed(s.reset_done, s.kernel_done);
        c.d2h_ms = elapsed(s.kernel_done, s.done);
        c.total_device_ms = elapsed(s.start, s.done);

        s.in_flight = false;
        s.prime_count = 0;
        s.last_prime = 0;
        s.host_owner.reset();
        return c;
    }

    void download() {
        if (!order_.empty()) fail("Internal error: CUDA pipeline must be drained before download");
        size_t words = static_cast<size_t>(words_for_bits(p_.length));
        if (p_.opt.only_twins) {
            CUDA_CHECK(cudaMemcpy(p_.twin_bits.data(), d_twin_.ptr, words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (!p_.twin_factor.empty())
                CUDA_CHECK(cudaMemcpy(p_.twin_factor.data(), d_twin_factor_.ptr,
                                      p_.twin_factor.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(p_.minus_bits.data(), d_minus_.ptr, words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(p_.plus_bits.data(), d_plus_.ptr, words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (!p_.minus_factor.empty()) {
                CUDA_CHECK(cudaMemcpy(p_.minus_factor.data(), d_minus_factor_.ptr,
                                      p_.minus_factor.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(p_.plus_factor.data(), d_plus_factor_.ptr,
                                      p_.plus_factor.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            }
        }
    }

private:
    struct Slot {
        DeviceBuffer<uint64_t> d_primes;
        DeviceBuffer<unsigned long long> d_removed;
        PinnedBuffer<uint64_t> h_primes;
        PinnedBuffer<unsigned long long> h_removed;
        cudaStream_t stream = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t h2d_done = nullptr;
        cudaEvent_t reset_done = nullptr;
        cudaEvent_t kernel_done = nullptr;
        cudaEvent_t done = nullptr;
        bool in_flight = false;
        uint64_t prime_count = 0;
        uint64_t last_prime = 0;
        std::shared_ptr<void> host_owner;
    };

    void ensure_device_capacity(Slot& s, size_t needed) {
        if (s.d_primes.count >= needed) return;
        size_t grown = needed + needed / 4 + 4096;
        s.d_primes.allocate(grown);
    }

    void ensure_staging_capacity(Slot& s, size_t needed) {
        if (s.h_primes.count >= needed) return;
        size_t grown = needed + needed / 4 + 4096;
        s.h_primes.allocate(grown);
    }

    Problem& p_;
    cudaDeviceProp prop_{};
    DeviceBuffer<uint32_t> d_twin_, d_minus_, d_plus_;
    DeviceBuffer<uint64_t> d_twin_factor_, d_minus_factor_, d_plus_factor_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::deque<size_t> order_;
    size_t next_slot_ = 0;
    double staging_seconds_ = 0.0;
    uint64_t direct_pinned_batches_ = 0;
    uint64_t fallback_staged_batches_ = 0;
    GprnArgs args_{};
    bool use_walk_ = false;
    uint32_t warps_per_block_ = 8;
    size_t shared_bytes_ = 0;
};
static std::string format_duration(double sec) {
    std::ostringstream oss;
    if (sec < 60) oss << std::fixed << std::setprecision(1) << sec << "s";
    else if (sec < 3600) oss << std::fixed << std::setprecision(1) << sec / 60 << "m";
    else oss << std::fixed << std::setprecision(2) << sec / 3600 << "h";
    return oss.str();
}

static double process_cpu_seconds() {
#if defined(_WIN32)
    FILETIME create_time{}, exit_time{}, kernel_time{}, user_time{};
    if (GetProcessTimes(GetCurrentProcess(), &create_time, &exit_time, &kernel_time, &user_time)) {
        ULARGE_INTEGER k{}, u{};
        k.LowPart = kernel_time.dwLowDateTime;
        k.HighPart = kernel_time.dwHighDateTime;
        u.LowPart = user_time.dwLowDateTime;
        u.HighPart = user_time.dwHighDateTime;
        return static_cast<double>(k.QuadPart + u.QuadPart) * 1.0e-7;
    }
#elif defined(CLOCK_PROCESS_CPUTIME_ID)
    timespec ts{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0)
        return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1.0e-9;
#endif
    std::clock_t c = std::clock();
    if (c == static_cast<std::clock_t>(-1)) return 0.0;
    return static_cast<double>(c) / static_cast<double>(CLOCKS_PER_SEC);
}

static std::string format_prime_rate(double primes_per_second) {
    double value = primes_per_second / 1.0e6;
    const char* unit = "M";
    if (value < 1.0) { value *= 1000.0; unit = "K"; }
    if (value < 1.0) { value *= 1000.0; unit = ""; }

    int precision = 0;
    if (value < 1000.0) precision = 1;
    if (value < 100.0) precision = 2;
    if (value < 10.0) precision = 3;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value << unit << " p/sec";
    return oss.str();
}

static std::string format_seconds_per_factor(double seconds) {
    std::ostringstream oss;
    oss << std::setprecision(4) << std::defaultfloat << seconds;
    return oss.str();
}

static std::string format_eta_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) return "n/a";

    uint64_t total = static_cast<uint64_t>(std::llround(seconds));
    uint64_t days = total / 86400;
    total %= 86400;
    uint64_t hours = total / 3600;
    total %= 3600;
    uint64_t minutes = total / 60;
    uint64_t secs = total % 60;

    std::ostringstream oss;
    if (days > 0) {
        oss << days << "d " << hours << "h";
    } else if (hours > 0) {
        oss << hours << "h " << minutes << "m";
    } else if (minutes > 0) {
        oss << minutes << "m " << secs << "s";
    } else {
        oss << secs << "s";
    }
    return oss.str();
}

static std::string format_eta_finish_time(double eta_seconds) {
    if (!std::isfinite(eta_seconds) || eta_seconds < 0.0) return {};

    // Prevent an accidentally absurd estimate from overflowing system_clock/time_t.
    constexpr double MAX_ETA_SECONDS = 100.0 * 366.0 * 86400.0;
    if (eta_seconds > MAX_ETA_SECONDS) return {};

    auto finish_point = std::chrono::system_clock::now()
        + std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::duration<double>(eta_seconds));
    std::time_t finish_time = std::chrono::system_clock::to_time_t(finish_point);
    std::tm local_tm{};
#if defined(_WIN32)
    if (localtime_s(&local_tm, &finish_time) != 0) return {};
#else
    if (localtime_r(&finish_time, &local_tm) == nullptr) return {};
#endif

    char buffer[64]{};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M", &local_tm) == 0)
        return {};
    return buffer;
}

static double estimate_eta_seconds(uint64_t current,
                                   uint64_t pmax,
                                   double p_range_per_second) {
    if (current >= pmax) return 0.0;
    if (!(p_range_per_second > 0.0) || !std::isfinite(p_range_per_second))
        return std::numeric_limits<double>::quiet_NaN();

    long double remaining = static_cast<long double>(pmax - current);
    long double eta = remaining / static_cast<long double>(p_range_per_second);
    if (!(eta >= 0.0L) || eta > static_cast<long double>(std::numeric_limits<double>::max()))
        return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(eta);
}

static double percent_done(uint64_t pmin, uint64_t pmax, uint64_t current) {
    if (pmax <= pmin || current <= pmin) return 0.0;
    long double numerator = static_cast<long double>(current - pmin);
    long double denominator = static_cast<long double>(pmax - pmin);
    long double result = 100.0L * numerator / denominator;
    if (result < 0.0L) result = 0.0L;
    if (result > 100.0L) result = 100.0L;
    return static_cast<double>(result);
}

static void print_progress_line(uint64_t largest_prime,
                                uint64_t pmin,
                                uint64_t pmax,
                                double prime_rate,
                                double p_range_per_second,
                                uint64_t factors_found,
                                uint64_t interval_factors,
                                double interval_wall_seconds,
                                double interval_cpu_seconds) {
    double factors_per_minute = interval_wall_seconds > 0.0
        ? 60.0 * static_cast<double>(interval_factors) / interval_wall_seconds : 0.0;

    std::ostringstream line;
    line << "  p=" << largest_prime << ", " << format_prime_rate(prime_rate)
         << ", " << factors_found << " factors found; last "
         << std::fixed << std::setprecision(1) << interval_wall_seconds << "s: +"
         << interval_factors << ", " << std::setprecision(2) << factors_per_minute
         << " f/min, ";

    if (interval_factors == 0) {
        line << "wall/CPU s/f n/a";
    } else {
        line << format_seconds_per_factor(interval_wall_seconds / static_cast<double>(interval_factors))
             << " wall s/f, "
             << format_seconds_per_factor(interval_cpu_seconds / static_cast<double>(interval_factors))
             << " CPU s/f";
    }

    double eta_seconds = estimate_eta_seconds(largest_prime, pmax, p_range_per_second);
    line << ", " << std::fixed << std::setprecision(1)
         << percent_done(pmin, pmax, largest_prime) << "% done, ETA ";
    if (std::isfinite(eta_seconds)) {
        line << format_eta_duration(eta_seconds);
        std::string finish_time = format_eta_finish_time(eta_seconds);
        if (!finish_time.empty()) line << " (finish " << finish_time << ")";
    } else {
        line << "n/a";
    }
    line << ".";
    std::cout << line.str() << '\n' << std::flush;
}

struct EfficiencyStatus {
    bool stop = false;
    bool window_ready = false;
    uint64_t window_removed = 0;
    double idle_seconds = 0.0;
    double average_seconds_per_factor = 0.0;
    std::string reason;
};

class EfficiencyMonitor {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    EfficiencyMonitor(const Options& opt, TimePoint start)
        : enabled_(opt.efficiency_enabled),
          max_factor_seconds_(opt.max_factor_seconds),
          max_average_factor_seconds_(opt.max_average_factor_seconds),
          window_seconds_(opt.efficiency_window_minutes * 60.0),
          start_(start), last_removal_(start) {}

    bool enabled() const { return enabled_; }

    EfficiencyStatus observe(TimePoint now, uint64_t newly_removed) {
        EfficiencyStatus status;
        if (!enabled_) return status;

        if (newly_removed != 0) {
            samples_.push_back({now, newly_removed});
            window_removed_ += newly_removed;
            last_removal_ = now;
        }

        const auto window = std::chrono::duration<double>(window_seconds_);
        const auto cutoff = now - std::chrono::duration_cast<Clock::duration>(window);
        while (!samples_.empty() && samples_.front().when < cutoff) {
            window_removed_ -= samples_.front().removed;
            samples_.pop_front();
        }

        status.window_removed = window_removed_;
        status.idle_seconds = std::chrono::duration<double>(now - last_removal_).count();
        const double elapsed = std::chrono::duration<double>(now - start_).count();
        status.window_ready = elapsed >= window_seconds_;
        status.average_seconds_per_factor = window_removed_ == 0
            ? std::numeric_limits<double>::infinity()
            : window_seconds_ / static_cast<double>(window_removed_);

        // Both limits use strict greater-than, matching the command wording
        // "exceeds S". The no-factor timer is active during window warm-up.
        if (status.idle_seconds > max_factor_seconds_) {
            status.stop = true;
            std::ostringstream oss;
            oss << "no newly removed term for " << std::fixed
                << std::setprecision(3) << status.idle_seconds
                << "s, limit=" << std::defaultfloat << std::setprecision(6)
                << max_factor_seconds_ << "s";
            status.reason = oss.str();
            return status;
        }

        if (status.window_ready &&
            status.average_seconds_per_factor > max_average_factor_seconds_) {
            status.stop = true;
            std::ostringstream oss;
            oss << "rolling " << std::defaultfloat << std::setprecision(6)
                << window_seconds_ / 60.0 << "m window removed "
                << status.window_removed << " term(s), average=";
            if (std::isfinite(status.average_seconds_per_factor))
                oss << std::defaultfloat << std::setprecision(6)
                    << status.average_seconds_per_factor << "s/factor";
            else
                oss << "infinite s/factor";
            oss << ", limit=" << std::defaultfloat << std::setprecision(6)
                << max_average_factor_seconds_ << "s/factor";
            status.reason = oss.str();
        }
        return status;
    }

private:
    struct Sample {
        TimePoint when;
        uint64_t removed;
    };

    bool enabled_ = false;
    double max_factor_seconds_ = 0.0;
    double max_average_factor_seconds_ = 0.0;
    double window_seconds_ = 0.0;
    TimePoint start_{};
    TimePoint last_removal_{};
    std::deque<Sample> samples_;
    uint64_t window_removed_ = 0;
};
// ------------------------- prime loop -------------------------

static uint64_t run_sieve(Problem& p) {
    uint64_t initial_terms = active_terms(p);
    uint64_t largest_prime = p.opt.min_prime;
    uint64_t primes_tested = 0;
    uint64_t factors_found = 0;
    uint64_t gpu_primes = 0, cpu_primes = 0;
    uint64_t gpu_batches = 0;
    double generator_wait_seconds = 0.0;
    double gpu_backpressure_seconds = 0.0;
    double gpu_h2d_ms = 0.0, gpu_reset_ms = 0.0;
    double gpu_kernel_ms = 0.0, gpu_d2h_ms = 0.0, gpu_total_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    EfficiencyMonitor efficiency(p.opt, start);
    EfficiencyStatus efficiency_status;
    bool efficiency_stopped = false;
    uint64_t efficiency_trigger_prime = 0;
    auto last_report_wall = start;
    auto next_report_wall = start + std::chrono::seconds(p.opt.progress_seconds);
    double last_report_cpu_seconds = process_cpu_seconds();
    uint64_t last_report_factors = 0;
    uint64_t last_report_primes = 0;

    struct RateSample {
        std::chrono::steady_clock::time_point wall;
        uint64_t primes_tested;
        uint64_t largest_prime;
    };
    struct RollingRates {
        double primes_per_second = 0.0;
        double p_range_per_second = 0.0;
    };

    std::deque<RateSample> rate_samples;
    rate_samples.push_back({start, 0, p.opt.min_prime});

    auto rolling_rates = [&](std::chrono::steady_clock::time_point now) {
        rate_samples.push_back({now, primes_tested, largest_prime});
        while (rate_samples.size() > 60) rate_samples.pop_front();
        const RateSample& first = rate_samples.front();
        double seconds = std::chrono::duration<double>(now - first.wall).count();

        RollingRates result;
        if (seconds > 0.0) {
            result.primes_per_second =
                static_cast<double>(primes_tested - first.primes_tested) / seconds;
            if (largest_prime >= first.largest_prime) {
                long double p_delta = static_cast<long double>(largest_prime - first.largest_prime);
                result.p_range_per_second = static_cast<double>(p_delta / seconds);
            }
        }
        return result;
    };

    auto maybe_report = [&](bool force) {
        if (p.opt.quiet) return;
        auto now = std::chrono::steady_clock::now();
        if (!force && now < next_report_wall) return;
        if (primes_tested == last_report_primes) return;
        double now_cpu = process_cpu_seconds();
        double interval_wall = std::chrono::duration<double>(now - last_report_wall).count();
        RollingRates rates = rolling_rates(now);
        print_progress_line(largest_prime, p.opt.min_prime, p.opt.max_prime,
                            rates.primes_per_second, rates.p_range_per_second,
                            factors_found,
                            factors_found - last_report_factors,
                            interval_wall,
                            std::max(0.0, now_cpu - last_report_cpu_seconds));
        last_report_wall = now;
        last_report_cpu_seconds = now_cpu;
        last_report_factors = factors_found;
        last_report_primes = primes_tested;
        next_report_wall = now + std::chrono::seconds(p.opt.progress_seconds);
    };

    HostSieve host(p);

    // For an all-GPU range, construct the CUDA context before starting prime
    // production.  If low primes are present, creation remains lazy so the CPU
    // changes the host bitmap before its one-time upload to the device.
    std::unique_ptr<GpuSieve> gpu;
    if (p.opt.min_prime >= p.opt.cpu_small_prime && p.opt.max_prime > p.opt.cpu_small_prime)
        gpu = std::make_unique<GpuSieve>(p);

    PrimeStream stream(p.opt.min_prime, p.opt.max_prime, p.opt.batch_primes,
                       p.opt.prime_mode, p.opt.mr_switch_sqrt, p.opt.segment_mib,
                       p.opt.prime_threads, p.opt.prime_region_batches,
                       p.opt.prime_prefetch, p.opt.cuda_streams, p.opt.quiet);
    if (!p.opt.quiet) {
        std::cout << "Prime generator: " << stream.description() << "\n"
                  << "Prime pipeline: " << p.opt.prime_prefetch
                  << " prefetched batch" << (p.opt.prime_prefetch == 1 ? "" : "es")
                  << ", CPU handles p <= " << p.opt.cpu_small_prime << "\n";
        if (efficiency.enabled()) {
            std::cout << "Lowest acceptable efficiency: no-factor limit="
                      << p.opt.max_factor_seconds << "s, rolling average limit="
                      << p.opt.max_average_factor_seconds << "s/factor, window="
                      << p.opt.efficiency_window_minutes << "m, automatic pmax="
                      << p.opt.max_prime << "\n";
        }
    }

    PrimeBatchPipeline pipeline(stream, p.opt.prime_prefetch);

    auto observe_efficiency = [&](uint64_t newly_removed) {
        if (!efficiency.enabled() || efficiency_stopped || g_interrupted) return;
        efficiency_status = efficiency.observe(std::chrono::steady_clock::now(), newly_removed);
        if (efficiency_status.stop) {
            efficiency_stopped = true;
            efficiency_trigger_prime = largest_prime;
        }
    };

    auto retire_one = [&](bool evaluate_efficiency) {
        auto wait_start = std::chrono::steady_clock::now();
        GpuCompletion c = gpu->retire_oldest();
        gpu_backpressure_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wait_start).count();
        primes_tested += c.prime_count;
        gpu_primes += c.prime_count;
        factors_found += c.removed;
        largest_prime = c.last_prime;
        ++gpu_batches;
        gpu_h2d_ms += c.h2d_ms;
        gpu_reset_ms += c.reset_ms;
        gpu_kernel_ms += c.kernel_ms;
        gpu_d2h_ms += c.d2h_ms;
        gpu_total_ms += c.total_device_ms;
        if (evaluate_efficiency) observe_efficiency(c.removed);
        maybe_report(false);
    };

    auto drain_gpu = [&](bool evaluate_efficiency) {
        while (gpu && gpu->in_flight() != 0) retire_one(evaluate_efficiency);
    };

    PrimeBatch batch;
    while (!g_interrupted && !efficiency_stopped &&
           pipeline.next(batch, generator_wait_seconds)) {
        if (batch.empty()) continue;

        size_t first_gpu_index = 0;
        if (batch.last() <= p.opt.cpu_small_prime) {
            first_gpu_index = batch.count;
        } else if (batch.first() <= p.opt.cpu_small_prime) {
            first_gpu_index = static_cast<size_t>(
                std::upper_bound(batch.data, batch.data + batch.count,
                                 p.opt.cpu_small_prime) - batch.data);
        }

        if (first_gpu_index != 0) {
            // Prime order is monotone.  CPU primes can only occur before the
            // first GPU submission, or in the single cutoff-crossing batch.
            // Draining here keeps the committed sieve boundary unambiguous.
            drain_gpu(true);
            if (efficiency_stopped) break;
            const uint64_t factors_before_cpu = factors_found;
            for (size_t i = 0; i < first_gpu_index; ++i) {
                uint64_t q = batch.data[i];
                factors_found += host.sieve_prime(q);
                ++primes_tested;
                ++cpu_primes;
                largest_prime = q;
            }
            observe_efficiency(factors_found - factors_before_cpu);
            maybe_report(false);
            if (efficiency_stopped) break;
        }

        if (first_gpu_index != batch.count) {
            if (!gpu) gpu = std::make_unique<GpuSieve>(p);
            while (!efficiency_stopped && gpu->in_flight() >= gpu->capacity())
                retire_one(true);
            if (efficiency_stopped) break;
            gpu->submit(batch, first_gpu_index, batch.count - first_gpu_index);
        }
    }

    // CUDA work already submitted cannot be cancelled safely because all
    // streams update the shared candidate bitmap. Retire it without changing
    // the first stop decision and report the actual committed prime boundary.
    drain_gpu(!g_interrupted && !efficiency_stopped);
    if (gpu) gpu->download();
    maybe_report(true);

    if (efficiency_stopped) {
        std::cout << "Stopped by lowest acceptable efficiency at p="
                  << efficiency_trigger_prime << ": " << efficiency_status.reason << ".";
        if (largest_prime > efficiency_trigger_prime)
            std::cout << " Drained already submitted CUDA work through p="
                      << largest_prime << ".";
        std::cout << "\n";
    } else if (g_interrupted) {
        std::cout << "Interrupted by signal after p=" << largest_prime << ".\n";
    }

    uint64_t remaining = active_terms(p);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Finished through p=" << largest_prime << " in " << format_duration(sec)
              << "; primes tested=" << primes_tested << " (CPU " << cpu_primes
              << ", GPU " << gpu_primes << "); removed=" << (initial_terms - remaining)
              << "; remaining=" << remaining << ".\n";
    if (!p.opt.quiet) {
        double staging_seconds = gpu ? gpu->staging_seconds() : 0.0;
        double host_active_seconds = std::max(0.0,
            sec - generator_wait_seconds - gpu_backpressure_seconds);
        std::cout << "Timing (overlap-aware; fields are not additive): producer wait="
                  << std::fixed << std::setprecision(2) << generator_wait_seconds
                  << "s, GPU backpressure wait=" << gpu_backpressure_seconds
                  << "s, host active=" << host_active_seconds
                  << "s, fallback staging=" << staging_seconds << "s.\n";
        if (stream.pinned_region_buffers() != 0) {
            double gib = static_cast<double>(stream.pinned_region_bytes()) /
                         (1024.0 * 1024.0 * 1024.0);
            std::cout << "Prime host pool: " << stream.pinned_region_buffers()
                      << "/" << stream.pinned_region_pool_limit()
                      << " pinned regions, " << std::setprecision(2) << gib
                      << " GiB allocated, growths="
                      << stream.pinned_region_growths() << ".\n";
        }
        if (gpu) {
            std::cout << "Prime batch path: direct pinned="
                      << gpu->direct_pinned_batches()
                      << ", fallback staged=" << gpu->fallback_staged_batches()
                      << ".\n";
        }
        if (gpu_batches != 0) {
            double kernel_rate = gpu_kernel_ms > 0.0
                ? static_cast<double>(gpu_primes) / (gpu_kernel_ms / 1000.0) : 0.0;
            double h2d_gbps = gpu_h2d_ms > 0.0
                ? (static_cast<double>(gpu_primes) * sizeof(uint64_t) / 1.0e9) /
                  (gpu_h2d_ms / 1000.0) : 0.0;
            std::cout << "CUDA events (summed across " << p.opt.cuda_streams
                      << " stream" << (p.opt.cuda_streams == 1 ? "" : "s") << ", "
                      << gpu_batches << " batches): H2D=" << gpu_h2d_ms / 1000.0
                      << "s (" << std::setprecision(2) << h2d_gbps << " GB/s), reset="
                      << std::setprecision(3) << gpu_reset_ms / 1000.0
                      << "s, kernel=" << gpu_kernel_ms / 1000.0
                      << "s (" << std::setprecision(2) << kernel_rate / 1.0e6
                      << "M p/s), D2H=" << std::setprecision(3) << gpu_d2h_ms / 1000.0
                      << "s, stream-total=" << gpu_total_ms / 1000.0 << "s.\n";
        }
    }
    return largest_prime;
}

} // namespace gprn_cuda

void display_banner() {
    printf("%s\n", "\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220");
    printf("%s\n", "       .oooooo.        ooooooooo.      ooooooooo.        .oooooo..o    oooooo     oooo       ");
    printf("%s\n", "      d8P'  `Y8b       `888   `Y88.    `888   `Y88.     d8P'    `Y8     `888.     .8'        ");
    printf("%s\n", "     888                888   .d88'     888   .d88'     Y88bo.           `888.   .8'         ");
    printf("%s\n", "     888                888ooo88P'      888ooo88P'       `'Y8888o.        `888. .8'          ");
    printf("%s\n", "     888     ooooo      888             888`88b.             `'Y88b        `888.8'           ");
    printf("%s\n", "     `88.    .88'  .o.  888        .o.  888  `88b.  .o. oo     .d8P .o.     `888'    .o.     ");
    printf("%s\n", "      `Y8bood8P'   Y8P o888o       Y8P o888o  o888o Y8P 8''88888P'  Y8P      `8'     Y8P     ");
    printf("%s\n", "\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220");
    printf("%s\n", "                    Generalized-Proth/Riesel-Siever: fixed k, sieve n                    ");
    printf("%s\n", "                              Version 1.0 CUDA by A.P. Nov 2026                             ");
}

#include "console_utf8.hpp"

static bool file_exists(const std::string& path) {
    std::ifstream in(path);
    return static_cast<bool>(in);
}

int main(int argc, char** argv) {
    prp_console::initialize_utf8_output();
    display_banner();
    using namespace gprn_cuda;
    try {
        std::signal(SIGINT, handle_interrupt);
        Options opt;
        parse_options(argc, argv, opt);
        if (opt.help) { print_help(); return 0; }
        if (opt.version) { std::cout << APP_VERSION << '\n'; return 0; }

        Problem p;
        p.opt = std::move(opt);
        if (!p.opt.input_terms.empty()) {
            load_input_terms(p);
        } else {
            configure_window(p);
            validate_problem(p);
            allocate_bitmaps(p, true);
        }
        allocate_factor_storage(p);
        apply_factor_file(p);

        if (!p.opt.output_terms.empty() && p.opt.output_terms != p.opt.input_terms &&
            file_exists(p.opt.output_terms))
            fail("output file already exists: " + p.opt.output_terms +
                 " (resume with -i FILE and choose a new -o, or remove/move the old file)");

        if (!p.opt.quiet) {
            const char* mode = p.opt.only_twins ? "paired" : "independent";
            std::cout << "GPRSV_N v" << APP_VERSION << "\n"
                      << "k=" << p.opt.k << ", base=" << p.opt.base
                      << ", n in [" << p.opt.nmin << ", " << p.opt.nmax << "] ("
                      << p.length << " exponents), " << mode << " mode, terms="
                      << active_terms(p) << ", p in (" << p.opt.min_prime << ", "
                      << p.opt.max_prime << "]\n"
                      << "window anchor=" << p.base_n << ", m=" << p.m
                      << ", giant steps=" << p.giants
                      << ", backend=" << (p.opt.backend == 0 ? "walk"
                                            : p.opt.backend == 1 ? "bsgs"
                                            : (p.length > 256 ? "bsgs" : "walk"))
                      << "\n";
        }

        uint64_t largest = p.opt.min_prime;
        if (!p.opt.apply_and_exit) largest = run_sieve(p);
        write_terms(p, largest);
        write_expressions(p);
        uint64_t factors = write_factors(p);
        if (!p.opt.output_factors.empty())
            std::cout << "Wrote " << factors << " new factors to " << p.opt.output_factors << ".\n";
        std::cout << "Wrote " << active_terms(p) << " remaining terms to " << p.opt.output_terms << ".\n";
        if (!p.opt.expression_file.empty())
            std::cout << "Wrote expressions to " << p.opt.expression_file << ".\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}

