/*
   GNCWSV.cu

   SPDX-License-Identifier: GPL-2.0-or-later
   Copyright (C) 2026 AstralPrisma (A.P.).

   CUDA sieve for Generalized Cullen / Woodall and Generalized Near
   Cullen / Near Woodall candidates with fixed base b and variable a:
       a*b^a-1, a*b^a+1,
       a*b^(a+1)-1, a*b^(a-1)-1, a*b^(a+1)+1, a*b^(a-1)+1.

   Prime generation / CUDA infrastructure is derived from the GPLv2-or-later
   GSRSV/mtsieve-based implementation.  The recurrence used for varying
   exponents follows the standard Generalized Cullen/Woodall sieve idea:
   test (-c)*b^(-e) == a (mod p) and update powers by fixed exponent gaps.

   Distributed under the GNU General Public License, version 2 or later.
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
#endif
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace gncw_cuda {
constexpr uint64_t PMAX_MAX = (UINT64_C(1) << 62) - 1;
constexpr uint32_t BMAX_MAX = (UINT32_C(1) << 31);
constexpr uint64_t AMAX_MAX = UINT64_C(0xfffffffe); // keeps a+1 in uint32 range
constexpr const char* APP_VERSION = "1.0";
constexpr const char* BUILD_ID = "20260812-1140-v13c";
enum class PrimeMode : int { Auto = 0, PrimeSieve, Segmented, MillerRabin };
enum class GncwMode : int {
    Unknown = 0,
    Woodall      = 1, // a*b^a-1      generalized Woodall
    Cullen       = 2, // a*b^a+1      generalized Cullen
    NearWoodall1 = 3, // a*b^(a+1)-1  == (n-1)*b^n-1
    NearWoodall2 = 4, // a*b^(a-1)-1  == (n+1)*b^n-1
    NearCullen1  = 5, // a*b^(a+1)+1  == (n-1)*b^n+1
    NearCullen2  = 6  // a*b^(a-1)+1  == (n+1)*b^n+1
};

struct Options {
    uint64_t min_a = 0;
    uint64_t max_a = 0;
    bool min_a_explicit = false;
    bool max_a_explicit = false;
    uint32_t base = 0;
    bool base_explicit = false;
    GncwMode mode = GncwMode::Unknown;
    bool mode_explicit = false;

    uint64_t min_prime = 1;          // primes p satisfy min_prime < p
    uint64_t max_prime = PMAX_MAX;   // and p <= max_prime
    bool min_prime_explicit = false;
    bool max_prime_explicit = false;

    std::string input_terms;
    std::string output_terms;
    std::string input_factors;
    std::string output_factors;
    bool apply_and_exit = false;

    int device = 0;
    int threads = 256;               // one CUDA block cooperates on one prime
    uint32_t sparse_hot_gaps = 6;    // compact path: specialized shared cache; RTX 4060 benchmark sweet spot; allowed 0,2,4,6,8,12,16
    int blocks = 0;                  // 0 = SM count * 8 resident/grid blocks
    uint64_t batch_primes = UINT64_C(1) << 18;
    uint64_t cpu_small_prime = 2;    // p=2 is structural after parity normalization
    uint32_t segment_mib = 8;
    PrimeMode prime_mode = PrimeMode::Auto;
    uint64_t mr_switch_sqrt = 100000000;
    uint32_t prime_prefetch = 16;
    uint32_t prime_threads = 0;
    uint32_t prime_region_batches = 12;
    uint32_t cuda_streams = 1;
    uint32_t progress_seconds = 60;
    bool verify_factors = false;
    bool quiet = false;

    bool help = false;
    bool version = false;
};

struct Problem {
    Options opt;
    uint64_t stride = 1;             // 2 for odd base (only even a can survive parity)
    uint64_t candidate_count = 0;
    std::vector<uint32_t> term_bits;

    // Allocated only when -O/--outputfactors is used. Stores first new factor.
    std::vector<uint64_t> factor;
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
        ::gncw_cuda::fail(_oss.str()); \
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

static std::string option_value(int& i, int argc, char** argv, const std::string& arg) {
    size_t eq = arg.find('=');
    if (eq != std::string::npos) return arg.substr(eq + 1);
    if (i + 1 >= argc) fail("Missing value after " + arg);
    return argv[++i];
}

static void parse_options(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { o.help = true; continue; }
        if (arg == "--version") { o.version = true; continue; }
        if (arg == "--applyandexit") { o.apply_and_exit = true; continue; }
        if (arg == "--verify") { o.verify_factors = true; continue; }
        if (arg == "--quiet") { o.quiet = true; continue; }

        auto is_long = [&](const char* name) {
            std::string n(name);
            return arg == n || arg.rfind(n + "=", 0) == 0;
        };
        auto short_value = [&](char c) -> std::string {
            if (arg.size() > 2) return arg.substr(2);
            if (i + 1 >= argc) fail(std::string("Missing value after -") + c);
            return argv[++i];
        };

        if (arg.rfind("-a", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.min_a = parse_u64(short_value('a'), 1, AMAX_MAX, "amin");
            o.min_a_explicit = true;
        } else if (is_long("--amin")) {
            o.min_a = parse_u64(option_value(i, argc, argv, arg), 1, AMAX_MAX, "amin");
            o.min_a_explicit = true;
        } else if (arg.rfind("-A", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.max_a = parse_u64(short_value('A'), 1, AMAX_MAX, "amax");
            o.max_a_explicit = true;
        } else if (is_long("--amax")) {
            o.max_a = parse_u64(option_value(i, argc, argv, arg), 1, AMAX_MAX, "amax");
            o.max_a_explicit = true;
        } else if (arg.rfind("-b", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.base = static_cast<uint32_t>(parse_u64(short_value('b'), 2, BMAX_MAX, "base"));
            o.base_explicit = true;
        } else if (is_long("--base")) {
            o.base = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, arg), 2, BMAX_MAX, "base"));
            o.base_explicit = true;
        } else if (arg.rfind("-m", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.mode = static_cast<GncwMode>(parse_int(short_value('m'), 1, 6, "mode"));
            o.mode_explicit = true;
        } else if (is_long("--mode")) {
            o.mode = static_cast<GncwMode>(parse_int(option_value(i, argc, argv, arg), 1, 6, "mode"));
            o.mode_explicit = true;
        } else if (arg.rfind("-p", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.min_prime = parse_u64(short_value('p'), 1, PMAX_MAX, "pmin");
            o.min_prime_explicit = true;
        } else if (is_long("--pmin")) {
            o.min_prime = parse_u64(option_value(i, argc, argv, arg), 1, PMAX_MAX, "pmin");
            o.min_prime_explicit = true;
        } else if (arg.rfind("-P", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.max_prime = parse_u64(short_value('P'), 2, PMAX_MAX, "pmax");
            o.max_prime_explicit = true;
        } else if (is_long("--pmax")) {
            o.max_prime = parse_u64(option_value(i, argc, argv, arg), 2, PMAX_MAX, "pmax");
            o.max_prime_explicit = true;
        } else if (arg.rfind("-w", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.batch_primes = parse_u64(short_value('w'), 128, UINT64_C(1) << 30, "batch-primes");
        } else if (is_long("--worksize") || is_long("--batch-primes")) {
            o.batch_primes = parse_u64(option_value(i, argc, argv, arg), 128, UINT64_C(1) << 30, "batch-primes");
        } else if (arg.rfind("-i", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.input_terms = short_value('i');
        } else if (is_long("--inputterms") || is_long("--input")) {
            o.input_terms = option_value(i, argc, argv, arg);
        } else if (arg.rfind("-o", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.output_terms = short_value('o');
        } else if (is_long("--outputterms") || is_long("--output")) {
            o.output_terms = option_value(i, argc, argv, arg);
        } else if (arg.rfind("-I", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.input_factors = short_value('I');
        } else if (is_long("--inputfactors")) {
            o.input_factors = option_value(i, argc, argv, arg);
        } else if (arg.rfind("-O", 0) == 0 && arg.rfind("--", 0) != 0) {
            o.output_factors = short_value('O');
        } else if (is_long("--outputfactors")) {
            o.output_factors = option_value(i, argc, argv, arg);
        } else if (is_long("--device")) {
            o.device = parse_int(option_value(i, argc, argv, arg), 0, 255, "device");
        } else if (is_long("--threads")) {
            o.threads = parse_int(option_value(i, argc, argv, arg), 32, 1024, "threads");
        } else if (is_long("--hot-gaps")) {
            const uint64_t h = parse_u64(option_value(i, argc, argv, arg), 0, 16, "hot-gaps");
            if (!(h == 0 || h == 2 || h == 4 || h == 6 || h == 8 || h == 12 || h == 16))
                fail("--hot-gaps must be one of 0, 2, 4, 6, 8, 12, or 16");
            o.sparse_hot_gaps = static_cast<uint32_t>(h);
        } else if (is_long("--blocks") || is_long("--gpuworkgroups")) {
            o.blocks = parse_int(option_value(i, argc, argv, arg), 0, 1000000, "blocks");
        } else if (is_long("--cpu-small-prime")) {
            o.cpu_small_prime = parse_u64(option_value(i, argc, argv, arg), 0, PMAX_MAX, "cpu-small-prime");
        } else if (is_long("--segment-mib")) {
            o.segment_mib = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, arg), 1, 2048, "segment-mib"));
        } else if (is_long("--mr-switch-sqrt")) {
            o.mr_switch_sqrt = parse_u64(option_value(i, argc, argv, arg), 1000, UINT32_MAX, "mr-switch-sqrt");
        } else if (is_long("--prime-generator")) {
            std::string v = option_value(i, argc, argv, arg);
            if (v == "auto") o.prime_mode = PrimeMode::Auto;
            else if (v == "primesieve" || v == "ps") o.prime_mode = PrimeMode::PrimeSieve;
            else if (v == "segmented") o.prime_mode = PrimeMode::Segmented;
            else if (v == "mr") o.prime_mode = PrimeMode::MillerRabin;
            else fail("prime-generator must be auto, primesieve, segmented, or mr");
        } else if (is_long("--prime-prefetch")) {
            o.prime_prefetch = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, arg), 1, 64, "prime-prefetch"));
        } else if (is_long("--prime-region-batches")) {
            o.prime_region_batches = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, arg), 1, 256, "prime-region-batches"));
        } else if (is_long("--cuda-streams")) {
            o.cuda_streams = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, arg), 1, 4, "cuda-streams"));
        } else if (is_long("--prime-threads")) {
            o.prime_threads = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, arg), 0, 64, "prime-threads"));
        } else if (arg == "--workers" || arg.rfind("--workers=", 0) == 0) {
            o.prime_threads = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, arg), 0, 64, "workers"));
        } else if (is_long("--progress-seconds") || is_long("--report-seconds")) {
            o.progress_seconds = static_cast<uint32_t>(parse_u64(option_value(i, argc, argv, arg), 1, 86400, "progress-seconds"));
        } else {
            fail("Unknown option: " + arg);
        }
    }

    if (o.threads % 32 != 0) fail("--threads must be a multiple of 32");
}

static void print_help() {
    std::cout
        << "GNCWSV v" << APP_VERSION << "\n"
        << "CUDA sieve for Generalized Cullen/Woodall and Near Cullen/Woodall forms with fixed base b.\n\n"
        << "Candidate modes (a varies; exponent is derived automatically):\n"
        << "  --mode 1   a*b^a-1       Generalized Woodall\n"
        << "  --mode 2   a*b^a+1       Generalized Cullen\n"
        << "  --mode 3   a*b^(a+1)-1   Near Woodall, 1st kind: (n-1)*b^n-1\n"
        << "  --mode 4   a*b^(a-1)-1   Near Woodall, 2nd kind: (n+1)*b^n-1\n"
        << "  --mode 5   a*b^(a+1)+1   Near Cullen,  1st kind: (n-1)*b^n+1\n"
        << "  --mode 6   a*b^(a-1)+1   Near Cullen,  2nd kind: (n+1)*b^n+1\n\n"
        << "Core options:\n"
        << "  -a, --amin A0            minimum a\n"
        << "  -A, --amax A1            maximum a\n"
        << "  -b, --base B              fixed base b\n"
        << "  -m, --mode {1|2|3|4|5|6}  one of the six forms above\n"
        << "  -p, --pmin P0             test primes p > P0\n"
        << "  -P, --pmax P1             test primes p <= P1 (e.g. 1e12)\n"
        << "  -i, --inputterms FILE     resume from plain expression-per-line file\n"
        << "  -o, --outputterms FILE    write survivors, one full expression per line\n"
        << "  -I, --inputfactors FILE   apply existing 'p | expression' factors\n"
        << "  -O, --outputfactors FILE  append newly found 'p | expression' factors\n"
        << "      --applyandexit        apply -I and write output without sieving\n\n"
        << "Input/output expression syntax:\n"
        << "  879536*3^879537-1\n"
        << "  864194*3^864195+1\n"
        << "Output starts with a '# GNCWSV ...' resume header, then one expression per line.\n"
        << "All '#' comment lines are ignored while reading candidate expressions.\n\n"
        << "CUDA / prime-generation options:\n"
        << "      --device D             CUDA device index (default 0)\n"
        << "      --threads T            CUDA threads/block; dense uses a block/prime, compact a thread/prime (default 256)\n"
        << "      --hot-gaps H           compact shared hot-gap specialization: 0, 2, 4, 6, 8, 12, or 16 (default 6)\n"
        << "      --blocks B             grid blocks; 0=auto (8/SM dense, 24/SM compact)\n"
        << "  -w, --batch-primes N       primes per producer batch (default 262144)\n"
        << "      --cpu-small-prime P    CPU handles p <= P (default 2)\n"
        << "      --prime-generator M    auto, primesieve, segmented, or mr\n"
        << "      --prime-prefetch N     queued prime batches (default 16)\n"
        << "      --prime-threads N      prime-generator workers; 0=auto\n"
        << "      --prime-region-batches N  batches per libprimesieve region (default 12)\n"
        << "      --cuda-streams N       CUDA streams/buffers, 1-4 (default 1)\n"
        << "      --progress-seconds N   progress/ETA print interval; rate is always last 60s (default 60)\n"
        << "      --segment-mib M        MiB per built-in worker segment (default 8)\n"
        << "      --verify               verify every reported factor on CPU\n"
        << "      --quiet                reduce progress output\n\n"
        << "The former efficiency/no-factor timeout system is intentionally removed.\n";
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

static uint64_t active_terms(const Problem& p) {
    return count_bits(p.term_bits);
}

static std::vector<uint32_t> collect_active_indices(const Problem& p) {
    if (p.candidate_count > static_cast<uint64_t>(UINT32_MAX))
        fail("Internal candidate lattice exceeds compact-index range");
    std::vector<uint32_t> out;
    out.reserve(static_cast<size_t>(active_terms(p)));
    for (size_t wi = 0; wi < p.term_bits.size(); ++wi) {
        uint32_t word = p.term_bits[wi];
        while (word) {
            uint32_t bit = 0;
#if defined(_MSC_VER)
            unsigned long b = 0;
            _BitScanForward(&b, word);
            bit = static_cast<uint32_t>(b);
#else
            bit = static_cast<uint32_t>(__builtin_ctz(word));
#endif
            const uint64_t idx = static_cast<uint64_t>(wi) * 32ULL + bit;
            if (idx < p.candidate_count) out.push_back(static_cast<uint32_t>(idx));
            word &= word - 1U;
        }
    }
    return out;
}

static bool a_to_index(const Problem& p, uint64_t a, uint64_t& idx) {
    if (a < p.opt.min_a || a > p.opt.max_a) return false;
    uint64_t d = a - p.opt.min_a;
    if (d % p.stride != 0) return false;
    idx = d / p.stride;
    return idx < p.candidate_count;
}

static uint64_t index_to_a(const Problem& p, uint64_t idx) {
    return p.opt.min_a + idx * p.stride;
}

static int mode_offset(GncwMode mode) {
    switch (mode) {
        case GncwMode::Woodall:
        case GncwMode::Cullen:       return 0;
        case GncwMode::NearWoodall1:
        case GncwMode::NearCullen1:  return +1;
        case GncwMode::NearWoodall2:
        case GncwMode::NearCullen2:  return -1;
        default: fail("Internal invalid GNCW mode");
    }
}

static int mode_sign(GncwMode mode) {
    switch (mode) {
        case GncwMode::Woodall:
        case GncwMode::NearWoodall1:
        case GncwMode::NearWoodall2: return -1;
        case GncwMode::Cullen:
        case GncwMode::NearCullen1:
        case GncwMode::NearCullen2:  return +1;
        default: fail("Internal invalid GNCW mode");
    }
}

static const char* mode_name(GncwMode mode) {
    switch (mode) {
        case GncwMode::Woodall:      return "Generalized Woodall";
        case GncwMode::Cullen:       return "Generalized Cullen";
        case GncwMode::NearWoodall1: return "Near Woodall 1st";
        case GncwMode::NearWoodall2: return "Near Woodall 2nd";
        case GncwMode::NearCullen1:  return "Near Cullen 1st";
        case GncwMode::NearCullen2:  return "Near Cullen 2nd";
        default: return "unknown";
    }
}

static uint64_t exponent_for_a(const Problem& p, uint64_t a) {
    const int off = mode_offset(p.opt.mode);
    if (off > 0) return a + 1;
    if (off == 0) return a;
    if (a == 0) fail("Internal exponent underflow");
    return a - 1;
}

static std::string term_text(const Problem& p, uint64_t a) {
    std::ostringstream oss;
    oss << a << '*' << p.opt.base << '^' << exponent_for_a(p, a)
        << (mode_sign(p.opt.mode) > 0 ? "+1" : "-1");
    return oss.str();
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

static bool quotient_is_exact_power(uint64_t q, uint32_t base, uint64_t exponent) {
    if (exponent == 0) return q == 1;
    uint64_t used = 0;
    while (q > 1 && used < exponent) {
        if (q % base != 0) return false;
        q /= base;
        ++used;
    }
    return q == 1 && used == exponent;
}

static bool term_equals_prime_host(const Problem& pr, uint64_t p, uint64_t a) {
    const int sign = mode_sign(pr.opt.mode);
    uint64_t product = 0;
    if (sign > 0) {
        if (p == 0) return false;
        product = p - 1;
    } else {
        if (p == std::numeric_limits<uint64_t>::max()) return false;
        product = p + 1;
    }
    if (a == 0 || product % a != 0) return false;
    return quotient_is_exact_power(product / a, pr.opt.base, exponent_for_a(pr, a));
}

static bool factor_is_valid(const Problem& pr, uint64_t factor, uint64_t a) {
    if (factor < 2) return false;
    uint64_t power = pow_mod_host(pr.opt.base, exponent_for_a(pr, a), factor);
    uint64_t r = mul_mod_host(a % factor, power, factor);
    if (mode_sign(pr.opt.mode) > 0)
        r = (r + 1 == factor) ? 0 : r + 1;
    else
        r = (r == 0) ? factor - 1 : r - 1;
    return r == 0 && !term_equals_prime_host(pr, factor, a);
}

// ------------------------- primality and prime streams -------------------------

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
// by mtsieve.  It is loaded dynamically so GSRSV remains one .cu source file
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

    void load() {
#if defined(_WIN32)
        static const char* names[] = {
            "primesieve.dll", "libprimesieve.dll"
        };
        for (const char* name : names) {
            handle_ = LoadLibraryA(name);
            if (handle_) { loaded_name_ = name; break; }
        }
        if (!handle_) {
            error_text_ = "primesieve.dll was not found";
            return;
        }
#else
        static const char* names[] = {
            "libprimesieve.so.13", "libprimesieve.so.12",
            "libprimesieve.so.11", "libprimesieve.so"
#if defined(__APPLE__)
            , "libprimesieve.dylib"
#endif
        };
        for (const char* name : names) {
            dlerror();
            handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (handle_) { loaded_name_ = name; break; }
        }
        if (!handle_) {
            const char* e = dlerror();
            error_text_ = e ? e : "libprimesieve was not found";
            return;
        }
#endif
        init_ = symbol<InitFn>("primesieve_init");
        free_ = symbol<FreeFn>("primesieve_free_iterator");
        jump_ = symbol<JumpFn>("primesieve_jump_to");
        skip_ = symbol<JumpFn>("primesieve_skipto");
        generate_ = symbol<GenerateFn>("primesieve_generate_next_primes");
        if (!available()) {
            error_text_ = "loaded library is missing the primesieve iterator API";
            close();
        }
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

// ------------------------- candidate construction and plain-text parsing -------------------------

struct ParsedExpression {
    uint64_t a = 0;
    uint32_t base = 0;
    uint64_t exponent = 0;
    int sign = 0;
    GncwMode mode = GncwMode::Unknown;
};

struct ResumeHeader {
    bool present = false;
    uint64_t p = 0;
    uint32_t base = 0;
    GncwMode mode = GncwMode::Unknown;
    uint64_t amin = 0;
    uint64_t amax = 0;
};

struct InputTermsData {
    std::vector<ParsedExpression> entries;
    ResumeHeader header;
};

static GncwMode mode_from_offset_sign(int offset, int sign) {
    if (offset == 0  && sign == -1) return GncwMode::Woodall;
    if (offset == 0  && sign == +1) return GncwMode::Cullen;
    if (offset == +1 && sign == -1) return GncwMode::NearWoodall1;
    if (offset == -1 && sign == -1) return GncwMode::NearWoodall2;
    if (offset == +1 && sign == +1) return GncwMode::NearCullen1;
    if (offset == -1 && sign == +1) return GncwMode::NearCullen2;
    fail("Internal invalid offset/sign pair");
}

static ParsedExpression parse_expression_line(const std::string& raw, uint64_t line_number = 0) {
    std::string line = raw;
    size_t comment = line.find("//");
    if (comment != std::string::npos) line.resize(comment);
    line = trim(line);

    std::smatch m;
    static const std::regex re(
        R"(^\s*(\d+)\s*\*\s*(\d+)\s*\^\s*(\d+)\s*([+-])\s*1\s*$)",
        std::regex::icase);
    if (!std::regex_match(line, m, re)) {
        std::ostringstream oss;
        if (line_number) oss << "line " << line_number << ": ";
        oss << "expected full expression like 879536*3^879537-1, got '" << raw << "'";
        fail(oss.str());
    }

    ParsedExpression x;
    x.a = parse_u64(m[1].str(), 1, AMAX_MAX, "a");
    x.base = static_cast<uint32_t>(parse_u64(m[2].str(), 2, BMAX_MAX, "base"));
    x.exponent = parse_u64(m[3].str(), 0, UINT64_C(0xffffffff), "exponent");
    x.sign = (m[4].str() == "+") ? +1 : -1;

    int offset = 0;
    if (x.exponent == x.a) offset = 0;
    else if (x.exponent == x.a + 1) offset = +1;
    else if (x.a >= 1 && x.exponent + 1 == x.a) offset = -1;
    else {
        std::ostringstream oss;
        if (line_number) oss << "line " << line_number << ": ";
        oss << "exponent must be a, a+1, or a-1: " << line;
        fail(oss.str());
    }
    if (offset == -1 && x.a < 2) {
        std::ostringstream oss;
        if (line_number) oss << "line " << line_number << ": ";
        oss << "2nd-kind Near Cullen/Woodall requires a>=2 (positive exponent): " << line;
        fail(oss.str());
    }
    x.mode = mode_from_offset_sign(offset, x.sign);
    return x;
}

static void configure_candidate_lattice(Problem& p) {
    p.stride = 1;
    if (p.opt.base & 1U) {
        // For odd b, b^e is odd. Odd a would make a*b^e +/- 1 even (>2),
        // so only even a need to enter the sieve.
        p.stride = 2;
        if (p.opt.min_a & 1ULL) ++p.opt.min_a;
        if (p.opt.max_a & 1ULL) --p.opt.max_a;
    }
    if (p.opt.max_a < p.opt.min_a)
        fail("No candidates remain after parity normalization");
    p.candidate_count = (p.opt.max_a - p.opt.min_a) / p.stride + 1;
}

static ResumeHeader parse_resume_header(const std::string& stripped, uint64_t line_number) {
    ResumeHeader h;
    static const std::regex re(
        R"(^#\s*GNCWSV\s+p=(\d+)\s+base=(\d+)\s+mode=(\d+)\s+amin=(\d+)\s+amax=(\d+)\s*$)",
        std::regex::icase);
    std::smatch m;
    if (!std::regex_match(stripped, m, re)) return h;

    h.present = true;
    h.p = parse_u64(m[1].str(), 1, PMAX_MAX, "resume p");
    h.base = static_cast<uint32_t>(parse_u64(m[2].str(), 2, BMAX_MAX, "resume base"));
    // Current GNCWSV output writes the public six-mode numbering directly
    // (1..6).  The previous parser accidentally classified the same GNCWSV
    // tag as a four-mode legacy format, which made valid mode 5/6 resume
    // files impossible to reload.
    h.mode = static_cast<GncwMode>(parse_int(m[3].str(), 1, 6, "resume mode"));
    h.amin = parse_u64(m[4].str(), 1, AMAX_MAX, "resume amin");
    h.amax = parse_u64(m[5].str(), 1, AMAX_MAX, "resume amax");
    if (h.amin > h.amax) {
        std::ostringstream oss;
        oss << "line " << line_number << ": resume header amin exceeds amax";
        fail(oss.str());
    }
    return h;
}

static InputTermsData read_expression_file(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) fail("Unable to open input terms file: " + filename);
    InputTermsData data;
    std::string raw;
    uint64_t line_number = 0;
    while (std::getline(in, raw)) {
        ++line_number;
        std::string stripped = trim(raw);
        if (stripped.empty() || stripped.rfind("//", 0) == 0) continue;
        if (stripped.rfind("#", 0) == 0) {
            ResumeHeader h = parse_resume_header(stripped, line_number);
            if (h.present) {
                if (data.header.present)
                    fail("Multiple GNCWSV resume headers in input terms file");
                data.header = h;
            }
            continue; // all # lines are comments for expression parsing
        }
        data.entries.push_back(parse_expression_line(raw, line_number));
    }
    if (data.entries.empty()) fail("Input terms file contains no candidate expressions");
    return data;
}

static void load_input_terms(Problem& p) {
    InputTermsData input = read_expression_file(p.opt.input_terms);
    std::vector<ParsedExpression>& entries = input.entries;
    const uint32_t file_base = entries.front().base;
    const GncwMode file_mode = entries.front().mode;

    for (const auto& x : entries) {
        if (x.base != file_base)
            fail("Input file contains multiple bases; GNCWSV requires one fixed base per run");
        if (x.mode != file_mode)
            fail("Input file contains multiple GNCW modes; use one --mode per run");
    }

    if (input.header.present) {
        if (input.header.base != file_base)
            fail("Resume header base does not match expressions in input file");
        if (input.header.mode != file_mode)
            fail("Resume header mode does not match expressions in input file");
        for (const auto& x : entries) {
            if (x.a < input.header.amin || x.a > input.header.amax)
                fail("Resume header a-range does not contain every expression in input file");
        }

        if (p.opt.min_prime_explicit && p.opt.min_prime > input.header.p) {
            std::ostringstream oss;
            oss << "--pmin=" << p.opt.min_prime << " is above resume header p="
                << input.header.p << "; this would skip an unsieved prime interval";
            fail(oss.str());
        }
        if (p.opt.min_prime < input.header.p) {
            if (p.opt.min_prime_explicit && !p.opt.quiet)
                std::cout << "Raising --pmin from " << p.opt.min_prime
                          << " to resume header p=" << input.header.p << ".\n";
            p.opt.min_prime = input.header.p;
        }
    }

    if (p.opt.base_explicit && p.opt.base != file_base)
        fail("--base does not match expressions in input file");
    if (p.opt.mode_explicit && p.opt.mode != file_mode)
        fail("--mode does not match expressions in input file");
    p.opt.base = file_base;
    p.opt.mode = file_mode;

    uint64_t file_min = AMAX_MAX, file_max = 0;
    for (const auto& x : entries) {
        file_min = std::min(file_min, x.a);
        file_max = std::max(file_max, x.a);
    }
    // Candidate files are explicit lists.  On resume, the historical amin/amax
    // in the header does not need to be kept as live lattice space: terms that
    // are no longer present can never reappear.  Shrink the implicit bounds to
    // the current file contents unless the user explicitly requested bounds.
    // The resume prime p is still honored above, so no prime interval is skipped.
    if (!p.opt.min_a_explicit) p.opt.min_a = file_min;
    if (!p.opt.max_a_explicit) p.opt.max_a = file_max;
    if (p.opt.min_a > p.opt.max_a) fail("amin must not exceed amax");
    if (mode_offset(p.opt.mode) < 0 && p.opt.min_a < 2)
        fail("Modes 4 and 6 require a>=2");

    configure_candidate_lattice(p);
    fill_bits(p.term_bits, p.candidate_count, false);

    uint64_t kept = 0, parity_dropped = 0;
    for (const auto& x : entries) {
        if (x.a < p.opt.min_a || x.a > p.opt.max_a) continue;
        uint64_t idx = 0;
        if (!a_to_index(p, x.a, idx)) {
            ++parity_dropped;
            continue;
        }
        if (!bit_test(p.term_bits, idx)) {
            bit_set(p.term_bits, idx);
            ++kept;
        }
    }
    if (kept == 0) fail("No input candidates remain in the selected range");
    if (parity_dropped && !p.opt.quiet)
        std::cout << "Discarded " << parity_dropped
                  << " trivially even input candidate(s) by parity.\n";
}

static void initialize_generated_terms(Problem& p) {
    if (!p.opt.min_a_explicit || !p.opt.max_a_explicit)
        fail("--amin and --amax are required without -i/--inputterms");
    if (!p.opt.base_explicit) fail("--base is required without -i/--inputterms");
    if (!p.opt.mode_explicit) fail("--mode {1|2|3|4|5|6} is required without -i/--inputterms");
    if (p.opt.min_a > p.opt.max_a) fail("amin must not exceed amax");
    if (mode_offset(p.opt.mode) < 0 && p.opt.min_a < 2)
        fail("Modes 4 and 6 require a>=2");

    configure_candidate_lattice(p);
    fill_bits(p.term_bits, p.candidate_count, true);
}

static void allocate_factor_storage(Problem& p) {
    if (p.opt.output_factors.empty()) return;
    p.factor.assign(static_cast<size_t>(p.candidate_count), 0);
}

static void normalize_and_validate_problem(Problem& p) {
    if (!p.opt.input_terms.empty()) load_input_terms(p);
    else initialize_generated_terms(p);

    // p=2 cannot divide any parity-surviving candidate.
    p.opt.min_prime = std::max<uint64_t>(p.opt.min_prime, 2);
    if (p.opt.max_prime <= p.opt.min_prime) fail("pmax must be greater than pmin");

    allocate_factor_storage(p);

    if (p.opt.output_terms.empty()) {
        std::ostringstream name;
        name << "gncw_b" << p.opt.base << "_m" << static_cast<int>(p.opt.mode)
             << "_a" << p.opt.min_a << '_' << p.opt.max_a << ".txt";
        p.opt.output_terms = name.str();
    }
}

// ------------------------- apply factor file -------------------------

static void apply_factor_file(Problem& p) {
    if (p.opt.input_factors.empty()) return;
    std::ifstream in(p.opt.input_factors);
    if (!in) fail("Unable to open factor file: " + p.opt.input_factors);

    std::string line;
    uint64_t line_number = 0, read = 0, applied = 0, unrelated = 0;
    while (std::getline(in, line)) {
        ++line_number;
        std::string stripped = trim(line);
        if (stripped.empty() || stripped.rfind("//", 0) == 0 || stripped.rfind("#", 0) == 0)
            continue;

        size_t bar = stripped.find('|');
        if (bar == std::string::npos)
            fail("Malformed factor line " + std::to_string(line_number) + ": " + line);

        uint64_t factor = parse_u64(trim(stripped.substr(0, bar)), 2, PMAX_MAX, "factor");
        ParsedExpression x = parse_expression_line(trim(stripped.substr(bar + 1)), line_number);
        ++read;

        // Factor files may contain several GNCW jobs.  Ignore other families/bases.
        if (x.base != p.opt.base || x.mode != p.opt.mode) {
            ++unrelated;
            continue;
        }

        uint64_t idx = 0;
        if (!a_to_index(p, x.a, idx)) continue;
        if (!bit_test(p.term_bits, idx)) continue;
        if (!factor_is_valid(p, factor, x.a))
            fail("Invalid factor on line " + std::to_string(line_number) + ": " + line);
        if (bit_clear(p.term_bits, idx)) ++applied;
    }

    if (!p.opt.quiet) {
        std::cout << "Applied " << applied << " of " << read << " input factor line(s)";
        if (unrelated) std::cout << " (ignored " << unrelated << " other base/mode line(s))";
        std::cout << ".\n";
    }
}

// ------------------------- CUDA exact modular arithmetic -------------------------

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

// Sparse-path Montgomery multiplier specialized for moduli below 2^62.
// It keeps the same R=2^64 representation as d_mont_mul(), but performs
// the product and REDC as two 32-bit limbs.  On NVIDIA GPUs this maps the
// multiplications to mul.wide.u32 instead of the much lower-throughput
// general 64x64 high-half path.
__device__ __forceinline__ uint64_t d_mont_mul_2x32(uint64_t a, uint64_t b, const Mont64& c) {
    const uint32_t a0 = static_cast<uint32_t>(a);
    const uint32_t a1 = static_cast<uint32_t>(a >> 32);
    const uint32_t b0 = static_cast<uint32_t>(b);
    const uint32_t b1 = static_cast<uint32_t>(b >> 32);
    const uint32_t n0 = static_cast<uint32_t>(c.mod);
    const uint32_t n1 = static_cast<uint32_t>(c.mod >> 32);
    const uint32_t ni = static_cast<uint32_t>(c.ninv);

    const uint64_t p00 = static_cast<uint64_t>(a0) * b0;
    const uint64_t p01 = static_cast<uint64_t>(a0) * b1;
    const uint64_t p10 = static_cast<uint64_t>(a1) * b0;
    const uint64_t p11 = static_cast<uint64_t>(a1) * b1;

    uint32_t t0 = static_cast<uint32_t>(p00);
    const uint64_t s1 = (p00 >> 32) + static_cast<uint32_t>(p01) + static_cast<uint32_t>(p10);
    uint32_t t1 = static_cast<uint32_t>(s1);
    const uint64_t s2 = (p01 >> 32) + (p10 >> 32) + static_cast<uint32_t>(p11) + (s1 >> 32);
    uint32_t t2 = static_cast<uint32_t>(s2);
    const uint64_t s3 = (p11 >> 32) + (s2 >> 32);
    uint32_t t3 = static_cast<uint32_t>(s3);
    uint32_t t4 = static_cast<uint32_t>(s3 >> 32);

    uint32_t m = static_cast<uint32_t>(static_cast<uint64_t>(t0) * ni);
    uint64_t u = static_cast<uint64_t>(t0) + static_cast<uint64_t>(m) * n0;
    uint64_t carry = u >> 32;
    u = static_cast<uint64_t>(t1) + static_cast<uint64_t>(m) * n1 + carry;
    t1 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t2) + carry; t2 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t3) + carry; t3 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t4) + carry; t4 = static_cast<uint32_t>(u);

    m = static_cast<uint32_t>(static_cast<uint64_t>(t1) * ni);
    u = static_cast<uint64_t>(t1) + static_cast<uint64_t>(m) * n0;
    carry = u >> 32;
    u = static_cast<uint64_t>(t2) + static_cast<uint64_t>(m) * n1 + carry;
    t2 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t3) + carry; t3 = static_cast<uint32_t>(u); carry = u >> 32;
    u = static_cast<uint64_t>(t4) + carry; t4 = static_cast<uint32_t>(u);

    // For this program c.mod < 2^62 and a,b < c.mod, so REDC(a*b) < 2*c.mod
    // and therefore the quotient after the two 32-bit reductions fits in 64 bits.
    uint64_t r = static_cast<uint64_t>(t2) | (static_cast<uint64_t>(t3) << 32);
    if (r >= c.mod) r -= c.mod;
    return r;
}

// Sparse fast path for p < 2^44.  In this range every Montgomery residue is
// also below 2^44, so a*b < 2^88.  The generic 2x32 routine above carries
// five 32-bit limbs to remain valid up to p < 2^62; here the product has only
// three live limbs.  Two radix-2^32 REDC steps can therefore be collapsed to
// short 64-bit accumulators without t3/t4 bookkeeping.  All multiplies are
// 32x32 (or narrower) promoted to uint64_t, so nvcc maps them to the native
// wide-u32 integer path.
__device__ __forceinline__ uint64_t d_mont_mul_fast44(uint64_t a, uint64_t b, const Mont64& c) {
    const uint32_t a0 = static_cast<uint32_t>(a);
    const uint32_t a1 = static_cast<uint32_t>(a >> 32);  // <= 12 bits
    const uint32_t b0 = static_cast<uint32_t>(b);
    const uint32_t b1 = static_cast<uint32_t>(b >> 32);  // <= 12 bits
    const uint32_t n0 = static_cast<uint32_t>(c.mod);
    const uint32_t n1 = static_cast<uint32_t>(c.mod >> 32); // <= 12 bits
    const uint32_t ni = static_cast<uint32_t>(c.ninv);

    // T = a*b = t0 + t1*B + t2*B^2, B=2^32.  Since a,b<2^44, t3=0.
    const uint64_t p00 = static_cast<uint64_t>(a0) * b0;
    const uint64_t cross = (p00 >> 32)
                         + static_cast<uint64_t>(a0) * b1
                         + static_cast<uint64_t>(a1) * b0;
    const uint32_t t0 = static_cast<uint32_t>(p00);
    uint32_t t1 = static_cast<uint32_t>(cross);
    uint64_t t2 = (cross >> 32) + static_cast<uint64_t>(a1) * b1;

    // First REDC limb: t0 + m0*n0 == 0 (mod B).
    const uint32_t m0 = static_cast<uint32_t>(static_cast<uint64_t>(t0) * ni);
    const uint64_t u0 = static_cast<uint64_t>(t0) + static_cast<uint64_t>(m0) * n0;
    const uint64_t u1 = static_cast<uint64_t>(t1)
                      + static_cast<uint64_t>(m0) * n1
                      + (u0 >> 32);
    t1 = static_cast<uint32_t>(u1);
    t2 += (u1 >> 32);

    // Second REDC limb.  After this division by B the result fits in <2*p.
    const uint32_t m1 = static_cast<uint32_t>(static_cast<uint64_t>(t1) * ni);
    const uint64_t v0 = static_cast<uint64_t>(t1) + static_cast<uint64_t>(m1) * n0;
    uint64_t r = t2 + static_cast<uint64_t>(m1) * n1 + (v0 >> 32);
    if (r >= c.mod) r -= c.mod;
    return r;
}

template <bool FAST44>
__device__ __forceinline__ uint64_t d_sparse_mont_mul(uint64_t a, uint64_t b, const Mont64& c) {
    if constexpr (FAST44) return d_mont_mul_fast44(a, b, c);
    else return d_mont_mul_2x32(a, b, c);
}

template <bool FAST44>
__device__ __forceinline__ uint64_t d_sparse_mont_pow_rep(uint64_t x, uint64_t e, const Mont64& c) {
    uint64_t r = c.rmod;
    while (e) {
        if (e & 1ULL) r = d_sparse_mont_mul<FAST44>(r, x, c);
        e >>= 1;
        if (e) x = d_sparse_mont_mul<FAST44>(x, x, c);
    }
    return r;
}

__device__ __forceinline__ uint64_t d_mont_pow_rep_2x32(uint64_t x, uint64_t e, const Mont64& c) {
    uint64_t r = c.rmod;
    while (e) {
        if (e & 1ULL) r = d_mont_mul_2x32(r, x, c);
        e >>= 1;
        if (e) x = d_mont_mul_2x32(x, x, c);
    }
    return r;
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

// Input/output are Montgomery representations.
__device__ __forceinline__ uint64_t d_mont_pow_rep(uint64_t x, uint64_t e, const Mont64& c) {
    uint64_t r = c.rmod;
    while (e) {
        if (e & 1ULL) r = d_mont_mul(r, x, c);
        e >>= 1;
        if (e) x = d_mont_mul(x, x, c);
    }
    return r;
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

// Exact inverse of a small uint32 base modulo p, avoiding a p-sized exponentiation.
__device__ __forceinline__ uint64_t d_inverse_base(uint32_t base, uint64_t p) {
    uint32_t r = static_cast<uint32_t>(p % base);
    if (r == 0) return 0;
    uint32_t inv_r = d_inverse_u32(r, base);
    if (inv_r == 0) return 0;
    uint64_t s = static_cast<uint64_t>(base - inv_r); // s*p == -1 (mod base)
    uint64_t q = p / base;
    uint64_t tail = (UINT64_C(1) + s * r) / base;
    return s * q + tail; // (1+s*p)/base
}

__device__ __forceinline__ bool d_quotient_is_exact_power(
    uint64_t q, uint32_t base, uint64_t exponent) {
    if (exponent == 0) return q == 1;
    uint64_t used = 0;
    while (q > 1 && used < exponent) {
        if (q % base != 0) return false;
        q /= base;
        ++used;
    }
    return q == 1 && used == exponent;
}

__device__ __forceinline__ bool d_term_equals_prime_gncw(
    uint64_t p, uint64_t a, uint32_t base, uint64_t exponent, int sign) {
    uint64_t product = sign > 0 ? (p - 1) : (p + 1);
    if (a == 0 || product % a != 0) return false;
    return d_quotient_is_exact_power(product / a, base, exponent);
}

// One CUDA block cooperates on one prime at a time.  For
//   N(a) = a*b^(a+offset) + sign, where offset is -1, 0, or +1,
// p divides N(a) iff
//   a == (-sign)*b^(-(a+offset)) (mod p).
// While scanning a downward by stride, the right hand side is updated by
// multiplication by b^stride.  Threads in a block scan interleaved a values.
//
// Dense-lattice path: unlike the original implementation, x_starts[] is not
// built serially by thread 0.  Every lane derives step^lane in parallel, which
// removes O(blockDim) serialized Montgomery multiplies from every prime.
__global__ void gncwsieve_dense_kernel(
    const uint64_t* primes, uint64_t prime_count,
    uint32_t base, int exponent_offset, int sign,
    uint64_t min_a, uint64_t max_a, uint64_t stride, uint64_t candidate_count,
    uint32_t* term_bits, uint64_t* factors,
    unsigned long long* removed_count) {

    __shared__ Mont64 mc;
    __shared__ uint64_t x0_m;
    __shared__ uint64_t step_m;
    __shared__ uint64_t jump_m;
    __shared__ uint64_t delta_a_m;
    __shared__ int valid_prime;

    for (uint64_t pi = blockIdx.x; pi < prime_count; pi += gridDim.x) {
        const uint64_t p = primes[pi];

        if (threadIdx.x == 0) {
            valid_prime = 0;
            const uint64_t base_mod = static_cast<uint64_t>(base) < p
                ? static_cast<uint64_t>(base)
                : static_cast<uint64_t>(base) % p;
            if (p > 2 && (p & 1ULL) && base_mod != 0 && candidate_count != 0) {
                mc = d_make_mont(p);
                const uint64_t inv_base = d_inverse_base(base, p);
                if (inv_base != 0) {
                    const uint64_t y_m = d_mont_mul(inv_base, mc.r2, mc);
                    const uint64_t max_exp = exponent_offset > 0 ? max_a + 1
                                           : exponent_offset < 0 ? max_a - 1
                                                                 : max_a;
                    x0_m = d_mont_pow_rep(y_m, max_exp, mc);
                    if (sign > 0 && x0_m != 0) x0_m = p - x0_m; // multiply by -1

                    uint64_t b_m = d_mont_mul(base_mod, mc.r2, mc);
                    step_m = b_m;
                    if (stride == 2) step_m = d_mont_mul(step_m, step_m, mc);
                    jump_m = d_mont_pow_rep(step_m, static_cast<uint64_t>(blockDim.x), mc);

                    const uint64_t delta_raw = stride * static_cast<uint64_t>(blockDim.x);
                    const uint64_t delta = delta_raw < p ? delta_raw : delta_raw % p;
                    delta_a_m = d_mont_mul(delta, mc.r2, mc);
                    valid_prime = 1;
                }
            }
        }
        __syncthreads();

        const uint64_t t = static_cast<uint64_t>(threadIdx.x);
        if (valid_prime && t < candidate_count) {
            uint64_t idx = candidate_count - 1 - t;
            uint64_t a = max_a - t * stride;

            // x(t) = x(0) * (b^stride)^t.  Computing this per lane costs only
            // O(log blockDim) dependent steps instead of the old 255-deep
            // thread-0 chain for a 256-thread block.
            uint64_t lane_step = d_mont_pow_rep(step_m, t, mc);
            uint64_t x_m = d_mont_mul(x0_m, lane_step, mc);
            const uint64_t a_mod = a < p ? a : a % p;
            uint64_t a_m = d_mont_mul(a_mod, mc.r2, mc);

            for (;;) {
                const uint64_t wi = idx >> 5;
                const uint32_t mask = UINT32_C(1) << (idx & 31);
                // Avoid the ordinary sieve special case where the candidate itself equals p.
                if (x_m == a_m && (term_bits[wi] & mask)) {
                    const uint64_t exponent = exponent_offset > 0 ? a + 1
                                              : exponent_offset < 0 ? a - 1
                                                                    : a;
                    if (!d_term_equals_prime_gncw(p, a, base, exponent, sign)) {
                        const uint32_t old = atomicAnd(term_bits + wi, ~mask);
                        if (old & mask) {
                            if (removed_count) atomicAdd(removed_count, 1ULL);
                            if (factors) factors[idx] = p;
                        }
                    }
                }

                if (idx < static_cast<uint64_t>(blockDim.x)) break;
                idx -= static_cast<uint64_t>(blockDim.x);
                a -= stride * static_cast<uint64_t>(blockDim.x);
                x_m = d_mont_mul(x_m, jump_m, mc);
                a_m = (a_m >= delta_a_m) ? (a_m - delta_a_m)
                                         : (a_m + (p - delta_a_m));
            }
        }
        __syncthreads();
    }
}

// Compact-list path, second generation.
//
// One CUDA thread owns one prime.  Threads in a warp therefore walk the same
// candidate/gap stream for 32 independent primes.  Each prime gets a small
// on-device table of the distinct gaps occurring in the current compact list:
//
//   Xmul[g] = (b^stride)^gap[g] * R (mod p)
//   Adel[g] = stride*gap[g]     * R (mod p)
//
// Both values are Montgomery representations.  After setup, every candidate
// transition is exactly one Montgomery multiply plus one modular subtraction;
// there is no per-candidate exponentiation and no per-candidate encode of a.
// Sparse compact path, third generation.
//
// One CUDA thread owns one prime.  X is deliberately kept in the ordinary
// residue domain, while each gap multiplier is kept in Montgomery form:
//
//   x_next = MontMul(x, (b^stride)^gap * R) = x*(b^stride)^gap (mod p).
//
// This means the hot loop can compare x directly with a (mod p).  The old
// 8-byte Montgomery a_delta entry disappears completely, as does the
// Montgomery a-state update.  For compact lists whose lattice gaps fit in one
// byte, the transition stream stores the gap itself and indexes a dense
// 1..max_gap x-multiplier table; irregular very-large-gap inputs retain the
// class-index fallback.
__global__ void gncwsieve_sparse_xonly_kernel(
    const uint64_t* primes, uint64_t prime_count,
    const uint32_t* active_indices, uint64_t active_count,
    const uint8_t* gap_steps8,
    const uint32_t* gap_ids, const uint32_t* unique_gaps,
    uint32_t gap_count, uint32_t max_gap, int direct_gap8,
    uint32_t base, int exponent_offset, int sign,
    uint64_t min_a, uint64_t stride,
    uint32_t* term_bits, uint64_t* factors,
    unsigned long long* removed_count,
    uint64_t* gap_x_table) {

    const uint64_t first_pi = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t pi_stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t pi = first_pi; pi < prime_count; pi += pi_stride) {
        const uint64_t p = primes[pi];
        if (p <= 2 || !(p & 1ULL) || active_count == 0) continue;

        const uint64_t base_mod = static_cast<uint64_t>(base) < p
            ? static_cast<uint64_t>(base)
            : static_cast<uint64_t>(base) % p;
        if (base_mod == 0) continue;

        const uint64_t inv_base = d_inverse_base(base, p);
        if (inv_base == 0) continue;

        const Mont64 mc = d_make_mont(p);
        const uint64_t inv_base_m = d_mont_mul_2x32(inv_base, mc.r2, mc);
        const uint64_t b_m = d_mont_mul_2x32(base_mod, mc.r2, mc);
        uint64_t step_m = b_m;
        if (stride == 2) step_m = d_mont_mul_2x32(step_m, step_m, mc);
        else if (stride != 1) step_m = d_mont_pow_rep_2x32(b_m, stride, mc);

        // Build one x multiplier per gap value for byte-sized gaps.  This is
        // denser than a class table, but with 8-byte entries it is still smaller
        // than the previous 16-byte class table for the normal GCW workloads
        // (e.g. max_gap=118 versus 83 classes in the user's benchmark).
        if (direct_gap8) {
            uint64_t power_m = mc.rmod;
            for (uint32_t g = 1; g <= max_gap; ++g) {
                power_m = d_mont_mul_2x32(power_m, step_m, mc);
                gap_x_table[(static_cast<uint64_t>(g) - 1ULL) * prime_count + pi] = power_m;
            }
        } else if (gap_count != 0) {
            const bool incremental = static_cast<uint64_t>(max_gap) <=
                                     static_cast<uint64_t>(gap_count) * 8ULL;
            if (incremental) {
                uint64_t power_m = mc.rmod;
                uint32_t next_gid = 0;
                for (uint32_t g = 1; g <= max_gap && next_gid < gap_count; ++g) {
                    power_m = d_mont_mul_2x32(power_m, step_m, mc);
                    if (g == unique_gaps[next_gid]) {
                        gap_x_table[static_cast<uint64_t>(next_gid) * prime_count + pi] = power_m;
                        ++next_gid;
                    }
                }
            } else {
                for (uint32_t gid = 0; gid < gap_count; ++gid) {
                    gap_x_table[static_cast<uint64_t>(gid) * prime_count + pi] =
                        d_mont_pow_rep_2x32(step_m, unique_gaps[gid], mc);
                }
            }
        }

        uint64_t pos = active_count - 1;
        const uint32_t last_lattice_idx = active_indices[pos];
        const uint64_t last_a = min_a + static_cast<uint64_t>(last_lattice_idx) * stride;
        const uint64_t exponent = exponent_offset > 0 ? last_a + 1
                                : exponent_offset < 0 ? last_a - 1
                                                      : last_a;

        // Power in Montgomery representation, then convert once.  Keeping x in
        // the ordinary domain makes every later comparison free of Montgomery
        // encoding and removes the a_delta table entirely.
        uint64_t x_m = d_mont_pow_rep_2x32(inv_base_m, exponent, mc);
        if (sign > 0 && x_m != 0) x_m = p - x_m;
        uint64_t x = d_mont_mul_2x32(x_m, 1ULL, mc);
        uint64_t a_mod = last_a < p ? last_a : last_a % p;

        for (;;) {
            if (x == a_mod) {
                // This path is extraordinarily rare.  Delay the active-index
                // load and bitmap address arithmetic until a congruence has
                // actually matched.
                const uint32_t lattice_idx = active_indices[pos];
                const uint64_t a = min_a + static_cast<uint64_t>(lattice_idx) * stride;
                const uint64_t wi = static_cast<uint64_t>(lattice_idx) >> 5;
                const uint32_t mask = UINT32_C(1) << (lattice_idx & 31);
                if ((term_bits[wi] & mask) != 0) {
                    const uint64_t e = exponent_offset > 0 ? a + 1
                                       : exponent_offset < 0 ? a - 1
                                                             : a;
                    if (!d_term_equals_prime_gncw(p, a, base, e, sign)) {
                        const uint32_t old = atomicAnd(term_bits + wi, ~mask);
                        if (old & mask) {
                            if (removed_count) atomicAdd(removed_count, 1ULL);
                            if (factors) factors[lattice_idx] = p;
                        }
                    }
                }
            }

            if (pos == 0) break;
            uint32_t gap;
            uint64_t x_mul_m;
            if (direct_gap8) {
                gap = static_cast<uint32_t>(gap_steps8[pos - 1]);
                x_mul_m = gap_x_table[(static_cast<uint64_t>(gap) - 1ULL) * prime_count + pi];
            } else {
                const uint32_t gid = gap_ids[pos - 1];
                gap = unique_gaps[gid];
                x_mul_m = gap_x_table[static_cast<uint64_t>(gid) * prime_count + pi];
            }

            // x is ordinary, x_mul_m is Montgomery.  The result is ordinary.
            x = d_mont_mul_2x32(x, x_mul_m, mc);

            const uint64_t delta_raw = stride * static_cast<uint64_t>(gap);
            const uint64_t delta = delta_raw < p ? delta_raw : delta_raw % p;
            a_mod = (a_mod >= delta) ? (a_mod - delta) : (a_mod + (p - delta));
            --pos;
        }
    }
}



// Sparse hot-gap path: keep the most common short-gap multipliers on chip.
// The benchmark worksets have geometric-like gap distributions, so gaps 1..16
// account for a large fraction of transitions after compaction.  The table is
// per-thread (one prime/thread) but lives in block shared memory, avoiding a
// global/L2 read for those transitions without increasing register pressure.

__device__ __forceinline__ void d_sparse_remove_hotgap_hit(
    uint64_t p, uint64_t a, uint64_t pos,
    const uint32_t* active_indices,
    uint32_t base, int exponent_offset, int sign,
    uint32_t* term_bits, uint64_t* factors,
    unsigned long long* removed_count) {
    const uint32_t lattice_idx = active_indices[pos];
    const uint64_t wi = static_cast<uint64_t>(lattice_idx) >> 5;
    const uint32_t mask = UINT32_C(1) << (lattice_idx & 31);
    if ((term_bits[wi] & mask) == 0) return;
    const uint64_t e = exponent_offset > 0 ? a + 1
                       : exponent_offset < 0 ? a - 1
                                             : a;
    if (d_term_equals_prime_gncw(p, a, base, e, sign)) return;
    const uint32_t old = atomicAnd(term_bits + wi, ~mask);
    if (old & mask) {
        if (removed_count) atomicAdd(removed_count, 1ULL);
        if (factors) factors[lattice_idx] = p;
    }
}

template <uint32_t HOT_GAPS, bool FAST44>
__global__ void gncwsieve_sparse_hotgap_kernel(
    const uint64_t* primes, uint64_t prime_count,
    const uint32_t* active_indices, uint64_t active_count,
    const uint8_t* gap_steps8, uint32_t max_gap,
    uint32_t base, int exponent_offset, int sign,
    uint64_t min_a, uint64_t max_a, uint64_t stride,
    uint32_t* term_bits, uint64_t* factors,
    unsigned long long* removed_count,
    uint64_t* gap_x_table) {

    extern __shared__ uint64_t hot_gap_mul[];
    const uint32_t tid = threadIdx.x;
    const uint64_t first_pi = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    const uint64_t pi_stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t pi = first_pi; pi < prime_count; pi += pi_stride) {
        const uint64_t p = primes[pi];
        if (p <= 2 || !(p & 1ULL) || active_count == 0) continue;

        const uint64_t base_mod = static_cast<uint64_t>(base) < p
            ? static_cast<uint64_t>(base)
            : static_cast<uint64_t>(base) % p;
        if (base_mod == 0) continue;
        const uint64_t inv_base = d_inverse_base(base, p);
        if (inv_base == 0) continue;

        const Mont64 mc = d_make_mont(p);
        const uint64_t inv_base_m = d_sparse_mont_mul<FAST44>(inv_base, mc.r2, mc);
        const uint64_t b_m = d_sparse_mont_mul<FAST44>(base_mod, mc.r2, mc);
        uint64_t step_m = b_m;
        if (stride == 2) step_m = d_sparse_mont_mul<FAST44>(step_m, step_m, mc);
        else if (stride != 1) step_m = d_sparse_mont_pow_rep<FAST44>(b_m, stride, mc);

        // Generate the same dense 1..max_gap multiplier sequence as v5, but
        // keep 1..16 on-chip and only write the cold rows to the global table.
        uint64_t power_m = mc.rmod;
        for (uint32_t g = 1; g <= max_gap; ++g) {
            power_m = d_sparse_mont_mul<FAST44>(power_m, step_m, mc);
            if constexpr (HOT_GAPS != 0) {
                if (g <= HOT_GAPS) {
                    hot_gap_mul[(static_cast<uint64_t>(g) - 1ULL) * blockDim.x + tid] = power_m;
                } else {
                    gap_x_table[(static_cast<uint64_t>(g) - 1ULL) * prime_count + pi] = power_m;
                }
            } else {
                gap_x_table[(static_cast<uint64_t>(g) - 1ULL) * prime_count + pi] = power_m;
            }
        }

        uint64_t pos = active_count - 1;
        const uint32_t last_lattice_idx = active_indices[pos];
        const uint64_t last_a = min_a + static_cast<uint64_t>(last_lattice_idx) * stride;
        const uint64_t exponent = exponent_offset > 0 ? last_a + 1
                                : exponent_offset < 0 ? last_a - 1
                                                      : last_a;
        uint64_t x_m = d_sparse_mont_pow_rep<FAST44>(inv_base_m, exponent, mc);
        if (sign > 0 && x_m != 0) x_m = p - x_m;
        uint64_t x = d_sparse_mont_mul<FAST44>(x_m, 1ULL, mc);

        // Once p exceeds the complete a range, a never wraps modulo p.  This
        // is the overwhelmingly dominant regime in deep GCW sieving and lets
        // the hot loop use a plain integer subtraction instead of modular
        // subtraction/conditional correction on every candidate.
        if (p > max_a) {
            uint64_t a_cur = last_a;
            for (;;) {
                if (x == a_cur) {
                    d_sparse_remove_hotgap_hit(p, a_cur, pos, active_indices,
                                               base, exponent_offset, sign,
                                               term_bits, factors, removed_count);
                }
                if (pos == 0) break;
                const uint32_t gap = static_cast<uint32_t>(gap_steps8[pos - 1]);
                uint64_t x_mul_m;
                if constexpr (HOT_GAPS != 0) {
                    x_mul_m = gap <= HOT_GAPS
                        ? hot_gap_mul[(static_cast<uint64_t>(gap) - 1ULL) * blockDim.x + tid]
                        : gap_x_table[(static_cast<uint64_t>(gap) - 1ULL) * prime_count + pi];
                } else {
                    x_mul_m = gap_x_table[(static_cast<uint64_t>(gap) - 1ULL) * prime_count + pi];
                }
                x = d_sparse_mont_mul<FAST44>(x, x_mul_m, mc);
                a_cur -= stride * static_cast<uint64_t>(gap);
                --pos;
            }
        } else {
            uint64_t a_mod = last_a < p ? last_a : last_a % p;
            for (;;) {
                if (x == a_mod) {
                    const uint32_t lattice_idx = active_indices[pos];
                    const uint64_t a = min_a + static_cast<uint64_t>(lattice_idx) * stride;
                    d_sparse_remove_hotgap_hit(p, a, pos, active_indices,
                                               base, exponent_offset, sign,
                                               term_bits, factors, removed_count);
                }
                if (pos == 0) break;
                const uint32_t gap = static_cast<uint32_t>(gap_steps8[pos - 1]);
                uint64_t x_mul_m;
                if constexpr (HOT_GAPS != 0) {
                    x_mul_m = gap <= HOT_GAPS
                        ? hot_gap_mul[(static_cast<uint64_t>(gap) - 1ULL) * blockDim.x + tid]
                        : gap_x_table[(static_cast<uint64_t>(gap) - 1ULL) * prime_count + pi];
                } else {
                    x_mul_m = gap_x_table[(static_cast<uint64_t>(gap) - 1ULL) * prime_count + pi];
                }
                x = d_sparse_mont_mul<FAST44>(x, x_mul_m, mc);
                const uint64_t delta_raw = stride * static_cast<uint64_t>(gap);
                const uint64_t delta = delta_raw < p ? delta_raw : delta_raw % p;
                a_mod = (a_mod >= delta) ? (a_mod - delta) : (a_mod + (p - delta));
                --pos;
            }
        }
    }
}

// ------------------------- CPU small-prime sieve -------------------------

static uint64_t sieve_prime_host(Problem& pr, uint64_t p) {
    if (p < 2 || pr.opt.base % p == 0) return 0;

    const uint64_t inv_base = inverse_mod_host(pr.opt.base, p);
    if (inv_base == 0) return 0;
    uint64_t x = pow_mod_host(inv_base, exponent_for_a(pr, pr.opt.max_a), p);
    if (mode_sign(pr.opt.mode) > 0 && x != 0) x = p - x;
    const uint64_t step = pow_mod_host(pr.opt.base, pr.stride, p);

    uint64_t removed = 0;
    for (uint64_t idx = pr.candidate_count; idx-- > 0;) {
        const uint64_t a = index_to_a(pr, idx);
        if (bit_test(pr.term_bits, idx) && x == a % p && !term_equals_prime_host(pr, p, a)) {
            if (bit_clear(pr.term_bits, idx)) {
                ++removed;
                if (!pr.factor.empty()) pr.factor[static_cast<size_t>(idx)] = p;
            }
        }
        if (idx != 0) x = mul_mod_host(x, step, p);
    }
    return removed;
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
        if (p_.opt.threads > prop_.maxThreadsPerBlock)
            fail("--threads exceeds device maxThreadsPerBlock");

        const uint64_t active_now = active_terms(p_);
        // compact-gap-v2 has a one-multiply inner transition, so switch much
        // earlier than the old sparse path: once at least 3/4 of lattice slots
        // are gone.  Dense remains best while the lattice is still close to full.
        sparse_mode_ = active_now != 0 &&
                       p_.candidate_count >= 8 &&
                       active_now <= p_.candidate_count / 4;
        kernel_threads_ = sparse_mode_ ? std::min<int>(p_.opt.threads, 256)
                                       : p_.opt.threads;
        grid_block_limit_ = p_.opt.blocks != 0
            ? p_.opt.blocks
            : prop_.multiProcessorCount * (sparse_mode_ ? 16 : 8);

        const size_t words = static_cast<size_t>(words_for_bits(p_.candidate_count));
        d_terms_.allocate(words);
        CUDA_CHECK(cudaMemcpy(d_terms_.ptr, p_.term_bits.data(),
                              words * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (!p_.factor.empty()) {
            d_factor_.allocate(static_cast<size_t>(p_.candidate_count));
            CUDA_CHECK(cudaMemcpy(d_factor_.ptr, p_.factor.data(),
                                  p_.factor.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }

        if (sparse_mode_) {
            std::vector<uint32_t> active_indices = collect_active_indices(p_);
            sparse_active_count_ = static_cast<uint64_t>(active_indices.size());
            d_active_indices_.allocate(active_indices.size());
            if (!active_indices.empty()) {
                CUDA_CHECK(cudaMemcpy(d_active_indices_.ptr, active_indices.data(),
                                      active_indices.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice));
            }

            std::vector<uint32_t> gaps;
            if (active_indices.size() > 1) {
                gaps.reserve(active_indices.size() - 1);
                for (size_t i = 1; i < active_indices.size(); ++i)
                    gaps.push_back(active_indices[i] - active_indices[i - 1]);
            }
            std::vector<uint32_t> unique_gaps = gaps;
            std::sort(unique_gaps.begin(), unique_gaps.end());
            unique_gaps.erase(std::unique(unique_gaps.begin(), unique_gaps.end()), unique_gaps.end());
            sparse_gap_count_ = static_cast<uint32_t>(unique_gaps.size());
            sparse_max_gap_ = unique_gaps.empty() ? 0U : unique_gaps.back();
            sparse_direct_gap8_ = sparse_max_gap_ != 0 && sparse_max_gap_ <= 255U;
            // v13 fast44 is implemented in the direct-gap sparse kernel.  It is
            // exact for the full run whenever maxP < 2^44; larger ranges and
            // non-u8 gap worksets retain the proven generic 2x32 path.
            sparse_fast44_ = sparse_direct_gap8_ &&
                             p_.opt.max_prime < (UINT64_C(1) << 44);
            sparse_hot_transitions_ = 0;
            if (sparse_direct_gap8_) {
                for (uint32_t g : gaps)
                    if (g <= p_.opt.sparse_hot_gaps) ++sparse_hot_transitions_;
            }

            if (sparse_direct_gap8_) {
                std::vector<uint8_t> gap_steps8(gaps.size());
                for (size_t i = 0; i < gaps.size(); ++i)
                    gap_steps8[i] = static_cast<uint8_t>(gaps[i]);
                d_sparse_gap_steps8_.allocate(gap_steps8.size());
                if (!gap_steps8.empty()) {
                    CUDA_CHECK(cudaMemcpy(d_sparse_gap_steps8_.ptr, gap_steps8.data(),
                                          gap_steps8.size() * sizeof(uint8_t),
                                          cudaMemcpyHostToDevice));
                }
            } else {
                std::vector<uint32_t> gap_ids(gaps.size());
                for (size_t i = 0; i < gaps.size(); ++i) {
                    const auto it = std::lower_bound(unique_gaps.begin(), unique_gaps.end(), gaps[i]);
                    gap_ids[i] = static_cast<uint32_t>(it - unique_gaps.begin());
                }
                d_sparse_gap_ids_.allocate(gap_ids.size());
                if (!gap_ids.empty()) {
                    CUDA_CHECK(cudaMemcpy(d_sparse_gap_ids_.ptr, gap_ids.data(),
                                          gap_ids.size() * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice));
                }
                d_sparse_unique_gaps_.allocate(unique_gaps.size());
                if (!unique_gaps.empty()) {
                    CUDA_CHECK(cudaMemcpy(d_sparse_unique_gaps_.ptr, unique_gaps.data(),
                                          unique_gaps.size() * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice));
                }
            }

            constexpr size_t table_budget_bytes = size_t(128) << 20;
            const size_t table_rows = sparse_direct_gap8_
                ? static_cast<size_t>(sparse_max_gap_)
                : static_cast<size_t>(sparse_gap_count_);
            const size_t bytes_per_prime = table_rows * sizeof(uint64_t);
            sparse_prime_chunk_cap_ = bytes_per_prime == 0
                ? std::numeric_limits<size_t>::max()
                : std::max<size_t>(1, table_budget_bytes / bytes_per_prime);
        }

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
            CUDA_CHECK(cudaMemset(slot->d_removed.ptr, 0, sizeof(unsigned long long)));
            slot->removed_cumulative = 0;
            slots_.push_back(std::move(slot));
        }

        if (!p_.opt.quiet) {
            const double density = p_.candidate_count != 0
                ? 100.0 * static_cast<double>(active_now) /
                  static_cast<double>(p_.candidate_count)
                : 0.0;
            std::cout << "CUDA device " << p_.opt.device << ": " << prop_.name
                      << ", SMs=" << prop_.multiProcessorCount
                      << ", grid blocks=" << grid_block_limit_;
            if (sparse_mode_)
                std::cout << ", threads/block=" << kernel_threads_ << ", one thread/prime\n";
            else
                std::cout << ", cooperative threads/prime=" << kernel_threads_ << "\n";
            std::cout << "Candidate kernel: " << (sparse_mode_ ? (sparse_direct_gap8_ ? "compact-gap-v7-hot6-fast44" : "compact-gap-v2") : "dense-lattice")
                      << ", live=" << active_now << "/" << p_.candidate_count
                      << " (" << std::fixed << std::setprecision(3) << density << "%)";
            if (sparse_mode_)
                std::cout << ", gap classes=" << sparse_gap_count_ << ", max gap=" << sparse_max_gap_
                          << ", gap table=x-only"
                          << (sparse_direct_gap8_ ? "/u8-direct" : "/class")
                          << ", mont=" << (sparse_fast44_ ? "fast44-3limb" : "2x32");
                if (sparse_direct_gap8_) {
                    const uint64_t transitions = sparse_active_count_ > 0 ? sparse_active_count_ - 1 : 0;
                    const double hot_pct = transitions != 0
                        ? 100.0 * static_cast<double>(sparse_hot_transitions_) / static_cast<double>(transitions)
                        : 0.0;
                    const size_t hot_shared_bytes = static_cast<size_t>(p_.opt.sparse_hot_gaps) *
                                                    static_cast<size_t>(kernel_threads_) * sizeof(uint64_t);
                    std::cout << ", hot gaps=" << p_.opt.sparse_hot_gaps << "/shared-specialized ("
                              << std::setprecision(1) << hot_pct << "% transitions)"
                              << ", shared/block=" << std::setprecision(1)
                              << (static_cast<double>(hot_shared_bytes) / 1024.0) << "KiB"
                              << ", p>amax direct-a";
                }
            std::cout << ".\n"
                      << "CUDA pipeline: " << slots_.size()
                      << " pinned buffer/stream" << (slots_.size() == 1 ? "" : "s")
                      << " (asynchronous H2D + kernel)\n";
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
    bool sparse_mode() const { return sparse_mode_; }
    uint64_t sparse_active_count() const { return sparse_active_count_; }

    bool should_rebuild_compact(uint64_t remaining) const {
        if (remaining == 0) return false;
        if (!sparse_mode_)
            return p_.candidate_count >= 8 && remaining <= p_.candidate_count / 4;
        // Repack after losing 20% of the compact workset.  The compact kernel
        // is candidate-linear, so stale terms translate almost directly into
        // wasted Montgomery recurrences.
        return sparse_active_count_ >= 2 && remaining * 5 <= sparse_active_count_ * 4;
    }

    void submit(const PrimeBatch& batch, size_t offset, size_t prime_count) {
        if (prime_count == 0) return;
        if (!batch.data || offset > batch.count || prime_count > batch.count - offset)
            fail("Internal invalid prime batch span");
        submit(batch.data + offset, prime_count, batch.owner, batch.pinned);
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
        CUDA_CHECK(cudaEventRecord(s.start, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_primes.ptr, h_source,
                                   prime_count * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaEventRecord(s.h2d_done, s.stream));
        // d_removed is cumulative per stream slot.  Do not launch an 8-byte
        // cudaMemset for every prime batch; it accounted for ~18 seconds in
        // the 3.5-minute RTX 4060 benchmark.
        CUDA_CHECK(cudaEventRecord(s.reset_done, s.stream));

        if (sparse_mode_) {
            size_t done = 0;
            while (done < prime_count) {
                const size_t chunk = std::min(prime_count - done, sparse_prime_chunk_cap_);
                ensure_sparse_table_capacity(s, chunk);
                const uint64_t needed_blocks =
                    (static_cast<uint64_t>(chunk) + static_cast<uint64_t>(kernel_threads_) - 1ULL) /
                    static_cast<uint64_t>(kernel_threads_);
                const int grid_blocks = static_cast<int>(std::max<uint64_t>(1ULL, std::min<uint64_t>(
                    static_cast<uint64_t>(grid_block_limit_), needed_blocks)));
                if (sparse_direct_gap8_) {
                    const size_t shared_bytes = static_cast<size_t>(p_.opt.sparse_hot_gaps) *
                                                static_cast<size_t>(kernel_threads_) * sizeof(uint64_t);
#define GNCWSV_LAUNCH_HOT_F(H, F) \
                    gncwsieve_sparse_hotgap_kernel<H, F><<<grid_blocks, kernel_threads_, shared_bytes, s.stream>>>( \
                        s.d_primes.ptr + done, chunk, \
                        d_active_indices_.ptr, sparse_active_count_, \
                        d_sparse_gap_steps8_.ptr, sparse_max_gap_, \
                        p_.opt.base, mode_offset(p_.opt.mode), mode_sign(p_.opt.mode), \
                        p_.opt.min_a, p_.opt.max_a, p_.stride, \
                        d_terms_.ptr, d_factor_.ptr, s.d_removed.ptr, \
                        s.d_gap_x_table.ptr)
#define GNCWSV_DISPATCH_HOT(F) \
                    switch (p_.opt.sparse_hot_gaps) { \
                        case 0:  GNCWSV_LAUNCH_HOT_F(0, F);  break; \
                        case 2:  GNCWSV_LAUNCH_HOT_F(2, F);  break; \
                        case 4:  GNCWSV_LAUNCH_HOT_F(4, F);  break; \
                        case 6:  GNCWSV_LAUNCH_HOT_F(6, F);  break; \
                        case 8:  GNCWSV_LAUNCH_HOT_F(8, F);  break; \
                        case 12: GNCWSV_LAUNCH_HOT_F(12, F); break; \
                        case 16: GNCWSV_LAUNCH_HOT_F(16, F); break; \
                        default: fail("Internal error: invalid --hot-gaps specialization"); \
                    }
                    if (sparse_fast44_) { GNCWSV_DISPATCH_HOT(true); }
                    else { GNCWSV_DISPATCH_HOT(false); }
#undef GNCWSV_DISPATCH_HOT
#undef GNCWSV_LAUNCH_HOT_F
                } else {
                    gncwsieve_sparse_xonly_kernel<<<grid_blocks, kernel_threads_, 0, s.stream>>>(
                        s.d_primes.ptr + done, chunk,
                        d_active_indices_.ptr, sparse_active_count_,
                        d_sparse_gap_steps8_.ptr,
                        d_sparse_gap_ids_.ptr, d_sparse_unique_gaps_.ptr,
                        sparse_gap_count_, sparse_max_gap_, 0,
                        p_.opt.base, mode_offset(p_.opt.mode), mode_sign(p_.opt.mode),
                        p_.opt.min_a, p_.stride,
                        d_terms_.ptr, d_factor_.ptr, s.d_removed.ptr,
                        s.d_gap_x_table.ptr);
                }
                CUDA_CHECK(cudaGetLastError());
                done += chunk;
            }
        } else {
            const int grid_blocks = static_cast<int>(std::min<uint64_t>(
                static_cast<uint64_t>(grid_block_limit_), static_cast<uint64_t>(prime_count)));
            gncwsieve_dense_kernel<<<grid_blocks, kernel_threads_, 0, s.stream>>>(
                s.d_primes.ptr, prime_count,
                p_.opt.base, mode_offset(p_.opt.mode), mode_sign(p_.opt.mode),
                p_.opt.min_a, p_.opt.max_a, p_.stride, p_.candidate_count,
                d_terms_.ptr, d_factor_.ptr, s.d_removed.ptr);
            CUDA_CHECK(cudaGetLastError());
        }
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
        const size_t slot_index = order_.front();
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
        const uint64_t removed_now = static_cast<uint64_t>(*s.h_removed.ptr);
        if (removed_now < s.removed_cumulative)
            fail("Internal CUDA removed counter moved backwards");
        c.removed = removed_now - s.removed_cumulative;
        s.removed_cumulative = removed_now;
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
        const size_t words = static_cast<size_t>(words_for_bits(p_.candidate_count));
        CUDA_CHECK(cudaMemcpy(p_.term_bits.data(), d_terms_.ptr,
                              words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (!p_.factor.empty())
            CUDA_CHECK(cudaMemcpy(p_.factor.data(), d_factor_.ptr,
                                  p_.factor.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

private:
    struct Slot {
        DeviceBuffer<uint64_t> d_primes;
        DeviceBuffer<uint64_t> d_gap_x_table;
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
        uint64_t removed_cumulative = 0;
    };

    void ensure_device_capacity(Slot& s, size_t needed) {
        if (s.d_primes.count >= needed) return;
        const size_t grown = needed + needed / 4 + 4096;
        s.d_primes.allocate(grown);
    }

    void ensure_staging_capacity(Slot& s, size_t needed) {
        if (s.h_primes.count >= needed) return;
        const size_t grown = needed + needed / 4 + 4096;
        s.h_primes.allocate(grown);
    }

    void ensure_sparse_table_capacity(Slot& s, size_t chunk_primes) {
        if (!sparse_mode_) return;
        const size_t rows = sparse_direct_gap8_
            ? static_cast<size_t>(sparse_max_gap_)
            : static_cast<size_t>(sparse_gap_count_);
        if (rows == 0) return;
        if (chunk_primes > std::numeric_limits<size_t>::max() / rows)
            fail("Sparse x-only gap table size overflow");
        const size_t needed = chunk_primes * rows;
        if (s.d_gap_x_table.count >= needed) return;
        s.d_gap_x_table.allocate(needed);
    }

    Problem& p_;
    cudaDeviceProp prop_{};
    DeviceBuffer<uint32_t> d_terms_;
    DeviceBuffer<uint64_t> d_factor_;
    DeviceBuffer<uint32_t> d_active_indices_;
    DeviceBuffer<uint8_t> d_sparse_gap_steps8_;
    DeviceBuffer<uint32_t> d_sparse_gap_ids_;
    DeviceBuffer<uint32_t> d_sparse_unique_gaps_;
    bool sparse_mode_ = false;
    uint64_t sparse_active_count_ = 0;
    uint32_t sparse_gap_count_ = 0;
    uint32_t sparse_max_gap_ = 0;
    bool sparse_direct_gap8_ = false;
    bool sparse_fast44_ = false;
    uint64_t sparse_hot_transitions_ = 0;
    size_t sparse_prime_chunk_cap_ = std::numeric_limits<size_t>::max();
    int kernel_threads_ = 256;
    int grid_block_limit_ = 0;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::deque<size_t> order_;
    size_t next_slot_ = 0;
    double staging_seconds_ = 0.0;
    uint64_t direct_pinned_batches_ = 0;
    uint64_t fallback_staged_batches_ = 0;
};

// ------------------------- output and verification -------------------------

static void write_terms(const Problem& p, uint64_t largest_prime) {
    std::ofstream out(p.opt.output_terms, std::ios::trunc);
    if (!out) fail("Unable to open output terms file: " + p.opt.output_terms);
    out << "# GNCWSV p=" << largest_prime
        << " base=" << p.opt.base
        << " mode=" << static_cast<int>(p.opt.mode)
        << " amin=" << p.opt.min_a
        << " amax=" << p.opt.max_a << '\n';
    for (uint64_t i = 0; i < p.candidate_count; ++i) {
        if (bit_test(p.term_bits, i)) out << term_text(p, index_to_a(p, i)) << '\n';
    }
}

static uint64_t write_factors(const Problem& p) {
    if (p.opt.output_factors.empty()) return 0;
    std::ofstream out(p.opt.output_factors, std::ios::app);
    if (!out) fail("Unable to open output factor file: " + p.opt.output_factors);
    uint64_t count = 0;
    for (uint64_t i = 0; i < p.candidate_count; ++i) {
        const uint64_t q = p.factor[static_cast<size_t>(i)];
        if (!q) continue;
        const uint64_t a = index_to_a(p, i);
        if (p.opt.verify_factors && !factor_is_valid(p, q, a))
            fail("Factor verification failed: " + std::to_string(q) + " | " + term_text(p, a));
        out << q << " | " << term_text(p, a) << '\n';
        ++count;
    }
    return count;
}

static std::string format_duration(double sec) {
    std::ostringstream oss;
    if (sec < 60) oss << std::fixed << std::setprecision(1) << sec << "s";
    else if (sec < 3600) oss << std::fixed << std::setprecision(1) << sec / 60 << "m";
    else oss << std::fixed << std::setprecision(2) << sec / 3600 << "h";
    return oss.str();
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

static std::string format_eta_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) return "n/a";
    uint64_t total = static_cast<uint64_t>(std::llround(seconds));
    uint64_t days = total / 86400; total %= 86400;
    uint64_t hours = total / 3600; total %= 3600;
    uint64_t minutes = total / 60;
    uint64_t secs = total % 60;
    std::ostringstream oss;
    if (days > 0) oss << days << "d " << hours << "h";
    else if (hours > 0) oss << hours << "h " << minutes << "m";
    else if (minutes > 0) oss << minutes << "m " << secs << "s";
    else oss << secs << "s";
    return oss.str();
}

static std::string format_eta_finish_time(double eta_seconds) {
    if (!std::isfinite(eta_seconds) || eta_seconds < 0.0) return {};
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
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M %Z", &local_tm) == 0)
        return {};
    return buffer;
}

static double estimate_eta_seconds(uint64_t current, uint64_t pmax, double p_range_per_second) {
    if (current >= pmax) return 0.0;
    if (!(p_range_per_second > 0.0) || !std::isfinite(p_range_per_second))
        return std::numeric_limits<double>::quiet_NaN();
    long double eta = static_cast<long double>(pmax - current)
                    / static_cast<long double>(p_range_per_second);
    if (!(eta >= 0.0L) || eta > static_cast<long double>(std::numeric_limits<double>::max()))
        return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(eta);
}

static double percent_done(uint64_t pmin, uint64_t pmax, uint64_t current) {
    if (pmax <= pmin || current <= pmin) return 0.0;
    long double result = 100.0L * static_cast<long double>(current - pmin)
                       / static_cast<long double>(pmax - pmin);
    if (result < 0.0L) result = 0.0L;
    if (result > 100.0L) result = 100.0L;
    return static_cast<double>(result);
}

static void print_progress_line(uint64_t largest_prime,
                                uint64_t pmin,
                                uint64_t pmax,
                                double prime_rate,
                                double p_range_per_second,
                                uint64_t removed,
                                uint64_t remaining) {
    std::ostringstream line;
    line << "  p=" << largest_prime << ", rate60=" << format_prime_rate(prime_rate)
         << ", removed=" << removed << ", remaining=" << remaining
         << ", " << std::fixed << std::setprecision(1)
         << percent_done(pmin, pmax, largest_prime) << "% done, ETA ";
    const double eta_seconds = estimate_eta_seconds(largest_prime, pmax, p_range_per_second);
    if (std::isfinite(eta_seconds)) {
        line << format_eta_duration(eta_seconds);
        const std::string finish_time = format_eta_finish_time(eta_seconds);
        if (!finish_time.empty()) line << " (finish " << finish_time << ")";
    } else {
        line << "n/a";
    }
    line << '.';
    std::cout << line.str() << '\n' << std::flush;
}

static uint64_t run_sieve(Problem& p) {
    const uint64_t initial_terms = active_terms(p);
    uint64_t largest_prime = p.opt.min_prime;
    uint64_t primes_tested = 0;
    uint64_t factors_found = 0;
    uint64_t gpu_primes = 0, cpu_primes = 0;
    uint64_t gpu_batches = 0;
    double generator_wait_seconds = 0.0;
    double gpu_backpressure_seconds = 0.0;
    double gpu_h2d_ms = 0.0, gpu_reset_ms = 0.0;
    double gpu_kernel_ms = 0.0, gpu_d2h_ms = 0.0, gpu_total_ms = 0.0;
    double gpu_staging_seconds_acc = 0.0;
    uint64_t gpu_direct_pinned_acc = 0, gpu_fallback_staged_acc = 0;
    const auto start = std::chrono::steady_clock::now();
    auto next_report_wall = start + std::chrono::seconds(p.opt.progress_seconds);
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

    // Progress speed is a true time-based rolling window, not a fixed number
    // of report points.  Samples are recorded whenever completed prime work is
    // retired, so --progress-seconds does not change the meaning of the rate.
    constexpr auto rate_window = std::chrono::seconds(60);
    std::deque<RateSample> rate_samples;
    rate_samples.push_back({start, 0, p.opt.min_prime});

    auto record_rate_sample = [&](std::chrono::steady_clock::time_point now) {
        if (!rate_samples.empty() &&
            rate_samples.back().primes_tested == primes_tested &&
            rate_samples.back().largest_prime == largest_prime)
            return;

        rate_samples.push_back({now, primes_tested, largest_prime});
        const auto cutoff = now - rate_window;

        // Keep the last sample at/before the 60-second cutoff plus all newer
        // samples.  That lets rolling_rates interpolate an exact cutoff even
        // though completions arrive in batches.
        while (rate_samples.size() >= 3 && rate_samples[1].wall <= cutoff)
            rate_samples.pop_front();
    };

    auto rolling_rates = [&](std::chrono::steady_clock::time_point now) {
        record_rate_sample(now);
        RollingRates result;
        if (rate_samples.empty()) return result;

        const auto cutoff = now - rate_window;
        long double base_primes = 0.0L;
        long double base_prime_value = static_cast<long double>(p.opt.min_prime);
        std::chrono::steady_clock::time_point base_wall = rate_samples.front().wall;

        if (rate_samples.front().wall > cutoff) {
            // The run is younger than 60 seconds (or there was no completed
            // sample old enough).  Use all available elapsed time.
            base_primes = static_cast<long double>(rate_samples.front().primes_tested);
            base_prime_value = static_cast<long double>(rate_samples.front().largest_prime);
        } else {
            // Normally the deque contains one sample on/before the cutoff and
            // the next sample after it.  Interpolate between those completion
            // points so the denominator is exactly 60 seconds.
            base_wall = cutoff;
            const RateSample& a = rate_samples.front();
            if (rate_samples.size() >= 2 && rate_samples[1].wall > a.wall) {
                const RateSample& b = rate_samples[1];
                const long double span = static_cast<long double>(
                    std::chrono::duration<double>(b.wall - a.wall).count());
                long double f = static_cast<long double>(
                    std::chrono::duration<double>(cutoff - a.wall).count()) / span;
                if (f < 0.0L) f = 0.0L;
                if (f > 1.0L) f = 1.0L;
                base_primes = static_cast<long double>(a.primes_tested)
                            + f * static_cast<long double>(b.primes_tested - a.primes_tested);
                base_prime_value = static_cast<long double>(a.largest_prime)
                                 + f * static_cast<long double>(b.largest_prime - a.largest_prime);
            } else {
                base_wall = a.wall;
                base_primes = static_cast<long double>(a.primes_tested);
                base_prime_value = static_cast<long double>(a.largest_prime);
            }
        }

        const double seconds = std::chrono::duration<double>(now - base_wall).count();
        if (seconds > 0.0) {
            long double tested_delta = static_cast<long double>(primes_tested) - base_primes;
            if (tested_delta < 0.0L) tested_delta = 0.0L;
            result.primes_per_second = static_cast<double>(tested_delta / seconds);

            long double p_delta = static_cast<long double>(largest_prime) - base_prime_value;
            if (p_delta < 0.0L) p_delta = 0.0L;
            result.p_range_per_second = static_cast<double>(p_delta / seconds);
        }
        return result;
    };

    auto maybe_report = [&](bool force) {
        if (p.opt.quiet) return;
        const auto now = std::chrono::steady_clock::now();
        if (!force && now < next_report_wall) return;
        if (primes_tested == last_report_primes) return;
        const RollingRates rates = rolling_rates(now);
        const uint64_t remaining = factors_found <= initial_terms
            ? initial_terms - factors_found : 0;
        print_progress_line(largest_prime, p.opt.min_prime, p.opt.max_prime,
                            rates.primes_per_second, rates.p_range_per_second,
                            factors_found, remaining);
        last_report_primes = primes_tested;
        next_report_wall = now + std::chrono::seconds(p.opt.progress_seconds);
    };

    // If all primes go to CUDA, upload the initial bitmap before production.
    // Otherwise construction stays lazy until the CPU cutoff has been handled.
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
                  << " prefetched batch" << (p.opt.prime_prefetch == 1 ? "" : "es") << "\n";
    }

    PrimeBatchPipeline pipeline(stream, p.opt.prime_prefetch);

    auto retire_one = [&]() {
        const auto wait_start = std::chrono::steady_clock::now();
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
        record_rate_sample(std::chrono::steady_clock::now());
        maybe_report(false);
    };

    auto drain_gpu = [&]() {
        while (gpu && gpu->in_flight() != 0) retire_one();
    };

    auto maybe_rebuild_gpu_for_compaction = [&]() {
        if (!gpu) return;
        uint64_t remaining = factors_found <= initial_terms
            ? initial_terms - factors_found : 0;
        if (!gpu->should_rebuild_compact(remaining)) return;

        // In-flight batches may remove more terms.  Drain first, then decide
        // again using the exact bitmap before paying for a rebuild.
        drain_gpu();
        remaining = factors_found <= initial_terms
            ? initial_terms - factors_found : 0;
        if (!gpu->should_rebuild_compact(remaining)) return;

        gpu->download();
        gpu_staging_seconds_acc += gpu->staging_seconds();
        gpu_direct_pinned_acc += gpu->direct_pinned_batches();
        gpu_fallback_staged_acc += gpu->fallback_staged_batches();
        if (!p.opt.quiet) {
            std::cout << "Rebuilding candidate workset at " << remaining
                      << " live term" << (remaining == 1 ? "" : "s")
                      << " to skip newly empty lattice positions.\n";
        }
        gpu.reset();
        gpu = std::make_unique<GpuSieve>(p);
    };

    PrimeBatch batch;
    while (!g_interrupted && pipeline.next(batch, generator_wait_seconds)) {
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
            drain_gpu();
            for (size_t i = 0; i < first_gpu_index; ++i) {
                const uint64_t q = batch.data[i];
                factors_found += sieve_prime_host(p, q);
                ++primes_tested;
                ++cpu_primes;
                largest_prime = q;
            }
            record_rate_sample(std::chrono::steady_clock::now());
            maybe_report(false);
        }

        if (first_gpu_index != batch.count) {
            size_t gpu_offset = first_gpu_index;
            while (gpu_offset < batch.count && !g_interrupted) {
                if (!gpu) gpu = std::make_unique<GpuSieve>(p);
                while (gpu->in_flight() >= gpu->capacity()) retire_one();
                maybe_rebuild_gpu_for_compaction();

                size_t submit_count = batch.count - gpu_offset;
                // A large producer batch amortizes pinned-buffer/event costs in
                // sparse mode.  While still dense, retire every 32768 primes so
                // the bitmap can compact at essentially the old responsiveness.
                if (!gpu->sparse_mode())
                    submit_count = std::min<size_t>(submit_count, size_t(1) << 15);

                gpu->submit(batch, gpu_offset, submit_count);
                gpu_offset += submit_count;
            }
        }
    }

    // Submitted streams share the bitmap, so retire them before download even after SIGINT.
    drain_gpu();
    if (gpu) gpu->download();
    maybe_report(true);

    if (g_interrupted)
        std::cout << "Interrupted by signal after p=" << largest_prime << ".\n";

    const uint64_t remaining = active_terms(p);
    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "Finished through p=" << largest_prime << " in " << format_duration(sec)
              << "; primes tested=" << primes_tested << " (CPU " << cpu_primes
              << ", GPU " << gpu_primes << "); removed=" << (initial_terms - remaining)
              << "; remaining=" << remaining << ".\n";

    if (!p.opt.quiet) {
        const double staging_seconds = gpu_staging_seconds_acc
            + (gpu ? gpu->staging_seconds() : 0.0);
        const double host_active_seconds = std::max(
            0.0, sec - generator_wait_seconds - gpu_backpressure_seconds);
        std::cout << "Timing (overlap-aware; fields are not additive): producer wait="
                  << std::fixed << std::setprecision(2) << generator_wait_seconds
                  << "s, GPU backpressure wait=" << gpu_backpressure_seconds
                  << "s, host active=" << host_active_seconds
                  << "s, fallback staging=" << staging_seconds << "s.\n";
        if (stream.pinned_region_buffers() != 0) {
            const double gib = static_cast<double>(stream.pinned_region_bytes()) /
                               (1024.0 * 1024.0 * 1024.0);
            std::cout << "Prime host pool: " << stream.pinned_region_buffers()
                      << "/" << stream.pinned_region_pool_limit()
                      << " pinned regions, " << std::setprecision(2) << gib
                      << " GiB allocated, growths="
                      << stream.pinned_region_growths() << ".\n";
        }
        if (gpu) {
            std::cout << "Prime batch path: direct pinned="
                      << (gpu_direct_pinned_acc + gpu->direct_pinned_batches())
                      << ", fallback staged="
                      << (gpu_fallback_staged_acc + gpu->fallback_staged_batches()) << ".\n";
        }
        if (gpu_batches != 0) {
            const double kernel_rate = gpu_kernel_ms > 0.0
                ? static_cast<double>(gpu_primes) / (gpu_kernel_ms / 1000.0) : 0.0;
            const double h2d_gbps = gpu_h2d_ms > 0.0
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

} // namespace gncw_cuda

void display_banner() {
    printf("%s\n","══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    printf("%s\n","        .oooooo.        ooooo      ooo       .oooooo.     oooooo   oooooo     oooo    .oooooo..o   oooooo     oooo        ");
    printf("%s\n","       d8P'  `Y8b       `888b.     `8'      d8P'  `Y8b     `888.    `888.     .8'    d8P'    `Y8    `888.     .8'         ");
    printf("%s\n","      888                8 `88b.    8      888              `888.   .8888.   .8'     Y88bo.          `888.   .8'          ");
    printf("%s\n","      888                8   `88b.  8      888               `888  .8'`888. .8'       `'Y8888o.       `888. .8'           ");
    printf("%s\n","      888     ooooo      8     `88b.8      888                `888.8'  `888.8'            `'Y88b       `888.8'            ");
    printf("%s\n","      `88.    .88'  .o.  8       `888  .o. `88b    ooo  .o.    `888'    `888'    .o. oo     .d8P .o.    `888'    .o.      ");
    printf("%s\n","       `Y8bood8P'   Y8P o8o        `8  Y8P  `Y8bood8P'  Y8P     `8'      `8'     Y8P 8''88888P'  Y8P     `8'     Y8P      ");
    printf("%s\n","══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    printf("%s\n","                                        Generalized (Near) Cullen / Woodall Siever                                        ");
    printf("%s\n","                                           Version 1.0 CUDA by A.P. August 2026                                           ");
}

#include "console_utf8.hpp"

int main(int argc, char** argv) {
    prp_console::initialize_utf8_output();
    display_banner();
    using namespace gncw_cuda;
    try {
        std::signal(SIGINT, handle_interrupt);
        Options opt;
        parse_options(argc, argv, opt);
        if (opt.help) { print_help(); return 0; }
        if (opt.version) { std::cout << APP_VERSION << '\n'; return 0; }

        Problem p; p.opt = std::move(opt);
        normalize_and_validate_problem(p);
        apply_factor_file(p);

        if (!p.opt.quiet) {
            std::cout << "GNCWSV v" << APP_VERSION << "\n"
                      << "Build: " << BUILD_ID << "\n"
                      << "mode " << static_cast<int>(p.opt.mode) << " (" << mode_name(p.opt.mode) << ")"
                      << ", a=" << p.opt.min_a << ".." << p.opt.max_a
                      << ", base=" << p.opt.base
                      << ", lattice stride=" << p.stride
                      << ", live candidates=" << active_terms(p)
                      << ", lattice slots=" << p.candidate_count
                      << ", p in (" << p.opt.min_prime << ", " << p.opt.max_prime << "]\n";
        }

        uint64_t largest = p.opt.min_prime;
        if (!p.opt.apply_and_exit) largest = run_sieve(p);
        write_terms(p, largest);
        uint64_t factors = write_factors(p);
        if (!p.opt.output_factors.empty()) std::cout << "Wrote " << factors << " new factors to " << p.opt.output_factors << ".\n";
        std::cout << "Wrote remaining terms to " << p.opt.output_terms << ".\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
