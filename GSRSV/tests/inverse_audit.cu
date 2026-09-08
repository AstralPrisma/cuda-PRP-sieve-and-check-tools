// Direct device inverse oracle for isolated GSRSV product-path experiments.
// Compile and run explicitly; this file never starts a sieve or touches queues.
#ifndef GSRSV_SOURCE
#define GSRSV_SOURCE "../src/GSRSV.cu"
#endif
#define main gsrsv_app_main
#include GSRSV_SOURCE
#undef main

#include <random>
#include <set>

#if !defined(__SIZEOF_INT128__) && !(defined(_MSC_VER) && defined(_M_X64))
#error "This audit requires native unsigned-128 host arithmetic (GCC/Clang or MSVC x64 intrinsics)."
#endif

namespace inverse_audit {
using twinsieve_cuda::Problem;
using twinsieve_cuda::TermType;
constexpr uint64_t CAP = (UINT64_C(1) << 62) - 1;
constexpr size_t MAX_PRIMES_PER_GROUP = 256;

struct Group {
    std::string label;
    int type = static_cast<int>(TermType::Factorial);
    uint32_t base = 0;
    uint32_t n = 1;
    bool real_factorial = false;
    bool real_primorial = false;
    bool allow_prime_two = false;
    std::vector<uint64_t> chunks;
    std::vector<uint32_t> original_primes;
};

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string quote(const std::string& value) {
    std::string result = "\"";
    for (unsigned char ch : value) {
        if (ch == '\\' || ch == '"') result += '\\';
        if (ch < 32) throw std::runtime_error("control character in audit label");
        result += static_cast<char>(ch);
    }
    return result + '"';
}

static uint64_t next_prime(uint64_t start) {
    uint64_t n = std::max<uint64_t>(3, start) | 1ULL;
    while (n <= CAP) {
        if (twinsieve_cuda::is_prime_mr(n)) return n;
        if (n > CAP - 2) break;
        n += 2;
    }
    throw std::runtime_error("no next audit prime inside GSRSV limit");
}

static uint64_t previous_prime(uint64_t start) {
    uint64_t n = start & 1ULL ? start : start - 1;
    for (; n >= 3; n -= 2)
        if (twinsieve_cuda::is_prime_mr(n)) return n;
    throw std::runtime_error("no previous odd audit prime");
}

static std::vector<uint32_t> independent_primes(uint32_t n) {
    std::vector<unsigned char> prime(static_cast<size_t>(n) + 1, 1);
    prime[0] = 0;
    if (n >= 1) prime[1] = 0;
    for (uint32_t d = 2; static_cast<uint64_t>(d) * d <= n; ++d) {
        if (!prime[d]) continue;
        for (uint64_t k = static_cast<uint64_t>(d) * d; k <= n; k += d)
            prime[static_cast<size_t>(k)] = 0;
    }
    std::vector<uint32_t> out;
    for (uint32_t p = 2; p <= n; ++p) if (prime[p]) out.push_back(p);
    return out;
}

static Group real_group(TermType kind, uint32_t n) {
    require(n <= 230563, "real audit construction is bounded at n=230563");
    Problem problem;
    problem.opt.term_type = kind;
    problem.opt.n = n;
    problem.opt.segment_mib = 1;
    twinsieve_cuda::build_product_chunks(problem);
    Group result;
    result.type = static_cast<int>(kind);
    result.n = n;
    result.real_factorial = kind == TermType::Factorial;
    result.real_primorial = kind == TermType::Primorial;
    result.label = (result.real_factorial ? "factorial_" : "primorial_") + std::to_string(n);
    result.chunks = std::move(problem.product_chunks);
    result.allow_prime_two = result.real_factorial && n == 1;
    require(!result.chunks.empty(), "real chunk builder returned no chunks");
    for (uint64_t x : result.chunks) require(x >= 1 && x <= CAP, "real chunk outside packing cap");
    if (result.real_primorial) {
        result.original_primes = independent_primes(n);
        require(!result.original_primes.empty() && result.original_primes.back() == n,
                "GSRSV primorial endpoint must itself be prime");
    }
    return result;
}

static std::vector<Group> groups() {
    std::vector<Group> result;
    for (uint32_t n : {1u, 2u, 7u, 10u, 100u, 1000u, 25206u})
        result.push_back(real_group(TermType::Factorial, n));
    for (uint32_t n : {2u, 3u, 7u, 97u, 104561u, 230563u})
        result.push_back(real_group(TermType::Primorial, n));
    std::mt19937_64 random(UINT64_C(20260908));
    for (uint32_t count : {0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u}) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            Group group;
            group.label = "synthetic_C" + std::to_string(count) + "_" +
                          (pattern == 0 ? "ones" : pattern == 1 ? "cap_edges" : "random");
            for (uint32_t i = 0; i < count; ++i) {
                uint64_t value = 1;
                if (pattern == 1) {
                    constexpr uint64_t edge[] = {CAP, CAP - 1, 1, 2, UINT32_MAX,
                                                  UINT64_C(1) << 32, CAP - 2};
                    value = edge[i % (sizeof(edge) / sizeof(edge[0]))];
                } else if (pattern == 2) value = 1 + random() % CAP;
                group.chunks.push_back(value);
            }
            result.push_back(std::move(group));
        }
    }
    for (uint64_t factor : {UINT64_C(3), UINT64_C(4294967291),
                            UINT64_C(4294967311), UINT64_C(2305843009213693951),
                            previous_prime(CAP)}) {
        Group group;
        group.label = "synthetic_zero_mod_" + std::to_string(factor);
        group.chunks.assign(33, CAP);
        group.chunks[16] = factor;
        result.push_back(std::move(group));
    }
    // The optional fast REDC helper also serves power-form code: test it too.
    for (const auto& pair : {std::pair<uint32_t,uint32_t>{2, 16}, {7, 325799},
                            {1337, 78647}, {32500, 32500}}) {
        Group group;
        group.label = "power_" + std::to_string(pair.first) + "_" + std::to_string(pair.second);
        group.type = static_cast<int>(TermType::BN);
        group.base = pair.first;
        group.n = pair.second;
        result.push_back(std::move(group));
    }
    return result;
}

static std::vector<uint64_t> prime_cases(const Group& group) {
    std::set<uint64_t> primes;
    for (uint64_t p : {UINT64_C(3), UINT64_C(5), UINT64_C(7), UINT64_C(17),
                       UINT64_C(257), UINT64_C(65537), UINT64_C(99991),
                       UINT64_C(100003), UINT64_C(1000003), UINT64_C(2147483647),
                       UINT64_C(4294967291), UINT64_C(4294967311),
                       UINT64_C(2305843009213693951)}) {
        require(twinsieve_cuda::is_prime_mr(p), "fixed audit modulus is not prime");
        primes.insert(p);
    }
    uint64_t low = UINT64_C(1) << 32, high = low, top = CAP;
    for (int i = 0; i < 8; ++i) {
        low = previous_prime(low - 1);
        high = next_prime(high + 1);
        top = previous_prime(top);
        primes.insert(low); primes.insert(high); primes.insert(top);
        top -= 2;
    }
    uint64_t after_n = group.n;
    for (int i = 0; i < 8; ++i) {
        after_n = next_prime(after_n + 1);
        primes.insert(after_n);
    }
    std::mt19937_64 random(UINT64_C(50814352));
    for (int i = 0; i < 16; ++i) {
        primes.insert(next_prime(3 + random() % (UINT32_MAX - UINT64_C(1000))));
        primes.insert(next_prime((UINT64_C(1) << 32) + random() % (CAP - (UINT64_C(1) << 32) - 65536)));
    }
    if (group.allow_prime_two) primes.insert(2);
    require(!primes.empty() && primes.size() <= MAX_PRIMES_PER_GROUP, "too many primes in one GPU group");
    for (uint64_t p : primes) {
        require(p <= CAP && twinsieve_cuda::is_prime_mr(p), "invalid generated audit prime");
        require(p != 2 || (group.real_factorial && group.n == 1), "p=2 only allowed for real 1!");
    }
    return {primes.begin(), primes.end()};
}

static uint64_t host_product(const Group& group, uint64_t p) {
    if (group.type == static_cast<int>(TermType::BN))
        return twinsieve_cuda::pow_mod_host(group.base, group.n, p);
    uint64_t packed = 1 % p;
    for (uint64_t chunk : group.chunks)
        packed = twinsieve_cuda::mul_mod_host(packed, chunk % p, p);
    if (group.real_factorial || group.real_primorial) {
        uint64_t direct = 1 % p;
        if (group.real_factorial) {
            for (uint32_t x = 2; x <= group.n && direct != 0; ++x)
                direct = twinsieve_cuda::mul_mod_host(direct, x % p, p);
        } else {
            for (uint32_t x : group.original_primes) {
                direct = twinsieve_cuda::mul_mod_host(direct, x % p, p);
                if (direct == 0) break;
            }
        }
        require(direct == packed, "real factorial/primorial differs from packed chunk product");
    }
    return packed;
}

__global__ void evaluate_inverse(const uint64_t* primes, uint64_t* output,
                                 uint32_t count, int type, uint32_t base, uint32_t n,
                                 const uint64_t* chunks, uint32_t chunk_count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = twinsieve_cuda::d_multiplier_inverse(primes[i], type, base, n, chunks, chunk_count);
}

struct DeviceMemory {
    uint64_t* primes = nullptr;
    uint64_t* chunks = nullptr;
    uint64_t* result = nullptr;
    ~DeviceMemory() { if(result) cudaFree(result); if(chunks) cudaFree(chunks); if(primes) cudaFree(primes); }
};

static void hash_u64(uint64_t& hash, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        hash ^= value & 255;
        hash *= UINT64_C(1099511628211);
        value >>= 8;
    }
}

static size_t run_group(const Group& group, uint64_t& digest) {
    const auto primes = prime_cases(group);
    require(group.chunks.size() <= UINT32_MAX, "chunk count overflow");
    for (uint64_t chunk : group.chunks) require(chunk <= CAP, "synthetic chunk exceeds cap");
    std::vector<uint64_t> expected, products;
    for (uint64_t p : primes) {
        const uint64_t product = host_product(group, p);
        const uint64_t inverse = product == 0 ? 0 : twinsieve_cuda::pow_mod_host(product, p - 2, p);
        require(inverse < p, "host inverse is noncanonical");
        require(product == 0 ? inverse == 0 : twinsieve_cuda::mul_mod_host(product, inverse, p) == 1,
                "host Fermat inverse failed independent multiplication check");
        products.push_back(product);
        expected.push_back(inverse);
    }
    if (twinsieve_cuda::g_interrupted) throw std::runtime_error("audit interrupted before GPU group");
    DeviceMemory device;
    CUDA_CHECK(cudaMalloc(&device.primes, primes.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&device.result, primes.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&device.chunks, std::max<size_t>(1, group.chunks.size()) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(device.primes, primes.data(), primes.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device.result, 0xa5, primes.size() * sizeof(uint64_t)));
    if (!group.chunks.empty())
        CUDA_CHECK(cudaMemcpy(device.chunks, group.chunks.data(), group.chunks.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    evaluate_inverse<<<(static_cast<unsigned>(primes.size()) + 127) / 128, 128>>>(
        device.primes, device.result, static_cast<uint32_t>(primes.size()), group.type, group.base, group.n,
        device.chunks, static_cast<uint32_t>(group.chunks.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (twinsieve_cuda::g_interrupted) throw std::runtime_error("audit interrupted after GPU group");
    std::vector<uint64_t> result(primes.size());
    CUDA_CHECK(cudaMemcpy(result.data(), device.result, result.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < primes.size(); ++i) {
        const bool ok = result[i] == expected[i] && result[i] < primes[i];
        std::cout << "{\"event\":\"inverse\",\"group\":" << quote(group.label)
                  << ",\"chunks\":" << group.chunks.size()
                  << ",\"p\":\"" << primes[i] << "\",\"product_mod\":\"" << products[i]
                  << "\",\"gpu_inverse\":\"" << result[i] << "\",\"cpu_inverse\":\"" << expected[i]
                  << "\",\"ok\":" << (ok ? "true" : "false") << "}\n";
        if (!ok) throw std::runtime_error("GPU inverse mismatch: " + group.label + " p=" + std::to_string(primes[i]));
        hash_u64(digest, static_cast<uint64_t>(group.type));
        hash_u64(digest, group.chunks.size());
        hash_u64(digest, primes[i]); hash_u64(digest, products[i]); hash_u64(digest, result[i]);
    }
    std::cout << "{\"event\":\"group\",\"group\":" << quote(group.label)
              << ",\"cases\":" << primes.size() << ",\"status\":\"PASS\"}\n" << std::flush;
    return primes.size();
}
} // namespace inverse_audit

int main(int argc, char** argv) {
    (void)argv;
    if (argc != 1) {
        std::cerr << "No run-time arguments. Choose GSRSV_SOURCE at compilation; this harness never starts a sieve.\n";
        return 2;
    }
    std::signal(SIGINT, twinsieve_cuda::handle_interrupt);
#ifdef _WIN32
    std::signal(SIGBREAK, twinsieve_cuda::handle_interrupt);
#endif
    try {
        const auto all = inverse_audit::groups();
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        inverse_audit::require(free_bytes >= 8 * 1024 * 1024, "less than 8 MiB free GPU memory");
        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::cout << "{\"event\":\"start\",\"source\":" << inverse_audit::quote(GSRSV_SOURCE)
                  << ",\"device\":" << inverse_audit::quote(properties.name)
                  << ",\"groups\":" << all.size() << ",\"max_primes_per_group\":256,\"cpu_threads\":1}\n" << std::flush;
        size_t cases = 0;
        uint64_t digest = UINT64_C(14695981039346656037);
        for (const auto& group : all) cases += inverse_audit::run_group(group, digest);
        std::cout << "{\"event\":\"summary\",\"status\":\"PASS\",\"groups\":" << all.size()
                  << ",\"inverse_cases\":" << cases << ",\"case_digest_fnv64\":\""
                  << std::hex << std::setw(16) << std::setfill('0') << digest << std::dec << "\"}\n" << std::flush;
        return 0;
    } catch (const std::exception& error) {
        if (twinsieve_cuda::g_interrupted) {
            std::cerr << "INTERRUPTED: " << error.what() << '\n';
            return 130;
        }
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
