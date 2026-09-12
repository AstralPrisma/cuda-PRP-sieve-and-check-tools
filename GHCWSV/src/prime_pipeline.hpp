// SPDX-License-Identifier: GPL-2.0-or-later
// Derived from GNCWSV / GSRSV; A.P. 2026; original mtsieve authors retained in LICENSE.
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

