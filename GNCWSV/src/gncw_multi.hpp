// SPDX-License-Identifier: GPL-2.0-or-later
// Included inside gncw_cuda after the existing single-mode implementation.
// Numerical kernels remain unchanged; one prime producer supplies all modes.
#pragma once

static uint32_t mode_bit(GncwMode mode) { return 1U << (static_cast<int>(mode) - 1); }
static bool same_multi_path(const std::string& a,const std::string& b) {
    auto left=std::filesystem::weakly_canonical(std::filesystem::absolute(std::filesystem::u8path(a)));
    auto right=std::filesystem::weakly_canonical(std::filesystem::absolute(std::filesystem::u8path(b)));
    if(std::filesystem::exists(left)&&std::filesystem::exists(right)&&std::filesystem::equivalent(left,right))return true;
    auto x=left.generic_u8string(),y=right.generic_u8string();
#if defined(_WIN32)
    std::transform(x.begin(),x.end(),x.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
    std::transform(y.begin(),y.end(),y.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
#endif
    return x==y;
}
static uint64_t multi_available_ram() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{}; status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullAvailPhys : 0;
#else
    std::ifstream in("/proc/meminfo"); std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream value(line.substr(13)); uint64_t kib = 0;
            value >> kib; return kib * 1024;
        }
    }
    return 0;
#endif
}
static void multi_memory_check(uint64_t needed) {
    const uint64_t available = multi_available_ram();
    if (available && (available < (UINT64_C(512) << 20) || needed > available / 2))
        fail("Insufficient free RAM for multi-mode candidate state; narrow the range or disable -O/--verify");
}
static std::string multi_sha(const std::string& body) {
    CheckpointSha256 sha; sha.update(reinterpret_cast<const uint8_t*>(body.data()), body.size());
    std::ostringstream hex; for (uint8_t byte : sha.final()) hex << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return hex.str();
}
struct MultiInput {
    bool saved = false;
    uint64_t p = 2, amin = 0, amax = 0;
    uint32_t base = 0, mask = 0;
    std::vector<ParsedExpression> terms;
};
static bool has_multi_header(const std::string& filename) {
    std::ifstream in(filename); if (!in) fail("Unable to open input terms file: " + filename);
    std::string line; std::getline(in, line);
    return trim(line).rfind("# GNCWSV-MULTI", 0) == 0;
}
static bool requires_multi_coordinator(const Options& opt) {
    if (opt.cpu_reference || (opt.mode_mask && (opt.mode_mask & (opt.mode_mask - 1)))) return true;
    // The old odd-base lattice assumes an even value is >2. Preserve the
    // actual prime 1*3^1-1=2, and exclude 1*2^1-1=1, in either selection form.
    if(opt.input_terms.empty() && opt.mode==GncwMode::Woodall && opt.min_a==1 && opt.base<=3)return true;
    if (opt.input_terms.empty()) return false;
    const auto bytes=std::filesystem::file_size(std::filesystem::u8path(opt.input_terms));
    if(bytes>(UINT64_C(512)<<20))fail("Input file exceeds 512 MiB");
    multi_memory_check(bytes*3);
    if (has_multi_header(opt.input_terms)) return true;
    const auto input = read_expression_file(opt.input_terms);
    uint32_t mask = 0; for (const auto& term : input.entries) {
        if(term.mode==GncwMode::Woodall && term.a==1 && term.base<=3)return true;
        mask |= mode_bit(term.mode);
    }
    return (mask & (mask - 1)) != 0;
}
static MultiInput read_multi_input(const std::string& filename) {
    MultiInput result;
    const auto input_bytes=std::filesystem::file_size(std::filesystem::u8path(filename));
    if(input_bytes>(UINT64_C(512)<<20))fail("Input file exceeds 512 MiB");
    multi_memory_check(input_bytes*3);
    if (!has_multi_header(filename)) {
        auto legacy = read_expression_file(filename);
        result.terms = std::move(legacy.entries);
        result.base = result.terms.front().base;
        result.amin = AMAX_MAX;
        for (const auto& x : result.terms) {
            if (x.base != result.base) fail("Multi-mode input must use one fixed base");
            result.mask |= mode_bit(x.mode); result.amin = std::min(result.amin, x.a); result.amax = std::max(result.amax, x.a);
        }
        if (legacy.header.present) {
            if (legacy.header.base != result.base || mode_bit(legacy.header.mode) != result.mask)
                fail("Legacy resume header does not match input modes/base");
            result.saved = true; result.p = legacy.header.p;
            result.amin = legacy.header.amin; result.amax = legacy.header.amax;
        }
    } else {
        const auto size = std::filesystem::file_size(std::filesystem::u8path(filename));
        if (size > (UINT64_C(512) << 20)) fail("Multi-mode input exceeds 512 MiB");
        multi_memory_check(size * 3);
        std::ifstream file(std::filesystem::u8path(filename), std::ios::binary);
        std::string raw((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        // Canonical LF digest; CRLF transport conversion is permitted, lone CR is not.
        std::string text; text.reserve(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] == '\r') { if (i + 1 == raw.size() || raw[i + 1] != '\n') fail("Invalid CR in checkpoint"); continue; }
            text.push_back(raw[i]);
        }
        const size_t footer = text.rfind("#SHA256 ");
        if (footer == std::string::npos || footer == 0 || text[footer - 1] != '\n' || footer + 73 != text.size() || text.back() != '\n')
            fail("Missing or truncated multi-mode SHA256 footer");
        const std::string body = text.substr(0, footer);
        if (multi_sha(body) != text.substr(footer + 8, 64)) fail("Multi-mode checkpoint SHA256 mismatch");
        std::istringstream in(body); std::string line; std::getline(in, line);
        static const std::regex header(R"(^# GNCWSV-MULTI v=1 p=(\d+) base=(\d+) modes=([1-6]{1,6}) amin=(\d+) amax=(\d+) count=(\d+)$)");
        std::smatch m; if (!std::regex_match(line, m, header)) fail("Invalid multi-mode checkpoint header");
        result.saved = true;
        result.p = parse_u64(m[1].str(), 2, PMAX_MAX, "checkpoint p");
        result.base = static_cast<uint32_t>(parse_u64(m[2].str(), 2, BMAX_MAX, "checkpoint base"));
        result.mask = parse_mode_selection(m[3].str());
        if (m[3].str() != mode_selection_text(result.mask)) fail("Checkpoint modes are not canonical");
        result.amin = parse_u64(m[4].str(), 1, AMAX_MAX, "checkpoint amin");
        result.amax = parse_u64(m[5].str(), result.amin, AMAX_MAX, "checkpoint amax");
        const uint64_t count = parse_u64(m[6].str(), 0, 20000000, "checkpoint count");
        multi_memory_check(count * sizeof(ParsedExpression));
        std::pair<int,uint64_t> previous{0,0};
        while (std::getline(in, line)) {
            if (line.empty() || line.front() == '#') fail("Unexpected line inside multi-mode checkpoint");
            const auto x = parse_expression_line(line);
            const std::pair<int,uint64_t> identity{static_cast<int>(x.mode), x.a};
            if (identity <= previous || result.terms.size() >= count) fail("Duplicate, unordered or extra checkpoint candidate");
            previous = identity; result.terms.push_back(x);
        }
        if (result.terms.size() != count) fail("Multi-mode checkpoint candidate count mismatch");
    }
    std::set<std::pair<int,uint64_t>> unique;
    for (const auto& x : result.terms) {
        if (x.base != result.base || !(result.mask & mode_bit(x.mode)) || x.a < result.amin || x.a > result.amax)
            fail("Checkpoint candidate is outside the declared modes/base/range");
        if (!unique.emplace(static_cast<int>(x.mode), x.a).second) fail("Duplicate candidate expression in input");
    }
    return result;
}

struct MultiRun { Options opt; std::vector<Problem> jobs; std::vector<ParsedExpression> tiny; };
static MultiRun make_multi_run(Options opt) {
    MultiRun run; MultiInput input;
    const bool resume = !opt.input_terms.empty();
    if (resume) {
        input = read_multi_input(opt.input_terms);
        if (opt.base_explicit && opt.base != input.base) fail("--base does not match input");
        if (opt.mode_explicit && opt.mode_mask != input.mask) fail("--mode must match all saved modes; modes cannot be added/dropped during resume");
        if (input.saved && opt.min_prime_explicit && opt.min_prime > input.p) fail("--pmin would skip an unsieved interval");
        opt.base = input.base; opt.mode_mask = input.mask;
        if (input.saved) opt.min_prime = std::max(opt.min_prime, input.p);
        if (!opt.min_a_explicit) opt.min_a = input.amin;
        if (!opt.max_a_explicit) opt.max_a = input.amax;
    } else if (!opt.base_explicit || !opt.mode_explicit || !opt.min_a_explicit || !opt.max_a_explicit)
        fail("Generation requires -b, -m, -a and -A");
    if (!opt.mode_mask || opt.min_a > opt.max_a) fail("Invalid mode selection or a range");
    if ((opt.mode_mask & ((1U<<3)|(1U<<5))) && opt.min_a < 2) fail("Modes 4 and 6 require amin >= 2");
    opt.min_prime = std::max<uint64_t>(opt.min_prime, 2);
    if (opt.max_prime < opt.min_prime) fail("pmax is below the saved/start prime boundary");
    if (opt.cpu_reference && opt.max_prime - opt.min_prime > 1000000 && !opt.apply_and_exit)
        fail("--cpu-reference is limited to a prime-value interval of 1000000");
    if (opt.output_terms.empty()) opt.output_terms = "gncw_b" + std::to_string(opt.base) + "_m" + mode_selection_text(opt.mode_mask) + "_a" + std::to_string(opt.min_a) + "_" + std::to_string(opt.max_a) + ".txt";
    if ((!opt.input_factors.empty() && same_multi_path(opt.output_terms,opt.input_factors)) ||
        (!opt.output_factors.empty() && (same_multi_path(opt.output_terms,opt.output_factors) || (!opt.input_terms.empty() && same_multi_path(opt.output_factors,opt.input_terms)))))
        fail("Candidate and factor output paths must not overwrite each other");
    run.opt = opt; run.jobs.reserve(6);
    for (int mode=1; mode<=6; ++mode) {
        if (!(opt.mode_mask & (1U<<(mode-1)))) continue;
        Problem p; p.opt=opt; p.opt.mode=static_cast<GncwMode>(mode); p.opt.mode_mask=1U<<(mode-1);
        std::vector<uint64_t> selected;
        if (resume) for (const auto& x:input.terms) if (x.mode==p.opt.mode && x.a>=opt.min_a && x.a<=opt.max_a) {
            if (x.base==2 && x.mode==GncwMode::Woodall && x.a==1) continue; // N=1 is not a prime candidate.
            if (x.base==3 && x.mode==GncwMode::Woodall && x.a==1) run.tiny.push_back(x);
            else if (!(x.base&1U) || !(x.a&1ULL)) selected.push_back(x.a);
        }
        if (!resume && opt.base==3 && mode==1 && opt.min_a==1) run.tiny.push_back(parse_expression_line("1*3^1-1"));
        if (resume && selected.empty()) { p.candidate_count=0; run.jobs.push_back(std::move(p)); continue; }
        if (resume) { p.opt.min_a=*std::min_element(selected.begin(),selected.end());p.opt.max_a=*std::max_element(selected.begin(),selected.end()); }
        if ((p.opt.base&1U) && p.opt.min_a==p.opt.max_a && (p.opt.min_a&1ULL)) {run.jobs.push_back(std::move(p));continue;}
        configure_candidate_lattice(p);
        const uint64_t bytes=words_for_bits(p.candidate_count)*sizeof(uint32_t)+((!opt.output_factors.empty()||opt.verify_factors)?p.candidate_count*sizeof(uint64_t):0);
        multi_memory_check(bytes);
        fill_bits(p.term_bits,p.candidate_count,!resume);
        if (!resume && p.opt.base==2 && mode==1 && p.opt.min_a==1) bit_clear(p.term_bits,0);
        if (resume) for (uint64_t a:selected) {uint64_t index=0;if(a_to_index(p,a,index))bit_set(p.term_bits,index);}
        allocate_factor_storage(p);
        if (opt.verify_factors && p.factor.empty())p.factor.assign(static_cast<size_t>(p.candidate_count),0);
        apply_factor_file(p); run.jobs.push_back(std::move(p));
    }
    return run;
}
static uint64_t multi_active(const MultiRun& run) {uint64_t n=run.tiny.size();for(const auto& job:run.jobs)n+=active_terms(job);return n;}
static void save_multi(const MultiRun& run,uint64_t boundary) {
    if (run.opt.verify_factors) for(const auto& p:run.jobs)for(size_t i=0;i<p.factor.size();++i)
        if(p.factor[i]&&!factor_is_valid(p,p.factor[i],index_to_a(p,i)))fail("Multi-mode factor verification failed; prior checkpoint preserved");
    const auto target=std::filesystem::absolute(std::filesystem::u8path(run.opt.output_terms)).lexically_normal();
    if(std::filesystem::is_symlink(target)||std::filesystem::is_directory(target))fail("Checkpoint destination must be a regular file, not a directory/symlink");
    auto temporary=target;temporary += ".tmp-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    if(std::filesystem::exists(temporary))fail("Temporary checkpoint already exists");
    try {
        std::ofstream out(temporary,std::ios::binary|std::ios::trunc);if(!out)fail("Cannot create temporary checkpoint");
        CheckpointSha256 digest;
        auto line=[&](const std::string& s){const auto value=s+'\n';out.write(value.data(),value.size());digest.update(reinterpret_cast<const uint8_t*>(value.data()),value.size());};
        line("# GNCWSV-MULTI v=1 p="+std::to_string(boundary)+" base="+std::to_string(run.opt.base)+" modes="+mode_selection_text(run.opt.mode_mask)+" amin="+std::to_string(run.opt.min_a)+" amax="+std::to_string(run.opt.max_a)+" count="+std::to_string(multi_active(run)));
        for(const auto& p:run.jobs) {
            for(const auto& x:run.tiny)if(x.mode==p.opt.mode)line("1*3^1-1");
            for(uint64_t i=0;i<p.candidate_count;++i)if(bit_test(p.term_bits,i))line(term_text(p,index_to_a(p,i)));
        }
        std::ostringstream hex;for(uint8_t byte:digest.final())hex<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(byte);
        out<<"#SHA256 "<<hex.str()<<'\n';out.flush();if(!out)fail("Checkpoint write failed");out.close();if(!out)fail("Checkpoint close failed");
#if defined(_WIN32)
        HANDLE file=CreateFileW(temporary.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file==INVALID_HANDLE_VALUE)fail("Cannot flush checkpoint file");
        const bool flushed=FlushFileBuffers(file)!=0;CloseHandle(file);if(!flushed)fail("Checkpoint flush failed");
        if(!MoveFileExW(temporary.c_str(),target.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))fail("Atomic checkpoint replacement failed");
#else
        const int fd=::open(temporary.c_str(),O_RDONLY);if(fd<0)fail("Cannot flush checkpoint file");
        const int synced=::fsync(fd);::close(fd);if(synced!=0)fail("Checkpoint flush failed");
        std::filesystem::rename(temporary,target);
        const int directory=::open(target.parent_path().c_str(),O_RDONLY|O_DIRECTORY);
        if(directory>=0){const int result=::fsync(directory);::close(directory);if(result!=0)fail("Checkpoint directory flush failed");}
#endif
    } catch(...) {std::error_code ignored;std::filesystem::remove(temporary,ignored);throw;}
}

static int run_multi_coordinator(Options opt) {
    MultiRun run=make_multi_run(std::move(opt));const Options& common=run.opt;
    uint64_t boundary=common.min_prime,primes=0;const uint64_t initial=multi_active(run);
    const auto started=std::chrono::steady_clock::now();auto checkpoint=started,reported=started;
    std::cout<<"GNCWSV v"<<APP_VERSION<<" multi-mode "<<mode_selection_text(common.mode_mask)<<", base="<<common.base<<", candidates="<<initial<<", shared prime range=("<<boundary<<','<<common.max_prime<<"]\n";
    for(const auto& p:run.jobs)std::cout<<"mode "<<static_cast<int>(p.opt.mode)<<": "<<active_terms(p)<<" candidates\n";
    save_multi(run,boundary); // A device/producer error cannot destroy the last safe input.
    std::vector<std::unique_ptr<GpuSieve>> gpu(run.jobs.size());
    std::vector<uint64_t> remaining;for(const auto& p:run.jobs)remaining.push_back(active_terms(p));
    auto have_work=[&](){return std::any_of(remaining.begin(),remaining.end(),[](uint64_t n){return n!=0;});};
    auto download=[&](){for(size_t m=0;m<gpu.size();++m)if(gpu[m])gpu[m]->download();};
    struct ProgressSample {
        std::chrono::steady_clock::time_point wall;
        uint64_t primes, boundary;
    };
    std::deque<ProgressSample> rates{{started,0,boundary}};
    auto tick=[&](bool force) {
        const auto now=std::chrono::steady_clock::now();
        rates.push_back({now,primes,boundary});while(rates.size()>2 && rates[1].wall<now-std::chrono::seconds(60))rates.pop_front();
        if(force || now-checkpoint>=std::chrono::seconds(common.checkpoint_seconds)) {download();save_multi(run,boundary);checkpoint=now;}
        if(!common.quiet && (force || now-reported>=std::chrono::seconds(common.progress_seconds))) {
            auto rate_start=rates.front().wall;long double base_primes=rates.front().primes;
            long double base_boundary=rates.front().boundary;
            const auto cutoff=now-std::chrono::seconds(60);
            if(rates.size()>1 && rate_start<cutoff && rates[1].wall>rate_start){
                const auto span=std::chrono::duration<long double>(rates[1].wall-rate_start).count();
                const auto part=std::clamp(std::chrono::duration<long double>(cutoff-rate_start).count()/span,0.0L,1.0L);
                base_primes+=part*(rates[1].primes-rates.front().primes);
                base_boundary+=part*(rates[1].boundary-rates.front().boundary);rate_start=cutoff;
            }
            const double seconds=std::chrono::duration<double>(now-rate_start).count();
            uint64_t live=run.tiny.size();for(auto n:remaining)live+=n;
            std::cout<<"p="<<boundary<<", "<<format_prime_rate(seconds>0?double(primes-base_primes)/seconds:0)<<", unique primes="<<primes<<", removed="<<initial-live<<", remaining="<<live;
            for(size_t m=0;m<run.jobs.size();++m)std::cout<<", m"<<static_cast<int>(run.jobs[m].opt.mode)<<'='<<remaining[m];
            // ETA uses prime-value distance, not the count of tested primes.
            // A resumed run starts at its saved common boundary.
            const double elapsed=std::chrono::duration<double>(now-started).count();
            const double range_rate=seconds>0?double(boundary-base_boundary)/seconds:0;
            const double eta=estimate_eta_seconds(boundary,common.max_prime,range_rate);
            std::ostringstream timing;
            timing<<", elapsed="<<format_duration(elapsed)<<", "<<std::fixed<<std::setprecision(1)
                  <<percent_done(common.min_prime,common.max_prime,boundary)<<"% done";
            if(!have_work())timing<<", no pending sieve candidates, ETA 0s";
            else {
                // Guard the integer conversion inside format_eta_duration.
                timing<<", ETA "<<(std::isfinite(eta) && eta>=0 && eta<9.0e18?format_eta_duration(eta):"n/a");
                const auto finish=format_eta_finish_time(eta);
                if(!finish.empty())timing<<" (finish "<<finish<<')';
            }
            std::cout<<timing.str();
            std::cout<<'\n'<<std::flush;reported=now;
        }
    };
    if(!common.apply_and_exit && have_work() && boundary<common.max_prime) {
        if(common.cpu_reference) {
            for(uint64_t q=boundary+1;q<=common.max_prime && !g_interrupted && have_work();++q) {
                if(!is_prime_mr(q))continue;
                for(size_t m=0;m<run.jobs.size();++m)if(remaining[m])remaining[m]-=sieve_prime_host(run.jobs[m],q);
                boundary=q;++primes;tick(false);
            }
            if(!g_interrupted && have_work())boundary=common.max_prime;
        } else {
            CUDA_CHECK(cudaSetDevice(common.device));
            PrimeStream stream(boundary,common.max_prime,common.batch_primes,common.prime_mode,common.mr_switch_sqrt,common.segment_mib,common.prime_threads,common.prime_region_batches,common.prime_prefetch,common.cuda_streams,common.quiet);
            if(!common.quiet)std::cout<<"Shared prime generator: "<<stream.description()<<"; one producer, per-mode CUDA worksets\n";
            PrimeBatchPipeline pipeline(stream,common.prime_prefetch);PrimeBatch batch;double producer_wait=0;
            while(!g_interrupted && have_work() && pipeline.next(batch,producer_wait)) {
                for(size_t at=0;at<batch.count && !g_interrupted && have_work();) {
                    if(batch.data[at]<=common.cpu_small_prime) {
                        const uint64_t q=batch.data[at++];
                        for(size_t m=0;m<run.jobs.size();++m)if(remaining[m])remaining[m]-=sieve_prime_host(run.jobs[m],q);
                        boundary=q;++primes;tick(false);continue;
                    }
                    for(size_t m=0;m<gpu.size();++m)if(remaining[m]) {
                        if(gpu[m]&&gpu[m]->should_rebuild_compact(remaining[m])) {gpu[m]->download();gpu[m].reset();}
                        if(!gpu[m])gpu[m]=std::make_unique<GpuSieve>(run.jobs[m]);
                    }
                    size_t group_cap=common.gpu_prime_chunk;
                    if(!group_cap) {
                        group_cap=65536;
                        for(size_t m=0;m<gpu.size();++m)if(remaining[m]&&!gpu[m]->validated_64k_group())group_cap=32768;
                    }
                    const size_t count=std::min<size_t>(batch.count-at,group_cap);
                    // Do not honor SIGINT inside this group: all selected modes
                    // must finish this exact chunk before the common boundary moves.
                    std::vector<bool> submitted(gpu.size(),false);
                    for(size_t m=0;m<gpu.size();++m)if(remaining[m]) {
                        gpu[m]->submit(batch,at,count);submitted[m]=true;
                    }
                    for(size_t m=0;m<gpu.size();++m)if(submitted[m]) {
                        const auto completed=gpu[m]->retire_oldest();
                        if(completed.prime_count!=count || completed.last_prime!=batch.data[at+count-1] || completed.removed>remaining[m])fail("Multi-mode completion boundary mismatch");
                        remaining[m]-=completed.removed;
                    }
                    boundary=batch.data[at+count-1];primes+=count;at+=count;tick(false);
                }
            }
            if(!g_interrupted && have_work())boundary=common.max_prime;
        }
    }
    download();tick(true);
    uint64_t factors=0;for(const auto& p:run.jobs)factors+=write_factors(p);
    std::cout<<(g_interrupted?"Interrupted":"Finished")<<"; common p="<<boundary<<", unique primes="<<primes<<", remaining="<<multi_active(run)<<", seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"\n";
    std::cout<<"Saved all modes to "<<common.output_terms<<"; factors written="<<factors<<"\n";
    return 0;
}
