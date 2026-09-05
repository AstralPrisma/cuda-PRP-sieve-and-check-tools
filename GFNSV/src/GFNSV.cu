/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */
// Integer-only GPU interval sieve for N=b^(2^n)+1, version 1.0.
// SHA-256-protected single-file checkpoints support Ctrl+C and continued sieving.
// Linux build: nvcc -O3 -std=c++17 -arch=sm_89 GFNSV.cu -o GFNSV
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "Shell32.lib")
#endif
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "gfnsv_state.hpp"
#if defined(_MSC_VER)
#include <boost/multiprecision/cpp_int.hpp>
#endif

using u64 = unsigned long long;
#if defined(_MSC_VER)
using u128 = boost::multiprecision::uint128_t;
#else
using u128 = unsigned __int128;
#endif
static std::atomic<int> stopped{0};
static_assert(std::atomic<int>::is_always_lock_free,
              "interrupt flag must be signal-safe and lock-free");
static void stop_handler(int) { stopped = 1; }
#ifdef _WIN32
static BOOL WINAPI console_handler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) {
        stopped = 1;
        return TRUE;
    }
    return FALSE;
}
#endif
static void install_stop_handlers() {
    std::signal(SIGINT, stop_handler);
#ifdef SIGBREAK
    std::signal(SIGBREAK, stop_handler);
#endif
#ifdef _WIN32
    if (!SetConsoleCtrlHandler(console_handler, TRUE))
        throw std::runtime_error("cannot install Ctrl+C handler");
#endif
}
#ifdef _WIN32
static std::vector<std::string> utf8_command_line() {
    int count=0;
    wchar_t** args=CommandLineToArgvW(GetCommandLineW(),&count);
    if(!args)throw std::runtime_error("cannot read Unicode command line");
    std::vector<std::string> result;
    try {
        result.reserve(static_cast<std::size_t>(count));
        for(int i=0;i<count;++i) {
            const int length=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,args[i],-1,nullptr,0,nullptr,nullptr);
            if(length<=0)throw std::runtime_error("invalid Unicode command line");
            std::string text(static_cast<std::size_t>(length),'\0');
            if(!WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,args[i],-1,text.data(),length,nullptr,nullptr))
                throw std::runtime_error("cannot convert command line to UTF-8");
            text.pop_back();
            result.push_back(std::move(text));
        }
    }catch(...) {LocalFree(args);throw;}
    LocalFree(args);
    return result;
}
#endif
static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
#define CUDA(x) ck((x), #x)
static constexpr u64 MAX_P = (1ULL << 62) - 1;
static constexpr unsigned ROOT_THREADS = 256;

struct Mont {
    u64 p, ni, r2, one;
};
__device__ __forceinline__ u64 mm(u64 a, u64 b, const Mont& m) {
    u64 lo=a*b, hi=__umul64hi(a,b), t=lo*m.ni;
    u64 tl=t*m.p, th=__umul64hi(t,m.p);
    u64 v=hi+th+(lo+tl<lo);
    return v>=m.p?v-m.p:v;
}
__device__ __forceinline__ Mont make_mont(u64 p) {
    u64 x=1;
#pragma unroll
    for (int i=0;i<6;++i) x*=2-p*x;
    Mont m{p,0-x,0,(0ULL-p)%p};
    x=m.one;
#pragma unroll 1
    for (int i=0;i<64;++i) x=x>=p-x?x-(p-x):x+x;
    m.r2=x;
    return m;
}
__device__ __forceinline__ u64 mpow(u64 a, u64 e, const Mont& m) {
    u64 r=m.one;
    while(e) {
        if(e&1) r=mm(r,a,m);
        e>>=1;
        if(e) a=mm(a,a,m);
    }
    return r;
}
__device__ bool prime(u64 p, const Mont& m) {
    u64 d=p-1;
    unsigned s=__ffsll(d)-1;
    d>>=s;
    const u64 witnesses[]={2,325,9375,28178,450775,9780504,1795265022};
    for(u64 a:witnesses) {
        a%=p;
        if(!a) continue;
        u64 x=mpow(mm(a,m.r2,m),d,m);
        if(x==m.one || x==p-m.one) continue;
        bool pass=false;
        for(unsigned j=1;j<s;++j) {
            x=mm(x,x,m);
            if(x==p-m.one) { pass=true; break; }
        }
        if(!pass) return false;
    }
    return true;
}
struct Prime {
    Mont m;
    u64 h, mult, stride, start_mod;
};
__global__ void prepare(u64 kfirst, unsigned count, unsigned n, u64 bstart,
                        Prime* found, unsigned* found_count) {
    unsigned idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=count) return;
    u64 p=((kfirst+idx)<<(n+1))+1;
    const unsigned trial_primes[]={3,5,7,11,13,17,19,23,29,31,37,41,43,47};
    for(unsigned s:trial_primes) if(p!=s && p%s==0) return;
    Mont m=make_mont(p);
    if(!prime(p,m)) return;
    u64 h=0;
    for(u64 a=2;;++a) {
        h=mpow(mm(a,m.r2,m),(p-1)>>(n+1),m);
        u64 x=h;
        for(unsigned j=0;j<n;++j) x=mm(x,x,m);
        if(x==p-m.one) break;
    }
    u64 mult=mm(h,h,m);
    Prime rec{m,h,mult,mpow(mult,ROOT_THREADS,m),bstart%p};
    found[atomicAdd(found_count,1)]=rec;
}
__device__ bool same_as_factor(u64 b, unsigned n, u64 p) {
    // p=N is not a proper factor; all products bounded by p-1.
    if(b>=p) return false;
    u64 v=b, target=p-1;
    for(unsigned j=0;j<n;++j) {
        if(v>target/v) return false;
        v*=v;
    }
    return v==target;
}
__device__ __forceinline__ void mark_root(u64 r,const Prime& rec,unsigned n,u64 bstart,u64 slots,u64* first_factor) {
    u64 delta=r>=rec.start_mod?r-rec.start_mod:rec.m.p-(rec.start_mod-r);
    if(delta&1) delta+=rec.m.p;
    u64 idx=delta>>1;
    while(idx<slots) {
        u64 b=bstart+2*idx;
        if(!same_as_factor(b,n,rec.m.p)) atomicCAS(first_factor+idx,0ULL,rec.m.p);
        // slots <= 64M, p <= 2^62-1: no overflow.
        idx+=rec.m.p;
    }
}
template<bool PairRoots>
__global__ void sieve_roots(const Prime* primes, unsigned n, u64 bstart,
                           u64 slots, u64* first_factor) {
    Prime rec=primes[blockIdx.x];
    const Mont& m=rec.m;
    u64 root=mm(rec.h,mpow(rec.mult,threadIdx.x,m),m);
    // For E=2^n, -h^(2j+1)=h^(2(j+E/2)+1); this halves modular multiplications.
    unsigned total=1U<<(n-(PairRoots?1:0));
    for(unsigned j=threadIdx.x;j<total;j+=ROOT_THREADS) {
        u64 r=mm(root,1,m);
        mark_root(r,rec,n,bstart,slots,first_factor);
        if(PairRoots)mark_root(m.p-r,rec,n,bstart,slots,first_factor);
        root=mm(root,rec.stride,m);
    }
}

static u64 mul_host(u64 a,u64 b,u64 p) { return u64(u128(a)*b%p); }
static u64 pow_host(u64 a,u64 e,u64 p) {
    u64 r=1;
    a%=p;
    while(e) { if(e&1) r=mul_host(r,a,p); e>>=1; if(e) a=mul_host(a,a,p); }
    return r;
}
static bool prime_host(u64 p) {
    if(p<2) return false;
    for(u64 s:{2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL}) {
        if(p==s) return true;
        if(p%s==0) return false;
    }
    u64 d=p-1; unsigned s=0; while(!(d&1)) { d>>=1; ++s; }
    for(u64 a:{2ULL,325ULL,9375ULL,28178ULL,450775ULL,9780504ULL,1795265022ULL}) {
        if(a%p==0) continue;
        u64 x=pow_host(a,d,p);
        if(x==1||x==p-1) continue;
        bool pass=false;
        for(unsigned j=1;j<s;++j) { x=mul_host(x,x,p); if(x==p-1) {pass=true;break;} }
        if(!pass) return false;
    }
    return true;
}
static bool same_host(u64 b,unsigned n,u64 p) {
    if(b>=p) return false;
    u64 v=b;
    for(unsigned j=0;j<n;++j) {if(v>(p-1)/v)return false;v*=v;}
    return v==p-1;
}
static u64 number(std::string s) {
    s.erase(std::remove(s.begin(),s.end(),'_'),s.end());
    if(s.empty()||s[0]=='-'||s[0]=='+') throw std::runtime_error("invalid unsigned integer: "+s);
    auto pos=s.find_first_of("eE");
    if(pos!=std::string::npos && s.rfind("0x",0)!=0 && s.rfind("0X",0)!=0) {
        u64 a=number(s.substr(0,pos)), e=number(s.substr(pos+1));
        if(e>19) throw std::runtime_error("integer exponent too large");
        for(u64 i=0;i<e;++i) {if(a>~0ULL/10)throw std::runtime_error("integer overflow");a*=10;}
        return a;
    }
    std::size_t used=0;
    u64 v=std::stoull(s,&used,(s.rfind("0x",0)==0||s.rfind("0X",0)==0)?16:10);
    if(used!=s.size()) throw std::runtime_error("invalid integer: "+s);
    return v;
}
struct Options {
    unsigned n=16,batch=65536,device=0;
    u64 bmin=0,bmax=0,pmin=3,pmax=0;
    bool verify=true,quiet=false,pairs=true;
    std::string out="cand_gpu.txt", factors, format="base";
    bool resume=false,n_given=false,bmin_given=false,bmax_given=false;
    bool pmin_given=false,pmax_given=false,format_given=false,out_given=false;
    std::string resume_file;
    double state_every=30.0;
};
static void usage() {
    std::cout<<"GFNSV CUDA 1.0 (integer-only, root/prime parallelism)\n"
      <<"  --n N --bmin B --bmax B --pmax P [options]\n"
      <<"Ranges are inclusive. Only even b>=2 are considered. n=1..20; p<=2^62-1.\n"
      <<"  --pmin P           inclusive lower factor bound (default 3)\n"
      <<"  --batch N          AP candidates per GPU batch (default 65536, max 1048576)\n"
      <<"  --device N         CUDA device (default 0)\n"
      <<"  --out FILE         sorted surviving bases, default cand_gpu.txt\n"
      <<"  --resume [FILE]    continue a saved CUDA checkpoint; otherwise uses --out\n"
      <<"  --state-every S    periodic save interval in seconds, default 30; 0=interrupt/end only\n"
      <<"  --checkpoint-info FILE  inspect a saved checkpoint without using the GPU\n"
      <<"  --factors FILE     one factor per removal (CPU verified by default)\n"
      <<"  --format base|expr output format\n"
      <<"  --no-verify        omit independent CPU factor verification\n"
      <<"  --root-pairs       paired roots r and -r (default, faster)\n"
      <<"  --full-roots       enumerate every root (reference path)\n"
      <<"  --quiet            hide batch progress\n"
      <<"Decimal/hex/underscores/integer scientific notation accepted.\n"
      <<"Ctrl+C drains the current GPU batch, saves --out atomically, and exits with code 130.\n"
      <<"Resume restores n/base range/progress from the same file; --pmax may extend the search.\n"
      <<"Checkpoint comments store progress, factors, and SHA-256; non-comment lines are survivors.\n"
      <<"Fresh output must not exist. Arbitrary sparse/legacy CPU files are not resume checkpoints.\n"
      <<"Maximum 64M candidate slots; partial and completed CUDA checkpoints can both resume.\n";
}
static Options options(int argc,char**argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto val=[&](){if(++i>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[i]);};
        if(a=="--n") {u64 v=number(val());if(v<1||v>20)throw std::runtime_error("n must be 1..20");o.n=unsigned(v);o.n_given=true;}
        else if(a=="--bmin") {o.bmin=number(val());o.bmin_given=true;}
        else if(a=="--bmax") {o.bmax=number(val());o.bmax_given=true;}
        else if(a=="--pmin") {o.pmin=number(val());o.pmin_given=true;}
        else if(a=="--pmax") {o.pmax=number(val());o.pmax_given=true;}
        else if(a=="--batch") {u64 v=number(val());if(v<1||v>1048576)throw std::runtime_error("batch must be 1..1048576");o.batch=unsigned(v);}
        else if(a=="--device") {u64 v=number(val());if(v>1024)throw std::runtime_error("invalid device");o.device=unsigned(v);}
        else if(a=="--out") {o.out=val();o.out_given=true;}
        else if(a=="--factors")o.factors=val();
        else if(a=="--format") {o.format=val();o.format_given=true;}
        else if(a=="--state-every") {
            const std::string text=val();std::size_t used=0;
            o.state_every=std::stod(text,&used);
            if(used!=text.size()||!std::isfinite(o.state_every)||o.state_every<0||o.state_every>86400)
                throw std::runtime_error("--state-every must be 0..86400 seconds");
        }
        else if(a=="--no-verify")o.verify=false;
        else if(a=="--root-pairs")o.pairs=true;
        else if(a=="--full-roots")o.pairs=false;
        else if(a=="--quiet")o.quiet=true;
        else if(a=="--help"||a=="-h") {usage();std::exit(0);}
        else if(a=="--resume") {
            if(o.resume)throw std::runtime_error("--resume specified more than once");
            o.resume=true;
            if(i+1<argc && argv[i+1][0]!='-')o.resume_file=val();
        }
        else throw std::runtime_error("unknown option: "+a);
    }
    if(!o.resume && (o.bmin<2||o.bmax<o.bmin||o.pmin<3||o.pmax<o.pmin||o.pmax>MAX_P))
        throw std::runtime_error("invalid b/p range (bmin>=2, pmax<=2^62-1)");
    if(o.format!="base"&&o.format!="expr")throw std::runtime_error("format must be base or expr");
    if(o.resume) {
        if(!o.resume_file.empty()&&!o.out_given)o.out=o.resume_file;
        if(o.resume_file.empty())o.resume_file=o.out;
    }
    if(o.out.empty())throw std::runtime_error("output path must be nonempty");
    if(!o.factors.empty() &&
       (gfnsv_state::same_path(o.out,o.factors)||
        (o.resume&&gfnsv_state::same_path(o.resume_file,o.factors))))
        throw std::runtime_error("candidate checkpoint and factor log must be different files");
    const bool replacing_resume=o.resume&&gfnsv_state::same_path(o.out,o.resume_file);
    if(!replacing_resume&&std::filesystem::exists(std::filesystem::u8path(o.out)))
        throw std::runtime_error("output exists; use --resume --out FILE to continue it");
    if(!o.resume&&!o.factors.empty()&&std::filesystem::exists(std::filesystem::u8path(o.factors)))
        throw std::runtime_error("factor output exists; choose a new path");
    return o;
}
template<class T> struct DeviceArray {
    T* p=nullptr;
    explicit DeviceArray(std::size_t n) {if(n)CUDA(cudaMalloc(&p,n*sizeof(T)));}
    ~DeviceArray(){if(p)cudaFree(p);}
    DeviceArray(const DeviceArray&)=delete;
    DeviceArray&operator=(const DeviceArray&)=delete;
};
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}

static u64 removed_count(const gfnsv_state::State& state) {
    return static_cast<u64>(std::count_if(state.factors.begin(),state.factors.end(),
        [](u64 factor){return factor!=0;}));
}

static void verify_factors(const gfnsv_state::State& state) {
    const u64 first=gfnsv_state::first_even(state),q=1ULL<<(state.n+1);
    const u64 frontier=gfnsv_state::next_p(state);
    std::vector<u64> used;
    for(std::size_t i=0;i<state.factors.size();++i)if(state.factors[i]) {
        const u64 b=first+2*static_cast<u64>(i),p=state.factors[i];
        if(p<state.pmin||p>state.pmax||p>=frontier||(p-1)%q||
           pow_host(b,1ULL<<state.n,p)!=p-1||same_host(b,state.n,p))
            throw std::runtime_error("CPU factor verification failed at b="+std::to_string(b));
        used.push_back(p);
    }
    std::sort(used.begin(),used.end());used.erase(std::unique(used.begin(),used.end()),used.end());
    for(u64 p:used)if(!prime_host(p))throw std::runtime_error("non-prime factor in checkpoint");
}

static void persist_checkpoint(const Options& opt,const gfnsv_state::State& state,
                               const char* reason,bool& factor_owned) {
    if(opt.verify)verify_factors(state);
    gfnsv_state::write_state_atomic(opt.out,state);
    std::cout<<"checkpoint: saved path="<<opt.out
             <<", next_p="<<gfnsv_state::next_p(state)
             <<", survivors="<<(state.factors.size()-removed_count(state))
             <<", reason="<<reason<<"\n";
    if(!opt.factors.empty()) {
        try {
            gfnsv_state::write_factors_atomic(opt.factors,state,factor_owned);
            factor_owned=true;
        }catch(const std::exception& e) {
            throw std::runtime_error(std::string("candidate checkpoint was saved, but factor export failed: ")+e.what());
        }
    }
}

static gfnsv_state::State initial_state(Options& opt) {
    gfnsv_state::State state;
    if(opt.resume) {
        state=gfnsv_state::read_state(opt.resume_file);
        if((opt.n_given&&opt.n!=state.n)||
           (opt.bmin_given&&opt.bmin!=state.bmin)||
           (opt.bmax_given&&opt.bmax!=state.bmax)||
           (opt.pmin_given&&opt.pmin!=state.pmin)||
           (opt.format_given&&(opt.format=="expr")!=state.expr))
            throw std::runtime_error("resume n/base range/pmin/format does not match saved checkpoint");
        if(opt.pmax_given) {
            const u64 q=1ULL<<(state.n+1);
            const u64 first_k=gfnsv_state::first_k(state.pmin,state.n);
            const u64 processed_bound=state.next_k>first_k?(state.next_k-1)*q+1:state.pmin-1;
            if(opt.pmax<state.pmin||opt.pmax>MAX_P||opt.pmax<processed_bound)
                throw std::runtime_error("--pmax cannot be below already processed factors or outside supported bounds");
            state.pmax=opt.pmax;
        }
        opt.n=state.n;opt.bmin=state.bmin;opt.bmax=state.bmax;
        opt.pmin=state.pmin;opt.pmax=state.pmax;opt.format=state.expr?"expr":"base";
        if(opt.verify)verify_factors(state);
        std::cout<<"resume: loaded path="<<opt.resume_file<<", next_p="<<gfnsv_state::next_p(state)
                 <<", survivors="<<(state.factors.size()-removed_count(state))<<"\n";
    }else {
        state.n=opt.n;state.bmin=opt.bmin;state.bmax=opt.bmax;
        state.pmin=opt.pmin;state.pmax=opt.pmax;state.expr=opt.format=="expr";
        state.next_k=gfnsv_state::first_k(state.pmin,state.n);
        const u64 slots=gfnsv_state::slot_count(state);
        state.factors.assign(static_cast<std::size_t>(slots),0);
        if(slots==0)state.next_k=gfnsv_state::last_k(state.pmax,state.n)+1;
    }
    return state;
}

static int checkpoint_info(const std::string& path) {
    const auto state=gfnsv_state::read_state(path);
    std::cout<<"checkpoint-info: n="<<state.n<<", bmin="<<state.bmin<<", bmax="<<state.bmax
             <<", pmin="<<state.pmin<<", pmax="<<state.pmax
             <<", next_p="<<gfnsv_state::next_p(state)<<", next_k="<<state.next_k
             <<", survivors="<<(state.factors.size()-removed_count(state))
             <<", complete="<<((state.factors.empty()||state.next_k>gfnsv_state::last_k(state.pmax,state.n))?"yes":"no")<<"\n";
    return 0;
}

void display_banner() {
    printf("%s\n","════════════════════════════════════════════════════════════════════════════════════════════════");
    printf("%s\n","       .oooooo.        oooooooooooo     ooooo      ooo      .oooooo..o   oooooo     oooo        ");
    printf("%s\n","      d8P'  `Y8b       `888'     `8     `888b.     `8'     d8P'    `Y8    `888.     .8'         ");
    printf("%s\n","     888                888              8 `88b.    8      Y88bo.          `888.   .8'          ");
    printf("%s\n","     888                888oooo8         8   `88b.  8       `'Y8888o.       `888. .8'           ");
    printf("%s\n","     888     ooooo      888    '         8     `88b.8           `'Y88b       `888.8'            ");
    printf("%s\n","     `88.    .88'  .o.  888         .o.  8       `888  .o. oo     .d8P .o.    `888'    .o.      ");
    printf("%s\n","      `Y8bood8P'   Y8P o888o        Y8P o8o        `8  Y8P 8''88888P'  Y8P     `8'     Y8P      ");
    printf("%s\n","════════════════════════════════════════════════════════════════════════════════════════════════");
    printf("%s\n","                                Generalized-Fermat-Number-Siever                                ");
    printf("%s\n","                               Version 1.0 CUDA by A.P. Sept 2026                               ");
}

int main(int argc,char**argv) {
    display_banner();
    std::fflush(stdout);
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    try {
        install_stop_handlers();
#ifdef _WIN32
        auto utf8_args=utf8_command_line();
        std::vector<char*> argument_pointers;
        argument_pointers.reserve(utf8_args.size());
        for(auto& arg:utf8_args)argument_pointers.push_back(arg.data());
        argc=static_cast<int>(argument_pointers.size());argv=argument_pointers.data();
#endif
        if(argc==1){usage();return 0;}
        if(argc==3&&std::string(argv[1])=="--checkpoint-info")return checkpoint_info(argv[2]);
        Options o=options(argc,argv);
        auto state=initial_state(o);
        const u64 first=gfnsv_state::first_even(state),slots=state.factors.size();
        bool factor_owned=o.resume;
        if(!o.factors.empty())gfnsv_state::validate_factor_destination(o.factors,state,factor_owned);
        const auto overall_started=Clock::now();
        double checkpoint_s=0;
        const auto persist=[&](const char* reason) {
            const auto t=Clock::now();
            persist_checkpoint(o,state,reason,factor_owned);
            checkpoint_s+=seconds(t);
        };
        // Create a complete, recoverable initial snapshot before any GPU work.
        // On resume the existing file is replaced only after validation succeeds.
        persist("initial");
        const auto interrupted=[&]() {
            std::cerr<<"interrupted: checkpoint saved; resume with --resume --out \""<<o.out<<"\"\n";
            return 130;
        };
        if(stopped) {persist("interrupt");return interrupted();}
        const u64 q=1ULL<<(o.n+1),klast=gfnsv_state::last_k(o.pmax,o.n);
        if(slots==0||state.next_k>klast) {
            persist("complete");
            std::cout<<"done: no new factor interval; survivors="<<(slots-removed_count(state))
                     <<", next_p="<<gfnsv_state::next_p(state)<<"\n";
            return 0;
        }
        CUDA(cudaSetDevice(o.device));
        cudaDeviceProp prop{};CUDA(cudaGetDeviceProperties(&prop,o.device));
        std::size_t free=0,total=0;CUDA(cudaMemGetInfo(&free,&total));
        std::size_t needed=slots*sizeof(u64)+o.batch*sizeof(Prime)+sizeof(unsigned);
        if(needed>free/2)throw std::runtime_error("GPU free memory too low for safe allocation");
        DeviceArray<u64> factors(slots);
        DeviceArray<Prime> primes(o.batch);
        DeviceArray<unsigned> count(1);
        CUDA(cudaMemcpy(factors.p,state.factors.data(),slots*sizeof(u64),cudaMemcpyHostToDevice));
        u64 np=0,nk=0;
        u64 batches=0,stop_after_batches=0;
        if(const char* text=std::getenv("GFNSV_STOP_AFTER_BATCHES"))stop_after_batches=number(text);
        double prep_s=0,sieve_s=0;
        auto start=Clock::now();
        auto last_saved=start;
        const auto snapshot=[&](const char* reason) {
            // Only called after all roots in the current batch have completed.
            CUDA(cudaMemcpy(state.factors.data(),factors.p,slots*sizeof(u64),cudaMemcpyDeviceToHost));
            persist(reason);
            last_saved=Clock::now();
        };
        std::cout<<"device="<<prop.name<<", n="<<o.n<<", candidates="<<slots
          <<", p=["<<gfnsv_state::next_p(state)<<","<<o.pmax<<"], batch="<<o.batch
          <<", root_pairs="<<(o.pairs?"yes":"no")<<", verify="<<(o.verify?"yes":"no")
          <<", state_every="<<o.state_every<<"\n";
        while(state.next_k<=klast) {
            if(stopped)break;
            const u64 k=state.next_k;
            unsigned batch=unsigned(std::min<u64>(o.batch,klast-k+1));
            unsigned found=0;
            auto t=Clock::now();
            CUDA(cudaMemset(count.p,0,sizeof(unsigned)));
            prepare<<<(batch+127)/128,128>>>(k,batch,o.n,first,primes.p,count.p);
            CUDA(cudaGetLastError());
            CUDA(cudaMemcpy(&found,count.p,sizeof(unsigned),cudaMemcpyDeviceToHost));
            prep_s+=seconds(t);
            t=Clock::now();
            if(found) {
                if(o.pairs)sieve_roots<true><<<found,ROOT_THREADS>>>(primes.p,o.n,first,slots,factors.p);
                else sieve_roots<false><<<found,ROOT_THREADS>>>(primes.p,o.n,first,slots,factors.p);
                CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
            }
            sieve_s+=seconds(t);np+=found;nk+=batch;
            // Publish the first unfinished k only after prepare AND every root
            // in this batch have completed. Interrupts never skip a partial p.
            state.next_k=k+batch;
            ++batches;
            if(!o.quiet)std::cout<<"progress: p="<<((k+batch-1)*q+1)<<", primes="<<np<<", elapsed_s="<<seconds(start)<<"\n";
            if(stop_after_batches!=0&&batches>=stop_after_batches)stopped=1;
            if(stopped)break;
            if(o.state_every>0&&seconds(last_saved)>=o.state_every)snapshot("periodic");
        }
        snapshot(stopped?"interrupt":"complete");
        if(stopped)return interrupted();
        const u64 removed=removed_count(state);
        std::cout<<std::fixed<<std::setprecision(6)<<"done: tested_ap="<<nk<<", primes="<<np<<", roots="<<(np*(1ULL<<o.n))
          <<", removed="<<removed<<", survivors="<<(slots-removed)<<", prepare_s="<<prep_s<<", sieve_s="<<sieve_s
          <<", gpu_wall_s="<<(prep_s+sieve_s)<<", checkpoint_s="<<checkpoint_s
          <<", next_p="<<gfnsv_state::next_p(state)<<", total_s="<<seconds(overall_started)<<"\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
