// SPDX-License-Identifier: GPL-2.0-or-later
// GHCWSV, Copyright (C) 2026 AstralPrisma (A.P.).
// Infrastructure derived from GNCWSV / GSRSV and GPL mtsieve (Mark Rodenkirch).
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "Shell32.lib")
#include <intrin.h>
#include <io.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#endif
#include "sha256.hpp"
#include "ghcw_model.hpp"
#include "console_utf8.hpp"
#include "work_batching.hpp"

namespace ghcw_sieve {
constexpr uint64_t PMAX_MAX=(uint64_t(1)<<62)-1;
constexpr size_t MAX_CANDIDATES=10000000;
constexpr size_t MAX_SNAPSHOT_BYTES=320*1024*1024;
constexpr uint32_t PLUS=1u, MINUS=2u;
static volatile std::sig_atomic_t g_interrupted=0;
static void handle_interrupt(int) {g_interrupted=1;}
#ifdef _WIN32
static BOOL WINAPI console_handler(DWORD kind) {
    if(kind==CTRL_C_EVENT || kind==CTRL_BREAK_EVENT){g_interrupted=1;return TRUE;}return FALSE;
}
#endif
[[noreturn]] static void fail(const std::string& s){throw std::runtime_error(s);}
static void cuda_check(cudaError_t e,const char* text) {if(e!=cudaSuccess)fail(std::string(text)+": "+cudaGetErrorString(e));}
#define CUDA_CHECK(expr) cuda_check((expr),#expr)
enum class PrimeMode {Auto,PrimeSieve,Segmented,MillerRabin};
#include "prime_pipeline.hpp"
#include "mont64.cuh"

// Extended Euclid coefficients fit int64: modulus <= 2^62-2. No inverse is
// assumed when gcd != 1; those primes always use the direct, complete path.
__host__ __device__ uint64_t inverse_exponent(uint64_t b,uint64_t modulus) {
    uint64_t r=modulus,nr=b%modulus;int64_t t=0,nt=1;
    while(nr){uint64_t q=r/nr,rr=r-q*nr;int64_t tt=t-int64_t(q)*nt;r=nr;nr=rr;t=nt;nt=tt;}
    if(r!=1)return 0;if(t<0)t+=int64_t(modulus);return uint64_t(t);
}
__host__ __device__ uint64_t power_capped(uint64_t a,uint64_t e,uint64_t cap) {
    uint64_t r=1;
    while(e){if(e&1){if(a>cap || r>cap/a)return cap+1;r*=a;}e>>=1;
        if(e){if(a>cap || a>cap/a)a=cap+1;else a*=a;}}
    return r;
}
__host__ __device__ bool term_equals_prime(uint32_t b,uint32_t n,int c,uint64_t p) {
    uint64_t want=c==1?p-1:p+1;
    uint64_t x=power_capped(b,n,want),y=power_capped(n,b,want);
    return x<=want && y<=want && x<=want/y && x*y==want;
}

__global__ void sieve_kernel(const uint64_t* primes,size_t count,const uint32_t* ns,
        size_t nc,uint32_t b,uint32_t sign_mask,uint32_t* alive,uint64_t* factors,bool direct,const uint64_t* coefficients) {
    constexpr size_t tile=256;
    const size_t jobs=count*((nc+tile-1)/tile);
    for(size_t job=blockIdx.x*blockDim.x+threadIdx.x;job<jobs;job+=size_t(blockDim.x)*gridDim.x){
        const size_t pi=job%count,first=(job/count)*tile,last=nc<first+tile?nc:first+tile;
        uint64_t p=primes[pi];if(p==2 || b%p==0 || !nc)continue;
        const Mont64 mc=d_make_mont(p);
        const uint64_t bm=d_mont_mul(b%p,mc.r2,mc);
        uint64_t e=direct||coefficients?0:inverse_exponent(b,p-1);
        bool transformed=e!=0 || coefficients;
        uint64_t step=coefficients?d_mont_mul(d_inverse_base(b,p),mc.r2,mc):e?d_mont_pow_rep(bm,p-1-e,mc):bm;
        uint64_t gaps[16];gaps[0]=step;
        for(int j=1;j<16;++j)gaps[j]=d_mont_mul(gaps[j-1],step,mc);
        uint64_t value=d_mont_pow_rep(step,ns[first],mc);
        if(transformed)value=d_mont_mul(value,1,mc);
        for(size_t i=first;i<last;++i){
            bool hit_plus,hit_minus;
            if(transformed) {
                uint64_t target=coefficients?coefficients[i]:uint64_t(ns[i])%p;
                hit_plus=(p-value)==target;hit_minus=value==target;
            }
            else {
                uint64_t nm=d_mont_mul(uint64_t(ns[i])%p,mc.r2,mc);
                uint64_t nb=d_mont_pow_rep(nm,b,mc);
                uint64_t product=d_mont_mul(value,nb,mc);
                hit_plus=product==p-mc.rmod;hit_minus=product==mc.rmod;
            }
            // Each sign has an independent atomic bit and factor slot. Never
            // remove the opposite sign, and preserve a prime equal to its divisor.
            if((sign_mask&PLUS) && hit_plus && !term_equals_prime(b,ns[i],1,p)
                    && (atomicAnd(alive+i,~PLUS)&PLUS))factors[2*i]=p;
            if((sign_mask&MINUS) && hit_minus && !term_equals_prime(b,ns[i],-1,p)
                    && (atomicAnd(alive+i,~MINUS)&MINUS))factors[2*i+1]=p;
            if(i+1<last){uint64_t gap=uint64_t(ns[i+1])-ns[i];
                uint64_t multiplier=gap<=16?gaps[gap-1]:d_mont_pow_rep(step,gap,mc);
                value=d_mont_mul(value,multiplier,mc);
            }
        }
    }
}

struct ModTest {uint64_t p;uint32_t b,n;};
__global__ void mod_test_kernel(const ModTest* tests,size_t count,uint64_t* products,uint64_t* inverses,uint64_t* targets){
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    auto t=tests[i];auto m=d_make_mont(t.p);
    auto bm=d_mont_mul(t.b%t.p,m.r2,m),nm=d_mont_mul(t.n%t.p,m.r2,m);
    auto product=d_mont_mul(d_mont_pow_rep(bm,t.n,m),d_mont_pow_rep(nm,t.b,m),m);
    products[i]=d_mont_mul(product,1,m);
    auto e=t.b%t.p?inverse_exponent(t.b,t.p-1):0;inverses[i]=e;targets[i]=0;
    if(e){auto step=d_mont_pow_rep(bm,t.p-1-e,m);targets[i]=d_mont_mul(d_mont_pow_rep(step,t.n,m),1,m);}
}
int selftest(){
    std::vector<ModTest> tests;
    for(uint64_t low:std::vector<uint64_t>{3ull,65521ull,4294967200ull,1000000000000ull,100000000000000ull,100000000000000000ull,PMAX_MAX-2000}){
        while(!is_prime_mr(low))++low;
        for(uint32_t b:{2u,3u,7u,325u,65537u,UINT32_MAX})for(uint32_t n:{2u,3u,17u,65537u,1000000u,UINT32_MAX})tests.push_back({low,b,n});
    }
    ModTest* dt=nullptr;uint64_t *dp=nullptr,*di=nullptr,*dv=nullptr;
    try{
        CUDA_CHECK(cudaMalloc(&dt,tests.size()*sizeof(ModTest)));CUDA_CHECK(cudaMalloc(&dp,tests.size()*8));
        CUDA_CHECK(cudaMalloc(&di,tests.size()*8));CUDA_CHECK(cudaMalloc(&dv,tests.size()*8));
        CUDA_CHECK(cudaMemcpy(dt,tests.data(),tests.size()*sizeof(ModTest),cudaMemcpyHostToDevice));
        mod_test_kernel<<<int((tests.size()+127)/128),128>>>(dt,tests.size(),dp,di,dv);CUDA_CHECK(cudaGetLastError());
        std::vector<uint64_t> products(tests.size()),inverses(tests.size()),targets(tests.size());
        CUDA_CHECK(cudaMemcpy(products.data(),dp,tests.size()*8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(inverses.data(),di,tests.size()*8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(targets.data(),dv,tests.size()*8,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<tests.size();++i){auto t=tests[i];
            auto want=mul_mod_host(pow_mod_host(t.b,t.n,t.p),pow_mod_host(t.n,t.b,t.p),t.p);
            if(products[i]!=want)fail("selftest modular product mismatch");
            bool valid=t.b%t.p && std::gcd(uint64_t(t.b),t.p-1)==1;
            if(bool(inverses[i])!=valid)fail("selftest inverse availability mismatch");
            if(valid && (mul_mod_host(t.b,inverses[i],t.p-1)!=1 || targets[i]!=pow_mod_host(pow_mod_host(t.b,t.p-1-inverses[i],t.p),t.n,t.p)))
                fail("selftest transformed target mismatch");
        }
    }catch(...){if(dt)cudaFree(dt);if(dp)cudaFree(dp);if(di)cudaFree(di);if(dv)cudaFree(dv);throw;}
    cudaFree(dt);cudaFree(dp);cudaFree(di);cudaFree(dv);
    std::cout<<"selftest: PASS, "<<tests.size()<<" GPU residues/transform targets verified, p up to2^62\n";return 0;
}

std::string sha(const std::string& bytes) {
    CheckpointSha256 h;h.update(reinterpret_cast<const uint8_t*>(bytes.data()),bytes.size());auto digest=h.final();
    std::ostringstream out;for(auto x:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(x);return out.str();
}
void atomic_text(const std::filesystem::path& path,const std::string& text) {
    auto tmp=path;
#ifdef _WIN32
    tmp+=".tmp."+std::to_string(GetCurrentProcessId());FILE* f=_wfopen(tmp.c_str(),L"wb");
#else
    tmp+=".tmp."+std::to_string(getpid());FILE* f=std::fopen(tmp.c_str(),"wb");
#endif
    if(!f)fail("cannot create output temporary file");
    bool ok=std::fwrite(text.data(),1,text.size(),f)==text.size() && std::fflush(f)==0;
#ifdef _WIN32
    if(ok)ok=_commit(_fileno(f))==0;
#else
    if(ok)ok=fsync(fileno(f))==0;
#endif
    if(std::fclose(f)!=0)ok=false;if(!ok)fail("output flush failed; previous file kept");
    if(std::filesystem::exists(path)){
        auto backup=path;backup+=".previous";
        std::filesystem::copy_file(path,backup,std::filesystem::copy_options::overwrite_existing);
    }
#ifdef _WIN32
    if(!MoveFileExW(tmp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))fail("output replace failed");
#else
    if(std::rename(tmp.c_str(),path.c_str()))fail("output replace failed");
    auto parent=path.parent_path();if(parent.empty())parent=".";
    int fd=open(parent.c_str(),O_RDONLY|O_DIRECTORY);if(fd>=0){fsync(fd);close(fd);}
#endif
}
struct Queue {
    uint32_t b=0;int c=1;uint64_t p=1;
    std::vector<uint32_t> ns,masks;
    uint32_t sign_mask() const {return c==0?PLUS|MINUS:c==1?PLUS:MINUS;}
    std::pair<size_t,size_t> sign_counts() const {
        size_t plus=0,minus=0;for(auto mask:masks){plus+=bool(mask&PLUS);minus+=bool(mask&MINUS);}return {plus,minus};
    }
    size_t terms() const {auto counts=sign_counts();return counts.first+counts.second;}
};
std::string serialize(const Queue& q) {
    if(q.ns.size()!=q.masks.size())fail("internal sign-mask length mismatch");
    std::ostringstream out;out<<"ABC "<<q.b<<"^$a*$a^"<<q.b;
    if(q.c==0)out<<"$b // GHCWSV v2";
    else out<<(q.c==1?"+1":"-1")<<" // GHCWSV v1";
    out<<" sieved_to="<<q.p<<" count="<<q.terms()<<"\n";
    for(size_t i=0;i<q.ns.size();++i){
        if(!q.masks[i] || (q.masks[i]&~q.sign_mask()) || (i && q.ns[i]<=q.ns[i-1]))fail("internal invalid candidate/sign ordering");
        if(q.c==0){
            if(q.masks[i]&MINUS)out<<q.ns[i]<<" -1\n";
            if(q.masks[i]&PLUS)out<<q.ns[i]<<" +1\n";
        }else out<<q.ns[i]<<"\n";
    }
    std::string body=out.str();return body+"#SHA256 "+sha(body)+"\n";
}
Queue deserialize(std::string content) {
    if(content.size()>MAX_SNAPSHOT_BYTES)fail("input exceeds 320MiB limit");
    // Permit Windows text editors to change LF to CRLF without changing the
    // semantic checkpoint; all other header/body edits require a new digest.
    content.erase(std::remove(content.begin(),content.end(),'\r'),content.end());
    auto end=content.rfind("#SHA256 ");
    if(end==std::string::npos || content.size()!=end+8+64+1 || content.back()!='\n')fail("missing/truncated GHCWSV checksum footer");
    std::string body=content.substr(0,end);
    if(sha(body)!=content.substr(end+8,64))fail("GHCWSV SHA256 mismatch");
    std::istringstream input(body);std::string line;std::getline(input,line);
    static const std::regex pattern(R"(^ABC ([0-9]+)\^\$a\*\$a\^([0-9]+)(\+1|-1|\$b) // GHCWSV v([12]) sieved_to=([0-9]+) count=([0-9]+)$)");
    std::smatch m;if(!std::regex_match(line,m,pattern))fail("unsupported GHCWSV header");
    bool both=m[3]=="$b";if((both && m[4]!="2") || (!both && m[4]!="1"))fail("header sign/version mismatch");
    uint64_t base=ghcw_model::number(m[1]);if(base!=ghcw_model::number(m[2]))fail("header base mismatch");
    Queue q;ghcw_model::validate({base,2,1});q.b=uint32_t(base);q.c=both?0:m[3]=="+1"?1:-1;
    q.p=ghcw_model::number(m[5]);if(q.p<1 || q.p>PMAX_MAX)fail("invalid saved sieve boundary");
    uint64_t count=ghcw_model::number(m[6]);if(count>MAX_CANDIDATES*(both?2u:1u))fail("too many candidates");
    uint64_t rows=0,previous_n=0;int previous_sign=0;
    static const std::regex pair_pattern(R"(^([0-9]+) ([+-]1)$)");
    while(std::getline(input,line)){
        uint64_t n;int sign=q.c;
        if(both){std::smatch pair;if(!std::regex_match(line,pair,pair_pattern))fail("both-sign rows require n and +1/-1");n=ghcw_model::number(pair[1]);sign=pair[2]=="+1"?1:-1;}
        else n=ghcw_model::number(line);
        ghcw_model::validate({q.b,n,sign});
        if(n<previous_n || (n==previous_n && sign<=previous_sign))fail("candidates must be strictly ordered by n, then sign (-1 before +1)");
        uint32_t mask=sign==1?PLUS:MINUS;
        if(!q.ns.empty() && n==q.ns.back())q.masks.back()|=mask;
        else {q.ns.push_back(uint32_t(n));q.masks.push_back(mask);}
        previous_n=n;previous_sign=sign;
        if(++rows>count || q.ns.size()>MAX_CANDIDATES)fail("extra/oversized candidate data");
    }
    if(rows!=count)fail("truncated candidate list");return q;
}
uint64_t number(const std::string& text) {
    static const std::regex pat(R"(^([0-9]+)(?:[eE]([0-9]+))?$)");std::smatch m;
    if(!std::regex_match(text,m,pat))fail("invalid integer: "+text);
    uint64_t x=ghcw_model::number(m[1]);if(m[2].matched){auto e=ghcw_model::number(m[2]);
        if(e>18)fail("exponent too large");for(uint64_t j=0;j<e;++j){if(x>UINT64_MAX/10)fail("integer overflow");x*=10;}}
    return x;
}
struct Options {
    std::string input,output,factors,algorithm="auto",work_batching="adaptive";uint64_t base=0,lo=0,hi=0,pmax=0;
    int sign=0,device=0,blocks=0,threads=128,prime_threads=8;
    uint64_t batch=8192,progress=1,checkpoint=60;bool cpu=false,verify=false,sign_given=false;
    PrimeMode prime_mode=PrimeMode::Auto;
};
void help(){std::cout<<"GHCWSV 1.2 CUDA - Generalized Hyper-Cullen / Woodall Siever\n"
 <<"Generate: GHCWSV -b B -n NMIN -N NMAX --sign +1|-1|both -P PMAX -o FILE\n"
 <<"Resume:   GHCWSV -i FILE -P PMAX -o FILE\n"
 <<"Ranges inclusive; candidates are b^n*n^b+/-1; b,n in [2,2^32-1].\n"
 <<"--algorithm auto|direct|transform --prime-generator auto|primesieve|segmented|mr\n"
 <<"--prime-threads 1..16 --batch-primes N --device N --blocks N --threads N\n"
 <<"--work-batching adaptive|fixed (adaptive; rebuild after 20% fewer n values)\n"
 <<"--progress-seconds N (1) --checkpoint-seconds N (60) --verify\n"
 <<"-O FACTORS (optional) --cpu-reference (small tests, no CUDA calls)\n"
 <<"--selftest runs bounded GPU arithmetic tests.\n"
 <<"A single checksummed candidate file holds the safe completed prime boundary.\n"
 <<"Both signs share modular work; v2 rows are n -1 / n +1; v1 single-sign files remain supported.\n";}
Options parse(int argc,char** argv){Options o;
 for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&](){if(++i>=argc)fail("missing value for "+a);return std::string(argv[i]);};
    if(a=="-b"||a=="--base")o.base=number(next());
    else if(a=="-n"||a=="--nmin")o.lo=number(next());else if(a=="-N"||a=="--nmax")o.hi=number(next());
    else if(a=="--sign"){auto s=next();if(s!="+1"&&s!="-1"&&s!="1"&&s!="both"&&s!="+/-1")fail("sign must be +1, -1 or both");o.sign=(s=="both"||s=="+/-1")?0:s=="-1"?-1:1;o.sign_given=true;}
    else if(a=="-P"||a=="--pmax")o.pmax=number(next());
    else if(a=="-i")o.input=next();else if(a=="-o")o.output=next();else if(a=="-O")o.factors=next();
    else if(a=="--algorithm")o.algorithm=next();
    else if(a=="--work-batching")o.work_batching=next();
    else if(a=="--cpu-reference")o.cpu=true;else if(a=="--verify")o.verify=true;
    else if(a=="--prime-generator"){auto s=next();if(s=="auto")o.prime_mode=PrimeMode::Auto;
        else if(s=="primesieve")o.prime_mode=PrimeMode::PrimeSieve;else if(s=="segmented")o.prime_mode=PrimeMode::Segmented;
        else if(s=="mr")o.prime_mode=PrimeMode::MillerRabin;else fail("unknown prime generator");}
    else if(a=="--prime-threads"){auto n=number(next());if(n<1||n>16)fail("prime-threads must be 1..16");o.prime_threads=int(n);}
    else if(a=="--device"){auto n=number(next());if(n>63)fail("device out of range");o.device=int(n);}
    else if(a=="--blocks"){auto n=number(next());if(n<1||n>65535)fail("blocks out of range");o.blocks=int(n);}
    else if(a=="--threads"){auto n=number(next());if(n<32||n>512||n%32)fail("threads must be a multiple of32 in32..512");o.threads=int(n);}
    else if(a=="--batch-primes"){o.batch=number(next());if(o.batch<1||o.batch>1048576)fail("batch-primes out of range");}
    else if(a=="--progress-seconds")o.progress=number(next());else if(a=="--checkpoint-seconds")o.checkpoint=number(next());
    else fail("unknown option: "+a);
 }
 if(o.output.empty() || o.pmax<2 || o.pmax>PMAX_MAX || !o.progress || !o.checkpoint)fail("output, pmax and positive report/checkpoint intervals required");
 if(o.algorithm!="auto" && o.algorithm!="direct" && o.algorithm!="transform")fail("algorithm must be auto, direct or transform");
 if(o.work_batching!="adaptive" && o.work_batching!="fixed")fail("work-batching must be adaptive or fixed");return o;
}
Queue load(const Options& o){Queue q;
 if(!o.input.empty()){
    auto path=std::filesystem::u8path(o.input);if(std::filesystem::file_size(path)>MAX_SNAPSHOT_BYTES)fail("input too large");
    std::ifstream f(path,std::ios::binary);std::string text((std::istreambuf_iterator<char>(f)),{});if(f.bad())fail("read error");q=deserialize(text);
    if((o.base&&o.base!=q.b)||(o.sign_given&&o.sign!=q.c)||o.lo||o.hi)fail("resume header mismatch; n-range overrides not allowed");
 }else{
    if(!o.sign_given)fail("new ranges require --sign +1, -1 or both");
    ghcw_model::validate({o.base,o.lo,1});ghcw_model::validate({o.base,o.hi,1});
    if(o.hi<o.lo || o.hi-o.lo+1>MAX_CANDIDATES)fail("invalid/oversized n range (limit10M slots)");
    if(std::filesystem::exists(std::filesystem::u8path(o.output)))fail("output already exists; resume with -i or choose another file");
    q.b=uint32_t(o.base);q.c=o.sign;
    for(uint64_t n=o.lo;n<=o.hi;++n){
        uint32_t mask=0;
        if((q.sign_mask()&PLUS) && ghcw_model::composite_reason({q.b,n,1}).empty())mask|=PLUS;
        if((q.sign_mask()&MINUS) && ghcw_model::composite_reason({q.b,n,-1}).empty())mask|=MINUS;
        if(mask){q.ns.push_back(uint32_t(n));q.masks.push_back(mask);}
    }
 }
 if(o.pmax<q.p)fail("pmax is below saved sieve boundary");return q;
}
template<class T>struct Device {T* p=nullptr;~Device(){if(p)cudaFree(p);}void alloc(size_t n){CUDA_CHECK(cudaMalloc(&p,n*sizeof(T)));}};
int run(int argc,char** argv){
 if(argc==1){help();return 0;}if(argc==2 && std::string(argv[1])=="--selftest")return selftest();
 if(argc==2 && std::string(argv[1])=="--version"){std::cout<<"GHCWSV 1.2\n";return 0;}
 for(int i=1;i<argc;++i)if(std::string(argv[i])=="--help" || std::string(argv[i])=="-h"){help();return 0;}
 auto o=parse(argc,argv);auto q=load(o);auto output=std::filesystem::u8path(o.output);
 if(!o.factors.empty()) {
    auto f=std::filesystem::absolute(std::filesystem::u8path(o.factors)).lexically_normal();
    for(const auto& name:{o.input,o.output}) if(!name.empty()) {
        auto p=std::filesystem::absolute(std::filesystem::u8path(name)).lexically_normal();
        if(f==p || (std::filesystem::exists(f)&&std::filesystem::exists(p)&&std::filesystem::equivalent(f,p)))
            fail("factor output must differ from candidate input/output");
    }
 }
 const uint64_t start_p=q.p;const size_t initial=q.terms(),initial_ns=q.ns.size();
 std::ofstream factors;if(!o.factors.empty()){factors.open(std::filesystem::u8path(o.factors),std::ios::app);if(!factors)fail("cannot open factors output");}
 auto save=[&](){atomic_text(output,serialize(q));auto counts=q.sign_counts();std::cout<<"checkpoint: sieved_to="<<q.p<<", survivors="<<counts.first+counts.second<<", plus="<<counts.first<<", minus="<<counts.second<<", file="<<o.output<<"\n";};
 save();if(q.ns.empty()||q.p==o.pmax)return 0;
 std::cout<<"GHCWSV 1.2: b="<<q.b<<", sign="<<(q.c==0?"both":q.c==1?"+1":"-1")<<", candidates="<<initial<<", n_values="<<q.ns.size()<<", p in ("<<q.p<<","<<o.pmax<<"]\n";
 const auto started=std::chrono::steady_clock::now();auto last_progress=started,last_save=started;
  uint64_t prime_count=0,gpu_batches=0;double gpu_seconds=0,wait_seconds=0,accept_seconds=0;
  struct RateSample {double seconds;uint64_t primes,p;};
  std::deque<RateSample> rate_samples{{0,0,start_p}};
 auto report=[&](bool force){auto now=std::chrono::steady_clock::now();double elapsed=std::chrono::duration<double>(now-started).count();
    if(force||std::chrono::duration<double>(now-last_progress).count()>=o.progress){
        while(rate_samples.size()>1 && elapsed-rate_samples[1].seconds>=30)rate_samples.pop_front();
        const auto recent=rate_samples.front();const double span=elapsed-recent.seconds;
        const double recent_rate=span>0?double(prime_count-recent.primes)/span:0;
        const double p_rate=span>0?double(q.p-recent.p)/span:0;
        rate_samples.push_back({elapsed,prime_count,q.p});
       auto counts=q.sign_counts();size_t survivors=counts.first+counts.second;
       std::cout<<"progress: p="<<q.p<<", survivors="<<survivors<<", plus="<<counts.first<<", minus="<<counts.second<<", removed="<<initial-survivors
       <<", primes_per_s="<<(elapsed?prime_count/elapsed:0)
       <<", recent_primes_per_s="<<recent_rate<<", p_per_s="<<p_rate<<", rate_window_s="<<span<<", elapsed_s="<<elapsed
       <<", eta_s=";
       if(q.ns.empty()||q.p==o.pmax)std::cout<<0;
       else if(p_rate>0)std::cout<<double(o.pmax-q.p)/p_rate;
       else std::cout<<"n/a";
       std::cout<<"\n";last_progress=now;}
    if(std::chrono::duration<double>(now-last_save).count()>=o.checkpoint){save();last_save=now;}
 };
 auto accept=[&](const std::vector<uint32_t>& alive,const std::vector<uint64_t>& fs){
    if(alive.size()!=q.ns.size()||fs.size()!=2*q.ns.size())fail("batch result length mismatch");
    std::vector<uint32_t> kept,masks;kept.reserve(q.ns.size());masks.reserve(q.ns.size());
    for(size_t i=0;i<q.ns.size();++i){
        if(alive[i]&~q.masks[i])fail("batch resurrected an excluded sign");
        for(unsigned slot=0;slot<2;++slot){uint32_t bit=1u<<slot;if(!(q.masks[i]&bit)||(alive[i]&bit))continue;
            int c=slot==0?1:-1;uint64_t p=fs[2*i+slot];
            if(p<2 || !is_prime_mr(p) || (mul_mod_host(pow_mod_host(q.b,q.ns[i],p),pow_mod_host(q.ns[i],q.b,p),p)+(c==1?1:p-1))%p!=0 || term_equals_prime(q.b,q.ns[i],c,p))
                fail("factor verification failed; refusing to save this batch");
            if(factors.is_open())factors<<p<<" | "<<ghcw_model::Expression{q.b,q.ns[i],c}.text()<<"\n";
        }
        if(alive[i]){kept.push_back(q.ns[i]);masks.push_back(alive[i]);}
    }q.ns.swap(kept);q.masks.swap(masks);if(factors.is_open()){factors.flush();if(!factors)fail("factor output write failed");}
 };
 if(o.cpu){
    if(o.pmax-q.p>1000000 || (o.pmax-q.p)*q.ns.size()>5000000)fail("CPU reference restricted to small validation ranges");
    for(uint64_t p=q.p+1;p<=o.pmax&&!g_interrupted;++p){if(!is_prime_mr(p))continue;
       std::vector<uint32_t> alive=q.masks;std::vector<uint64_t> fs(2*q.ns.size());
       for(size_t i=0;i<q.ns.size();++i){
           auto product=mul_mod_host(pow_mod_host(q.b,q.ns[i],p),pow_mod_host(q.ns[i],q.b,p),p);
           for(unsigned slot=0;slot<2;++slot){uint32_t bit=1u<<slot;int c=slot==0?1:-1;
               if((alive[i]&bit) && (product+(c==1?1:p-1))%p==0 && !term_equals_prime(q.b,q.ns[i],c,p)){alive[i]&=~bit;fs[2*i+slot]=p;}
           }
       }
       accept(alive,fs);q.p=p;++prime_count;report(false);
    }
 }else{
    CUDA_CHECK(cudaSetDevice(o.device));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,o.device));
    std::cout<<"CUDA device: "<<prop.name<<", algorithm="<<o.algorithm<<", fallback=direct, prime_threads="<<o.prime_threads<<"\n";
    constexpr size_t term_budget=ghcw_work::Batching::term_budget;
    ghcw_work::Batching sizing(o.batch,initial_ns,o.work_batching=="adaptive");
    const uint64_t producer_batch=sizing.producer_batch();
    std::cout<<"work-batching: primes="<<sizing.selected()<<", mode="<<o.work_batching
             <<", producer_primes="<<producer_batch<<", candidate_tile=256, pairs_per_kernel<="<<term_budget<<"\n";
    Device<uint64_t> dp,df,dc;Device<uint32_t> dn,da;dp.alloc(producer_batch);dn.alloc(initial_ns);da.alloc(initial_ns);df.alloc(2*initial_ns);dc.alloc(initial_ns);
    PrimeStream stream(q.p,o.pmax,producer_batch,o.prime_mode,100000000,8,o.prime_threads,12,4,1,false);
    PrimeBatchPipeline pipeline(stream,4);std::cout<<"Prime generator: "<<stream.description()<<"\n";
    PrimeBatch produced;
    while(!g_interrupted&&!q.ns.empty()&&pipeline.next(produced,wait_seconds)){
      size_t cursor=0;
      while(cursor<produced.count&&!g_interrupted&&!q.ns.empty()){
       if(sizing.refresh(q.ns.size()))
           std::cout<<"workset-rebuild: p="<<q.p<<", n_values="<<q.ns.size()
                    <<", gpu_primes="<<sizing.selected()<<", producer_primes="<<producer_batch<<"\n";
       PrimeBatch batch=produced;
       batch.data=produced.data+cursor;
       batch.count=ghcw_work::next_chunk(produced.count,cursor,sizing.selected());
       auto begin=std::chrono::steady_clock::now();size_t count=q.ns.size();
       CUDA_CHECK(cudaMemcpy(dn.p,q.ns.data(),count*4,cudaMemcpyHostToDevice));
       std::vector<uint32_t> alive=q.masks;std::vector<uint64_t> fs(2*count);
       CUDA_CHECK(cudaMemcpy(da.p,alive.data(),count*4,cudaMemcpyHostToDevice));
       CUDA_CHECK(cudaMemset(df.p,0,count*16));
       size_t root_count=0;std::vector<uint64_t> ordered,coefficients;
       // When every n^b fits below this batch's smallest prime, coefficients
       // can be prepared once on the CPU and compared directly without a pow.
       if(o.algorithm=="auto" && power_capped(q.ns.back(),q.b,batch.first()-1)<batch.first()) {
           coefficients.reserve(count);for(auto n:q.ns)coefficients.push_back(power_capped(n,q.b,batch.first()-1));
           CUDA_CHECK(cudaMemcpy(dc.p,coefficients.data(),count*8,cudaMemcpyHostToDevice));
       }
       if(coefficients.empty() && (o.algorithm=="transform" || (o.algorithm=="auto" && q.b>64)) && (q.b&1)) {
           ordered.assign(batch.data,batch.data+batch.count);
           auto cut=std::partition(ordered.begin(),ordered.end(),[&](uint64_t p){return p>2 && q.b%p && std::gcd(uint64_t(q.b),p-1)==1;});
           root_count=size_t(cut-ordered.begin());
       }
       CUDA_CHECK(cudaMemcpy(dp.p,ordered.empty()?batch.data:ordered.data(),batch.count*8,cudaMemcpyHostToDevice));
       const size_t per_launch=std::max(size_t(1),term_budget/batch.count);
       for(size_t offset=0;offset<count;offset+=per_launch){
           size_t take=std::min(per_launch,count-offset);
           auto launch=[&](size_t pi,size_t np,bool direct){if(!np)return;
               size_t jobs=np*((take+255)/256);
               int blocks=o.blocks?o.blocks:std::max(1,std::min(int((jobs+o.threads-1)/o.threads),prop.multiProcessorCount*4));
               sieve_kernel<<<blocks,o.threads>>>(dp.p+pi,np,dn.p+offset,take,q.b,q.sign_mask(),da.p+offset,df.p+2*offset,direct,coefficients.empty()?nullptr:dc.p+offset);
               CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
           };
           // Keep each warp on one arithmetic path. The original sorted batch
           // boundary is committed only after BOTH groups and all tiles finish.
           launch(0,root_count,false);launch(root_count,batch.count-root_count,true);
       }
       CUDA_CHECK(cudaMemcpy(alive.data(),da.p,count*4,cudaMemcpyDeviceToHost));
       CUDA_CHECK(cudaMemcpy(fs.data(),df.p,count*16,cudaMemcpyDeviceToHost));
       gpu_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
       const auto accept_start=std::chrono::steady_clock::now();
       accept(alive,fs);accept_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-accept_start).count();
       // Commit only this fully completed chunk. The producer's remaining tail
       // stays owned by 'produced', including across workset resize/checkpoint.
       q.p=batch.last();prime_count+=batch.count;++gpu_batches;cursor+=batch.count;report(false);
      }
    }
 }
 if(!g_interrupted)q.p=o.pmax;
 save();report(true);
 auto counts=q.sign_counts();std::cout<<"done: primes="<<prime_count<<", survivors="<<counts.first+counts.second<<", plus="<<counts.first<<", minus="<<counts.second<<", producer_wait_s="<<wait_seconds
 <<", gpu_batch_wall_s="<<gpu_seconds<<", host_accept_s="<<accept_seconds<<", gpu_batches="<<gpu_batches
 <<", result="<<(g_interrupted?"INTERRUPTED":"COMPLETE")<<"\n";
 return g_interrupted?130:0;
}
}

void display_banner() {
    printf("%s\n","\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90");
    printf("%s\n","        .oooooo.        ooooo   ooooo       .oooooo.     oooooo   oooooo     oooo    .oooooo..o   oooooo     oooo        ");
    printf("%s\n","       d8P'  `Y8b       `888'   `888'      d8P'  `Y8b     `888.    `888.     .8'    d8P'    `Y8    `888.     .8'         ");
    printf("%s\n","      888                888     888      888              `888.   .8888.   .8'     Y88bo.          `888.   .8'          ");
    printf("%s\n","      888                888ooooo888      888               `888  .8'`888. .8'       `'Y8888o.       `888. .8'           ");
    printf("%s\n","      888     ooooo      888     888      888                `888.8'  `888.8'            `'Y88b       `888.8'            ");
    printf("%s\n","      `88.    .88'  .o.  888     888  .o. `88b    ooo  .o.    `888'    `888'    .o. oo     .d8P .o.    `888'    .o.      ");
    printf("%s\n","       `Y8bood8P'   Y8P o888o   o888o Y8P  `Y8bood8P'  Y8P     `8'      `8'     Y8P 8''88888P'  Y8P     `8'     Y8P      ");
    printf("%s\n","\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90");
    printf("%s\n","                                        Generalized Hyper Cullen / Woodall Siever                                        ");
    printf("%s\n","                                            Version 1.2 CUDA by A.P. Sep 2026                                            ");
}

int main(int argc,char** argv){prp_console::initialize_utf8_output();std::cout.setf(std::ios::unitbuf);
 display_banner();
 std::signal(SIGINT,ghcw_sieve::handle_interrupt);std::signal(SIGTERM,ghcw_sieve::handle_interrupt);
#ifdef _WIN32
 SetConsoleCtrlHandler(ghcw_sieve::console_handler,TRUE);
#endif
 try{
#ifdef _WIN32
    int count=0;LPWSTR* wide=CommandLineToArgvW(GetCommandLineW(),&count);
    if(!wide)throw std::runtime_error("cannot read Unicode arguments");
    std::vector<std::string> utf8;
    try {for(int i=0;i<count;++i){
        int length=WideCharToMultiByte(CP_UTF8,0,wide[i],-1,nullptr,0,nullptr,nullptr);
        if(length<1)throw std::runtime_error("invalid Unicode argument");
        std::string arg(size_t(length),'\0');
        if(!WideCharToMultiByte(CP_UTF8,0,wide[i],-1,arg.data(),length,nullptr,nullptr))throw std::runtime_error("Unicode conversion failed");
        arg.resize(size_t(length)-1);utf8.push_back(std::move(arg));
    }}catch(...){LocalFree(wide);throw;}LocalFree(wide);
    std::vector<char*> args;for(auto& arg:utf8)args.push_back(arg.data());
    return ghcw_sieve::run(count,args.data());
#else
    return ghcw_sieve::run(argc,argv);
#endif
 }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
