// SPDX-License-Identifier: GPL-2.0-or-later
// GFPPS, Copyright (C) 2026 AstralPrisma (A.P.).
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "ntt_backend.cuh"
#include "sha256.hpp"
#include <cub/cub.cuh>
#include <boost/multiprecision/cpp_int.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#pragma comment(lib, "Shell32.lib")
#else
#include <unistd.h>
#include <fcntl.h>
#endif

namespace gfpps {
using boost::multiprecision::cpp_int;
using gfpps_ntt::cuda_check;
constexpr char kVersion[]="1.0";
constexpr uint32_t BITS=15, BASE=1u<<BITS, MASK=BASE-1;
constexpr uint32_t P0=998244353, P1=1004535809;
constexpr uint32_t MAX_LIMBS=1u<<20;
std::atomic<bool> stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free, "signal flag must be lock-free");
void signal_handler(int) { stop_requested.store(true, std::memory_order_relaxed); }
#ifdef _WIN32
BOOL WINAPI console_handler(DWORD kind) {
    if(kind==CTRL_C_EVENT || kind==CTRL_BREAK_EVENT) { signal_handler(0); return TRUE; }
    return FALSE;
}
#endif

template<class T> struct Device {
    T* p=nullptr;
    Device()=default;
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
    ~Device(){ if(p) cudaFree(p); }
    void allocate(size_t n) { if(p) throw std::runtime_error("duplicate allocation"); cuda_check(cudaMalloc(&p,n*sizeof(T)),"allocate"); }
};

struct CarryCompose {
    __host__ __device__ uint32_t operator()(uint32_t a,uint32_t b) const {
        // b after a: g_b | (p_b & g_a), p_b & p_a.
        return (b&1u) | (((b>>1)&(a&1u))) | ((a&b)&2u);
    }
};
__global__ void encode(const uint32_t* digits,uint32_t* r0,uint32_t* r1,
                       int count,int len,uint32_t r20,uint32_t r21) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<len) { uint32_t x=i<count?digits[i]:0;
        r0[i]=gfpps_ntt::mul_mod_mont_2prime(x,r20,P0);
        r1[i]=gfpps_ntt::mul_mod_mont_2prime(x,r21,P1); }
}
__global__ void crt_coeff(const uint32_t* r0,const uint32_t* r1,uint64_t* c,
                          int len,int count,uint32_t scale,const uint32_t* add,int add_count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) c[i]=(i<len?gfpps_ntt::crt2_reconstruct_shoup(r0[i],r1[i])*scale:0)
                        +(add && i<add_count?add[i]:0);
}
__global__ void relax(const uint64_t* in,uint64_t* out,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) out[i]=(in[i]&MASK)+(i?in[i-1]>>BITS:0);
}
__global__ void carry_maps(const uint64_t* c,uint32_t* maps,int count,uint32_t* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {
        if(c[i]>=2*uint64_t(BASE)) atomicOr(error,1u);
        maps[i]=uint32_t(c[i]>=BASE)|(uint32_t(c[i]==MASK)<<1);
    }
}
__global__ void finish_carry(const uint64_t* c,const uint32_t* prefix,uint32_t* out,
                            int count,bool allow_discard,uint32_t* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) out[i]=uint32_t(c[i]+(i?(prefix[i-1]&1u):0))&MASK;
    if(i==count-1 && !allow_discard && (prefix[i]&1u)) atomicOr(error,2u);
}
__global__ void prepare_subtract(const uint32_t* sum,const uint32_t* modulus,
                                uint32_t* maps,int m,uint32_t* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m && sum[i]) atomicOr(error,4u); // REDC numerator must be divisible by R.
    if(i<=m) {
        int64_t diff=int64_t(sum[m+i])-modulus[i];
        maps[i]=uint32_t(diff<0)|(uint32_t(diff==0)<<1);
    }
}
__global__ void finish_subtract(const uint32_t* sum,const uint32_t* modulus,
                               const uint32_t* prefix,uint32_t* out,int m,uint32_t* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<=m) {
        uint32_t d=sum[m+i];
        if(!(prefix[m]&1u)) {
            int64_t v=int64_t(d)-modulus[i]-(i?(prefix[i-1]&1u):0);
            d=uint32_t(v<0?v+BASE:v);
        }
        if(i<m) out[i]=d;
        else if(d) atomicOr(error,8u);
    }
}

std::vector<uint32_t> limbs(const cpp_int& x,size_t size) {
    if(x<0) throw std::runtime_error("negative radix export");
    std::vector<uint32_t> v;
    export_bits(x,std::back_inserter(v),BITS,false);
    while(!v.empty() && !v.back()) v.pop_back();
    if(v.size()>size) throw std::runtime_error("radix export overflow");
    v.resize(size,0); return v;
}
cpp_int from_limbs(const std::vector<uint32_t>& v) {
    cpp_int x; import_bits(x,v.begin(),v.end(),BITS,false); return x;
}
cpp_int inverse_power2(const cpp_int& n,unsigned bits) {
    cpp_int inv=1;
    for(unsigned have=1;have<bits;) {
        if(stop_requested.load(std::memory_order_relaxed)) throw std::runtime_error("interrupted during Montgomery setup");
        unsigned next=std::min(2*have,bits);
        cpp_int power=cpp_int(1)<<next, mask=power-1;
        cpp_int product=((n&mask)*inv)&mask;
        inv=(inv*((power+2-product)&mask))&mask;
        have=next;
    }
    return inv;
}

class Montgomery {
    int m_,len_,log_;
    cpp_int n_,radix_power_;
    cudaStream_t stream_=nullptr;
    cudaGraph_t graphs_[2]{nullptr,nullptr};
    cudaGraphExec_t exec_[2]{nullptr,nullptr};
    uint32_t witness_,r20_,r21_,inv0_,inv1_;
    std::vector<int> offsets_;
    Device<uint32_t> work0_,work1_,nf0_,nf1_,if0_,if1_,fw0_,fw1_,iw0_,iw1_;
    Device<uint32_t> state_,mod_,t_,q_,sum_,maps_,prefix_,error_;
    Device<uint64_t> ca_,cb_;
    Device<uint8_t> scan_temp_;
    size_t scan_bytes_=0;
    int blocks(int count) const { return (count+255)/256; }
    void spectrum(const uint32_t* digits,uint32_t* a,uint32_t* b) {
        encode<<<blocks(len_),256,0,stream_>>>(digits,a,b,m_,len_,r20_,r21_);
        gfpps_ntt::ntt2_forward_dif_mont(a,b,log_,fw0_.p,fw1_.p,offsets_,stream_);
    }
    void convolution(const uint32_t* digits,const uint32_t* c0=nullptr,const uint32_t* c1=nullptr) {
        spectrum(digits,work0_.p,work1_.p);
        gfpps_ntt::ntt2_inverse_product_dit_mont(work0_.p,work1_.p,log_,iw0_.p,iw1_.p,offsets_,inv0_,inv1_,c0,c1,stream_);
    }
    void scan(int count) {
        cuda_check(cub::DeviceScan::InclusiveScan(scan_temp_.p,scan_bytes_,maps_.p,prefix_.p,CarryCompose{},count,stream_),"carry scan");
    }
    void normalize(uint32_t* out,int count,bool discard=false) {
        // M0<2^58 and B=2^15 imply M3<2B; finish the arbitrary-length
        // carry chain with a prefix scan, not a finite dependency halo.
        relax<<<blocks(count),256,0,stream_>>>(ca_.p,cb_.p,count);
        relax<<<blocks(count),256,0,stream_>>>(cb_.p,ca_.p,count);
        relax<<<blocks(count),256,0,stream_>>>(ca_.p,cb_.p,count);
        carry_maps<<<blocks(count),256,0,stream_>>>(cb_.p,maps_.p,count,error_.p);
        scan(count);
        finish_carry<<<blocks(count),256,0,stream_>>>(cb_.p,prefix_.p,out,count,discard,error_.p);
    }
    void reduce() {
        convolution(t_.p,if0_.p,if1_.p);
        crt_coeff<<<blocks(m_),256,0,stream_>>>(work0_.p,work1_.p,ca_.p,len_,m_,1,nullptr,0);
        normalize(q_.p,m_,true); // q=(T mod R)*(-N^-1) mod R.
        convolution(q_.p,nf0_.p,nf1_.p);
        const int count=2*m_+1;
        crt_coeff<<<blocks(count),256,0,stream_>>>(work0_.p,work1_.p,ca_.p,len_,count,1,t_.p,2*m_);
        normalize(sum_.p,count);
        prepare_subtract<<<blocks(m_+1),256,0,stream_>>>(sum_.p,mod_.p,maps_.p,m_,error_.p);
        scan(m_+1);
        finish_subtract<<<blocks(m_+1),256,0,stream_>>>(sum_.p,mod_.p,prefix_.p,state_.p,m_,error_.p);
    }
    void iteration(uint32_t scale) {
        convolution(state_.p);
        crt_coeff<<<blocks(2*m_),256,0,stream_>>>(work0_.p,work1_.p,ca_.p,len_,2*m_,scale,nullptr,0);
        normalize(t_.p,2*m_);
        reduce();
    }
    void destroy_graphs() noexcept {
        for(int i=0;i<2;++i) {
            if(exec_[i]) cudaGraphExecDestroy(exec_[i]); exec_[i]=nullptr;
            if(graphs_[i]) cudaGraphDestroy(graphs_[i]); graphs_[i]=nullptr;
        }
    }
public:
    Montgomery(const cpp_int& modulus,uint32_t witness,bool use_graphs,int block_cap)
        : n_(modulus),witness_(witness) {
        unsigned needed=boost::multiprecision::msb(n_*witness_)+1;
        m_=int((needed+BITS-1)/BITS);
        if(m_<1 || m_>int(MAX_LIMBS)) throw std::runtime_error("modulus exceeds supported limb/NTT limit");
        radix_power_=cpp_int(1)<<(BITS*m_);
        if(radix_power_<witness_*n_) throw std::runtime_error("Montgomery slack check failed");
        cpp_int bound=cpp_int(witness_)*m_*MASK*MASK+BASE;
        if(bound>=(cpp_int(1)<<58) || bound>=cpp_int(P0)*P1) throw std::runtime_error("CRT/carry coefficient bound exceeded");
        log_=1; while((1<<log_)<2*m_) ++log_; len_=1<<log_;
        if(log_>21) throw std::runtime_error("NTT length exceeds 2^21");
        gfpps_ntt::block_cap=block_cap;
        offsets_=gfpps_ntt::make_twiddle_offsets(log_);
        r20_=gfpps_ntt::pow_mod_host(2,64,P0); r21_=gfpps_ntt::pow_mod_host(2,64,P1);
        inv0_=gfpps_ntt::pow_mod_host(len_,P0-2,P0); inv1_=gfpps_ntt::pow_mod_host(len_,P1-2,P1);
        size_t free_bytes=0,total_bytes=0;
        cuda_check(cudaMemGetInfo(&free_bytes,&total_bytes),"query GPU memory");
        if(uint64_t(len_)*96+32*1024*1024 > free_bytes) throw std::runtime_error("insufficient free GPU memory");
        cuda_check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"create stream");
        try {
            for(auto p:{&work0_,&work1_,&nf0_,&nf1_,&if0_,&if1_,&fw0_,&fw1_,&iw0_,&iw1_}) p->allocate(len_);
            state_.allocate(m_); mod_.allocate(m_+1); t_.allocate(2*m_); q_.allocate(m_);
            sum_.allocate(2*m_+1); maps_.allocate(2*m_+1); prefix_.allocate(2*m_+1);
            ca_.allocate(2*m_+1); cb_.allocate(2*m_+1); error_.allocate(1);
            cuda_check(cudaMemsetAsync(error_.p,0,sizeof(uint32_t),stream_),"clear status");
            for(int count:{m_,m_+1,2*m_,2*m_+1}) {
                size_t bytes=0;
                cuda_check(cub::DeviceScan::InclusiveScan(nullptr,bytes,maps_.p,prefix_.p,CarryCompose{},count,stream_),"scan workspace query");
                scan_bytes_=std::max(scan_bytes_,bytes);
            }
            scan_temp_.allocate(std::max(size_t(1),scan_bytes_));
            for(int j=0;j<2;++j) for(int inverse=0;inverse<2;++inverse) {
                uint32_t p=j?P1:P0;
                auto table=gfpps_ntt::make_twiddle_table_host(log_,p,3,inverse!=0);
                uint32_t rmod=uint32_t((uint64_t(1)<<32)%p);
                for(auto& x:table) x=uint32_t(uint64_t(x)*rmod%p);
                uint32_t* dest=inverse?(j?iw1_.p:iw0_.p):(j?fw1_.p:fw0_.p);
                cuda_check(cudaMemcpyAsync(dest,table.data(),table.size()*4,cudaMemcpyHostToDevice,stream_),"copy roots");
                cuda_check(cudaStreamSynchronize(stream_),"upload roots");
            }
            auto md=limbs(n_,m_+1);
            cuda_check(cudaMemcpyAsync(mod_.p,md.data(),md.size()*4,cudaMemcpyHostToDevice,stream_),"upload modulus");
            spectrum(mod_.p,nf0_.p,nf1_.p);
            cpp_int inverse=inverse_power2(n_,BITS*m_);
            cpp_int nprime=(radix_power_-inverse)&(radix_power_-1);
            if(((n_*nprime+1)&(radix_power_-1))!=0) throw std::runtime_error("Montgomery inverse check failed");
            auto ni=limbs(nprime,m_);
            cuda_check(cudaMemcpyAsync(q_.p,ni.data(),ni.size()*4,cudaMemcpyHostToDevice,stream_),"upload inverse");
            spectrum(q_.p,if0_.p,if1_.p);
            set(limbs(radix_power_%n_,m_));
            // Exercise kernels/CUB outside graph capture; reset the state after warmup.
            iteration(1); check(); set(limbs(radix_power_%n_,m_));
            if(use_graphs) for(int i=0;i<2;++i) {
                cuda_check(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal),"begin capture");
                iteration(i?witness_:1);
                cuda_check(cudaStreamEndCapture(stream_,&graphs_[i]),"end capture");
                cuda_check(cudaGraphInstantiate(&exec_[i],graphs_[i],nullptr,nullptr,0),"instantiate graph");
            }
        } catch(...) { destroy_graphs(); if(stream_) cudaStreamDestroy(stream_); stream_=nullptr; throw; }
        std::cout<<"montgomery: radix_bits="<<BITS<<", limbs="<<m_<<", ntt_length="<<len_
                 <<", ntt_primes=2, products_per_bit=3, graphs="<<(use_graphs?"yes":"no")<<"\n";
    }
    ~Montgomery() { destroy_graphs(); if(stream_) cudaStreamDestroy(stream_); }
    int size() const { return m_; }
    const cpp_int& r() const { return radix_power_; }
    void set(const std::vector<uint32_t>& v) {
        if(v.size()!=size_t(m_)) throw std::runtime_error("state limb count mismatch");
        cuda_check(cudaMemcpyAsync(state_.p,v.data(),m_*4,cudaMemcpyHostToDevice,stream_),"upload state");
        cuda_check(cudaStreamSynchronize(stream_),"upload state sync");
    }
    void step(bool multiply) {
        if(exec_[multiply?1:0]) cuda_check(cudaGraphLaunch(exec_[multiply?1:0],stream_),"launch step");
        else iteration(multiply?witness_:1);
    }
    void check() {
        cuda_check(cudaGetLastError(),"last launch");
        uint32_t error=0;
        cuda_check(cudaMemcpyAsync(&error,error_.p,4,cudaMemcpyDeviceToHost,stream_),"read arithmetic status");
        cuda_check(cudaStreamSynchronize(stream_),"synchronize");
        if(error) throw std::runtime_error("GPU arithmetic invariant failed, flags="+std::to_string(error));
    }
    std::vector<uint32_t> get() {
        check(); std::vector<uint32_t> v(m_);
        cuda_check(cudaMemcpy(v.data(),state_.p,m_*4,cudaMemcpyDeviceToHost),"read state"); return v;
    }
    cpp_int decode() {
        cuda_check(cudaMemsetAsync(t_.p,0,2*m_*4,stream_),"decode clear");
        cuda_check(cudaMemcpyAsync(t_.p,state_.p,m_*4,cudaMemcpyDeviceToDevice,stream_),"decode input");
        reduce(); return from_limbs(get());
    }
};

uint64_t number(const std::string& s) {
    if(s.empty() || s[0]=='-' || s[0]=='+') throw std::runtime_error("invalid unsigned number: "+s);
    int base=(s.size()>2 && s[0]=='0' && (s[1]=='x'||s[1]=='X'))?16:10;
    size_t start=base==16?2:0;
    if(start==s.size()) throw std::runtime_error("empty numeric literal");
    for(size_t i=start;i<s.size();++i) if(!(base==16?std::isxdigit((unsigned char)s[i]):std::isdigit((unsigned char)s[i])))
        throw std::runtime_error("invalid numeric literal: "+s);
    size_t end=0; uint64_t v=std::stoull(s,&end,base); if(end!=s.size()) throw std::runtime_error("invalid numeric tail"); return v;
}
struct Expression { uint64_t k,n; char family; int c;
    std::string text() const { return std::to_string(k)+"*"+std::to_string(n)+family+(c==1?"+1":"-1"); }
};
Expression parse(const std::string& text) {
    static const std::regex pattern(R"(^\s*(?:(0[xX][0-9a-fA-F]+|[0-9]+)\s*\*\s*)?(0[xX][0-9a-fA-F]+|[0-9]+)\s*([!#])\s*([+-])\s*1\s*$)");
    std::smatch m; if(!std::regex_match(text,m,pattern)) throw std::runtime_error("expected k*n!+/-1 or k*n#+/-1");
    Expression e{m[1].matched?number(m[1]):1,number(m[2]),m[3].str()[0],m[4]=="+"?1:-1};
    if(e.k==0 || e.n>10000000) throw std::runtime_error("requires k>=1 and 0<=n<=10000000");
    return e;
}
cpp_int product_tree(const std::vector<uint32_t>& values,size_t lo,size_t hi) {
    if(stop_requested.load()) throw std::runtime_error("interrupted during modulus construction");
    if(hi-lo<32) { cpp_int p=1; for(size_t i=lo;i<hi;++i) p*=values[i]; return p; }
    size_t mid=lo+(hi-lo)/2;
    return product_tree(values,lo,mid)*product_tree(values,mid,hi);
}
cpp_int modulus(const Expression& e) {
    if(e.family=='!' && e.n>2) {
        long double estimate=std::lgamma((long double)e.n+1)/std::log(2.L);
        if(estimate>static_cast<long double>(MAX_LIMBS)*BITS+1024) throw std::runtime_error("factorial exceeds supported size limit");
    }
    std::vector<uint32_t> factors;
    if(e.family=='!') { factors.reserve(size_t(e.n)); for(uint32_t i=2;i<=e.n;++i) factors.push_back(i); }
    else {
        std::vector<bool> composite(size_t(e.n)+1,false);
        for(uint32_t i=2;i<=e.n;++i) {
            if(stop_requested.load()) throw std::runtime_error("interrupted during prime generation");
            if(!composite[i]) {
                factors.push_back(i);
                if(uint64_t(i)*i<=e.n) for(uint64_t j=uint64_t(i)*i;j<=e.n;j+=i) composite[size_t(j)]=true;
            }
        }
    }
    cpp_int n=product_tree(factors,0,factors.size())*e.k+e.c;
    if(n<2) throw std::runtime_error("candidate must be at least 2");
    if(boost::multiprecision::msb(n)+9>MAX_LIMBS*BITS) throw std::runtime_error("candidate exceeds supported bit limit");
    return n;
}

std::array<uint8_t,32> digest(const std::vector<uint8_t>& data) {
    CheckpointSha256 h; h.update(data.data(),data.size()); return h.final();
}
std::array<uint8_t,32> modulus_hash(const cpp_int& n) {
    std::vector<uint8_t> bytes; export_bits(n,std::back_inserter(bytes),8,false); return digest(bytes);
}
void append64(std::vector<uint8_t>& out,uint64_t x) { for(int j=0;j<8;++j) out.push_back(uint8_t(x>>(j*8))); }
std::vector<uint8_t> checkpoint_data(const Expression& e,uint32_t witness,const cpp_int& n,
                                    uint64_t bits,uint64_t done,const std::vector<uint32_t>& v) {
    std::vector<uint8_t> out={'G','F','P','P','S','0','0','1'};
    for(uint64_t x:{uint64_t(1),uint64_t(e.family=='!'?1:2),e.k,e.n,uint64_t(e.c+1),uint64_t(witness),uint64_t(v.size()),
                    uint64_t(boost::multiprecision::msb(n)+1),bits,done}) append64(out,x);
    auto nh=modulus_hash(n); out.insert(out.end(),nh.begin(),nh.end());
    for(uint32_t x:v) for(int j=0;j<4;++j) out.push_back(uint8_t(x>>(j*8)));
    auto hash=digest(out); out.insert(out.end(),hash.begin(),hash.end()); return out;
}
void atomic_write(const std::filesystem::path& path,const std::vector<uint8_t>& data) {
    auto temp=path;
#ifdef _WIN32
    temp+=".tmp."+std::to_string(GetCurrentProcessId());
#else
    temp+=".tmp."+std::to_string(getpid());
#endif
#ifdef _WIN32
    FILE* f=_wfopen(temp.c_str(),L"wb");
#else
    FILE* f=std::fopen(temp.c_str(),"wb");
#endif
    if(!f) throw std::runtime_error("cannot create checkpoint temporary file");
    bool ok=std::fwrite(data.data(),1,data.size(),f)==data.size() && std::fflush(f)==0;
#ifdef _WIN32
    if(ok) ok=_commit(_fileno(f))==0;
#else
    if(ok) ok=fsync(fileno(f))==0;
#endif
    if(std::fclose(f)!=0) ok=false;
    if(!ok) throw std::runtime_error("checkpoint write/flush failed; previous checkpoint kept");
#ifdef _WIN32
    if(!MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("checkpoint replace failed; previous checkpoint kept");
#else
    if(std::rename(temp.c_str(),path.c_str())!=0) throw std::runtime_error("checkpoint replace failed");
    auto parent=path.parent_path(); if(parent.empty()) parent=".";
    int fd=open(parent.c_str(),O_RDONLY|O_DIRECTORY); if(fd>=0){ fsync(fd); close(fd); }
#endif
}
std::pair<uint64_t,std::vector<uint32_t>> read_checkpoint(const std::filesystem::path& path,
        const Expression& e,uint32_t witness,const cpp_int& n,uint64_t bits,int m,const cpp_int& r) {
    uint64_t size=std::filesystem::file_size(path);
    if(size!=152+uint64_t(m)*4 || m<1 || m>int(MAX_LIMBS)) throw std::runtime_error("checkpoint length/limb mismatch");
    std::ifstream input(path,std::ios::binary); std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if(!input.read(reinterpret_cast<char*>(bytes.data()),bytes.size())) throw std::runtime_error("checkpoint read failed");
    std::array<uint8_t,32> saved{}; std::copy(bytes.end()-32,bytes.end(),saved.begin()); bytes.resize(bytes.size()-32);
    if(digest(bytes)!=saved) throw std::runtime_error("checkpoint SHA-256 mismatch");
    if(std::memcmp(bytes.data(),"GFPPS001",8)!=0) throw std::runtime_error("not a GFPPS checkpoint");
    size_t pos=8;
    auto take64=[&](){ uint64_t x=0; for(int j=0;j<8;++j) x|=uint64_t(bytes[pos++])<<(8*j); return x; };
    for(uint64_t want:{uint64_t(1),uint64_t(e.family=='!'?1:2),e.k,e.n,uint64_t(e.c+1),uint64_t(witness),uint64_t(m),
                       uint64_t(boost::multiprecision::msb(n)+1),bits})
        if(take64()!=want) throw std::runtime_error("checkpoint parameters do not match");
    uint64_t done=take64(); if(done>bits) throw std::runtime_error("checkpoint progress out of range");
    auto nh=modulus_hash(n); if(!std::equal(nh.begin(),nh.end(),bytes.begin()+pos)) throw std::runtime_error("checkpoint modulus mismatch"); pos+=32;
    std::vector<uint32_t> v(m);
    for(auto& x:v) { x=0; for(int j=0;j<4;++j) x|=uint32_t(bytes[pos++])<<(j*8); if(x>=BASE) throw std::runtime_error("checkpoint digit out of range"); }
    cpp_int value=from_limbs(v); if(value>=n || (done==0 && value!=r%n)) throw std::runtime_error("checkpoint residue not canonical/initial");
    return {done,std::move(v)};
}

void display_banner() {
    printf("%s\n","\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90");
    printf("%s\n","       .oooooo.        oooooooooooo     ooooooooo.       ooooooooo.        .oooooo..o          ");
    printf("%s\n","      d8P'  `Y8b       `888'     `8     `888   `Y88.     `888   `Y88.     d8P'    `Y8          ");
    printf("%s\n","     888                888              888   .d88'      888   .d88'     Y88bo.               ");
    printf("%s\n","     888                888oooo8         888ooo88P'       888ooo88P'       `'Y8888o.           ");
    printf("%s\n","     888     ooooo      888    '         888              888                  `'Y88b          ");
    printf("%s\n","     `88.    .88'  .o.  888         .o.  888         .o.  888         .o. oo     .d8P .o.      ");
    printf("%s\n","      `Y8bood8P'   Y8P o888o        Y8P o888o        Y8P o888o        Y8P 8''88888P'  Y8P      ");
    printf("%s\n","\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90");
    printf("%s\n","                        Generalized-Factorial / Primorial-Primes-Seeker                        ");
    printf("%s\n","                               Version 1.0 CUDA by A.P. Sep 2026                               ");
}

int run(int argc,char** argv) {
    std::string expression,checkpoint; uint32_t witness=2; uint64_t max_bits=0,every=100000,progress=100000;
    bool resume=false,verify=false,graphs=true,print_residue=false,help=false,interval_given=false; int blocks=96;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        auto next=[&]()->std::string { if(++i>=argc) throw std::runtime_error("missing value for "+arg); return argv[i]; };
        if(arg=="--check") expression=next();
        else if(arg=="--witness") { uint64_t v=number(next()); if(v<2||v>255) throw std::runtime_error("witness must be 2..255"); witness=uint32_t(v); }
        else if(arg=="--max-bits") max_bits=number(next());
        else if(arg=="--checkpoint") checkpoint=next();
        else if(arg=="--checkpoint-every-bits") { every=number(next()); interval_given=true; }
        else if(arg=="--progress-every-bits") { progress=number(next()); if(!progress) throw std::runtime_error("progress interval must be positive"); }
        else if(arg=="--resume-checkpoint") resume=true;
        else if(arg=="--verify-cpp-int") verify=true;
        else if(arg=="--no-graphs") graphs=false;
        else if(arg=="--print-residue") print_residue=true;
        else if(arg=="--force-ntt-blocks") { uint64_t v=number(next()); if(v<1||v>4096) throw std::runtime_error("blocks must be 1..4096"); blocks=int(v); }
        else if(arg=="-h" || arg=="--help") help=true;
        else throw std::runtime_error("unknown option: "+arg);
    }
    if(help || expression.empty()) {
        display_banner();
        std::cout<<"  GFPPS --check \"k*n!+/-1\" [options]\n  GFPPS --check \"k*n#+/-1\" [options]\n"
                 <<"n# = product of all primes <= n; 0! = 0# = 1. Quote the expression.\n"
                 <<"Options: --witness 2..255 (default2), --max-bits N (0=full),\n"
                 <<" --checkpoint FILE --checkpoint-every-bits N --resume-checkpoint,\n"
                 <<" --progress-every-bits N (default100000) --force-ntt-blocks N --no-graphs,\n"
                 <<" --verify-cpp-int (explicit, <=8192-bit N), --print-residue.\n"
                 <<"Fermat PRP only, not a primality proof. Independently verify important results.\n";
        return 0;
    }
    if(resume && checkpoint.empty()) throw std::runtime_error("resume requires --checkpoint");
    if(interval_given && checkpoint.empty()) throw std::runtime_error("checkpoint interval requires --checkpoint");
    if(checkpoint.empty()) std::cout<<"checkpoint: disabled; use --checkpoint FILE to enable save/resume\n";
    else std::cout<<"checkpoint: enabled, periodic_bits="<<every<<", file="<<checkpoint<<"\n";
    auto started=std::chrono::steady_clock::now(); Expression e=parse(expression);
    std::cout<<"GFPPS "<<kVersion<<": N="<<e.text()<<", witness="<<witness<<"; constructing modulus\n";
    cpp_int n=modulus(e), exponent=n-1;
    uint64_t bits=boost::multiprecision::msb(exponent)+1;
    unsigned nbits=boost::multiprecision::msb(n)+1;
    const unsigned log_shift=nbits>53?nbits-53:0;
    const double top=static_cast<double>(n>>log_shift);
    const uint64_t decimal_estimate=static_cast<uint64_t>(std::floor(std::log10(top)+log_shift*std::log10(2.0)))+1;
    std::cout<<"modulus: bits="<<nbits<<", digits_estimate="<<decimal_estimate
             <<", construction_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"\n";
    if(verify && nbits>8192) throw std::runtime_error("explicit cpp_int verification capped at 8192 bits");
    if(n==2) { std::cout<<"gfpps-check: N="<<e.text()<<", result=PRP, reason=trivial-prime-2\n"; return 0; }
    if((n&1)==0) { std::cout<<"gfpps-check: N="<<e.text()<<", result=COMPOSITE, factor=2\n"; return 0; }
    uint32_t g=std::gcd(witness,static_cast<uint32_t>(n%witness));
    if(g>1) { if(n==g) throw std::runtime_error("invalid witness: divisible by candidate");
        std::cout<<"gfpps-check: N="<<e.text()<<", result=COMPOSITE, factor="<<g<<"\n"; return 0; }
    if(print_residue && nbits>8192) throw std::runtime_error("print-residue capped at8192bits");
    uint64_t done=0;
    const auto path=std::filesystem::u8path(checkpoint);
    std::vector<uint32_t> saved_digits;
    if(resume) {
        const unsigned needed=boost::multiprecision::msb(n*witness)+1;
        const int m=int((needed+BITS-1)/BITS);
        const cpp_int r=cpp_int(1)<<(BITS*m);
        auto saved=read_checkpoint(path,e,witness,n,bits,m,r);
        done=saved.first; saved_digits=std::move(saved.second);
    } else if(!checkpoint.empty() && std::filesystem::exists(path)) {
        throw std::runtime_error("checkpoint exists; use --resume-checkpoint or another path");
    }
    Montgomery gpu(n,witness,graphs,blocks);
    if(resume) {
        gpu.set(saved_digits);
        std::cout<<"checkpoint: resumed, processed_bits="<<done<<"/"<<bits<<"\n";
    }
    auto save=[&](const char* reason) {
        if(!checkpoint.empty()) {
            auto digits=gpu.get();
            if(from_limbs(digits)>=n) throw std::runtime_error("refusing noncanonical checkpoint");
            atomic_write(path,checkpoint_data(e,witness,n,bits,done,digits));
            std::cout<<"checkpoint: saved, processed_bits="<<done<<", reason="<<reason<<"\n";
        }
    };
    if(!resume) save("initial");
    uint64_t start_done=done,limit=max_bits?std::min(bits,done+std::min(max_bits,bits-done)):bits;
    uint64_t last_saved=done,last_progress=done;
    auto exp_started=std::chrono::steady_clock::now();
    while(done<limit && !stop_requested.load(std::memory_order_relaxed)) {
        gpu.step(boost::multiprecision::bit_test(exponent,static_cast<unsigned>(bits-1-done)));
        ++done;
        const bool report_due=done-last_progress>=progress;
        const bool checkpoint_due=!checkpoint.empty() && every && done-last_saved>=every;
        if((done-start_done)%128==0 || done==limit || report_due || checkpoint_due) {
            gpu.check();
            if(checkpoint_due) { save("periodic"); last_saved=done; }
            if(report_due || done==limit) {
                double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-exp_started).count();
                double eta=done>start_done?seconds*(bits-done)/(done-start_done):0;
                std::cout<<"progress: "<<std::fixed<<std::setprecision(2)<<100.0*done/bits<<"%, bits="<<done<<"/"<<bits
                         <<", elapsed_s="<<seconds<<", eta_s="<<eta<<"\n";
                last_progress=done;
            }
        }
    }
    gpu.check(); bool interrupted=stop_requested.load();
    save(interrupted?"interrupt":done==bits?"complete":"partial");
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-exp_started).count();
    cpp_int residue=gpu.decode();
    if(verify) {
        cpp_int prefix=exponent>>(bits-done);
        cpp_int expected=boost::multiprecision::powm(cpp_int(witness),prefix,n);
        if(residue!=expected) throw std::runtime_error("independent cpp_int residue mismatch");
    }
    auto checksum0=static_cast<uint32_t>(residue%1000000007), checksum1=static_cast<uint32_t>(residue%1000000009);
    std::cout<<"gfpps-check: N="<<e.text()<<", witness="<<witness<<", processed_bits="<<done<<"/"<<bits
             <<", exponentiation_seconds="<<std::setprecision(6)<<elapsed
             <<", total_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
             <<", checksum="<<checksum0<<":"<<checksum1<<", verification="<<(verify?"cpp_int-match":"not-requested")
             <<", result="<<(interrupted?"INTERRUPTED":done<bits?"PARTIAL":residue==1?"PRP":"COMPOSITE")<<"\n";
    if(print_residue) std::cout<<"residue="<<residue<<"\n";
    return interrupted?130:0;
}
} // namespace gfpps
int main(int argc,char** argv) {
    std::cout.setf(std::ios::unitbuf); std::cerr.setf(std::ios::unitbuf);
    std::signal(SIGINT,gfpps::signal_handler); std::signal(SIGTERM,gfpps::signal_handler);
#ifdef _WIN32
    SetConsoleCtrlHandler(gfpps::console_handler,TRUE);
#endif
    try {
#ifdef _WIN32
        int count=0; LPWSTR* wide=CommandLineToArgvW(GetCommandLineW(),&count);
        if(!wide) throw std::runtime_error("cannot read Unicode command line");
        std::vector<std::string> utf8;
        try {
            for(int i=0;i<count;++i) {
                int length=WideCharToMultiByte(CP_UTF8,0,wide[i],-1,nullptr,0,nullptr,nullptr);
                if(length<1) throw std::runtime_error("invalid Unicode argument");
                std::string arg(static_cast<size_t>(length),'\0');
                if(!WideCharToMultiByte(CP_UTF8,0,wide[i],-1,arg.data(),length,nullptr,nullptr)) throw std::runtime_error("argument conversion failed");
                arg.resize(static_cast<size_t>(length)-1); utf8.push_back(std::move(arg));
            }
        } catch(...) { LocalFree(wide); throw; }
        LocalFree(wide); std::vector<char*> args;
        for(auto& arg:utf8) args.push_back(arg.data());
        return gfpps::run(count,args.data());
#else
        return gfpps::run(argc,argv);
#endif
    }
    catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<"\n"; return gfpps::stop_requested.load()?130:1; }
}
