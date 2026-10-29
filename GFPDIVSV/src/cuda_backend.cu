// SPDX-License-Identifier: GPL-2.0-or-later
// Experimental GFPDIVSV GPU remainder backend. No floating point arithmetic.
#include "cuda_backend.hpp"
#include <cuda_runtime.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

namespace gfpdiv_cuda {
using u64 = uint64_t;
constexpr u64 max_modulus = (u64(1) << 62) - 1;
static void check(cudaError_t e, const char* operation) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(e));
}
template<class T> struct Buffer {
    T* p = nullptr;
    size_t capacity = 0;
    ~Buffer() { if (p) cudaFree(p); }
    void reserve(size_t count) {
        if (capacity >= count) return;
        if (p) { check(cudaFree(p), "free previous CUDA buffer"); p = nullptr; capacity = 0; }
        check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "allocate CUDA buffer");
        capacity = count;
    }
};
struct Mont { u64 p, negative_inverse, one, r2; };

// a<p, b<2^64 => a*b<p*R. REDC needs only one subtraction.
// p<2^62 also bounds hi(T)+hi(m*p)+carry below 2^63.
__device__ __forceinline__ u64 multiply(u64 a, u64 b, const Mont& c) {
    const u64 low = a*b;
    const u64 m = low*c.negative_inverse;
    u64 value = __umul64hi(a,b) + __umul64hi(m,c.p) + u64(low != 0);
    if (value >= c.p) value -= c.p;
    return value;
}
__device__ __forceinline__ Mont make_mont(u64 p) {
    u64 inverse = 1;
    #pragma unroll
    for (int i=0; i<6; ++i) inverse *= 2-p*inverse;
    const u64 one = (u64(0)-p)%p; // exactly 2^64 modulo p
    u64 r2 = one;
    #pragma unroll 1
    for (int i=0; i<64; ++i) {
        r2 += r2; // r2<p<2^62, so no word overflow
        if (r2 >= p) r2 -= p;
    }
    return {p, u64(0)-inverse, one, r2};
}
__device__ __forceinline__ u64 power_encoded(u64 x, uint32_t e, const Mont& c) {
    u64 y=c.one;
    while (e) {
        if (e&1) y=multiply(y,x,c);
        e>>=1;
        if (e) x=multiply(x,x,c);
    }
    return y;
}
__global__ void initialize(const u64* primes, size_t count, Mont* contexts, u64* totals) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<count) { const Mont c=make_mont(primes[i]); contexts[i]=c; totals[i]=c.one; }
}
template<int Chains>
__global__ void initialize_group(const Mont* contexts, size_t count, uint32_t chunks,
                                 u64* accumulators) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const Mont c=contexts[i];
    const uint32_t q=chunks/Chains, r=chunks%Chains;
    // power_encoded(R^2, q) is R^(q+1) in ordinary word notation.
    // k raw-factor REDCs therefore leave product*R, not a scaled residue.
    const u64 seed=power_encoded(c.r2,q,c);
    const u64 extra=multiply(seed,c.r2,c);
    #pragma unroll
    for (int k=0;k<Chains;++k) accumulators[size_t(k)*count+i]=k<int(r)?extra:seed;
}
template<int Chains>
__global__ void product_tile(const Mont* contexts, size_t count, const u64* chunks,
                            uint32_t begin, uint32_t end, uint32_t total,
                            u64* accumulators) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const Mont c=contexts[i];
    u64 values[Chains];
    #pragma unroll
    for (int k=0;k<Chains;++k) values[k]=accumulators[size_t(k)*count+i];
    // begin/end are whole chain groups except the final partial group.
    for (uint32_t j=begin; j<end; j+=Chains) {
        #pragma unroll
        for (int k=0;k<Chains;++k)
            if (j+k<total && j+k<end) values[k]=multiply(values[k],chunks[j+k],c);
    }
    #pragma unroll
    for (int k=0;k<Chains;++k) accumulators[size_t(k)*count+i]=values[k];
}
template<int Chains>
__global__ void finish_group(const Mont* contexts, size_t count, const u64* accumulators,
                            u64* totals) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const Mont c=contexts[i];
    u64 group=accumulators[i];
    #pragma unroll
    for (int k=1;k<Chains;++k) group=multiply(group,accumulators[size_t(k)*count+i],c);
    totals[i]=multiply(multiply(totals[i],totals[i],c),group,c);
}
__global__ void decode(const Mont* contexts, size_t count, u64* totals) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<count) totals[i]=multiply(totals[i],1,contexts[i]);
}

struct Backend::Impl {
    Buffer<u64> chunks, primes, accumulators, totals;
    Buffer<Mont> contexts;
    std::vector<std::pair<uint32_t,uint32_t>> groups;
    cudaStream_t stream=nullptr;
    cudaEvent_t started=nullptr, ended=nullptr;
    int chains;
    uint32_t tile;
    double elapsed=0;
    std::string identity;
    Impl(const Groups& input, int device, int c, uint32_t t):chains(c),tile(t) {
        if ((chains!=1&&chains!=2&&chains!=4&&chains!=8)||tile<8||tile%8)
            throw std::runtime_error("CUDA chains must be1/2/4/8; tile must be a positive multiple of8");
        check(cudaSetDevice(device),"select CUDA device");
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,device),"query CUDA device");
        int driver=0,runtime=0;check(cudaDriverGetVersion(&driver),"query CUDA driver version");check(cudaRuntimeGetVersion(&runtime),"query CUDA runtime version");
        std::ostringstream key;key<<prop.major<<'.'<<prop.minor<<':'<<prop.multiProcessorCount<<':'<<driver<<':'<<runtime<<':'<<chains<<':'<<tile<<':';
        key<<std::hex<<std::setfill('0');for(unsigned char byte:prop.uuid.bytes)key<<std::setw(2)<<unsigned(byte);identity=key.str();
        std::cout<<"CUDA device="<<prop.name<<", SMs="<<prop.multiProcessorCount
                 <<", independent_chains="<<chains<<", tile_chunks="<<tile<<"\n";
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create CUDA stream");
        check(cudaEventCreate(&started),"create CUDA event");
        check(cudaEventCreate(&ended),"create CUDA event");
        std::vector<u64> all;
        for (const auto& group:input) {
            if (all.size()+group.size()>UINT32_MAX) throw std::runtime_error("CUDA product representation too large");
            groups.push_back({uint32_t(all.size()),uint32_t(group.size())});
            all.insert(all.end(),group.begin(),group.end());
        }
        chunks.reserve(std::max<size_t>(1,all.size()));
        if (!all.empty()) check(cudaMemcpyAsync(chunks.p,all.data(),all.size()*sizeof(u64),cudaMemcpyHostToDevice,stream),"upload product chunks");
        check(cudaStreamSynchronize(stream),"commit product upload");
    }
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        if (started) cudaEventDestroy(started);
        if (ended) cudaEventDestroy(ended);
        if (stream) cudaStreamDestroy(stream);
    }
    template<int C> bool run_groups(size_t n, const std::function<bool()>& stop) {
        const unsigned blocks=unsigned((n+127)/128);
        for (auto [offset,size]:groups) {
            if (stop()) return false;
            initialize_group<C><<<blocks,128,0,stream>>>(contexts.p,n,size,accumulators.p);
            check(cudaGetLastError(),"initialize product group");
            for (uint32_t begin=0;begin<size;) {
                if (stop()) return false;
                const uint32_t end=uint32_t(std::min<uint64_t>(size,uint64_t(begin)+tile));
                product_tile<C><<<blocks,128,0,stream>>>(contexts.p,n,chunks.p+offset,begin,end,size,accumulators.p);
                check(cudaGetLastError(),"CUDA product tile");
                check(cudaStreamSynchronize(stream),"complete CUDA product tile");
                begin=end;
            }
            finish_group<C><<<blocks,128,0,stream>>>(contexts.p,n,accumulators.p,totals.p);
            check(cudaGetLastError(),"reduce product group");
        }
        return !stop();
    }
    bool calculate(const std::vector<u64>& input,std::vector<u64>& output,
                   const std::function<bool()>& stop) {
        const size_t n=input.size();
        if (!n) {output.clear();return true;}
        if (n>1048576) throw std::runtime_error("CUDA batch exceeds1048576 primes");
        for (u64 p:input) if (p<3||p>max_modulus||!(p&1)) throw std::runtime_error("CUDA requires odd modulus3..2^62-1");
        primes.reserve(n);contexts.reserve(n);accumulators.reserve(n*chains);totals.reserve(n);
        check(cudaMemcpyAsync(primes.p,input.data(),n*sizeof(u64),cudaMemcpyHostToDevice,stream),"upload sieve primes");
        check(cudaEventRecord(started,stream),"start CUDA timing");
        initialize<<<unsigned((n+127)/128),128,0,stream>>>(primes.p,n,contexts.p,totals.p);
        check(cudaGetLastError(),"initialize modular contexts");
        bool complete=false;
        switch(chains) {
            case 1:complete=run_groups<1>(n,stop);break;
            case 2:complete=run_groups<2>(n,stop);break;
            case 4:complete=run_groups<4>(n,stop);break;
            case 8:complete=run_groups<8>(n,stop);break;
        }
        if (!complete) {check(cudaStreamSynchronize(stream),"discard interrupted CUDA batch");output.clear();return false;}
        decode<<<unsigned((n+127)/128),128,0,stream>>>(contexts.p,n,totals.p);
        check(cudaGetLastError(),"decode modular residues");
        check(cudaEventRecord(ended,stream),"finish CUDA timing");
        output.resize(n);
        check(cudaMemcpyAsync(output.data(),totals.p,n*sizeof(u64),cudaMemcpyDeviceToHost,stream),"download modular residues");
        check(cudaStreamSynchronize(stream),"complete CUDA batch");
        float milliseconds=0;check(cudaEventElapsedTime(&milliseconds,started,ended),"read CUDA timing");
        elapsed+=milliseconds/1000.;
        return true;
    }
};
Backend::Backend(const Groups& groups,int device,int chains,uint32_t tile)
    :impl_(std::make_unique<Impl>(groups,device,chains,tile)) {}
Backend::~Backend()=default;
bool Backend::remainders(const std::vector<u64>& p,std::vector<u64>& r,
                         const std::function<bool()>& stop) { return impl_->calculate(p,r,stop); }
double Backend::stream_seconds()const{return impl_->elapsed;}
std::string Backend::tuning_identity()const{return impl_->identity;}
void Backend::reset_timing(){impl_->elapsed=0;}

int selftest() {
    using boost::multiprecision::uint128_t;
    u64 random=0xf3a85c1e2d640b79ULL;
    auto next=[&](){random^=random<<13;random^=random>>7;random^=random<<17;return random;};
    std::vector<u64> moduli={3,5,7,17,31,65537,4294967291ULL,4294967311ULL,
                            (u64(1)<<61)-1,max_modulus,max_modulus-2};
    for(int i=0;i<53;++i) moduli.push_back((next()&max_modulus)|3);
    size_t comparisons=0;
    for(unsigned length:{0u,1u,2u,3u,7u,8u,9u,31u,127u,1025u,6501u}) {
        Groups groups(3);
        for(unsigned i=0;i<length;++i)groups[0].push_back(i==0?UINT64_MAX:i==1?u64(1)<<63:next());
        groups[1]={UINT64_MAX,0x8000000000000000ULL,3};
        // Include an empty lower-weight group: it must square, not disappear.
        std::vector<u64> expected;
        for(u64 p:moduli) {
            u64 value=1;
            for(const auto& group:groups) {
                value=uint64_t(uint128_t(value)*value%p);
                for(u64 word:group)value=uint64_t(uint128_t(value)*word%p);
            }
            expected.push_back(value);
        }
        for(int chains:{1,2,4,8}) {
            Backend engine(groups,0,chains,256);std::vector<u64> actual;
            if(!engine.remainders(moduli,actual,[]{return false;})||actual!=expected)
                throw std::runtime_error("CUDA exact modular selftest failed at length="+std::to_string(length)+", chains="+std::to_string(chains));
            comparisons+=actual.size();
            if(engine.remainders(moduli,actual,[]{return true;}))throw std::runtime_error("CUDA cancellation selftest failed");
        }
    }
    std::cout<<"CUDA_SELFTEST_PASS comparisons="<<comparisons
             <<" (full-word operands, odd moduli through2^62-1, chains1/2/4/8, empty groups, cancellation)\n";
    return 0;
}
}
