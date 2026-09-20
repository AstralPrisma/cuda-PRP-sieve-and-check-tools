// SPDX-License-Identifier: GPL-2.0-or-later
// GHCWPS, Copyright (C) 2026 AstralPrisma (A.P.).
// The 15-bit path preserves existing checkpoints; the 18-bit path is optional.
#define ghcwps ghcwps15
#define GHCWPS_RADIX_BITS 15
#include "ghcwps_profile.cuh"
#undef GHCWPS_RADIX_BITS
#undef ghcwps
#define ghcwps ghcwps18
#define GHCWPS_RADIX_BITS 18
#include "ghcwps_profile.cuh"
#undef GHCWPS_RADIX_BITS
#undef ghcwps
#include "console_utf8.hpp"
#define ghcwps ghcwps21
#define GHCWPS_RADIX_BITS 21
#include "ghcwps_profile.cuh"
#undef GHCWPS_RADIX_BITS
#undef ghcwps
#include "ghcwps_banner.hpp"

namespace adaptive {
using boost::multiprecision::cpp_int;
void signal_handler(int sig) {
    ghcwps15::signal_handler(sig); ghcwps18::signal_handler(sig);
    ghcwps21::signal_handler(sig);
}
#ifdef _WIN32
BOOL WINAPI console_handler(DWORD kind) {
    if(kind==CTRL_C_EVENT || kind==CTRL_BREAK_EVENT) { signal_handler(0); return TRUE; }
    return FALSE;
}
#endif
bool stopped() { return ghcwps15::stop_requested.load(std::memory_order_relaxed); }
struct Plan { bool safe=false; uint64_t limbs=0,length=0; };
Plan plan(const cpp_int& n,uint32_t witness,unsigned radix) {
    const uint64_t needed=boost::multiprecision::msb(n*witness)+1;
    Plan p; p.limbs=(needed+radix-1)/radix; p.length=2;
    while(p.length<2*p.limbs)p.length*=2;
    const cpp_int base=cpp_int(1)<<radix, mask=base-1;
    const cpp_int bound=cpp_int(witness)*p.limbs*mask*mask+base;
    p.safe=p.limbs>=1 && p.limbs<=(1u<<20) && p.length<=(1u<<21)
        && bound<(cpp_int(1)<<58) && bound<cpp_int(998244353)*1004535809;
    return p;
}

// Use measured transform reductions and the measured 524288-point equal-length
// case. The 5090's 32768-point case is deliberately excluded.
int choose(const Plan& old,const Plan& fast,int architecture,uint32_t witness,bool graphs,int blocks) {
    if(!old.safe || !fast.safe || witness!=2 || !graphs || blocks!=256)return 15;
    if(architecture==89 && fast.length==524288 && old.length==524288)return 18;
    if(fast.length>=old.length || fast.length>262144)return 15;
    if(architecture==89 && fast.length>=32768)return 18;
    if(architecture==120 && fast.length>=65536)return 18;
    return 15;
}

int run(int argc,char** argv) {
    std::vector<char*> forwarded;forwarded.push_back(argv[0]);
    int requested=0,checkpoint_radix=0,blocks=256;uint32_t witness=2;
    std::string expression,checkpoint;bool resume=false,cpu=false,graphs=true,help=false;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--radix-bits") {
            if(++i>=argc)throw std::runtime_error("missing value for --radix-bits");
            const std::string value=argv[i];
            if(value=="auto")requested=0;
            else if(value=="15")requested=15;
            else if(value=="18")requested=18;
            else if(value=="21")requested=21;
            else throw std::runtime_error("radix-bits must be auto, 15, 18 or 21");
            continue;
        }
        forwarded.push_back(argv[i]);
        if(arg=="--resume-checkpoint")resume=true;
        else if(arg=="--cpu-reference")cpu=true;
        else if(arg=="--no-graphs")graphs=false;
        else if(arg=="--help" || arg=="-h" || arg=="--version")help=true;
        else if(arg=="--check" || arg=="--checkpoint" || arg=="--witness" || arg=="--force-ntt-blocks"
             || arg=="--max-bits" || arg=="--checkpoint-every-bits" || arg=="--progress-every-bits") {
            if(++i>=argc)throw std::runtime_error("missing value for "+arg);
            forwarded.push_back(argv[i]);const std::string value=argv[i];
            if(arg=="--check")expression=value;
            else if(arg=="--checkpoint")checkpoint=value;
            else if(arg=="--witness") {
                auto v=ghcw_model::number(value);if(v<2 || v>255)throw std::runtime_error("witness must be 2..255");witness=uint32_t(v);
            } else if(arg=="--force-ntt-blocks") {
                auto v=ghcw_model::number(value);if(v<1 || v>4096)throw std::runtime_error("blocks must be 1..4096");blocks=int(v);
            } else ghcw_model::number(value);
        } else if(arg!="--verify-cpp-int" && arg!="--print-residue")
            throw std::runtime_error("unknown option: "+arg);
    }
    auto dispatch=[&](int radix) {
        if(radix==21)return ghcwps21::run(int(forwarded.size()),forwarded.data());
        return radix==18?ghcwps18::run(int(forwarded.size()),forwarded.data()):ghcwps15::run(int(forwarded.size()),forwarded.data());
    };
    if(help || expression.empty())return dispatch(15);
    if(resume) {
        if(checkpoint.empty())throw std::runtime_error("resume requires --checkpoint");
        std::ifstream file(std::filesystem::u8path(checkpoint),std::ios::binary);
        char tag[8];if(!file.read(tag,8))throw std::runtime_error("cannot read checkpoint format");
        // Close before dispatch: Windows otherwise denies atomic replacement
        // while this format-probe handle remains open across the resumed run.
        file.close();
        if(std::memcmp(tag,"GHCWPS01",8)==0)checkpoint_radix=15;
        else if(std::memcmp(tag,"GHCW18A1",8)==0)checkpoint_radix=18;
        else if(std::memcmp(tag,"GHCW21A1",8)==0)checkpoint_radix=21;
        else throw std::runtime_error("unsupported checkpoint radix/format");
        if(requested && requested!=checkpoint_radix)throw std::runtime_error("requested radix conflicts with checkpoint");
        // The selected implementation verifies the complete digest and metadata
        // before creating any CUDA context or consuming saved state.
        std::cout<<"radix-selection: bits="<<checkpoint_radix<<", reason=checkpoint\n";
        return dispatch(checkpoint_radix);
    }
    if(requested) {
        std::cout<<"radix-selection: bits="<<requested<<", reason=explicit\n";
        return dispatch(requested); // Constructor independently enforces bounds.
    }
    if(cpu)return dispatch(15);
    auto e=ghcw_model::parse(expression);
    if(!ghcw_model::composite_reason(e).empty())return dispatch(15);
    cpp_int n=ghcw_model::modulus(e,stopped);
    auto old=plan(n,witness,15),fast=plan(n,witness,18),wide=plan(n,witness,21);
    int chosen=15;
    // Avoid even device discovery when a transform reduction cannot apply.
    if(old.safe && fast.safe && fast.length<=old.length && witness==2 && graphs && blocks==256) {
        int device=0;cudaDeviceProp prop{};
        gfpps_ntt::cuda_check(cudaGetDevice(&device),"select radix get device");
        gfpps_ntt::cuda_check(cudaGetDeviceProperties(&prop,device),"select radix device properties");
        chosen=choose(old,fast,prop.major*10+prop.minor,witness,graphs,blocks);
        // RTX4060 follow-up: retain existing policy at equal lengths. Admit
        // only measured21-bit transform reductions with exact CRT bounds.
        if(prop.major*10+prop.minor==89 && wide.safe && wide.length<old.length
            && wide.length<fast.length && (wide.length==32768 || wide.length==65536))chosen=21;
    }
    std::cout<<"radix-selection: bits="<<chosen<<", reason="<<(chosen!=15?"validated-profile":"baseline")<<"\n";
    n=0;return dispatch(chosen);
}
}

int main(int argc,char** argv) {
    prp_console::initialize_utf8_output();
    std::cout.setf(std::ios::unitbuf);std::cerr.setf(std::ios::unitbuf);
    std::signal(SIGINT,adaptive::signal_handler);std::signal(SIGTERM,adaptive::signal_handler);
#ifdef _WIN32
    SetConsoleCtrlHandler(adaptive::console_handler,TRUE);
#endif
    try {
#ifdef _WIN32
        int count=0;LPWSTR* wide=CommandLineToArgvW(GetCommandLineW(),&count);
        if(!wide)throw std::runtime_error("cannot read Unicode command line");
        std::vector<std::string> utf8;
        try {
            for(int i=0;i<count;++i) {
                int length=WideCharToMultiByte(CP_UTF8,0,wide[i],-1,nullptr,0,nullptr,nullptr);
                if(length<1)throw std::runtime_error("invalid Unicode argument");
                std::string arg(static_cast<size_t>(length),'\0');
                if(!WideCharToMultiByte(CP_UTF8,0,wide[i],-1,arg.data(),length,nullptr,nullptr))throw std::runtime_error("argument conversion failed");
                arg.resize(size_t(length)-1);utf8.push_back(std::move(arg));
            }
        } catch(...) {LocalFree(wide);throw;}
        LocalFree(wide);std::vector<char*> args;
        for(auto& arg:utf8)args.push_back(arg.data());
        return adaptive::run(count,args.data());
#else
        return adaptive::run(argc,argv);
#endif
    } catch(const std::exception& e) {std::cerr<<"error: "<<e.what()<<"\n";return adaptive::stopped()?130:1;}
}
