// SPDX-License-Identifier: GPL-2.0-or-later
// Catena Siderum Siever 1.2, A.P. 2026.
#include "cssv_common.hpp"
#include "cssv_arithmetic.cuh"
#include "cssv_state.hpp"
#include "cssv_candidates.hpp"
#include "cssv_primes.hpp"
#include "console_utf8.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib,"Shell32.lib")
#endif
namespace cssv {
using Clock=std::chrono::steady_clock;
std::atomic<bool> stopped{false};
static_assert(std::atomic<bool>::is_always_lock_free,"signal flag must be lock-free");
void stop_handler(int){stopped.store(true,std::memory_order_relaxed);}
#ifdef _WIN32
BOOL WINAPI console_handler(DWORD kind){if(kind==CTRL_C_EVENT||kind==CTRL_BREAK_EVENT){stop_handler(0);return TRUE;}return FALSE;}
#endif
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double interval(const std::string& s){size_t used=0;double d=std::stod(s,&used);require(used==s.size()&&std::isfinite(d)&&d>0,"time interval must be finite and positive");return d;}
bool alias(const std::filesystem::path& a,const std::filesystem::path& b){
 auto x=std::filesystem::weakly_canonical(std::filesystem::absolute(a)),y=std::filesystem::weakly_canonical(std::filesystem::absolute(b));
 if(std::filesystem::exists(x)&&std::filesystem::exists(y)&&std::filesystem::equivalent(x,y))return true;
#ifdef _WIN32
 return CompareStringOrdinal(x.c_str(),-1,y.c_str(),-1,TRUE)==CSTR_EQUAL;
#else
 return x==y;
#endif
}
#ifdef __CUDACC__
void cuda_ok(cudaError_t e,const char* label){if(e!=cudaSuccess)throw std::runtime_error(std::string(label)+": "+cudaGetErrorString(e));}
template<class T>struct Device{T* p=nullptr;explicit Device(size_t n){cuda_ok(cudaMalloc(&p,n*sizeof(T)),"allocate");}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;};
__global__ void kernel(Config c,const Mod* primes,size_t n,const Entry* entries,size_t count,unsigned chunk,u64* factors){
 size_t k=blockIdx.x*blockDim.x+threadIdx.x,offset=size_t(blockIdx.y)*chunk;
 if(k<n&&offset<count){u64 take=count-offset;if(take>chunk)take=chunk;hd_scan(c,primes[k],entries+offset,take,factors);}
}
#endif
struct Options {
 Config cfg;std::set<std::string> explicit_fields;std::string backend="auto",generator="auto",format="candidates";bool export_only=false;
 std::filesystem::path out,state,input,stopfile;bool resume=false;double checkpoint=60,progress=1;
 unsigned batch=1024,chunk=256,device=0;
};
void help(){std::cout<<"Catena Siderum Siever 1.3 CPU/CUDA (p <= 2^62-1)\n"
 <<"New: CSSV --start X --end Y --terms M --step 1|2 --sign +|-|both --pmax 1e10 --out FILE\n"
 <<"Resume: same command + --resume (may increase --pmax; state defaults to FILE.state)\n"
 <<"Continue elsewhere: CSSV -i FILE.state -P 2e10 -o NEWFILE\n"
 <<"--state FILE --checkpoint-seconds 60 --progress-seconds 1 --stop-file FILE\n"
 <<"--info FILE.state: verify and display the saved actual frontier without GPU work\n"
 <<"--format candidates|factors (default candidates); --export-state FILE.state --out FILE\n"
 <<"--batch-primes 1024 (adaptive ceiling) --candidate-chunk 256 --device 0\n"
 <<"--backend auto|cpu|gpu --prime-generator auto|primesieve|segmented|mr --reference\n"
 <<"Exact integral scientific notation supported (1e10, 3.5e15). Ctrl+C saves safely.\n"
 <<"Default output: checked CSSV v2 survivor list (x sign), plus resumable .state.\n"
 <<"Legacy --format factors: 0=no proper factor; both columns minus plus.\n";}
Options parse(int argc,char** argv){Options o;for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&](){require(++i<argc,"missing value for "+a);return std::string(argv[i]);};auto integer=[&](){return parse_integer(next());};
 if(a=="--start"){o.cfg.start=integer();o.explicit_fields.insert("start");}
 else if(a=="--end"){o.cfg.end=integer();o.explicit_fields.insert("end");}
 else if(a=="--terms"){o.cfg.terms=integer();o.explicit_fields.insert("terms");}
 else if(a=="--step"||a=="--mode"){u64 n=integer();require(n==1||n==2,"step must be 1 or 2");o.cfg.step=unsigned(n);o.explicit_fields.insert("step");}
 else if(a=="--sign"){auto s=next();require(s=="+"||s=="-"||s=="both","sign must be +, - or both");o.cfg.both=s=="both";o.cfg.sign=s=="+"?1:-1;o.explicit_fields.insert("sign");}
 else if(a=="--pmin"||a=="-p"){o.cfg.pmin=integer();o.explicit_fields.insert("pmin");}
 else if(a=="--pmax"||a=="-P"){o.cfg.pmax=integer();o.explicit_fields.insert("pmax");}
 else if(a=="--out"||a=="-o")o.out=std::filesystem::u8path(next());
 else if(a=="--state")o.state=std::filesystem::u8path(next());
 else if(a=="--input"||a=="-i")o.input=std::filesystem::u8path(next());
 else if(a=="--export-state"){o.input=std::filesystem::u8path(next());o.export_only=true;}
 else if(a=="--format")o.format=next();
 else if(a=="--resume")o.resume=true;
 else if(a=="--stop-file")o.stopfile=std::filesystem::u8path(next());
 else if(a=="--backend")o.backend=next();
 else if(a=="--prime-generator")o.generator=next();
 else if(a=="--checkpoint-seconds")o.checkpoint=interval(next());
 else if(a=="--progress-seconds")o.progress=interval(next());
 else if(a=="--batch-primes"){u64 n=integer();require(n>=1&&n<=65536,"batch-primes must be 1..65536");o.batch=unsigned(n);}
 else if(a=="--candidate-chunk"){u64 n=integer();require(n>=1&&n<=4096,"candidate-chunk must be 1..4096");o.chunk=unsigned(n);}
 else if(a=="--device"){u64 n=integer();require(n<=1024,"invalid device");o.device=unsigned(n);}
 else if(a=="--reference")o.cfg.reference=true;
 else throw std::runtime_error("unknown option: "+a);
 }
 require(!o.out.empty(),"--out required");require(!(o.resume&&!o.input.empty()),"choose --resume or -i, not both");
 if(o.state.empty()){o.state=o.out;o.state+=".state";}
 require(!alias(o.out,o.state),"output and state must be different paths");if(!o.input.empty())require(!alias(o.input,o.out),"input state and output must differ");
 auto out_lock=o.out,state_lock=o.state;out_lock+=".lock";state_lock+=".lock";require(!alias(o.out,state_lock)&&!alias(o.state,out_lock),"data files must differ from lock sidecars");
 if(!o.stopfile.empty())require(!alias(o.stopfile,o.out)&&!alias(o.stopfile,o.state)&&(o.input.empty()||!alias(o.stopfile,o.input)),"stop-file must differ from data files");
 require(o.backend=="auto"||o.backend=="cpu"||o.backend=="gpu","invalid backend");require(o.generator=="auto"||o.generator=="primesieve"||o.generator=="segmented"||o.generator=="mr","invalid prime generator");require(o.format=="candidates"||o.format=="factors","format must be candidates or factors");if(o.export_only)require(o.explicit_fields.empty(),"export cannot override saved parameters or depth");return o;}
Snapshot initial(const Options& o){Snapshot s;bool loaded=o.resume||!o.input.empty();
 if(loaded){s=load_state((o.resume?o.state:o.input).u8string());auto same=[&](const char* name,u64 a,u64 b){if(o.explicit_fields.count(name))require(a==b,std::string("resume parameter mismatch: ")+name);};
  same("start",o.cfg.start,s.cfg.start);same("end",o.cfg.end,s.cfg.end);same("terms",o.cfg.terms,s.cfg.terms);same("step",o.cfg.step,s.cfg.step);same("pmin",o.cfg.pmin,s.cfg.pmin);
  if(o.explicit_fields.count("sign"))require(o.cfg.sign==s.cfg.sign&&o.cfg.both==s.cfg.both,"resume sign mismatch");
  if(o.explicit_fields.count("pmax"))s.cfg.pmax=o.cfg.pmax;
  require(s.cfg.pmax>=s.frontier&&s.cfg.pmax<=MAX_P&&s.cfg.pmax>=s.cfg.pmin,"new pmax must cover saved frontier and be <=2^62-1");s.cfg.reference=o.cfg.reference;
 }else{s.cfg=o.cfg;if(!o.explicit_fields.count("end"))s.cfg.end=s.cfg.start;
  require(s.cfg.start&&s.cfg.end>=s.cfg.start&&s.cfg.end-s.cfg.start<2000000&&s.cfg.terms&&s.cfg.terms<=1000000&&(s.cfg.step==1||s.cfg.step==2)&&(s.cfg.sign==1||s.cfg.sign==-1),"invalid range/terms/step/sign");
  require(s.cfg.end<UINT64_MAX&&s.cfg.terms-1<=(UINT64_MAX-1-s.cfg.end)/s.cfg.step,"last base + 1 overflows uint64");
  require(s.cfg.pmin>=2&&s.cfg.pmin<=s.cfg.pmax&&s.cfg.pmax<=MAX_P,"require 2<=pmin<=pmax<=2^62-1");s.frontier=s.cfg.pmin-1;s.factors.assign(size_t(s.cfg.end-s.cfg.start+1)*(s.cfg.both?2:1),NONE);
 }
 if(!o.resume){require(!std::filesystem::exists(o.out),"output already exists; use --resume or another path");require(!std::filesystem::exists(o.state)||(!o.input.empty()&&alias(o.state,o.input)),"state already exists; use --resume or another path");}
 return s;}
int run(int argc,char** argv){if(argc==1){help();return 0;}if(argc==3&&std::string(argv[1])=="--info"){auto s=load_state(argv[2]);size_t remaining=std::count(s.factors.begin(),s.factors.end(),NONE);std::cout<<"CSSVST01 checksum=valid, start="<<s.cfg.start<<", end="<<s.cfg.end<<", terms="<<s.cfg.terms<<", step="<<s.cfg.step<<", sign="<<(s.cfg.both?"both":s.cfg.sign==1?"+":"-")<<", pmin="<<s.cfg.pmin<<", actual_sieved_to="<<s.frontier<<", target="<<s.cfg.pmax<<", factorless_entries="<<remaining<<", complete="<<(s.frontier==s.cfg.pmax?"yes":"no")<<"\n";return 0;}for(int i=1;i<argc;++i)if(std::string(argv[i])=="--help"||std::string(argv[i])=="-h"){help();return 0;}
 Options o=parse(argc,argv);PathLock output_lock(o.out.u8string());if(o.export_only){require(!std::filesystem::exists(o.out),"output already exists");auto s=load_state(o.input.u8string());if(o.format=="candidates")write_candidates(o.out.u8string(),s);else write_factors(o.out.u8string(),s);std::cout<<"Exported "<<o.format<<" at actual_sieved_to="<<s.frontier<<"; no computation performed\n";return 0;}PathLock state_lock(o.state.u8string());Snapshot s=initial(o);const Config& c=s.cfg;
 std::cout<<"Catena Siderum Siever 1.3: start="<<c.start<<", end="<<c.end<<", terms="<<c.terms<<", step="<<c.step<<", sign="<<(c.both?"both":c.sign==1?"+":"-")<<", primes=("<<s.frontier<<","<<c.pmax<<"]\n";
 std::cout<<"state="<<o.state.u8string()<<", checkpoint_seconds="<<o.checkpoint<<", progress_seconds="<<o.progress<<"; safe stop at completed prime-batch boundary\n";
 const size_t count=size_t(c.end-c.start+1),width=c.both?2:1;std::vector<Entry> all(count),active;u64 not_prime=0;
 for(size_t i=0;i<count;++i){u64 x=c.start+i,minus=capped_value(c,x,-1),plus=capped_value(c,x,1);all[i]={u64(i),x,minus,plus,0};if((c.both||c.sign==-1)&&minus<2)++not_prime;}
 auto rebuild=[&](){active.clear();for(size_t i=0;i<count;++i){Entry e=all[i];if((c.both||c.sign==-1)&&e.minus_value>=2&&(s.factors[i*width]==NONE||c.reference))e.mask|=1;if((c.both||c.sign==1)&&e.plus_value>=2&&(s.factors[i*width+(c.both?1:0)]==NONE||c.reference))e.mask|=2;if(e.mask)active.push_back(e);}};
 auto stopped_now=[&](){if(!o.stopfile.empty()&&std::filesystem::exists(o.stopfile))stopped.store(true);return stopped.load();};
 auto save=[&](const char* reason){save_state(o.state.u8string(),s);if(o.format=="candidates")write_candidates(o.out.u8string(),s);else write_factors(o.out.u8string(),s);std::cout<<"checkpoint: p="<<s.frontier<<", reason="<<reason<<", state="<<o.state.u8string()<<"\n";};
 save("initial");if(stopped_now()){std::cout<<"result=INTERRUPTED, p="<<s.frontier<<"\n";return 130;}
 bool gpu=false;
#ifdef __CUDACC__
 if(o.backend!="cpu"){int devices=0;auto e=cudaGetDeviceCount(&devices);if(e==cudaSuccess&&devices>int(o.device)){cuda_ok(cudaSetDevice(o.device),"select device");gpu=true;}else if(o.backend=="gpu")throw std::runtime_error("requested CUDA device unavailable");}
 std::unique_ptr<Device<u64>> df;std::unique_ptr<Device<Mod>> dp;std::unique_ptr<Device<Entry>> de;
 if(gpu){df=std::make_unique<Device<u64>>(s.factors.size());dp=std::make_unique<Device<Mod>>(o.batch);de=std::make_unique<Device<Entry>>(count);cuda_ok(cudaMemcpy(df->p,s.factors.data(),s.factors.size()*8,cudaMemcpyHostToDevice),"upload factors");cudaDeviceProp prop{};cuda_ok(cudaGetDeviceProperties(&prop,o.device),"device properties");std::cout<<"device="<<prop.name<<"\n";}
#else
 require(o.backend!="gpu","this build has no CUDA backend");
#endif
 std::cout<<"backend="<<(gpu?"CUDA":"CPU")<<", arithmetic=exact u32 / Montgomery62, compact="<<(c.reference?"no":"yes")<<"\nInitializing prime generator...\n";
 PrimeGenerator generator(s.frontier+1,c.pmax,o.generator,&stopped);std::cout<<"Prime generator: "<<generator.description()<<"\n";
 auto began=Clock::now(),last_save=began,last_report=began;u64 initial_frontier=s.frontier,last_frontier=s.frontier,primes_done=0,last_primes=0;
 unsigned next_batch=gpu?std::min(o.batch,1024u):std::min(o.batch,16u);u64 removed=0,minus_removed=0,plus_removed=0;
 auto counts=[&](){removed=minus_removed=plus_removed=0;for(size_t i=0;i<s.factors.size();++i)if(s.factors[i]!=NONE){++removed;if((c.both&&i%2==0)||(!c.both&&c.sign==-1))++minus_removed;else ++plus_removed;}};
 auto report=[&](bool force){if(!force&&seconds(last_report)<o.progress)return;counts();double dt=seconds(last_report);std::cout<<std::fixed<<std::setprecision(3)<<"progress: p="<<s.frontier<<"/"<<c.pmax<<", primes="<<primes_done<<", p/sec="<<(primes_done-last_primes)/std::max(dt,1e-9)<<", removed="<<removed<<", remaining="<<s.factors.size()-removed-not_prime<<", minus_removed="<<minus_removed<<", plus_removed="<<plus_removed<<", active_starts="<<active.size()<<", elapsed_s="<<seconds(began)<<", eta_s=";
  if(s.frontier==c.pmax)std::cout<<0;else if(s.frontier>last_frontier)std::cout<<double(c.pmax-s.frontier)*dt/double(s.frontier-last_frontier);else std::cout<<"n/a";std::cout<<", batch="<<next_batch<<"\n";last_report=Clock::now();last_frontier=s.frontier;last_primes=primes_done;};
 while(s.frontier<c.pmax&&!stopped_now()){
  rebuild();if(active.empty()){s.frontier=c.pmax;std::cout<<"All eligible candidates already have proper factors; no further arithmetic needed.\n";break;}
  auto tick=Clock::now();auto ps=generator.next(next_batch);u64 generated_frontier=generator.frontier();size_t used=ps.size();auto before=s.factors;
  if(!ps.empty()){
   std::vector<Mod> mods;mods.reserve(ps.size());for(auto p:ps)mods.push_back(make_mod(p));
#ifdef __CUDACC__
   if(gpu){cuda_ok(cudaMemcpy(dp->p,mods.data(),mods.size()*sizeof(Mod),cudaMemcpyHostToDevice),"upload primes");cuda_ok(cudaMemcpy(de->p,active.data(),active.size()*sizeof(Entry),cudaMemcpyHostToDevice),"upload survivors");unsigned chunk=o.chunk;if(c.terms>10000)chunk=std::min(chunk,32u);chunk=std::max(chunk,unsigned((active.size()+65534)/65535));
    kernel<<<dim3(unsigned((mods.size()+127)/128),unsigned((active.size()+chunk-1)/chunk)),128>>>(c,dp->p,mods.size(),de->p,active.size(),chunk,df->p);cuda_ok(cudaGetLastError(),"sieve launch");cuda_ok(cudaDeviceSynchronize(),"sieve synchronize");cuda_ok(cudaMemcpy(s.factors.data(),df->p,s.factors.size()*8,cudaMemcpyDeviceToHost),"download factors");
   }else
#endif
   {used=0;for(const auto& m:mods){hd_scan(c,m,active.data(),active.size(),s.factors.data());++used;if(stopped_now())break;}if(used<ps.size())generated_frontier=ps[used-1];}
   std::unordered_set<u64> verified;for(size_t i=0;i<s.factors.size();++i)if(before[i]!=s.factors[i]){u64 q=s.factors[i],x=c.start+i/width;int sign=c.both?(i%2?1:-1):c.sign;require(q>=c.pmin&&q>s.frontier&&q<=generated_frontier,"factor outside completed batch");if(verified.insert(q).second)require(prime64(q),"GPU factor is not prime");u64 product=1;for(u64 j=0;j<c.terms;++j){u64 a=x+c.step*j;product=cpu_mulmod(product,cpu_powmod(a,a+1,q),q);}require((sign==1?product==q-1:product==1)&&capped_value(c,x,sign)>q,"GPU factor verification failed; checkpoint not advanced");}
  }
  require(generated_frontier>=s.frontier&&generated_frontier<=c.pmax,"prime generator frontier invalid");s.frontier=generated_frontier;primes_done+=used;
  double dt=seconds(tick);if(dt>.6&&next_batch>1)next_batch=std::max(1u,next_batch/2);else if(dt<.15&&next_batch<o.batch)next_batch=std::min(o.batch,next_batch*2);
  report(false);if(seconds(last_save)>=o.checkpoint){save("periodic");last_save=Clock::now();}if(generator.done())break;
 }
 bool interrupted=stopped_now()&&s.frontier<c.pmax;save(interrupted?"interrupt":"complete");rebuild();report(true);
 std::cout<<"CSSV "<<(gpu?"CUDA":"CPU")<<", primes="<<primes_done<<", removed="<<removed<<", starts="<<count<<", seconds="<<seconds(began)<<", result="<<(interrupted?"INTERRUPTED":"COMPLETE")<<", actual_sieved_to="<<s.frontier<<", started_at="<<initial_frontier<<"\n";return interrupted?130:0;
}
}
int main(int argc,char** argv){prp_console::initialize_utf8_output();std::cout.setf(std::ios::unitbuf);std::cerr.setf(std::ios::unitbuf);std::signal(SIGINT,cssv::stop_handler);std::signal(SIGTERM,cssv::stop_handler);
#ifdef _WIN32
 SetConsoleCtrlHandler(cssv::console_handler,TRUE);
#endif
 try{
#ifdef _WIN32
  int count=0;LPWSTR* wide=CommandLineToArgvW(GetCommandLineW(),&count);cssv::require(wide!=nullptr,"cannot read Unicode command line");std::vector<std::string> values;
  try{for(int i=0;i<count;++i){int n=WideCharToMultiByte(CP_UTF8,0,wide[i],-1,nullptr,0,nullptr,nullptr);cssv::require(n>0,"invalid Unicode argument");std::string s(n,'\0');cssv::require(WideCharToMultiByte(CP_UTF8,0,wide[i],-1,s.data(),n,nullptr,nullptr)!=0,"Unicode conversion failed");s.resize(n-1);values.push_back(std::move(s));}}catch(...){LocalFree(wide);throw;}LocalFree(wide);std::vector<char*> args;for(auto& s:values)args.push_back(s.data());return cssv::run(count,args.data());
#else
  return cssv::run(argc,argv);
#endif
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 2;}}
