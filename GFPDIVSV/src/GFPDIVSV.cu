// SPDX-License-Identifier: GPL-2.0-or-later
// Fixed n! or n#; remove ONE factor, then sieve quotient +/- 1.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "factorial_expression.hpp"
#include "sha256.hpp"
#include <array>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <numeric>
#include <cerrno>
#include <memory>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include "primesieve_iterator.h"
#include "cuda_backend.hpp"
#ifdef GFPDIVSV_USE_GMP
#include <boost/multiprecision/gmp.hpp>
using Integer=boost::multiprecision::mpz_int;
#else
using Integer=boost::multiprecision::cpp_int;
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/sysinfo.h>
#endif
namespace {
using gfpps_model::cpp_int;
namespace fs=std::filesystem;
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free,"signal flag must be lock-free");
void stop_signal(int){interrupted.store(true,std::memory_order_relaxed);}
#ifdef _WIN32
BOOL WINAPI console_stop(DWORD type){if(type==CTRL_C_EVENT||type==CTRL_BREAK_EVENT){stop_signal(0);return TRUE;}return FALSE;}
#endif
bool stopped(){return interrupted.load(std::memory_order_relaxed);}
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
constexpr uint64_t MAX_P=(uint64_t(1)<<62)-1,MAX_BITS=16000000;
struct Options{
 uint64_t n=0,pmax=100000000,low=2,high=0,max_batches=0;
 int signs=3;char family='!';std::string input,output,stop_file,factors,prime_library;
 double progress=1,checkpoint=60;bool n_set=false,family_set=false,sign_set=false,bounds_set=false;
 std::string gpu_product="valuation";
 uint64_t gpu_batch=32768,gpu_chains=4,gpu_tile=4096,device=0;
 bool verify_cuda=false;
 std::string prime_mode="auto",tuning_cache;
 uint64_t prime_threads=1,prime_span=33554432;
 bool batch_set=false,threads_set=false,retune=false;
};
uint64_t number(std::string s){
 // Scientific integer input is exact; never convert sieve bounds through double.
 if(s.size()>2&&s[0]=='0'&&(s[1]=='x'||s[1]=='X'))return gfpps_model::integer(s);
 auto p=s.find_first_of("eE");if(p==std::string::npos)return gfpps_model::integer(s);
 if(s.find_first_of("eE",p+1)!=std::string::npos)throw std::runtime_error("invalid scientific integer");
 auto mantissa=s.substr(0,p),exponent=s.substr(p+1);auto dot=mantissa.find('.');size_t fractional=0;
 if(dot!=std::string::npos){fractional=mantissa.size()-dot-1;mantissa.erase(dot,1);}
 uint64_t e=gfpps_model::integer(exponent);if(e>19||e<fractional)throw std::runtime_error("scientific value is not a supported integer");
 uint64_t value=gfpps_model::integer(mantissa);for(size_t i=0;i<e-fractional;++i){if(value>UINT64_MAX/10)throw std::runtime_error("integer overflow");value*=10;}return value;
}
Options options(int argc,char**argv){
 Options o;for(int i=1;i<argc;++i){std::string arg=argv[i];auto next=[&](){if(++i>=argc)throw std::runtime_error("missing value for "+arg);return std::string(argv[i]);};
  if(arg=="--n"){o.n=number(next());o.n_set=true;}
  else if(arg=="--family"){auto f=next();if(f!="factorial"&&f!="primorial")throw std::runtime_error("family must be factorial or primorial");o.family=f=="factorial"?'!':'#';o.family_set=true;}
  else if(arg=="--sign"){auto s=next();if(s!="both"&&s!="+"&&s!="-")throw std::runtime_error("sign must be both, + or -");o.signs=s=="both"?3:s=="-"?1:2;o.sign_set=true;}
  else if(arg=="--divisor-min"){o.low=number(next());o.bounds_set=true;}
  else if(arg=="--divisor-max"){o.high=number(next());o.bounds_set=true;}
  else if(arg=="-P"||arg=="--pmax")o.pmax=number(next());
  else if(arg=="-i"||arg=="--input")o.input=next();
  else if(arg=="-o"||arg=="--out")o.output=next();
  else if(arg=="-O"||arg=="--factors")o.factors=next();
  else if(arg=="--stop-file")o.stop_file=next();
  else if(arg=="--primesieve-library")o.prime_library=next();
  else if(arg=="--prime-mode")o.prime_mode=next();
  else if(arg=="--prime-threads"){o.prime_threads=number(next());o.threads_set=true;}
  else if(arg=="--prime-span")o.prime_span=number(next());
  else if(arg=="--tuning-cache")o.tuning_cache=next();
  else if(arg=="--retune")o.retune=true;
  else if(arg=="--gpu-product")o.gpu_product=next();
  else if(arg=="--batch-primes"||arg=="--gpu-batch-primes"){o.gpu_batch=number(next());o.batch_set=true;}
  else if(arg=="--gpu-chains")o.gpu_chains=number(next());
  else if(arg=="--gpu-tile-chunks")o.gpu_tile=number(next());
  else if(arg=="--device")o.device=number(next());
  else if(arg=="--verify-cuda")o.verify_cuda=true;
  else if(arg=="--max-batches")o.max_batches=number(next());
  else if(arg=="--checkpoint-seconds")o.checkpoint=number(next());
  else if(arg=="--progress-seconds")o.progress=number(next());
  else if(arg=="-h"||arg=="--help"){
   std::cout<<"GFPDIVSV 1.1 CUDA - Generalized Factorial/Primorial Factor-Removal Siever\n"
    <<"Generate: GFPDIVSV --n N --family factorial|primorial --sign +|-|both\n"
    <<"Resume: GFPDIVSV -i FILE -P 1e11 [-o NEWFILE]\n"
    <<"Default -P 1e8; automatically named output; existing files are never silently overwritten.\n"
    <<"Factorial divisors:2..n-1. Primorial:odd primes below the largest prime factor.\n"
    <<"--divisor-min D --divisor-max D --device 0\n"
    <<"--batch-primes N (manual override,16..1048576), --gpu-chains 1|2|4|8 (4)\n"
    <<"--gpu-tile-chunks 4096 (multiple of8), --gpu-product valuation|raw\n"
    <<"--checkpoint-seconds 60 --progress-seconds 1 --stop-file FILE -O FACTORS\n"
    <<"--primesieve-library FILE, or GFPDIVSV_PRIMESIEVE_LIBRARY.\n"
    <<"--prime-mode auto|array|iterator|pipeline (auto), --prime-threads 1..16 (manual override)\n"
    <<"Auto calibrates real CUDA work, caches the choice, and adds supplier threads only on measured starvation.\n"
    <<"--tuning-cache FILE (default OUTPUT.tuning), --retune; --prime-span 33554432\n"
    <<"Library search: executable directory, its lib directory, then system.\n"
    <<"--verify-cuda checks every modular residue against independent exact arithmetic.\n"
    <<"--selftest performs bounded CUDA arithmetic tests.\n"
    <<"--selftest-primes checks ordered prime streams; --selftest-supplier checks adaptive hysteresis.\n"
    <<"--max-batches N bounds a PARTIAL test; never implies target completion.\n"
    <<"CUDA is required for sieving. Host preprocessing and independent factor verification remain enabled.\n";std::exit(0);
  }else throw std::runtime_error("unknown option: "+arg);
 }
 if(!o.checkpoint||!o.progress||o.pmax>MAX_P)throw std::runtime_error("invalid resource or sieve limits");
 if(o.gpu_product!="valuation"&&o.gpu_product!="raw")throw std::runtime_error("gpu-product must be valuation or raw");
 if((o.prime_mode!="auto"&&o.prime_mode!="array"&&o.prime_mode!="iterator"&&o.prime_mode!="pipeline")||o.prime_threads<1||o.prime_threads>16||o.prime_span<1048576||o.prime_span>1073741824)throw std::runtime_error("invalid prime supply options");
 if(o.prime_mode!="pipeline"&&o.prime_mode!="auto"&&o.prime_threads!=1)throw std::runtime_error("multiple prime threads require auto or pipeline mode");
 if(o.gpu_batch<16||o.gpu_batch>1048576||(o.gpu_chains!=1&&o.gpu_chains!=2&&o.gpu_chains!=4&&o.gpu_chains!=8)||o.gpu_tile<8||o.gpu_tile>65536||o.gpu_tile%8||o.device>63)throw std::runtime_error("invalid CUDA resource limits");
 if(o.input.empty()&&(!o.n_set||!o.family_set))throw std::runtime_error("new search requires --n and --family");
 if(o.input.empty()&&o.output.empty())o.output="gfpdiv_"+std::string(o.family=='!'?"factorial":"primorial")+"_n"+std::to_string(o.n)+"_"+(o.signs==3?"both":o.signs==1?"minus":"plus")+"_p"+std::to_string(o.pmax)+".txt";
 return o;
}
std::string hash(const std::string& s){CheckpointSha256 h;h.update(reinterpret_cast<const uint8_t*>(s.data()),s.size());auto d=h.final();std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto b:d)out<<std::setw(2)<<unsigned(b);return out.str();}
std::string base_hash(const cpp_int& n){std::string bytes;export_bits(n,std::back_inserter(bytes),8,false);return hash(bytes);}
void atomic_write(const fs::path& path,const std::string& data){
 auto temp=path;
#ifdef _WIN32
 temp+=".tmp."+std::to_string(GetCurrentProcessId());FILE* f=_wfopen(temp.c_str(),L"wb");
#else
 temp+=".tmp."+std::to_string(getpid());FILE* f=fopen(temp.c_str(),"wb");
#endif
 if(!f)throw std::runtime_error("cannot create checkpoint temporary file");
 bool ok=fwrite(data.data(),1,data.size(),f)==data.size()&&fflush(f)==0;
#ifdef _WIN32
 if(ok)ok=_commit(_fileno(f))==0;
#else
 if(ok)ok=fsync(fileno(f))==0;
#endif
 if(fclose(f)!=0)ok=false;if(!ok)throw std::runtime_error("checkpoint flush failed; previous file preserved");
#ifdef _WIN32
 if(!MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("checkpoint replace failed");
#else
 if(rename(temp.c_str(),path.c_str())!=0)throw std::runtime_error("checkpoint replace failed");
 auto parent=path.parent_path();if(parent.empty())parent=".";int fd=open(parent.c_str(),O_RDONLY|O_DIRECTORY);if(fd<0)throw std::runtime_error("cannot sync checkpoint directory");int result=fsync(fd);close(fd);if(result)throw std::runtime_error("checkpoint directory fsync failed");
#endif
}
struct State{
 uint64_t n=0,low=2,high=0,p=0;char family='!';int signs=3;std::string base_sha;
 std::vector<uint8_t> mask;
 uint64_t count()const{uint64_t total=0;for(auto m:mask)total+=(m&1)+((m>>1)&1);return total;}
 std::string expression(uint32_t t,int sign)const{return std::to_string(n)+family+"/"+std::to_string(t)+(sign==1?"+1":"-1");}
 std::string encode()const{
  std::ostringstream out;out<<"GFPDIVSV 1\nn "<<n<<"\nfamily "<<family<<"\ndivisor_min "<<low<<"\ndivisor_max "<<high<<"\nsign_mask "<<signs<<"\nsieved_to "<<p<<"\nbase_sha256 "<<base_sha<<"\ncandidates "<<count()<<"\n";
  for(uint32_t t=2;t<mask.size();++t){if(mask[t]&1)out<<t<<" -1\n";if(mask[t]&2)out<<t<<" 1\n";}
  auto body=out.str();return body+"sha256 "+hash(body)+"\n";
 }
};
State read_state(const fs::path& file){
 auto size=fs::file_size(file);if(size>512ull*1024*1024)throw std::runtime_error("checkpoint exceeds size limit");
 std::ifstream in(file,std::ios::binary);std::string text((std::istreambuf_iterator<char>(in)),{});if(text.size()!=size)throw std::runtime_error("checkpoint read failed");
 auto pos=text.rfind("sha256 ");if(pos==std::string::npos||!pos||text[pos-1]!='\n'||text.size()!=pos+7+64+1||text.back()!='\n'||hash(text.substr(0,pos))!=text.substr(pos+7,64))throw std::runtime_error("checkpoint SHA-256 mismatch");
 std::istringstream fields(text.substr(0,pos));auto take=[&](const char* key){std::string k,v;if(!(fields>>k>>v)||k!=key)throw std::runtime_error(std::string("invalid checkpoint field: ")+key);return v;};
 if(take("GFPDIVSV")!="1")throw std::runtime_error("unsupported checkpoint version");State s;
 s.n=number(take("n"));auto family=take("family");if(family!="!"&&family!="#")throw std::runtime_error("invalid checkpoint family");s.family=family[0];s.low=number(take("divisor_min"));s.high=number(take("divisor_max"));s.signs=int(number(take("sign_mask")));s.p=number(take("sieved_to"));s.base_sha=take("base_sha256");auto count=number(take("candidates"));
 if(s.n<3||s.n>10000000||s.low<2||s.high>=s.n||s.low>s.high||s.signs<1||s.signs>3||s.p<s.n||s.p>MAX_P||count>2*s.n||s.base_sha.size()!=64)throw std::runtime_error("invalid checkpoint metadata");
 auto domain=gfpps_model::divisors(s.n,s.family,s.low,s.high);std::vector<uint8_t> allowed(size_t(s.n)+1);for(auto t:domain)allowed[t]=s.signs;s.mask.resize(size_t(s.n)+1);
 uint64_t previous=0;for(uint64_t i=0;i<count;++i){uint64_t t;int sign;if(!(fields>>t>>sign)||t>=s.mask.size()||(sign!=-1&&sign!=1))throw std::runtime_error("invalid candidate row");int bit=sign==-1?1:2;uint64_t key=t*2+(sign==1);if(key<=previous||!(allowed[t]&bit))throw std::runtime_error("invalid candidate domain/order");previous=key;s.mask[t]|=bit;}
 std::string extra;if(fields>>extra)throw std::runtime_error("unexpected checkpoint tail");return s;
}
std::vector<uint64_t> remainders(const Integer& a,const std::vector<uint64_t>& moduli){
 size_t width=1;while(width<moduli.size())width*=2;std::vector<Integer> tree(2*width,Integer(1));
 for(size_t i=0;i<moduli.size();++i)tree[width+i]=moduli[i];
 for(size_t i=width-1;i;i--){if(stopped())throw std::runtime_error("interrupted remainder tree");tree[i]=tree[2*i]*tree[2*i+1];}
 std::vector<uint64_t> result(moduli.size());
 std::function<void(size_t,const Integer&)> descend=[&](size_t i,const Integer& value){
  if(stopped())throw std::runtime_error("interrupted remainder tree");if(i>=width){if(i-width<result.size())result[i-width]=value.convert_to<uint64_t>();return;}
  Integer left=value%tree[2*i],right=value%tree[2*i+1];descend(2*i,left);descend(2*i+1,right);
 };Integer root=a%tree[1];descend(1,root);return result;
}
struct Hit{uint32_t divisor;int sign;uint64_t factor;};
bool equal_candidate(const cpp_int& a,uint32_t t,int sign,uint64_t prime){return a==cpp_int(t)*(prime-sign);}
uint64_t apply(State& s,const cpp_int& a,const std::vector<Hit>& hits,std::ofstream& log){
 uint64_t removed=0;for(auto h:hits){int bit=h.sign<0?1:2;if(s.mask[h.divisor]&bit&&!equal_candidate(a,h.divisor,h.sign,h.factor)){
  s.mask[h.divisor]&=uint8_t(~bit);++removed;if(log.is_open()){log<<h.factor<<" | "<<s.expression(h.divisor,h.sign)<<'\n';if(!log)throw std::runtime_error("factor log write failed");}
 }}return removed;
}
// Encode factorial prime valuations as binary-weight groups. This avoids
// multiplying n-1 ordinary factorial factors for every sieve prime.
// For primorials every valuation is1, so one packed prime group remains.
gfpdiv_cuda::Groups product_groups(const State& s,const Integer& exact,bool raw) {
 gfpdiv_cuda::Groups low_to_high(1);std::vector<uint64_t> packed(1,1);
 auto add=[&](size_t bit,uint64_t factor){
  if(low_to_high.size()<=bit){low_to_high.resize(bit+1);packed.resize(bit+1,1);}
  auto& current=packed[bit];
  if(current>UINT64_MAX/factor){low_to_high[bit].push_back(current);current=1;}
  current*=factor;
 };
 if(raw){for(auto factor:gfpps_model::factors(s.n,s.family,stopped))add(0,factor);}
 else{
  auto primes=gfpps_model::factors(s.n,'#',stopped);
  for(auto prime:primes){
   if(stopped())throw std::runtime_error("interrupted product encoding");
   uint64_t valuation=1;
   if(s.family=='!'){valuation=0;for(uint64_t q=s.n/prime;q;q/=prime)valuation+=q;}
   for(size_t bit=0;valuation;++bit,valuation>>=1)if(valuation&1)add(bit,prime);
  }
 }
 for(size_t i=0;i<packed.size();++i)if(packed[i]!=1)low_to_high[i].push_back(packed[i]);
 gfpdiv_cuda::Groups result(low_to_high.rbegin(),low_to_high.rend());
 // Prove the representation equals the independently built GMP/Boost mother
 // product before uploading it. No GPU result is trusted for this equality.
 std::function<Integer(const std::vector<uint64_t>&,size_t,size_t)> multiply_words;
 multiply_words=[&](const std::vector<uint64_t>& v,size_t lo,size_t hi)->Integer{
  if(stopped())throw std::runtime_error("interrupted product representation validation");
  if(hi-lo<16){Integer p=1;for(size_t i=lo;i<hi;++i)p*=v[i];return p;}
  size_t mid=lo+(hi-lo)/2;Integer left=multiply_words(v,lo,mid),right=multiply_words(v,mid,hi);return left*right;
 };
 Integer reconstructed=1;
 for(const auto& group:result){reconstructed*=reconstructed;reconstructed*=multiply_words(group,0,group.size());}
 if(reconstructed!=exact)throw std::runtime_error("CUDA product representation mismatch");
 size_t words=0;for(const auto& group:result)words+=group.size();
 std::cout<<"CUDA product: groups="<<result.size()<<", words="<<words
          <<", encoding="<<(raw?"raw-packed":"prime-valuation-binary")<<", exact_host_equality=yes\n";
 return result;
}
void verify_gpu_hits(const State& s,const Integer& a,const std::vector<Hit>& hits) {
 if(hits.empty())return;
 std::map<uint64_t,uint64_t> checked;
 for(auto h:hits){
  if(h.factor<=s.n||h.divisor>=h.factor)throw std::runtime_error("CUDA hit outside modular search domain");
  checked.emplace(h.factor,0);
 }
 // Repeated single-word divisions of a multi-million-bit mother product can
 // dominate early dense ranges. Use an independent CPU remainder tree for
 // larger hit sets, still checking EVERY proposed factor before any mutation.
 if(checked.size()>=16){
  std::vector<uint64_t> moduli;moduli.reserve(checked.size());
  for(auto [p,value]:checked)moduli.push_back(p);
  auto values=remainders(a,moduli);size_t i=0;
  for(auto& row:checked)row.second=values[i++];
 }else for(auto& row:checked){Integer residue=a%row.first;row.second=residue.convert_to<uint64_t>();}
 for(auto h:hits){
  auto found=checked.find(h.factor);
  const uint64_t expected=h.sign<0?h.divisor:h.factor-h.divisor;
  if(found->second!=expected)throw std::runtime_error("independent CPU verification rejected a CUDA factor");
 }
}
uint64_t inverse(uint64_t a,uint64_t p){uint64_t r=1,e=p-2;while(e){if(e&1)r=r*a%p;a=a*a%p;e>>=1;}return r;}
std::vector<Hit> cancellation_checks(const State& s,const Integer& a){
 // Only primes completely removed from A can divide A/t +/- 1 below n.
 // Factor every divisor and compare exact valuations, rather than relying on
 // an assumed large-n shortcut. A mod p^(v+1) recovers (A/p^v) mod p.
 struct Check{uint32_t t,p;uint64_t power;};std::vector<Check> checks;std::vector<uint64_t> moduli;
 std::vector<uint32_t> spf(size_t(s.n)+1);for(uint32_t p=2;p<=s.n;++p)if(!spf[p])for(uint32_t k=p;k<=s.n;k+=p)if(!spf[k])spf[k]=p;
 for(uint32_t t=2;t<s.mask.size();++t)if(s.mask[t]){
  uint32_t rest=t;while(rest>1){uint32_t p=spf[rest],v=0;uint64_t power=1;while(rest%p==0){rest/=p;++v;power*=p;}
   uint64_t valuation=0;if(s.family=='#')valuation=1;else for(uint64_t q=s.n/p;q;q/=p)valuation+=q;
   if(valuation==v){checks.push_back({t,p,power});moduli.push_back(power*p);}
  }
 }
 std::vector<Hit> hits;
 for(size_t start=0;start<checks.size();start+=2048){size_t end=std::min(checks.size(),start+2048);auto rem=remainders(a,{moduli.begin()+start,moduli.begin()+end});
  for(size_t i=start;i<end;++i){const auto& c=checks[i];if(rem[i-start]%c.power)throw std::runtime_error("p-adic cancellation invariant failed");uint64_t quotient=(rem[i-start]/c.power)*inverse(c.t/c.power,c.p)%c.p;
   if(quotient==1)hits.push_back({c.t,-1,c.p});if(quotient==c.p-1)hits.push_back({c.t,1,c.p});
  }
 }return hits;
}
fs::path executable_directory(){
#ifdef _WIN32
 std::vector<wchar_t> name(32768);DWORD size=GetModuleFileNameW(nullptr,name.data(),DWORD(name.size()));
 if(!size||size>=name.size())throw std::runtime_error("cannot locate executable directory");return fs::path(std::wstring(name.data(),size)).parent_path();
#else
 std::error_code ec;auto name=fs::read_symlink("/proc/self/exe",ec);if(ec)throw std::runtime_error("cannot locate executable directory");return name.parent_path();
#endif
}
class Primes{
 using Generate=void*(*)(uint64_t,uint64_t,int);using Free=void(*)(void*);
 using IterInit=void(*)(primesieve_iterator*);using IterJump=void(*)(primesieve_iterator*,uint64_t,uint64_t);
 IterInit iter_init=nullptr,iter_free=nullptr,iter_generate=nullptr;IterJump iter_jump=nullptr;
 primesieve_iterator iterator_{};bool iter_ready_=false,use_iterator_=false;
 std::string identity_="built-in-segmented";
 void* library=nullptr;Generate generate=nullptr;Free release=nullptr;uint64_t next_,end_,segment_low_=0;bool exhausted_=false;std::vector<uint32_t> base_;std::vector<uint64_t> cache_;size_t index_=0;
 void* symbol(const char* name){
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library),name));
#else
  return dlsym(library,name);
#endif
 }
 void fill(){
  cache_.clear();index_=0;if(segment_low_>end_)return;uint64_t hi=std::min(end_,segment_low_+1048575);std::vector<bool> composite(size_t(hi-segment_low_+1));
  for(auto p:base_){if(uint64_t(p)*p>hi)break;uint64_t start=std::max(uint64_t(p)*p,segment_low_+(p-segment_low_%p)%p);for(uint64_t x=start;x<=hi;x+=p)composite[size_t(x-segment_low_)]=true;}
  for(uint64_t x=segment_low_;x<=hi;++x)if(x>=2&&!composite[size_t(x-segment_low_)])cache_.push_back(x);segment_low_=hi+1;
 }
public:
 Primes(uint64_t first,uint64_t last,const std::string& selected,bool persistent=false,bool quiet=false):use_iterator_(persistent),next_(first),end_(last),segment_low_(first){
  const char* names[]={
#ifdef _WIN32
   "primesieve.dll","libprimesieve.dll"
#else
   "libprimesieve.so.12","libprimesieve.so"
#endif
  };
  std::vector<fs::path> paths;std::string explicit_library=selected;
  if(explicit_library.empty())if(const char* value=std::getenv("GFPDIVSV_PRIMESIEVE_LIBRARY"))explicit_library=value;
  if(!explicit_library.empty()){
   auto file=fs::absolute(fs::u8path(explicit_library));if(!fs::is_regular_file(file))throw std::runtime_error("explicit primesieve library is not a regular file");paths.push_back(file);
  }else{
   const auto dir=executable_directory();for(auto name:names)paths.push_back(dir/name);
   for(auto name:names)paths.push_back(dir/"lib"/name);for(auto name:names)paths.push_back(fs::u8path(name));
  }
  std::string loaded;
  for(const auto& path:paths){
#ifdef _WIN32
   DWORD flags=LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;if(path.is_absolute())flags|=LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
   library=static_cast<void*>(LoadLibraryExW(path.c_str(),nullptr,flags));
#else
   library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
#endif
   if(library){loaded=path.u8string();break;}
  }
  if(!library&&!explicit_library.empty())throw std::runtime_error("cannot load explicitly selected primesieve library");
  if(library){generate=reinterpret_cast<Generate>(symbol("primesieve_generate_n_primes"));release=reinterpret_cast<Free>(symbol("primesieve_free"));if(!generate||!release)throw std::runtime_error("incompatible libprimesieve API");
   auto version=reinterpret_cast<const char*(*)()>(symbol("primesieve_version"));const char* version_text=version?version():nullptr;
   identity_=loaded+":"+(version_text?version_text:"unknown");
   if(use_iterator_&&(!version_text||std::string(version_text).rfind("12.",0)!=0))throw std::runtime_error("persistent prime supply requires the primesieve12.x iterator ABI; bundle/select version12.x");
   auto set_threads=reinterpret_cast<void(*)(int)>(symbol("primesieve_set_num_threads"));if(set_threads)set_threads(1);
   if(use_iterator_){
    iter_init=reinterpret_cast<IterInit>(symbol("primesieve_init"));iter_free=reinterpret_cast<IterInit>(symbol("primesieve_free_iterator"));iter_generate=reinterpret_cast<IterInit>(symbol("primesieve_generate_next_primes"));iter_jump=reinterpret_cast<IterJump>(symbol("primesieve_jump_to"));
    if(!iter_init||!iter_free||!iter_generate||!iter_jump)throw std::runtime_error("libprimesieve iterator API unavailable");
    iter_init(&iterator_);iter_ready_=true;iter_jump(&iterator_,first,last);
   }
   if(!quiet)std::cout<<"prime_generator=libprimesieve C API ("<<(use_iterator_?"persistent iterator":"array")<<"), library="<<loaded<<"\n";
  }else{
   if(last>1000000000000ull)throw std::runtime_error("libprimesieve required above 1e12; install/bundle the library");
   uint32_t root=uint32_t(std::sqrt((long double)last));while(uint64_t(root+1)*(root+1)<=last)++root;
   base_=gfpps_model::factors(root,'#');std::cout<<"prime_generator=built-in segmented Eratosthenes\n";
  }
 }
 ~Primes(){if(iter_ready_)iter_free(&iterator_);if(library){
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(library));
#else
  dlclose(library);
#endif
 }}
 const std::string& identity()const{return identity_;}
 void range(uint64_t first,uint64_t last){
  next_=first;end_=last;segment_low_=first;exhausted_=false;cache_.clear();index_=0;
  if(iter_ready_)iter_jump(&iterator_,first,last);
 }
 std::vector<uint64_t> next(size_t count){
  if(exhausted_||next_>end_)return {};
  std::vector<uint64_t> out;
  out.reserve(count);
  if(iter_ready_){
   while(out.size()<count){
    ++iterator_.i;if(iterator_.i>=iterator_.size)iter_generate(&iterator_);
    if(iterator_.is_error||!iterator_.size)throw std::runtime_error("libprimesieve iterator failure");
    uint64_t p=iterator_.primes[iterator_.i];if(p==UINT64_MAX||p<next_)throw std::runtime_error("prime iterator order failure");
    if(p>end_){exhausted_=true;break;}out.push_back(p);next_=p+1;
   }
  }else if(generate){
   errno=0;auto data=static_cast<uint64_t*>(generate(count,next_,13)); // primesieve 12 UINT64_PRIMES enum
   if(!data||errno==EDOM){if(data)release(data);throw std::runtime_error("libprimesieve failed");}
   for(size_t i=0;i<count;++i){if(data[i]>end_){exhausted_=true;break;}if(data[i]<next_) {release(data);throw std::runtime_error("prime generator order failure");}out.push_back(data[i]);next_=data[i]+1;}release(data);
  }else while(out.size()<count){if(index_==cache_.size()){fill();if(cache_.empty()){if(segment_low_>end_){exhausted_=true;break;}continue;}}out.push_back(cache_[index_++]);next_=out.back()+1;}
  return out;
 }
};
// Bounded, ordered producer queue. Only consumed CUDA-verified batches may
// advance State::p; generated/prefetched ranges never advance a checkpoint.
class PrimePipeline{
 struct Region{std::vector<uint64_t> primes;};
 std::mutex mutex_;std::condition_variable ready_;bool cancel_=false;
 uint64_t first_,last_,span_,assigned_=0,consumed_=0,capacity_;
 std::map<uint64_t,Region> completed_;std::exception_ptr error_;
 std::vector<std::unique_ptr<Primes>> generators_;std::vector<std::thread> workers_;
 std::vector<uint64_t> current_;size_t index_=0;double producer_seconds_=0;
 bool cancelled(){std::lock_guard<std::mutex> lock(mutex_);return cancel_;}
 void worker(size_t id){try{
  while(true){uint64_t job,lo,hi;
   {std::unique_lock<std::mutex> lock(mutex_);ready_.wait(lock,[&]{return cancel_||assigned_<consumed_+capacity_;});if(cancel_)return;
    if(assigned_>(last_-first_)/span_)return;job=assigned_++;lo=first_+job*span_;hi=std::min(last_,lo+span_-1);}
   auto started=Clock::now();auto& generator=*generators_[id];generator.range(lo,hi);Region result;
   while(!cancelled()){auto values=generator.next(32768);if(values.empty())break;result.primes.insert(result.primes.end(),values.begin(),values.end());}
   {std::lock_guard<std::mutex> lock(mutex_);if(cancel_)return;producer_seconds_+=seconds(started);completed_.emplace(job,std::move(result));}ready_.notify_all();
  }
 }catch(...){std::lock_guard<std::mutex> lock(mutex_);if(!error_)error_=std::current_exception();cancel_=true;ready_.notify_all();}}
public:
 PrimePipeline(uint64_t first,uint64_t last,const std::string& library,size_t threads,uint64_t span):first_(first),last_(last),span_(span),capacity_(2*threads){
  if(first>last||!threads||!span)throw std::runtime_error("invalid prime pipeline range");
  for(size_t i=0;i<threads;++i)generators_.push_back(std::make_unique<Primes>(first,last,library,true,i!=0));
  try{for(size_t i=0;i<threads;++i)workers_.emplace_back([this,i]{worker(i);});}catch(...){stop();throw;}
 }
 void stop(){{std::lock_guard<std::mutex> lock(mutex_);cancel_=true;}ready_.notify_all();for(auto& t:workers_)if(t.joinable())t.join();}
 ~PrimePipeline(){stop();}
 std::vector<uint64_t> next(size_t count,const std::function<bool()>& should_stop){
  std::vector<uint64_t> result;result.reserve(count);
  while(result.size()<count){
   if(should_stop())return {};
   if(index_==current_.size()){
    if(consumed_>(last_-first_)/span_)break;
    std::unique_lock<std::mutex> lock(mutex_);
    while(!cancel_&&!completed_.count(consumed_)){ready_.wait_for(lock,std::chrono::milliseconds(50));if(should_stop())return {};}
    if(error_)std::rethrow_exception(error_);if(cancel_)return {};
    auto it=completed_.find(consumed_);current_=std::move(it->second.primes);index_=0;completed_.erase(it);++consumed_;lock.unlock();ready_.notify_all();
   }
   size_t take=std::min(count-result.size(),current_.size()-index_);
   result.insert(result.end(),current_.begin()+index_,current_.begin()+index_+take);index_+=take;
  }
  return result;
 }
 double producer_seconds(){std::lock_guard<std::mutex> lock(mutex_);return producer_seconds_;}
};
struct TuneChoice{std::string mode="iterator";uint64_t batch=32768,threads=1;};
struct SupplierPolicy{
 bool frozen=false,probing=false;uint64_t previous_threads=0;double previous_rate=0;
 uint64_t observe(uint64_t current,uint64_t cap,double rate,double wait_fraction){
  if(frozen)return current;
  if(!std::isfinite(rate)||rate<=0||!std::isfinite(wait_fraction)){frozen=true;return current;}
  if(probing&&rate<previous_rate*1.03){frozen=true;return previous_threads;}
  if(wait_fraction<0.08||current>=cap){frozen=true;return current;}
  previous_rate=rate;previous_threads=current;probing=true;return std::min(cap,current*2);
 }
};
int supplier_selftest(){
 auto require=[](bool condition){if(!condition)throw std::runtime_error("supplier policy selftest failed");};
 SupplierPolicy saturated;require(saturated.observe(1,8,100,0.01)==1&&saturated.frozen);
 SupplierPolicy reject;require(reject.observe(1,8,100,0.2)==2);require(reject.observe(2,8,102,0.2)==1&&reject.frozen);require(reject.observe(1,8,200,0.9)==1);
 SupplierPolicy accept;require(accept.observe(1,8,100,0.2)==2);require(accept.observe(2,8,140,0.2)==4);require(accept.observe(4,8,141,0.2)==2&&accept.frozen);
 SupplierPolicy filled;require(filled.observe(1,8,100,0.2)==2);require(filled.observe(2,8,140,0.01)==2&&filled.frozen);
 SupplierPolicy cap;require(cap.observe(4,7,100,0.2)==7);require(cap.observe(7,7,140,0.2)==7&&cap.frozen);
 SupplierPolicy invalid;require(invalid.observe(1,8,0,0.9)==1&&invalid.frozen);
 std::cout<<"SUPPLIER_SELFTEST_PASS checks=12 (starvation,3-percent hysteresis,rollback,freeze,memory cap)\n";return 0;
}
uint64_t available_memory(){
#ifdef _WIN32
 MEMORYSTATUSEX status{};status.dwLength=sizeof(status);if(GlobalMemoryStatusEx(&status))return status.ullAvailPhys;
#else
 struct sysinfo status{};if(sysinfo(&status)==0)return (uint64_t(status.freeram)+status.bufferram)*status.mem_unit;
#endif
 return 512ull*1024*1024;
}
uint64_t automatic_thread_cap(uint64_t p){
 const double root=std::sqrt(double(p));
 // Bound per-iterator base-prime storage plus bounded region/prefetch buffers.
 const double base=root>100?8*root/std::max(1.,std::log(root)-1.1):4096;
 const uint64_t per_thread=uint64_t(base*1.5)+64ull*1024*1024;
 const uint64_t budget=std::min<uint64_t>(512ull*1024*1024,available_memory()/4);
 return std::max<uint64_t>(1,std::min<uint64_t>({8,std::max(1u,std::thread::hardware_concurrency()),budget/std::max<uint64_t>(1,per_thread)}));
}
std::string tune_key(const Options& o,const State& s,const gfpdiv_cuda::Backend& gpu){
 unsigned bucket=0;for(uint64_t p=s.p;p>>=1;)++bucket;
 unsigned density=0;for(uint64_t count=s.count();count>>=1;)++density;
 std::string library=o.prime_library;if(library.empty())if(auto value=std::getenv("GFPDIVSV_PRIMESIEVE_LIBRARY"))library=value;
 Primes probe(s.p+1,o.pmax,library,true,true);
 return hash("GFPDIVSV1.1-tune1:"+gpu.tuning_identity()+":"+s.base_sha+":"+std::string(1,s.family)+":"+std::to_string(s.low)+":"+std::to_string(s.high)+":"+std::to_string(s.signs)+":"+std::to_string(density)+":"+o.gpu_product+":"+std::to_string(bucket)+":"+std::to_string(o.prime_span)+":"+std::to_string(std::thread::hardware_concurrency())+":"+probe.identity()+":"+std::to_string(o.batch_set?o.gpu_batch:0)+":"+std::to_string(o.threads_set?o.prime_threads:0));
}
bool load_tune(const fs::path& file,const std::string& key,TuneChoice& choice){
 try{
  if(!fs::is_regular_file(file)||fs::file_size(file)>4096)return false;
  std::ifstream in(file,std::ios::binary);std::string text((std::istreambuf_iterator<char>(in)),{});
  auto footer=text.rfind("sha256 ");if(footer==std::string::npos||text.size()!=footer+72||hash(text.substr(0,footer))!=text.substr(footer+7,64))return false;
  std::istringstream fields(text.substr(0,footer));std::string magic,version,label,saved,extra;
  TuneChoice candidate;
  if(!(fields>>magic>>version)||magic!="GFPDIVTUNE"||version!="1")return false;
  if(!(fields>>label>>saved)||label!="key"||saved!=key)return false;
  if(!(fields>>label>>candidate.mode)||label!="mode"||(candidate.mode!="iterator"&&candidate.mode!="pipeline"))return false;
  if(!(fields>>label>>candidate.batch)||label!="batch"||candidate.batch<16||candidate.batch>1048576)return false;
  if(!(fields>>label>>candidate.threads)||label!="threads"||candidate.threads<1||candidate.threads>16||(candidate.mode=="iterator"&&candidate.threads!=1)||(fields>>extra))return false;
  choice=candidate;return true;
 }catch(...){return false;}
}
void store_tune(const fs::path& file,const std::string& key,const TuneChoice& choice){
 try{
  if(fs::exists(file)){std::ifstream existing(file,std::ios::binary);std::string header;std::getline(existing,header);if(header!="GFPDIVTUNE 1"){std::cerr<<"warning: unrelated tuning-cache file preserved\n";return;}}
  std::string body="GFPDIVTUNE 1\nkey "+key+"\nmode "+choice.mode+"\nbatch "+std::to_string(choice.batch)+"\nthreads "+std::to_string(choice.threads)+"\n";
  atomic_write(file,body+"sha256 "+hash(body)+"\n");
 }catch(const std::exception& error){std::cerr<<"warning: tuning cache not saved: "<<error.what()<<"\n";}
}
TuneChoice calibrate(const Options& o,const State& s,const Integer& a,const cpp_int& exact,gfpdiv_cuda::Backend& gpu,const std::function<bool()>& stop){
 TuneChoice choice;choice.batch=o.gpu_batch;choice.threads=o.prime_threads;
 if(stop())return choice;
 Primes generator(s.p+1,o.pmax,o.prime_library,true,true);
 auto began=Clock::now();auto pool=generator.next(524288);const double generation=seconds(began);
 if(stop()||pool.size()<4096){choice.mode=o.threads_set?"pipeline":"iterator";return choice;}
 std::vector<uint64_t> warm(pool.begin(),pool.begin()+std::min<size_t>(32768,pool.size())),ignored;
 if(!gpu.remainders(warm,ignored,stop))return choice;
 std::vector<uint64_t> candidates=o.batch_set?std::vector<uint64_t>{o.gpu_batch}:std::vector<uint64_t>{32768,131072,262144,524288};
 std::map<uint64_t,std::vector<double>> timings;std::vector<uint64_t> baseline;std::vector<uint8_t> baseline_mask;
 auto tuning_started=Clock::now();
 for(int repetition=0;repetition<2;++repetition){
  auto order=candidates;if(repetition)std::reverse(order.begin(),order.end());
  for(uint64_t batch_size:order){
   if(stop())return choice;
   if(!timings.empty()&&seconds(tuning_started)>12)break;
   State trial=s;std::ofstream no_log;std::vector<uint64_t> all;all.reserve(pool.size());auto trial_started=Clock::now();
   for(size_t offset=0;offset<pool.size();offset+=batch_size){
    std::vector<uint64_t> batch(pool.begin()+offset,pool.begin()+std::min<size_t>(pool.size(),offset+batch_size)),residues;
    if(!gpu.remainders(batch,residues,stop))return choice;
    std::vector<Hit> hits;
    for(size_t i=0;i<batch.size();++i){const uint64_t p=batch[i],minus=residues[i],plus=minus?p-minus:0;
     if(minus>=p)throw std::runtime_error("autotune residue outside canonical range");
     if(minus>=trial.low&&minus<=trial.high&&(trial.mask[minus]&1))hits.push_back({uint32_t(minus),-1,p});
     if(plus>=trial.low&&plus<=trial.high&&(trial.mask[plus]&2))hits.push_back({uint32_t(plus),1,p});
    }
    verify_gpu_hits(trial,a,hits);apply(trial,exact,hits,no_log);all.insert(all.end(),residues.begin(),residues.end());
   }
   const double elapsed=seconds(trial_started);timings[batch_size].push_back(elapsed);
   if(baseline.empty()){baseline=std::move(all);baseline_mask=std::move(trial.mask);}
   else if(all!=baseline||trial.mask!=baseline_mask)throw std::runtime_error("autotune batch correctness mismatch");
   std::cout<<"autotune_trial: batch="<<batch_size<<", primes="<<pool.size()<<", verified_wall_s="<<elapsed<<"\n";
  }
 }
 double best=1e300;std::map<uint64_t,double> scores;
 for(const auto& row:timings){double score=std::accumulate(row.second.begin(),row.second.end(),0.)/row.second.size();scores[row.first]=score;best=std::min(best,score);}
 for(auto row:scores)if(row.second<=best/0.97){choice.batch=row.first;break;}
 choice.mode=(o.threads_set||generation>=0.03*best)?"pipeline":"iterator";
 std::cout<<"autotune_selected: batch="<<choice.batch<<", mode="<<choice.mode<<", threads="<<choice.threads<<", calibration_s="<<seconds(tuning_started)<<", prime_sample_s="<<generation<<"\n";
 return choice;
}
int run(int argc,char**argv){
 if(argc==2&&std::string(argv[1])=="--selftest-supplier")return supplier_selftest();
 if(argc==2&&std::string(argv[1])=="--selftest-primes"){
  size_t tests=0;
  for(auto range:std::vector<std::pair<uint64_t,uint64_t>>{{3,3},{4,4},{113,126},{114,126},{127,127},{1000000,1010000},{10000000000ull,10000050000ull},{1547695832912ull,1547695932911ull}}){
   Primes baseline(range.first,range.second,"",false,true);std::vector<uint64_t> expected;
   while(true){auto values=baseline.next(257);if(values.empty())break;expected.insert(expected.end(),values.begin(),values.end());}
   for(size_t threads:{0u,1u,2u,4u}){
    std::vector<uint64_t> actual;
    if(!threads){Primes generator(range.first,range.second,"",true,true);while(true){auto values=generator.next(511);if(values.empty())break;actual.insert(actual.end(),values.begin(),values.end());}}
    else{PrimePipeline generator(range.first,range.second,"",threads,1009);while(true){auto values=generator.next(511,[]{return false;});if(values.empty())break;actual.insert(actual.end(),values.begin(),values.end());}}
    if(actual!=expected)throw std::runtime_error("prime sequence selftest mismatch");++tests;
   }
  }
  std::cout<<"prime_sequence_selftest=PASS, cases="<<tests<<"\n";return 0;
 }
 if(argc==2&&(std::string(argv[1])=="--selftest"||std::string(argv[1])=="--selftest-cuda"))return gfpdiv_cuda::selftest();
 auto o=options(argc,argv);State s;
 if(!o.input.empty()){
  s=read_state(fs::u8path(o.input));if((o.n_set&&o.n!=s.n)||(o.family_set&&o.family!=s.family)||(o.sign_set&&o.signs!=s.signs)||(o.bounds_set&&(o.low!=s.low||(o.high&&o.high!=s.high))))throw std::runtime_error("resume parameters do not match saved search");
  if(o.output.empty())o.output=o.input;
 }else{if(o.n<3||o.n>10000000)throw std::runtime_error("requires 3<=n<=10000000");s.n=o.n;s.family=o.family;s.low=o.low;s.high=o.high?o.high:o.n-1;s.signs=o.signs;s.mask.resize(size_t(s.n)+1);auto domain=gfpps_model::divisors(s.n,s.family,s.low,s.high);for(auto t:domain)s.mask[t]=s.signs;std::cout<<"divisor_positions="<<domain.size()<<", constructed_candidates="<<s.count()<<"\n";}
 if(o.pmax<s.n||o.pmax<s.p)throw std::runtime_error("pmax must be at least n and the saved tested depth");
 auto output=fs::absolute(fs::u8path(o.output)).lexically_normal();
 if(fs::exists(output)&&(o.input.empty()||!fs::equivalent(output,fs::u8path(o.input))))throw std::runtime_error("output exists; explicitly resume that file or select a new filename");
 if(!o.factors.empty()&&(fs::absolute(fs::u8path(o.factors)).lexically_normal()==output||(!o.input.empty()&&fs::absolute(fs::u8path(o.factors)).lexically_normal()==fs::absolute(fs::u8path(o.input)).lexically_normal())))throw std::runtime_error("factor log must not overwrite a candidate file");
 if(o.prime_mode=="auto"){
  const fs::path profile=o.tuning_cache.empty()?fs::u8path(o.output+".tuning"):fs::u8path(o.tuning_cache);
  auto overlap=[](const fs::path& x,const fs::path& y){return fs::absolute(x).lexically_normal()==fs::absolute(y).lexically_normal()||(fs::exists(x)&&fs::exists(y)&&fs::equivalent(x,y));};
  for(const auto& name:std::vector<std::string>{o.output,o.input,o.factors,o.stop_file})if(!name.empty()&&overlap(profile,fs::u8path(name)))throw std::runtime_error("tuning cache must not overlap candidates, factor logs or stop files");
 }
 std::ofstream log;if(!o.factors.empty()){log.open(fs::u8path(o.factors),std::ios::app);if(!log)throw std::runtime_error("cannot open factor log");}
 auto start=Clock::now();std::cout<<"GFPDIVSV 1.1 n="<<s.n<<s.family<<", divisors="<<s.low<<".."<<s.high<<", signs="<<s.signs<<"\nconstructing fixed product...\n";
 cpp_int exact;Integer a;
#ifdef GFPDIVSV_USE_GMP
 // GMP has tuned factorial/primorial product construction (including FFT at
 // large sizes). Do not first construct a multi-million-bit cpp_int tree.
 if(s.family=='!'&&std::lgamma((long double)s.n+1)/std::log(2.L)>MAX_BITS+1024)throw std::runtime_error("factorial exceeds supported bit limit");
 if(s.family=='!')mpz_fac_ui(a.backend().data(),s.n);else mpz_primorial_ui(a.backend().data(),s.n);
 if(boost::multiprecision::msb(a)+1>MAX_BITS)throw std::runtime_error("base product exceeds supported bit limit");
 std::vector<uint8_t> bytes((boost::multiprecision::msb(a)+8)/8);size_t used=0;mpz_export(bytes.data(),&used,-1,1,-1,0,a.backend().data());
 boost::multiprecision::import_bits(exact,bytes.begin(),bytes.begin()+used,8,false);
#else
 exact=gfpps_model::base_product(s.n,s.family,MAX_BITS,stopped);a=exact;
#endif
 auto sha=base_hash(exact);if(!s.base_sha.empty()&&s.base_sha!=sha)throw std::runtime_error("base product checksum mismatch");s.base_sha=sha;
 std::cout<<"construction_s="<<seconds(start)<<"\n";
 uint64_t removed=0;
 if(o.input.empty()){std::cout<<"checking primes cancelled by each divisor...\n";auto hits=cancellation_checks(s,a);removed+=apply(s,exact,hits,log);s.p=s.n;}
 auto save=[&](){if(log.is_open()){log.flush();if(!log)throw std::runtime_error("factor log flush failed");}atomic_write(output,s.encode());std::cout<<"checkpoint: saved p="<<s.p<<", survivors="<<s.count()<<", file="<<o.output<<"\n";};
 save();auto last_save=Clock::now(),last_report=Clock::now();
 auto gpu=std::make_unique<gfpdiv_cuda::Backend>(product_groups(s,a,o.gpu_product=="raw"),int(o.device),int(o.gpu_chains),uint32_t(o.gpu_tile));
 std::cout<<"host_integer_backend="
#ifdef GFPDIVSV_USE_GMP
 <<"GMP exact construction/verification"
#else
 <<"Boost exact construction/verification"
#endif
 <<", sieve_backend=CUDA, batch_primes="<<(o.prime_mode=="auto"&&!o.batch_set?"auto":std::to_string(o.gpu_batch))<<", base_bits="<<boost::multiprecision::msb(exact)+1<<"\n";
 uint64_t tested=0,batches=0;bool complete=s.p==o.pmax;
 auto should_stop=[&](){return stopped()||(!o.stop_file.empty()&&fs::exists(fs::u8path(o.stop_file)));};
 const bool automatic=o.prime_mode=="auto";
 const fs::path cache=o.tuning_cache.empty()?fs::u8path(o.output+".tuning"):fs::u8path(o.tuning_cache);
 auto same_file=[](const fs::path& x,const fs::path& y){return fs::absolute(x).lexically_normal()==fs::absolute(y).lexically_normal()||(fs::exists(x)&&fs::exists(y)&&fs::equivalent(x,y));};
 if(automatic&&(same_file(cache,output)||(!o.input.empty()&&same_file(cache,fs::u8path(o.input)))||(!o.factors.empty()&&same_file(cache,fs::u8path(o.factors)))||(!o.stop_file.empty()&&same_file(cache,fs::u8path(o.stop_file)))))throw std::runtime_error("tuning cache must not overlap candidates, factor logs or stop files");
 const std::string key=automatic&&!complete&&!should_stop()?tune_key(o,s,*gpu):std::string();
 const uint64_t thread_cap=automatic_thread_cap(s.p+1);
 if(automatic&&!complete&&!should_stop()){
  TuneChoice choice;bool cached=!o.retune&&load_tune(cache,key,choice);
  if(!cached&&!o.retune&&!o.input.empty())cached=load_tune(fs::u8path(o.input+".tuning"),key,choice);
  if(!cached)choice=calibrate(o,s,a,exact,*gpu,should_stop);
  if(o.batch_set)choice.batch=o.gpu_batch;if(o.threads_set){choice.threads=o.prime_threads;choice.mode="pipeline";}else choice.threads=std::min(choice.threads,thread_cap);
  o.prime_mode=choice.mode;o.gpu_batch=choice.batch;o.prime_threads=choice.threads;
  if(!should_stop())store_tune(cache,key,choice);
  std::cout<<"autotune_cache="<<(cached?"hit":"miss")<<", mode="<<o.prime_mode<<", batch="<<o.gpu_batch<<", threads="<<o.prime_threads<<", automatic_thread_cap="<<thread_cap<<"\n";
 }
 gpu->reset_timing();
 std::unique_ptr<Primes> primes;std::unique_ptr<PrimePipeline> pipeline;
 if(!complete&&!should_stop()){if(o.prime_mode=="pipeline")pipeline=std::make_unique<PrimePipeline>(s.p+1,o.pmax,o.prime_library,size_t(o.prime_threads),o.prime_span);
 else primes=std::make_unique<Primes>(s.p+1,o.pmax,o.prime_library,o.prime_mode=="iterator");}
 std::cout<<"prime_supply="<<o.prime_mode<<", prime_threads="<<o.prime_threads<<", region_span="<<o.prime_span<<"\n";
 const auto sieve_start=Clock::now();double prime_seconds=0,compute_seconds=0,verify_seconds=0,checkpoint_seconds=0;
 auto window_started=Clock::now();uint64_t window_primes=0,window_batches=0;double window_wait=0;
 SupplierPolicy supplier;supplier.frozen=!automatic||o.threads_set||o.prime_mode!="pipeline";
 std::cout<<"preprocessing_s="<<seconds(start)<<"\n";
 while(!complete&&!should_stop()&&(!o.max_batches||batches<o.max_batches)){
  auto prime_started=Clock::now();auto batch=pipeline?pipeline->next(size_t(o.gpu_batch),should_stop):primes->next(size_t(o.gpu_batch));const double prime_wait=seconds(prime_started);prime_seconds+=prime_wait;window_wait+=prime_wait;if(should_stop())break;if(batch.empty()){s.p=o.pmax;complete=true;break;}
  auto compute_started=Clock::now();std::vector<Hit> hits;bool cancelled=false;
  {
   std::vector<uint64_t> residues;
   try{cancelled=!gpu->remainders(batch,residues,should_stop);}catch(...){save();throw;}
   if(!cancelled){
    if(o.verify_cuda){auto expected=remainders(a,batch);if(residues!=expected)throw std::runtime_error("CPU/GPU full-batch residue mismatch");}
    for(size_t j=0;j<batch.size();++j){auto p=batch[j],minus=residues[j],plus=minus?p-minus:0;
     if(minus>=p)throw std::runtime_error("CUDA residue outside canonical range");
     if(minus>=s.low&&minus<=s.high&&(s.mask[minus]&1))hits.push_back({uint32_t(minus),-1,p});
     if(plus>=s.low&&plus<=s.high&&(s.mask[plus]&2))hits.push_back({uint32_t(plus),1,p});
    }
   }
  }
  compute_seconds+=seconds(compute_started);
  if(cancelled)break; // No partial batch is committed; resume replays it in full.
  auto verify_started=Clock::now();
  verify_gpu_hits(s,a,hits);
  removed+=apply(s,exact,hits,log);s.p=batch.back();tested+=batch.size();++batches;
  if(s.p==o.pmax)complete=true;
  verify_seconds+=seconds(verify_started);
  window_primes+=batch.size();++window_batches;
  if(!complete&&!supplier.frozen&&window_batches>=8&&seconds(window_started)>=3){
   const double window_wall=seconds(window_started),rate=window_primes/window_wall,wait_fraction=window_wait/window_wall;
   const uint64_t next_threads=supplier.observe(o.prime_threads,thread_cap,rate,wait_fraction);
   std::cout<<"supplier_probe: threads="<<o.prime_threads<<", primes_per_s="<<rate<<", wait_fraction="<<wait_fraction<<", next="<<next_threads<<", frozen="<<supplier.frozen<<"\n";
   if(next_threads!=o.prime_threads){save();pipeline.reset();o.prime_threads=next_threads;pipeline=std::make_unique<PrimePipeline>(s.p+1,o.pmax,o.prime_library,size_t(o.prime_threads),o.prime_span);}
   if(supplier.frozen)store_tune(cache,key,TuneChoice{o.prime_mode,o.gpu_batch,o.prime_threads});
   window_started=Clock::now();window_primes=0;window_batches=0;window_wait=0;
  }
  if(seconds(last_report)>=o.progress){std::cout<<"progress: p="<<s.p<<", primes="<<tested<<", survivors="<<s.count()<<", removed="<<removed<<", primes_per_s="<<tested/seconds(start)<<", elapsed_s="<<seconds(start)<<"\n";last_report=Clock::now();}
  if(seconds(last_save)>=o.checkpoint){auto t=Clock::now();save();checkpoint_seconds+=seconds(t);last_save=Clock::now();}
 }
 if(complete)s.p=o.pmax;save();std::cout<<"done: p="<<s.p<<", survivors="<<s.count()<<", removed="<<removed<<", primes="<<tested<<", elapsed_s="<<seconds(start)<<", result="<<(complete?"COMPLETE":should_stop()?"INTERRUPTED":"PARTIAL")<<"\n";
 std::cout<<"timing: sieve_wall_s="<<seconds(sieve_start)<<", prime_generation_s="<<prime_seconds<<", compute_wall_s="<<compute_seconds<<", factor_verification_s="<<verify_seconds<<", checkpoint_s="<<checkpoint_seconds;
 std::cout<<", cuda_stream_s="<<gpu->stream_seconds();
 if(pipeline){pipeline->stop();std::cout<<", producer_worker_wall_sum_s="<<pipeline->producer_seconds();}
 std::cout<<"\n";
 return should_stop()&&!complete?130:0;
}
}
int main(int argc,char**argv){std::cout.setf(std::ios::unitbuf);std::signal(SIGINT,stop_signal);std::signal(SIGTERM,stop_signal);
#ifdef _WIN32
SetConsoleCtrlHandler(console_stop,TRUE);
#endif
try{return run(argc,argv);}catch(const std::exception& error){std::cerr<<"error: "<<error.what()<<"\n";return stopped()?130:1;}}
