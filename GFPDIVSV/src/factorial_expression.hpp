// SPDX-License-Identifier: GPL-2.0-or-later
// Shared exact construction for GFPPS and the independent single-factor sieve.
#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
namespace gfpps_model {
using boost::multiprecision::cpp_int;
using Cancel=std::function<bool()>;
inline void check_cancel(const Cancel& stop){if(stop&&stop())throw std::runtime_error("interrupted during exact construction");}
inline uint64_t integer(const std::string& s){
 if(s.empty())throw std::runtime_error("empty integer");
 size_t start=s.size()>2&&s[0]=='0'&&(s[1]=='x'||s[1]=='X')?2:0;
 for(size_t i=start;i<s.size();++i)if(!(start?std::isxdigit((unsigned char)s[i]):std::isdigit((unsigned char)s[i])))throw std::runtime_error("invalid unsigned integer");
 if(start==s.size())throw std::runtime_error("empty integer");size_t used=0;auto n=std::stoull(s,&used,start?16:10);if(used!=s.size())throw std::runtime_error("invalid integer tail");return n;
}
struct Expression {
 uint64_t k=1,n=0;char family='!';int c=1;uint64_t divisor=1;
 std::string text()const{
  // Preserve the exact legacy spelling for checkpoint/result consumers.
  const auto prefix=divisor==1||k!=1?std::to_string(k)+"*":"";
  return prefix+std::to_string(n)+family+(divisor>1?"/"+std::to_string(divisor):"")+(c==1?"+1":"-1");
 }
};
inline Expression parse(const std::string& input){
 if(input.size()>512)throw std::runtime_error("expression too long");
 static const std::regex pattern(R"(^\s*(?:(0[xX][0-9a-fA-F]+|[0-9]+)\s*\*\s*)?(0[xX][0-9a-fA-F]+|[0-9]+)\s*([!#])(?:\s*/\s*(0[xX][0-9a-fA-F]+|[0-9]+))?\s*([+-])\s*1\s*$)");
 std::smatch m;if(!std::regex_match(input,m,pattern))throw std::runtime_error("expected k*n!+/-1, k*n#+/-1, n!/d+/-1 or n#/q+/-1");
 Expression e{m[1].matched?integer(m[1]):1,integer(m[2]),m[3].str()[0],m[5]=="+"?1:-1,m[4].matched?integer(m[4]):1};
 if(!e.k||e.n>10000000)throw std::runtime_error("requires k>=1 and n<=10000000");
 if(m[4].matched&&(e.divisor<2||e.divisor>=e.n))throw std::runtime_error("single-factor removal requires 2<=divisor<n");
 return e;
}
inline bool prime32(uint32_t p){if(p<2)return false;if(!(p&1))return p==2;for(uint32_t q=3;uint64_t(q)*q<=p;q+=2)if(p%q==0)return false;return true;}
inline std::vector<uint32_t> factors(uint64_t n,char family,const Cancel& stop={}){
 if(n>10000000||(family!='!'&&family!='#'))throw std::runtime_error("unsupported product bounds");
 std::vector<uint32_t> out;
 if(family=='!'){out.reserve(size_t(n));for(uint32_t i=2;i<=n;++i){if((i&65535)==0)check_cancel(stop);out.push_back(i);}}
 else{
  std::vector<bool> composite(size_t(n)+1,false);
  for(uint32_t i=2;i<=n;++i){if((i&65535)==0)check_cancel(stop);if(!composite[i]){out.push_back(i);if(uint64_t(i)*i<=n)for(uint64_t j=uint64_t(i)*i;j<=n;j+=i)composite[size_t(j)]=true;}}
 }return out;
}
inline cpp_int product_tree(const std::vector<uint32_t>& a,size_t lo,size_t hi,const Cancel& stop={}){
 check_cancel(stop);if(hi-lo<32){cpp_int v=1;for(size_t i=lo;i<hi;++i)v*=a[i];return v;}
 size_t mid=lo+(hi-lo)/2;cpp_int left=product_tree(a,lo,mid,stop),right=product_tree(a,mid,hi,stop);return left*right;
}
inline cpp_int base_product(uint64_t n,char family,uint64_t max_bits,const Cancel& stop={}){
 if(family=='!'&&n>2&&std::lgamma((long double)n+1)/std::log(2.L)>max_bits+1024)throw std::runtime_error("factorial exceeds supported bit limit");
 auto a=factors(n,family,stop);auto v=product_tree(a,0,a.size(),stop);
 if(boost::multiprecision::msb(v)+1>max_bits)throw std::runtime_error("base product exceeds supported bit limit");return v;
}
inline void validate_divisor(const Expression& e){
 if(e.divisor==1)return;
 if(e.divisor<2||e.divisor>=e.n)throw std::runtime_error("requires 2<=divisor<n");
 if(e.family=='#'){
  // q=2 is accepted for explicit diagnostic checks. The search sieve excludes
  // it because n#/2 +/- 1 is even (apart from tiny exceptional candidates).
  if(!prime32(uint32_t(e.divisor)))throw std::runtime_error("primorial removal requires a prime divisor");
  uint64_t largest=e.n;while(largest>=2&&!prime32(uint32_t(largest)))--largest;
  if(e.divisor>=largest)throw std::runtime_error("the largest primorial factor is excluded");
 }
}
inline cpp_int construct(const Expression& e,uint64_t max_bits,const Cancel& stop={}){
 validate_divisor(e);cpp_int a=base_product(e.n,e.family,max_bits+64,stop);
 if(a%e.divisor!=0)throw std::runtime_error("divisor does not divide the unscaled product exactly");
 cpp_int result=(a/e.divisor)*e.k+e.c;
 if(result<2)throw std::runtime_error("candidate must be at least 2");
 if(boost::multiprecision::msb(result)+9>max_bits)throw std::runtime_error("candidate exceeds supported bit limit");return result;
}
inline std::vector<uint32_t> divisors(uint64_t n,char family,uint64_t low,uint64_t high){
 if(n<3||n>10000000||low<2||high>=n||low>high)throw std::runtime_error("invalid divisor search bounds");
 std::vector<uint32_t> out;
 if(family=='!'){for(uint32_t t=uint32_t(low);t<=high;++t)out.push_back(t);}
 else if(family=='#'){auto p=factors(n,'#');for(auto q:p)if(q>=3&&q>=low&&q<=high&&q<p.back())out.push_back(q);}
 else throw std::runtime_error("unknown product family");return out;
}
}
