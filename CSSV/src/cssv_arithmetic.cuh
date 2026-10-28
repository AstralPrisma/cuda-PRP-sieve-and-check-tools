// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "cssv_common.hpp"
#include <algorithm>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CSSV_HD __host__ __device__
#else
#define CSSV_HD
#endif
namespace cssv {
struct Mod {u64 p,ninv,r2,one;};
struct Entry {u64 index,x,minus_value,plus_value;unsigned mask;};
CSSV_HD inline u64 hd_high(u64 a,u64 b) {
#if defined(__CUDA_ARCH__)
 return __umul64hi(a,b);
#elif defined(_MSC_VER)
 u64 hi;_umul128(a,b,&hi);return hi;
#else
 return u64((static_cast<unsigned __int128>(a)*b)>>64);
#endif
}
CSSV_HD inline u64 hd_mul(u64 a,u64 b,const Mod& m) {
 if(m.p<=UINT32_MAX)return a*b%m.p;
 // a,b<p<2^62: REDC quotient <2*p<2^63, high-word sum cannot overflow.
 u64 lo=a*b,hi=hd_high(a,b),t=lo*m.ninv,mlo=t*m.p,mhi=hd_high(t,m.p);
 u64 out=hi+mhi+u64(lo+mlo<lo);return out>=m.p?out-m.p:out;
}
inline Mod make_mod(u64 p) {
 if(p<=UINT32_MAX)return {p,0,0,1};
 u64 inv=1;for(int i=0;i<6;++i)inv*=2-p*inv;
 u64 one=(u64(0)-p)%p,r2=one;for(int i=0;i<64;++i)r2=r2>=p-r2?r2-(p-r2):r2+r2;
 return {p,u64(0)-inv,r2,one};
}
CSSV_HD inline u64 hd_power(u64 a,u64 exp,const Mod& m) {
 a%=m.p;if(m.p>UINT32_MAX)a=hd_mul(a,m.r2,m);
 u64 value=m.one;while(exp){if(exp&1)value=hd_mul(value,a,m);exp>>=1;if(exp)a=hd_mul(a,a,m);}return value;
}
CSSV_HD inline u64 hd_term(u64 a,const Mod& m) {return a%m.p?hd_power(a,(a+1)%(m.p-1),m):0;}
CSSV_HD inline u64 hd_inverse_term(u64 a,const Mod& m) {u64 e=(a+1)%(m.p-1);return hd_power(a,e?m.p-1-e:0,m);}
CSSV_HD inline void hd_mark(u64* factors,u64 idx,u64 p) {
#ifdef __CUDA_ARCH__
 atomicMin(reinterpret_cast<unsigned long long*>(factors+idx),static_cast<unsigned long long>(p));
#else
 if(p<factors[idx])factors[idx]=p;
#endif
}
CSSV_HD inline void hd_match(const Config& c,const Entry& e,u64 product,const Mod& m,u64* factors) {
 const u64 base=e.index*(c.both?2:1);
 if((e.mask&1)&&product==m.one&&e.minus_value>m.p)hd_mark(factors,base,m.p);
 if((e.mask&2)&&product==m.p-m.one&&e.plus_value>m.p)hd_mark(factors,base+(c.both?1:0),m.p);
}
CSSV_HD inline void hd_scan(const Config& c,const Mod& m,const Entry* entries,u64 count,u64* factors) {
 if(m.p!=2&&c.terms>=m.p)return;
 if(c.reference||c.terms<8) {
  for(u64 i=0;i<count;++i){u64 product=m.one;for(u64 j=0;j<c.terms;++j){product=hd_mul(product,hd_term(entries[i].x+c.step*j,m),m);if(!product)break;}hd_match(c,entries[i],product,m,factors);}return;
 }
 // Separate step=2 lanes. Sparse gaps choose rebuild rather than walking
 // large eliminated intervals; dense consecutive starts retain sliding speed.
 for(unsigned lane=0;lane<c.step;++lane){
  bool initialized=false;u64 previous=0,product=m.one,zeros=0;
  for(u64 i=0;i<count;++i){const Entry& e=entries[i];if(e.x%c.step!=lane)continue;
   u64 gap=initialized?(e.x-previous)/c.step:NONE;
   if(!initialized||gap>=c.terms/2){product=m.one;zeros=0;for(u64 j=0;j<c.terms;++j){u64 v=hd_term(e.x+c.step*j,m);if(v)product=hd_mul(product,v,m);else ++zeros;}}
   else for(u64 j=0;j<gap;++j){u64 outgoing=previous+c.step*j,incoming=outgoing+c.step*c.terms;if(outgoing%m.p)product=hd_mul(product,hd_inverse_term(outgoing,m),m);else --zeros;u64 v=hd_term(incoming,m);if(v)product=hd_mul(product,v,m);else ++zeros;}
   if(!zeros)hd_match(c,e,product,m,factors);previous=e.x;initialized=true;
  }
 }
}
inline u64 capped_value(const Config& c,u64 x,int sign) {
 const u64 cap=MAX_P,limit=cap+1;
 auto multiply=[&](u64 a,u64 b){return b&&a>limit/b?limit+1:a*b;};
 u64 product=1;for(u64 i=0;i<c.terms;++i){u64 a=x+c.step*i,e=a+1,power=1;while(e){if(e&1){power=multiply(power,a);if(power>limit)return cap+1;}e>>=1;if(e)a=multiply(a,a);}product=multiply(product,power);if(product>limit)return cap+1;}
 u64 n=sign==1?product+1:product-1;return std::min(n,cap+1);
}
}
