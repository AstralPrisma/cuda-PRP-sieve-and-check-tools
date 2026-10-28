// SPDX-License-Identifier: GPL-2.0-or-later
// Optional host-only acceleration. Device kernels and checkpoint arithmetic do
// not depend on GMP. Linux builds link the system libgmp; portable builds do not.
#pragma once
#include "factorial_expression.hpp"
#include <gmp.h>
namespace gfpps_host {
using gfpps_model::cpp_int;
struct Integer{mpz_t z;Integer(){mpz_init(z);}Integer(const Integer&)=delete;~Integer(){mpz_clear(z);}};
inline void load(Integer& out,const cpp_int& n){std::vector<uint8_t> bytes;export_bits(n,std::back_inserter(bytes),8,false);mpz_import(out.z,bytes.size(),-1,1,-1,0,bytes.data());}
inline cpp_int read(const Integer& n){std::vector<uint8_t> bytes((mpz_sizeinbase(n.z,2)+7)/8);size_t used=0;mpz_export(bytes.data(),&used,-1,1,-1,0,n.z);cpp_int out=0;import_bits(out,bytes.begin(),bytes.begin()+used,8,false);return out;}
inline cpp_int construct(const gfpps_model::Expression& e,uint64_t max_bits,const gfpps_model::Cancel& stop){
 gfpps_model::validate_divisor(e);gfpps_model::check_cancel(stop);
 if(e.family=='!'&&e.n>2&&std::lgamma((long double)e.n+1)/std::log(2.L)>max_bits+1024)throw std::runtime_error("factorial exceeds supported bit limit");
 Integer a;if(e.family=='!')mpz_fac_ui(a.z,e.n);else mpz_primorial_ui(a.z,e.n);
 if(mpz_sizeinbase(a.z,2)>max_bits+64)throw std::runtime_error("base product exceeds supported bit limit");
 if(!mpz_divisible_ui_p(a.z,e.divisor))throw std::runtime_error("inexact factor removal");
 mpz_divexact_ui(a.z,a.z,e.divisor);gfpps_model::check_cancel(stop);
 cpp_int n=read(a)*e.k+e.c;if(n<2)throw std::runtime_error("candidate must be at least 2");
 if(boost::multiprecision::msb(n)+9>max_bits)throw std::runtime_error("candidate exceeds supported bit limit");return n;
}
inline cpp_int inverse_power2(const cpp_int& n,unsigned bits){Integer a,r,inv;load(a,n);mpz_setbit(r.z,bits);if(!mpz_invert(inv.z,a.z,r.z))throw std::runtime_error("Montgomery inverse does not exist");return read(inv);}
inline bool check_inverse(const cpp_int& n,const cpp_int& negative_inverse,unsigned bits){Integer a,b,result;load(a,n);load(b,negative_inverse);mpz_mul(result.z,a.z,b.z);mpz_add_ui(result.z,result.z,1);mpz_fdiv_r_2exp(result.z,result.z,bits);return mpz_sgn(result.z)==0;}
}
