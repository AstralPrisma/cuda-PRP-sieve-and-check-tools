// SPDX-License-Identifier: GPL-2.0-or-later
// GHCW family model, Copyright (C) 2026 AstralPrisma (A.P.).
#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <functional>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <cmath>

namespace ghcw_model {
using boost::multiprecision::cpp_int;
constexpr uint64_t parameter_max = UINT32_MAX;
constexpr uint64_t max_modulus_bits = (uint64_t(1)<<20)*15-16;
struct Expression {
    uint64_t b=0,n=0; int c=1;
    std::string text() const {
        return std::to_string(b)+"^"+std::to_string(n)+"*"+
               std::to_string(n)+"^"+std::to_string(b)+(c==1?"+1":"-1");
    }
};
inline uint64_t number(const std::string& value) {
    if(value.empty() || value.size()>20 || value.find_first_not_of("0123456789")!=std::string::npos)
        throw std::runtime_error("expected an unsigned decimal integer");
    size_t used=0; auto n=std::stoull(value,&used);
    if(used!=value.size()) throw std::runtime_error("invalid integer suffix");
    return n;
}
inline void validate(const Expression& e) {
    if(e.b<2 || e.n<2 || e.b>parameter_max || e.n>parameter_max || (e.c!=1 && e.c!=-1))
        throw std::runtime_error("requires 2 <= b,n <= 4294967295 and sign +/-1");
}
inline Expression parse(const std::string& text) {
    if(text.size()>256) throw std::runtime_error("expression is too long");
    static const std::regex pattern(R"(^\s*([0-9]+)\s*\^\s*([0-9]+)\s*\*\s*([0-9]+)\s*\^\s*([0-9]+)\s*([+-])\s*1\s*$)");
    std::smatch m;
    if(!std::regex_match(text,m,pattern)) throw std::runtime_error("expected b^n*n^b+/-1; quote the expression");
    Expression e{number(m[1]),number(m[2]),m[5]=="+"?1:-1};
    if(e.n!=number(m[3]) || e.b!=number(m[4])) throw std::runtime_error("crossed bases/exponents do not match");
    validate(e);return e;
}
inline std::string composite_reason(const Expression& e) {
    validate(e);
    if((e.b&1) && (e.n&1)) return "factor-2";
    auto d=std::gcd(e.b,e.n);
    if(e.c==-1 && d>1) return "difference-of-powers-gcd="+std::to_string(d);
    while(d && !(d&1))d>>=1;
    if(e.c==1 && d>1) return "sum-of-odd-powers-divisor="+std::to_string(d);
    return {};
}
inline cpp_int power(uint64_t base,uint64_t exponent,const std::function<bool()>& stopped={}) {
    cpp_int r=1,a=base;
    while(exponent) {
        if(stopped && stopped()) throw std::runtime_error("interrupted during modulus construction");
        if(exponent&1) r*=a;
        exponent>>=1;if(exponent)a*=a;
    }
    return r;
}
inline cpp_int modulus(const Expression& e,const std::function<bool()>& stopped={}) {
    validate(e);
    long double estimate=e.n*std::log2((long double)e.b)+e.b*std::log2((long double)e.n);
    if(estimate>max_modulus_bits-1) throw std::runtime_error("candidate exceeds supported bit limit");
    cpp_int value=power(e.b,e.n,stopped)*power(e.n,e.b,stopped)+e.c;
    if(boost::multiprecision::msb(value)+1>max_modulus_bits) throw std::runtime_error("candidate exceeds supported bit limit");
    return value;
}
}
