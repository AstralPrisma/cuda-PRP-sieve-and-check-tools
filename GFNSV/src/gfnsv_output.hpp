/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */
#pragma once
#include "gfnsv_state.hpp"
#include "gfnsv_compact.hpp"

namespace gfnsv_output {
inline bool exists(const std::string& path) {
    return gfnsv_state::detail::entry_exists(gfnsv_state::native_path(path));
}
inline std::string first_line(const std::string& path) {
    std::ifstream in(gfnsv_state::native_path(path),std::ios::binary);
    std::string line;
    if(!in||!gfnsv_state::detail::line(in,line))
        throw std::runtime_error("cannot read sieve input header: "+path);
    return line;
}
inline bool legacy_state(const std::string& path) {
    if(!exists(path))return false;
    const auto line=first_line(path);
    return line.rfind("#GFNSV_STATE ",0)==0||line.rfind("#GFNSV_COMPLETE ",0)==0;
}
inline std::string sidecar(const std::string& path) {return path+".sieve-state";}
inline bool is_view_format(const std::string& format) {return format=="G"||format=="A";}
inline std::string view_format(const std::string& path) {
    if(!exists(path))return "G";
    const auto line=first_line(path);
    if(line.rfind("GFN ",0)==0)return "G";
    if(line.rfind("ABC ",0)==0)return "A";
    return "";
}
inline std::string resume_state(const std::string& input) {
    if(exists(input)&&gfnsv_compact::is_compact(input))return input;
    if(legacy_state(input))return input;
    if(exists(input)&&view_format(input).empty())
        throw std::runtime_error("resume input is not a GFNSV GFN/ABC view or CUDA v3 state; plain PRP queues must not be restored in place");
    const auto checkpoint=sidecar(input);
    if(!exists(checkpoint))
        throw std::runtime_error("this old bare GFN/ABC needs its legacy "+checkpoint+" for one-time migration; new v4 GFN files resume by themselves");
    if(gfnsv_compact::is_compact(checkpoint))
        throw std::runtime_error("legacy bare view has a non-v3 companion; open the self-contained checkpoint directly instead");
    return checkpoint;
}
inline gfnsv_state::State read_resume_state(const std::string& path,std::string* format=nullptr) {
    if(gfnsv_compact::is_compact(path))return gfnsv_compact::read_state(path,format);
    auto state=gfnsv_state::read_state(path);
    if(format)*format=state.expr?"expr":"base";
    return state;
}
inline gfnsv_state::U64 sieved_to(const gfnsv_state::State& state) {
    return (std::min)(state.pmax,gfnsv_state::next_p(state)-1);
}
inline std::string view_prefix(unsigned n,const std::string& format) {
    gfnsv_state::check_n(n);
    if(format=="G")return "GFN n="+std::to_string(n)+" // Sieved to ";
    return "ABC $a^"+std::to_string(gfnsv_state::U64(1)<<n)+"+1 // Sieved to ";
}
inline void validate_view_identity(const std::string& path,const gfnsv_state::State& state) {
    if(!exists(path)||gfnsv_compact::is_compact(path)||legacy_state(path))return;
    const auto line=first_line(path),prefix=view_prefix(state.n,view_format(path));
    if(line.rfind(prefix,0)!=0)
        throw std::runtime_error("GFN/ABC header does not match the companion checkpoint's GFN index");
    const auto bound=gfnsv_state::detail::number(line.substr(prefix.size()));
    if(bound>sieved_to(state))
        throw std::runtime_error("GFN/ABC progress is ahead of its checkpoint; refusing inconsistent input");
    std::ifstream in(gfnsv_state::native_path(path),std::ios::binary);
    std::string record;
    if(!in||!gfnsv_state::detail::line(in,record))
        throw std::runtime_error("cannot reread GFN/ABC input");
    gfnsv_state::U64 previous=0;
    while(gfnsv_state::detail::line(in,record)) {
        const auto b=gfnsv_state::detail::number(record);
        if(b<state.bmin||b>state.bmax||(b&1)||b<=previous)
            throw std::runtime_error("GFN/ABC bases are outside the checkpoint range, odd, duplicated or unordered; recover from the state into a new output");
        previous=b;
    }
    // The SHA-256-verified state is authoritative. A stale, missing or
    // truncated candidate body is rebuilt from it, never used to infer removals.
}
} // namespace gfnsv_output
