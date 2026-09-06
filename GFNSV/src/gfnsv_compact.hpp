/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */
#pragma once
#include "gfnsv_state.hpp"

namespace gfnsv_compact {
using gfnsv_state::State;
using gfnsv_state::U64;
namespace sd = gfnsv_state::detail;

namespace detail {
inline bool marker(const std::string& line) { return line.rfind("#GFNSV_",0)==0; }
inline bool view(const std::string& format) { return format=="G" || format=="A"; }
inline void check_format(const std::string& format) {
    if (!view(format) && format!="base" && format!="expr")
        throw std::runtime_error("invalid compact checkpoint format");
}
inline U64 sieved_to(const State& s) {
    return (std::min)(s.pmax,gfnsv_state::next_p(s)-1);
}
inline std::string view_header(const State& s,const std::string& format) {
    return (format=="G" ? "GFN n="+std::to_string(s.n) :
        "ABC $a^"+std::to_string(U64(1)<<s.n)+"+1")+
        " // format=4 Sieved to "+std::to_string(sieved_to(s));
}
inline std::string metadata(const State& s,U64 alive,const std::string& format) {
    auto result=sd::header(s,alive);
    const auto version=result.find("version=3");
    result.replace(version,9,"version=4");
    result.resize(result.rfind(" format="));
    return result+" format="+format;
}
inline std::string candidate(U64 b,unsigned n,const std::string& format) {
    return std::to_string(b)+(format=="expr"?"^"+std::to_string(U64(1)<<n)+"+1":"");
}
inline U64 parse_candidate(const std::string& record,unsigned n,bool expr) {
    if (!expr) return sd::number(record);
    const std::string suffix="^"+std::to_string(U64(1)<<n)+"+1";
    if (record.size()<=suffix.size() || record.compare(record.size()-suffix.size(),suffix.size(),suffix)!=0)
        throw std::runtime_error("compact candidate expression has the wrong exponent or suffix");
    return sd::number(record.substr(0,record.size()-suffix.size()));
}
inline std::uintmax_t file_size(const std::filesystem::path& path) {
    const auto bytes=std::filesystem::file_size(path);
    if (bytes<100 || bytes>gfnsv_state::max_slots*96+4096)
        throw std::runtime_error("compact checkpoint file size is outside supported bounds");
    return bytes;
}
} // namespace detail

// Detection deliberately does not validate v4. Recognizable corrupt v4 must
// reach its strict reader rather than falling back to an older sidecar.
inline bool is_compact(const std::string& path) {
    std::ifstream in(gfnsv_state::native_path(path),std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint: "+path);
    std::string first,second;
    if (!sd::line(in,first)) return false;
    if (first.find("// format=")!=std::string::npos) return true;
    if (detail::marker(first)) {
        if (first.rfind("#GFNSV_STATE version=3 ",0)==0 ||
            first.rfind("#GFNSV_COMPLETE version=3 ",0)==0) return false;
        return true;
    }
    if (!sd::line(in,second)) return false;
    return detail::marker(second);
}

inline State read_state(const std::string& path,std::string* format_out=nullptr) {
    const auto native=gfnsv_state::native_path(path);
    const auto bytes=detail::file_size(native);
    std::ifstream in(native,std::ios::binary);
    if (!in) throw std::runtime_error("cannot open compact checkpoint: "+path);
    std::string record,view_header;
    if (!sd::line(in,record)) throw std::runtime_error("compact checkpoint header is missing");
    if (record.rfind("GFN ",0)==0 || record.rfind("ABC ",0)==0) {
        view_header=record;
        if (!sd::line(in,record)) throw std::runtime_error("compact checkpoint metadata is missing");
    }
    const auto fields=sd::tokens(record);
    if (fields.size()!=13 || (fields[0]!="#GFNSV_STATE" && fields[0]!="#GFNSV_COMPLETE") ||
        fields[1]!="version=4" || fields[2]!="engine=cuda")
        throw std::runtime_error("resume requires an intact GFNSV CUDA compact v4 checkpoint");
    State state;
    const U64 n=sd::number(sd::field(fields,3,"n"));
    if (n<1 || n>20) throw std::runtime_error("checkpoint n must be in 1..20");
    state.n=static_cast<unsigned>(n);
    state.bmin=sd::number(sd::field(fields,4,"bmin"));
    state.bmax=sd::number(sd::field(fields,5,"bmax"));
    state.pmin=sd::number(sd::field(fields,6,"start_pmin"));
    const U64 declared_next=sd::number(sd::field(fields,7,"pmin"));
    state.pmax=sd::number(sd::field(fields,8,"pmax"));
    state.next_k=sd::number(sd::field(fields,9,"next_k"));
    const U64 slots=sd::number(sd::field(fields,10,"slots"));
    const U64 alive=sd::number(sd::field(fields,11,"alive_count"));
    const std::string format=sd::field(fields,12,"format");
    detail::check_format(format);
    state.expr=format=="expr";
    sd::validate_metadata(state);
    if (slots!=gfnsv_state::slot_count(state) || alive>slots ||
        declared_next!=gfnsv_state::next_p(state) || record!=detail::metadata(state,alive,format))
        throw std::runtime_error("compact checkpoint metadata is inconsistent");
    if (detail::view(format) ? view_header!=detail::view_header(state,format) : !view_header.empty())
        throw std::runtime_error("compact candidate header disagrees with its metadata");
    if (bytes<alive*2+record.size()+view_header.size()+80 || bytes>alive*96+4096)
        throw std::runtime_error("compact file length disagrees with its survivor count");
    sd::Sha256 hash;
    if (!view_header.empty()) hash.update(view_header+"\n");
    hash.update(record+"\n");
    state.factors.assign(static_cast<std::size_t>(slots),1);
    state.historical_removed.assign(static_cast<std::size_t>(slots),1);
    const U64 first=gfnsv_state::first_even(state);
    U64 previous=0;
    for (U64 i=0;i<alive;++i) {
        if (!sd::line(in,record)) throw std::runtime_error("compact checkpoint truncated before all survivors");
        const U64 b=detail::parse_candidate(record,state.n,state.expr);
        if (b<state.bmin || b>state.bmax || (b&1) || b<=previous)
            throw std::runtime_error("compact survivors are out of range, odd, duplicated or unordered");
        const auto index=static_cast<std::size_t>((b-first)/2);
        state.factors[index]=0;
        state.historical_removed[index]=0;
        previous=b;
        hash.update(record+"\n");
    }
    if (!sd::line(in,record)) throw std::runtime_error("compact checkpoint footer is missing");
    if (record!="#GFNSV_END count="+std::to_string(alive)+" sha256="+hash.final_hex())
        throw std::runtime_error("compact checkpoint count or SHA-256 mismatch");
    if (sd::line(in,record)) throw std::runtime_error("extra content after compact checkpoint footer");
    (void)sd::validate(state);
    if (format_out) *format_out=format;
    return state;
}

inline void write_state_atomic(const std::string& path,const State& state,const std::string& format,
                               bool allow_replace) {
    detail::check_format(format);
    const U64 alive=sd::validate(state);
    sd::AtomicWriter out(path);
    sd::Sha256 hash;
    const auto put=[&](const std::string& line) { out.append(line+"\n");hash.update(line+"\n"); };
    if (detail::view(format)) put(detail::view_header(state,format));
    put(detail::metadata(state,alive,format));
    const U64 first=gfnsv_state::first_even(state);
    for (std::size_t i=0;i<state.factors.size();++i)
        if (!state.factors[i]) put(detail::candidate(first+2*static_cast<U64>(i),state.n,format));
    out.append("#GFNSV_END count="+std::to_string(alive)+" sha256="+hash.final_hex()+"\n");
    out.commit(allow_replace);
}

// Factor logs are optional supplements. Validate the complete log before
// changing any factor, and never let a log remove a current survivor.
inline void merge_factor_log(const std::string& path,State& state) {
    (void)sd::validate(state);
    const auto native=gfnsv_state::native_path(path);
    const auto bytes=std::filesystem::file_size(native);
    if (bytes<80 || bytes>gfnsv_state::max_slots*96+4096)
        throw std::runtime_error("factor log size is outside supported bounds");
    std::ifstream in(native,std::ios::binary);
    std::string record;
    if (!in || !sd::line(in,record) || record!=sd::factor_header(state))
        throw std::runtime_error("factor log does not match the checkpoint n/base range/start_pmin");
    sd::Sha256 hash;hash.update(record+"\n");
    auto merged=state.factors;
    U64 count=0,previous=0;
    bool ended=false;
    while (sd::line(in,record)) {
        if (record.rfind("#GFNSV_FACTORS_END",0)==0) { ended=true;break; }
        const auto fields=sd::tokens(record);
        if (fields.size()!=3 || fields[1]!="|") throw std::runtime_error("malformed factor log record");
        const U64 p=sd::number(fields[0]);
        const U64 b=detail::parse_candidate(fields[2],state.n,true);
        if (p<=1 || b<state.bmin || b>state.bmax || (b&1) || b<=previous)
            throw std::runtime_error("factor log records are invalid, out of range or unordered");
        sd::validate_factor(p,state);
        const auto index=static_cast<std::size_t>((b-gfnsv_state::first_even(state))/2);
        if (!state.factors[index]) throw std::runtime_error("factor log attempts to remove a surviving candidate");
        if (state.factors[index]!=1 && state.factors[index]!=p)
            throw std::runtime_error("factor log conflicts with a known checkpoint factor");
        merged[index]=p;
        previous=b;
        ++count;
        if (count>state.factors.size()) throw std::runtime_error("too many factor log records");
        hash.update(record+"\n");
    }
    if (!ended || record!="#GFNSV_FACTORS_END count="+std::to_string(count)+" sha256="+hash.final_hex())
        throw std::runtime_error("factor log is truncated or its count/SHA-256 is invalid");
    if (sd::line(in,record)) throw std::runtime_error("extra content after factor log footer");
    state.factors.swap(merged);
}
} // namespace gfnsv_compact
