/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */
#pragma once
// GFNSV CUDA checkpoint format v3. Standalone C++17, no CUDA dependency.
// State::pmin is the original inclusive lower factor bound; next_k is the
// first unprocessed k for p=k*2^(n+1)+1. A zero factor marks a survivor.
// Text is ASCII and the digest covers header/body with canonical LF endings.
// Reading a CRLF copy is supported; legacy/plain candidate files are rejected.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#endif

namespace gfnsv_state {
using U64 = unsigned long long;
inline constexpr U64 max_factor = (U64(1) << 62) - 1;
inline constexpr U64 max_slots = U64(64) * 1024 * 1024;
struct State {
    unsigned n = 16;
    U64 bmin = 0, bmax = 0, pmin = 3, pmax = 0, next_k = 0;
    bool expr = false;
    std::vector<U64> factors;
    // Compact checkpoints omit old factors. A masked factor value of 1 is
    // an already removed candidate whose factor is not available locally.
    std::vector<std::uint8_t> historical_removed;
};

inline bool is_historical_removed(const State& s,std::size_t index) {
    return s.historical_removed.size()==s.factors.size() && index<s.factors.size() &&
        s.historical_removed[index]==1 && s.factors[index]==1;
}

inline void check_n(unsigned n) {
    if (n < 1 || n > 20) throw std::runtime_error("checkpoint n must be in 1..20");
}
inline U64 first_even(const State& s) {
    // UINT64_MAX is odd; the empty [MAX,MAX] range has no first even base.
    return s.bmin == (std::numeric_limits<U64>::max)() ? 0 : s.bmin + (s.bmin & 1);
}
inline U64 slot_count(const State& s) {
    if (s.bmax < s.bmin) throw std::runtime_error("checkpoint base range is reversed");
    if (s.bmin == (std::numeric_limits<U64>::max)()) return 0;
    const U64 first = first_even(s);
    return first > s.bmax ? 0 : (s.bmax - first) / 2 + 1;
}
inline U64 first_k(U64 pmin, unsigned n) {
    check_n(n);
    const U64 q = U64(1) << (n + 1);
    return pmin <= 2 ? 1 : (pmin - 2) / q + 1;
}
inline U64 last_k(U64 pmax, unsigned n) {
    check_n(n);
    return pmax < 2 ? 0 : (pmax - 1) / (U64(1) << (n + 1));
}
inline U64 next_p(const State& s) {
    check_n(s.n);
    const U64 q = U64(1) << (s.n + 1);
    if (s.next_k > ((std::numeric_limits<U64>::max)() - 1) / q)
        throw std::runtime_error("checkpoint next factor overflow");
    return s.next_k * q + 1;
}
inline std::filesystem::path native_path(const std::string& s) {
    if (s.find('\0')!=std::string::npos) throw std::runtime_error("file path contains a NUL byte");
    // Callers supply UTF-8 (Windows main converts GetCommandLineW once).
    return std::filesystem::u8path(s);
}
inline bool same_path(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return false;
    const auto pa = native_path(a), pb = native_path(b);
    std::error_code ec;
    if (std::filesystem::equivalent(pa, pb, ec) && !ec) return true;
    const auto ca = std::filesystem::weakly_canonical(std::filesystem::absolute(pa));
    const auto cb = std::filesystem::weakly_canonical(std::filesystem::absolute(pb));
#ifdef _WIN32
    auto wa = ca.native(), wb = cb.native();
    std::transform(wa.begin(), wa.end(), wa.begin(), [](wchar_t c) { return std::towlower(c); });
    std::transform(wb.begin(), wb.end(), wb.begin(), [](wchar_t c) { return std::towlower(c); });
    return wa == wb;
#else
    return ca == cb;
#endif
}

namespace detail {
inline bool entry_exists(const std::filesystem::path& path) {
    std::error_code ec;
    const auto status=std::filesystem::symlink_status(path,ec);
    if (ec && ec!=std::errc::no_such_file_or_directory)
        throw std::filesystem::filesystem_error("cannot inspect output path",path,ec);
    return status.type()!=std::filesystem::file_type::not_found;
}
class Sha256 {
    std::array<std::uint32_t, 8> state_{{0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u}};
    std::array<unsigned char, 64> buffer_{};
    std::size_t used_ = 0;
    U64 bytes_ = 0;
    static std::uint32_t rotr(std::uint32_t v, unsigned n) { return (v >> n) | (v << (32 - n)); }
    void transform(const unsigned char* p) {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        std::uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i) w[i] = (std::uint32_t(p[4*i]) << 24) |
            (std::uint32_t(p[4*i+1]) << 16) | (std::uint32_t(p[4*i+2]) << 8) | p[4*i+3];
        for (unsigned i = 16; i < 64; ++i) w[i] = w[i-16] +
            (rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3)) + w[i-7] +
            (rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10));
        auto a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            const auto t1=h+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            const auto t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
        state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }
public:
    void update(const std::string& s) {
        const auto* p = reinterpret_cast<const unsigned char*>(s.data());
        std::size_t len = s.size();
        if (len > ((std::numeric_limits<U64>::max)() / 8) - bytes_)
            throw std::runtime_error("SHA-256 input too large");
        bytes_ += len;
        while (len) {
            const std::size_t take = (std::min)(len, std::size_t(64) - used_);
            std::copy(p, p+take, buffer_.begin()+used_);
            p+=take;len-=take;used_+=take;
            if (used_==64) { transform(buffer_.data()); used_=0; }
        }
    }
    std::string final_hex() {
        const U64 bits=bytes_*8;
        buffer_[used_++]=0x80;
        if (used_>56) { std::fill(buffer_.begin()+used_,buffer_.end(),0); transform(buffer_.data()); used_=0; }
        std::fill(buffer_.begin()+used_,buffer_.begin()+56,0);
        for (unsigned i=0;i<8;++i) buffer_[63-i]=static_cast<unsigned char>(bits>>(8*i));
        transform(buffer_.data());
        std::ostringstream out;
        out<<std::hex<<std::setfill('0');
        for (const auto v:state_) out<<std::setw(8)<<v;
        return out.str();
    }
};

inline U64 number(const std::string& s) {
    if (s.empty() || (s.size()>1 && s.front()=='0')) throw std::runtime_error("invalid canonical checkpoint integer");
    U64 value=0;
    for (const char c:s) {
        if (c<'0'||c>'9'||value>((std::numeric_limits<U64>::max)()-(c-'0'))/10)
            throw std::runtime_error("invalid or overflowing checkpoint integer");
        value=value*10+(c-'0');
    }
    return value;
}
inline std::vector<std::string> tokens(const std::string& line) {
    std::vector<std::string> result;
    std::size_t start=0;
    for (;;) {
        const auto end=line.find(' ',start);
        const auto token=line.substr(start,end==std::string::npos ? end : end-start);
        if (token.empty()) throw std::runtime_error("noncanonical checkpoint spacing");
        result.push_back(token);
        if (end==std::string::npos) return result;
        start=end+1;
    }
}
inline std::string field(const std::vector<std::string>& t, std::size_t pos, const std::string& name) {
    const std::string prefix=name+"=";
    if (pos>=t.size() || t[pos].compare(0,prefix.size(),prefix)!=0)
        throw std::runtime_error("missing or misplaced checkpoint field: "+name);
    return t[pos].substr(prefix.size());
}
inline bool line(std::istream& in, std::string& result) {
    // A bounded reader prevents corrupt files allocating an unbounded string.
    char storage[1026];
    in.getline(storage,sizeof storage);
    if (in.bad()) throw std::runtime_error("cannot read checkpoint file");
    if (in.fail() && !in.eof()) throw std::runtime_error("checkpoint line exceeds 1024 bytes");
    if (!in.gcount() && in.eof()) return false;
    const auto read_bytes=in.gcount();
    // gcount includes the delimiter only when it was consumed.
    const std::size_t len=std::size_t(read_bytes)-(in.eof()?0:1);
    result.assign(storage,len);
    if (!result.empty() && result.back()=='\r') result.pop_back();
    if (result.size()>1024) throw std::runtime_error("checkpoint line too long");
    for (const unsigned char c:result)
        if (c<32 || c>126) throw std::runtime_error("checkpoint must contain ASCII text");
    return true;
}
inline void validate_metadata(const State& s) {
    check_n(s.n);
    if (s.bmin<2 || s.bmax<s.bmin) throw std::runtime_error("invalid checkpoint base range");
    if (s.pmin<3 || s.pmax<s.pmin || s.pmax>max_factor)
        throw std::runtime_error("invalid checkpoint factor range");
    if (slot_count(s)>max_slots) throw std::runtime_error("checkpoint exceeds 64M candidate slots");
    if (s.next_k<first_k(s.pmin,s.n) || s.next_k>last_k(s.pmax,s.n)+1)
        throw std::runtime_error("checkpoint frontier is outside factor range");
    (void)next_p(s);
}
inline bool complete(const State& s) { return slot_count(s)==0 || s.next_k>last_k(s.pmax,s.n); }
inline void validate_factor(U64 p, const State& s) {
    if (!p) return;
    if (p<s.pmin || p>s.pmax || p>=next_p(s) || p%(U64(1)<<(s.n+1))!=1)
        throw std::runtime_error("checkpoint factor lies outside completed prefix");
}
inline U64 validate(const State& s) {
    validate_metadata(s);
    if (s.factors.size()!=slot_count(s)) throw std::runtime_error("checkpoint factor vector size mismatch");
    if (!s.historical_removed.empty() && s.historical_removed.size()!=s.factors.size())
        throw std::runtime_error("checkpoint historical removal mask size mismatch");
    U64 alive=0;
    for (std::size_t i=0;i<s.factors.size();++i) {
        const auto p=s.factors[i];
        if (!s.historical_removed.empty()) {
            if (s.historical_removed[i]>1) throw std::runtime_error("invalid historical removal mask bit");
            if (s.historical_removed[i] && !p)
                throw std::runtime_error("historically removed candidate was revived");
        }
        if (!is_historical_removed(s,i)) validate_factor(p,s);
        if (!p) ++alive;
    }
    return alive;
}
inline std::string header(const State& s,U64 alive) {
    return std::string(complete(s)?"#GFNSV_COMPLETE":"#GFNSV_STATE")+
        " version=3 engine=cuda n="+std::to_string(s.n)+" bmin="+std::to_string(s.bmin)+
        " bmax="+std::to_string(s.bmax)+" start_pmin="+std::to_string(s.pmin)+
        " pmin="+std::to_string(next_p(s))+" pmax="+std::to_string(s.pmax)+
        " next_k="+std::to_string(s.next_k)+" slots="+std::to_string(slot_count(s))+
        " alive_count="+std::to_string(alive)+" format="+(s.expr?"expr":"base");
}
inline std::string record(const State& s,U64 slot) {
    const U64 b=first_even(s)+2*slot, p=s.factors[std::size_t(slot)];
    if (p) return "#FACTOR "+std::to_string(b)+" "+std::to_string(p);
    return std::to_string(b)+(s.expr?"^"+std::to_string(U64(1)<<s.n)+"+1":"");
}
inline std::string factor_header(const State& s) {
    return "#GFNSV_FACTORS version=1 n="+std::to_string(s.n)+" bmin="+std::to_string(s.bmin)+
        " bmax="+std::to_string(s.bmax)+" start_pmin="+std::to_string(s.pmin);
}

// Stream into a unique, exclusively created sibling file. Only a durable,
// complete file is published. Replacement never deletes the previous state.
class AtomicWriter {
    std::filesystem::path path_,temp_;
    std::string pending_;
    bool durable_=false,installed_=false;
#ifdef _WIN32
    HANDLE fd_=INVALID_HANDLE_VALUE;
#else
    int fd_=-1;
#endif
    std::runtime_error error(const std::string& message) const {
#ifdef _WIN32
        const auto code=GetLastError();
#else
        const auto code=errno;
#endif
        return std::runtime_error(message+" (error "+std::to_string(code)+"): "+temp_.u8string());
    }
    void flush_buffer() {
        std::size_t done=0;
        while (done<pending_.size()) {
#ifdef _WIN32
            DWORD written=0;
            if (!WriteFile(fd_,pending_.data()+done,static_cast<DWORD>((std::min)(pending_.size()-done,std::size_t(1<<20))),&written,nullptr) || !written)
                throw error("cannot write checkpoint temporary file");
            done+=written;
#else
            const auto written=::write(fd_,pending_.data()+done,pending_.size()-done);
            if (written<0 && errno==EINTR) continue;
            if (written<=0) throw error("cannot write checkpoint temporary file");
            done+=std::size_t(written);
#endif
        }
        pending_.clear();
    }
public:
    explicit AtomicWriter(const std::string& path):path_(native_path(path)) {
        if (path.empty()) throw std::runtime_error("checkpoint output path is empty");
        static std::atomic<U64> sequence{0};
#ifdef _WIN32
        const U64 pid=GetCurrentProcessId();
#else
        const U64 pid=static_cast<U64>(getpid());
#endif
        const auto clock=std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned attempt=0;attempt<64;++attempt) {
            temp_=path_;
            temp_+=std::filesystem::u8path(".tmp."+std::to_string(pid)+"."+std::to_string(clock)+"."+std::to_string(sequence.fetch_add(1)));
#ifdef _WIN32
            fd_=CreateFileW(temp_.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
            if (fd_!=INVALID_HANDLE_VALUE) return;
            if (GetLastError()!=ERROR_FILE_EXISTS && GetLastError()!=ERROR_ALREADY_EXISTS) throw error("cannot create checkpoint temporary file");
#else
            fd_=::open(temp_.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
            if (fd_>=0) return;
            if (errno!=EEXIST) throw error("cannot create checkpoint temporary file");
#endif
        }
        throw std::runtime_error("cannot allocate unique checkpoint temporary filename");
    }
    AtomicWriter(const AtomicWriter&)=delete;
    AtomicWriter& operator=(const AtomicWriter&)=delete;
    ~AtomicWriter() {
#ifdef _WIN32
        if (fd_!=INVALID_HANDLE_VALUE) CloseHandle(fd_);
#else
        if (fd_>=0) ::close(fd_);
#endif
        if (!durable_&&!installed_) { std::error_code ignored; std::filesystem::remove(temp_,ignored); }
    }
    void append(const std::string& s) {
        pending_+=s;
        if (pending_.size()>=65536) flush_buffer();
    }
    void commit(bool allow_replace=true) {
        flush_buffer();
#ifdef _WIN32
        if (!FlushFileBuffers(fd_)) throw error("cannot flush checkpoint temporary file");
        durable_=true;
        if (!CloseHandle(fd_)) throw error("cannot close checkpoint temporary file");
        fd_=INVALID_HANDLE_VALUE;
        const DWORD flags=MOVEFILE_WRITE_THROUGH|(allow_replace?MOVEFILE_REPLACE_EXISTING:0);
        if (!MoveFileExW(temp_.c_str(),path_.c_str(),flags)) throw error("cannot install checkpoint; complete temporary file retained");
#else
        while (::fsync(fd_)!=0) { if (errno!=EINTR) throw error("cannot flush checkpoint temporary file"); }
        durable_=true;
        const int oldfd=fd_;fd_=-1;
        if (::close(oldfd)!=0) throw error("cannot close checkpoint temporary file");
        if (allow_replace) {
            if (::rename(temp_.c_str(),path_.c_str())!=0) throw error("cannot install checkpoint; complete temporary file retained");
        } else {
            // Linux renameat2 works on mounts (including WSL DrvFS) that
            // refuse hard links. Other POSIX systems use atomic link.
            bool renamed=false;
#if defined(__linux__) && defined(SYS_renameat2)
            if (::syscall(SYS_renameat2,AT_FDCWD,temp_.c_str(),AT_FDCWD,path_.c_str(),1U)==0) renamed=true;
            else if (errno!=ENOSYS && errno!=EINVAL && errno!=EOPNOTSUPP)
                throw error("cannot install new output; complete temporary file retained");
#endif
            if (!renamed) {
                if (::link(temp_.c_str(),path_.c_str())==0) {
                    if (::unlink(temp_.c_str())!=0) throw error("output installed but temporary link retained");
                } else {
                    if (errno!=EPERM && errno!=EOPNOTSUPP && errno!=ENOSYS && errno!=EINVAL)
                        throw error("cannot install new output; complete temporary file retained");
                    // FAT/exFAT mounts may support neither operation. An
                    // existing file is still refused; ordinary rename keeps
                    // publication atomic. As with exclusive ownership of the
                    // primary checkpoint, callers must not concurrently use
                    // this output path from an unrelated process.
                    if (entry_exists(path_)) {
                        errno=EEXIST;
                        throw error("output already exists; complete temporary file retained");
                    }
                    if (::rename(temp_.c_str(),path_.c_str())!=0)
                        throw error("cannot install new output; complete temporary file retained");
                }
            }
        }
#endif
        installed_=true;
#ifndef _WIN32
        auto parent=path_.parent_path();if (parent.empty()) parent=".";
        const int dirfd=::open(parent.c_str(),O_RDONLY|O_DIRECTORY);
        if (dirfd<0) throw error("checkpoint installed but cannot open parent directory for sync");
        int rc;do { rc=::fsync(dirfd); } while (rc!=0&&errno==EINTR);
        const int saved_errno=errno;::close(dirfd);errno=saved_errno;
        if (rc!=0) throw error("checkpoint installed but cannot sync parent directory");
#endif
    }
};
} // namespace detail

inline State read_state(const std::string& path) {
    const auto p=native_path(path);
    const auto file_bytes=std::filesystem::file_size(p);
    if (file_bytes<100 || file_bytes>max_slots*96+4096)
        throw std::runtime_error("checkpoint file size is outside supported bounds");
    std::ifstream in(p,std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint: "+path);
    std::string line;
    if (!detail::line(in,line)) throw std::runtime_error("checkpoint header is missing");
    const auto fields=detail::tokens(line);
    if (fields.size()!=13 || (fields[0]!="#GFNSV_STATE" && fields[0]!="#GFNSV_COMPLETE") ||
        fields[1]!="version=3" || fields[2]!="engine=cuda")
        throw std::runtime_error("resume requires a GFNSV CUDA v3 state; legacy/plain candidates are not resumable");
    State s;
    const U64 n=detail::number(detail::field(fields,3,"n"));
    if (n<1 || n>20) throw std::runtime_error("checkpoint n must be in 1..20");
    s.n=static_cast<unsigned>(n);
    s.bmin=detail::number(detail::field(fields,4,"bmin"));
    s.bmax=detail::number(detail::field(fields,5,"bmax"));
    s.pmin=detail::number(detail::field(fields,6,"start_pmin"));
    const U64 declared_next_p=detail::number(detail::field(fields,7,"pmin"));
    s.pmax=detail::number(detail::field(fields,8,"pmax"));
    s.next_k=detail::number(detail::field(fields,9,"next_k"));
    const U64 slots=detail::number(detail::field(fields,10,"slots"));
    const U64 alive=detail::number(detail::field(fields,11,"alive_count"));
    const auto format=detail::field(fields,12,"format");
    if (format!="base"&&format!="expr") throw std::runtime_error("invalid checkpoint format");
    s.expr=format=="expr";
    detail::validate_metadata(s);
    if (slots!=slot_count(s) || alive>slots || declared_next_p!=next_p(s) ||
        ((fields[0]=="#GFNSV_COMPLETE")!=detail::complete(s)) || line!=detail::header(s,alive))
        throw std::runtime_error("checkpoint metadata is inconsistent");
    // At least one digit and LF per slot, plus header/footer. Reject inflated
    // slot declarations before allocating the factor vector.
    if (file_bytes<slots*2+line.size()+80 || file_bytes>slots*96+4096)
        throw std::runtime_error("checkpoint file length disagrees with slot count");
    detail::Sha256 hash;hash.update(line+"\n");
    s.factors.resize(static_cast<std::size_t>(slots),0);
    U64 seen_alive=0;
    for (U64 i=0;i<slots;++i) {
        if (!detail::line(in,line)) throw std::runtime_error("checkpoint truncated before all candidate records");
        const U64 b=first_even(s)+2*i;
        if (line.compare(0,8,"#FACTOR ")==0) {
            const auto record=detail::tokens(line);
            if (record.size()!=3 || detail::number(record[1])!=b)
                throw std::runtime_error("checkpoint factor records are missing, duplicated or reordered");
            const U64 factor=detail::number(record[2]);
            if (!factor) throw std::runtime_error("checkpoint factor cannot be zero");
            detail::validate_factor(factor,s);
            s.factors[static_cast<std::size_t>(i)]=factor;
        } else ++seen_alive;
        if (line!=detail::record(s,i))
            throw std::runtime_error("checkpoint candidate records are missing, duplicated or reordered");
        hash.update(line+"\n");
    }
    if (!detail::line(in,line)) throw std::runtime_error("checkpoint footer is missing");
    const std::string expected="#GFNSV_END count="+std::to_string(slots)+" survivors="+
        std::to_string(alive)+" sha256="+hash.final_hex();
    if (seen_alive!=alive || line!=expected) throw std::runtime_error("checkpoint count or SHA-256 mismatch");
    if (detail::line(in,line)) throw std::runtime_error("extra content after checkpoint footer");
    return s;
}

inline void write_state_atomic(const std::string& path,const State& s,bool allow_replace=true) {
    const U64 alive=detail::validate(s);
    if (std::find(s.factors.begin(),s.factors.end(),U64(1))!=s.factors.end())
        throw std::runtime_error("CUDA v3 cannot store unknown historical factors; use compact v4 output");
    detail::AtomicWriter out(path);
    detail::Sha256 hash;
    auto put=[&](const std::string& line) { const auto text=line+"\n";out.append(text);hash.update(text); };
    put(detail::header(s,alive));
    for (U64 i=0;i<s.factors.size();++i) put(detail::record(s,i));
    out.append("#GFNSV_END count="+std::to_string(s.factors.size())+" survivors="+
        std::to_string(alive)+" sha256="+hash.final_hex()+"\n");
    out.commit(allow_replace);
}

inline void validate_factor_destination(const std::string& path,const State& s,bool allow_existing) {
    if (path.empty()) throw std::runtime_error("factor output path is empty");
    detail::validate_metadata(s);
    const auto p=native_path(path);
    const std::string expected=detail::factor_header(s);
    if (detail::entry_exists(p)) {
        if (!allow_existing) throw std::runtime_error("factor output already exists; choose a new path or resume its checkpoint: "+path);
        std::ifstream in(p,std::ios::binary);
        std::string first;
        if (!in || !detail::line(in,first) || first!=expected)
            throw std::runtime_error("existing factor output does not belong to this checkpoint: "+path);
    }
}

inline void write_factors_atomic(const std::string& path,const State& s,bool allow_existing) {
    (void)detail::validate(s);
    validate_factor_destination(path,s,allow_existing);
    const std::string expected=detail::factor_header(s);
    detail::AtomicWriter out(path);
    detail::Sha256 hash;
    auto put=[&](const std::string& line) { const auto text=line+"\n";out.append(text);hash.update(text); };
    put(expected);
    U64 count=0;
    for (U64 i=0;i<s.factors.size();++i) if (s.factors[static_cast<std::size_t>(i)]>1) {
        put(std::to_string(s.factors[static_cast<std::size_t>(i)])+" | "+
            std::to_string(first_even(s)+2*i)+"^"+std::to_string(U64(1)<<s.n)+"+1");
        ++count;
    }
    out.append("#GFNSV_FACTORS_END count="+std::to_string(count)+" sha256="+hash.final_hex()+"\n");
    out.commit(allow_existing);
}
} // namespace gfnsv_state
