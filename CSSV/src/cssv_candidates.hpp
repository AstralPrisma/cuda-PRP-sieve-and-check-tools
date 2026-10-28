// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "cssv_state.hpp"
#include <sstream>
namespace cssv {
inline void write_candidates(const std::string& path,const Snapshot& state) {
 state_detail::validate(state);const auto& c=state.cfg;const size_t width=c.both?2:1;
 auto eligible=[&](size_t i){int sign=c.both?(i%2?1:-1):c.sign;u64 x=c.start+i/width;return state.factors[i]==NONE&&!(x==1&&c.terms==1&&sign==-1);};
 size_t count=0;for(size_t i=0;i<state.factors.size();++i)if(eligible(i))++count;
 std::ostringstream body;body<<"# CSSV v2 start="<<c.start<<" end="<<c.end<<" terms="<<c.terms<<" step="<<c.step<<" signs="<<(c.both?"both":c.sign==1?"plus":"minus")<<" pmin="<<c.pmin<<" sieved_to="<<state.frontier<<" count="<<count<<"\n";
 for(size_t i=0;i<state.factors.size();++i)if(eligible(i)){int sign=c.both?(i%2?1:-1):c.sign;body<<c.start+i/width<<' '<<(sign==1?"+1":"-1")<<'\n';}
 std::string text=body.str();CheckpointSha256 hash;hash.update(reinterpret_cast<const uint8_t*>(text.data()),text.size());auto bytes=hash.final();const char* hex="0123456789abcdef";text+="#SHA256 ";for(auto b:bytes){text+=hex[b>>4];text+=hex[b&15];}text+='\n';
 state_detail::atomic_write(std::filesystem::u8path(path),reinterpret_cast<const uint8_t*>(text.data()),text.size());
}
}
