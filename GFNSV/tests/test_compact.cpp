/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */
#include "../src/gfnsv_compact.hpp"
#include <iostream>
#include <iterator>

namespace fs=std::filesystem;
using gfnsv_state::State;
using gfnsv_state::U64;
namespace compact=gfnsv_compact;
namespace sd=gfnsv_state::detail;
static fs::path output;
static unsigned checks=0,sequence=0;

static void check(bool condition,const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class F> static void rejects(F function,const char* message) {
    bool rejected=false;
    try {function();} catch (const std::exception&) {rejected=true;}
    check(rejected,message);
}
static std::string unique(const char* label) {
    return (output/(std::to_string(++sequence)+"-"+label)).u8string();
}
static std::string text(const std::string& path) {
    std::ifstream in(gfnsv_state::native_path(path),std::ios::binary);
    if (!in) throw std::runtime_error("cannot read test output");
    return std::string(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
}
static void fixture(const std::string& path,const std::string& contents) {
    std::ofstream out(gfnsv_state::native_path(path),std::ios::binary|std::ios::trunc);
    out.write(contents.data(),static_cast<std::streamsize>(contents.size()));
    if (!out) throw std::runtime_error("cannot write test fixture");
}
static std::string replace(std::string value,const std::string& before,const std::string& after) {
    const auto pos=value.find(before);
    if (pos==std::string::npos) throw std::runtime_error("test replacement anchor missing");
    value.replace(pos,before.size(),after);
    return value;
}
static State sample() {
    State s;s.n=1;s.bmin=2;s.bmax=12;s.pmin=3;s.pmax=100;s.next_k=10;
    s.factors={0,0,0,5,0,5};
    return s;
}
static std::vector<U64> alive(const State& s) {
    std::vector<U64> result;
    for (std::size_t i=0;i<s.factors.size();++i)
        if (!s.factors[i]) result.push_back(gfnsv_state::first_even(s)+2*static_cast<U64>(i));
    return result;
}
static std::string factor_document(const State& s,const std::vector<std::string>& records) {
    std::string result=sd::factor_header(s)+"\n";
    for (const auto& record:records) result+=record+"\n";
    sd::Sha256 hash;hash.update(result);
    return result+"#GFNSV_FACTORS_END count="+std::to_string(records.size())+" sha256="+hash.final_hex()+"\n";
}
static void reject_document(const std::string& contents,const char* message) {
    const auto path=unique("bad-compact.txt");fixture(path,contents);
    rejects([&]{(void)compact::read_state(path);},message);
}

static void roundtrips() {
    const State original=sample();
    const auto legacy=unique("legacy-v3.txt");
    gfnsv_state::write_state_atomic(legacy,original,false);
    check(!compact::is_compact(legacy),"v3 must route to its original reader");
    auto restored=gfnsv_state::read_state(legacy);
    check(restored.factors==original.factors && restored.historical_removed.empty(),"v3 factor roundtrip changed");
    rejects([&]{(void)compact::read_state(legacy);},"compact reader accepted v3");

    for (const std::string format:{"G","A","base","expr"}) {
        const auto path=unique("roundtrip.txt");
        compact::write_state_atomic(path,original,format,false);
        check(compact::is_compact(path),"v4 detection failed");
        std::string parsed="unchanged";
        restored=compact::read_state(path,&parsed);
        check(parsed==format && restored.expr==(format=="expr"),"output format did not roundtrip");
        check(alive(restored)==alive(original) && restored.next_k==original.next_k,"compact survivor/frontier roundtrip changed");
        check(restored.factors[3]==1 && gfnsv_state::is_historical_removed(restored,3),"old removal did not become a masked sentinel");
        check(restored.historical_removed.size()==original.factors.size() && !restored.historical_removed[0],"compact mask not initialized correctly");
        check(text(path).find("#FACTOR ")==std::string::npos,"compact writer leaked per-removal factor records");
        rejects([&]{gfnsv_state::write_state_atomic(unique("sentinel-v3.txt"),restored,false);},"v3 writer accepted unknown factors");
        const auto same=unique("canonical-copy.txt");
        compact::write_state_atomic(same,restored,format,false);
        check(text(same)==text(path),"compact read/write was not canonical");
        const auto old=text(path);
        rejects([&]{compact::write_state_atomic(path,restored,format,false);},"exclusive write replaced an existing file");
        check(text(path)==old,"exclusive write damaged prior output");
        std::string crlf;
        for (char c:old) {if (c=='\n') crlf+='\r';crlf+=c;}
        const auto crlf_path=unique("crlf.txt");fixture(crlf_path,crlf);
        check(alive(compact::read_state(crlf_path))==alive(original),"CRLF compact file rejected");
        const auto no_final_lf=unique("no-final-lf.txt");fixture(no_final_lf,old.substr(0,old.size()-1));
        check(alive(compact::read_state(no_final_lf))==alive(original),"complete footer without terminal LF rejected");
    }
    for (const std::string format:{"G","A","base","expr"}) {
        State s=sample();s.bmin=8;s.bmax=8;s.factors={5};
        const auto empty=unique("zero-survivors.txt");
        compact::write_state_atomic(empty,s,format,false);
        const auto loaded=compact::read_state(empty);
        check(alive(loaded).empty() && loaded.factors==std::vector<U64>{1},"zero-survivor roundtrip failed");
        s.bmin=3;s.bmax=3;s.factors.clear();s.next_k=25;
        const auto slots=unique("zero-slots.txt");compact::write_state_atomic(slots,s,format,false);
        check(compact::read_state(slots).factors.empty(),"zero-slot roundtrip failed");
    }
    State edge=sample();edge.bmin=(std::numeric_limits<U64>::max)();edge.bmax=edge.bmin;edge.factors.clear();edge.next_k=25;
    const auto edge_path=unique("max-odd-empty.txt");compact::write_state_atomic(edge_path,edge,"G",false);
    check(compact::read_state(edge_path).factors.empty(),"UINT64_MAX empty range failed");
}

static void corruption() {
    const auto path=unique("canonical-g.txt");
    compact::write_state_atomic(path,sample(),"G",false);
    const auto good=text(path);
    reject_document(replace(good,"\n2\n","\n3\n"),"odd survivor accepted");
    reject_document(replace(good,"\n2\n","\n0\n"),"out-of-range survivor accepted");
    reject_document(replace(good,"\n2\n","\n02\n"),"noncanonical integer accepted");
    reject_document(replace(good,"\n2\n4\n","\n4\n2\n"),"unordered survivors accepted");
    reject_document(replace(good,"\n2\n4\n","\n2\n2\n"),"duplicate survivors accepted");
    reject_document(replace(good,"\n2\n","\n"),"missing survivor accepted");
    reject_document(replace(good,"\n2\n","\n2 3\n"),"extra candidate columns accepted");
    reject_document(replace(good,"\n2\n","\n2\n8\n"),"extra survivor accepted");
    reject_document(replace(good,"alive_count=4","alive_count=3"),"wrong alive count accepted");
    reject_document(replace(good,"#GFNSV_END count=4","#GFNSV_END count=3"),"wrong footer count accepted");
    reject_document(replace(good,"slots=6","slots=7"),"wrong slot count accepted");
    reject_document(replace(good,"n=1 bmin=","n=2 bmin="),"wrong metadata n accepted");
    reject_document(replace(good,"GFN n=1","GFN n=2"),"wrong view n accepted");
    reject_document(replace(good,"Sieved to 40","Sieved to 41"),"wrong view frontier accepted");
    reject_document(replace(good,"pmin=41","pmin=45"),"wrong next prime accepted");
    reject_document(replace(good,"next_k=10","next_k=0"),"frontier before start accepted");
    reject_document(replace(good,"next_k=10","next_k=26"),"frontier beyond pmax accepted");
    reject_document(replace(good,"#GFNSV_STATE","#GFNSV_COMPLETE"),"incorrect completion marker accepted");
    reject_document(replace(good,"engine=cuda","engine=cpu"),"wrong engine accepted");
    reject_document(replace(good,"format=G","format=base"),"view/metadata format mismatch accepted");
    reject_document(good+"junk\n","data after footer accepted");
    reject_document(good.substr(0,good.find("#GFNSV_END")),"missing footer accepted");
    reject_document(good.substr(0,good.size()-10),"truncated checksum accepted");
    reject_document(replace(good,"\n2\n","\n"+std::string(1025,'2')+"\n"),"oversized line accepted");
    reject_document(replace(good,"\n2\n",std::string("\n2\0\n",4)),"embedded NUL accepted");

    for (const auto& bad:{replace(good,"version=4","version=5"),
                         replace(good,"// format=4","// format=5"),
                         good.substr(0,good.find('\n')+1),
                         std::string("GFN n=1 // format="),
                         replace(good,"#GFNSV_STATE version=4","#GFNSV_STATX version=4")}) {
        const auto damaged=unique("detect-damaged-v4.txt");fixture(damaged,bad);
        check(compact::is_compact(damaged),"damaged v4 escaped strict compact dispatch");
        rejects([&]{(void)compact::read_state(damaged);},"damaged v4 was read successfully");
    }
    const auto legacy_view=unique("old-bare-gfn.txt");fixture(legacy_view,"GFN n=1 // Sieved to 40\n2\n4\n");
    check(!compact::is_compact(legacy_view),"old bare view misidentified as compact");
    const auto pexpr=unique("canonical-expr.txt");compact::write_state_atomic(pexpr,sample(),"expr",false);
    reject_document(replace(text(pexpr),"2^2+1","2^4+1"),"wrong candidate exponent accepted");
    reject_document(replace(text(pexpr),"2^2+1","2^2-1"),"wrong candidate sign accepted");
    rejects([&]{compact::write_state_atomic(unique("bad-format.txt"),sample(),"N",false);},"unsupported output format accepted");
    State huge=sample();huge.bmax=2*gfnsv_state::max_slots;huge.factors.clear();
    std::string inflated=compact::detail::view_header(huge,"G")+"\n"+
        compact::detail::metadata(huge,gfnsv_state::max_slots,"G")+"\n#GFNSV_END count=0 sha256="+std::string(64,'0')+"\n";
    reject_document(inflated,"inflated survivor count allocated/read despite a tiny file");
}

static void masks_and_logs() {
    auto s=sample();s.factors[3]=1;
    rejects([&]{(void)sd::validate(s);},"unmasked sentinel accepted");
    s.historical_removed.assign(5,0);
    rejects([&]{(void)sd::validate(s);},"short historical mask accepted");
    s.historical_removed.assign(6,0);s.historical_removed[3]=2;
    rejects([&]{(void)sd::validate(s);},"invalid historical mask bit accepted");
    s.historical_removed[3]=1;
    check(sd::validate(s)==4 && gfnsv_state::is_historical_removed(s,3),"valid masked sentinel rejected");
    s.factors[3]=0;
    rejects([&]{(void)sd::validate(s);},"historical candidate revival accepted");
    s.factors[3]=5;
    check(sd::validate(s)==4 && !gfnsv_state::is_historical_removed(s,3),"recovered known historical factor skipped");
    check(!gfnsv_state::is_historical_removed(s,6),"out-of-range historical lookup accepted");

    const auto v3=unique("v3-factor.txt");gfnsv_state::write_state_atomic(v3,sample(),false);
    const auto bad_v3=unique("v3-sentinel.txt");fixture(bad_v3,replace(text(v3),"#FACTOR 8 5","#FACTOR 8 1"));
    rejects([&]{(void)gfnsv_state::read_state(bad_v3);},"v3 reader accepted factor one");

    const auto checkpoint=unique("merge-state.txt");compact::write_state_atomic(checkpoint,sample(),"G",false);
    const auto known_log=unique("known-factors.txt");gfnsv_state::write_factors_atomic(known_log,sample(),false);
    auto loaded=compact::read_state(checkpoint);
    const auto unknown_log=unique("unknown-factors.txt");gfnsv_state::write_factors_atomic(unknown_log,loaded,false);
    check(text(unknown_log).find(" | ")==std::string::npos && text(unknown_log).find("count=0")!=std::string::npos,
          "unknown historical factors leaked into optional factor log");
    compact::merge_factor_log(unknown_log,loaded);
    check(loaded.factors[3]==1,"empty optional log altered unknown factors");
    compact::merge_factor_log(known_log,loaded);
    check(loaded.factors==sample().factors && !gfnsv_state::is_historical_removed(loaded,3),"factor log did not restore known factors");
    compact::merge_factor_log(known_log,loaded);
    check(loaded.factors==sample().factors,"repeat factor merge changed known factors");
    gfnsv_state::write_state_atomic(unique("recovered-v3.txt"),loaded,false);
    check(true,"recovered known factors must permit v3 output");

    auto expect_bad_log=[&](const std::string& content,const char* message) {
        const auto path=unique("bad-factors.txt");fixture(path,content);
        const auto before=loaded.factors;
        const auto mask=loaded.historical_removed;
        rejects([&]{compact::merge_factor_log(path,loaded);},message);
        check(loaded.factors==before && loaded.historical_removed==mask,"rejected factor log changed state");
    };
    const auto good=text(known_log);
    expect_bad_log(replace(good,"n=1","n=2"),"wrong log n accepted");
    expect_bad_log(replace(good,"bmax=12","bmax=14"),"wrong log range accepted");
    expect_bad_log(replace(good,"start_pmin=3","start_pmin=5"),"wrong log start accepted");
    expect_bad_log(good.substr(0,good.find("#GFNSV_FACTORS_END")),"truncated log accepted");
    expect_bad_log(replace(good,"count=2","count=1"),"wrong log count accepted");
    expect_bad_log(good+"junk\n","log trailing data accepted");
    expect_bad_log(factor_document(loaded,{"1 | 8^2+1"}),"log sentinel accepted");
    expect_bad_log(factor_document(loaded,{"41 | 8^2+1"}),"log factor at unprocessed frontier accepted");
    expect_bad_log(factor_document(loaded,{"5 | 2^2+1"}),"log removed a current survivor");
    expect_bad_log(factor_document(loaded,{"5 | 8^2+1","5 | 8^2+1"}),"duplicate log base accepted");
    expect_bad_log(factor_document(loaded,{"5 | 12^2+1","5 | 8^2+1"}),"unordered log bases accepted");
    expect_bad_log(factor_document(loaded,{"13 | 8^2+1"}),"conflicting known factor accepted");
    expect_bad_log(factor_document(loaded,{"5 | 8^4+1"}),"wrong log exponent accepted");

    State earlier=sample();earlier.factors[5]=0;
    const auto old=unique("historical-only.txt");compact::write_state_atomic(old,earlier,"G",false);
    auto continued=compact::read_state(old);continued.factors[5]=5;
    const auto new_log=unique("new-only-factors.txt");gfnsv_state::write_factors_atomic(new_log,continued,false);
    check(text(new_log).find("5 | 12^2+1")!=std::string::npos && text(new_log).find("5 | 8^2+1")==std::string::npos,
          "optional log did not retain only available factors");
    const auto newer=unique("continued.txt");compact::write_state_atomic(newer,continued,"G",false);
    auto again=compact::read_state(newer);compact::merge_factor_log(new_log,again);
    check(again.factors[3]==1 && again.factors[5]==5,"partial known-factor history did not resume");
}

int main(int argc,char** argv) {
    try {
        if (argc>=3 && std::string(argv[1])=="--verify") {
            for (int i=2;i<argc;++i) {
                std::string format;
                const auto state=compact::read_state(argv[i],&format);
                const auto survivors=alive(state);
                std::cout<<"verified format="<<format<<" n="<<state.n<<" bmin="<<state.bmin
                         <<" bmax="<<state.bmax<<" next_k="<<state.next_k
                         <<" alive="<<survivors.size()<<" complete="<<(sd::complete(state)?"yes":"no")<<" bases=";
                for (std::size_t j=0;j<survivors.size();++j) {
                    if (j) std::cout<<",";
                    std::cout<<survivors[j];
                }
                std::cout<<" path="<<argv[i]<<"\n";
            }
            return 0;
        }
        if (argc!=2) throw std::runtime_error("usage: test_compact NEW_OUTPUT_DIRECTORY");
        output=gfnsv_state::native_path(argv[1]);
        if (fs::exists(output) || !fs::create_directories(output)) throw std::runtime_error("test output directory must be new");
        roundtrips();corruption();masks_and_logs();
        std::cout<<"compact selftest ok: "<<checks<<" checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"compact selftest failed: "<<error.what()<<"\n";
        return 1;
    }
}
