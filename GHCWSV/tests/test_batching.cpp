#include "work_batching.hpp"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    using ghcw_work::Batching;
    Batching adaptive(8192,171428,true), fixed(8192,171428,false);
    assert(adaptive.selected()==11 && adaptive.producer_batch()==8192);
    assert(fixed.selected()==11 && fixed.producer_batch()==11);
    assert(!adaptive.refresh(171428));
    assert(!adaptive.refresh(140000));
    assert(adaptive.refresh(130000) && adaptive.selected()==15);
    assert(adaptive.refresh(20544) && adaptive.selected()==97);
    assert(!fixed.refresh(20544) && fixed.selected()==11);
    assert(!adaptive.refresh(0));
    Batching huge(8192,2100002,true);
    assert(huge.selected()==1);
    assert(huge.refresh(1) && huge.selected()==8192);
    Batching tiny(3,10000000,true);
    assert(tiny.refresh(1) && tiny.selected()==3);
    // Every producer index must be consumed exactly once despite arbitrary
    // changes of GPU chunk sizes and a partially processed final chunk.
    for (size_t total : {1,11,17,97,8192,1048576}) {
        size_t cursor=0, visited=0;
        while(cursor<total) {
            const size_t n=ghcw_work::next_chunk(total,cursor,visited%2?97:11);
            assert(n>0 && cursor+n<=total);
            for(size_t i=cursor;i<cursor+n;++i)assert(i==visited++);
            cursor+=n;
        }
        assert(visited==total && ghcw_work::next_chunk(total,total,1)==0);
    }
    // Exhaustive threshold rounding and bounded chunk arithmetic.
    for(size_t basis=1;basis<=10000;++basis) {
        Batching sizing(1048576,basis,true);
        const size_t just_above=(basis*4)/5+1;
        assert(!sizing.refresh(just_above));
        const size_t trigger=(basis*4)/5;
        if(trigger) {
            assert(sizing.refresh(trigger));
            assert(sizing.selected()*trigger<=Batching::term_budget);
        }
    }
    std::cout<<"PASS: adaptive threshold, fixed baseline, work budget, producer tail cursor\n";
}
