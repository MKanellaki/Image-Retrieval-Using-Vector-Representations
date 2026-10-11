#include <cassert>
#include <cmath>
#include <unordered_set>
#include <vector>
#include "metrics.hpp"
using namespace image_search;
static bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }
int main() {
    const std::unordered_set<ImageIndex> rel = {1, 2, 3, 4, 5};
    { // all relevant on top
        auto m = evaluate_query({1,2,3,4,5,100,101,102,103,104}, rel);
        assert(close(m.recall5,1) && close(m.recall10,1) && close(m.ap10,1) && m.returned==10);
    }
    { // nothing relevant
        auto m = evaluate_query({100,101,102,103,104,105,106,107,108,109}, rel);
        assert(close(m.recall5,0) && close(m.recall10,0) && close(m.ap10,0));
    }
    { // relevant at ranks 2, 4, 9: R@5=2/5, R@10=3/5, AP=(1/2+2/4+3/9)/5
        auto m = evaluate_query({100,1,101,2,102,103,104,105,3,106}, rel);
        assert(close(m.recall5,0.4) && close(m.recall10,0.6));
        assert(close(m.ap10,(0.5+0.5+3.0/9.0)/5.0));
    }
    { // only 3 results returned, all relevant: missing slots count as non-relevant
        auto m = evaluate_query({1,2,3}, rel);
        assert(close(m.recall5,0.6) && close(m.recall10,0.6) && close(m.ap10,0.6) && m.returned==3);
    }
    { // relevant item at rank 6: counts for @10 but not @5
        auto m = evaluate_query({100,101,102,103,104,1,105,106,107,108}, rel);
        assert(close(m.recall5,0) && close(m.recall10,0.2) && close(m.ap10,(1.0/6.0)/5.0));
    }
    { // duplicate result is counted once but still occupies its rank
        auto m = evaluate_query({1,1,2}, rel);
        assert(close(m.recall10,0.4) && close(m.ap10,(1.0+2.0/3.0)/5.0));
    }
    { // empty result list
        auto m = evaluate_query({}, rel);
        assert(close(m.recall10,0) && m.returned==0);
    }
    { // summarize
        QueryMetrics a, b;
        a.recall5=1; a.recall10=1; a.ap10=1; a.returned=10;
        b.recall5=0; b.recall10=0.2; b.ap10=0.1; b.returned=4;
        auto s = summarize({a,b}, {2.0, 4.0});
        assert(s.queries==2 && s.queries_with_fewer_than_10==1);
        assert(close(s.mean_recall5,0.5) && close(s.mean_recall10,0.6) && close(s.map10,0.55) && close(s.avg_query_ms,3.0));
        assert(summarize({}, {}).queries==0);
    }
    { // empty relevant set: no division by zero, everything is 0
       auto m = evaluate_query({1,2,3}, {});
       assert(close(m.recall5,0) && close(m.recall10,0) && close(m.ap10,0) && m.returned==3);
    }
    return 0;
}