#include <cassert>
#include <vector>
#include "distance.hpp"
#include "lsh.hpp"
#include "types.hpp"
using namespace image_search;

int main() {
    // 4 points in 2D: [0,1] near (0,0), [2,3] far at (10,10)
    Matrix data(4, 2);
    data.raw() = {0.f, 0.f,  0.01f, 0.01f,  10.f, 10.f,  10.1f, 10.f};
    const float q[] = {0.02f, 0.02f};

    { // nearest neighbor is ranked first
        LSHIndex lsh(data, {.k = 2, .L = 10, .w = 1.0, .seed = 42});
        auto res = lsh.query(q, 10);
        assert(!res.empty() && res[0] == 1);
    }
    { // results are sorted by distance and contain no duplicates
        LSHIndex lsh(data, {.k = 1, .L = 5, .w = 20.0, .seed = 1});
        auto res = lsh.query(q, 10);
        for (std::size_t i = 1; i < res.size(); ++i) {
            assert(res[i] != res[i - 1]);
            assert(l2_squared(data.row(res[i - 1]), q, 2) <= l2_squared(data.row(res[i]), q, 2));
        }
    }
    { // dataset with fewer than 10 points returns at most data.rows()
        LSHIndex lsh(data, {.k = 1, .L = 5, .w = 50.0, .seed = 1});
        assert(lsh.query(q, 10).size() <= data.rows());
    }
    { // max_results limit respected
        LSHIndex lsh(data, {.k = 1, .L = 5, .w = 50.0, .seed = 1});
        assert(lsh.query(q, 2).size() <= 2);
    }
    { // empty dataset
        Matrix empty(0, 2);
        LSHIndex lsh(empty);
        assert(lsh.query(q, 10).empty() && lsh.size() == 0);
    }
    { // determinism: same seed produces identical results
        LSHParams p{.k = 2, .L = 4, .w = 1.0, .seed = 99};
        assert(LSHIndex(data, p).query(q, 10) == LSHIndex(data, p).query(q, 10));
    }
    return 0;
}
