#include <cassert>
#include <vector>
#include "distance.hpp"
#include "ivfflat.hpp"
#include "metrics.hpp"
#include "types.hpp"
using namespace image_search;

int main() {
    // 4 points in 2D: [0,1] near (0,0), [2,3] far at (10,10)
    Matrix data(4, 2);
    data.raw() = {0.f, 0.f,  0.01f, 0.01f,  10.f, 10.f,  10.1f, 10.f};
    const float q[] = {0.02f, 0.02f};

    { // nearest neighbor is ranked first
        IVFFlatIndex ivf(data, {.k = 2, .nprobe = 2, .max_iter = 10, .seed = 42});
        auto res = ivf.query(q, 10);
        assert(!res.empty() && res[0] == 1);
    }
    { // results are sorted by distance and contain no duplicates
        IVFFlatIndex ivf(data, {.k = 2, .nprobe = 2, .max_iter = 10, .seed = 1});
        auto res = ivf.query(q, 10);
        for (std::size_t i = 1; i < res.size(); ++i) {
            assert(res[i] != res[i - 1]);
            assert(l2_squared(data.row(res[i - 1]), q, 2) <= l2_squared(data.row(res[i]), q, 2));
        }
    }
    { // dataset with fewer than 10 points returns at most data.rows()
        IVFFlatIndex ivf(data, {.k = 2, .nprobe = 2, .max_iter = 10, .seed = 1});
        assert(ivf.query(q, 10).size() <= data.rows());
    }
    { // max_results limit respected
        IVFFlatIndex ivf(data, {.k = 2, .nprobe = 2, .max_iter = 10, .seed = 1});
        assert(ivf.query(q, 2).size() <= 2);
    }
    { // nprobe limits the number of clusters to search
        IVFFlatIndex ivf(data, {.k = 2, .nprobe = 1, .max_iter = 10, .seed = 42});
        auto res = ivf.query(q, 10);
        // with nprobe = 1, only the closest cluster near (0,0) is checked
        assert(res.size() == 2);
        assert(res[0] == 1 && res[1] == 0);
    }
    { // dataset smaller than k
        Matrix small(2, 2);
        small.raw() = {1.f, 1.f,  5.f, 5.f};
        IVFFlatIndex ivf(small, {.k = 5, .nprobe = 2, .seed = 1});
        assert(ivf.size() == 2);
        const float sq[] = {1.1f, 1.1f};
        auto res = ivf.query(sq, 2);
        assert(!res.empty() && res[0] == 0);
    }
    { // empty dataset
        Matrix empty(0, 2);
        IVFFlatIndex ivf(empty);
        assert(ivf.query(q, 10).empty() && ivf.size() == 0);
    }
    { // determinism: same seed produces identical results
        IVFFlatParams p{.k = 2, .nprobe = 2, .max_iter = 10, .seed = 99};
        assert(IVFFlatIndex(data, p).query(q, 10) == IVFFlatIndex(data, p).query(q, 10));
    }
    { // integration with evaluate_query metrics
        IVFFlatIndex ivf(data, {.k = 2, .nprobe = 2, .max_iter = 10, .seed = 1});
        auto res = ivf.query(q, 10);
        std::unordered_set<ImageIndex> relevant = {1};
        auto metrics = evaluate_query(res, relevant);
        assert(metrics.recall5 == 1.0);
        assert(metrics.ap10 == 1.0);
    }
    { // higher dimension test (e.g. 128-dim descriptor)
        constexpr std::size_t n = 50;
        constexpr std::size_t dim = 128;
        Matrix data128(n, dim);
        for (std::size_t i = 0; i < n * dim; ++i) {
            data128.raw()[i] = static_cast<float>(i % 17) * 0.1f;
        }
        std::vector<float> q128(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            q128[d] = data128.raw()[d]; // identical to first point
        }
        IVFFlatIndex ivf128(data128, {.k = 4, .nprobe = 2, .max_iter = 10, .seed = 7});
        auto res128 = ivf128.query(q128.data(), 5);
        assert(!res128.empty());
        assert(res128[0] == 0); // exact match is index 0
    }
    return 0;
}
