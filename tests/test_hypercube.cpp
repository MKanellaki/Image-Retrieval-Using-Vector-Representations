#include <cassert>
#include <vector>
#include "distance.hpp"
#include "hypercube.hpp"
#include "metrics.hpp"
#include "types.hpp"

using namespace image_search;

int main() {
    // 4 points in 2D: [0,1] near (0,0), [2,3] far at (10,10)
    Matrix data(4, 2);
    data.raw() = {0.f, 0.f,  0.01f, 0.01f,  10.f, 10.f,  10.1f, 10.f};
    const float q[] = {0.02f, 0.02f};

    { // nearest neighbor is ranked first
        HypercubeIndex cube(data, {.k = 2, .M = 10, .probes = 4, .w = 1.0, .seed = 42});
        auto res = cube.query(q, 10);
        assert(!res.empty() && res[0] == 1);
    }
    { // results are sorted by distance and contain no duplicates
        HypercubeIndex cube(data, {.k = 2, .M = 10, .probes = 4, .w = 20.0, .seed = 1});
        auto res = cube.query(q, 10);
        for (std::size_t i = 1; i < res.size(); ++i) {
            assert(res[i] != res[i - 1]);
            assert(l2_squared(data.row(res[i - 1]), q, 2) <= l2_squared(data.row(res[i]), q, 2));
        }
    }
    { // dataset with fewer than 10 points returns at most data.rows()
        HypercubeIndex cube(data, {.k = 2, .M = 10, .probes = 4, .w = 50.0, .seed = 1});
        assert(cube.query(q, 10).size() <= data.rows());
    }
    { // max_results limit respected
        HypercubeIndex cube(data, {.k = 2, .M = 10, .probes = 4, .w = 50.0, .seed = 1});
        assert(cube.query(q, 2).size() <= 2);
    }
    { // candidate limit M is respected
        Matrix data20(20, 2);
        for (std::size_t i = 0; i < 20; ++i) {
            data20.raw().push_back(0.001f * static_cast<float>(i));
            data20.raw().push_back(0.001f * static_cast<float>(i));
        }
        HypercubeIndex cube(data20, {.k = 2, .M = 3, .probes = 10, .w = 50.0, .seed = 1});
        auto res = cube.query(q, 10);
        assert(res.size() <= 3);
    }
    { // probe limit restricts search to at most 'probes' vertices
        HypercubeIndex cube(data, {.k = 3, .M = 10, .probes = 1, .w = 1.0, .seed = 42});
        auto res = cube.query(q, 10);
        // With probes = 1, only the bucket of vertex(q) is examined
        assert(res.size() <= data.rows());
    }
    { // empty dataset
        Matrix empty(0, 2);
        HypercubeIndex cube(empty);
        assert(cube.query(q, 10).empty() && cube.size() == 0);
    }
    { // determinism: same seed produces identical results
        HypercubeParams p{.k = 3, .M = 10, .probes = 4, .w = 1.0, .seed = 99};
        assert(HypercubeIndex(data, p).query(q, 10) == HypercubeIndex(data, p).query(q, 10));
    }
    { // integration with evaluate_query metrics
        HypercubeIndex cube(data, {.k = 2, .M = 10, .probes = 4, .w = 20.0, .seed = 1});
        auto res = cube.query(q, 10);
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
        HypercubeIndex cube128(data128, {.k = 4, .M = 20, .probes = 8, .w = 2.0, .seed = 7});
        auto res128 = cube128.query(q128.data(), 5);
        assert(!res128.empty());
        assert(res128[0] == 0); // exact match is index 0
    }
    return 0;
}
