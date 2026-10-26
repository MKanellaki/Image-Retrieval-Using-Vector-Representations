#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include "distance.hpp"
#include "exact.hpp"
using namespace image_search;
static int checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
static std::vector<ImageIndex> naive(const Matrix& m, const float* q, std::size_t k) {
    std::vector<ImageIndex> idx(m.rows());
    std::iota(idx.begin(), idx.end(), 0u);
    std::stable_sort(idx.begin(), idx.end(), [&](ImageIndex a, ImageIndex b) {
        return l2_squared(m.row(a), q, m.cols()) < l2_squared(m.row(b), q, m.cols()); });
    idx.resize(std::min(k, idx.size()));
    return idx;
}
int main() {
    std::mt19937 rng(1);
    // random data with many exact duplicates -> lots of tied distances
    for (std::size_t dim : {1u, 3u, 16u, 128u})
        for (std::size_t n : {1u, 2u, 9u, 10u, 11u, 200u}) {
            Matrix m(n, dim);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < dim; ++j) m.row(i)[j] = static_cast<float>(rng() % 3);
            ExactIndex ex(m);
            CHECK(ex.size() == n);
            for (int t = 0; t < 20; ++t) {
                std::vector<float> q(dim);
                for (auto& v : q) v = static_cast<float>(rng() % 3);
                for (std::size_t k : {1u, 5u, 10u, 300u}) CHECK(ex.query(q.data(), k) == naive(m, q.data(), k));
            }
        }
    // query that is a database row comes back first (lowest index among duplicates)
    Matrix m(4, 2);
    float v[4][2] = {{1,1},{5,5},{1,1},{9,9}};
    for (int i = 0; i < 4; ++i) { m.row(i)[0] = v[i][0]; m.row(i)[1] = v[i][1]; }
    ExactIndex ex(m);
    auto r = ex.query(v[2], 3);
    CHECK(r.size() == 3 && r[0] == 0 && r[1] == 2 && r[2] == 1);
    CHECK(ex.query(v[0], 0).empty());
    ExactIndex empty; float z[2] = {0,0};
    CHECK(empty.size() == 0 && empty.query(z, 5).empty());
    Matrix none(0, 2); ExactIndex en(none);
    CHECK(en.query(z, 5).empty());
    std::printf("test_exact: all %d checks passed\n", checks);
}