#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

#include "features.hpp"

using namespace image_search;

// n rows of dimension dim; row i starts with the value i, so a sampled row can be traced back.
static Matrix numbered(std::size_t n, std::size_t dim) {
    Matrix m(0, dim);
    std::vector<float> row(dim, 0.f);
    for (std::size_t i = 0; i < n; ++i) { row[0] = static_cast<float>(i); m.append_row(row.data()); }
    return m;
}
static std::vector<std::size_t> picked(const Matrix& s) {
    std::vector<std::size_t> v;
    for (std::size_t i = 0; i < s.rows(); ++i) v.push_back(static_cast<std::size_t>(s.row(i)[0]));
    return v;
}

int main() {
    // fewer (or exactly as many) rows than S: everything is returned unchanged
    {
        Matrix m = numbered(5, 3);
        assert(sample_descriptors(m, 5, 1).raw() == m.raw());
        assert(sample_descriptors(m, 50, 1).raw() == m.raw());
    }
    // S == 0 and empty input
    {
        Matrix s = sample_descriptors(numbered(10, 4), 0, 1);
        assert(s.rows() == 0 && s.cols() == 4);
        Matrix e = sample_descriptors(Matrix(0, 4), 5, 1);
        assert(e.rows() == 0);
    }
    // exactly S distinct rows, in their original order, with intact contents
    {
        Matrix m = numbered(100, 8);
        for (std::uint64_t seed = 1; seed <= 20; ++seed) {
            Matrix s = sample_descriptors(m, 10, seed);
            assert(s.rows() == 10 && s.cols() == 8);
            auto p = picked(s);
            for (std::size_t i = 1; i < p.size(); ++i) assert(p[i - 1] < p[i]);   // sorted, no duplicates
            for (std::size_t i = 0; i < s.rows(); ++i)                          // row content untouched
                for (std::size_t j = 1; j < 8; ++j) assert(s.row(i)[j] == m.row(p[i])[j]);
            assert(p.back() < 100);
        }
    }
    // reproducible for the same seed; the seed actually matters
    {
        Matrix m = numbered(100, 4);
        assert(sample_descriptors(m, 10, 7).raw() == sample_descriptors(m, 10, 7).raw());
        bool differs = false;
        for (std::uint64_t seed = 2; seed <= 6; ++seed)
            differs |= sample_descriptors(m, 10, seed).raw() != sample_descriptors(m, 10, 1).raw();
        assert(differs);
    }
    // golden value: the sample depends only on (data, S, seed) on every platform
    {
        auto p = picked(sample_descriptors(numbered(20, 2), 5, 42));
        assert((p == std::vector<std::size_t>{0, 1, 2, 6, 9}));
    }
    // uniformity: every row is chosen with probability S/n (here 3/10)
    {
        Matrix m = numbered(10, 1);
        std::vector<int> count(10, 0);
        const int trials = 20000;
        for (int t = 0; t < trials; ++t)
            for (std::size_t i : picked(sample_descriptors(m, 3, derive_seed(7, t)))) ++count[i];
        for (int c : count) assert(c > 5600 && c < 6400);      // expected 6000, standard deviation about 65
    }
    // derive_seed: deterministic, matches the SplitMix64 reference value, no collisions
    {
        assert(derive_seed(0, 0) == 0xE220A8397B1DCDAFULL);    // first output of SplitMix64 seeded with 0
        assert(derive_seed(1, 0) == 10451216379200822465ULL);
        assert(derive_seed(1, 1) == 13757245211066428519ULL);
        assert(derive_seed(1, 0) != derive_seed(2, 0));
        std::set<std::uint64_t> seen;
        for (std::uint64_t i = 0; i < 20000; ++i) seen.insert(derive_seed(1, i));
        assert(seen.size() == 20000);
    }
    return 0;
}