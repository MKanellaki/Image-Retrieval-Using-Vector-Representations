#include <algorithm>
#include <cassert>
#include <random>
#include <set>
#include <vector>
#include "exact.hpp"
#include "search_index.hpp"
using namespace image_search;

namespace {
Matrix random_db(std::size_t n, std::size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    Matrix m(n, dim);
    for (float& x : m.raw()) x = u(rng);
    return m;
}

Args args_for(Method m) {
    Args a;
    a.method = m;
    a.seed = 1;
    a.ivfflat.kclusters = 8;
    a.ivfpq.kclusters = 8;
    a.ivfpq.M = 4;          // 16 / 4 = 4 dimensions per sub-vector
    a.ivfpq.nbits = 4;
    return a;
}
}  // namespace

int main() {
    const Matrix db = random_db(300, 16, 1);
    const Matrix queries = random_db(10, 16, 2);

    { // Exact through make_search equals the index itself, for any k
        const SearchFn search = make_search(args_for(Method::Exact), db);
        const ExactIndex ex(db);
        for (std::size_t i = 0; i < queries.rows(); ++i)
            for (std::size_t k : {1u, 10u, 50u, 100u})
                assert(search(queries.row(i), k) == ex.query(queries.row(i), k));
    }
    { // asking for more rows than exist gives every row once
        const SearchFn search = make_search(args_for(Method::Exact), db);
        auto r = search(queries.row(0), 1000);
        assert(r.size() == db.rows());
        std::sort(r.begin(), r.end());
        assert(std::adjacent_find(r.begin(), r.end()) == r.end());
    }
    { // every method: at most k rows, valid and distinct; Top-10 is a prefix of Top-C
        for (Method m : {Method::Exact, Method::LSH, Method::Hypercube, Method::IVFFlat, Method::IVFPQ}) {
            const SearchFn search = make_search(args_for(m), db);
            for (std::size_t i = 0; i < queries.rows(); ++i) {
                const auto top_c = search(queries.row(i), 50);
                const auto top_10 = search(queries.row(i), 10);
                assert(top_c.size() <= 50 && top_10.size() <= 10);
                assert(std::set<ImageIndex>(top_c.begin(), top_c.end()).size() == top_c.size());
                for (ImageIndex row : top_c) assert(row < db.rows());
                assert(std::equal(top_10.begin(), top_10.end(), top_c.begin()));
            }
        }
    }
    { // the Exact and IVF methods return a full Top-C on this data
        for (Method m : {Method::Exact, Method::IVFFlat})
            assert(make_search(args_for(m), db)(queries.row(0), 50).size() == 50);
    }
    return 0;
}