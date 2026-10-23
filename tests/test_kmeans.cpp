#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include "kmeans.hpp"
using namespace image_search;

static Matrix blobs(std::size_t per_blob, std::size_t dim, const std::vector<float>& offsets, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.2f);
    Matrix m(0, dim);
    std::vector<float> row(dim);
    for (float off : offsets)
        for (std::size_t i = 0; i < per_blob; ++i) {
            for (std::size_t j = 0; j < dim; ++j) row[j] = off + noise(rng);
            m.append_row(row.data());
        }
    return m;
}
template <typename F> static bool throws_invalid(F f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

int main() {
    // 1. Three well-separated blobs are recovered (each true blob has exactly one center nearby).
    {
        const std::vector<float> offs = {0.f, 10.f, 20.f};
        Matrix data = blobs(100, 8, offs, 1);
        KMeansParams p; p.k = 3; p.seed = 5;
        Matrix c = kmeans(data, p);
        assert(c.rows() == 3 && c.cols() == 8);
        std::vector<int> used(3, 0);
        for (float off : offs) {
            int best = -1; float bd = 1e30f;
            for (std::size_t i = 0; i < 3; ++i) {
                float d = std::fabs(c.row(i)[0] - off);
                if (d < bd) { bd = d; best = (int)i; }
            }
            assert(bd < 0.5f);
            ++used[best];
        }
        for (int u : used) assert(u == 1);
    }
    // 2. Same seed -> identical centers.
    {
        Matrix data = blobs(50, 16, {0.f, 3.f, 6.f, 9.f}, 2);
        KMeansParams p; p.k = 4; p.seed = 11;
        Matrix a = kmeans(data, p), b = kmeans(data, p);
        assert(a.raw() == b.raw());
    }
    // 3. Invalid arguments.
    {
        Matrix data = blobs(5, 4, {0.f}, 3);
        KMeansParams p; p.k = 0;
        assert(throws_invalid([&] { kmeans(data, p); }));
        p.k = 6;  // more clusters than the 5 points
        assert(throws_invalid([&] { kmeans(data, p); }));
        p.k = 2;
        assert(throws_invalid([&] { kmeans(Matrix(0, 4), p); }));
    }
    // 4. k == n with distinct points: every point becomes (essentially) its own center.
    {
        Matrix data = blobs(1, 4, {0.f, 5.f, 10.f, 15.f, 20.f, 25.f}, 4);
        KMeansParams p; p.k = 6;
        Matrix c = kmeans(data, p);
        for (std::size_t i = 0; i < data.rows(); ++i) {
            float d2 = 0; const float* q = data.row(i); const float* r = c.row(nearest_center(q, c));
            for (std::size_t j = 0; j < 4; ++j) d2 += (q[j]-r[j])*(q[j]-r[j]);
            assert(d2 < 1e-6f);
        }
    }
    // 5. Identical points and k > number of distinct points: forces the empty-cluster path.
    {
        Matrix data(0, 4); const float v[4] = {1.f, 2.f, 3.f, 4.f};
        for (int i = 0; i < 10; ++i) data.append_row(v);
        KMeansParams p; p.k = 3;
        Matrix c = kmeans(data, p);
        assert(c.rows() == 3);
        for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 4; ++j) assert(std::fabs(c.row(i)[j] - v[j]) < 1e-5f);
    }
    // 6. Some clusters empty after the first assignment, but the data has k distinct groups.
    {
        Matrix data = blobs(20, 4, {0.f, 0.001f, 50.f}, 6);   // two nearly coincident blobs + one far
        KMeansParams p; p.k = 3; p.seed = 3;
        Matrix c = kmeans(data, p);
        // every center must be a real, finite point and every point must have a center nearby
        for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 4; ++j) assert(std::isfinite(c.row(i)[j]));
        auto labels = assign_all(data, c);
        std::vector<int> cnt(3, 0); for (auto l : labels) { assert(l < 3); ++cnt[l]; }
        for (int x : cnt) assert(x > 0);   // no dead center
    }
    // 7. assign_all agrees with nearest_center.
    {
        Matrix data = blobs(30, 8, {0.f, 4.f}, 7);
        KMeansParams p; p.k = 2;
        Matrix c = kmeans(data, p);
        auto labels = assign_all(data, c);
        for (std::size_t i = 0; i < data.rows(); ++i) assert(labels[i] == nearest_center(data.row(i), c));
    }
    // 8. kmeans_fit: labels are consistent with the returned centers, and kmeans() agrees with it.
    {
        Matrix data = blobs(30, 8, {0.f, 4.f, 8.f}, 8);
        KMeansParams p; p.k = 3; p.seed = 2;
        KMeansResult r = kmeans_fit(data, p);
        assert(r.centers.rows() == 3 && r.labels.size() == data.rows());
        assert(r.labels == assign_all(data, r.centers));
        assert(kmeans(data, p).raw() == r.centers.raw());
    }
    // 9. Consistency also holds when the loop is cut short by max_iter, i.e. when the labels
    //    of the last assignment step would be out of date with respect to the updated centers.
    //    (Overlapping blobs and several seeds, so that some points do change cluster.)
    {
        Matrix data = blobs(40, 4, {0.f, 0.5f, 1.0f}, 9);
        for (std::uint64_t seed = 1; seed <= 20; ++seed) {
            for (int iters : {1, 2, 3}) {
                KMeansParams p; p.k = 3; p.seed = seed; p.max_iter = iters;
                KMeansResult r = kmeans_fit(data, p);
                assert(r.labels == assign_all(data, r.centers));
                for (auto l : r.labels) assert(l < 3);
            }
        }
    }
    // 10. kmeans_fit with a single cluster, and with max_iter == 0 (initial centers only).
    {
        Matrix data = blobs(10, 4, {0.f, 5.f}, 10);
        KMeansParams p; p.k = 1;
        KMeansResult r = kmeans_fit(data, p);
        for (auto l : r.labels) assert(l == 0);
        p.k = 2; p.max_iter = 0;
        r = kmeans_fit(data, p);
        assert(r.centers.rows() == 2 && r.labels == assign_all(data, r.centers));
    }
    return 0;
}