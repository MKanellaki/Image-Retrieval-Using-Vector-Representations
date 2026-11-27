#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include "matching.hpp"
using namespace image_search;

namespace {
// One descriptor per row; keypoint k at (10k + 0.5, 20k + 0.25) + offset, so all points differ
ImageFeatures image(const std::vector<std::vector<float>>& rows, double offset = 0.0) {
    ImageFeatures f;
    f.width = f.height = 1000;
    f.descriptors = Matrix(rows.size(), kSiftDim);
    for (std::size_t k = 0; k < rows.size(); ++k) {
        std::copy(rows[k].begin(), rows[k].end(), f.descriptors.row(k));
        f.points.emplace_back(10.0 * k + 0.5 + offset, 20.0 * k + 0.25 + offset);
    }
    return f;
}

// 128-vector equal to `base`, with the first entries replaced by `head`
std::vector<float> vec(float base, const std::vector<float>& head = {}) {
    std::vector<float> v(kSiftDim, base);
    std::copy(head.begin(), head.end(), v.begin());
    return v;
}

// n random whole-number descriptors in [0, 255], like real SIFT
std::vector<std::vector<float>> random_rows(std::mt19937& rng, std::size_t n) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<std::vector<float>> rows(n, std::vector<float>(kSiftDim));
    for (auto& r : rows) for (auto& x : r) x = float(byte(rng));
    return rows;
}

// Noisy copy of a descriptor: each value moved by up to +-spread, kept in [0, 255]
std::vector<float> noisy(std::mt19937& rng, std::vector<float> v, int spread) {
    std::uniform_int_distribution<int> d(-spread, spread);
    for (auto& x : v) x = float(std::clamp(int(x) + d(rng), 0, 255));
    return v;
}

// Textbook version: real distances with sqrt, accept when d1 < rho * d2
std::vector<Match2D> reference(const ImageFeatures& q, const ImageFeatures& c, double rho) {
    std::vector<Match2D> out;
    if (c.size() < 2) return out;
    for (std::size_t i = 0; i < q.size(); ++i) {
        double d1 = std::numeric_limits<double>::infinity(), d2 = d1;
        std::size_t j1 = 0;
        for (std::size_t j = 0; j < c.size(); ++j) {
            double s = 0;
            for (std::size_t k = 0; k < kSiftDim; ++k) {
                const double d = double(q.descriptors.row(i)[k]) - double(c.descriptors.row(j)[k]);
                s += d * d;
            }
            const double d = std::sqrt(s);
            if (d < d1) { d2 = d1; d1 = d; j1 = j; } else if (d < d2) d2 = d;
        }
        if (d1 < rho * d2) out.emplace_back(q.points[i], c.points[j1]);
    }
    return out;
}

bool same(const std::vector<Match2D>& a, const std::vector<Match2D>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (a[k].p1 != b[k].p1 || a[k].p2 != b[k].p2) return false;
    return true;
}

template <typename F> bool throws_invalid(F f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}
}  // namespace

int main() {
    { // rho must be in (0, 1)
        const ImageFeatures a = image({vec(0), vec(1)});
        for (double rho : {0.0, 1.0, -0.5, 1.5, std::nan("")})
            assert(throws_invalid([&] { match_sift(a, a, rho); }));
        assert(!throws_invalid([&] { match_sift(a, a, 0.8); }));
    }
    { // fewer than 2 candidate descriptors: no second neighbour, so nothing passes
        const ImageFeatures q = image({vec(0), vec(5)});
        assert(match_sift(q, image({}), 0.8).empty());
        assert(match_sift(q, image({vec(0)}), 0.8).empty());
        assert(match_sift(image({}), q, 0.8).empty());   // empty query
    }
    { // ratio threshold: distances 1 and 2 (ratio 0.5), then 3 and 5 (ratio 0.6)
        const ImageFeatures q = image({vec(0)});
        const ImageFeatures c = image({vec(0, {0, 2}), vec(0, {1})}, 100.0);   // nearest is index 1
        const auto m = match_sift(q, c, 0.8);
        assert(m.size() == 1 && m[0].p1 == q.points[0] && m[0].p2 == c.points[1]);
        assert(match_sift(q, c, 0.55).size() == 1);
        assert(match_sift(q, c, 0.45).empty());

        const ImageFeatures c2 = image({vec(0, {3}), vec(0, {0, 3, 4})});
        assert(match_sift(q, c2, 0.65).size() == 1);
        assert(match_sift(q, c2, 0.55).empty());
    }
    { // ties are ambiguous: d1 == d2 is never accepted, also when both are 0
        const ImageFeatures q = image({vec(10)});
        assert(match_sift(q, image({vec(10, {13}), vec(10, {7})}), 0.99).empty());
        assert(match_sift(q, image({vec(10), vec(10), vec(200)}), 0.99).empty());
    }
    { // the old nearest becomes the second nearest: squared distances 100, far, 1
        const ImageFeatures q = image({vec(50)});
        const ImageFeatures c = image({vec(50, {60}), vec(200), vec(50, {51})});
        const auto m = match_sift(q, c, 0.5);   // 1 / 10 = 0.1
        assert(m.size() == 1 && m[0].p2 == c.points[2]);
        assert(match_sift(q, c, 0.09).empty());  // d2 is 10, not the far one
    }
    { // matches come out in query order, each with its own candidate point
        const ImageFeatures q = image({vec(0), vec(100), vec(200)});
        const ImageFeatures c = image({vec(200), vec(255), vec(0), vec(100)}, 3.0);
        const auto m = match_sift(q, c, 0.8);
        assert(m.size() == 3);
        assert(m[0].p1 == q.points[0] && m[0].p2 == c.points[2]);
        assert(m[1].p1 == q.points[1] && m[1].p2 == c.points[3]);
        assert(m[2].p1 == q.points[2] && m[2].p2 == c.points[0]);
        assert(same(m, match_sift(q, c, 0.8)));   // deterministic
    }
    { // many query descriptors may match one candidate keypoint (no cross-check)
        const ImageFeatures q = image({vec(0, {1}), vec(0, {0, 1})});
        const ImageFeatures c = image({vec(0), vec(250)});
        const auto m = match_sift(q, c, 0.8);
        assert(m.size() == 2 && m[0].p2 == c.points[0] && m[1].p2 == c.points[0]);
    }
    { // descriptor widths must agree
        ImageFeatures bad;
        bad.descriptors = Matrix(2, 64);
        bad.points = {Point2D(0, 0), Point2D(1, 1)};
        assert(throws_invalid([&] { match_sift(image({vec(0)}), bad, 0.8); }));
    }
    { // the inputs are not modified
        const ImageFeatures q = image({vec(0), vec(100)}), c = image({vec(0), vec(100), vec(200)});
        const ImageFeatures q2 = q, c2 = c;
        match_sift(q, c, 0.8);
        assert(q.descriptors.raw() == q2.descriptors.raw() && q.points == q2.points);
        assert(c.descriptors.raw() == c2.descriptors.raw() && c.points == c2.points);
    }
    { // worst-case cancellation: values near 255 make ||q||^2 + ||b||^2 - 2 q.b about 16.6
      // million minus 16.6 million (still below 2^24, so exact); distances 1 and 4 must survive
        const ImageFeatures q = image({vec(255)});
        const ImageFeatures c = image({vec(255, {253}), vec(255, {254})});   // squared 4, then 1
        const auto m = match_sift(q, c, 0.8);
        assert(m.size() == 1 && m[0].p2 == c.points[1]);
        assert(match_sift(q, c, 0.55).size() == 1);
        assert(match_sift(q, c, 0.45).empty());

        const ImageFeatures c0 = image({vec(255, {254}), vec(255)});   // exact copy: d1 = 0
        const auto m0 = match_sift(q, c0, 0.01);
        assert(m0.size() == 1 && m0[0].p2 == c0.points[1]);

        // far away: squared 8323200 and 8322691, ratio ~0.99997, rejected even at 0.99
        assert(match_sift(image({vec(0)}), image({vec(255), vec(255, {254})}), 0.99).empty());
    }
    { // random descriptors agree with the textbook sqrt version
        std::mt19937 rng(7);
        for (int trial = 0; trial < 20; ++trial) {
            auto qr = random_rows(rng, 60), cr = random_rows(rng, 40);
            for (std::size_t k = 0; k < qr.size(); k += 2) cr.push_back(noisy(rng, qr[k], 12));
            const ImageFeatures q = image(qr), c = image(cr, 7.0);
            for (double rho : {0.6, 0.8, 0.95})
                assert(same(match_sift(q, c, rho), reference(q, c, rho)));
            assert(match_sift(q, c, 0.8).size() >= 25);   // most of the 30 noisy copies pass
        }
    }
    { // query sizes around the 256-row block: every row is matched once, last partial block included
        std::mt19937 rng(11);
        for (std::size_t nq : {255u, 256u, 257u, 511u, 512u, 513u, 700u}) {
            auto qr = random_rows(rng, nq), cr = random_rows(rng, 50);
            for (std::size_t k = 0; k < nq; k += 3) cr.push_back(noisy(rng, qr[k], 10));
            const ImageFeatures q = image(qr), c = image(cr, 5.0);
            const auto m = match_sift(q, c, 0.8);
            assert(same(m, reference(q, c, 0.8)));
            assert(m.size() >= nq / 3 - 5);
            assert(m.back().p1 == q.points[nq - 1 - (nq - 1) % 3]);   // last block reached
        }
    }
    { // many candidate rows, true neighbours at the very end
        std::mt19937 rng(3);
        auto cr = random_rows(rng, 3000), qr = random_rows(rng, 5);
        cr.insert(cr.end(), qr.begin(), qr.end());   // exact copies at 3000..3004
        const ImageFeatures q = image(qr), c = image(cr);
        const auto m = match_sift(q, c, 0.8);
        assert(m.size() == 5);
        for (std::size_t k = 0; k < m.size(); ++k) assert(m[k].p2 == c.points[3000 + k]);
        assert(same(m, reference(q, c, 0.8)));
    }
    return 0;
}