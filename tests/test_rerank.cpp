#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <set>
#include <vector>
#include "matching.hpp"
#include "rerank.hpp"
using namespace image_search;

namespace {
// n features with distinct random whole-number descriptors at random positions in [0, 500)^2
ImageFeatures random_features(std::mt19937& rng, std::size_t n) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<double> pos(0.0, 500.0);
    ImageFeatures f;
    f.width = f.height = 500;
    f.descriptors = Matrix(n, kSiftDim);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < kSiftDim; ++j) f.descriptors.row(i)[j] = float(byte(rng));
        f.points.emplace_back(pos(rng), pos(rng));
    }
    return f;
}

// Same descriptors, every point moved by (dx, dy)
ImageFeatures shifted(const ImageFeatures& f, double dx, double dy) {
    ImageFeatures g = f;
    for (Point2D& p : g.points) p = Point2D(p.x + dx, p.y + dy);
    return g;
}

CandidateResult cand(std::size_t rank, std::size_t matches, bool success, std::size_t inliers, double median) {
    CandidateResult c;
    c.row = static_cast<ImageIndex>(100 + rank);
    c.original_rank = rank;
    c.matches = matches;
    c.success = success;
    c.inliers = success ? inliers : 0;
    c.median_error = success ? median : std::nan("");
    return c;
}

std::vector<std::size_t> ranks_of(const std::vector<CandidateResult>& v) {
    std::vector<std::size_t> r;
    for (const CandidateResult& c : v) r.push_back(c.original_rank);
    return r;
}
}  // namespace

int main() {
    { // pair seeds: reproducible, different for different pairs, independent of the rank
        assert(pair_seed(1, 3, 42) == pair_seed(1, 3, 42));
        std::set<std::uint64_t> seen;
        for (std::size_t q = 0; q < 20; ++q)
            for (ImageIndex row = 0; row < 50; ++row) seen.insert(pair_seed(1, q, row));
        assert(seen.size() == 20 * 50);
        assert(pair_seed(1, 3, 42) != pair_seed(2, 3, 42));
        assert(pair_seed(1, 3, 42) != pair_seed(1, 42, 3));
    }
    { // a translated copy verifies: every descriptor matches, every match is an inlier
        std::mt19937 rng(1);
        const ImageFeatures q = random_features(rng, 40);
        const CandidateResult r = verify_candidate(q, shifted(q, 20, 10), 0.8, {.seed = 5});
        assert(r.matches == 40);
        assert(r.success && r.inliers == 40 && r.median_error < 1e-6);
        const auto p = r.H.project(Point2D(100, 200));
        assert(p && std::abs(p->x - 120) < 1e-6 && std::abs(p->y - 210) < 1e-6);
    }
    { // wrong positions are matched but rejected by RANSAC
        std::mt19937 rng(2);
        const ImageFeatures q = random_features(rng, 50);
        ImageFeatures c = shifted(q, -15, 30);
        std::uniform_real_distribution<double> pos(0.0, 500.0);
        for (std::size_t i = 0; i < 10; ++i) c.points[i] = Point2D(pos(rng), pos(rng));   // 10 outliers
        const CandidateResult r = verify_candidate(q, c, 0.8, {.seed = 9});
        assert(r.matches == 50);
        assert(r.success && r.inliers == 40);
    }
    { // the result is exactly what matching + RANSAC give (noisy points: median clearly > 0)
        std::mt19937 rng(6);
        const ImageFeatures q = random_features(rng, 60);
        ImageFeatures c = shifted(q, 12, -8);
        std::uniform_real_distribution<double> jitter(-1.0, 1.0), pos(0.0, 500.0);
        for (Point2D& p : c.points) p = Point2D(p.x + jitter(rng), p.y + jitter(rng));
        for (std::size_t i = 0; i < 15; ++i) c.points[i] = Point2D(pos(rng), pos(rng));
        const RansacParams params{.tau = 3.0, .seed = 21};
        const CandidateResult r = verify_candidate(q, c, 0.8, params);
        const RansacResult direct = estimate_ransac(match_sift(q, c, 0.8), params);
        assert(r.success == direct.success && r.success);
        assert(r.inliers == direct.inliers.count() && r.inliers >= 45);
        assert(r.median_error == direct.inliers.median_error && r.median_error > 0.1);
        assert(r.H.m == direct.H.m);
    }
    { // failures: no second neighbour (0 matches) and fewer than 4 matches
        std::mt19937 rng(3);
        const ImageFeatures q = random_features(rng, 30);
        const CandidateResult none = verify_candidate(q, random_features(rng, 1), 0.8, {});
        assert(none.matches == 0 && !none.success && none.inliers == 0 && std::isnan(none.median_error));

        ImageFeatures three = shifted(q, 5, 5);
        three.descriptors = Matrix(0, kSiftDim);
        three.points.clear();
        for (std::size_t i = 0; i < 3; ++i) {                     // copies of 3 query descriptors
            three.descriptors.append_row(q.descriptors.row(i));
            three.points.push_back(q.points[i]);
        }
        three.descriptors.append_row(random_features(rng, 1).descriptors.row(0));   // so ni >= 2
        three.points.emplace_back(1, 1);
        const CandidateResult r3 = verify_candidate(q, three, 0.8, {});
        assert(r3.matches == 3 && !r3.success && r3.inliers == 0 && std::isnan(r3.median_error));
    }
    { // same seed, same result
        std::mt19937 rng(4);
        const ImageFeatures q = random_features(rng, 30);
        ImageFeatures c = shifted(q, 7, -3);
        for (std::size_t i = 0; i < 8; ++i) c.points[i] = Point2D(3.0 * i, 400.0 - i);
        const CandidateResult a = verify_candidate(q, c, 0.8, {.seed = 11});
        const CandidateResult b = verify_candidate(q, c, 0.8, {.seed = 11});
        assert(a.inliers == b.inliers && a.H.m == b.H.m);
    }
    { // rerank by inliers: count, then median error, then original rank; failures last
        const std::vector<CandidateResult> v = {
            cand(0, 50, false, 0, 0),     // failed
            cand(1, 30, true, 12, 1.5),
            cand(2, 40, true, 30, 2.0),
            cand(3, 35, true, 12, 0.5),   // ties rank 1 on inliers, smaller median
            cand(4, 20, true, 12, 1.5),   // ties rank 1 on inliers and median -> after it
            cand(5, 10, false, 0, 0),     // failed, after rank 0
        };
        const auto r = rerank_by_inliers(v);
        assert((ranks_of(r) == std::vector<std::size_t>{2, 3, 1, 4, 0, 5}));
        assert(r[0].inliers == 30 && r[0].row == 102);
        assert(std::isnan(r[4].median_error) && r[4].inliers == 0);
    }
    { // a failure keeps 0 inliers even if the field was left set, and never beats a success
        CandidateResult bad = cand(0, 99, false, 0, 0);
        bad.inliers = 500;
        const auto r = rerank_by_inliers({bad, cand(1, 4, true, 4, 2.9)});
        assert((ranks_of(r) == std::vector<std::size_t>{1, 0}));
    }
    { // rerank by matches: count, then original rank
        const std::vector<CandidateResult> v = {
            cand(0, 10, true, 8, 1), cand(1, 40, false, 0, 0), cand(2, 25, true, 20, 1),
            cand(3, 40, true, 30, 1), cand(4, 0, false, 0, 0),
        };
        assert((ranks_of(rerank_by_matches(v)) == std::vector<std::size_t>{1, 3, 2, 0, 4}));
    }
    { // the result does not depend on the input order, and nothing is lost
        std::vector<CandidateResult> v;
        std::mt19937 rng(5);
        for (std::size_t k = 0; k < 50; ++k) {
            const bool ok = rng() % 3 != 0;
            v.push_back(cand(k, rng() % 20, ok, 4 + rng() % 5, double(rng() % 4) * 0.5));
        }
        const auto expect_i = ranks_of(rerank_by_inliers(v)), expect_m = ranks_of(rerank_by_matches(v));
        for (int t = 0; t < 10; ++t) {
            std::shuffle(v.begin(), v.end(), rng);
            assert(ranks_of(rerank_by_inliers(v)) == expect_i);
            assert(ranks_of(rerank_by_matches(v)) == expect_m);
        }
        std::vector<std::size_t> sorted = expect_i;
        std::sort(sorted.begin(), sorted.end());
        for (std::size_t k = 0; k < 50; ++k) assert(sorted[k] == k);
    }
    { // rows_of keeps the order; empty input stays empty
        const auto r = rerank_by_inliers({cand(0, 5, true, 5, 1), cand(1, 9, true, 9, 1)});
        assert((rows_of(r) == std::vector<ImageIndex>{101, 100}));
        assert(rerank_by_inliers({}).empty() && rerank_by_matches({}).empty() && rows_of({}).empty());
    }
    return 0;
}