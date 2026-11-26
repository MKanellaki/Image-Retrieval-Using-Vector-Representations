#include <cassert>
#include <cmath>
#include <vector>
#include "homography.hpp"
#include "ransac.hpp"
using namespace image_search;

namespace {
bool close(double a, double b, double eps = 1e-6) { return std::abs(a - b) < eps; }

// Compare up to scale: divide both by their h33
bool same_homography(const Homography& A, const Homography& B) {
    for (std::size_t i = 0; i < 9; ++i)
        if (!close(A.m[i] / A.m[8], B.m[i] / B.m[8])) return false;
    return true;
}
}  // namespace

int main() {
    const Homography shift(1, 0, 20,  0, 1, 10,  0, 0, 1);

    // 5 correct matches of (x, y) -> (x + 20, y + 10) + 2 wrong ones (tutorial slide 15)
    const std::vector<Point2D> pts = {{0, 0}, {100, 0}, {0, 100}, {100, 100}, {50, 30}};
    std::vector<Match2D> m;
    for (const Point2D& p : pts) m.emplace_back(p, Point2D(p.x + 20, p.y + 10));
    m.emplace_back(30, 30, 200, 150);
    m.emplace_back(70, 50, 15, 180);

    { // adaptive limit
        assert(adaptive_trial_limit(1, 2, 0.99, 2000) == 72);     // w = 0.5 (lecture example)
        assert(adaptive_trial_limit(1, 2, 0.99, 50) == 50);       // never increases
        assert(adaptive_trial_limit(0, 10, 0.99, 2000) == 2000);  // w = 0
        assert(adaptive_trial_limit(10, 10, 0.99, 2000) == 0);    // w = 1
    }
    { // finds the 5 inliers, rejects the 2 outliers, recovers the translation
        const RansacResult r = estimate_ransac(m, {.tau = 3, .T_max = 2000, .p = 0.99, .adaptive = true, .seed = 1});
        assert(r.success && !r.refit_failed);
        assert((r.inliers.inlier_indices == std::vector<std::size_t>{0, 1, 2, 3, 4}));
        assert(same_homography(r.H, shift));
        assert(r.inliers.median_error < 1e-6);
        assert(r.trials < 2000);    // adaptive limit stopped early
    }
    { // same for many seeds, with a fixed trial count (the adaptive limit only gives success
      // with probability ~p, so some seeds may legitimately stop early with a worse model)
        for (std::uint64_t seed = 0; seed < 200; ++seed) {
            const RansacResult r = estimate_ransac(m, {.T_max = 500, .adaptive = false, .seed = seed});
            assert(r.success && r.inliers.count() == 5);
        }
    }
    { // all matches are inliers: w = 1 stops sampling after the first valid hypothesis
        const std::vector<Match2D> clean(m.begin(), m.begin() + 5);
        const RansacResult r = estimate_ransac(clean, {.seed = 3});
        assert(r.success && r.inliers.count() == 5 && r.trials < 2000);
    }
    { // without the adaptive limit all T_max trials run
        const RansacResult r = estimate_ransac(m, {.T_max = 300, .adaptive = false});
        assert(r.success && r.trials == 300);
    }
    { // reproducibility
        const RansacResult a = estimate_ransac(m, {.seed = 7});
        const RansacResult b = estimate_ransac(m, {.seed = 7});
        assert(a.H.m == b.H.m && a.inliers.inlier_indices == b.inliers.inlier_indices && a.trials == b.trials);
    }
    { // failures: fewer than 4 matches, every quadruple degenerate (all points collinear)
        assert(!estimate_ransac({m.begin(), m.begin() + 3}).success);
        std::vector<Match2D> line;
        for (int i = 0; i < 10; ++i) line.emplace_back(i, i, i, 2.0 * i);
        const RansacResult r = estimate_ransac(line, {.T_max = 100});
        assert(!r.success && r.trials == 100);
    }
    return 0;
}