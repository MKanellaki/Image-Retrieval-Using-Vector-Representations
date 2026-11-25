#include "ransac.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <random>

namespace image_search {

namespace {

// Uniform integer in [0, bound), bound > 0, by rejection sampling
std::uint64_t uniform_below(std::mt19937_64& rng, std::uint64_t bound) {
    const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max()
                              - std::numeric_limits<std::uint64_t>::max() % bound;
    std::uint64_t x;
    do { x = rng(); } while (x >= limit);
    return x % bound;
}

// 4 distinct match indices in [0, n), n >= 4
std::array<std::size_t, 4> sample_four(std::mt19937_64& rng, std::size_t n) {
    std::array<std::size_t, 4> s{};
    for (std::size_t k = 0; k < 4; ++k) {
        do {
            s[k] = static_cast<std::size_t>(uniform_below(rng, n));
        } while (std::find(s.begin(), s.begin() + k, s[k]) != s.begin() + k);
    }
    return s;
}

struct Hypothesis {
    Homography H;
    InlierResult inliers;   // evaluated on ALL matches, median_error already filled in
};

// better(): more inliers; on a tie, smaller median inlier error
bool better(const InlierResult& cand, const std::optional<Hypothesis>& best) {
    if (!best) return true;
    if (cand.count() != best->inliers.count()) return cand.count() > best->inliers.count();
    return cand.median_error < best->inliers.median_error;   // NaN (0 inliers) never wins
}

}  // namespace

int adaptive_trial_limit(std::size_t inliers, std::size_t total, double p, int current_limit) {
    if (total == 0 || inliers == 0) return current_limit;   // w = 0: log(1) = 0, keep the limit
    if (inliers >= total) return 0;                         // w = 1: every match agrees, stop

    const double w = static_cast<double>(inliers) / static_cast<double>(total);
    const double denom = std::log1p(-std::pow(w, 4));       // log(1 - w^4), accurate for small w
    if (!(denom < 0.0)) return current_limit;               // w^4 underflowed to 0

    const double t_new = std::ceil(std::log1p(-p) / denom);
    if (!(t_new < static_cast<double>(current_limit))) return current_limit;   // also inf / NaN
    return static_cast<int>(t_new);
}

RansacResult estimate_ransac(const std::vector<Match2D>& matches, const RansacParams& params) {
    RansacResult res;
    const std::size_t M = matches.size();
    if (M < 4) return res;

    std::mt19937_64 rng(params.seed);
    std::optional<Hypothesis> best;
    int limit = params.T_max;

    std::vector<Point2D> pts1(4), pts2(4);   // reused every trial
    for (int t = 1; t <= params.T_max; ++t) {
        if (t > limit) break;
        res.trials = t;

        // 1. sample 4 distinct matches, reject degenerate quadruples in EITHER image
        const auto s = sample_four(rng, M);
        for (std::size_t k = 0; k < 4; ++k) { pts1[k] = matches[s[k]].p1; pts2[k] = matches[s[k]].p2; }
        if (is_degenerate_quad(pts1, pts2)) continue;

        // 2. minimal normalized DLT (returns nullopt if not finite / not invertible)
        const auto H = estimate_homography_dlt(pts1, pts2);
        if (!H) continue;

        // 3. inliers over all M matches
        InlierResult cand = evaluate_inliers(*H, matches, params.tau);

        // 4. keep the best hypothesis
        if (!better(cand, best)) continue;
        best = Hypothesis{*H, std::move(cand)};

        // 5. adaptive limit from w = |I_best| / M
        if (params.adaptive)
            limit = adaptive_trial_limit(best->inliers.count(), M, params.p, limit);
    }

    if (!best || best->inliers.count() < 4) return res;

    // 6. refit on the m = |I_best| inliers only, then final inliers over ALL M matches
    if (auto refit = refit_homography(matches, best->inliers.inlier_indices, params.tau)) {
        res.H = refit->H;
        res.inliers = std::move(refit->inliers);
    } else {
        // refit invalid (or < 4 final inliers): keep the best minimal-sample hypothesis,
        // whose inliers were already evaluated on all M matches
        res.refit_failed = true;
        res.H = best->H;
        res.inliers = std::move(best->inliers);
    }

    res.success = true;
    return res;
}

}  // namespace image_search