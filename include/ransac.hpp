#pragma once
// RANSAC homography estimation over accepted SIFT matches.
// Builds on homography.hpp: normalized DLT, degeneracy checks, inlier evaluation and refit.
#include <cstddef>
#include <cstdint>
#include <vector>
#include "homography.hpp"

namespace image_search {

struct RansacParams {
    double tau = 3.0;           // inlier iff e_i(H) < tau (pixels)
    int    T_max = 2000;        // maximum number of trials
    double p = 0.99;            // desired probability of at least one clean sample
    bool   adaptive = true;     // shrink the trial limit from the best inlier ratio
    std::uint64_t seed = 1;     // use derive_seed(...) per (query, candidate) pair
};

struct RansacResult {
    bool success = false;       // false: M < 4, no valid hypothesis, or < 4 final inliers
    Homography H;               // final (refitted) homography, meaningful only if success
    InlierResult inliers;       // final inliers over ALL matches: indices, errors, median_error
    int trials = 0;             // trials run, including rejected (degenerate/invalid) samples
    bool refit_failed = false;  // refit gave no usable H, H is the best minimal-sample hypothesis
};

RansacResult estimate_ransac(const std::vector<Match2D>& matches, const RansacParams& params = {});

// New trial limit after the best hypothesis improved (exposed for tests).
//   w = 0  -> current_limit unchanged
//   w = 1  -> 0 (stop sampling, go to refit)
//   else   -> min(current_limit, ceil(log(1-p) / log(1-w^4)))
int adaptive_trial_limit(std::size_t inliers, std::size_t total, double p, int current_limit);

}  // namespace image_search