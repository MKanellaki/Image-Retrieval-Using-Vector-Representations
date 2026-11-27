#pragma once
// SIFT descriptor matching with Lowe's ratio test (own implementation, no OpenCV matcher)
//
// For every query descriptor q_i, find its two nearest candidate descriptors by Euclidean
// distance, d1 <= d2, and accept (p_i, p'_j1) when d1 < rho * d2, 0 < rho < 1
// Compared on squared distances (D1 < rho^2 * D2), which is the same test without sqrt
#include <cstddef>
#include <vector>

#include "features.hpp"
#include "homography.hpp"   // Match2D

namespace image_search {

// Matches in query order: matches[k].p1 is in the query, matches[k].p2 in the candidate
// A candidate with fewer than 2 descriptors has no second neighbour, so nothing passes
// Ties: d1 == d2 is ambiguous and rejected (strict <); among equal nearest distances the
// lowest candidate index is taken, so the result is deterministic
// Throws std::invalid_argument if rho is not in (0, 1) or descriptor widths differ
std::vector<Match2D> match_sift(const ImageFeatures& query, const ImageFeatures& candidate, double rho);

}  // namespace image_search