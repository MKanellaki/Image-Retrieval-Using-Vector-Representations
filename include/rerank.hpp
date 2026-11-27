#pragma once
// Geometric verification of the Top-C candidates of one query and the rankings built from it
//
// For every candidate: SIFT matches with Lowe's ratio test, then RANSAC on those matches
// A candidate whose verification fails (fewer than 4 matches, no valid homography) gets
// 0 inliers and no median error (written as NA)
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "features.hpp"
#include "homography.hpp"
#include "ransac.hpp"
#include "types.hpp"

namespace image_search {

struct CandidateResult {
    ImageIndex row = 0;               // database row of the candidate
    std::size_t original_rank = 0;    // 0-based position in the retrieval ranking
    std::size_t matches = 0;          // accepted SIFT matches (M)
    bool success = false;             // RANSAC found a valid homography
    std::size_t inliers = 0;          // final inliers, 0 when !success
    double median_error = std::numeric_limits<double>::quiet_NaN();   // NaN (NA) when !success
    Homography H;                     // query -> candidate, meaningful only when success
};

// Seed of the RANSAC run for one (query, candidate) pair. Depends on the database row,
// not on the rank, so the same pair gets the same seed for every C and every method
std::uint64_t pair_seed(std::uint64_t seed, std::size_t query_index, ImageIndex row);

// Matches query against candidate with ratio rho, then runs RANSAC with `ransac`
// (its seed field is used as given). row and original_rank are left for the caller
CandidateResult verify_candidate(const ImageFeatures& query, const ImageFeatures& candidate,
                                 double rho, const RansacParams& ransac);

// Final ranking: more inliers first; on a tie the smaller median error; then the original
// rank. Failed candidates count as 0 inliers and lose every median-error tie
std::vector<CandidateResult> rerank_by_inliers(std::vector<CandidateResult> candidates);

// Baseline for the experiments: more accepted matches first, then the original rank
std::vector<CandidateResult> rerank_by_matches(std::vector<CandidateResult> candidates);

// The database rows in the given order (input for evaluate_query)
std::vector<ImageIndex> rows_of(const std::vector<CandidateResult>& candidates);

}  // namespace image_search