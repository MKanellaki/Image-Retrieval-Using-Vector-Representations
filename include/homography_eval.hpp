#pragma once
// Accuracy of the estimated homographies against the HPatches ground truth H_1->j
//
// Only relevant pairs are evaluated: query I1 and a database image Ij (2 <= j <= 6) of the
// same sequence. A relevant pair outside Top-C, a failed RANSAC and an invalid homography
// all count as failures in both rates and never get a corner error (it is never set to 0)
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "dataset.hpp"
#include "homography.hpp"
#include "rerank.hpp"
#include "types.hpp"

namespace image_search {

// A 3x3 homography stored as 9 whitespace-separated numbers, row-major (the H_1_j format)
// Throws if the file is missing, has a different number of values or is not invertible
Homography read_homography_file(const std::string& path);

// H_1->j for every HPatches row of db (id "<sequence>/<j>"), read from <hp_dir>/<sequence>/H_1_<j>
// MIRFlickr rows have no ground truth and are left out
std::unordered_map<ImageIndex, Homography> load_ground_truth(const std::string& hp_dir, const std::vector<ImageRef>& db);

// Valid estimate: RANSAC succeeded with at least 4 final inliers and H has finite
// elements and is numerically invertible
bool is_valid_estimate(const CandidateResult& c);

// One relevant pair of a query
struct PairEvaluation {
    ImageIndex row = 0;           // database row of Ij
    bool in_top_c = false;        // Ij was among the C candidates, so RANSAC ran on it
    bool valid = false;           // valid estimate (implies in_top_c)
    double corner_error = std::numeric_limits<double>::quiet_NaN();   // pixels of Ij, only if valid
};

// Evaluates the relevant pairs of one query, sorted by row
//   top_c         the verified candidates (any order)
//   relevant      the relevant rows of the query (relevant_rows)
//   ground_truth  load_ground_truth, must contain every relevant row
//   width, height size of the query image I1, its corners are (0,0) (w,0) (w,h) (0,h)
// A valid H that sends a corner of I1 to infinity has no corner error and is counted as invalid
std::vector<PairEvaluation> evaluate_relevant_pairs(
    const std::vector<CandidateResult>& top_c, const std::unordered_set<ImageIndex>& relevant,
    const std::unordered_map<ImageIndex, Homography>& ground_truth, double width, double height);

struct HomographySummary {
    std::size_t pairs = 0;        // denominator of the rates: all relevant pairs (23 * 5 = 115)
    std::size_t in_top_c = 0;
    std::size_t valid = 0;
    std::size_t accurate1 = 0, accurate3 = 0, accurate5 = 0;   // valid and E_corner < 1, 3, 5 px
    double mean_corner_error = std::numeric_limits<double>::quiet_NaN();   // over valid, NaN (NA) if none

    // count / pairs, 0 when there are no pairs
    double rate(std::size_t count) const;
};

// Works on any subset of pairs, so the same code gives overall, illumination and viewpoint
HomographySummary summarize_homographies(const std::vector<PairEvaluation>& pairs);

}  // namespace image_search