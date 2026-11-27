#include "rerank.hpp"
#include <algorithm>
#include <cmath>
#include "matching.hpp"
#include "seed.hpp"

namespace image_search {

std::uint64_t pair_seed(std::uint64_t seed, std::size_t query_index, ImageIndex row) {
    return derive_seed(derive_seed(seed, query_index), row);
}

CandidateResult verify_candidate(const ImageFeatures& query, const ImageFeatures& candidate,
                                 double rho, const RansacParams& ransac) {
    CandidateResult out;
    const std::vector<Match2D> matches = match_sift(query, candidate, rho);
    out.matches = matches.size();

    const RansacResult r = estimate_ransac(matches, ransac);   // fails by itself when M < 4
    if (r.success) {
        out.success = true;
        out.inliers = r.inliers.count();
        out.median_error = r.inliers.median_error;
        out.H = r.H;
    }
    return out;
}

namespace {

// Median errors compared with NaN (no homography) after every real value
bool smaller_median(double a, double b) {
    if (std::isnan(a)) return false;
    if (std::isnan(b)) return true;
    return a < b;
}

}  // namespace

std::vector<CandidateResult> rerank_by_inliers(std::vector<CandidateResult> candidates) {
    std::sort(candidates.begin(), candidates.end(), [](const CandidateResult& a, const CandidateResult& b) {
        const std::size_t ia = a.success ? a.inliers : 0, ib = b.success ? b.inliers : 0;
        if (ia != ib) return ia > ib;
        if (smaller_median(a.median_error, b.median_error)) return true;
        if (smaller_median(b.median_error, a.median_error)) return false;
        return a.original_rank < b.original_rank;
    });
    return candidates;
}

std::vector<CandidateResult> rerank_by_matches(std::vector<CandidateResult> candidates) {
    std::sort(candidates.begin(), candidates.end(), [](const CandidateResult& a, const CandidateResult& b) {
        if (a.matches != b.matches) return a.matches > b.matches;
        return a.original_rank < b.original_rank;
    });
    return candidates;
}

std::vector<ImageIndex> rows_of(const std::vector<CandidateResult>& candidates) {
    std::vector<ImageIndex> rows;
    rows.reserve(candidates.size());
    for (const CandidateResult& c : candidates) rows.push_back(c.row);
    return rows;
}

}  // namespace image_search