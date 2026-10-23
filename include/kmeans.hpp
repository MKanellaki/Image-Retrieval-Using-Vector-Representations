#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "types.hpp"

namespace image_search {

struct KMeansParams {
    std::size_t k = 0;
    int max_iter = 25;
    double tol = 1e-4;          // stop when the fraction of reassigned points falls below this
    std::uint64_t seed = 1;
};

struct KMeansResult {
    Matrix centers;                       // k x dim
    std::vector<std::uint32_t> labels;    // labels[i] = nearest center of data.row(i),
                                          // consistent with the returned centers
};

KMeansResult kmeans_fit(const Matrix& data, const KMeansParams& params);

// Convenience wrapper: centers only.
Matrix kmeans(const Matrix& data, const KMeansParams& params);

// Index of the nearest center to a single point. Requires centers.rows() > 0
// (checked with assert in debug builds).
std::size_t nearest_center(const float* point, const Matrix& centers);

// Nearest center for every row of `data`.
std::vector<std::uint32_t> assign_all(const Matrix& data, const Matrix& centers);

}  // namespace image_search