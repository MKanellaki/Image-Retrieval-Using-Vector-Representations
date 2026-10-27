#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "types.hpp"

namespace image_search {

struct IVFFlatParams {
    int k = 10;              // Number of clusters (centroids / inverted lists)
    int nprobe = 2;          // Number of closest centroids to probe
    int max_iter = 15;       // Maximum iterations for k-means
    std::uint32_t seed = 1;  // Random seed for reproducibility
};

class IVFFlatIndex {
public:
    IVFFlatIndex() = default;
    IVFFlatIndex(const Matrix& dataset, const IVFFlatParams& params = {});

    // Returns up to max_results nearest ImageIndex results, sorted best (closest) first
    std::vector<ImageIndex> query(const float* q, std::size_t max_results = 10) const;

    // Number of points in the index
    std::size_t size() const;

private:
    const Matrix* dataset_ = nullptr;
    IVFFlatParams params_;
    std::size_t dim_ = 0;
    Matrix centroids_;                                    // Centroids in R^d
    std::vector<std::vector<ImageIndex>> inverted_lists_; // Inverted lists for each centroid
};

}  // namespace image_search
