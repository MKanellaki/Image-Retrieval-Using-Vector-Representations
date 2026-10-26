#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "types.hpp"

namespace image_search {

struct IVFPQParams {
    int k = 10;              // Number of coarse clusters (centroids / inverted lists)
    int m = 4;               // Number of sub-quantizers (sub-vectors)
    int k_sub = 256;         // Codebook size per sub-quantizer
    int nprobe = 2;          // Number of closest coarse centroids to probe
    int max_iter = 15;       // Maximum iterations for k-means
    bool rerank = false;     // Whether to rerank candidates with exact distance
    std::uint32_t seed = 1;  // Random seed for reproducibility
};

class IVFPQIndex {
public:
    IVFPQIndex() = default;
    IVFPQIndex(const Matrix& dataset, const IVFPQParams& params = {});

    // Returns up to max_results nearest ImageIndex results, sorted best (closest) first
    std::vector<ImageIndex> query(const float* q, std::size_t max_results = 10) const;

    // Number of points in the index
    std::size_t size() const;

private:
    struct PQEntry {
        ImageIndex id;
        std::vector<int> code; // m sub-vector codebook indices
    };

    const Matrix* dataset_ = nullptr;
    IVFPQParams params_;
    std::size_t dim_ = 0;
    std::size_t m_ = 0;      // Effective number of sub-quantizers

    Matrix coarse_centroids_;                             // k coarse centroids in R^dim
    std::vector<Matrix> sub_centroids_;                   // m codebooks, each (k_sub x sub_dim)
    std::vector<std::size_t> sub_dims_;                   // Dimension of each sub-vector
    std::vector<std::size_t> sub_offsets_;                // Starting coordinate of each sub-vector
    std::vector<std::vector<PQEntry>> inverted_lists_;    // Inverted lists for each coarse centroid

    // Finds closest coarse centroid index
    int find_closest_coarse_centroid(const float* p) const;

    // Runs k-means on a set of sub-vectors
    Matrix run_kmeans_sub(const std::vector<const float*>& points, std::size_t sub_dim, int k, std::uint32_t seed);

    // Builds coarse and product quantizers
    void build_index();
};

}  // namespace image_search
