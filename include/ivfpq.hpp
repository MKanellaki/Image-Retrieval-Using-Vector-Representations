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
    // One coarse cluster. Entry e is ids[e]; its PQ code is the m_ sub-codes stored at
    // codes[e * m_ .. e * m_ + m_). Only one of the two code arrays is used: 8-bit codes
    // when the codebooks have at most 256 centers (nbits <= 8), 16-bit otherwise.
    struct InvertedList {
        std::vector<ImageIndex> ids;
        std::vector<std::uint8_t> codes8;
        std::vector<std::uint16_t> codes16;
    };

    const Matrix* dataset_ = nullptr;
    IVFPQParams params_;
    std::size_t dim_ = 0;
    std::size_t m_ = 0;       // Effective number of sub-quantizers
    std::size_t k_sub_ = 0;   // Effective codebook size (same for every sub-space)
    bool wide_codes_ = false; // true -> codes16, false -> codes8

    Matrix coarse_centroids_;                             // k coarse centroids in R^dim
    std::vector<Matrix> sub_centroids_;                   // m codebooks, each (k_sub x sub_dim)
    std::vector<std::size_t> sub_dims_;                   // Dimension of each sub-vector
    std::vector<std::size_t> sub_offsets_;                // Starting coordinate of each sub-vector
    std::vector<InvertedList> inverted_lists_;            // Inverted lists for each coarse centroid

    // Builds coarse and product quantizers
    void build_index();
};

}  // namespace image_search
