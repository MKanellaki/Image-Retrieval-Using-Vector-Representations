#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "types.hpp"

namespace image_search {

struct HypercubeParams {
    int k = 10;              // Dimension of the hypercube (d')
    int M = 500;             // Maximum candidate points to evaluate
    int probes = 10;         // Maximum hypercube vertices to probe
    double w = 0.05;         // Window size on the projection line
    std::uint32_t seed = 1;  // Random seed for reproducibility
};

class HypercubeIndex {
public:
    HypercubeIndex() = default;
    HypercubeIndex(const Matrix& dataset, const HypercubeParams& params = {});

    // Returns up to max_results nearest ImageIndex results, sorted best (closest) first
    std::vector<ImageIndex> query(const float* q, std::size_t max_results = 10) const;

    // Number of points in the index
    std::size_t size() const;

private:
    struct HashFunction {
        std::vector<float> v; // Random Gaussian vector in R^d: v_j ~ N(0, 1)
        float t = 0.0f;       // Random shift: t ~ U[0, w)
    };

    const Matrix* dataset_ = nullptr;
    HypercubeParams params_;
    std::size_t dim_ = 0;
    std::vector<HashFunction> funcs_; // k projection functions

    // Hypercube vertices: 2^k buckets
    std::vector<std::vector<ImageIndex>> buckets_;

    // Computes single hash value: h_i(p) = floor((v . p + t) / w)
    int compute_h(const float* p, const HashFunction& hf) const;

    // Computes bit f_i(h_i(p)) in {0, 1}: a pure function of (seed, i, h)
    int compute_f(std::size_t func_idx, int h_val) const;

    // Maps point p to a k-bit hypercube vertex index
    std::uint32_t compute_vertex(const float* p) const;
};

}  // namespace image_search
