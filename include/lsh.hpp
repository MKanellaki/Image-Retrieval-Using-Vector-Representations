#pragma once
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "types.hpp"

namespace image_search {

struct LSHParams {
    int k = 4;              // Number of hash functions per table g
    int L = 5;              // Number of hash tables
    double w = 0.05;        // Window size on the projection line
    std::uint32_t seed = 1; // Random seed for reproducibility
};

class LSHIndex {
public:
    LSHIndex() = default;
    LSHIndex(const Matrix& dataset, const LSHParams& params = {});

    // Binding to a temporary would leave a dangling pointer, so it is rejected
    // at compile time, with or without params.
    LSHIndex(Matrix&&, const LSHParams& = {}) = delete;

    // Returns up to max_results nearest ImageIndex results, sorted best (closest) first.
    // Ties in distance are broken by ascending row index, so output is deterministic.
    std::vector<ImageIndex> query(const float* q, std::size_t max_results = 10) const;

    // Number of points in the index
    std::size_t size() const;

private:
    struct HashFunction {
        std::vector<float> v; // Random Gaussian vector in R^d: v_j ~ N(0, 1)
        float t = 0.0f;       // Random shift: t ~ U[0, w)
    };

    struct HashTable {
        std::vector<HashFunction> funcs; // k hash functions h_1, ..., h_k
        // Hash table: maps 64-bit composite key g(p) -> list of point IDs
        std::unordered_map<std::uint64_t, std::vector<ImageIndex>> buckets;
    };

    const Matrix* dataset_ = nullptr;   // not owned; must outlive the index
    LSHParams params_;
    std::size_t dim_ = 0;
    std::vector<HashTable> tables_; // L hash tables

    // Computes single hash value: h_i(p) = floor((v . p + t) / w)
    int compute_h(const float* p, const HashFunction& hf) const;

    // Combines k hash values into a 64-bit composite bucket key g(p)
    std::uint64_t compute_g(const float* p, const HashTable& table) const;
};

}  // namespace image_search
