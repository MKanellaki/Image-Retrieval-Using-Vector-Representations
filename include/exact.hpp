#pragma once
#include <cstddef>
#include <vector>
#include "types.hpp"

namespace image_search {

// Exhaustive (brute-force) nearest-neighbour search under the Euclidean metric.
// Computes the distance from the query to every database vector and keeps the
// best max_results. Serves as the ground-truth baseline for LSH, Hypercube,
// IVFFlat and IVFPQ. Has no parameters and no randomness.
class ExactIndex {
public:
    ExactIndex() = default;

    // Stores a pointer to `dataset`, which must outlive the index.
    explicit ExactIndex(const Matrix& dataset);

    // Binding to a temporary would leave a dangling pointer, so it is rejected
    // at compile time: ExactIndex ex(make_database(...)) does not compile.
    ExactIndex(Matrix&&) = delete;

    // Returns the min(max_results, size()) nearest rows, sorted best (closest) first,
    // so the result has fewer than max_results entries when the database is smaller
    // (and is empty for a default-constructed index or max_results == 0).
    // `q` must point to dim() floats. Ties in distance are broken by ascending
    // row index, so output is deterministic.
    std::vector<ImageIndex> query(const float* q, std::size_t max_results = 10) const;

    // Number of points in the index
    std::size_t size() const;

    // Dimension of the indexed vectors; a query must have exactly this many floats.
    std::size_t dim() const { return dim_; }

private:
    const Matrix* dataset_ = nullptr;   // not owned; must outlive the index
    std::size_t dim_ = 0;
};

}  // namespace image_search