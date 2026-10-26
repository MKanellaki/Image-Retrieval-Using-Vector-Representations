#include "exact.hpp"

#include <algorithm>
#include <queue>
#include <utility>

#include "distance.hpp"

namespace image_search {

ExactIndex::ExactIndex(const Matrix& dataset)
    : dataset_(&dataset), dim_(dataset.cols()) {}

std::vector<ImageIndex> ExactIndex::query(const float* q, std::size_t max_results) const {
    if (!dataset_ || max_results == 0) {
        return {};
    }

    const std::size_t n = dataset_->rows();
    const std::size_t k = std::min(max_results, n);

    // Max-heap of the k best (distance, id) seen so far; top() is the worst of them.
    // Pair comparison orders by distance first, then id, giving a deterministic tie-break.
    using Entry = std::pair<float, ImageIndex>;
    std::priority_queue<Entry> heap;

    for (std::size_t i = 0; i < n; ++i) {
        // Squared L2 preserves the ordering of L2, so the sqrt is unnecessary.
        const float dist = l2_squared(dataset_->row(i), q, dim_);
        const Entry e{dist, static_cast<ImageIndex>(i)};
        if (heap.size() < k) {
            heap.push(e);
        } else if (e < heap.top()) {
            heap.pop();
            heap.push(e);
        }
    }

    // Drain the heap (worst first) into the result back-to-front -> closest first.
    std::vector<ImageIndex> results(heap.size());
    for (std::size_t pos = results.size(); pos-- > 0;) {
        results[pos] = heap.top().second;
        heap.pop();
    }
    return results;
}

std::size_t ExactIndex::size() const {
    return dataset_ ? dataset_->rows() : 0;
}

}  // namespace image_search