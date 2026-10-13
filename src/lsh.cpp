#include "lsh.hpp"
#include "distance.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace image_search {

// h(p) = floor((v . p + t) / w)
int LSHIndex::compute_h(const float* p, const HashFunction& hf) const {
    float dot = 0.0f;
    for (std::size_t d = 0; d < dim_; ++d) {
        dot += p[d] * hf.v[d];
    }
    return static_cast<int>(std::floor((dot + hf.t) / params_.w));
}

// Combines the k hash values into a single 64-bit key
std::uint64_t LSHIndex::compute_g(const float* p, const HashTable& table) const {
    std::uint64_t key = 0;
    for (const auto& hf : table.funcs) {
        const int h_val = compute_h(p, hf);
        key = key * 31ULL + static_cast<std::uint64_t>(static_cast<std::int64_t>(h_val));
    }
    return key;
}

LSHIndex::LSHIndex(const Matrix& dataset, const LSHParams& params)
    : dataset_(&dataset), params_(params), dim_(dataset.cols()) {

    // Empty input
    if (dataset.empty() || dim_ == 0) {
        return;
    }

    std::mt19937 rng(params_.seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni_dist(0.0f, static_cast<float>(params_.w));

    // Generate L hash tables with k functions each
    tables_.resize(params_.L);
    for (int l = 0; l < params_.L; ++l) {
        tables_[l].funcs.resize(params_.k);
        for (int i = 0; i < params_.k; ++i) {
            tables_[l].funcs[i].v.resize(dim_);
            for (std::size_t d = 0; d < dim_; ++d) {
                tables_[l].funcs[i].v[d] = norm_dist(rng);
            }
            tables_[l].funcs[i].t = uni_dist(rng);
        }
    }

    // Insert database images into the hash tables
    for (std::size_t i = 0; i < dataset.rows(); ++i) {
        const float* point = dataset.row(i);
        for (int l = 0; l < params_.L; ++l) {
            const std::uint64_t key = compute_g(point, tables_[l]);
            tables_[l].buckets[key].push_back(static_cast<ImageIndex>(i));
        }
    }
}

std::vector<ImageIndex> LSHIndex::query(const float* q, std::size_t max_results) const {
    // Nothing to search
    if (!dataset_ || dataset_->empty()) {
        return {};
    }

    // Collect candidates across all L tables without duplicates
    std::unordered_set<ImageIndex> candidates;
    for (int l = 0; l < params_.L; ++l) {
        const std::uint64_t key = compute_g(q, tables_[l]);
        auto it = tables_[l].buckets.find(key);
        if (it != tables_[l].buckets.end()) {
            for (ImageIndex id : it->second) {
                candidates.insert(id);
            }
        }
    }

    struct Candidate {
        ImageIndex id;
        float dist;
    };

    // Compute exact distances to candidate points
    std::vector<Candidate> evaluated;
    evaluated.reserve(candidates.size());
    for (ImageIndex id : candidates) {
        const float dist = l2_squared(dataset_->row(id), q, dim_);
        evaluated.push_back({id, dist});
    }

    // Sort candidates by ascending distance
    std::sort(evaluated.begin(), evaluated.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.dist < b.dist;
              });

    // Return top results up to max_results
    const std::size_t count = std::min(max_results, evaluated.size());
    std::vector<ImageIndex> results;
    results.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        results.push_back(evaluated[i].id);
    }

    return results;
}

std::size_t LSHIndex::size() const {
    return dataset_ ? dataset_->rows() : 0;
}

}  // namespace image_search