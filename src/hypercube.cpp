#include "hypercube.hpp"
#include "distance.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace image_search {

// h_i(p) = floor((v . p + t) / w)
int HypercubeIndex::compute_h(const float* p, const HashFunction& hf) const {
    float dot = 0.0f;
    for (std::size_t d = 0; d < dim_; ++d) {
        dot += p[d] * hf.v[d];
    }
    return static_cast<int>(std::floor((dot + hf.t) / params_.w));
}

// f_i: Z -> {0, 1} with memoized coin flips
int HypercubeIndex::compute_f(std::size_t func_idx, int h_val) const {
    auto& map = f_maps_[func_idx];
    auto it = map.find(h_val);
    if (it != map.end()) {
        return it->second;
    }
    std::uniform_int_distribution<int> coin(0, 1);
    const int bit = coin(rng_);
    map[h_val] = bit;
    return bit;
}

// Maps point p to a k-bit hypercube vertex
std::uint32_t HypercubeIndex::compute_vertex(const float* p) const {
    std::uint32_t vertex = 0;
    for (int i = 0; i < params_.k; ++i) {
        const int h_val = compute_h(p, funcs_[i]);
        const int bit = compute_f(i, h_val);
        if (bit) {
            vertex |= (1U << i);
        }
    }
    return vertex;
}

HypercubeIndex::HypercubeIndex(const Matrix& dataset, const HypercubeParams& params)
    : dataset_(&dataset), params_(params), dim_(dataset.cols()), rng_(params.seed) {

    // Empty dataset or invalid dimensions
    if (dataset.empty() || dim_ == 0 || params_.k <= 0) {
        return;
    }

    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni_dist(0.0f, static_cast<float>(params_.w));

    // Generate k projection functions
    funcs_.resize(params_.k);
    for (int i = 0; i < params_.k; ++i) {
        funcs_[i].v.resize(dim_);
        for (std::size_t d = 0; d < dim_; ++d) {
            funcs_[i].v[d] = norm_dist(rng_);
        }
        funcs_[i].t = uni_dist(rng_);
    }

    f_maps_.resize(params_.k);
    buckets_.resize(1U << params_.k);

    // Insert database images into hypercube vertex buckets
    for (std::size_t i = 0; i < dataset.rows(); ++i) {
        const float* point = dataset.row(i);
        const std::uint32_t vertex = compute_vertex(point);
        buckets_[vertex].push_back(static_cast<ImageIndex>(i));
    }
}

// Recursive helper to explore vertices with exactly 'remaining_bits' differences
static void explore_hamming(std::uint32_t base_vtx, int start, int remaining_bits, std::uint32_t current_mask,
                            int k, std::vector<std::uint32_t>& probe_list, int max_probes) {
    if (static_cast<int>(probe_list.size()) >= max_probes) {
        return;
    }
    if (remaining_bits == 0) {
        probe_list.push_back(base_vtx ^ current_mask);
        return;
    }
    for (int i = start; i <= k - remaining_bits; ++i) {
        explore_hamming(base_vtx, i + 1, remaining_bits - 1, current_mask | (1U << i),
                        k, probe_list, max_probes);
        if (static_cast<int>(probe_list.size()) >= max_probes) {
            return;
        }
    }
}

std::vector<ImageIndex> HypercubeIndex::query(const float* q, std::size_t max_results) const {
    if (!dataset_ || dataset_->empty() || params_.k <= 0 || params_.M <= 0 || params_.probes <= 0 || max_results == 0) {
        return {};
    }

    const std::uint32_t query_vtx = compute_vertex(q);

    // Collect list of hypercube vertices to probe in increasing Hamming distance
    std::vector<std::uint32_t> vertices_to_probe;
    for (int d = 0; d <= params_.k; ++d) {
        explore_hamming(query_vtx, 0, d, 0, params_.k, vertices_to_probe, params_.probes);
        if (static_cast<int>(vertices_to_probe.size()) >= params_.probes) {
            break;
        }
    }

    std::unordered_set<ImageIndex> seen;
    struct Candidate {
        ImageIndex id;
        float dist;
    };
    std::vector<Candidate> evaluated;

    for (std::uint32_t vtx : vertices_to_probe) {
        if (vtx >= buckets_.size()) {
            continue;
        }
        for (ImageIndex id : buckets_[vtx]) {
            if (seen.insert(id).second) {
                const float dist = l2_squared(dataset_->row(id), q, dim_);
                evaluated.push_back({id, dist});
                if (static_cast<int>(evaluated.size()) >= params_.M) {
                    break;
                }
            }
        }
        if (static_cast<int>(evaluated.size()) >= params_.M) {
            break;
        }
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

std::size_t HypercubeIndex::size() const {
    return dataset_ ? dataset_->rows() : 0;
}

}  // namespace image_search
