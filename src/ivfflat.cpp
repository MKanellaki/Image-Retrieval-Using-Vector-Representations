#include "ivfflat.hpp"
#include "distance.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace image_search {

int IVFFlatIndex::find_closest_centroid(const float* p) const {
    int best_c = 0;
    float best_dist = l2_squared(p, centroids_.row(0), dim_);
    for (std::size_t c = 1; c < centroids_.rows(); ++c) {
        float d = l2_squared(p, centroids_.row(c), dim_);
        if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
        }
    }
    return best_c;
}

void IVFFlatIndex::run_kmeans() {
    const std::size_t n = dataset_->rows();
    const std::size_t k = std::min(static_cast<std::size_t>(params_.k), n);

    centroids_ = Matrix(k, dim_);
    std::mt19937 rng(params_.seed);

    // If dataset has fewer points than k, each point is its own centroid
    if (k == n) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t d = 0; d < dim_; ++d) {
                centroids_.row(i)[d] = dataset_->row(i)[d];
            }
        }
        return;
    }

    // k-means++ initialization
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);
    std::size_t first = dist(rng);
    for (std::size_t d = 0; d < dim_; ++d) {
        centroids_.row(0)[d] = dataset_->row(first)[d];
    }

    std::vector<float> min_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        min_d[i] = l2_squared(dataset_->row(i), centroids_.row(0), dim_);
    }

    for (std::size_t c = 1; c < k; ++c) {
        float sum_d = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            sum_d += min_d[i];
        }

        std::size_t chosen = 0;
        if (sum_d > 0.0f) {
            std::uniform_real_distribution<float> rdist(0.0f, sum_d);
            float r = rdist(rng);
            float current = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                current += min_d[i];
                if (current >= r) {
                    chosen = i;
                    break;
                }
            }
        } else {
            chosen = dist(rng);
        }

        for (std::size_t d = 0; d < dim_; ++d) {
            centroids_.row(c)[d] = dataset_->row(chosen)[d];
        }

        for (std::size_t i = 0; i < n; ++i) {
            float d = l2_squared(dataset_->row(i), centroids_.row(c), dim_);
            if (d < min_d[i]) {
                min_d[i] = d;
            }
        }
    }

    // Lloyd's algorithm
    std::vector<int> assignments(n, -1);

    for (int iter = 0; iter < params_.max_iter; ++iter) {
        bool changed = false;

        // E-step: assign points to closest centroid
        for (std::size_t i = 0; i < n; ++i) {
            int nearest = find_closest_centroid(dataset_->row(i));
            if (assignments[i] != nearest) {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if (!changed && iter > 0) {
            break;
        }

        // M-step: update centroids
        std::vector<std::vector<float>> sums(k, std::vector<float>(dim_, 0.0f));
        std::vector<int> counts(k, 0);

        for (std::size_t i = 0; i < n; ++i) {
            int c = assignments[i];
            counts[c]++;
            for (std::size_t d = 0; d < dim_; ++d) {
                sums[c][d] += dataset_->row(i)[d];
            }
        }

        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (std::size_t d = 0; d < dim_; ++d) {
                    centroids_.row(c)[d] = sums[c][d] / counts[c];
                }
            }
        }
    }
}

IVFFlatIndex::IVFFlatIndex(const Matrix& dataset, const IVFFlatParams& params)
    : dataset_(&dataset), params_(params), dim_(dataset.cols()) {

    // Empty input
    if (dataset.empty() || dim_ == 0 || params_.k <= 0) {
        return;
    }

    // Compute cluster centroids with k-means
    run_kmeans();

    // Insert database images into inverted lists
    inverted_lists_.resize(centroids_.rows());
    for (std::size_t i = 0; i < dataset.rows(); ++i) {
        int c = find_closest_centroid(dataset.row(i));
        inverted_lists_[c].push_back(static_cast<ImageIndex>(i));
    }
}

std::vector<ImageIndex> IVFFlatIndex::query(const float* q, std::size_t max_results) const {
    // Nothing to search
    if (!dataset_ || dataset_->empty() || centroids_.empty() || max_results == 0) {
        return {};
    }

    // Compute distance to all centroids
    struct CentroidDist {
        int id;
        float dist;
    };
    std::vector<CentroidDist> c_dists;
    c_dists.reserve(centroids_.rows());
    for (std::size_t c = 0; c < centroids_.rows(); ++c) {
        float d = l2_squared(q, centroids_.row(c), dim_);
        c_dists.push_back({static_cast<int>(c), d});
    }

    // Sort centroids by distance to query
    std::sort(c_dists.begin(), c_dists.end(),
              [](const CentroidDist& a, const CentroidDist& b) {
                  return a.dist < b.dist;
              });

    // Probe the top nprobe closest centroids
    std::size_t probes = std::min(static_cast<std::size_t>(params_.nprobe), centroids_.rows());

    struct Candidate {
        ImageIndex id;
        float dist;
    };

    // Compute exact distances to candidate points in probed lists
    std::vector<Candidate> evaluated;
    for (std::size_t p = 0; p < probes; ++p) {
        int c_id = c_dists[p].id;
        for (ImageIndex id : inverted_lists_[c_id]) {
            float dist = l2_squared(dataset_->row(id), q, dim_);
            evaluated.push_back({id, dist});
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

std::size_t IVFFlatIndex::size() const {
    return dataset_ ? dataset_->rows() : 0;
}

}  
