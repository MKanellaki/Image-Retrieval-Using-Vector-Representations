#include "ivfpq.hpp"
#include "distance.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace image_search {

int IVFPQIndex::find_closest_coarse_centroid(const float* p) const {
    int best_c = 0;
    float best_dist = l2_squared(p, coarse_centroids_.row(0), dim_);
    for (std::size_t c = 1; c < coarse_centroids_.rows(); ++c) {
        float d = l2_squared(p, coarse_centroids_.row(c), dim_);
        if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
        }
    }
    return best_c;
}

Matrix IVFPQIndex::run_kmeans_sub(const std::vector<const float*>& points, std::size_t sub_dim, int k, std::uint32_t seed) {
    const std::size_t n = points.size();
    const std::size_t actual_k = std::min(static_cast<std::size_t>(k), n);

    Matrix centroids(actual_k, sub_dim);
    std::mt19937 rng(seed);

    if (actual_k == n) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t d = 0; d < sub_dim; ++d) {
                centroids.row(i)[d] = points[i][d];
            }
        }
        return centroids;
    }

    // k-means++ initialization
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);
    std::size_t first = dist(rng);
    for (std::size_t d = 0; d < sub_dim; ++d) {
        centroids.row(0)[d] = points[first][d];
    }

    std::vector<float> min_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        min_d[i] = l2_squared(points[i], centroids.row(0), sub_dim);
    }

    for (std::size_t c = 1; c < actual_k; ++c) {
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

        for (std::size_t d = 0; d < sub_dim; ++d) {
            centroids.row(c)[d] = points[chosen][d];
        }

        for (std::size_t i = 0; i < n; ++i) {
            float d = l2_squared(points[i], centroids.row(c), sub_dim);
            if (d < min_d[i]) {
                min_d[i] = d;
            }
        }
    }

    // Lloyd's iterations
    std::vector<int> assignments(n, -1);
    for (int iter = 0; iter < params_.max_iter; ++iter) {
        bool changed = false;

        // E-step: assign points to closest sub-centroid
        for (std::size_t i = 0; i < n; ++i) {
            int best_c = 0;
            float best_dist = l2_squared(points[i], centroids.row(0), sub_dim);
            for (std::size_t c = 1; c < actual_k; ++c) {
                float d = l2_squared(points[i], centroids.row(c), sub_dim);
                if (d < best_dist) {
                    best_dist = d;
                    best_c = static_cast<int>(c);
                }
            }
            if (assignments[i] != best_c) {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if (!changed && iter > 0) {
            break;
        }

        // M-step: update sub-centroids
        std::vector<std::vector<float>> sums(actual_k, std::vector<float>(sub_dim, 0.0f));
        std::vector<int> counts(actual_k, 0);

        for (std::size_t i = 0; i < n; ++i) {
            int c = assignments[i];
            counts[c]++;
            for (std::size_t d = 0; d < sub_dim; ++d) {
                sums[c][d] += points[i][d];
            }
        }

        for (std::size_t c = 0; c < actual_k; ++c) {
            if (counts[c] > 0) {
                for (std::size_t d = 0; d < sub_dim; ++d) {
                    centroids.row(c)[d] = sums[c][d] / counts[c];
                }
            }
        }
    }

    return centroids;
}

void IVFPQIndex::build_index() {
    const std::size_t n = dataset_->rows();

    // 1. Determine sub-vector dimensions and offsets
    m_ = std::max<std::size_t>(1, std::min(static_cast<std::size_t>(params_.m), dim_));
    sub_dims_.resize(m_);
    sub_offsets_.resize(m_);

    std::size_t base_dim = dim_ / m_;
    std::size_t rem = dim_ % m_;
    std::size_t curr_offset = 0;
    for (std::size_t s = 0; s < m_; ++s) {
        sub_offsets_[s] = curr_offset;
        sub_dims_[s] = base_dim + (s < rem ? 1 : 0);
        curr_offset += sub_dims_[s];
    }

    // 2. Coarse clustering on the full dataset
    std::vector<const float*> all_points(n);
    for (std::size_t i = 0; i < n; ++i) {
        all_points[i] = dataset_->row(i);
    }
    coarse_centroids_ = run_kmeans_sub(all_points, dim_, params_.k, params_.seed);

    // 3. Find coarse cluster assignments and compute residuals
    std::vector<int> coarse_assignments(n);
    std::vector<std::vector<float>> residuals(n, std::vector<float>(dim_));

    for (std::size_t i = 0; i < n; ++i) {
        int c = find_closest_coarse_centroid(dataset_->row(i));
        coarse_assignments[i] = c;
        const float* pt = dataset_->row(i);
        const float* center = coarse_centroids_.row(c);
        for (std::size_t d = 0; d < dim_; ++d) {
            residuals[i][d] = pt[d] - center[d];
        }
    }

    // 4. Train codebooks for each sub-space on residuals
    sub_centroids_.resize(m_);
    int effective_k_sub = std::max(1, std::min(params_.k_sub, static_cast<int>(n)));

    for (std::size_t s = 0; s < m_; ++s) {
        std::vector<const float*> sub_points(n);
        for (std::size_t i = 0; i < n; ++i) {
            sub_points[i] = residuals[i].data() + sub_offsets_[s];
        }
        sub_centroids_[s] = run_kmeans_sub(sub_points, sub_dims_[s], effective_k_sub, params_.seed + s * 13 + 7);
    }

    // 5. Quantize residuals and populate inverted lists
    inverted_lists_.resize(coarse_centroids_.rows());
    for (std::size_t i = 0; i < n; ++i) {
        int c = coarse_assignments[i];
        std::vector<int> code(m_);

        for (std::size_t s = 0; s < m_; ++s) {
            const float* sub_res = residuals[i].data() + sub_offsets_[s];
            int best_sub = 0;
            float min_d = l2_squared(sub_res, sub_centroids_[s].row(0), sub_dims_[s]);
            for (std::size_t j = 1; j < sub_centroids_[s].rows(); ++j) {
                float d = l2_squared(sub_res, sub_centroids_[s].row(j), sub_dims_[s]);
                if (d < min_d) {
                    min_d = d;
                    best_sub = static_cast<int>(j);
                }
            }
            code[s] = best_sub;
        }

        inverted_lists_[c].push_back({static_cast<ImageIndex>(i), code});
    }
}

IVFPQIndex::IVFPQIndex(const Matrix& dataset, const IVFPQParams& params)
    : dataset_(&dataset), params_(params), dim_(dataset.cols()) {

    // Empty input
    if (dataset.empty() || dim_ == 0 || params_.k <= 0) {
        return;
    }

    build_index();
}

std::vector<ImageIndex> IVFPQIndex::query(const float* q, std::size_t max_results) const {
    // Nothing to search
    if (!dataset_ || dataset_->empty() || coarse_centroids_.empty() || max_results == 0) {
        return {};
    }

    // Compute distance to coarse centroids
    struct CentroidDist {
        int id;
        float dist;
    };
    std::vector<CentroidDist> c_dists;
    c_dists.reserve(coarse_centroids_.rows());
    for (std::size_t c = 0; c < coarse_centroids_.rows(); ++c) {
        float d = l2_squared(q, coarse_centroids_.row(c), dim_);
        c_dists.push_back({static_cast<int>(c), d});
    }

    // Sort coarse centroids by distance to query
    std::sort(c_dists.begin(), c_dists.end(),
              [](const CentroidDist& a, const CentroidDist& b) {
                  return a.dist < b.dist;
              });

    // Probe the top nprobe coarse centroids
    std::size_t probes = std::min(static_cast<std::size_t>(params_.nprobe), coarse_centroids_.rows());

    struct Candidate {
        ImageIndex id;
        float dist;
    };
    std::vector<Candidate> evaluated;

    for (std::size_t p = 0; p < probes; ++p) {
        int c_id = c_dists[p].id;
        const float* c_center = coarse_centroids_.row(c_id);

        // Residual of query vector with respect to coarse centroid
        std::vector<float> q_res(dim_);
        for (std::size_t d = 0; d < dim_; ++d) {
            q_res[d] = q[d] - c_center[d];
        }

        // Asymmetric Distance Computation (ADC) Lookup Table
        // lut[s][j] = || q_res[s] - sub_centroid[s][j] ||^2
        std::vector<std::vector<float>> lut(m_);
        for (std::size_t s = 0; s < m_; ++s) {
            const std::size_t k_sub = sub_centroids_[s].rows();
            lut[s].resize(k_sub);
            const float* q_sub = q_res.data() + sub_offsets_[s];
            for (std::size_t j = 0; j < k_sub; ++j) {
                lut[s][j] = l2_squared(q_sub, sub_centroids_[s].row(j), sub_dims_[s]);
            }
        }

        // Scan inverted list and compute ADC distance
        for (const auto& entry : inverted_lists_[c_id]) {
            float dist = 0.0f;
            for (std::size_t s = 0; s < m_; ++s) {
                dist += lut[s][entry.code[s]];
            }
            if (params_.rerank) {
                dist = l2_squared(dataset_->row(entry.id), q, dim_);
            }
            evaluated.push_back({entry.id, dist});
        }
    }

    // Sort candidates by ascending distance
    std::sort(evaluated.begin(), evaluated.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.dist != b.dist) {
                      return a.dist < b.dist;
                  }
                  return a.id < b.id;
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

std::size_t IVFPQIndex::size() const {
    return dataset_ ? dataset_->rows() : 0;
}

}  // namespace image_search
