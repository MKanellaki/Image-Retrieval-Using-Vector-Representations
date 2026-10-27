#include "ivfpq.hpp"
#include "distance.hpp"
#include "kmeans.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace image_search {

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

    // 2. Coarse clustering on the full dataset; labels give each point's inverted list
    KMeansParams coarse;
    coarse.k = std::min(static_cast<std::size_t>(params_.k), n);
    coarse.max_iter = params_.max_iter;
    coarse.seed = params_.seed;
    KMeansResult coarse_km = kmeans_fit(*dataset_, coarse);
    coarse_centroids_ = std::move(coarse_km.centers);

    // 3. Residuals with respect to the assigned coarse centroid
    Matrix residuals(n, dim_);
    for (std::size_t i = 0; i < n; ++i) {
        const float* pt = dataset_->row(i);
        const float* center = coarse_centroids_.row(coarse_km.labels[i]);
        float* r = residuals.row(i);
        for (std::size_t d = 0; d < dim_; ++d) {
            r[d] = pt[d] - center[d];
        }
    }

    // 4. Lay out the inverted lists: ids in database order, and each point's slot in its list
    k_sub_ = std::min(static_cast<std::size_t>(std::max(1, params_.k_sub)), n);
    if (k_sub_ > 65536) {
        throw std::invalid_argument("IVFPQ: k_sub > 65536 does not fit a 16-bit code");
    }
    wide_codes_ = k_sub_ > 256;

    inverted_lists_.resize(coarse_centroids_.rows());
    std::vector<std::size_t> slot(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto& ids = inverted_lists_[coarse_km.labels[i]].ids;
        slot[i] = ids.size();
        ids.push_back(static_cast<ImageIndex>(i));
    }
    for (auto& list : inverted_lists_) {
        if (wide_codes_) {
            list.codes16.resize(list.ids.size() * m_);
        } else {
            list.codes8.resize(list.ids.size() * m_);
        }
    }

    // 5. Train a codebook per sub-space on the residuals; its labels are the PQ codes
    sub_centroids_.resize(m_);
    for (std::size_t s = 0; s < m_; ++s) {
        Matrix sub(n, sub_dims_[s]);
        for (std::size_t i = 0; i < n; ++i) {
            const float* r = residuals.row(i) + sub_offsets_[s];
            std::copy(r, r + sub_dims_[s], sub.row(i));
        }

        KMeansParams kp;
        kp.k = k_sub_;
        kp.max_iter = params_.max_iter;
        kp.seed = params_.seed + s * 13 + 7;
        KMeansResult km = kmeans_fit(sub, kp);
        sub_centroids_[s] = std::move(km.centers);

        for (std::size_t i = 0; i < n; ++i) {
            auto& list = inverted_lists_[coarse_km.labels[i]];
            const std::size_t at = slot[i] * m_ + s;
            if (wide_codes_) {
                list.codes16[at] = static_cast<std::uint16_t>(km.labels[i]);
            } else {
                list.codes8[at] = static_cast<std::uint8_t>(km.labels[i]);
            }
        }
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
    std::vector<float> q_res(dim_);
    std::vector<float> lut(m_ * k_sub_);   // lut[s * k_sub_ + j] = || q_res[s] - sub_centroid[s][j] ||^2

    for (std::size_t p = 0; p < probes; ++p) {
        int c_id = c_dists[p].id;
        const float* c_center = coarse_centroids_.row(c_id);

        // Residual of query vector with respect to coarse centroid
        for (std::size_t d = 0; d < dim_; ++d) {
            q_res[d] = q[d] - c_center[d];
        }

        // Asymmetric Distance Computation (ADC) lookup table
        for (std::size_t s = 0; s < m_; ++s) {
            const float* q_sub = q_res.data() + sub_offsets_[s];
            for (std::size_t j = 0; j < k_sub_; ++j) {
                lut[s * k_sub_ + j] = l2_squared(q_sub, sub_centroids_[s].row(j), sub_dims_[s]);
            }
        }

        // Scan inverted list and compute ADC distance (codes is the list's 8- or 16-bit array)
        const InvertedList& list = inverted_lists_[c_id];
        auto scan = [&](const auto& codes) {
            for (std::size_t e = 0; e < list.ids.size(); ++e) {
                const auto* code = codes.data() + e * m_;
                float dist = 0.0f;
                for (std::size_t s = 0; s < m_; ++s) {
                    dist += lut[s * k_sub_ + code[s]];
                }
                if (params_.rerank) {
                    dist = l2_squared(dataset_->row(list.ids[e]), q, dim_);
                }
                evaluated.push_back({list.ids[e], dist});
            }
        };
        if (wide_codes_) {
            scan(list.codes16);
        } else {
            scan(list.codes8);
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
