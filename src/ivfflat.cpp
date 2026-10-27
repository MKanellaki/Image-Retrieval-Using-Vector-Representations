#include "ivfflat.hpp"
#include "distance.hpp"
#include "kmeans.hpp"
#include <algorithm>
#include <utility>

namespace image_search {

IVFFlatIndex::IVFFlatIndex(const Matrix& dataset, const IVFFlatParams& params)
    : dataset_(&dataset), params_(params), dim_(dataset.cols()) {

    // Empty input
    if (dataset.empty() || dim_ == 0 || params_.k <= 0) {
        return;
    }

    // Cluster the database with k-means; the labels already say which list each point goes in
    KMeansParams kp;
    kp.k = std::min(static_cast<std::size_t>(params_.k), dataset.rows());
    kp.max_iter = params_.max_iter;
    kp.seed = params_.seed;
    KMeansResult km = kmeans_fit(dataset, kp);
    centroids_ = std::move(km.centers);

    // Insert database images into inverted lists
    inverted_lists_.resize(centroids_.rows());
    for (std::size_t i = 0; i < dataset.rows(); ++i) {
        inverted_lists_[km.labels[i]].push_back(static_cast<ImageIndex>(i));
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

}  // namespace image_search