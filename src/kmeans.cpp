#include "kmeans.hpp"
#include "distance.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>

namespace image_search {

namespace {

// Nearest center and its squared distance (ties -> lowest index).
inline std::size_t nearest_with_dist(const float* p, const Matrix& centers, float& best_d2) {
    assert(centers.rows() > 0);
    const std::size_t d = centers.cols();
    std::size_t best = 0;
    best_d2 = std::numeric_limits<float>::max();
    for (std::size_t c = 0; c < centers.rows(); ++c) {
        const float v = l2_squared(p, centers.row(c), d);
        if (v < best_d2) {
            best_d2 = v;
            best = c;
        }
    }
    return best;
}

inline double uniform01(std::mt19937_64& rng) {  // [0, 1)
    return static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
}

void copy_row(float* dst, const float* src, std::size_t d) {
    std::copy(src, src + d, dst);
}

Matrix init_plusplus(const Matrix& data, std::size_t k, std::mt19937_64& rng) {
    const std::size_t n = data.rows(), d = data.cols();
    Matrix centers(k, d);

    copy_row(centers.row(0), data.row(static_cast<std::size_t>(rng() % n)), d);

    std::vector<float> min_d2(n);
    for (std::size_t i = 0; i < n; ++i) min_d2[i] = l2_squared(data.row(i), centers.row(0), d);

    std::vector<double> prefix(n);
    for (std::size_t t = 1; t < k; ++t) {
        // Normalize by the max so partial sums stay well scaled.
        float max_d2 = 0.0f;
        for (float v : min_d2) max_d2 = std::max(max_d2, v);

        std::size_t pick;
        if (max_d2 <= 0.0f) {
            pick = static_cast<std::size_t>(rng() % n);
        } else {
            double acc = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                acc += static_cast<double>(min_d2[i] / max_d2);
                prefix[i] = acc;
            }
            // x in (0, total]; first r with P(r) >= x has P(r-1) < x.
            const double x = acc * (1.0 - uniform01(rng));
            pick = static_cast<std::size_t>(std::lower_bound(prefix.begin(), prefix.end(), x) - prefix.begin());
            if (pick >= n) pick = n - 1;
        }

        copy_row(centers.row(t), data.row(pick), d);
        for (std::size_t i = 0; i < n; ++i) {
            const float v = l2_squared(data.row(i), centers.row(t), d);
            if (v < min_d2[i]) min_d2[i] = v;
        }
    }
    return centers;
}

}  // namespace

std::size_t nearest_center(const float* point, const Matrix& centers) {
    assert(centers.rows() > 0);
    float d2;
    return nearest_with_dist(point, centers, d2);
}

std::vector<std::uint32_t> assign_all(const Matrix& data, const Matrix& centers) {
    std::vector<std::uint32_t> labels(data.rows());
    for (std::size_t i = 0; i < data.rows(); ++i)
        labels[i] = static_cast<std::uint32_t>(nearest_center(data.row(i), centers));
    return labels;
}

KMeansResult kmeans_fit(const Matrix& data, const KMeansParams& params) {
    const std::size_t n = data.rows(), d = data.cols(), k = params.k;
    if (n == 0 || k == 0) throw std::invalid_argument("kmeans: empty data or k == 0");
    if (k > n) throw std::invalid_argument("kmeans: k larger than number of points");

    std::mt19937_64 rng(params.seed);
    Matrix centers = init_plusplus(data, k, rng);

    constexpr std::uint32_t kUnassigned = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> labels(n, kUnassigned);
    std::vector<float> d2(n, 0.0f);
    std::vector<std::size_t> counts(k);
    std::vector<double> sums(k * d);

    for (int iter = 0; iter < params.max_iter; ++iter) {
        // Assignment.
        std::size_t changed = 0;
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t i = 0; i < n; ++i) {
            const auto c = static_cast<std::uint32_t>(nearest_with_dist(data.row(i), centers, d2[i]));
            if (c != labels[i]) {
                labels[i] = c;
                ++changed;
            }
            ++counts[c];
        }

        // Empty clusters: steal the farthest point from a cluster with > 1 member.
        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] != 0) continue;
            std::size_t far = n;
            float far_d2 = -1.0f;
            for (std::size_t i = 0; i < n; ++i) {
                if (counts[labels[i]] > 1 && d2[i] > far_d2) {
                    far_d2 = d2[i];
                    far = i;
                }
            }
            if (far == n) break;  // cannot happen when k <= n, kept as a guard
            --counts[labels[far]];
            labels[far] = static_cast<std::uint32_t>(c);
            counts[c] = 1;
            d2[far] = 0.0f;
            ++changed;
        }

        // Update: centers become cluster means.
        std::fill(sums.begin(), sums.end(), 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const float* p = data.row(i);
            double* s = &sums[static_cast<std::size_t>(labels[i]) * d];
            for (std::size_t j = 0; j < d; ++j) s[j] += p[j];
        }
        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;
            float* ctr = centers.row(c);
            const double inv = 1.0 / static_cast<double>(counts[c]);
            for (std::size_t j = 0; j < d; ++j) ctr[j] = static_cast<float>(sums[c * d + j] * inv);
        }

        if (static_cast<double>(changed) / static_cast<double>(n) < params.tol) break;
    }

    // Labels so far belong to the centers of the previous iteration; redo them
    // against the final centers so that (centers, labels) are consistent.
    labels = assign_all(data, centers);
    return KMeansResult{std::move(centers), std::move(labels)};
}

Matrix kmeans(const Matrix& data, const KMeansParams& params) {
    return kmeans_fit(data, params).centers;
}

}  // namespace image_search