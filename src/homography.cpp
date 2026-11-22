#include "homography.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace image_search {

Point2D::Point2D(double x_, double y_) : x(x_), y(y_) {}

bool Point2D::operator==(const Point2D& o) const {
    return x == o.x && y == o.y;
}

bool Point2D::operator!=(const Point2D& o) const {
    return !(*this == o);
}

double Point2D::distance_to(const Point2D& o) const {
    double dx = x - o.x;
    double dy = y - o.y;
    return std::sqrt(dx * dx + dy * dy);
}

double Point2D::squared_distance_to(const Point2D& o) const {
    double dx = x - o.x;
    double dy = y - o.y;
    return dx * dx + dy * dy;
}

Match2D::Match2D(Point2D src, Point2D dst) : p1(src), p2(dst) {}

Match2D::Match2D(double x, double y, double u, double v) : p1(x, y), p2(u, v) {}

Point2D Normalization::transform(const Point2D& p) const {
    return Point2D(s * (p.x - mean_x), s * (p.y - mean_y));
}

Point2D Normalization::inverse_transform(const Point2D& p) const {
    return Point2D(p.x / s + mean_x, p.y / s + mean_y);
}

// =========================================================================
// Homography implementation
// =========================================================================

Homography::Homography(const std::array<double, 9>& elements) : m(elements) {}

Homography::Homography(double m00, double m01, double m02,
                       double m10, double m11, double m12,
                       double m20, double m21, double m22)
    : m{m00, m01, m02, m10, m11, m12, m20, m21, m22} {}

double Homography::operator()(std::size_t r, std::size_t c) const {
    return m[r * 3 + c];
}

double& Homography::operator()(std::size_t r, std::size_t c) {
    return m[r * 3 + c];
}

Homography Homography::identity() {
    return Homography(1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0);
}

bool Homography::is_finite() const {
    for (double val : m) {
        if (!std::isfinite(val)) {
            return false;
        }
    }
    return true;
}

double Homography::determinant() const {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

double Homography::frobenius_norm() const {
    double sum_sq = 0.0;
    for (double val : m) {
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

bool Homography::is_invertible(double det_tol, double max_cond) const {
    if (!is_finite()) {
        return false;
    }
    double fnorm = frobenius_norm();
    if (fnorm < 1e-15) {
        return false;
    }
    double det = determinant();
    // Scale-invariant determinant check: det / ||H||_F^3
    if (std::abs(det) / (fnorm * fnorm * fnorm) < det_tol) {
        return false;
    }

    Eigen::MatrixXd mat(3, 3);
    mat << m[0], m[1], m[2],
           m[3], m[4], m[5],
           m[6], m[7], m[8];

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat);
    const auto& sv = svd.singularValues();
    if (sv(2) < 1e-15) {
        return false;
    }
    double cond = sv(0) / sv(2);
    if (!std::isfinite(cond) || cond > max_cond) {
        return false;
    }
    return true;
}

std::optional<Homography> Homography::inverse(double det_tol) const {
    if (!is_finite()) {
        return std::nullopt;
    }
    double det = determinant();
    if (std::abs(det) < det_tol) {
        return std::nullopt;
    }
    double inv_det = 1.0 / det;

    std::array<double, 9> inv = {
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,

        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,

        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det
    };

    Homography res(inv);
    if (!res.is_finite()) {
        return std::nullopt;
    }
    return res;
}

void Homography::normalize_scale() {
    if (std::abs(m[8]) > 1e-12) {
        double s = 1.0 / m[8];
        for (double& val : m) {
            val *= s;
        }
    } else {
        double fnorm = frobenius_norm();
        if (fnorm > 1e-15) {
            double s = 1.0 / fnorm;
            for (double& val : m) {
                val *= s;
            }
        }
    }
}

std::optional<Point2D> Homography::project(const Point2D& p, double denom_tol) const {
    double c = m[6] * p.x + m[7] * p.y + m[8];
    if (std::abs(c) < denom_tol || !std::isfinite(c)) {
        return std::nullopt;
    }
    double a = m[0] * p.x + m[1] * p.y + m[2];
    double b = m[3] * p.x + m[4] * p.y + m[5];

    double u = a / c;
    double v = b / c;
    if (!std::isfinite(u) || !std::isfinite(v)) {
        return std::nullopt;
    }
    return Point2D(u, v);
}

double Homography::reprojection_error(const Point2D& p1, const Point2D& p2, double denom_tol) const {
    auto proj = project(p1, denom_tol);
    if (!proj) {
        return std::numeric_limits<double>::infinity();
    }
    return proj->distance_to(p2);
}

// =========================================================================
// Geometric and Degeneracy Checks
// =========================================================================

bool has_duplicate_points(const std::vector<Point2D>& pts, double tol) {
    const std::size_t n = pts.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (pts[i].distance_to(pts[j]) < tol) {
                return true;
            }
        }
    }
    return false;
}

bool has_collinear_triplet(const std::vector<Point2D>& pts, double tol) {
    const std::size_t n = pts.size();
    if (n < 3) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            for (std::size_t k = j + 1; k < n; ++k) {
                const Point2D& a = pts[i];
                const Point2D& b = pts[j];
                const Point2D& c = pts[k];

                double area2 = std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
                double d_ab = a.distance_to(b);
                double d_ac = a.distance_to(c);
                double d_bc = b.distance_to(c);
                double max_edge = std::max({d_ab, d_ac, d_bc});

                if (max_edge < 1e-12) {
                    return true;
                }
                double perp_dist = area2 / max_edge;
                if (perp_dist < tol) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool is_degenerate_quad(const std::vector<Point2D>& pts1,
                        const std::vector<Point2D>& pts2,
                        double collinear_tol,
                        double min_area) {
    if (pts1.size() != 4 || pts2.size() != 4) {
        return true;
    }

    // 1. Check duplicate points in either image
    if (has_duplicate_points(pts1) || has_duplicate_points(pts2)) {
        return true;
    }

    // 2. Check collinear triplets in either image
    if (has_collinear_triplet(pts1, collinear_tol) || has_collinear_triplet(pts2, collinear_tol)) {
        return true;
    }

    // 3. Check for concave quadrilateral (one point inside the triangle of the other three)
    auto point_in_triangle = [](const Point2D& p, const Point2D& a, const Point2D& b, const Point2D& c) {
        auto cross = [](const Point2D& p1, const Point2D& p2, const Point2D& p3) {
            return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
        };
        double d1 = cross(p, a, b);
        double d2 = cross(p, b, c);
        double d3 = cross(p, c, a);
        bool has_neg = (d1 < -1e-9) || (d2 < -1e-9) || (d3 < -1e-9);
        bool has_pos = (d1 > 1e-9) || (d2 > 1e-9) || (d3 > 1e-9);
        return !(has_neg && has_pos);
    };

    if (point_in_triangle(pts1[3], pts1[0], pts1[1], pts1[2]) ||
        point_in_triangle(pts1[2], pts1[0], pts1[1], pts1[3]) ||
        point_in_triangle(pts1[1], pts1[0], pts1[2], pts1[3]) ||
        point_in_triangle(pts1[0], pts1[1], pts1[2], pts1[3])) {
        return true;
    }

    if (point_in_triangle(pts2[3], pts2[0], pts2[1], pts2[2]) ||
        point_in_triangle(pts2[2], pts2[0], pts2[1], pts2[3]) ||
        point_in_triangle(pts2[1], pts2[0], pts2[2], pts2[3]) ||
        point_in_triangle(pts2[0], pts2[1], pts2[2], pts2[3])) {
        return true;
    }

    // 4. Oriented projective consistency check (cheirality / line-side preservation)
    auto line_eval = [](const Point2D& a, const Point2D& b, const Point2D& p) {
        return (b.y - a.y) * (p.x - a.x) - (b.x - a.x) * (p.y - a.y);
    };

    const std::size_t pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    const std::size_t others[6][2] = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};

    for (int idx = 0; idx < 6; ++idx) {
        std::size_t i = pairs[idx][0], j = pairs[idx][1];
        std::size_t k = others[idx][0], l = others[idx][1];

        double d1_k = line_eval(pts1[i], pts1[j], pts1[k]);
        double d1_l = line_eval(pts1[i], pts1[j], pts1[l]);
        double d2_k = line_eval(pts2[i], pts2[j], pts2[k]);
        double d2_l = line_eval(pts2[i], pts2[j], pts2[l]);

        double prod1 = d1_k * d1_l;
        double prod2 = d2_k * d2_l;

        if ((prod1 < -1e-9 && prod2 > 1e-9) || (prod1 > 1e-9 && prod2 < -1e-9)) {
            return true;
        }
    }

    // 5. Quad minimum area check
    auto tri_area = [](const Point2D& a, const Point2D& b, const Point2D& c) {
        return 0.5 * std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    };

    double area1 = std::max(tri_area(pts1[0], pts1[1], pts1[2]) + tri_area(pts1[0], pts1[2], pts1[3]),
                            tri_area(pts1[0], pts1[1], pts1[3]) + tri_area(pts1[1], pts1[2], pts1[3]));
    double area2 = std::max(tri_area(pts2[0], pts2[1], pts2[2]) + tri_area(pts2[0], pts2[2], pts2[3]),
                            tri_area(pts2[0], pts2[1], pts2[3]) + tri_area(pts2[1], pts2[2], pts2[3]));

    if (area1 < min_area || area2 < min_area) {
        return true;
    }

    return false;
}

// =========================================================================
// Hartley Normalization & DLT
// =========================================================================

std::optional<Normalization> compute_normalization(const std::vector<Point2D>& pts) {
    const std::size_t n = pts.size();
    if (n < 2) {
        return std::nullopt;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;
    for (const auto& p : pts) {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= static_cast<double>(n);
    mean_y /= static_cast<double>(n);

    double sum_dist = 0.0;
    for (const auto& p : pts) {
        double dx = p.x - mean_x;
        double dy = p.y - mean_y;
        sum_dist += std::sqrt(dx * dx + dy * dy);
    }
    double mean_dist = sum_dist / static_cast<double>(n);
    if (mean_dist < 1e-12 || !std::isfinite(mean_dist)) {
        return std::nullopt;
    }

    double s = std::sqrt(2.0) / mean_dist;
    return Normalization{s, mean_x, mean_y};
}

std::optional<Homography> estimate_homography_dlt(const std::vector<Point2D>& pts1,
                                                 const std::vector<Point2D>& pts2) {
    const std::size_t n = pts1.size();
    if (n < 4 || pts2.size() != n) {
        return std::nullopt;
    }

    if (n == 4 && is_degenerate_quad(pts1, pts2)) {
        return std::nullopt;
    }

    auto norm1 = compute_normalization(pts1);
    auto norm2 = compute_normalization(pts2);
    if (!norm1 || !norm2) {
        return std::nullopt;
    }

    // Build system matrix A: 2n x 9
    Eigen::MatrixXd A(2 * n, 9);
    for (std::size_t i = 0; i < n; ++i) {
        Point2D p1 = norm1->transform(pts1[i]);
        Point2D p2 = norm2->transform(pts2[i]);

        double x = p1.x, y = p1.y;
        double u = p2.x, v = p2.y;

        A.row(2 * i)     << -x, -y, -1.0,  0.0,  0.0,  0.0,  u * x, u * y, u;
        A.row(2 * i + 1) <<  0.0,  0.0,  0.0, -x, -y, -1.0,  v * x, v * y, v;
    }

    // Solve Ah = 0 via SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const auto& sv = svd.singularValues();

    // Check condition / rank: 8th singular value (index 7) should be non-zero
    if (sv(7) < 1e-10) {
        return std::nullopt;
    }

    // Smallest singular value is at index 8
    Eigen::VectorXd h = svd.matrixV().col(8);

    Eigen::Matrix3d H_n;
    H_n << h(0), h(1), h(2),
           h(3), h(4), h(5),
           h(6), h(7), h(8);

    // Normalization matrices:
    // T1: shifts centroid of pts1 to origin, scales by s1
    // T2: shifts centroid of pts2 to origin, scales by s2
    // H = T2^{-1} * H_n * T1
    Eigen::Matrix3d T1;
    T1 << norm1->s, 0.0, -norm1->s * norm1->mean_x,
          0.0, norm1->s, -norm1->s * norm1->mean_y,
          0.0, 0.0, 1.0;

    Eigen::Matrix3d T2_inv;
    T2_inv << 1.0 / norm2->s, 0.0, norm2->mean_x,
              0.0, 1.0 / norm2->s, norm2->mean_y,
              0.0, 0.0, 1.0;

    Eigen::Matrix3d H_denorm = T2_inv * H_n * T1;

    Homography result;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            result(r, c) = H_denorm(r, c);
        }
    }

    result.normalize_scale();

    if (!result.is_finite() || !result.is_invertible()) {
        return std::nullopt;
    }

    return result;
}

std::optional<Homography> estimate_homography_dlt(const std::vector<Match2D>& matches) {
    std::vector<Point2D> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto& m : matches) {
        pts1.push_back(m.p1);
        pts2.push_back(m.p2);
    }
    return estimate_homography_dlt(pts1, pts2);
}

// =========================================================================
// Inlier Evaluation & Refit
// =========================================================================

std::size_t InlierResult::count() const {
    return inlier_indices.size();
}

double compute_median(std::vector<double> errors) {
    if (errors.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(errors.begin(), errors.end());
    std::size_t n = errors.size();
    if (n % 2 == 1) {
        return errors[n / 2];
    }
    return 0.5 * (errors[n / 2 - 1] + errors[n / 2]);
}

InlierResult evaluate_inliers(const Homography& H,
                             const std::vector<Match2D>& matches,
                             double tau,
                             double denom_tol) {
    InlierResult res;
    res.inlier_indices.reserve(matches.size());
    res.inlier_errors.reserve(matches.size());

    for (std::size_t i = 0; i < matches.size(); ++i) {
        double err = H.reprojection_error(matches[i].p1, matches[i].p2, denom_tol);
        if (std::isfinite(err) && err < tau) {
            res.inlier_indices.push_back(i);
            res.inlier_errors.push_back(err);
        }
    }

    res.median_error = compute_median(res.inlier_errors);
    return res;
}

std::optional<RefitResult> refit_homography(const std::vector<Match2D>& all_matches,
                                           const std::vector<std::size_t>& best_inlier_indices,
                                           double tau) {
    const std::size_t m = best_inlier_indices.size();
    if (m < 4) {
        return std::nullopt;
    }

    std::vector<Match2D> inlier_matches;
    inlier_matches.reserve(m);
    for (std::size_t idx : best_inlier_indices) {
        if (idx < all_matches.size()) {
            inlier_matches.push_back(all_matches[idx]);
        }
    }
    if (inlier_matches.size() < 4) {
        return std::nullopt;
    }

    auto H_refit = estimate_homography_dlt(inlier_matches);
    if (!H_refit) {
        return std::nullopt;
    }

    // Re-evaluate on ALL original matches
    InlierResult final_inliers = evaluate_inliers(*H_refit, all_matches, tau);
    if (final_inliers.count() < 4) {
        return std::nullopt;
    }

    return RefitResult{*H_refit, final_inliers};
}

// =========================================================================
// Corner Error Evaluation (PDF Page 3)
// =========================================================================

double compute_corner_error(const Homography& H_est,
                            const Homography& H_gt,
                            const std::array<Point2D, 4>& corners) {
    double total_err = 0.0;
    for (const auto& pt : corners) {
        auto p_est = H_est.project(pt);
        auto p_gt = H_gt.project(pt);
        if (!p_est || !p_gt) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        total_err += p_est->distance_to(*p_gt);
    }
    return total_err / 4.0;
}

double compute_corner_error(const Homography& H_est,
                            const Homography& H_gt,
                            double width,
                            double height) {
    std::array<Point2D, 4> corners = {
        Point2D(0.0, 0.0),
        Point2D(width, 0.0),
        Point2D(width, height),
        Point2D(0.0, height)
    };
    return compute_corner_error(H_est, H_gt, corners);
}

}  // namespace image_search
