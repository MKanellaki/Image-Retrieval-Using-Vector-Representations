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

}  // namespace image_search
