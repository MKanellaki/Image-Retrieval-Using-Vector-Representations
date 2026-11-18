#include "homography.hpp"

#include <cmath>
#include <limits>

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

}  // namespace image_search
