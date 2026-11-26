#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace image_search {

// 2D Point with double precision coordinates.
struct Point2D {
    double x = 0.0;
    double y = 0.0;

    Point2D() = default;
    Point2D(double x_, double y_);

    bool operator==(const Point2D& o) const;
    bool operator!=(const Point2D& o) const;

    double distance_to(const Point2D& o) const;
    double squared_distance_to(const Point2D& o) const;
};

// 2D Point match / correspondence between source image (p1) and target image (p2).
struct Match2D {
    Point2D p1;  // (x, y) in source / query image
    Point2D p2;  // (u, v) in destination / candidate image

    Match2D() = default;
    Match2D(Point2D src, Point2D dst);
    Match2D(double x, double y, double u, double v);
};

// Hartley isotropic normalization parameters:
// p_norm = s * (p - mean)
struct Normalization {
    double s = 1.0;
    double mean_x = 0.0;
    double mean_y = 0.0;

    Point2D transform(const Point2D& p) const;
    Point2D inverse_transform(const Point2D& p) const;
};

// 3x3 Projective Homography matrix (double precision, row-major storage).
class Homography {
public:
    // Elements:
    // [0] [1] [2]
    // [3] [4] [5]
    // [6] [7] [8]
    std::array<double, 9> m = {1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0};

    Homography() = default;
    explicit Homography(const std::array<double, 9>& elements);
    Homography(double m00, double m01, double m02,
               double m10, double m11, double m12,
               double m20, double m21, double m22);

    double operator()(std::size_t r, std::size_t c) const;
    double& operator()(std::size_t r, std::size_t c);

    static Homography identity();

    // Verify all 9 elements are finite (not NaN, not Inf).
    bool is_finite() const;

    // Matrix determinant.
    double determinant() const;

    // Frobenius norm: sqrt(sum(m_ij^2)).
    double frobenius_norm() const;

    // Numerical invertibility check (finite, non-zero determinant, bounded condition number).
    bool is_invertible(double det_tol = 1e-12, double max_cond = 1e10) const;

    // Analytical 3x3 matrix inverse. Returns std::nullopt if not invertible.
    std::optional<Homography> inverse(double det_tol = 1e-12) const;

    // Normalizes scale so that m[8] == 1.0 (if |m[8]| > 1e-12) or Frobenius norm == 1.0.
    void normalize_scale();

    // Project point p = (x, y) via projective transformation:
    // (a, b, c)^T = H * (x, y, 1)^T
    // Returns (a/c, b/c) if c != 0 (|c| >= denom_tol) and finite.
    // Returns std::nullopt if denominator check fails.
    std::optional<Point2D> project(const Point2D& p, double denom_tol = 1e-12) const;

    // Reprojection error ||p2 - H(p1)||_2 in pixels.
    // Returns infinity if denominator check fails or projection is not finite.
    double reprojection_error(const Point2D& p1, const Point2D& p2, double denom_tol = 1e-12) const;
};

// =========================================================================
// Geometric and Degeneracy Checks
// =========================================================================

// 1. Check for duplicate points in a set of points (pairwise distance < tol).
bool has_duplicate_points(const std::vector<Point2D>& pts, double tol = 1e-6);

// 2. Check for collinear triplets among points (all combinations of 3 points).
// A triple is (near) collinear if eta = |(b-a) x (c-a)| / L^2 < tol, L = longest side
bool has_collinear_triplet(const std::vector<Point2D>& pts, double tol = 1e-4);

// 3. Check if 4 correspondences form a degenerate or nearly degenerate quad:
//    - Duplicate points in pts1 or pts2
//    - Collinear triplets in pts1 or pts2
//    - Directed line / orientation consistency check (same side of every line ij in both images)
//    - Minimum quad area check (shoelace area >= min_area)
bool is_degenerate_quad(const std::vector<Point2D>& pts1,
                        const std::vector<Point2D>& pts2,
                        double collinear_tol = 1e-4,
                        double min_area = 1.0);

// =========================================================================
// Hartley Normalization & DLT
// =========================================================================

// Hartley isotropic normalization for N >= 2 points:
// Shifts centroid to origin and scales average Euclidean distance to sqrt(2).
// Returns std::nullopt if points are coincident or average distance < 1e-12.
std::optional<Normalization> compute_normalization(const std::vector<Point2D>& pts);

// Normalized Direct Linear Transformation (DLT) using SVD.
// Solves Ah = 0 in normalized coordinates, then denormalizes: H = T2^{-1} * H_n * T1.
// Validates finite elements and numerical invertibility.
// Returns std::nullopt if matches < 4, normalization fails, or SVD system is degenerate.
std::optional<Homography> estimate_homography_dlt(const std::vector<Match2D>& matches);

std::optional<Homography> estimate_homography_dlt(const std::vector<Point2D>& pts1,
                                                 const std::vector<Point2D>& pts2);

// =========================================================================
// Inlier Evaluation & Refit
// =========================================================================

// Inlier evaluation result for a homography on a set of matches.
struct InlierResult {
    std::vector<std::size_t> inlier_indices;
    std::vector<double> inlier_errors;
    double median_error = 0.0;

    std::size_t count() const;
};

// Computes the median of a vector of errors.
// Returns NaN if vector is empty.
double compute_median(std::vector<double> errors);

// Evaluate inliers for a given homography on matches using threshold tau (pixels):
// e_i(H) < tau.
// Handles denominator check during projection.
InlierResult evaluate_inliers(const Homography& H,
                             const std::vector<Match2D>& matches,
                             double tau = 3.0,
                             double denom_tol = 1e-12);

// Refit result containing the refined homography and final inliers evaluated on ALL matches.
struct RefitResult {
    Homography H;
    InlierResult inliers;
};

// Refit homography using m = inlier_indices.size() >= 4 inliers:
// 1. Computes new normalization T, T' from the m inliers.
// 2. Solves 2m x 9 system via SVD to obtain H_refit = T'^{-1} * H_n * T.
// 3. Re-evaluates reprojection errors on ALL M original matches to obtain final inliers.
// Returns std::nullopt if m < 4 or estimation fails.
std::optional<RefitResult> refit_homography(const std::vector<Match2D>& all_matches,
                                           const std::vector<std::size_t>& best_inlier_indices,
                                           double tau = 3.0);

// =========================================================================
// Corner Error Evaluation (PDF Page 3)
// =========================================================================

// Corner error between estimated homography and ground-truth homography:
// E_corner = 1/4 * sum_{i=1}^4 || H_est(a_i) - H_gt(a_i) ||_2
// Returns NaN if projection fails on any corner.
double compute_corner_error(const Homography& H_est,
                            const Homography& H_gt,
                            const std::array<Point2D, 4>& corners);

// Convenience overload for image dimensions width x height:
// Uses corners: (0, 0), (width, 0), (width, height), (0, height).
double compute_corner_error(const Homography& H_est,
                            const Homography& H_gt,
                            double width,
                            double height);

}  // namespace image_search
