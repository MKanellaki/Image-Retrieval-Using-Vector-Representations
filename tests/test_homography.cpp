#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "homography.hpp"

using namespace image_search;

static bool close(double a, double b, double tol = 1e-7) {
    return std::fabs(a - b) < tol;
}

int main() {
    std::cout << "Running test_homography..." << std::endl;

    // 1. Point2D and Match2D basics
    {
        Point2D p1(10.0, 20.0);
        Point2D p2(13.0, 24.0);
        assert(close(p1.distance_to(p2), 5.0));
        assert(close(p1.squared_distance_to(p2), 25.0));
        assert(p1 == Point2D(10.0, 20.0));
        assert(p1 != p2);

        Match2D m1(p1, p2);
        assert(m1.p1 == p1 && m1.p2 == p2);
        Match2D m2(1.0, 2.0, 3.0, 4.0);
        assert(m2.p1.x == 1.0 && m2.p2.y == 4.0);
    }

    // 2. Identity Homography properties
    {
        Homography I = Homography::identity();
        assert(I.is_finite());
        assert(close(I.determinant(), 1.0));
        assert(I.is_invertible());

        Point2D p(42.5, 99.1);
        auto proj = I.project(p);
        assert(proj.has_value());
        assert(close(proj->x, 42.5) && close(proj->y, 99.1));
        assert(close(I.reprojection_error(p, p), 0.0));

        auto inv = I.inverse();
        assert(inv.has_value());
        assert(close(inv->determinant(), 1.0));
        assert(close((*inv)(0, 0), 1.0) && close((*inv)(1, 1), 1.0) && close((*inv)(2, 2), 1.0));
    }

    // 3. Finite, Determinant, Inversion and Invertibility checks
    {
        // Non-finite matrix
        Homography nan_H(1, 0, 0, 0, std::numeric_limits<double>::quiet_NaN(), 0, 0, 0, 1);
        assert(!nan_H.is_finite());
        assert(!nan_H.is_invertible());
        assert(!nan_H.inverse().has_value());

        Homography inf_H(1, 0, 0, 0, 1, 0, 0, 0, std::numeric_limits<double>::infinity());
        assert(!inf_H.is_finite());
        assert(!inf_H.is_invertible());

        // Singular matrix (rank 1 / rank 2)
        Homography singular_H(1, 2, 3, 2, 4, 6, 1, 2, 3);
        assert(singular_H.is_finite());
        assert(close(singular_H.determinant(), 0.0));
        assert(!singular_H.is_invertible());
        assert(!singular_H.inverse().has_value());

        // General invertible projective matrix
        Homography H(1.5, 0.2, 10.0,
                    -0.1, 1.2, 20.0,
                     0.001, 0.002, 1.0);
        assert(H.is_finite());
        assert(H.is_invertible());
        auto H_inv = H.inverse();
        assert(H_inv.has_value());

        // Verify H * H_inv = I
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                double val = 0.0;
                for (int k = 0; k < 3; ++k) {
                    val += H(r, k) * (*H_inv)(k, c);
                }
                double expected = (r == c) ? 1.0 : 0.0;
                assert(close(val, expected, 1e-6));
            }
        }
    }

    // 4. Projection and Denominator check (c != 0)
    {
        // Line at infinity: c = h20*x + h21*y + h22
        // Suppose h20 = 1, h21 = 0, h22 = -10. Points with x = 10 have c = 0.
        Homography H(1, 0, 0,
                     0, 1, 0,
                     1, 0, -10);

        Point2D p_valid(20.0, 5.0); // c = 20 - 10 = 10 != 0
        auto proj_valid = H.project(p_valid);
        assert(proj_valid.has_value());
        assert(close(proj_valid->x, 2.0));
        assert(close(proj_valid->y, 0.5));

        Point2D p_zero_denom(10.0, 5.0); // c = 10 - 10 = 0
        auto proj_fail = H.project(p_zero_denom);
        assert(!proj_fail.has_value());
        assert(std::isinf(H.reprojection_error(p_zero_denom, Point2D(0, 0))));

        // Extremely near zero denominator (< denom_tol)
        Point2D p_near_zero(10.0 + 1e-14, 5.0);
        auto proj_near_fail = H.project(p_near_zero, 1e-12);
        assert(!proj_near_fail.has_value());
    }

    // 5. Duplicate points check
    {
        std::vector<Point2D> distinct = {
            {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}
        };
        assert(!has_duplicate_points(distinct));

        std::vector<Point2D> with_dup = {
            {0.0, 0.0}, {10.0, 0.0}, {0.0, 0.0}, {0.0, 10.0}
        };
        assert(has_duplicate_points(with_dup));

        std::vector<Point2D> near_dup = {
            {0.0, 0.0}, {10.0, 0.0}, {1e-8, 1e-8}, {0.0, 10.0}
        };
        assert(has_duplicate_points(near_dup, 1e-6));
    }

    // 6. Collinear triplets check
    {
        std::vector<Point2D> general_pos = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 100.0}, {0.0, 100.0}
        };
        assert(!has_collinear_triplet(general_pos));

        // Points 0, 1, 2 on horizontal line y = 0
        std::vector<Point2D> collinear = {
            {0.0, 0.0}, {50.0, 0.0}, {100.0, 0.0}, {50.0, 50.0}
        };
        assert(has_collinear_triplet(collinear));

        // Diagonal collinearity: (0,0), (10,10), (20,20)
        std::vector<Point2D> diag_collinear = {
            {0.0, 0.0}, {10.0, 10.0}, {20.0, 20.0}, {-5.0, 15.0}
        };
        assert(has_collinear_triplet(diag_collinear));
    }

    // 7. Degenerate and nearly degenerate quad check
    {
        std::vector<Point2D> square1 = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 100.0}, {0.0, 100.0}
        };
        std::vector<Point2D> square2 = {
            {10.0, 10.0}, {110.0, 10.0}, {110.0, 110.0}, {10.0, 110.0}
        };
        // Valid non-degenerate quads
        assert(!is_degenerate_quad(square1, square2));

        // Duplicate points in quad
        std::vector<Point2D> dup_quad = {
            {0.0, 0.0}, {100.0, 0.0}, {0.0, 0.0}, {0.0, 100.0}
        };
        assert(is_degenerate_quad(dup_quad, square2));

        // Collinear triplet in quad
        std::vector<Point2D> col_quad = {
            {0.0, 0.0}, {50.0, 0.0}, {100.0, 0.0}, {0.0, 100.0}
        };
        assert(is_degenerate_quad(col_quad, square2));

        // Concave quad (point inside triangle of the other three) is NOT degenerate by itself
        std::vector<Point2D> concave_quad = {
            {0.0, 0.0}, {100.0, 0.0}, {50.0, 100.0}, {50.0, 20.0}
        };
        std::vector<Point2D> concave_shifted;
        for (const auto& p : concave_quad) concave_shifted.emplace_back(p.x + 10.0, p.y + 10.0);
        assert(!is_degenerate_quad(concave_quad, concave_shifted));
        // but paired with a convex quad the orientation check rejects it (no homography does that)
        assert(is_degenerate_quad(concave_quad, square2));

        // Bowtie / self-intersecting twist (crossing diagonals order flipped)
        std::vector<Point2D> bowtie = {
            {0.0, 0.0}, {100.0, 100.0}, {100.0, 0.0}, {0.0, 100.0}
        };
        assert(is_degenerate_quad(square1, bowtie));

        // Tiny collapsed area (< min_area)
        std::vector<Point2D> tiny_quad = {
            {0.0, 0.0}, {0.1, 0.0}, {0.1, 0.1}, {0.0, 0.1}
        };
        assert(is_degenerate_quad(tiny_quad, square2, 1e-4, 1.0));
    }

    // 8. Hartley isotropic normalization
    {
        std::vector<Point2D> pts = {
            {10.0, 20.0},
            {150.0, 40.0},
            {120.0, 200.0},
            {30.0, 180.0}
        };

        auto norm = compute_normalization(pts);
        assert(norm.has_value());

        // Centroid of normalized points must be at origin (0, 0)
        double mean_nx = 0.0, mean_ny = 0.0;
        double sum_dist = 0.0;
        for (const auto& p : pts) {
            Point2D np = norm->transform(p);
            mean_nx += np.x;
            mean_ny += np.y;
            sum_dist += std::sqrt(np.x * np.x + np.y * np.y);

            // Inverse transform check
            Point2D reconstructed = norm->inverse_transform(np);
            assert(close(p.x, reconstructed.x));
            assert(close(p.y, reconstructed.y));
        }
        mean_nx /= pts.size();
        mean_ny /= pts.size();
        double avg_dist = sum_dist / pts.size();

        assert(close(mean_nx, 0.0));
        assert(close(mean_ny, 0.0));
        assert(close(avg_dist, std::sqrt(2.0))); // Average distance from origin = sqrt(2)

        // Degenerate points (all coincident)
        std::vector<Point2D> coincident = {
            {5.0, 5.0}, {5.0, 5.0}, {5.0, 5.0}, {5.0, 5.0}
        };
        assert(!compute_normalization(coincident).has_value());
    }

    // 9. Normalized DLT: exact 4-point recovery
    {
        // Ground truth projective homography
        Homography H_gt(1.2, 0.2, 25.0,
                       -0.1, 1.1, -15.0,
                        0.0004, -0.0002, 1.0);

        std::vector<Point2D> pts1 = {
            {20.0, 30.0},
            {250.0, 45.0},
            {220.0, 280.0},
            {40.0, 260.0}
        };

        std::vector<Point2D> pts2;
        std::vector<Match2D> matches;
        for (const auto& p : pts1) {
            auto proj = H_gt.project(p);
            assert(proj.has_value());
            pts2.push_back(*proj);
            matches.emplace_back(p, *proj);
        }

        auto H_est = estimate_homography_dlt(matches);
        assert(H_est.has_value());

        // Check reprojection error for all points is practically zero
        for (std::size_t i = 0; i < 4; ++i) {
            double err = H_est->reprojection_error(pts1[i], pts2[i]);
            assert(close(err, 0.0, 1e-8));
        }

        // Compare matrix elements normalized to H(2,2) == 1.0
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t c = 0; c < 3; ++c) {
                assert(close((*H_est)(r, c), H_gt(r, c), 1e-6));
            }
        }

        // Corner error between H_est and H_gt should be negligible
        double c_err = compute_corner_error(*H_est, H_gt, 300.0, 300.0);
        assert(close(c_err, 0.0, 1e-8));
    }

    // 10. Normalized DLT: Overdetermined system (N = 20 correspondences)
    {
        Homography H_gt(0.9, -0.15, 35.0,
                        0.15, 0.9, 15.0,
                        0.0002, 0.0001, 1.0);

        std::vector<Match2D> matches;
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist_x(10.0, 400.0);
        std::uniform_real_distribution<double> dist_y(10.0, 400.0);

        for (int i = 0; i < 20; ++i) {
            Point2D p1(dist_x(rng), dist_y(rng));
            auto p2 = H_gt.project(p1);
            assert(p2.has_value());
            matches.emplace_back(p1, *p2);
        }

        auto H_est = estimate_homography_dlt(matches);
        assert(H_est.has_value());

        for (const auto& m : matches) {
            double err = H_est->reprojection_error(m.p1, m.p2);
            assert(close(err, 0.0, 1e-8));
        }
    }

    // 11. DLT Rejection of degenerate configurations
    {
        // Fewer than 4 points
        std::vector<Match2D> three_pts = {
            {{0,0}, {0,0}}, {{10,0}, {10,0}}, {{0,10}, {0,10}}
        };
        assert(!estimate_homography_dlt(three_pts).has_value());

        // 4 points with collinear triplet
        std::vector<Match2D> col_matches = {
            {{0,0}, {0,0}}, {{50,0}, {50,0}}, {{100,0}, {100,0}}, {{10,10}, {10,10}}
        };
        assert(!estimate_homography_dlt(col_matches).has_value());

        // 4 points with duplicate points
        std::vector<Match2D> dup_matches = {
            {{0,0}, {0,0}}, {{100,0}, {100,0}}, {{0,0}, {0,0}}, {{0,100}, {0,100}}
        };
        assert(!estimate_homography_dlt(dup_matches).has_value());
    }

    // 12. Inlier evaluation and Median calculation
    {
        // Odd error list: {0.5, 1.2, 2.8} -> median is 1.2
        assert(close(compute_median({2.8, 0.5, 1.2}), 1.2));

        // Even error list: {0.4, 1.0, 2.0, 3.6} -> median is (1.0 + 2.0) / 2 = 1.5
        assert(close(compute_median({3.6, 0.4, 2.0, 1.0}), 1.5));

        // Empty list -> NaN
        assert(std::isnan(compute_median({})));

        // Inlier test with tau = 3.0
        Homography H_gt = Homography::identity();
        std::vector<Match2D> matches = {
            {{10, 10}, {10.5, 10.0}},   // err = 0.5 < 3 (inlier)
            {{20, 20}, {21.0, 20.0}},   // err = 1.0 < 3 (inlier)
            {{30, 30}, {32.0, 30.0}},   // err = 2.0 < 3 (inlier)
            {{40, 40}, {90.0, 40.0}},   // err = 50.0 >= 3 (outlier)
            {{50, 50}, {50.0, 50.5}}    // err = 0.5 < 3 (inlier)
        };

        auto res = evaluate_inliers(H_gt, matches, 3.0);
        assert(res.count() == 4);
        assert(res.inlier_indices == std::vector<std::size_t>({0, 1, 2, 4}));
        // inlier errors: 0.5, 1.0, 2.0, 0.5 -> sorted: 0.5, 0.5, 1.0, 2.0 -> median = (0.5 + 1.0) / 2 = 0.75
        assert(close(res.median_error, 0.75));
    }

    // 13. Refit (Επανεκτίμηση) on inliers
    {
        Homography H_gt(1.1, 0.05, 10.0,
                       -0.05, 1.1, 15.0,
                        0.0001, 0.0002, 1.0);

        std::vector<Match2D> all_matches;
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> dist(20.0, 300.0);
        std::normal_distribution<double> noise(0.0, 0.2); // small inlier noise

        // 12 inliers
        for (int i = 0; i < 12; ++i) {
            Point2D p1(dist(rng), dist(rng));
            auto p2 = H_gt.project(p1);
            assert(p2.has_value());
            Point2D p2_noisy(p2->x + noise(rng), p2->y + noise(rng));
            all_matches.emplace_back(p1, p2_noisy);
        }

        // 8 severe outliers
        for (int i = 0; i < 8; ++i) {
            Point2D p1(dist(rng), dist(rng));
            Point2D p2_outlier(dist(rng) + 500.0, dist(rng) + 500.0);
            all_matches.emplace_back(p1, p2_outlier);
        }

        // Suppose hypothesis found 7 of the inliers
        std::vector<std::size_t> sample_inliers = {0, 1, 2, 3, 4, 5, 6};

        auto refit_res = refit_homography(all_matches, sample_inliers, 3.0);
        assert(refit_res.has_value());

        // The refitted homography re-evaluated on ALL 20 matches should find all 12 true inliers!
        assert(refit_res->inliers.count() == 12);
        assert(refit_res->inliers.median_error < 1.0);

        // Refit with fewer than 4 inliers must fail
        assert(!refit_homography(all_matches, {0, 1, 2}, 3.0).has_value());
    }

    // 14. Corner error calculation
    {
        Homography H_gt = Homography::identity();
        Homography H_shift(1, 0, 4.0,
                           0, 1, 3.0,
                           0, 0, 1.0); // shifts by (4, 3), distance = 5

        double c_err = compute_corner_error(H_shift, H_gt, 100.0, 100.0);
        // Each corner shifted by 5 px, average corner error = 5.0
        assert(close(c_err, 5.0));

        // When estimated equals ground truth, error = 0
        assert(close(compute_corner_error(H_gt, H_gt, 100.0, 100.0), 0.0));
    }

    std::cout << "All homography tests passed successfully!" << std::endl;
    return 0;
}
