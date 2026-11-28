#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "homography_eval.hpp"

using namespace image_search;
namespace fs = std::filesystem;

namespace {
bool close(double a, double b, double eps = 1e-6) { return std::abs(a - b) < eps; }

template <class Ex, class F>
bool throws(F&& f) {
    try { f(); } catch (const Ex&) { return true; } catch (...) { return false; }
    return false;
}

void write_file(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

Homography multiply(const Homography& A, const Homography& B) {
    Homography C;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c) {
            double s = 0;
            for (std::size_t k = 0; k < 3; ++k) s += A(r, k) * B(k, c);
            C(r, c) = s;
        }
    return C;
}

Homography translation(double dx, double dy) { return Homography(1, 0, dx, 0, 1, dy, 0, 0, 1); }

// A real ground truth (v_bird H_1_2 values): scale, shear and a projective part
const Homography kGt(0.32788, -0.00026656, 168.52, -0.087696, 0.49289, 72.043,
                     -0.00025798, 4.6006e-06, 0.9984);

CandidateResult verified(ImageIndex row, const Homography& H, std::size_t inliers = 20) {
    CandidateResult c;
    c.row = row;
    c.matches = inliers + 5;
    c.success = true;
    c.inliers = inliers;
    c.median_error = 1.0;
    c.H = H;
    return c;
}

CandidateResult failed(ImageIndex row) {
    CandidateResult c;
    c.row = row;
    c.matches = 3;
    return c;
}

PairEvaluation pair(ImageIndex row, bool in_top_c, bool valid, double err) {
    PairEvaluation e;
    e.row = row;
    e.in_top_c = in_top_c;
    e.valid = valid;
    e.corner_error = valid ? err : std::nan("");
    return e;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("homography_eval_test_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);

    { // reading H_1_j files
        const fs::path p = dir / "H_ok";
        write_file(p, "0.32788 -0.00026656 168.52\n-0.087696 0.49289 72.043\n-0.00025798 4.6006e-06 0.9984\n");
        const Homography H = read_homography_file(p.string());
        assert(H.m == kGt.m);

        write_file(dir / "H_spaces", "  1 0 5   0 1 -7\n\n 0 0 1  ");   // any whitespace layout
        assert(read_homography_file((dir / "H_spaces").string()).m == translation(5, -7).m);

        assert(throws<std::runtime_error>([&] { read_homography_file((dir / "missing").string()); }));
        write_file(dir / "H_8", "1 0 0 0 1 0 0 0");
        assert(throws<std::runtime_error>([&] { read_homography_file((dir / "H_8").string()); }));
        write_file(dir / "H_10", "1 0 0 0 1 0 0 0 1 4");
        assert(throws<std::runtime_error>([&] { read_homography_file((dir / "H_10").string()); }));
        write_file(dir / "H_text", "1 0 0 0 x 0 0 0 1");
        assert(throws<std::runtime_error>([&] { read_homography_file((dir / "H_text").string()); }));
        write_file(dir / "H_singular", "1 2 3 2 4 6 0 0 1");
        assert(throws<std::runtime_error>([&] { read_homography_file((dir / "H_singular").string()); }));
        write_file(dir / "H_empty", "");
        assert(throws<std::runtime_error>([&] { read_homography_file((dir / "H_empty").string()); }));
    }
    { // ground truth for every HPatches row of the database, none for MIRFlickr rows
        const fs::path hp = dir / "hp";
        write_file(hp / "v_a" / "H_1_2", "1 0 2 0 1 0 0 0 1");
        write_file(hp / "v_a" / "H_1_3", "1 0 3 0 1 0 0 0 1");
        write_file(hp / "i_b" / "H_1_2", "1 0 0 0 1 0 0 0 1");
        const std::vector<ImageRef> db = {
            {"v_a/2", "x"}, {"im1.jpg", "y"}, {"v_a/3", "x"}, {"i_b/2", "x"}, {"im2.jpg", "y"},
        };
        const auto gt = load_ground_truth(hp.string(), db);
        assert(gt.size() == 3);
        assert(gt.at(0).m == translation(2, 0).m);
        assert(gt.at(2).m == translation(3, 0).m);
        assert(gt.at(3).m == Homography::identity().m);
        assert(!gt.count(1) && !gt.count(4));

        const std::vector<ImageRef> missing = {{"v_a/2", "x"}, {"v_a/4", "x"}};
        assert(throws<std::runtime_error>([&] { load_ground_truth(hp.string(), missing); }));
        const std::vector<ImageRef> bad_id = {{"v_a/x", "x"}};
        assert(throws<std::runtime_error>([&] { load_ground_truth(hp.string(), bad_id); }));
        assert(load_ground_truth(hp.string(), {}).empty());
    }
    { // valid estimate: success, >= 4 inliers, finite, invertible
        assert(is_valid_estimate(verified(0, kGt)));
        assert(is_valid_estimate(verified(0, kGt, 4)));
        assert(!is_valid_estimate(verified(0, kGt, 3)));
        assert(!is_valid_estimate(failed(0)));
        CandidateResult not_ok = verified(0, kGt);
        not_ok.success = false;
        assert(!is_valid_estimate(not_ok));
        assert(!is_valid_estimate(verified(0, Homography(1, 2, 3, 2, 4, 6, 0, 0, 1))));   // singular
        Homography nan_h = kGt;
        nan_h.m[4] = std::nan("");
        assert(!is_valid_estimate(verified(0, nan_h)));
        Homography inf_h = kGt;
        inf_h.m[2] = INFINITY;
        assert(!is_valid_estimate(verified(0, inf_h)));
    }
    { // corner errors of the relevant pairs of one query
        // rows 10..14 relevant; 10 exact, 11 shifted by (3,4) -> 5 px, 12 failed,
        // 13 outside Top-C, 14 sends a corner of I1 to infinity; rows 0, 1 not relevant
        const std::unordered_map<ImageIndex, Homography> gt = {
            {10, kGt}, {11, kGt}, {12, kGt}, {13, kGt}, {14, kGt}};
        const Homography to_infinity(0, 0, 1, 0, 1, 0, 1, 0, 0);   // (0,0) -> third coordinate 0
        assert(is_valid_estimate(verified(14, to_infinity)));
        const std::vector<CandidateResult> top_c = {
            verified(0, Homography(1, 2, 3, 2, 4, 6, 0, 0, 1)),   // not relevant, never checked
            verified(14, to_infinity),
            verified(11, multiply(translation(3, 4), kGt)),
            failed(12),
            verified(1, kGt),
            verified(10, kGt),
        };
        const auto r = evaluate_relevant_pairs(top_c, {10, 11, 12, 13, 14}, gt, 640, 480);
        assert(r.size() == 5);
        for (std::size_t k = 0; k < 5; ++k) assert(r[k].row == 10 + k);
        assert(r[0].in_top_c && r[0].valid && r[0].corner_error < 1e-9);
        assert(r[1].in_top_c && r[1].valid && close(r[1].corner_error, 5.0));
        assert(r[2].in_top_c && !r[2].valid && std::isnan(r[2].corner_error));
        assert(!r[3].in_top_c && !r[3].valid && std::isnan(r[3].corner_error));
        assert(r[4].in_top_c && !r[4].valid && std::isnan(r[4].corner_error));

        // the error is measured at the corners of I1 (its size matters for a non-translation)
        const std::vector<CandidateResult> scaled = {verified(10, multiply(kGt, Homography(1.01, 0, 0, 0, 1.01, 0, 0, 0, 1)))};
        const double small = evaluate_relevant_pairs(scaled, {10}, gt, 100, 100)[0].corner_error;
        const double large = evaluate_relevant_pairs(scaled, {10}, gt, 1000, 1000)[0].corner_error;
        assert(small > 0 && large > 5 * small);

        // same error as compute_corner_error with the corners of I1
        const Homography est = multiply(kGt, Homography(1, 0.01, -2, 0.005, 0.99, 3, 1e-5, 0, 1));
        const double direct = compute_corner_error(est, kGt, 640, 480);
        assert(evaluate_relevant_pairs({verified(10, est)}, {10}, gt, 640, 480)[0].corner_error == direct);

        // a relevant row without ground truth is an error, even outside Top-C
        assert(throws<std::invalid_argument>([&] { evaluate_relevant_pairs({}, {10, 99}, gt, 640, 480); }));
        assert(evaluate_relevant_pairs(top_c, {}, gt, 640, 480).empty());
        // nothing verified: every pair is a failure without error
        for (const PairEvaluation& e : evaluate_relevant_pairs({}, {10, 11}, gt, 640, 480))
            assert(!e.in_top_c && !e.valid && std::isnan(e.corner_error));
    }
    { // summary: rates over all pairs, mean only over valid, strict thresholds
        const std::vector<PairEvaluation> v = {
            pair(0, true, true, 0.5), pair(1, true, true, 1.0), pair(2, true, true, 3.0),
            pair(3, true, true, 5.0), pair(4, true, true, 10.5),
            pair(5, true, false, 0), pair(6, false, false, 0), pair(7, false, false, 0),
        };
        const HomographySummary s = summarize_homographies(v);
        assert(s.pairs == 8 && s.in_top_c == 6 && s.valid == 5);
        assert(s.accurate1 == 1 && s.accurate3 == 2 && s.accurate5 == 3);
        assert(close(s.mean_corner_error, 20.0 / 5));
        assert(close(s.rate(s.valid), 5.0 / 8) && close(s.rate(s.accurate5), 3.0 / 8));

        const HomographySummary none = summarize_homographies({pair(0, true, false, 0), pair(1, false, false, 0)});
        assert(none.pairs == 2 && none.valid == 0 && std::isnan(none.mean_corner_error));
        assert(none.rate(none.valid) == 0.0 && none.rate(none.in_top_c) == 0.5);

        const HomographySummary empty = summarize_homographies({});
        assert(empty.pairs == 0 && std::isnan(empty.mean_corner_error) && empty.rate(0) == 0.0);

        // 115 pairs with 23 accurate: 20 %
        std::vector<PairEvaluation> split(115, pair(0, false, false, 0));
        for (std::size_t k = 0; k < 23; ++k) split[k] = pair(k, true, true, 2.0);
        assert(close(summarize_homographies(split).rate(summarize_homographies(split).accurate5), 0.2));
    }

    fs::remove_all(dir);
    return 0;
}