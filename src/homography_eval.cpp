#include "homography_eval.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace image_search {

Homography read_homography_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open homography file: " + path);

    std::array<double, 9> m{};
    for (double& v : m)
        if (!(in >> v)) throw std::runtime_error("expected 9 numbers in homography file: " + path);
    std::string extra;
    if (in >> extra) throw std::runtime_error("more than 9 values in homography file: " + path);

    const Homography H(m);
    if (!H.is_finite() || !H.is_invertible())
        throw std::runtime_error("homography file is not a finite invertible matrix: " + path);
    return H;
}

std::unordered_map<ImageIndex, Homography> load_ground_truth(const std::string& hp_dir, const std::vector<ImageRef>& db) {
    std::unordered_map<ImageIndex, Homography> gt;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const std::string& id = db[i].id;
        const std::size_t slash = id.find('/');
        if (slash == std::string::npos) continue;          // MIRFlickr image

        const std::string sequence = id.substr(0, slash);
        const std::string j = id.substr(slash + 1);
        if (sequence.empty() || j.empty() ||
            !std::all_of(j.begin(), j.end(), [](char c) { return c >= '0' && c <= '9'; }))
            throw std::runtime_error("not an HPatches image id: " + id);

        const fs::path path = fs::path(hp_dir) / sequence / ("H_1_" + j);
        gt.emplace(static_cast<ImageIndex>(i), read_homography_file(path.string()));
    }
    return gt;
}

bool is_valid_estimate(const CandidateResult& c) {
    return c.success && c.inliers >= 4 && c.H.is_finite() && c.H.is_invertible();
}

std::vector<PairEvaluation> evaluate_relevant_pairs(
    const std::vector<CandidateResult>& top_c, const std::unordered_set<ImageIndex>& relevant,
    const std::unordered_map<ImageIndex, Homography>& ground_truth, double width, double height) {
    std::unordered_map<ImageIndex, const CandidateResult*> by_row;
    for (const CandidateResult& c : top_c) by_row.emplace(c.row, &c);

    std::vector<ImageIndex> rows(relevant.begin(), relevant.end());
    std::sort(rows.begin(), rows.end());

    std::vector<PairEvaluation> out;
    out.reserve(rows.size());
    for (ImageIndex row : rows) {
        const auto gt = ground_truth.find(row);
        if (gt == ground_truth.end())
            throw std::invalid_argument("no ground-truth homography for relevant row " + std::to_string(row));

        PairEvaluation e;
        e.row = row;
        const auto it = by_row.find(row);
        if (it != by_row.end()) {
            e.in_top_c = true;
            if (is_valid_estimate(*it->second)) {
                const double err = compute_corner_error(it->second->H, gt->second, width, height);
                if (std::isfinite(err)) {
                    e.valid = true;
                    e.corner_error = err;
                }
            }
        }
        out.push_back(e);
    }
    return out;
}

double HomographySummary::rate(std::size_t count) const {
    return pairs == 0 ? 0.0 : static_cast<double>(count) / static_cast<double>(pairs);
}

HomographySummary summarize_homographies(const std::vector<PairEvaluation>& pairs) {
    HomographySummary s;
    s.pairs = pairs.size();
    double sum = 0.0;
    for (const PairEvaluation& e : pairs) {
        if (e.in_top_c) ++s.in_top_c;
        if (!e.valid) continue;
        ++s.valid;
        sum += e.corner_error;
        if (e.corner_error < 1.0) ++s.accurate1;
        if (e.corner_error < 3.0) ++s.accurate3;
        if (e.corner_error < 5.0) ++s.accurate5;
    }
    if (s.valid > 0) s.mean_corner_error = sum / static_cast<double>(s.valid);
    return s;
}

}  // namespace image_search