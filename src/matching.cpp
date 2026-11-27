#include "matching.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

namespace image_search {

namespace {

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Query rows per block: the block of dot products is kBlock x ni floats
// (256 x 10000 = ~10 MB), so memory stays bounded for any image size
constexpr std::size_t kBlock = 256;

}  // namespace

// Squared distances by the expansion ||q - b||^2 = ||q||^2 + ||b||^2 - 2 q.b, with the
// dot products of a block of query rows against all candidate rows as one matrix product
// SIFT values are whole numbers in [0, 255], so every norm and dot product is an integer
// below 128 * 255^2 < 2^24 and is exact in float; the sum is formed in double, so the
// distances are exact and equal to the direct per-coordinate computation
std::vector<Match2D> match_sift(const ImageFeatures& query, const ImageFeatures& candidate, double rho) {
    if (!(rho > 0.0 && rho < 1.0))
        throw std::invalid_argument("match_sift: rho must be in (0, 1)");
    const Matrix& Qm = query.descriptors;
    const Matrix& Im = candidate.descriptors;
    if (Qm.rows() > 0 && Im.rows() > 0 && Qm.cols() != Im.cols())
        throw std::invalid_argument("match_sift: descriptor widths differ");

    std::vector<Match2D> matches;
    const std::size_t nq = Qm.rows(), ni = Im.rows(), dim = Qm.cols();
    if (nq == 0 || ni < 2) return matches;

    const Eigen::Map<const RowMatrix> Q(Qm.raw().data(), nq, dim);
    const Eigen::Map<const RowMatrix> I(Im.raw().data(), ni, dim);
    const Eigen::VectorXf qn = Q.rowwise().squaredNorm();
    const Eigen::VectorXf in = I.rowwise().squaredNorm();

    const double rho2 = rho * rho;
    RowMatrix dots;
    for (std::size_t r0 = 0; r0 < nq; r0 += kBlock) {
        const std::size_t rows = std::min(kBlock, nq - r0);
        dots.noalias() = Q.middleRows(r0, rows) * I.transpose();   // rows x ni

        for (std::size_t r = 0; r < rows; ++r) {
            const double a = qn[r0 + r];
            double d1 = std::numeric_limits<double>::infinity();   // nearest squared distance
            double d2 = d1;                                         // second nearest
            std::size_t j1 = 0;
            for (std::size_t j = 0; j < ni; ++j) {
                const double d = a + static_cast<double>(in[j]) - 2.0 * static_cast<double>(dots(r, j));
                if (d < d1) { d2 = d1; d1 = d; j1 = j; }
                else if (d < d2) { d2 = d; }
            }
            if (d1 < rho2 * d2)
                matches.push_back(Match2D{query.points[r0 + r], candidate.points[j1]});
        }
    }
    return matches;
}

}  // namespace image_search