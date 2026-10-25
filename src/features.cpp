#include "features.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

namespace image_search {

// SIFT extraction

struct SiftExtractor::Impl {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();   // OpenCV default parameters
};

SiftExtractor::SiftExtractor() : impl_(std::make_unique<Impl>()) {}
SiftExtractor::~SiftExtractor() = default;
SiftExtractor::SiftExtractor(SiftExtractor&&) noexcept = default;
SiftExtractor& SiftExtractor::operator=(SiftExtractor&&) noexcept = default;

Matrix SiftExtractor::extract(const std::string& path) const {
    cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(gray.empty())
        throw std::runtime_error("cannot read image: " + path);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat desc;
    impl_->sift->detectAndCompute(gray, cv::noArray(), keypoints, desc);

    Matrix out(0, kSiftDim);
    if(desc.empty())
        return out;                                   // valid: zero descriptors

    if(desc.cols != static_cast<int>(kSiftDim))
        throw std::runtime_error("unexpected SIFT descriptor size in " + path);
    if(desc.type() != CV_32F)
        desc.convertTo(desc, CV_32F);

    out = Matrix(static_cast<std::size_t>(desc.rows), kSiftDim);
    for(int r = 0; r < desc.rows; ++r) {             // row-wise copy: no continuity assumption
        const float* src = desc.ptr<float>(r);
        std::copy(src, src + kSiftDim, out.row(static_cast<std::size_t>(r)));
    }
    return out;
}

// sampling

namespace {

// Uniform integer in [0, bound), bound > 0, by rejection sampling
std::uint64_t uniform_below(std::mt19937_64& rng, std::uint64_t bound) {
    const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max()
                              - std::numeric_limits<std::uint64_t>::max() % bound;
    std::uint64_t x;
    do { x = rng(); } while (x >= limit);
    return x % bound;
}

}  // namespace

std::uint64_t derive_seed(std::uint64_t seed, std::uint64_t index) {
    std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (index + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

Matrix sample_descriptors(const Matrix& descriptors, std::size_t S, std::uint64_t seed) {
    const std::size_t n = descriptors.rows();
    if(n <= S)
        return descriptors;                           // fewer than S: use all of them

    // Partial Fisher-Yates: the first S slots end up as a uniform S-subset
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::mt19937_64 rng(seed);
    for(std::size_t i = 0; i < S; ++i) {
        std::size_t j = i + static_cast<std::size_t>(uniform_below(rng, n - i));
        std::swap(idx[i], idx[j]);
    }
    idx.resize(S);
    std::sort(idx.begin(), idx.end());                // keep original order

    Matrix out(0, descriptors.cols());
    out.raw().reserve(S * descriptors.cols());
    for (std::size_t i : idx)
        out.append_row(descriptors.row(i));
    return out;
}

ExtractorFactory sift_extractor_factory() {
    cv::setNumThreads(1);
    return [] {
        auto ex = std::make_shared<SiftExtractor>();
        return ExtractFn([ex](const std::string& p) { return ex->extract(p); });
    };
}

}