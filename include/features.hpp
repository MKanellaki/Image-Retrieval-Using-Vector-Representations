#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <functional>
#include <vector>
#include "seed.hpp"
#include "types.hpp"


namespace image_search {

constexpr std::size_t kSiftDim = 128;

// All SIFT features of one image. points[i] is the location of descriptors.row(i)
// Coordinates are in pixels of the full-resolution image (the frame the HPatches H_1_j uses)
struct ImageFeatures {
    std::uint32_t width = 0, height = 0;   // image size (I1 corners for the corner error)
    std::vector<Point2D> points;
    Matrix descriptors{0, kSiftDim};       // n x 128
    std::size_t size() const { return points.size(); }
};

// Reads an image as grayscale and runs OpenCV SIFT (default parameters) on it
//   extract_features: keypoint positions + descriptors, points[i] belongs to descriptors.row(i),
//                     plus the image width and height. Used by geometric verification
//   extract:          only the n x 128 descriptor matrix (n may be 0). Used by BoVW;
//                     always equals extract_features(path).descriptors
// OpenCV is used only inside features.cpp
// The same functions are used for HPatches (.ppm) and MIRFlickr (.jpg)
// Throws std::runtime_error if the file cannot be read
// a missing or corrupt image is an input error, not "an image with zero descriptors"
// Not thread-safe on a single instance: give each thread its own extractorclass
class SiftExtractor {
public:
    SiftExtractor();
    ~SiftExtractor();

    SiftExtractor(SiftExtractor&&) noexcept;
    SiftExtractor& operator=(SiftExtractor&&) noexcept;
    SiftExtractor(const SiftExtractor&) = delete;
    SiftExtractor& operator=(const SiftExtractor&) = delete;

    ImageFeatures extract_features(const std::string& path) const;
    Matrix extract(const std::string& path) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Chooses up to S rows uniformly at random, without replacement (all rows if there are fewer than S)
// Selected rows keep their original relative order
// The result depends only on (descriptors, S, seed)
// Use a different seed per image (e.g. derive_seed(global_seed, image_index), from seed.hpp),
// otherwise images with equal descriptor counts get the same row positions.
Matrix sample_descriptors(const Matrix& descriptors, std::size_t S, std::uint64_t seed);

// Extracts the descriptors of one image. A factory makes one per worker thread.
using ExtractFn        = std::function<Matrix(const std::string& path)>;
using ExtractorFactory = std::function<ExtractFn()>;

// Factory that gives every worker its own SiftExtractor.
ExtractorFactory sift_extractor_factory();

}