#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <functional>
#include "seed.hpp"
#include "types.hpp"

namespace image_search {

constexpr std::size_t kSiftDim = 128;

// Reads an image as grayscale and returns all SIFT descriptors as an n x 128 matrix (n may be 0)
//  OpenCV is used only inside features.cpp
// The same function is used for HPatches (.ppm) and MIRFlickr (.jpg)
// Throws std::runtime_error if the file cannot be read
// a missing or corrupt image is an input error, not "an image with zero descriptors"
// Not thread-safe on a single instance: give each thread its own extractor
class SiftExtractor {
public:
    SiftExtractor();
    ~SiftExtractor();

    SiftExtractor(SiftExtractor&&) noexcept;
    SiftExtractor& operator=(SiftExtractor&&) noexcept;
    SiftExtractor(const SiftExtractor&) = delete;
    SiftExtractor& operator=(const SiftExtractor&) = delete;

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