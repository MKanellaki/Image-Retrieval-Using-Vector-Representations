#pragma once
// On-disk store of per-image SIFT features (keypoint positions + descriptors), one file
// per image, used by geometric verification so that SIFT never runs at query time
//
// File layout (native byte order, like the other caches):
//   magic "SIFTFT01", u64 width, u64 height, u64 n,
//   n x 2 float (x, y), n x 128 uint8 descriptors
// OpenCV SIFT descriptors are whole numbers in [0, 255], so uint8 is lossless;
// save_features throws if a value is not, instead of silently rounding
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "dataset.hpp"
#include "features.hpp"

namespace image_search {

// "v_bird/2"  -> <dir>/hpatches/v_bird/2.feat
// "im123.jpg" -> <dir>/mir/im123.feat
// Throws std::invalid_argument for an empty id or one that could escape dir (absolute, "..")
std::string feature_path(const std::string& dir, const std::string& id);

// Writes "<path>.tmp" and renames it, so an interrupted run never leaves a half-written file
// Creates the parent directories
void save_features(const std::string& path, const ImageFeatures& f);

// Throws std::runtime_error for a missing, truncated or foreign file
ImageFeatures load_features(const std::string& path);

// Extracts the features of one image. A factory makes one per worker thread
using FeatureExtractFn        = std::function<ImageFeatures(const std::string& path)>;
using FeatureExtractorFactory = std::function<FeatureExtractFn()>;

// Factory that gives every worker its own SiftExtractor
FeatureExtractorFactory sift_feature_factory();

// Extracts and saves every image whose file does not exist yet (resumable: rerunning
// after a crash only does the missing ones). threads = 0 -> hardware concurrency
// The first exception from any worker stops the others and is rethrown
// Returns how many images were extracted (0 if everything was already stored)
std::size_t build_feature_store(const std::vector<ImageRef>& images, const std::string& dir,
                                const FeatureExtractorFactory& factory, unsigned threads = 0);

}  // namespace image_search