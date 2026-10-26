#pragma once
// Builds every image vector the experiments need, once, and caches them
//
// Order
//   1. Vocabulary: extract the training images (all six images of every train sequence), 
//      sample up to S descriptors from image i with derive_seed(seed, i), run k-means
//   2. Encode the validation and test HPatches queries (image 1) and database images (2..6)
//   3. Encode MIRFlickr in number order, skipping images with zero descriptors, until
//      mir_target valid images are found or the files run out. Any D is the first D rows
//
// The cache file name contains everything that changes the vectors: K, S, seed, mir_target and a hash of the split file's contents
// Runs that differ only in -D, -set or the search method reuse it and skip SIFT extraction entirely
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bow.hpp"
#include "dataset.hpp"
#include "types.hpp"

namespace image_search {

constexpr std::size_t kMirFlickrTarget = 25000;   // largest D in the experiments

struct BowConfig {
    std::string hp_dir;
    std::string mir_dir;
    std::string split_file;
    std::string cache_dir = "cache";
    std::size_t K = 256;
    std::size_t S = 500;
    std::uint64_t seed = 1;
    std::size_t mir_target = kMirFlickrTarget;
    unsigned threads = 0;                 // 0 -> hardware concurrency
    bool verbose = true;                  // progress on stderr
};

// Image vectors in a fixed order, with their ids ("v_bird/2", "im123.jpg").
struct EncodedImages {
    std::vector<std::string> ids;
    Matrix vectors;                       // ids.size() x K
};

struct BowData {
    Matrix centers;                       // the vocabulary, K x 128
    EncodedImages validation_queries;     // order of make_queries(..., Validation)
    EncodedImages validation_db;          // order of hpatches_database(..., Validation)
    EncodedImages test_queries;
    EncodedImages test_db;
    EncodedImages mirflickr;              // first valid images in number order
    std::size_t mir_skipped = 0;          // zero-descriptor images skipped while filling it
};

// The values that identify a cache file. Stored in the file and checked on load.
struct CacheKey {
    std::size_t K = 0, S = 0, mir_target = 0;
    std::uint64_t seed = 0;
    std::uint64_t split_hash = 0;         // FNV-1a 64 of the split file's bytes
};

CacheKey cache_key(const BowConfig& config);
std::string cache_path(const BowConfig& config, const CacheKey& key);

// Runs steps 1-3 without touching the cache.
BowData build_bow_data(const BowConfig& config, const ExtractorFactory& factory);

// Loads the cache if it exists and matches; otherwise builds and writes it.
// After loading, the HPatches ids are checked against the current split file and
// hp_dir, so a stale cache is reported instead of silently used.
BowData load_or_build_bow_data(const BowConfig& config, const ExtractorFactory& factory);

// The cache file itself. Written to "<path>.tmp" and renamed, so an interrupted run
// never leaves a half-written cache behind.
void save_bow_cache(const std::string& path, const CacheKey& key, const BowData& data);
BowData load_bow_cache(const std::string& path, const CacheKey& expected);

// Search database for one query set: its 115 HPatches rows followed by the first D
// MIRFlickr rows. `refs` receives matching ImageRefs (id only, empty path), which is
// what relevant_rows() needs. Throws if fewer than D MIRFlickr images were encoded.
Matrix make_database(const BowData& data, Split which, std::size_t D, std::vector<ImageRef>* refs);

const EncodedImages& queries_for(const BowData& data, Split which);

}  // namespace image_search
