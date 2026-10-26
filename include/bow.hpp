#pragma once
// Bag of visual words: the shared vocabulary and the image 
// K-histogram encoding.
//
// Database descriptors are never kept
// each image is turned into its K-length histogram as soon as it is extracted and its descriptors are dropped

// All parallel functions give bit-identical output for any thread count
// every image is written to its own fixed output row, and any randomness is seeded per image (derive_seed(seed, i)), never per thread
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dataset.hpp"
#include "features.hpp"
#include "types.hpp"

namespace image_search {

// The shared visual vocabulary: K centers in R^128, trained on the training split only
class Vocabulary {
public:
    Vocabulary() = default;
    explicit Vocabulary(Matrix centers) : centers_(std::move(centers)) {}

    std::size_t size() const { return centers_.rows(); }   // K
    std::size_t dim() const { return centers_.cols(); }    // 128 for SIFT
    const Matrix& centers() const { return centers_; }

    // Assigns every descriptor to its nearest center (ties -> lowest index) andreturns the K-dimensional histogram divided by n_I (non-negative, sums to 1)
    // Throws std::invalid_argument if descriptors is empty or has the wrong width
    std::vector<float> encode(const Matrix& descriptors) const;

    // Same, written into out[0..K)
    void encode_into(const Matrix& descriptors, float* out) const;

    void save(const std::string& path) const;
    static Vocabulary load(const std::string& path);

private:
    Matrix centers_;
};

// Trains the vocabulary from already-sampled training descriptors (k-means, k-means++ init)
Vocabulary train_vocabulary(const Matrix& sampled, std::size_t K, std::uint64_t seed);

// 0 -> std::thread::hardware_concurrency() (at least 1)
unsigned resolve_threads(unsigned requested);

// Step 1 input: extracts every image and keeps up to S descriptors of image i, sampled with derive_seed(seed, i)
// Rows are concatenated in image order
Matrix collect_training_descriptors(const std::vector<ImageRef>& images, std::size_t S,
                                    std::uint64_t seed, const ExtractorFactory& factory,
                                    unsigned threads);

// Step 2: one K-histogram per image, row i = images[i]
// Every image must yield at least one descriptor (true for HPatches) otherwise throws
Matrix encode_images(const std::vector<ImageRef>& images, const Vocabulary& vocab,
                     const ExtractorFactory& factory, unsigned threads);

// Step 3: walks candidates in order, skips images with zero descriptors, and stops once target valid images are encoded (or the candidates run out) 
// Candidates are processed in batches of batch (0 -> automatic)
// a batch is accepted strictly in candidate order, so the result does not depend on the thread count or the batch size
struct ValidPrefix {
    std::vector<std::string> ids;   // ids of the accepted images, in order
    Matrix vectors;                 // ids.size() x K
    std::size_t skipped = 0;        // images rejected for having zero descriptors
};
ValidPrefix encode_first_valid(const std::vector<ImageRef>& candidates, std::size_t target,
                               const Vocabulary& vocab, const ExtractorFactory& factory,
                               unsigned threads, std::size_t batch = 0);

// Binary matrix file (magic, rows, cols, row-major float32 in native byte order)
void save_vectors(const std::string& path, const Matrix& vectors);
Matrix load_vectors(const std::string& path);

}  // namespace image_search
