#include "bow.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "binary_io.hpp"
#include "kmeans.hpp"

namespace image_search {

namespace {

constexpr char kVectorsMagic[9] = "IMSVEC01";
constexpr char kVocabMagic[9]   = "IMSVOC01";

// Runs body(i, extract) for every i in [0, n) on up to threads workers
// Each worker makes its own ExtractFn from the factory
// Work is handed out with an atomic counter, so which thread handles which i varies, 
// but body must only write to slot i, which makes the result independent of the thread count
// The first exception thrown by any worker stops the others and is rethrown here
template <class Body>
void parallel_for(std::size_t n, unsigned threads, const ExtractorFactory& factory, Body&& body) {
    if (n == 0) return;
    const std::size_t workers = std::min<std::size_t>(resolve_threads(threads), n);

    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::exception_ptr error;
    std::mutex error_mutex;

    auto worker = [&] {
        try {
            ExtractFn extract = factory();
            while (!stop.load(std::memory_order_relaxed)) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) return;
                body(i, extract);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
            stop.store(true, std::memory_order_relaxed);
        }
    };

    if(workers == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (std::size_t t = 0; t < workers; ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
    }
    if(error) std::rethrow_exception(error);
}

void check_width(const Matrix& descriptors, std::size_t dim, const ImageRef& image) {
    if(descriptors.cols() != dim)
        throw std::runtime_error(image.id + ": descriptors have width " +
                                 std::to_string(descriptors.cols()) + ", vocabulary expects " +
                                 std::to_string(dim));
}

void write_matrix_file(const std::string& path, const char (&magic)[9], const Matrix& m) {
    binio::Writer w(path);
    w.magic(magic);
    w.matrix(m);
    w.close();
}

Matrix read_matrix_file(const std::string& path, const char (&magic)[9]) {
    binio::Reader r(path);
    r.expect_magic(magic);
    Matrix m = r.matrix();
    r.expect_end();
    return m;
}

}  // namespace

// ===Vocabulary===

void Vocabulary::encode_into(const Matrix& descriptors, float* out) const {
    if(centers_.empty()) throw std::invalid_argument("encode: empty vocabulary");
    if(descriptors.empty()) throw std::invalid_argument("encode: image has no descriptors");
    if(descriptors.cols() != dim()) {
        throw std::invalid_argument("encode: descriptor width does not match the vocabulary");
    }

    const std::size_t K = size();
    std::vector<std::uint32_t> counts(K, 0);
    for(std::size_t t = 0; t < descriptors.rows(); ++t) {
        ++counts[nearest_center(descriptors.row(t), centers_)];
    }
    const double n = static_cast<double>(descriptors.rows());
    for(std::size_t j = 0; j < K; ++j) {
        out[j] = static_cast<float>(static_cast<double>(counts[j]) / n);
    }
}

std::vector<float> Vocabulary::encode(const Matrix& descriptors) const {
    std::vector<float> y(size());
    encode_into(descriptors, y.data());
    return y;
}

void Vocabulary::save(const std::string& path) const { write_matrix_file(path, kVocabMagic, centers_); }

Vocabulary Vocabulary::load(const std::string& path) {
    return Vocabulary(read_matrix_file(path, kVocabMagic));
}

Vocabulary train_vocabulary(const Matrix& sampled, std::size_t K, std::uint64_t seed) {
    if(K == 0) throw std::invalid_argument("train_vocabulary: K must be positive");
    if(sampled.rows() < K) {
        throw std::invalid_argument("train_vocabulary: " + std::to_string(sampled.rows()) + " training descriptors are fewer than K = " + std::to_string(K));
    }
    KMeansParams params;
    params.k = K;
    params.seed = seed;
    return Vocabulary(kmeans_fit(sampled, params).centers);
}

// ===pipeline steps===

unsigned resolve_threads(unsigned requested) {
    if (requested > 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 1;
}

Matrix collect_training_descriptors(const std::vector<ImageRef>& images, std::size_t S, std::uint64_t seed, const ExtractorFactory& factory, unsigned threads) {
    if(S == 0) throw std::invalid_argument("collect_training_descriptors: S must be positive");

    // Only the (at most S) sampled rows of each image are kept
    std::vector<Matrix> slots(images.size());
    parallel_for(images.size(), threads, factory, [&](std::size_t i, const ExtractFn& extract) {
        const Matrix d = extract(images[i].path);
        if(!d.empty() && d.cols() != kSiftDim) check_width(d, kSiftDim, images[i]);
        slots[i] = sample_descriptors(d, S, derive_seed(seed, i));
    });

    std::size_t total = 0;
    for(const auto& s : slots) total += s.rows();
    if(total == 0) throw std::runtime_error("no SIFT descriptors in any training image");

    Matrix out(total, kSiftDim);
    std::size_t row = 0;
    for(auto& s : slots) {
        std::copy(s.raw().begin(), s.raw().end(), out.row(row));
        row += s.rows();
        s = Matrix();   // release as we go
    }
    return out;
}

Matrix encode_images(const std::vector<ImageRef>& images, const Vocabulary& vocab, const ExtractorFactory& factory, unsigned threads) {
    Matrix out(images.size(), vocab.size());
    parallel_for(images.size(), threads, factory, [&](std::size_t i, const ExtractFn& extract) {
        const Matrix d = extract(images[i].path);
        if (d.empty()) throw std::runtime_error("no SIFT descriptors in " + images[i].id);
        check_width(d, vocab.dim(), images[i]);
        vocab.encode_into(d, out.row(i));
    });   // descriptors of image i are freed here, only its histogram remains
    return out;
}

ValidPrefix encode_first_valid(const std::vector<ImageRef>& candidates, std::size_t target, const Vocabulary& vocab, const ExtractorFactory& factory, unsigned threads, std::size_t batch) {
    const std::size_t K = vocab.size();
    ValidPrefix result;
    result.vectors = Matrix(0, K);
    result.vectors.raw().reserve(std::min(target, candidates.size()) * K);

    const unsigned workers = resolve_threads(threads);
    if (batch == 0) batch = std::max<std::size_t>(256, 32 * static_cast<std::size_t>(workers));

    Matrix buffer(batch, K);
    std::vector<char> valid(batch);

    std::size_t pos = 0;
    while (pos < candidates.size() && result.ids.size() < target) {
        // Never extract more images than could still be needed
        const std::size_t m = std::min({batch, candidates.size() - pos, target - result.ids.size()});

        parallel_for(m, workers, factory, [&](std::size_t i, const ExtractFn& extract) {
            const ImageRef& image = candidates[pos + i];
            const Matrix d = extract(image.path);
            valid[i] = !d.empty();
            if (valid[i]) {
                check_width(d, vocab.dim(), image);
                vocab.encode_into(d, buffer.row(i));
            }
        });

        // Accept strictly in candidate order, an empty image is replaced by the next one
        for (std::size_t i = 0; i < m; ++i) {
            if (!valid[i]) {
                ++result.skipped;
                continue;
            }
            result.vectors.append_row(buffer.row(i));
            result.ids.push_back(candidates[pos + i].id);
        }
        pos += m;
    }
    return result;
}

// ===matrix files===

void save_vectors(const std::string& path, const Matrix& vectors) {
    write_matrix_file(path, kVectorsMagic, vectors);
}

Matrix load_vectors(const std::string& path) { return read_matrix_file(path, kVectorsMagic); }

}  // namespace image_search
