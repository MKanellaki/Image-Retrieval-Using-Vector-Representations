#include "feature_store.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "binary_io.hpp"
#include "bow.hpp"   // resolve_threads

namespace fs = std::filesystem;

namespace image_search {

namespace {

constexpr char kFeatMagic[9] = "SIFTFT01";
constexpr std::uint64_t kMaxKeypoints = std::uint64_t{1} << 22;   // sanity bound before allocating

}  // namespace

std::string feature_path(const std::string& dir, const std::string& id) {
    if (id.empty())
        throw std::invalid_argument("feature_path: empty image id");
    const fs::path rel(id);
    if (rel.is_absolute())
        throw std::invalid_argument("feature_path: absolute image id: " + id);
    for (const auto& part : rel)
        if (part == "..")
            throw std::invalid_argument("feature_path: image id leaves the store: " + id);

    // HPatches ids are "<sequence>/<k>", MIRFlickr ids are file names
    const bool hpatches = id.find('/') != std::string::npos;
    fs::path out = fs::path(dir) / (hpatches ? "hpatches" : "mir") / rel;
    out.replace_extension(".feat");   // "2" -> "2.feat", "im123.jpg" -> "im123.feat"
    return out.string();
}

void save_features(const std::string& path, const ImageFeatures& f) {
    const std::size_t n = f.size();
    if (f.descriptors.rows() != n || f.descriptors.cols() != kSiftDim)
        throw std::invalid_argument("save_features: descriptors do not match the points: " + path);

    std::vector<float> xy;
    xy.reserve(2 * n);
    for (const Point2D& p : f.points) {
        xy.push_back(static_cast<float>(p.x));   // kp.pt is float, so this is lossless
        xy.push_back(static_cast<float>(p.y));
    }

    const auto& src = f.descriptors.raw();
    std::vector<std::uint8_t> d(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        const float v = src[i];
        if (!(v >= 0.0f && v <= 255.0f) || v != std::nearbyint(v))
            throw std::runtime_error("descriptor value is not a byte in " + path);
        d[i] = static_cast<std::uint8_t>(v);
    }

    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);

    const std::string tmp = path + ".tmp";
    {
        binio::Writer w(tmp);
        w.magic(kFeatMagic);
        w.u64(f.width);
        w.u64(f.height);
        w.u64(n);
        w.bytes(xy.data(), xy.size() * sizeof(float));
        w.bytes(d.data(), d.size());
        w.close();
    }
    fs::rename(tmp, path);
}

ImageFeatures load_features(const std::string& path) {
    binio::Reader r(path);
    r.expect_magic(kFeatMagic);

    ImageFeatures f;
    const std::uint64_t w = r.u64(), h = r.u64(), n = r.u64();
    if (w > UINT32_MAX || h > UINT32_MAX) r.fail("implausible image size");
    if (n > kMaxKeypoints) r.fail("implausible keypoint count");
    f.width  = static_cast<std::uint32_t>(w);
    f.height = static_cast<std::uint32_t>(h);

    std::vector<float> xy(2 * n);
    r.bytes(xy.data(), xy.size() * sizeof(float));
    std::vector<std::uint8_t> d(n * kSiftDim);
    r.bytes(d.data(), d.size());
    r.expect_end();

    f.points.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        f.points.emplace_back(xy[2 * i], xy[2 * i + 1]);
    f.descriptors = Matrix(n, kSiftDim);
    std::copy(d.begin(), d.end(), f.descriptors.raw().begin());   // uint8 -> float
    return f;
}

FeatureExtractorFactory sift_feature_factory() {
    return [] {
        auto ex = std::make_shared<SiftExtractor>();
        return FeatureExtractFn([ex](const std::string& p) { return ex->extract_features(p); });
    };
}

std::size_t build_feature_store(const std::vector<ImageRef>& images, const std::string& dir,
                                const FeatureExtractorFactory& factory, unsigned threads) {
    // Decide what is missing up front, so the return value and the work list agree
    std::vector<std::size_t> todo;
    for (std::size_t i = 0; i < images.size(); ++i)
        if (!fs::exists(feature_path(dir, images[i].id)))
            todo.push_back(i);
    if (todo.empty()) return 0;

    const std::size_t workers = std::min<std::size_t>(resolve_threads(threads), todo.size());
    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::exception_ptr error;
    std::mutex error_mutex;

    auto worker = [&] {
        try {
            FeatureExtractFn extract = factory();
            while (!stop.load(std::memory_order_relaxed)) {
                const std::size_t k = next.fetch_add(1, std::memory_order_relaxed);
                if (k >= todo.size()) return;
                const ImageRef& img = images[todo[k]];
                save_features(feature_path(dir, img.id), extract(img.path));
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
            stop.store(true, std::memory_order_relaxed);
        }
    };

    if (workers == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (std::size_t t = 0; t < workers; ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
    }
    if (error) std::rethrow_exception(error);
    return todo.size();
}

}  // namespace image_search