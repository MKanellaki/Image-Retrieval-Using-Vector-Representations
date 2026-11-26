// Tests for feature_store. They use a fake extractor instead of SIFT, so they need
// neither OpenCV nor the real datasets. Run with `make test`.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "feature_store.hpp"

using namespace image_search;
namespace fs = std::filesystem;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        ++g_checks;                                                                    \
        if (!(cond)) {                                                                 \
            ++g_failures;                                                              \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
        }                                                                              \
    } while (0)

template <class Ex, class F>
bool throws(F&& f) {
    try { f(); } catch (const Ex&) { return true; } catch (...) { return false; }
    return false;
}

static bool same(const ImageFeatures& a, const ImageFeatures& b) {
    return a.width == b.width && a.height == b.height && a.points == b.points &&
           a.descriptors.rows() == b.descriptors.rows() &&
           a.descriptors.cols() == b.descriptors.cols() &&
           a.descriptors.raw() == b.descriptors.raw();
}

// Deterministic fake features: depend only on the path. Points have fractional
// coordinates (like real SIFT), descriptors are whole numbers in [0, 255]
static ImageFeatures fake_features(const std::string& path, std::size_t n) {
    std::uint32_t s = 2166136261u;
    for (char c : path) s = (s ^ static_cast<unsigned char>(c)) * 16777619u;
    auto next = [&] { s = s * 1664525u + 1013904223u; return s >> 8; };

    ImageFeatures f;
    f.width = 640;
    f.height = 480;
    f.descriptors = Matrix(n, kSiftDim);
    for (std::size_t i = 0; i < n; ++i) {
        f.points.emplace_back((next() % 64000) / 100.0f, (next() % 48000) / 100.0f);
        for (std::size_t j = 0; j < kSiftDim; ++j)
            f.descriptors.row(i)[j] = static_cast<float>(next() % 256);
    }
    return f;
}

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

static void test_feature_path(const std::string& dir) {
    CHECK(fs::path(feature_path(dir, "v_bird/2")) == fs::path(dir) / "hpatches" / "v_bird" / "2.feat");
    CHECK(fs::path(feature_path(dir, "i_ajuntament/1")) == fs::path(dir) / "hpatches" / "i_ajuntament" / "1.feat");
    CHECK(fs::path(feature_path(dir, "im123.jpg")) == fs::path(dir) / "mir" / "im123.feat");
    CHECK(feature_path(dir, "v_bird/2") != feature_path(dir, "v_bird/3"));
    CHECK(feature_path(dir, "v_bird/2") != feature_path(dir, "i_bird/2"));
    CHECK(throws<std::invalid_argument>([&] { feature_path(dir, ""); }));
    CHECK(throws<std::invalid_argument>([&] { feature_path(dir, "/etc/passwd"); }));
    CHECK(throws<std::invalid_argument>([&] { feature_path(dir, "../escape.jpg"); }));
    CHECK(throws<std::invalid_argument>([&] { feature_path(dir, "v_bird/../../x"); }));
}

static void test_roundtrip(const std::string& dir) {
    // normal image, nested directory that does not exist yet
    const ImageFeatures f = fake_features("a", 37);
    const std::string p = feature_path(dir, "v_bird/2");
    save_features(p, f);
    CHECK(fs::exists(p));
    CHECK(!fs::exists(p + ".tmp"));
    CHECK(same(load_features(p), f));

    // zero keypoints: valid, and the image size is kept
    ImageFeatures empty;
    empty.width = 500;
    empty.height = 333;
    const std::string pe = feature_path(dir, "im7.jpg");
    save_features(pe, empty);
    const ImageFeatures le = load_features(pe);
    CHECK(le.size() == 0 && le.descriptors.rows() == 0 && le.descriptors.cols() == kSiftDim);
    CHECK(le.width == 500 && le.height == 333);

    // descriptor values 0 and 255 survive
    ImageFeatures edge = fake_features("edge", 2);
    edge.descriptors.row(0)[0] = 0.0f;
    edge.descriptors.row(1)[127] = 255.0f;
    const std::string px = feature_path(dir, "im8.jpg");
    save_features(px, edge);
    CHECK(same(load_features(px), edge));

    // file size is exactly the documented layout: 8 magic + 3*8 header + n*(8 + 128)
    CHECK(fs::file_size(p) == 8 + 24 + 37 * (2 * sizeof(float) + kSiftDim));
}

static void test_save_rejects(const std::string& dir) {
    const std::string p = (fs::path(dir) / "rejects" / "x.feat").string();

    ImageFeatures frac = fake_features("frac", 3);
    frac.descriptors.row(1)[5] = 12.5f;
    CHECK(throws<std::runtime_error>([&] { save_features(p, frac); }));

    ImageFeatures neg = fake_features("neg", 3);
    neg.descriptors.row(0)[0] = -1.0f;
    CHECK(throws<std::runtime_error>([&] { save_features(p, neg); }));

    ImageFeatures big = fake_features("big", 3);
    big.descriptors.row(2)[0] = 256.0f;
    CHECK(throws<std::runtime_error>([&] { save_features(p, big); }));

    ImageFeatures mismatch = fake_features("mm", 3);
    mismatch.points.pop_back();
    CHECK(throws<std::invalid_argument>([&] { save_features(p, mismatch); }));

    CHECK(!fs::exists(p));   // nothing written by a rejected save
}

static void test_load_rejects(const std::string& dir) {
    const std::string good = feature_path(dir, "im20.jpg");
    save_features(good, fake_features("good", 10));
    const std::string bytes = slurp(good);
    const std::string bad = (fs::path(dir) / "bad.feat").string();
    auto write = [&](const std::string& s) { std::ofstream(bad, std::ios::binary | std::ios::trunc) << s; };

    CHECK(throws<std::runtime_error>([&] { load_features((fs::path(dir) / "missing.feat").string()); }));

    write(bytes.substr(0, bytes.size() - 1));                       // truncated descriptors
    CHECK(throws<std::runtime_error>([&] { load_features(bad); }));

    write(bytes.substr(0, 20));                                     // truncated header
    CHECK(throws<std::runtime_error>([&] { load_features(bad); }));

    write(bytes + "x");                                             // trailing bytes
    CHECK(throws<std::runtime_error>([&] { load_features(bad); }));

    std::string wrong = bytes;
    wrong[0] = 'X';                                                 // wrong magic
    write(wrong);
    CHECK(throws<std::runtime_error>([&] { load_features(bad); }));

    std::string huge = bytes;                                       // absurd n, must not allocate it
    const std::uint64_t n = ~std::uint64_t{0} >> 1;
    std::memcpy(&huge[8 + 16], &n, sizeof n);
    write(huge);
    CHECK(throws<std::runtime_error>([&] { load_features(bad); }));

    write("");
    CHECK(throws<std::runtime_error>([&] { load_features(bad); }));
}

static void test_build_store(const std::string& root) {
    const std::string dir = (fs::path(root) / "store").string();
    std::vector<ImageRef> images;
    for (int k = 2; k <= 6; ++k)
        images.push_back({"v_seq/" + std::to_string(k), "/fake/v_seq/" + std::to_string(k) + ".ppm"});
    for (int k = 1; k <= 20; ++k)
        images.push_back({"im" + std::to_string(k) + ".jpg", "/fake/mir/im" + std::to_string(k) + ".jpg"});

    std::atomic<int> calls{0};
    auto factory = [&]() -> FeatureExtractorFactory {
        return [&] {
            return FeatureExtractFn([&](const std::string& path) {
                ++calls;
                return fake_features(path, 5 + path.size() % 7);
            });
        };
    };

    // first run extracts everything, every file holds that image's features
    CHECK(build_feature_store(images, dir, factory(), 4) == images.size());
    CHECK(calls == static_cast<int>(images.size()));
    for (const ImageRef& img : images)
        CHECK(same(load_features(feature_path(dir, img.id)), fake_features(img.path, 5 + img.path.size() % 7)));

    // second run: nothing to do, nothing extracted, files untouched
    const std::string before = slurp(feature_path(dir, "im3.jpg"));
    calls = 0;
    CHECK(build_feature_store(images, dir, factory(), 4) == 0);
    CHECK(calls == 0);
    CHECK(slurp(feature_path(dir, "im3.jpg")) == before);

    // resumable: delete two files, only those are redone
    fs::remove(feature_path(dir, "v_seq/4"));
    fs::remove(feature_path(dir, "im17.jpg"));
    calls = 0;
    CHECK(build_feature_store(images, dir, factory(), 3) == 2);
    CHECK(calls == 2);
    CHECK(fs::exists(feature_path(dir, "v_seq/4")) && fs::exists(feature_path(dir, "im17.jpg")));

    // same result with 1 thread and with many
    const std::string d1 = (fs::path(root) / "store1").string();
    const std::string d8 = (fs::path(root) / "store8").string();
    build_feature_store(images, d1, factory(), 1);
    build_feature_store(images, d8, factory(), 8);
    for (const ImageRef& img : images)
        CHECK(slurp(feature_path(d1, img.id)) == slurp(feature_path(d8, img.id)));

    // an extractor error is rethrown, and the failing image leaves no file
    const std::string de = (fs::path(root) / "store_err").string();
    auto failing = [&]() -> FeatureExtractorFactory {
        return [] {
            return FeatureExtractFn([](const std::string& path) {
                if (path.find("im5.jpg") != std::string::npos) throw std::runtime_error("cannot read image: " + path);
                return fake_features(path, 4);
            });
        };
    };
    CHECK(throws<std::runtime_error>([&] { build_feature_store(images, de, failing(), 4); }));
    CHECK(!fs::exists(feature_path(de, "im5.jpg")));

    // empty input is fine
    CHECK(build_feature_store({}, dir, factory(), 4) == 0);
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("feature_store_test_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);

    std::printf("feature_path\n");
    test_feature_path(dir.string());
    std::printf("save/load round trip\n");
    test_roundtrip(dir.string());
    std::printf("save rejects bad input\n");
    test_save_rejects(dir.string());
    std::printf("load rejects bad files\n");
    test_load_rejects(dir.string());
    std::printf("build_feature_store\n");
    test_build_store(dir.string());

    fs::remove_all(dir);
    std::printf("test_feature_store: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}