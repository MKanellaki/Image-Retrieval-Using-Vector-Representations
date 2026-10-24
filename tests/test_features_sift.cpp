// Tests for SiftExtractor. Needs OpenCV (linked through features.cpp); the test itself does not.
// Images are written as binary PPM files by hand, in a temporary directory.
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "features.hpp"

namespace fs = std::filesystem;
using namespace image_search;

// 256x256 grayscale-looking P6 image. flat: constant gray (SIFT finds nothing);
// otherwise 60 deterministic random rectangles (SIFT finds keypoints).
static void write_ppm(const fs::path& path, bool flat) {
    const int w = 256, h = 256;
    std::vector<unsigned char> px(static_cast<std::size_t>(w) * h, 128);
    if (!flat) {
        std::uint32_t s = 12345;
        auto next = [&] { s = s * 1664525u + 1013904223u; return s >> 8; };
        for (int k = 0; k < 60; ++k) {
            const int x0 = next() % w, y0 = next() % h, rw = 6 + next() % 30, rh = 6 + next() % 30;
            const unsigned char v = static_cast<unsigned char>(20 + next() % 220);
            for (int y = y0; y < y0 + rh && y < h; ++y)
                for (int x = x0; x < x0 + rw && x < w; ++x) px[static_cast<std::size_t>(y) * w + x] = v;
        }
    }
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    for (unsigned char v : px) { out.put(static_cast<char>(v)); out.put(static_cast<char>(v)); out.put(static_cast<char>(v)); }
}

template <typename F> static bool throws_runtime(F f) {
    try { f(); } catch (const std::runtime_error&) { return true; }
    return false;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("features_test_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string tex = (dir / "textured.ppm").string();
    const std::string flat = (dir / "flat.ppm").string();
    write_ppm(tex, false);
    write_ppm(flat, true);
    { std::ofstream(dir / "garbage.ppm", std::ios::binary) << "this is not an image at all"; }
    { std::ofstream(dir / "empty.ppm"); }

    SiftExtractor ex;

    // a textured image gives n x 128 non-negative descriptors, and extraction is deterministic
    const Matrix d = ex.extract(tex);
    assert(d.cols() == kSiftDim && d.rows() > 0);
    for (float v : d.raw()) assert(v >= 0.0f && v <= 255.0f);
    assert(ex.extract(tex).raw() == d.raw());

    // a flat image is valid input with zero keypoints: an empty matrix, not an error
    const Matrix none = ex.extract(flat);
    assert(none.rows() == 0 && none.cols() == kSiftDim);

    // missing, empty or corrupt files are input errors
    assert(throws_runtime([&] { ex.extract((dir / "missing.ppm").string()); }));
    assert(throws_runtime([&] { ex.extract((dir / "empty.ppm").string()); }));
    assert(throws_runtime([&] { ex.extract((dir / "garbage.ppm").string()); }));

    // the extractor can be moved, and the moved-to object still works
    SiftExtractor moved = std::move(ex);
    assert(moved.extract(tex).raw() == d.raw());

    // sampling the real descriptors gives at most S rows with the right dimension
    const Matrix s = sample_descriptors(d, 10, derive_seed(1, 0));
    assert(s.cols() == kSiftDim && s.rows() == (d.rows() < 10 ? d.rows() : 10));

    fs::remove_all(dir);
    return 0;
}