// Tests for bow / bow_pipeline. They use a fake extractor instead of SIFT, so they need
// neither OpenCV nor the real datasets. Run with `make test`.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "bow.hpp"
#include "bow_pipeline.hpp"
#include "kmeans.hpp"
#include "seed.hpp"

using namespace image_search;
namespace fs = std::filesystem;

// ------------------------------------------------------------- tiny test harness

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

static bool same(const Matrix& a, const Matrix& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() && a.raw() == b.raw();   // bit-exact
}

// ------------------------------------------------------------- fakes

// Deterministic fake "SIFT": the descriptors depend only on the path. Paths listed in
// `empty` give zero descriptors; others give 5..24 rows of `dim` values in [0, 1).
struct FakeImages {
    std::set<std::string> empty;
    std::size_t dim = kSiftDim;

    Matrix extract(const std::string& path) const {
        Matrix m(0, dim);
        if (empty.count(path)) return m;
        std::mt19937 rng(static_cast<unsigned>(std::hash<std::string>{}(path)));
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        const std::size_t n = 5 + rng() % 20;
        m = Matrix(n, dim);
        for (float& v : m.raw()) v = u(rng);
        return m;
    }
    ExtractorFactory factory() const {
        return [this] { return ExtractFn([this](const std::string& p) { return extract(p); }); };
    }
};

static ExtractorFactory forbidden_factory() {
    return [] { return ExtractFn([](const std::string& p) -> Matrix {
        throw std::logic_error("extractor called on cache hit: " + p); }); };
}

static std::vector<ImageRef> refs(std::size_t n, const std::string& prefix = "img") {
    std::vector<ImageRef> out;
    for (std::size_t i = 0; i < n; ++i) out.push_back({prefix + std::to_string(i), prefix + std::to_string(i)});
    return out;
}

static Vocabulary random_vocab(std::size_t K, std::size_t dim, unsigned seed) {
    Matrix c(K, dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (float& v : c.raw()) v = u(rng);
    return Vocabulary(std::move(c));
}

// ------------------------------------------------------------- encoding

static void test_slide_example() {
    // K = 4 words, n = 8 descriptors: h = (4, 1, 3, 0) -> y = (0.5, 0.125, 0.375, 0).
    Matrix centers(4, 2);
    const float c[] = {0, 0, 10, 0, 0, 10, 10, 10};
    std::copy(c, c + 8, centers.raw().begin());
    const Vocabulary vocab(centers);

    Matrix d(0, 2);
    const float pts[][2] = {{0.1f, 0.2f}, {-0.3f, 0.0f}, {0.0f, 1.0f}, {1.0f, -1.0f},   // word 0
                            {9.0f, 0.5f},                                               // word 1
                            {0.5f, 9.5f}, {-1.0f, 11.0f}, {0.0f, 10.0f}};               // word 2
    for (const auto& p : pts) d.append_row(p);

    const std::vector<float> y = vocab.encode(d);
    CHECK(y.size() == 4);
    CHECK(y[0] == 0.5f && y[1] == 0.125f && y[2] == 0.375f && y[3] == 0.0f);
    CHECK(y[0] + y[1] + y[2] + y[3] == 1.0f);
}

static void test_histogram_counts_and_sum() {
    const std::size_t K = 7, dim = 16;
    const Vocabulary vocab = random_vocab(K, dim, 3);
    FakeImages fake;
    fake.dim = dim;
    for (int img = 0; img < 20; ++img) {
        const Matrix d = fake.extract("x" + std::to_string(img));
        const std::vector<float> y = vocab.encode(d);

        std::vector<int> expected(K, 0);   // brute-force nearest center
        for (std::size_t t = 0; t < d.rows(); ++t) {
            std::size_t best = 0;
            float best_d = INFINITY;
            for (std::size_t c = 0; c < K; ++c) {
                float s = 0;
                for (std::size_t j = 0; j < dim; ++j) {
                    const float diff = d.row(t)[j] - vocab.centers().row(c)[j];
                    s += diff * diff;
                }
                if (s < best_d) { best_d = s; best = c; }
            }
            ++expected[best];
        }
        double sum = 0;
        for (std::size_t c = 0; c < K; ++c) {
            sum += y[c];
            CHECK(y[c] >= 0.0f);
            CHECK(std::lround(y[c] * static_cast<double>(d.rows())) == expected[c]);
        }
        CHECK(std::fabs(sum - 1.0) < 1e-5);
    }
}

static void test_encode_rejects_bad_input() {
    const Vocabulary vocab = random_vocab(4, 8, 1);
    CHECK(throws<std::invalid_argument>([&] { vocab.encode(Matrix(0, 8)); }));
    CHECK(throws<std::invalid_argument>([&] { vocab.encode(Matrix(3, 9)); }));
    CHECK(throws<std::invalid_argument>([&] { Vocabulary().encode(Matrix(3, 8)); }));
}

// ------------------------------------------------------------- parallel steps

static void test_encode_images_thread_independent() {
    FakeImages fake;
    const Vocabulary vocab = random_vocab(10, kSiftDim, 5);
    const auto images = refs(37);
    const Matrix one = encode_images(images, vocab, fake.factory(), 1);
    CHECK(one.rows() == 37 && one.cols() == 10);
    for (unsigned t : {2u, 3u, 8u, 64u}) CHECK(same(one, encode_images(images, vocab, fake.factory(), t)));
    for (std::size_t i = 0; i < images.size(); ++i) {   // row i is image i
        const auto y = vocab.encode(fake.extract(images[i].path));
        CHECK(std::equal(y.begin(), y.end(), one.row(i)));
    }
}

static void test_encode_images_rejects_empty_image() {
    FakeImages fake;
    fake.empty = {"img4"};
    const Vocabulary vocab = random_vocab(3, kSiftDim, 1);
    CHECK(throws<std::runtime_error>([&] { encode_images(refs(9), vocab, fake.factory(), 1); }));
    CHECK(throws<std::runtime_error>([&] { encode_images(refs(9), vocab, fake.factory(), 4); }));
}

static void test_worker_exception_propagates() {
    const Vocabulary vocab = random_vocab(3, kSiftDim, 1);
    ExtractorFactory bad = [] { return ExtractFn([](const std::string& p) -> Matrix {
        throw std::runtime_error("cannot read image: " + p); }); };
    CHECK(throws<std::runtime_error>([&] { encode_images(refs(50), vocab, bad, 4); }));
}

static void test_training_sample() {
    FakeImages fake;
    const auto images = refs(30, "train");
    const std::size_t S = 12;
    const std::uint64_t seed = 42;

    Matrix expected(0, kSiftDim);   // per image: sample_descriptors with derive_seed(seed, i)
    std::size_t expected_rows = 0;
    for (std::size_t i = 0; i < images.size(); ++i) {
        const Matrix d = fake.extract(images[i].path);
        expected_rows += std::min(d.rows(), S);
        const Matrix s = sample_descriptors(d, S, derive_seed(seed, i));
        for (std::size_t r = 0; r < s.rows(); ++r) expected.append_row(s.row(r));
    }
    for (unsigned t : {1u, 2u, 5u}) {
        const Matrix got = collect_training_descriptors(images, S, seed, fake.factory(), t);
        CHECK(got.rows() == expected_rows);
        CHECK(same(got, expected));
    }
    CHECK(!same(collect_training_descriptors(images, S, seed + 1, fake.factory(), 1), expected));

    const Vocabulary vocab = train_vocabulary(expected, 8, seed);
    CHECK(vocab.size() == 8 && vocab.dim() == kSiftDim);
    CHECK(throws<std::invalid_argument>([&] { train_vocabulary(Matrix(5, kSiftDim), 8, seed); }));
}

// MIRFlickr rule: first `target` images with at least one descriptor, in order.
static void test_first_valid_skip_and_replace() {
    FakeImages fake;
    const auto candidates = refs(20, "im");
    fake.empty = {"im0", "im3", "im4", "im9", "im15"};
    const Vocabulary vocab = random_vocab(6, kSiftDim, 9);

    const std::vector<std::string> want = {"im1", "im2", "im5", "im6", "im7",
                                           "im8", "im10", "im11", "im12", "im13"};
    const ValidPrefix ref = encode_first_valid(candidates, 10, vocab, fake.factory(), 1, 1);
    CHECK(ref.ids == want);
    CHECK(ref.vectors.rows() == 10 && ref.vectors.cols() == 6);
    CHECK(ref.skipped == 4);   // im0, im3, im4, im9; im15 is never reached
    for (std::size_t i = 0; i < want.size(); ++i) {
        const auto y = vocab.encode(fake.extract(want[i]));
        CHECK(std::equal(y.begin(), y.end(), ref.vectors.row(i)));
    }

    // Same answer for any thread count and batch size.
    for (unsigned t : {1u, 2u, 4u})
        for (std::size_t b : {1u, 2u, 3u, 7u, 64u, 0u}) {
            const ValidPrefix got = encode_first_valid(candidates, 10, vocab, fake.factory(), t, b);
            CHECK(got.ids == want && got.skipped == ref.skipped && same(got.vectors, ref.vectors));
        }

    // Running out of files: every valid image, nothing more.
    const ValidPrefix all = encode_first_valid(candidates, 100, vocab, fake.factory(), 3, 4);
    CHECK(all.ids.size() == 15 && all.skipped == 5 && all.vectors.rows() == 15);

    // The first D rows of a longer run equal a run with target D (so any D is a prefix).
    const ValidPrefix five = encode_first_valid(candidates, 5, vocab, fake.factory(), 2, 0);
    CHECK(five.ids == std::vector<std::string>(want.begin(), want.begin() + 5));
    CHECK(std::equal(five.vectors.raw().begin(), five.vectors.raw().end(), ref.vectors.raw().begin()));

    const ValidPrefix none = encode_first_valid(candidates, 0, vocab, fake.factory(), 2, 0);
    CHECK(none.ids.empty() && none.vectors.rows() == 0 && none.skipped == 0);
}

// ------------------------------------------------------------- files

static fs::path temp_dir() {
    const fs::path dir = fs::temp_directory_path() / ("bow_test_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

static void test_vectors_roundtrip(const fs::path& dir) {
    Matrix m(13, 7);
    std::mt19937 rng(1);
    std::normal_distribution<float> n;
    for (float& v : m.raw()) v = n(rng);
    m.raw()[3] = -0.0f;
    m.raw()[4] = 1e-38f;
    const std::string p = (dir / "v.bin").string();
    save_vectors(p, m);
    CHECK(same(load_vectors(p), m));

    save_vectors(p, Matrix(0, 7));
    const Matrix e = load_vectors(p);
    CHECK(e.rows() == 0 && e.cols() == 7);

    const Vocabulary vocab = random_vocab(5, kSiftDim, 2);
    vocab.save((dir / "vocab.bin").string());
    CHECK(same(Vocabulary::load((dir / "vocab.bin").string()).centers(), vocab.centers()));

    // Truncated file and wrong file type are errors, not garbage.
    save_vectors(p, m);
    fs::resize_file(p, fs::file_size(p) - 5);
    CHECK(throws<std::runtime_error>([&] { load_vectors(p); }));
    CHECK(throws<std::runtime_error>([&] { load_vectors((dir / "vocab.bin").string()); }));
    CHECK(throws<std::runtime_error>([&] { load_vectors((dir / "missing.bin").string()); }));
}

static EncodedImages fake_set(std::size_t n, std::size_t K, const std::string& prefix, unsigned seed) {
    EncodedImages e;
    e.vectors = Matrix(n, K);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u;
    for (float& v : e.vectors.raw()) v = u(rng);
    for (std::size_t i = 0; i < n; ++i) e.ids.push_back(prefix + std::to_string(i));
    return e;
}

static void test_cache_roundtrip(const fs::path& dir) {
    const std::size_t K = 6;
    BowData d;
    d.centers = fake_set(K, kSiftDim, "c", 1).vectors;
    d.validation_queries = fake_set(2, K, "vq", 2);
    d.validation_db = fake_set(10, K, "vd", 3);
    d.test_queries = fake_set(3, K, "tq", 4);
    d.test_db = fake_set(15, K, "td", 5);
    d.mirflickr = fake_set(7, K, "im", 6);
    d.mir_skipped = 2;
    const CacheKey key{K, 500, 25000, 1, 0xabcdefULL};

    const std::string p = (dir / "sub" / "cache.bin").string();   // parent is created
    save_bow_cache(p, key, d);
    CHECK(!fs::exists(p + ".tmp"));
    const BowData r = load_bow_cache(p, key);
    CHECK(same(r.centers, d.centers));
    for (auto [a, b] : {std::pair{&r.validation_queries, &d.validation_queries},
                        {&r.validation_db, &d.validation_db}, {&r.test_queries, &d.test_queries},
                        {&r.test_db, &d.test_db}, {&r.mirflickr, &d.mirflickr}})
        CHECK(a->ids == b->ids && same(a->vectors, b->vectors));
    CHECK(r.mir_skipped == 2);

    for (CacheKey other : {CacheKey{K + 1, 500, 25000, 1, 0xabcdefULL}, CacheKey{K, 400, 25000, 1, 0xabcdefULL},
                           CacheKey{K, 500, 25000, 2, 0xabcdefULL}, CacheKey{K, 500, 25000, 1, 0x1ULL}})
        CHECK(throws<std::runtime_error>([&] { load_bow_cache(p, other); }));

    fs::resize_file(p, fs::file_size(p) - 1);
    CHECK(throws<std::runtime_error>([&] { load_bow_cache(p, key); }));

    // make_database: HPatches rows, then the first D MIRFlickr rows.
    std::vector<ImageRef> db_refs;
    const Matrix db = make_database(d, Split::Test, 4, &db_refs);
    CHECK(db.rows() == 19 && db.cols() == K && db_refs.size() == 19);
    CHECK(db_refs[0].id == "td0" && db_refs[15].id == "im0" && db_refs[18].id == "im3");
    CHECK(std::equal(d.mirflickr.vectors.row(3), d.mirflickr.vectors.row(3) + K, db.row(18)));
    CHECK(make_database(d, Split::Validation, 0, nullptr).rows() == 10);
    CHECK(throws<std::runtime_error>([&] { make_database(d, Split::Test, 8, nullptr); }));
    CHECK(throws<std::invalid_argument>([&] { make_database(d, Split::Train, 1, nullptr); }));

    // shuffle_database: a seeded permutation that keeps every row with its id.
    Matrix sh = db;
    std::vector<ImageRef> sh_refs = db_refs;
    shuffle_database(sh, &sh_refs, 3);
    CHECK(sh.rows() == db.rows() && sh.cols() == K && sh_refs.size() == db_refs.size());
    std::size_t moved = 0;
    std::set<std::string> seen;
    for (std::size_t i = 0; i < sh.rows(); ++i) {
        std::size_t j = 0;
        while (j < db_refs.size() && db_refs[j].id != sh_refs[i].id) ++j;
        CHECK(j < db_refs.size() && std::equal(db.row(j), db.row(j) + K, sh.row(i)));
        seen.insert(sh_refs[i].id);
        if (j != i) ++moved;
    }
    CHECK(seen.size() == db_refs.size());   // a permutation: nothing lost or duplicated
    CHECK(moved > 0);

    Matrix again = db;                      // same seed, same order
    std::vector<ImageRef> again_refs = db_refs;
    shuffle_database(again, &again_refs, 3);
    CHECK(same(again, sh));
    for (std::size_t i = 0; i < again_refs.size(); ++i) CHECK(again_refs[i].id == sh_refs[i].id);

    Matrix other = db;                      // a different seed gives a different order
    shuffle_database(other, nullptr, 4);
    CHECK(!same(other, sh));

    std::vector<ImageRef> bad_refs(2);
    CHECK(throws<std::invalid_argument>([&] { shuffle_database(again, &bad_refs, 3); }));
    Matrix empty(0, K);
    std::vector<ImageRef> no_refs;
    shuffle_database(empty, &no_refs, 3);
    CHECK(empty.rows() == 0 && no_refs.empty());
}

// End to end on a tiny fake dataset on disk: build, cache hit, cache key changes.
static void test_pipeline(const fs::path& dir) {
    const fs::path hp = dir / "hp", mir = dir / "mir";
    const std::vector<std::pair<std::string, std::string>> split = {
        {"i_a", "train"}, {"v_b", "train"}, {"i_c", "validation"}, {"v_d", "validation"},
        {"i_e", "test"}, {"v_f", "test"}};
    std::ofstream sf(dir / "split.txt");
    for (const auto& [name, s] : split) {
        sf << name << ' ' << s << '\n';
        fs::create_directories(hp / name);
        for (int n = 1; n <= 6; ++n) std::ofstream(hp / name / (std::to_string(n) + ".ppm")) << "x";
    }
    sf.close();
    fs::create_directories(mir);
    for (int n : {1, 2, 3, 10, 11, 12, 20}) std::ofstream(mir / ("im" + std::to_string(n) + ".jpg")) << "x";

    FakeImages fake;
    fake.empty = {(mir / "im2.jpg").string(), (mir / "im10.jpg").string()};

    BowConfig cfg;
    cfg.hp_dir = hp.string();
    cfg.mir_dir = mir.string();
    cfg.split_file = (dir / "split.txt").string();
    cfg.cache_dir = (dir / "cache").string();
    cfg.K = 4;
    cfg.S = 6;
    cfg.seed = 7;
    cfg.mir_target = 4;
    cfg.threads = 3;
    cfg.verbose = false;

    const BowData built = load_or_build_bow_data(cfg, fake.factory());
    const std::string path = cache_path(cfg, cache_key(cfg));
    CHECK(fs::exists(path));
    CHECK(built.centers.rows() == 4 && built.centers.cols() == kSiftDim);
    CHECK(built.validation_queries.ids == (std::vector<std::string>{"i_c/1", "v_d/1"}));
    CHECK(built.test_db.ids.size() == 10 && built.test_db.ids[5] == "v_f/2");
    CHECK(built.mirflickr.ids == (std::vector<std::string>{"im1.jpg", "im3.jpg", "im11.jpg", "im12.jpg"}));
    CHECK(built.mir_skipped == 2);

    // Same result with one thread, built from scratch.
    BowConfig one = cfg;
    one.threads = 1;
    const BowData serial = build_bow_data(one, fake.factory());
    CHECK(same(serial.centers, built.centers) && same(serial.mirflickr.vectors, built.mirflickr.vectors) &&
          same(serial.test_queries.vectors, built.test_queries.vectors));

    // Second run: served from the cache, extractor never called.
    const BowData cached = load_or_build_bow_data(cfg, forbidden_factory());
    CHECK(same(cached.centers, built.centers) && same(cached.validation_db.vectors, built.validation_db.vectors));

    // The relevant rows of a query line up with make_database.
    std::vector<ImageRef> db_refs;
    const Matrix db = make_database(cached, Split::Validation, 3, &db_refs);
    CHECK(db.rows() == 13);
    const auto rel = relevant_rows(db_refs, "v_d");
    CHECK(rel == (std::unordered_set<ImageIndex>{5, 6, 7, 8, 9}));

    // After the shuffle main does, relevant_rows still finds the same five images.
    Matrix sh = db;
    std::vector<ImageRef> sh_refs = db_refs;
    shuffle_database(sh, &sh_refs, cfg.seed);
    const auto sh_rel = relevant_rows(sh_refs, "v_d");
    CHECK(sh_rel.size() == 5);
    for (ImageIndex row : sh_rel) {
        CHECK(sh_refs[row].id.rfind("v_d/", 0) == 0);
        std::size_t j = 0;
        while (j < db_refs.size() && db_refs[j].id != sh_refs[row].id) ++j;
        CHECK(j < db_refs.size() && std::equal(db.row(j), db.row(j) + db.cols(), sh.row(row)));
    }

    // A different seed or split file gives a different cache file.
    BowConfig s2 = cfg;
    s2.seed = 8;
    CHECK(cache_path(s2, cache_key(s2)) != path);
    std::ofstream(dir / "split.txt", std::ios::app) << '\n';
    CHECK(cache_path(cfg, cache_key(cfg)) != path);
}

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"slide example h=(4,1,3,0)", test_slide_example},
        {"histogram counts and sum", test_histogram_counts_and_sum},
        {"encode rejects bad input", test_encode_rejects_bad_input},
        {"encode_images thread independent", test_encode_images_thread_independent},
        {"encode_images rejects empty image", test_encode_images_rejects_empty_image},
        {"worker exception propagates", test_worker_exception_propagates},
        {"training sample", test_training_sample},
        {"MIRFlickr skip-and-replace", test_first_valid_skip_and_replace},
    };
    for (const auto& [name, fn] : tests) {
        std::printf("%s\n", name);
        fn();
    }

    const fs::path dir = temp_dir();
    std::printf("vectors file roundtrip\n");
    test_vectors_roundtrip(dir);
    std::printf("cache roundtrip\n");
    test_cache_roundtrip(dir);
    std::printf("pipeline end to end\n");
    test_pipeline(dir);
    fs::remove_all(dir);

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
