#include "bow_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "binary_io.hpp"

namespace fs = std::filesystem;

namespace image_search {

namespace {

constexpr char kCacheMagic[9]    = "IMSBOWC1";
constexpr char kCacheEndMagic[9] = "IMSBOWE1";
// Bump whenever the meaning of the vectors changes (encoding, SIFT settings, file layout), so old caches are rejected instead of reused
constexpr std::uint64_t kCacheVersion = 1;

class Log {
public:
    explicit Log(bool on) : on_(on), start_(std::chrono::steady_clock::now()) {}
    template <class... Args>
    void operator()(const char* fmt, Args... args) const {
        if (!on_) return;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        std::fprintf(stderr, "[bow %7.1fs] ", s);
        if constexpr (sizeof...(Args) == 0) std::fputs(fmt, stderr);
        else std::fprintf(stderr, fmt, args...);
        std::fputc('\n', stderr);
    }

private:
    bool on_;
    std::chrono::steady_clock::time_point start_;
};

std::uint64_t fnv1a64_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open split file: " + path);
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::istreambuf_iterator<char> it(in), end; it != end; ++it) {
        h ^= static_cast<unsigned char>(*it);
        h *= 0x100000001b3ULL;
    }
    return h;
}

// All six images of every training sequence, sequence by sequence (seq/1, ..., seq/6)
std::vector<ImageRef> training_images(const std::string& hp_dir, const std::vector<Sequence>& seqs) {
    const auto first = make_queries(hp_dir, seqs, Split::Train);
    const auto rest = hpatches_database(hp_dir, seqs, Split::Train);
    std::vector<ImageRef> out;
    out.reserve(first.size() + rest.size());
    const std::size_t per_seq = first.empty() ? 0 : rest.size() / first.size();
    for (std::size_t j = 0; j < first.size(); ++j) {
        out.push_back(first[j].image);
        for (std::size_t t = 0; t < per_seq; ++t) out.push_back(rest[j * per_seq + t]);
    }
    return out;
}

std::vector<ImageRef> query_images(const std::vector<Query>& queries) {
    std::vector<ImageRef> out;
    out.reserve(queries.size());
    for (const auto& q : queries) out.push_back(q.image);
    return out;
}

EncodedImages encode_set(const std::vector<ImageRef>& images, const Vocabulary& vocab,
                         const ExtractorFactory& factory, unsigned threads) {
    EncodedImages e;
    e.ids.reserve(images.size());
    for (const auto& im : images) e.ids.push_back(im.id);
    e.vectors = encode_images(images, vocab, factory, threads);
    return e;
}

std::vector<std::string> ids_of(const std::vector<ImageRef>& images) {
    std::vector<std::string> ids;
    ids.reserve(images.size());
    for (const auto& im : images) ids.push_back(im.id);
    return ids;
}

// The cached HPatches ids must be exactly what the current split file and hp_dir produce
void check_hpatches_ids(const BowConfig& config, const BowData& data) {
    const auto seqs = read_split(config.split_file);
    auto same = [](const std::vector<std::string>& a, const std::vector<std::string>& b, const char* what) {
        if (a != b) throw std::runtime_error(std::string("cached ") + what + " do not match the split file");
    };
    for (Split s : {Split::Validation, Split::Test}) {
        const bool val = s == Split::Validation;
        same(ids_of(query_images(make_queries(config.hp_dir, seqs, s))),
             (val ? data.validation_queries : data.test_queries).ids, "queries");
        same(ids_of(hpatches_database(config.hp_dir, seqs, s)),
             (val ? data.validation_db : data.test_db).ids, "database images");
    }
}

void write_set(binio::Writer& w, const EncodedImages& e) {
    w.u64(e.ids.size());
    for (const auto& id : e.ids) w.str(id);
    w.matrix(e.vectors);
}

EncodedImages read_set(binio::Reader& r, std::size_t K) {
    EncodedImages e;
    const std::uint64_t n = r.u64();
    for (std::uint64_t i = 0; i < n; ++i) e.ids.push_back(r.str());
    e.vectors = r.matrix();
    if (e.vectors.rows() != e.ids.size() || e.vectors.cols() != K)
        r.fail("image set has inconsistent shape");
    return e;
}

}  // namespace

CacheKey cache_key(const BowConfig& config) {
    CacheKey k;
    k.K = config.K;
    k.S = config.S;
    k.seed = config.seed;
    k.mir_target = config.mir_target;
    k.split_hash = fnv1a64_file(config.split_file);
    return k;
}

std::string cache_path(const BowConfig& config, const CacheKey& key) {
    char name[160];
    std::snprintf(name, sizeof name, "bow_K%zu_S%zu_seed%llu_mir%zu_split%016llx.bin", key.K, key.S,
                  static_cast<unsigned long long>(key.seed), key.mir_target,
                  static_cast<unsigned long long>(key.split_hash));
    return (fs::path(config.cache_dir) / name).string();
}

BowData build_bow_data(const BowConfig& config, const ExtractorFactory& factory) {
    const Log log(config.verbose);
    const unsigned threads = resolve_threads(config.threads);
    const auto seqs = read_split(config.split_file);
    BowData data;

    // 1. Vocabulary from the training split only
    const auto train = training_images(config.hp_dir, seqs);
    log("step 1/3: extracting %zu training images (S = %zu, %u threads)", train.size(), config.S, threads);
    Matrix sampled = collect_training_descriptors(train, config.S, config.seed, factory, threads);
    log("          k-means: %zu descriptors -> K = %zu", sampled.rows(), config.K);
    const Vocabulary vocab = train_vocabulary(sampled, config.K, config.seed);
    sampled = Matrix();
    data.centers = vocab.centers();

    // 2. HPatches validation/test queries and database images
    log("step 2/3: encoding HPatches validation/test images");
    data.validation_queries = encode_set(query_images(make_queries(config.hp_dir, seqs, Split::Validation)),
                                         vocab, factory, threads);
    data.validation_db = encode_set(hpatches_database(config.hp_dir, seqs, Split::Validation),
                                    vocab, factory, threads);
    data.test_queries = encode_set(query_images(make_queries(config.hp_dir, seqs, Split::Test)),
                                   vocab, factory, threads);
    data.test_db = encode_set(hpatches_database(config.hp_dir, seqs, Split::Test), vocab, factory, threads);

    // 3. MIRFlickr distractors, in number order, skipping images without descriptors
    const auto mir = mirflickr_ordered(config.mir_dir);
    log("step 3/3: encoding MIRFlickr until %zu valid images (%zu files)", config.mir_target, mir.size());
    ValidPrefix prefix = encode_first_valid(mir, config.mir_target, vocab, factory, threads);
    data.mirflickr.ids = std::move(prefix.ids);
    data.mirflickr.vectors = std::move(prefix.vectors);
    data.mir_skipped = prefix.skipped;
    log("          %zu valid, %zu skipped (no descriptors)", data.mirflickr.ids.size(), data.mir_skipped);
    if (data.mirflickr.ids.size() < config.mir_target)
        log("          warning: only %zu valid MIRFlickr images; larger D will be rejected",
            data.mirflickr.ids.size());
    return data;
}

void save_bow_cache(const std::string& path, const CacheKey& key, const BowData& data) {
    const fs::path target(path);
    if (target.has_parent_path()) fs::create_directories(target.parent_path());
    const std::string tmp = path + ".tmp";
    {
        binio::Writer w(tmp);
        w.magic(kCacheMagic);
        w.u64(kCacheVersion);
        w.u64(key.K);
        w.u64(key.S);
        w.u64(key.mir_target);
        w.u64(key.seed);
        w.u64(key.split_hash);
        w.matrix(data.centers);
        write_set(w, data.validation_queries);
        write_set(w, data.validation_db);
        write_set(w, data.test_queries);
        write_set(w, data.test_db);
        write_set(w, data.mirflickr);
        w.u64(data.mir_skipped);
        w.magic(kCacheEndMagic);
        w.close();
    }
    fs::rename(tmp, target);
}

BowData load_bow_cache(const std::string& path, const CacheKey& expected) {
    binio::Reader r(path);
    r.expect_magic(kCacheMagic);
    if (r.u64() != kCacheVersion) r.fail("cache written by a different version");
    CacheKey got;
    got.K = r.u64();
    got.S = r.u64();
    got.mir_target = r.u64();
    got.seed = r.u64();
    got.split_hash = r.u64();
    if (got.K != expected.K || got.S != expected.S || got.mir_target != expected.mir_target ||
        got.seed != expected.seed || got.split_hash != expected.split_hash)
        r.fail("cache was built with different parameters");

    BowData data;
    data.centers = r.matrix();
    if (data.centers.rows() != expected.K) r.fail("vocabulary has the wrong size");
    data.validation_queries = read_set(r, expected.K);
    data.validation_db = read_set(r, expected.K);
    data.test_queries = read_set(r, expected.K);
    data.test_db = read_set(r, expected.K);
    data.mirflickr = read_set(r, expected.K);
    data.mir_skipped = r.u64();
    r.expect_magic(kCacheEndMagic);
    r.expect_end();
    return data;
}

BowData load_or_build_bow_data(const BowConfig& config, const ExtractorFactory& factory) {
    const Log log(config.verbose);
    const CacheKey key = cache_key(config);
    const std::string path = cache_path(config, key);

    if (fs::exists(path)) {
        try {
            BowData data = load_bow_cache(path, key);
            check_hpatches_ids(config, data);
            log("loaded cached vectors from %s", path.c_str());
            return data;
        } catch (const std::exception& e) {
            log("cache %s unusable (%s); rebuilding", path.c_str(), e.what());
        }
    }

    BowData data = build_bow_data(config, factory);
    save_bow_cache(path, key, data);
    log("saved vectors to %s", path.c_str());
    return data;
}

const EncodedImages& queries_for(const BowData& data, Split which) {
    if (which == Split::Validation) return data.validation_queries;
    if (which == Split::Test) return data.test_queries;
    throw std::invalid_argument("queries come from the validation or test split only");
}

Matrix make_database(const BowData& data, Split which, std::size_t D, std::vector<ImageRef>* refs) {
    if (which != Split::Validation && which != Split::Test)
        throw std::invalid_argument("the database is built for the validation or test split only");
    const EncodedImages& hp = which == Split::Validation ? data.validation_db : data.test_db;
    const EncodedImages& mir = data.mirflickr;
    if (D > mir.ids.size())
        throw std::runtime_error("D = " + std::to_string(D) + " but only " +
                                 std::to_string(mir.ids.size()) + " valid MIRFlickr images were encoded");

    const std::size_t K = hp.vectors.cols();
    Matrix db(hp.ids.size() + D, K);
    std::copy(hp.vectors.raw().begin(), hp.vectors.raw().end(), db.raw().begin());
    std::copy_n(mir.vectors.raw().begin(), D * K, db.raw().begin() + static_cast<std::ptrdiff_t>(hp.vectors.raw().size()));

    if (refs) {
        refs->clear();
        refs->reserve(db.rows());
        for (const auto& id : hp.ids) refs->push_back({id, ""});
        for (std::size_t i = 0; i < D; ++i) refs->push_back({mir.ids[i], ""});
    }
    return db;
}

}  // namespace image_search
