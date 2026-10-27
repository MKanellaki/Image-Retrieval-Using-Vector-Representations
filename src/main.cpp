// ./search: builds (or loads) the BoW vectors, builds the chosen index over the m query set's database, runs every query and writes the results file
#include <chrono>
#include <climits>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.hpp"
#include "bow_pipeline.hpp"
#include "dataset.hpp"
#include "exact.hpp"
#include "features.hpp"
#include "hypercube.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"
#include "lsh.hpp"
#include "metrics.hpp"
#include "output.hpp"

using namespace image_search;

namespace {

using Clock = std::chrono::steady_clock;

// The one way main calls every method: a query vector in, ranked database rows out
using SearchFn = std::function<std::vector<ImageIndex>(const float*)>;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int to_int(std::size_t v) { return v > static_cast<std::size_t>(INT_MAX) ? INT_MAX : static_cast<int>(v); }

// Builds the chosen index and wraps its query() in a SearchFn
// The index is owned by the lambda (shared_ptr, because std::function needs a copyable callable)
// Every index keeps a pointer to `db`, so `db` must outlive the result
SearchFn make_search(const Args& args, const Matrix& db) {
    const std::size_t n = db.rows();
    switch(args.method) {
        case Method::Exact: {
            auto ix = std::make_shared<const ExactIndex>(db);
            return [ix](const float* q) { return ix->query(q, kOutputRanks); };
        }
        case Method::LSH: {
            std::fprintf(stderr, "[search] LSH: k = %d, L = %d, w = %g\n", args.lsh.k, args.lsh.L, args.lsh.w);
            auto ix = std::make_shared<const LSHIndex>(db, args.lsh);
            return [ix](const float* q) { return ix->query(q, kOutputRanks); };
        }
        case Method::Hypercube: {
            std::fprintf(stderr, "[search] Hypercube: kproj = %d, M = %d, probes = %d, w = %g\n", args.hypercube.k, args.hypercube.M, args.hypercube.probes, args.hypercube.w);
            auto ix = std::make_shared<const HypercubeIndex>(db, args.hypercube);
            return [ix](const float* q) { return ix->query(q, kOutputRanks); };
        }
        case Method::IVFFlat: {
            IVFFlatParams p;
            p.k = to_int(resolve_kclusters(args.ivfflat.kclusters, n));
            p.nprobe = to_int(args.ivfflat.nprobe);
            p.seed = args.seed;
            std::fprintf(stderr, "[search] IVFFlat: kclusters = %d, nprobe = %d\n", p.k, p.nprobe);
            auto ix = std::make_shared<const IVFFlatIndex>(db, p);
            return [ix](const float* q) { return ix->query(q, kOutputRanks); };
        }
        case Method::IVFPQ: {
            IVFPQParams p;
            p.k = to_int(resolve_kclusters(args.ivfpq.kclusters, n));
            p.nprobe = to_int(args.ivfpq.nprobe);
            p.m = to_int(args.ivfpq.M);
            p.k_sub = 1 << args.ivfpq.nbits;     // nbits <= 16, checked by parse_args
            p.seed = args.seed;
            std::fprintf(stderr, "[search] IVFPQ: kclusters = %d, nprobe = %d, M = %d, nbits = %zu\n", p.k, p.nprobe, p.m, args.ivfpq.nbits);
            auto ix = std::make_shared<const IVFPQIndex>(db, p);
            return [ix](const float* q) { return ix->query(q, kOutputRanks); };
        }
    }
    throw std::logic_error("unknown method");
}

// "v_bird/1" -> "v_bird"
std::string sequence_of(const std::string& query_id) {
    const auto slash = query_id.find('/');
    if(slash == std::string::npos || slash == 0) {
        throw std::runtime_error("query id '" + query_id + "' is not <sequence>/<n>");
    }
    return query_id.substr(0, slash);
}

Category category_of(const std::string& sequence) {
    if(sequence.rfind("i_", 0) == 0) return Category::Illumination;
    if(sequence.rfind("v_", 0) == 0) return Category::Viewpoint;
    throw std::runtime_error("sequence '" + sequence + "' must start with 'i_' or 'v_'");
}

void print_summary(const char* label, const Summary& s) {
    std::fprintf(stderr,
                "[search]   %-13s %3zu queries  R@5 %.4f  R@10 %.4f  mAP@10 %.4f  "
                "avg %.4f ms  (<10 results: %zu)\n",
                label, s.queries, s.mean_recall5, s.mean_recall10, s.map10, s.avg_query_ms, s.queries_with_fewer_than_10);
}

Summary summarize_category(const std::vector<QueryResult>& results, Category which) {
    std::vector<QueryMetrics> metrics;
    std::vector<double> times;
    for(const auto& r : results) {
        if(category_of(r.sequence) != which) continue;
        metrics.push_back(r.metrics);
        times.push_back(r.query_ms);
    }
    return summarize(metrics, times);
}

int run(const Args& args) {
    // 1. Image vectors (cached; only K, S, seed and the split file change them)
    BowConfig config;
    config.threads = 4;   // aoid crashes due to memory
    config.hp_dir = args.hp_dir;
    config.mir_dir = args.mir_dir;
    config.split_file = args.split_file;
    config.K = args.K;
    config.S = args.S;
    config.seed = args.seed;
    const BowData data = load_or_build_bow_data(config, sift_extractor_factory());

    // 2. Database: the split's HPatches rows + the first D MIRFlickr rows
    std::size_t D = args.D;
    const std::size_t available = data.mirflickr.ids.size();
    if(D > available) {
        std::fprintf(stderr,
                    "[search] warning: -D %zu requested but only %zu valid MIRFlickr images exist "
                    "(%zu skipped for having no SIFT descriptors); using D = %zu\n",
                    D, available, data.mir_skipped, available);
        D = available;
    }
    std::vector<ImageRef> refs;
    const Matrix db = make_database(data, args.query_set, D, &refs);
    const EncodedImages& queries = queries_for(data, args.query_set);
    if(queries.vectors.cols() != db.cols())
        throw std::runtime_error("query vectors and database vectors have different dimensions");
    std::fprintf(stderr, "[search] database: %zu rows (%zu HPatches + %zu MIRFlickr), %zu queries\n", db.rows(), db.rows() - D, D, queries.ids.size());

    // 3. Index (build time is reported but is not part of the query time)
    const auto build_start = Clock::now();
    const SearchFn search = make_search(args, db);
    std::fprintf(stderr, "[search] %s index built in %.1f ms\n", method_name(args.method), ms_since(build_start));

    // 4. Queries: only the query() call is timed
    std::vector<QueryResult> results;
    results.reserve(queries.ids.size());
    for(std::size_t i = 0; i < queries.ids.size(); ++i) {
        const float* q = queries.vectors.row(i);

        const auto t0 = Clock::now();
        std::vector<ImageIndex> ranked = search(q);
        const double ms = ms_since(t0);

        QueryResult r;
        r.sequence = sequence_of(queries.ids[i]);
        r.ranked = std::move(ranked);
        r.query_ms = ms;
        r.metrics = evaluate_query(r.ranked, relevant_rows(refs, r.sequence));
        results.push_back(std::move(r));
    }

    // 5. Output file, then the per-category breakdown for the report on stderr
    const Summary all = write_output_file(args.output, method_name(args.method), results, refs);
    std::fprintf(stderr, "[search] wrote %s (set = %s, D = %zu)\n", args.output.c_str(), args.query_set == Split::Validation ? "validation" : "test", D);
    print_summary("all", all);
    print_summary("illumination", summarize_category(results, Category::Illumination));
    print_summary("viewpoint", summarize_category(results, Category::Viewpoint));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string program = argc > 0 ? argv[0] : "search";

    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const ArgsError& e) {
        std::cerr << program << ": " << e.what() << "\n\n" << usage(program);
        return 2;
    }
    if (args.help) {
        std::cout << usage(program);
        return 0;
    }

    try {
        return run(args);
    } catch (const ArgsError& e) {          // e.g. -kclusters larger than the database
        std::cerr << program << ": " << e.what() << '\n';
        return 2;
    } catch (const std::exception& e) {
        std::cerr << program << ": error: " << e.what() << '\n';
        return 1;
    }
}