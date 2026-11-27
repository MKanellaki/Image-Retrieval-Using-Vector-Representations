// ./search: builds (or loads) the BoW vectors, builds the chosen index over the m query set's database, runs every query and writes the results file
#include <chrono>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.hpp"
#include "bow_pipeline.hpp"
#include "dataset.hpp"
#include "features.hpp"
#include "metrics.hpp"
#include "output.hpp"
#include "search_index.hpp"

using namespace image_search;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
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
    Matrix db = make_database(data, args.query_set, D, &refs);
    // Seeded row shuffle (refs move with the rows), so the relevant HPatches images are
    // spread among the distractors instead of all sitting in rows 0..114. Without it,
    // any method that stops after a fixed number of candidates sees them first.
    shuffle_database(db, &refs, args.seed);
    const EncodedImages& queries = queries_for(data, args.query_set);
    if(queries.vectors.cols() != db.cols())
        throw std::runtime_error("query vectors and database vectors have different dimensions");
    std::fprintf(stderr, "[search] database: %zu rows (%zu HPatches + %zu MIRFlickr, shuffled with seed %llu), %zu queries\n", db.rows(), db.rows() - D, D, static_cast<unsigned long long>(args.seed), queries.ids.size());

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
        std::vector<ImageIndex> ranked = search(q, kOutputRanks);
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