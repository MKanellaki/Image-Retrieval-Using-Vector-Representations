#include "output.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace image_search {

namespace fs = std::filesystem;

namespace {

// Builds the whole file in memory
// Throws (std::out_of_range) before anything reaches a stream or the file system, so callers never see partial output
// Using a local buffer also leaves the caller's stream flags untouched
std::string format_output(const std::string& method, const std::vector<QueryResult>& results, const std::vector<ImageRef>& db, Summary& summary) {
    std::ostringstream buf;
    buf << std::fixed << std::setprecision(kOutputPrecision);

    std::vector<QueryMetrics> per_query;
    std::vector<double> query_ms;
    per_query.reserve(results.size());
    query_ms.reserve(results.size());

    buf << method << '\n';

    for(const QueryResult& r : results) {
        buf << "Query: " << r.sequence << '\n';

        for(std::size_t i = 0; i < kOutputRanks; ++i) {
            buf << "Rank-" << (i + 1) << ": ";
            if(i < r.ranked.size()) {
                const ImageIndex row = r.ranked[i];
                if(row >= db.size())
                    throw std::out_of_range("write_output: row " + std::to_string(row) +
                                            " for query '" + r.sequence +
                                            "' is outside the database (size " +
                                            std::to_string(db.size()) + ")");
                buf << db[row].id;
            } else {
                buf << kMissingRank;
            }
            buf << '\n';
        }

        buf << "Recall@5: " << r.metrics.recall5 << '\n'
            << "Recall@10: " << r.metrics.recall10 << '\n'
            << "AP@10: " << r.metrics.ap10 << '\n';

        per_query.push_back(r.metrics);
        query_ms.push_back(r.query_ms);
    }

    summary = summarize(per_query, query_ms);

    buf << "Mean Recall@5: " << summary.mean_recall5 << '\n'
        << "Mean Recall@10: " << summary.mean_recall10 << '\n'
        << "mAP@10: " << summary.map10 << '\n'
        << "Average query time: " << summary.avg_query_ms << '\n';

    return buf.str();
}

}  // namespace

Summary write_output(std::ostream& out, const std::string& method, const std::vector<QueryResult>& results, const std::vector<ImageRef>& db) {
    Summary s;
    out << format_output(method, results, db, s);
    return s;
}

Summary write_output_file(const std::string& path, const std::string& method, const std::vector<QueryResult>& results, const std::vector<ImageRef>& db) {
    // 1. Validate and format everything first 
    // a bad row throws here, before the file system is touched, so any existing file survives
    Summary s;
    const std::string text = format_output(method, results, db, s);

    // 2. Write to a sibling temp file, so the rename below stays on one file system and therefore replaces `path` in a single step
    const fs::path target(path);
    fs::path tmp = target;
    tmp += ".tmp";

    {
        std::ofstream file(tmp, std::ios::out | std::ios::trunc | std::ios::binary);
        if(!file)
            throw std::runtime_error("cannot open output file: " + tmp.string());
        file << text;
        file.close();
        if(!file) {
            std::error_code ignore;
            fs::remove(tmp, ignore);
            throw std::runtime_error("failed writing output file: " + tmp.string());
        }
    }

    // 3. Replace the old result
    // If this fails, the old file is still intact
    std::error_code ec;
    fs::rename(tmp, target, ec);
    if(ec) {
        std::error_code ignore;
        fs::remove(tmp, ignore);
        throw std::runtime_error("cannot replace output file " + path + ": " + ec.message());
    }
    return s;
}

}  // namespace image_search