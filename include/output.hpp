#pragma once
// Writes the results file in the format:
//
//   <METHOD NAME>
//   Query: <sequence_id>
//   Rank-1: <image_id>
//   ...
//   Rank-10: <image_id>
//   Recall@5: <double>
//   Recall@10: <double>
//   AP@10: <double>
//   ...                         (one block per query)
//   Mean Recall@5: <double>
//   Mean Recall@10: <double>
//   mAP@10: <double>
//   Average query time: <double>      (milliseconds)
//
// image_id is the ImageRef::id of the database row: "v_bird/2" for HPatches,
// "im123.jpg" for MIRFlickr
// When a method returns fewer than 10 results, the missing Rank lines are still written, with kMissingRank as the id,
// so every query block has the same number of lines
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "dataset.hpp"
#include "metrics.hpp"
#include "types.hpp"

namespace image_search {

inline constexpr std::size_t kOutputRanks = 10;
inline constexpr const char* kMissingRank = "-";
inline constexpr int kOutputPrecision = 6;    // digits after the decimal point

struct QueryResult {
    std::string sequence;               // query sequence id, e.g. "v_bird"
    std::vector<ImageIndex> ranked;     // database rows returned, best first
    QueryMetrics metrics;               // from evaluate_query(ranked, relevant)
    double query_ms = 0;                // search time for this query
};

// Writes the whole file to "out" and returns the summary it printed
// (so the caller can also report queries_with_fewer_than_10)
// "db" maps a row index to its ImageRef, every row in "ranked" must be < db.size(),
// otherwise std::out_of_range is thrown and nothing is written
// The caller's stream formatting flags are left untouched
Summary write_output(std::ostream& out, const std::string& method,
                     const std::vector<QueryResult>& results,
                     const std::vector<ImageRef>& db);

// Same, to a file
// The text is fully built and validated first, then written to `path + ".tmp"` and renamed over "path"
// a failure (bad row, I/O error) never destroys a previous good result
// Throws std::out_of_range for a bad row
// (the file system is not touched) and std::runtime_error for I/O failures
// (the temp file is removed and "path" keeps its old contents)
Summary write_output_file(const std::string& path, const std::string& method,
                          const std::vector<QueryResult>& results,
                          const std::vector<ImageRef>& db);

}  // namespace image_search