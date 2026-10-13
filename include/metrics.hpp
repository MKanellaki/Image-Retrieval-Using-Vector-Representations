#pragma once
#include <unordered_set>
#include <vector>
#include "types.hpp"

namespace image_search {

struct QueryMetrics {
    double recall5 = 0, recall10 = 0, ap10 = 0;
    std::size_t returned = 0;       // how many results the method actually gave
};

// ranked = returned rows, best first (at most 10 are used) 
// relevant = the 5 relevant rows
// Missing positions count as non-relevant
QueryMetrics evaluate_query(const std::vector<ImageIndex>& ranked, const std::unordered_set<ImageIndex>& relevant);

struct Summary {
    double mean_recall5 = 0, mean_recall10 = 0, map10 = 0;
    double avg_query_ms = 0;
    std::size_t queries = 0;
    std::size_t queries_with_fewer_than_10 = 0;   // to be reported in the experimental study
};

// Averages over any subset of queries, so the same code gives overall, illumination-only and viewpoint-only summaries
Summary summarize(const std::vector<QueryMetrics>& per_query, const std::vector<double>& query_ms);

}