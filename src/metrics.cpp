
#include "metrics.hpp"

#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <vector>

namespace image_search {

QueryMetrics evaluate_query(const std::vector<ImageIndex>& ranked, const std::unordered_set<ImageIndex>& relevant) {
    QueryMetrics metrics;

    // Number of results actually returned by the search method
    metrics.returned = ranked.size();

    // Nothing to measure
    if(relevant.empty()) {
        return metrics;
    }
    const double num_relevant = static_cast<double>(relevant.size());

    std::size_t hits5 = 0;
    std::size_t hits10 = 0;
    std::size_t hits = 0;
    double precision_sum = 0.0;

    // Avoid counting the same image more than once
    std::unordered_set<ImageIndex> seen;

    const std::size_t limit = std::min<std::size_t>(10, ranked.size());

    for(std::size_t i = 0; i < limit; ++i) {
        const ImageIndex& image = ranked[i];

        // Only the first occurrence of an image counts
        if(!seen.insert(image).second) {
            continue;
        }

        const bool is_relevant = relevant.count(image) > 0;
        if(!is_relevant) {
            continue;
        }

        ++hits;
        ++hits10;
        if(i < 5) {
            ++hits5;
        }

        // i starts from zero so the rank is i + 1
        precision_sum += static_cast<double>(hits) / static_cast<double>(i + 1);
    }

    metrics.recall5 = static_cast<double>(hits5) / num_relevant;

    metrics.recall10 = static_cast<double>(hits10) / num_relevant;

    metrics.ap10 = precision_sum / num_relevant;

    return metrics;
}

Summary summarize(const std::vector<QueryMetrics>& per_query, const std::vector<double>& query_ms) {
    Summary summary;

    summary.queries = per_query.size();

    // Empty input: return the default-initialized summary
    if(per_query.empty()) {
        return summary;
    }

    double sum_recall5 = 0.0;
    double sum_recall10 = 0.0;
    double sum_ap10 = 0.0;
    double sum_query_ms = 0.0;

    for(const QueryMetrics& metrics : per_query) {
        sum_recall5 += metrics.recall5;
        sum_recall10 += metrics.recall10;
        sum_ap10 += metrics.ap10;

        if(metrics.returned < 10) {
            ++summary.queries_with_fewer_than_10;
        }
    }

    const double num_queries = static_cast<double>(per_query.size());

    summary.mean_recall5 = sum_recall5 / num_queries;
    summary.mean_recall10 = sum_recall10 / num_queries;
    summary.map10 = sum_ap10 / num_queries;

    // Average the available query timings
    if(!query_ms.empty()) {
        for(double ms : query_ms) {
            sum_query_ms += ms;
        }

        summary.avg_query_ms = sum_query_ms / static_cast<double>(query_ms.size());
    }

    return summary;
}

}
