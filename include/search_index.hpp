#pragma once
// Builds the index chosen on the command line and wraps it in one call shape, so ./search
// (top 10) and ./verify (top C candidates) run every method the same way
#include <cstddef>
#include <functional>
#include <vector>

#include "args.hpp"
#include "types.hpp"

namespace image_search {

// A query vector and how many results are wanted in, ranked database rows out (best first)
// An approximate method may return fewer than k rows
using SearchFn = std::function<std::vector<ImageIndex>(const float* q, std::size_t k)>;

// Builds the index of args.method over db and logs its parameters on stderr
// The index is owned by the returned function; it keeps a pointer to db, so db must outlive it
SearchFn make_search(const Args& args, const Matrix& db);

}  // namespace image_search