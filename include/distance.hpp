
#pragma once
#include <cstddef>
 
namespace image_search {
 
// Squared Euclidean distance
float l2_squared(const float* a, const float* b, std::size_t dim);
 
}