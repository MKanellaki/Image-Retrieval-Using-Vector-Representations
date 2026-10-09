#include "distance.hpp"

namespace image_search {

float l2_squared(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;

    // Iterate through each dimension of the vectors
    for (std::size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

}