#include <cassert>
#include "distance.hpp"

int main() {
    const float a[] = {0.f, 0.f}, b[] = {3.f, 4.f};
    assert(image_search::l2_squared(a, b, 2) == 25.f);   // 3^2 + 4^2
    assert(image_search::l2_squared(a, a, 2) == 0.f);    // distance to itself
    return 0;
}