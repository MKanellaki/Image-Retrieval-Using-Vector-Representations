#pragma once
// Seed mixing shared by every module that needs reproducible per-item randomness
#include <cstdint>

namespace image_search {

// Mixes a global seed with a per-item index into an independent-looking seed
constexpr std::uint64_t derive_seed(std::uint64_t seed, std::uint64_t index) noexcept {
    std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (index + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace image_search