#pragma once
// Command-line parsing for ./search
//
//   ./search -hp <dir> -mir <dir> -split <file> -vocab <K> -D <int>
//            -set <validation|test> -o <file> [-S <int>] [-seed <int>]
//            -exact
//          | -lsh       [-k <int>] [-L <int>] [-w <double>]
//          | -hypercube [-kproj <int>] [-w <double>] [-M <int>] [-probes <int>]
//          | -ivfflat   [-kclusters <int>] [-nprobe <int>]
//          | -ivfpq     [-kclusters <int>] [-nprobe <int>] [-M <int>] [-nbits <int>]
//
// Options may come in any order
// Exactly one method flag is required, and every method option must belong to that method 
// (-M means different things for -hypercube and -ivfpq, so it is only accepted for those two)
// Each option may be given at most once
// Anything wrong throws ArgsError with a one-line message.
//
// Defaults are the ones in the assignment spec
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "hypercube.hpp"
#include "lsh.hpp"
#include "types.hpp"

namespace image_search {

class ArgsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class Method { Exact, LSH, Hypercube, IVFFlat, IVFPQ };

struct IVFFlatOptions {
    std::optional<std::size_t> kclusters;   // unset -> ceil(sqrt(n)), see resolve_kclusters
    std::size_t nprobe = 5;
};

struct IVFPQOptions {
    std::optional<std::size_t> kclusters;   // unset -> ceil(sqrt(n)), see resolve_kclusters
    std::size_t nprobe = 5;
    std::size_t M = 16;                     // sub-vectors; K must be divisible by M
    std::size_t nbits = 8;                  // 2^nbits centroids per sub-space
};

struct Args {
    // Shared options
    std::string hp_dir;                     // -hp
    std::string mir_dir;                    // -mir
    std::string split_file;                 // -split
    std::size_t K = 0;                      // -vocab
    std::size_t S = 500;                    // -S
    std::size_t D = 0;                      // -D
    Split query_set = Split::Validation;    // -set
    std::string output;                     // -o
    std::uint32_t seed = 1;                 // -seed (also copied into lsh.seed and hypercube.seed)

    Method method = Method::Exact;

    // Options of each method
    // Only the chosen method's are read from the command line, the others keep their defaults.
    LSHParams lsh;                          // spec defaults: k = 4, L = 5, w = 0.05
    HypercubeParams hypercube{.k = 10, .M = 500, .probes = 10, .w = 0.05};
    IVFFlatOptions ivfflat;
    IVFPQOptions ivfpq;

    bool help = false;                      // -h / -help / --help was given; nothing else is checked
};

// `argv[0]` is the program name and is skipped.
Args parse_args(int argc, const char* const* argv);

// The tokens after the program name.
Args parse_args(const std::vector<std::string>& tokens);

std::string usage(const std::string& program);

// The name written as the first line of the output file
const char* method_name(Method m);

// The number of IVF clusters for a database of n vectors: the requested value, or ceil(sqrt(n)) when none was given
// Throws ArgsError if the request exceeds n
std::size_t resolve_kclusters(const std::optional<std::size_t>& requested, std::size_t n);

}  // namespace image_search