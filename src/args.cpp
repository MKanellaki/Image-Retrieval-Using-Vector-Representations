#include "args.hpp"

#include <cctype>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <map>
#include <system_error>

namespace image_search {

namespace {

constexpr std::size_t kMirFlickrSize = 25000;   // MIRFlickr-25K
constexpr unsigned kMaxKproj = 24;              // the hypercube allocates 2^kproj buckets
constexpr unsigned kMaxNbits = 16;              // PQ trains 2^nbits centroids per sub-space

const std::map<std::string, Method>& method_flags() {
    static const std::map<std::string, Method> m = {
        {"-exact", Method::Exact},     {"-lsh", Method::LSH}, 
        {"-hypercube", Method::Hypercube},
        {"-ivfflat", Method::IVFFlat}, {"-ivfpq", Method::IVFPQ},
    };
    return m;
}

const char* method_flag(Method m) {
    switch(m) {
        case Method::Exact: return "-exact";
        case Method::LSH: return "-lsh";
        case Method::Hypercube: return "-hypercube";
        case Method::IVFFlat: return "-ivfflat";
        case Method::IVFPQ: return "-ivfpq";
    }
    return "?";
}

const std::vector<std::string>& shared_options() {
    static const std::vector<std::string> v = {"-hp", "-mir", "-split", "-vocab", "-S", "-D",  "-set", "-o",     "-seed"};
    return v;
}

const std::vector<std::string>& required_options() {
    static const std::vector<std::string> v = {"-hp", "-mir", "-split", "-vocab", "-D", "-set", "-o"};
    return v;
}

// Every method option and the methods that accept it
const std::map<std::string, std::vector<Method>>& method_options() {
    static const std::map<std::string, std::vector<Method>> m = {
        {"-k", {Method::LSH}},
        {"-L", {Method::LSH}},
        {"-w", {Method::LSH, Method::Hypercube}},
        {"-kproj", {Method::Hypercube}},
        {"-M", {Method::Hypercube, Method::IVFPQ}},
        {"-probes", {Method::Hypercube}},
        {"-kclusters", {Method::IVFFlat, Method::IVFPQ}},
        {"-nprobe", {Method::IVFFlat, Method::IVFPQ}},
        {"-nbits", {Method::IVFPQ}},
    };
    return m;
}

bool is_shared(const std::string& s) {
    for(const auto& o : shared_options()){
        if(o == s) return true;
    }
    return false;
}

bool takes_value(const std::string& s) { 
    return is_shared(s) || method_options().count(s) != 0;
}

bool is_help(const std::string& s) {
    return s == "-h" || s == "-help" || s == "--help";
}

bool is_option_name(const std::string& s) {
    return takes_value(s) || method_flags().count(s) != 0 || is_help(s);
}

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for(const auto& s : v) {
        out += (out.empty() ? "" : ", ") + s;
    }
    return out;
}

// A non-negative decimal integer in [lo, hi]. No sign, spaces or trailing characters.
std::uint64_t parse_uint(const std::string& flag, const std::string& value, std::uint64_t lo, std::uint64_t hi) {
    std::uint64_t x = 0;
    const char* first = value.data();
    const char* last = first + value.size();
    const auto [end, ec] = std::from_chars(first, last, x);
    const bool digits_only = !value.empty() && std::isdigit(static_cast<unsigned char>(value[0]));
    if(ec == std::errc::result_out_of_range && digits_only && end == last){
        throw ArgsError(flag + ": value " + value + " is too large (at most " + std::to_string(hi) + ")");
    }
    if(ec != std::errc() || end != last || !digits_only) {
        const std::string want = lo == 0 ? "a non-negative integer" : "a positive integer";
        throw ArgsError(flag + ": expected " + want + ", got '" + value + "'");
    }
    if(x < lo) throw ArgsError(flag + ": must be at least " + std::to_string(lo) + ", got " + value);
    if(x > hi) throw ArgsError(flag + ": must be at most " + std::to_string(hi) + ", got " + value);
    return x;
}

int parse_int(const std::string& flag, const std::string& value, int lo, int hi = INT_MAX) {
    return static_cast<int>(parse_uint(flag, value, static_cast<std::uint64_t>(lo), static_cast<std::uint64_t>(hi)));
}

std::size_t parse_size(const std::string& flag, const std::string& value, std::size_t lo, std::size_t hi = SIZE_MAX) {
    return static_cast<std::size_t>(parse_uint(flag, value, lo, hi));
}

// A finite double > 0.
double parse_positive_double(const std::string& flag, const std::string& value) {
    const bool starts_ok = !value.empty() && !std::isspace(static_cast<unsigned char>(value[0]));
    char* end = nullptr;
    errno = 0;
    const double x = starts_ok ? std::strtod(value.c_str(), &end) : 0.0;
    if(!starts_ok || end != value.c_str() + value.size() || errno == ERANGE || !std::isfinite(x)){
        throw ArgsError(flag + ": expected a number, got '" + value + "'");
    }
    if(!(x > 0.0)) throw ArgsError(flag + ": must be greater than 0, got " + value);
    return x;
}

Split parse_set(const std::string& value) {
    if (value == "validation") return Split::Validation;
    if (value == "test") return Split::Test;
    throw ArgsError("-set: expected 'validation' or 'test', got '" + value + "'");
}

}  // namespace

const char* method_name(Method m) {
    switch (m) {
        case Method::Exact: return "Exact Search";
        case Method::LSH: return "LSH";
        case Method::Hypercube: return "Hypercube";
        case Method::IVFFlat: return "IVFFlat";
        case Method::IVFPQ: return "IVFPQ";
    }
    return "?";
}

std::string usage(const std::string& program) {
    return "usage: " + program +
           " -hp <dir> -mir <dir> -split <file> -vocab <K> -D <int>\n"
           "       -set <validation|test> -o <file> [-S <int>] [-seed <int>] <method>\n"
           "\n"
           "shared defaults: -S 500, -seed 1 (the others are required)\n"
           "\n"
           "methods (exactly one; their options are optional):\n"
           "  -exact\n"
           "  -lsh        [-k 4] [-L 5] [-w 0.05]\n"
           "  -hypercube  [-kproj 10] [-w 0.05] [-M 500] [-probes 10]\n"
           "  -ivfflat    [-kclusters ceil(sqrt(n))] [-nprobe 5]\n"
           "  -ivfpq      [-kclusters ceil(sqrt(n))] [-nprobe 5] [-M 16] [-nbits 8]\n";
}

Args parse_args(int argc, const char* const* argv) {
    std::vector<std::string> tokens;
    for(int i = 1; i < argc; ++i) {
        tokens.emplace_back(argv[i]);
    }
    return parse_args(tokens);
}

Args parse_args(const std::vector<std::string>& tokens) {
    Args args;

    // 1. Split the tokens into the method flag and (option, value) pairs
    std::optional<Method> method;
    std::map<std::string, std::string> values;
    for(std::size_t i = 0; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];
        if(is_help(t)) {
            args.help = true;
            return args;
        }
        if(const auto it = method_flags().find(t); it != method_flags().end()) {
            if(method && *method == it->second) throw ArgsError(t + " given twice");
            if(method) {
                throw ArgsError(std::string("more than one method given: ") + method_flag(*method) + " and " + t);
            }
            method = it->second;
            continue;
        }
        if(takes_value(t)) {
            if (i + 1 >= tokens.size()) throw ArgsError(t + ": missing value");
            const std::string& v = tokens[i + 1];
            if (is_option_name(v)) throw ArgsError(t + ": missing value (next is the option " + v + ")");
            if (values.count(t)) throw ArgsError(t + " given twice");
            values[t] = v;
            ++i;
            continue;
        }
        if(!t.empty() && t[0] == '-') throw ArgsError("unknown option '" + t + "'");
        throw ArgsError("unexpected argument '" + t + "'");
    }

    // 2. A method, the required shared options, and only that method's options
    if (!method) throw ArgsError("no search method given; use one of -exact, -lsh, -hypercube, -ivfflat, -ivfpq");
    args.method = *method;

    std::vector<std::string> missing;
    for(const auto& r : required_options()) {
        if (!values.count(r)) {
            missing.push_back(r);
        }
    }
    if(!missing.empty()) {
        throw ArgsError((missing.size() == 1 ? "missing required option " : "missing required options ") + join(missing));
    }

    for(const auto& [flag, value] : values) {
        if(is_shared(flag)) continue;
        const auto& owners = method_options().at(flag);
        bool ok = false;
        std::vector<std::string> names;
        for(Method m : owners) {
            ok = ok || m == args.method;
            names.emplace_back(method_flag(m));
        }
        if (!ok) {
            throw ArgsError(flag + " is not an option of " + method_flag(args.method) + " (it belongs to " + join(names) + ")");
        }
    }

    // 3. Values
    auto get = [&](const std::string& flag) -> const std::string* {
        const auto it = values.find(flag);
        return it == values.end() ? nullptr : &it->second;
    };

    args.hp_dir = *get("-hp");
    args.mir_dir = *get("-mir");
    args.split_file = *get("-split");
    args.output = *get("-o");
    for(const char* f : {"-hp", "-mir", "-split", "-o"}) {
        if(get(f)->empty()) {
            throw ArgsError(std::string(f) + ": value is empty");
        }
    }
    args.K = parse_size("-vocab", *get("-vocab"), 1);
    args.D = parse_size("-D", *get("-D"), 0, kMirFlickrSize);
    args.query_set = parse_set(*get("-set"));
    if(const auto* v = get("-S")) args.S = parse_size("-S", *v, 1);
    if(const auto* v = get("-seed")) args.seed = static_cast<std::uint32_t>(parse_uint("-seed", *v, 0, UINT32_MAX));

    switch (args.method) {
        case Method::Exact:
            break;
        case Method::LSH:
            if (const auto* v = get("-k")) args.lsh.k = parse_int("-k", *v, 1);
            if (const auto* v = get("-L")) args.lsh.L = parse_int("-L", *v, 1);
            if (const auto* v = get("-w")) args.lsh.w = parse_positive_double("-w", *v);
            break;
        case Method::Hypercube:
            if (const auto* v = get("-kproj")) args.hypercube.k = parse_int("-kproj", *v, 1, kMaxKproj);
            if (const auto* v = get("-w")) args.hypercube.w = parse_positive_double("-w", *v);
            if (const auto* v = get("-M")) args.hypercube.M = parse_int("-M", *v, 1);
            if (const auto* v = get("-probes")) args.hypercube.probes = parse_int("-probes", *v, 1);
            break;
        case Method::IVFFlat:
            if (const auto* v = get("-kclusters")) args.ivfflat.kclusters = parse_size("-kclusters", *v, 1);
            if (const auto* v = get("-nprobe")) args.ivfflat.nprobe = parse_size("-nprobe", *v, 1);
            break;
        case Method::IVFPQ:
            if (const auto* v = get("-kclusters")) args.ivfpq.kclusters = parse_size("-kclusters", *v, 1);
            if (const auto* v = get("-nprobe")) args.ivfpq.nprobe = parse_size("-nprobe", *v, 1);
            if (const auto* v = get("-M")) args.ivfpq.M = parse_size("-M", *v, 1);
            if (const auto* v = get("-nbits")) args.ivfpq.nbits = parse_size("-nbits", *v, 1, kMaxNbits);
            if (args.K % args.ivfpq.M != 0)
                throw ArgsError("-ivfpq: -vocab " + std::to_string(args.K) + " is not divisible by -M " +
                                std::to_string(args.ivfpq.M));
            break;
    }

    args.lsh.seed = args.seed;
    args.hypercube.seed = args.seed;
    return args;
}

std::size_t resolve_kclusters(const std::optional<std::size_t>& requested, std::size_t n) {
    if(n == 0) throw ArgsError("-kclusters: the database is empty");
    if(requested) {
        if(*requested > n) {
            throw ArgsError("-kclusters " + std::to_string(*requested) + " is larger than the database (n = " + std::to_string(n) + ")");
        }
        return *requested;
    }
    // ceil(sqrt(n)) in integers: the smallest c with c * c >= n
    std::size_t c = static_cast<std::size_t>(std::sqrt(static_cast<double>(n)));
    while (c * c < n) ++c;
    while (c > 1 && (c - 1) * (c - 1) >= n) --c;
    return c;
}

}  // namespace image_search