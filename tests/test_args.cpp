#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include "args.hpp"
using namespace image_search;
static int checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)

using V = std::vector<std::string>;
static V split_words(const std::string& s) { V v; std::istringstream in(s); std::string w; while (in >> w) v.push_back(w); return v; }
static const std::string kBase = "-hp hp -mir mir -split split.txt -vocab 256 -D 1000 -set validation -o out.txt";
static Args P(const std::string& s) { return parse_args(split_words(s)); }
static bool bad(const std::string& s, const std::string& must_contain = "") {
    try { P(s); } catch (const ArgsError& e) { return std::string(e.what()).find(must_contain) != std::string::npos; }
    return false;
}

int main() {
    // defaults of every method (the spec's)
    { Args a = P(kBase + " -exact");
      CHECK(a.method == Method::Exact && a.hp_dir == "hp" && a.mir_dir == "mir" && a.split_file == "split.txt");
      CHECK(a.K == 256 && a.D == 1000 && a.S == 500 && a.seed == 1 && a.query_set == Split::Validation && a.output == "out.txt"); }
    { Args a = P(kBase + " -lsh"); CHECK(a.lsh.k == 4 && a.lsh.L == 5 && a.lsh.w == 0.05); }
    { Args a = P(kBase + " -hypercube"); CHECK(a.hypercube.k == 10 && a.hypercube.M == 500 && a.hypercube.probes == 10 && a.hypercube.w == 0.05); }
    { Args a = P(kBase + " -ivfflat"); CHECK(!a.ivfflat.kclusters && a.ivfflat.nprobe == 5); }
    { Args a = P(kBase + " -ivfpq"); CHECK(!a.ivfpq.kclusters && a.ivfpq.nprobe == 5 && a.ivfpq.M == 16 && a.ivfpq.nbits == 8); }
    // the spec's example, options in any order, seed propagated
    { Args a = P("-hp hpatches -mir mirflickr -split split.txt -vocab 256 -S 500 -D 10000 -set validation -o output.txt -ivfflat -kclusters 100 -nprobe 10 -seed 1");
      CHECK(a.method == Method::IVFFlat && *a.ivfflat.kclusters == 100 && a.ivfflat.nprobe == 10 && a.D == 10000); }
    { Args a = P("-lsh -k 6 -L 9 -w 0.25 -seed 42 -hp hp -mir mir -split s -vocab 8 -D 5 -o o -S 300 -set test");
      CHECK(a.lsh.k == 6 && a.lsh.L == 9 && a.lsh.w == 0.25 && a.lsh.seed == 42 && a.hypercube.seed == 42 && a.seed == 42 && a.S == 300 && a.query_set == Split::Test); }
    { Args a = P(kBase + " -hypercube -kproj 12 -w 1e-1 -M 100 -probes 3 -seed 7");
      CHECK(a.hypercube.k == 12 && a.hypercube.w == 0.1 && a.hypercube.M == 100 && a.hypercube.probes == 3 && a.hypercube.seed == 7); }
    { Args a = P(kBase + " -ivfpq -M 8 -nbits 4 -nprobe 2 -kclusters 50"); CHECK(a.ivfpq.M == 8 && a.ivfpq.nbits == 4 && a.ivfpq.nprobe == 2 && *a.ivfpq.kclusters == 50); }
    // -M means different things; the other method's value must not leak
    { Args a = P(kBase + " -ivfpq -M 8"); CHECK(a.hypercube.M == 500); }
    { Args a = P(kBase + " -hypercube -M 8"); CHECK(a.ivfpq.M == 16); }
    // help
    CHECK(P("-h").help && P("--help").help && P(kBase + " -lsh -help").help);

    // structure errors
    CHECK(bad(kBase, "no search method"));
    CHECK(bad(kBase + " -lsh -exact", "more than one method"));
    CHECK(bad(kBase + " -lsh -lsh", "given twice"));
    CHECK(bad("-exact -hp hp", "missing required options"));
    CHECK(bad("-exact -hp hp -mir m -split s -vocab 4 -D 1 -set test", "-o"));
    CHECK(bad(kBase + " -exact -D 5", "given twice"));
    CHECK(bad(kBase + " -exact -bogus 1", "unknown option"));
    CHECK(bad(kBase + " -exact stray", "unexpected argument"));
    CHECK(bad(kBase + " -lsh -k", "missing value"));
    CHECK(bad(kBase + " -lsh -k -L 3", "missing value"));
    CHECK(bad(kBase + " -exact -o", "missing value"));
    // options of the wrong method
    CHECK(bad(kBase + " -exact -k 3", "not an option"));
    CHECK(bad(kBase + " -lsh -nprobe 3", "not an option"));
    CHECK(bad(kBase + " -ivfflat -M 4", "not an option"));
    CHECK(bad(kBase + " -hypercube -nbits 4", "not an option"));
    CHECK(bad(kBase + " -ivfpq -w 0.1", "not an option"));
    // value errors
    for (const char* s : {" -lsh -k 0", " -lsh -k -3", " -lsh -k 2.5", " -lsh -k abc", " -lsh -k 3x", " -lsh -k +3", " -lsh -L 99999999999999999999",
                          " -lsh -w 0", " -lsh -w -1", " -lsh -w abc", " -lsh -w 1x", " -lsh -w inf", " -lsh -w nan", " -lsh -w 1e999",
                          " -hypercube -kproj 0", " -hypercube -kproj 25", " -hypercube -probes 0", " -ivfpq -nbits 0", " -ivfpq -nbits 17",
                          " -ivfpq -M 0", " -ivfpq -M 7", " -ivfflat -nprobe 0", " -ivfflat -kclusters 0", " -lsh -k 3000000000"})
        CHECK(bad(kBase + s));
    CHECK(bad("-hp hp -mir mir -split s -vocab 0 -D 1 -set test -o o -exact", "-vocab"));
    CHECK(bad("-hp hp -mir mir -split s -vocab 4 -D 25001 -set test -o o -exact", "-D"));
    CHECK(bad("-hp hp -mir mir -split s -vocab 4 -D 1 -set train -o o -exact", "-set"));
    CHECK(bad("-hp hp -mir mir -split s -vocab 4 -D 1 -set test -o o -exact -S 0", "-S"));
    CHECK(bad("-hp hp -mir mir -split s -vocab 4 -D 1 -set test -o o -exact -seed -1", "-seed"));
    CHECK(bad("-hp hp -mir mir -split s -vocab 4 -D 1 -set test -o o -exact -seed 4294967296", "-seed"));
    CHECK(P("-hp hp -mir mir -split s -vocab 4 -D 0 -set test -o o -exact -seed 4294967295").seed == 4294967295u);
    CHECK(bad("-hp hp -mir mir -split s -vocab 100 -D 1 -set test -o o -ivfpq", "divisible"));     // default M = 16
    CHECK(P("-hp hp -mir mir -split s -vocab 96 -D 1 -set test -o o -ivfpq").ivfpq.M == 16);
    CHECK(P("-hp hp -mir mir -split s -vocab 100 -D 1 -set test -o o -ivfflat").method == Method::IVFFlat);   // divisibility only for ivfpq
    CHECK(P("-hp hp -mir mir -split s -vocab 4 -D 25000 -set test -o o -exact").D == 25000);

    // resolve_kclusters == ceil(sqrt(n)) for many n, computed independently
    for (std::size_t n = 1; n <= 200000; ++n) {
        std::size_t c = 0; while (c * c < n) ++c;   // smallest c with c*c >= n
        if (resolve_kclusters(std::nullopt, n) != c) { std::fprintf(stderr, "ceil sqrt mismatch n=%zu\n", n); return 1; }
    }
    CHECK(resolve_kclusters(std::nullopt, 1115) == 34 && resolve_kclusters(std::nullopt, 25115) == 159);
    CHECK(resolve_kclusters(7, 10) == 7 && resolve_kclusters(10, 10) == 10);
    bool t = false; try { resolve_kclusters(11, 10); } catch (const ArgsError&) { t = true; } CHECK(t);
    t = false; try { resolve_kclusters(std::nullopt, 0); } catch (const ArgsError&) { t = true; } CHECK(t);
    CHECK(std::string(method_name(Method::Exact)) == "Exact Search");
    CHECK(usage("./search").find("-ivfpq") != std::string::npos);
    std::printf("test_args: all %d checks passed\n", checks);
}