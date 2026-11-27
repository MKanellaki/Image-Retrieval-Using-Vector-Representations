#include "search_index.hpp"
#include <climits>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include "exact.hpp"
#include "hypercube.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"
#include "lsh.hpp"

namespace image_search {

namespace {

int to_int(std::size_t v) { return v > static_cast<std::size_t>(INT_MAX) ? INT_MAX : static_cast<int>(v); }

// Wraps any index with a query(q, k) member; shared_ptr because std::function needs a copyable callable
template <class Index>
SearchFn wrap(std::shared_ptr<const Index> ix) {
    return [ix](const float* q, std::size_t k) { return ix->query(q, k); };
}

}  // namespace

SearchFn make_search(const Args& args, const Matrix& db) {
    const std::size_t n = db.rows();
    switch (args.method) {
        case Method::Exact:
            return wrap(std::make_shared<const ExactIndex>(db));
        case Method::LSH:
            std::fprintf(stderr, "[search] LSH: k = %d, L = %d, w = %g\n", args.lsh.k, args.lsh.L, args.lsh.w);
            return wrap(std::make_shared<const LSHIndex>(db, args.lsh));
        case Method::Hypercube:
            std::fprintf(stderr, "[search] Hypercube: kproj = %d, M = %d, probes = %d, w = %g\n",
                         args.hypercube.k, args.hypercube.M, args.hypercube.probes, args.hypercube.w);
            return wrap(std::make_shared<const HypercubeIndex>(db, args.hypercube));
        case Method::IVFFlat: {
            IVFFlatParams p;
            p.k = to_int(resolve_kclusters(args.ivfflat.kclusters, n));
            p.nprobe = to_int(args.ivfflat.nprobe);
            p.seed = args.seed;
            std::fprintf(stderr, "[search] IVFFlat: kclusters = %d, nprobe = %d\n", p.k, p.nprobe);
            return wrap(std::make_shared<const IVFFlatIndex>(db, p));
        }
        case Method::IVFPQ: {
            IVFPQParams p;
            p.k = to_int(resolve_kclusters(args.ivfpq.kclusters, n));
            p.nprobe = to_int(args.ivfpq.nprobe);
            p.m = to_int(args.ivfpq.M);
            p.k_sub = 1 << args.ivfpq.nbits;     // nbits <= 16, checked by parse_args
            p.seed = args.seed;
            std::fprintf(stderr, "[search] IVFPQ: kclusters = %d, nprobe = %d, M = %d, nbits = %zu\n",
                         p.k, p.nprobe, p.m, args.ivfpq.nbits);
            return wrap(std::make_shared<const IVFPQIndex>(db, p));
        }
    }
    throw std::logic_error("unknown method");
}

}  // namespace image_search