#pragma once
#include <string>
#include <unordered_set>
#include <vector>
#include "types.hpp"

namespace image_search {

struct Sequence {
    std::string name;       // e.g. "v_bird"
    Split split;
    Category category;
};

// One line per sequence: "<name> <train|validation|test>".
std::vector<Sequence> read_split(const std::string& split_file);

struct ImageRef {
    std::string id;     // "v_bird/2" for HPatches, "im123.jpg" for MIRFlickr
    std::string path;   // where to imread it from
};

// Queries: image 1 of every sequence in the requested split.
struct Query {
    std::string sequence;
    Category category;
    ImageRef image;
};
std::vector<Query> make_queries(const std::string& hp_dir,
                                const std::vector<Sequence>& seqs, Split which);

// HPatches database images (images 2..6 of each sequence in `which`),
// in a fixed, documented order.
std::vector<ImageRef> hpatches_database(const std::string& hp_dir,
                                        const std::vector<Sequence>& seqs, Split which);

// All MIRFlickr images, sorted by ascending image number. The "first D valid images"
// rule (skip images with zero SIFT descriptors) is applied by the caller,
// because validity is only known after feature extraction.
std::vector<ImageRef> mirflickr_ordered(const std::string& mir_dir);

// For a query's sequence, the database rows that are relevant (exactly five).
std::unordered_set<ImageIndex> relevant_rows(const std::vector<ImageRef>& db,
                                             const std::string& sequence);

}