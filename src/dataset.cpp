#include "../include/dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace image_search {

namespace {

constexpr int kFirstDbImage = 2;  // HPatches images 2..6 form the database
constexpr int kLastDbImage = 6;
constexpr std::size_t kRelevantPerQuery = 5;

Split parse_split(const std::string& s, const std::string& where) {
    if (s == "train") return Split::Train;
    if (s == "validation") return Split::Validation;
    if (s == "test") return Split::Test;
    throw std::runtime_error(where + ": unknown split '" + s + "'");
}

Category parse_category(const std::string& name, const std::string& where) {
    if (name.rfind("i_", 0) == 0) return Category::Illumination;
    if (name.rfind("v_", 0) == 0) return Category::Viewpoint;
    throw std::runtime_error(where + ": sequence '" + name +
                             "' must start with 'i_' or 'v_'");
}

ImageRef hp_image(const std::string& hp_dir, const std::string& seq, int number) {
    const std::string n = std::to_string(number);
    const fs::path path = fs::path(hp_dir) / seq / (n + ".ppm");
    if (!fs::exists(path))
       throw std::runtime_error("HPatches image not found: " + path.string());
    return ImageRef{seq + "/" + n, path.string()};
}

// Matches exactly "im<digits>.jpg"; returns the number via `out`.
bool parse_mir_name(const std::string& name, unsigned long long& out) {
    static const std::string prefix = "im";
    static const std::string suffix = ".jpg";
    if (name.size() <= prefix.size() + suffix.size()) return false;
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
    const std::size_t begin = prefix.size();
    const std::size_t end = name.size() - suffix.size();
    unsigned long long value = 0;
    for (std::size_t i = begin; i < end; ++i) {
        const char c = name[i];
        if (c < '0' || c > '9') return false;
        value = value * 10 + static_cast<unsigned long long>(c - '0');
    }
    out = value;
    return true;
}

}  // namespace

std::vector<Sequence> read_split(const std::string& split_file) {
    std::ifstream in(split_file);
    if (!in) throw std::runtime_error("cannot open split file: " + split_file);

    std::vector<Sequence> seqs;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        std::istringstream ls(line);  // also strips a trailing '\r' as whitespace
        std::string name, split;
        if (!(ls >> name)) continue;  // blank line
        const std::string where = split_file + ":" + std::to_string(line_no);
        if (!(ls >> split)) throw std::runtime_error(where + ": missing split");
        seqs.push_back({name, parse_split(split, where), parse_category(name, where)});
    }
    return seqs;
}

std::vector<Query> make_queries(const std::string& hp_dir,
                                const std::vector<Sequence>& seqs, Split which) {
    std::vector<Query> queries;
    for (const auto& s : seqs) {
        if (s.split != which) continue;
        queries.push_back({s.name, s.category, hp_image(hp_dir, s.name, 1)});
    }
    return queries;
}

// Order: sequences in split-file order, then image 2..6 within each sequence.
std::vector<ImageRef> hpatches_database(const std::string& hp_dir,
                                        const std::vector<Sequence>& seqs, Split which) {
    std::vector<ImageRef> db;
    for (const auto& s : seqs) {
        if (s.split != which) continue;
        for (int n = kFirstDbImage; n <= kLastDbImage; ++n)
            db.push_back(hp_image(hp_dir, s.name, n));
    }
    return db;
}

std::vector<ImageRef> mirflickr_ordered(const std::string& mir_dir) {
    if (!fs::is_directory(mir_dir))
        throw std::runtime_error("MIRFlickr directory not found: " + mir_dir);

    struct Entry {
        unsigned long long number;
        std::string name;
    };
    std::vector<Entry> entries;
    for (const auto& de : fs::directory_iterator(mir_dir)) {
        if (!de.is_regular_file()) continue;
        const std::string name = de.path().filename().string();
        unsigned long long number = 0;
        if (parse_mir_name(name, number)) entries.push_back({number, name});
    }
    // Numeric order (im2 before im10); name as tie-break for determinism.
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.number != b.number ? a.number < b.number : a.name < b.name;
    });

    if (entries.empty())
        throw std::runtime_error("no im<number>.jpg files found in " + mir_dir);

    std::vector<ImageRef> out;
    out.reserve(entries.size());
    for (const auto& e : entries)
        out.push_back({e.name, (fs::path(mir_dir) / e.name).string()});
    return out;
}

std::unordered_set<ImageIndex> relevant_rows(const std::vector<ImageRef>& db,
                                             const std::string& sequence) {
    const std::string prefix = sequence + "/";
    std::unordered_set<ImageIndex> rows;
    for (std::size_t i = 0; i < db.size(); ++i)
        if (db[i].id.compare(0, prefix.size(), prefix) == 0)
            rows.insert(static_cast<ImageIndex>(i));
    if (rows.size() != kRelevantPerQuery)
        throw std::runtime_error("expected " + std::to_string(kRelevantPerQuery) +
                                 " relevant images for '" + sequence + "', found " +
                                 std::to_string(rows.size()));
    return rows;
}

}