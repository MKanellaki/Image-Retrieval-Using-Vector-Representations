// Unit tests for dataset.cpp. Builds a tiny fake HPatches/MIRFlickr tree in a
// temporary directory, so no real data is needed.
#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "dataset.hpp"

namespace fs = std::filesystem;
using namespace image_search;

static void touch(const fs::path& p) {
    fs::create_directories(p.parent_path());
    std::ofstream(p).put('x');
}

template <typename F>
static bool throws(F f) {
    try { f(); } catch (const std::runtime_error&) { return true; }
    return false;
}

int main() {
    const fs::path root = fs::temp_directory_path() / ("dataset_test_" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root);

    // --- split file: CRLF line ending, blank line, trailing spaces ---
    {
        std::ofstream out(root / "split.txt", std::ios::binary);
        out << "i_a train\r\n"
               "\n"
               "i_ab test  \n"
               "v_x test\n"
               "v_y validation\n";
    }
    const auto seqs = read_split((root / "split.txt").string());
    assert(seqs.size() == 4);
    assert(seqs[0].name == "i_a" && seqs[0].split == Split::Train && seqs[0].category == Category::Illumination);
    assert(seqs[1].name == "i_ab" && seqs[1].split == Split::Test);
    assert(seqs[2].name == "v_x" && seqs[2].category == Category::Viewpoint);
    assert(seqs[3].split == Split::Validation);

    // --- bad split files are rejected with a clear error ---
    {
        std::ofstream(root / "bad1.txt") << "i_a training\n";   // unknown split
        std::ofstream(root / "bad2.txt") << "x_a train\n";      // bad prefix
        std::ofstream(root / "bad3.txt") << "i_a\n";            // missing split
    }
    assert(throws([&] { read_split((root / "bad1.txt").string()); }));
    assert(throws([&] { read_split((root / "bad2.txt").string()); }));
    assert(throws([&] { read_split((root / "bad3.txt").string()); }));
    assert(throws([&] { read_split((root / "missing.txt").string()); }));

    // --- fake HPatches tree ---
    const std::string hp = (root / "hp").string();
    for (const char* s : {"i_a", "i_ab", "v_x", "v_y"})
        for (int n = 1; n <= 6; ++n) touch(root / "hp" / s / (std::to_string(n) + ".ppm"));

    // --- queries: image 1 of every sequence in the set, in split-file order ---
    const auto q = make_queries(hp, seqs, Split::Test);
    assert(q.size() == 2);
    assert(q[0].sequence == "i_ab" && q[0].image.id == "i_ab/1");
    assert(q[0].image.path == (root / "hp" / "i_ab" / "1.ppm").string());
    assert(q[1].sequence == "v_x" && q[1].category == Category::Viewpoint);

    // --- database: images 2..6 of each sequence; training sequences excluded ---
    const auto db = hpatches_database(hp, seqs, Split::Test);
    assert(db.size() == 10);
    assert(db[0].id == "i_ab/2" && db[4].id == "i_ab/6" && db[5].id == "v_x/2" && db[9].id == "v_x/6");

    // --- relevant rows: exactly the 5 images of the sequence ---
    const auto rel = relevant_rows(db, "i_ab");
    assert(rel.size() == 5);
    for (ImageIndex r = 0; r < 5; ++r) assert(rel.count(r) == 1);
    assert(relevant_rows(db, "v_x").count(5) == 1);
    // A sequence that is not in this database is an error, not an empty set.
    assert(throws([&] { relevant_rows(db, "v_y"); }));
    // "i_a" must not match "i_ab/..." (prefix collision): it is not in the test database.
    assert(throws([&] { relevant_rows(db, "i_a"); }));

    // --- MIRFlickr: numeric order, junk ignored ---
    for (const char* n : {"im10.jpg", "im2.jpg", "im1.jpg", "im100.jpg", "im9.jpg"})
        touch(root / "mir" / n);
    for (const char* n : {"im3.jpeg", "imX.jpg", "im.jpg", "README.txt", "IM4.jpg"})
        touch(root / "mir" / n);
    fs::create_directories(root / "mir" / "im7.jpg");   // a directory with a matching name
    const auto mir = mirflickr_ordered((root / "mir").string());
    assert(mir.size() == 5);
    const char* expected[] = {"im1.jpg", "im2.jpg", "im9.jpg", "im10.jpg", "im100.jpg"};
    for (std::size_t i = 0; i < 5; ++i) {
        assert(mir[i].id == expected[i]);
        assert(mir[i].path == (root / "mir" / expected[i]).string());
    }
    assert(throws([&] { mirflickr_ordered((root / "no_such_dir").string()); }));

    // --- a wrong HPatches path or an empty MIRFlickr folder fails loudly ---
    assert(throws([&] { make_queries((root / "no_hp").string(), seqs, Split::Test); }));
    assert(throws([&] { hpatches_database((root / "no_hp").string(), seqs, Split::Test); }));
    fs::create_directories(root / "empty_mir");
    assert(throws([&] { mirflickr_ordered((root / "empty_mir").string()); }));

    fs::remove_all(root);
    return 0;
}