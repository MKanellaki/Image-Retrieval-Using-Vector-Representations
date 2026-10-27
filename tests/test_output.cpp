#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include "output.hpp"
using namespace image_search;
static int checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
static std::vector<std::string> lines(const std::string& s) { std::vector<std::string> v; std::istringstream in(s); std::string l; while (std::getline(in, l)) v.push_back(l); return v; }
int main() {
    std::vector<ImageRef> db;
    for (int i = 2; i <= 6; ++i) db.push_back({"v_a/" + std::to_string(i), ""});
    for (int i = 2; i <= 6; ++i) db.push_back({"i_b/" + std::to_string(i), ""});
    db.push_back({"im123.jpg", ""}); db.push_back({"im7.jpg", ""});
    std::unordered_set<ImageIndex> rel_a{0,1,2,3,4}, rel_b{5,6,7,8,9};

    std::vector<QueryResult> res;
    { QueryResult r; r.sequence = "v_a"; r.ranked = {0, 10, 1, 11, 2, 3, 4, 5, 6, 7}; r.metrics = evaluate_query(r.ranked, rel_a); r.query_ms = 1.5; res.push_back(r); }
    { QueryResult r; r.sequence = "i_b"; r.ranked = {5, 6, 10}; r.metrics = evaluate_query(r.ranked, rel_b); r.query_ms = 0.5; res.push_back(r); }   // fewer than 10

    std::ostringstream os; os << std::scientific << std::setprecision(2);
    const std::ios_base::fmtflags flags = os.flags();
    Summary s = write_output(os, "LSH", res, db);
    CHECK(os.flags() == flags);
    auto L = lines(os.str());
    CHECK(L.size() == 1 + 2 * (1 + 10 + 3) + 4);
    CHECK(L[0] == "LSH" && L[1] == "Query: v_a" && L[2] == "Rank-1: v_a/2" && L[3] == "Rank-2: im123.jpg" && L[11] == "Rank-10: i_b/4");
    CHECK(L[12].rfind("Recall@5: ", 0) == 0 && L[13].rfind("Recall@10: ", 0) == 0 && L[14].rfind("AP@10: ", 0) == 0);
    CHECK(L[15] == "Query: i_b" && L[16] == "Rank-1: i_b/2" && L[18] == "Rank-3: im123.jpg" && L[19] == "Rank-4: -" && L[25] == "Rank-10: -");
    CHECK(L[26].rfind("Recall@5:", 0) == 0);
    CHECK(L[L.size()-4].rfind("Mean Recall@5: ", 0) == 0 && L[L.size()-3].rfind("Mean Recall@10: ", 0) == 0 && L[L.size()-2].rfind("mAP@10: ", 0) == 0 && L[L.size()-1] == "Average query time: 1.000000");
    // v_a: relevant rows sit at ranks 1, 3, 5, 6, 7 -> AP@10 = (1/1 + 2/3 + 3/5 + 4/6 + 5/7) / 5
    { std::ostringstream want; want << std::fixed << std::setprecision(6)
          << "AP@10: " << (1.0/1 + 2.0/3 + 3.0/5 + 4.0/6 + 5.0/7) / 5;
      CHECK(L[14] == want.str()); }
    // i_b: relevant rows at ranks 1, 2 -> AP@10 = (1/1 + 2/2) / 5 = 0.4
    CHECK(L[28] == "AP@10: 0.400000");
    CHECK(L[12] == "Recall@5: 0.600000" && L[13] == "Recall@10: 1.000000");
    CHECK(L[26] == "Recall@5: 0.400000" && L[27] == "Recall@10: 0.400000");   // i_b: rows 5,6 relevant, 10 not -> 2/5
    CHECK(s.queries == 2 && s.queries_with_fewer_than_10 == 1);
    CHECK(L[L.size()-4] == "Mean Recall@5: 0.500000" && L[L.size()-3] == "Mean Recall@10: 0.700000");

    // invalid row: nothing written
    { QueryResult bad = res[0]; bad.ranked[3] = 999; std::ostringstream o2; bool t = false;
      try { write_output(o2, "LSH", {res[0], bad}, db); } catch (const std::out_of_range&) { t = true; }
      CHECK(t && o2.str().empty()); }
    // empty results: header and zero means, no crash
    { std::ostringstream o3; Summary e = write_output(o3, "Exact Search", {}, db); auto E = lines(o3.str());
      CHECK(E.size() == 5 && E[0] == "Exact Search" && E[1] == "Mean Recall@5: 0.000000" && e.queries == 0); }
    // file variant
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("out_test_" + std::to_string(::getpid())); fs::remove_all(dir); fs::create_directories(dir);
    const std::string f = (dir / "o.txt").string();
    write_output_file(f, "LSH", res, db);
    { std::ifstream in(f); std::stringstream ss; ss << in.rdbuf(); CHECK(ss.str() == os.str()); }
    bool t = false; try { write_output_file((dir / "no" / "o.txt").string(), "LSH", res, db); } catch (const std::runtime_error&) { t = true; } CHECK(t);
    // invalid row with an existing file: what happens to the old content?
    { QueryResult bad = res[0]; bad.ranked[0] = 999; bool th = false;
      try { write_output_file(f, "LSH", {bad}, db); } catch (const std::out_of_range&) { th = true; }
      std::ifstream in(f); std::stringstream ss; ss << in.rdbuf();
      CHECK(th); CHECK(!ss.str().empty());
      std::ifstream in2(f); std::stringstream s2; s2 << in2.rdbuf(); CHECK(s2.str() == os.str());   // old result intact
      CHECK(!fs::exists(f + ".tmp")); }
    // success leaves no temp file and overwrites an existing result
    { std::vector<QueryResult> one{res[0]}; write_output_file(f, "Exact Search", one, db);
      std::ifstream in(f); std::string first; std::getline(in, first); CHECK(first == "Exact Search" && !fs::exists(f + ".tmp")); }
    // target is a directory: rename fails, error reported, no temp left behind
    { fs::create_directories(dir / "d"); bool th = false;
      try { write_output_file((dir / "d").string(), "LSH", res, db); } catch (const std::runtime_error&) { th = true; }
      CHECK(th && !fs::exists((dir / "d").string() + ".tmp") && fs::is_directory(dir / "d")); }
    fs::remove_all(dir);
    std::printf("test_output: all %d checks passed\n", checks);
}