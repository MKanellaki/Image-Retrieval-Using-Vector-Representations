#pragma once
// Internal helpers for the binary files written by bow (not part of the public API)
// Values are written in native byte order 
// the files are caches, not an exchange format
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "types.hpp"

namespace image_search::binio {

class Writer {
public:
    explicit Writer(const std::string& path) : path_(path), out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) throw std::runtime_error("cannot open for writing: " + path);
    }

    void bytes(const void* p, std::size_t n) {
        out_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
        if (!out_) throw std::runtime_error("write failed: " + path_);
    }
    void magic(const char (&m)[9]) { bytes(m, 8); }
    void u64(std::uint64_t v) { bytes(&v, sizeof v); }
    void str(const std::string& s) { u64(s.size()); bytes(s.data(), s.size()); }
    void matrix(const Matrix& m) {
        u64(m.rows());
        u64(m.cols());
        bytes(m.raw().data(), m.raw().size() * sizeof(float));
    }
    void close() {
        out_.close();
        if (!out_) throw std::runtime_error("write failed: " + path_);
    }

private:
    std::string path_;
    std::ofstream out_;
};

class Reader {
public:
    explicit Reader(const std::string& path) : path_(path), in_(path, std::ios::binary) {
        if(!in_) throw std::runtime_error("cannot open for reading: " + path);
        in_.seekg(0, std::ios::end);
        remaining_ = static_cast<std::uint64_t>(in_.tellg());
        in_.seekg(0, std::ios::beg);
    }

    void bytes(void* p, std::size_t n) {
        need(n);
        in_.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
        if (!in_) fail("read error");
        remaining_ -= n;
    }
    void expect_magic(const char (&m)[9]) {
        char got[8];
        bytes(got, 8);
        if (std::memcmp(got, m, 8) != 0)
            fail(std::string("bad magic, expected '") + std::string(m, 8) + "'");
    }
    std::uint64_t u64() { std::uint64_t v; bytes(&v, sizeof v); return v; }
    std::string str() {
        const std::uint64_t n = u64();
        need(n);
        std::string s(static_cast<std::size_t>(n), '\0');
        bytes(s.data(), s.size());
        return s;
    }
    Matrix matrix() {
        const std::uint64_t rows = u64(), cols = u64();
        if (cols != 0 && rows > remaining_ / sizeof(float) / cols) fail("truncated matrix");
        Matrix m(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        bytes(m.raw().data(), m.raw().size() * sizeof(float));
        return m;
    }
    void expect_end() { if (remaining_ != 0) fail("trailing bytes"); }

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error(path_ + ": " + what);
    }

private:
    void need(std::uint64_t n) const { if (n > remaining_) fail("truncated file"); }

    std::string path_;
    std::ifstream in_;
    std::uint64_t remaining_ = 0;
};

}  // namespace image_search::binio
