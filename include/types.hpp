#pragma once
// Shared basic types. No dependencies on OpenCV so that only features.cpp needs it.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace image_search {

// Dense row-major float matrix. Used for SIFT descriptors (n x 128),
// k-means centers (K x 128) and image vectors (n_images x K).
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    bool empty() const { return rows_ == 0; }

    float*       row(std::size_t i)       { return data_.data() + i * cols_; }
    const float* row(std::size_t i) const { return data_.data() + i * cols_; }

    void append_row(const float* r) { data_.insert(data_.end(), r, r + cols_); ++rows_; }
    std::vector<float>& raw() { return data_; }
    const std::vector<float>& raw() const { return data_; }

private:
    std::size_t rows_ = 0, cols_ = 0;
    std::vector<float> data_;
};

using ImageIndex = std::uint32_t;   // row index into the database matrix

enum class Split { Train, Validation, Test };
enum class Category { Illumination, Viewpoint };   // i_* / v_* prefix of the sequence name

}  // namespace image_search