#pragma once

#include <cstddef>

namespace image_search {

// 2D Point with double precision coordinates.
struct Point2D {
    double x = 0.0;
    double y = 0.0;

    Point2D() = default;
    Point2D(double x_, double y_);

    bool operator==(const Point2D& o) const;
    bool operator!=(const Point2D& o) const;

    double distance_to(const Point2D& o) const;
    double squared_distance_to(const Point2D& o) const;
};

// 2D Point match / correspondence between source image (p1) and target image (p2).
struct Match2D {
    Point2D p1;  // (x, y) in source / query image
    Point2D p2;  // (u, v) in destination / candidate image

    Match2D() = default;
    Match2D(Point2D src, Point2D dst);
    Match2D(double x, double y, double u, double v);
};

// Hartley isotropic normalization parameters:
// p_norm = s * (p - mean)
struct Normalization {
    double s = 1.0;
    double mean_x = 0.0;
    double mean_y = 0.0;

    Point2D transform(const Point2D& p) const;
    Point2D inverse_transform(const Point2D& p) const;
};

}  // namespace image_search
