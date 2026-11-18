#include "homography.hpp"

#include <cmath>

namespace image_search {

Point2D::Point2D(double x_, double y_) : x(x_), y(y_) {}

bool Point2D::operator==(const Point2D& o) const {
    return x == o.x && y == o.y;
}

bool Point2D::operator!=(const Point2D& o) const {
    return !(*this == o);
}

double Point2D::distance_to(const Point2D& o) const {
    double dx = x - o.x;
    double dy = y - o.y;
    return std::sqrt(dx * dx + dy * dy);
}

double Point2D::squared_distance_to(const Point2D& o) const {
    double dx = x - o.x;
    double dy = y - o.y;
    return dx * dx + dy * dy;
}

Match2D::Match2D(Point2D src, Point2D dst) : p1(src), p2(dst) {}

Match2D::Match2D(double x, double y, double u, double v) : p1(x, y), p2(u, v) {}

Point2D Normalization::transform(const Point2D& p) const {
    return Point2D(s * (p.x - mean_x), s * (p.y - mean_y));
}

Point2D Normalization::inverse_transform(const Point2D& p) const {
    return Point2D(p.x / s + mean_x, p.y / s + mean_y);
}

}  // namespace image_search
