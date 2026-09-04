#include "ImmersedGeometry.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace immersed {
namespace {

double Dot(const Point &a, const Point &b) {
    double result = 0.;
    for (int i = 0; i < 3; ++i) result += a[size_t(i)] * b[size_t(i)];
    return result;
}

Point Cross(const Point &a, const Point &b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }

} // namespace

std::vector<Patch> SpherePatches(const Point &center, double radius, double target_area, const BoundaryImmittance &zv, const BoundaryImmittance &yp) {
    if (!(radius > 0.) || !(target_area > 0.)) throw std::runtime_error("Sphere radius and patch area must be positive");
    const double total_area = 4. * std::numbers::pi * radius * radius;
    const int count = std::max(6, int(std::lround(total_area / target_area)));
    const double area = total_area / count;
    const double golden_angle = std::numbers::pi * (3. - std::sqrt(5.));
    std::vector<Patch> patches;
    patches.reserve(size_t(count));
    for (int i = 0; i < count; ++i) {
        const double z = 1. - 2. * (i + .5) / count;
        const double radial = std::sqrt(std::max(0., 1. - z * z));
        const double angle = golden_angle * i;
        const Point normal{radial * std::cos(angle), radial * std::sin(angle), z};
        Point position{};
        for (int d = 0; d < 3; ++d) position[size_t(d)] = center[size_t(d)] + radius * normal[size_t(d)];
        patches.push_back({position, normal, area, zv, yp});
    }
    return patches;
}

std::vector<Patch> SquarePatches(const Point &center, const Point &u, const Point &v, double side, double target_area, const BoundaryImmittance &zv, const BoundaryImmittance &yp) {
    if (!(side > 0.) || !(target_area > 0.)) throw std::runtime_error("Square side and patch area must be positive");
    if (std::abs(Dot(u, u) - 1.) > 1e-6 || std::abs(Dot(v, v) - 1.) > 1e-6 || std::abs(Dot(u, v)) > 1e-6)
        throw std::runtime_error("Square patch axes must be orthonormal");
    const Point normal = Cross(u, v);
    const int divisions = std::max(1, int(std::lround(side / std::sqrt(target_area))));
    const double width = side / divisions;
    std::vector<Patch> patches;
    patches.reserve(size_t(divisions) * divisions);
    for (int y = 0; y < divisions; ++y) {
        for (int x = 0; x < divisions; ++x) {
            const double du = -side / 2. + (x + .5) * width;
            const double dv = -side / 2. + (y + .5) * width;
            Point position{};
            for (int d = 0; d < 3; ++d)
                position[size_t(d)] = center[size_t(d)] + du * u[size_t(d)] + dv * v[size_t(d)];
            patches.push_back({position, normal, width * width, zv, yp});
        }
    }
    return patches;
}

} // namespace immersed
