#include "ImmersedDelta.h"

#include <cmath>
#include <stdexcept>

namespace immersed {
namespace {

struct AxisWeights {
    std::vector<int> Index;
    std::vector<double> Weight;
};

AxisWeights LagrangeAxis(double coordinate, int count, int order) {
    if (order < 1 || order % 2 == 0) throw std::runtime_error("Lagrange order must be positive and odd");
    const int width = order + 1;
    const int first = int(std::floor(coordinate)) - order / 2;
    if (first < 0 || first + width > count) throw std::runtime_error("Lagrange delta support leaves the grid");
    AxisWeights result;
    result.Index.reserve(size_t(width));
    result.Weight.reserve(size_t(width));
    for (int i = 0; i < width; ++i) {
        const int xi = first + i;
        double weight = 1.;
        for (int j = 0; j < width; ++j) {
            if (i == j) continue;
            const int xj = first + j;
            weight *= (coordinate - xj) / double(xi - xj);
        }
        result.Index.push_back(xi);
        result.Weight.push_back(weight);
    }
    return result;
}

} // namespace

DeltaStencil LagrangeDelta(const Point &position, const Point &origin, double spacing, const std::array<int, 3> &shape, int order, int staggered_axis) {
    if (!(spacing > 0.)) throw std::runtime_error("Delta grid spacing must be positive");
    AxisWeights axes[3];
    for (int d = 0; d < 3; ++d) {
        const double offset = d == staggered_axis ? .5 : 0.;
        axes[d] = LagrangeAxis((position[d] - origin[d]) / spacing - offset, shape[d], order);
    }

    DeltaStencil result;
    const size_t entries = axes[0].Index.size() * axes[1].Index.size() * axes[2].Index.size();
    result.Index.reserve(entries);
    result.Weight.reserve(entries);
    for (size_t z = 0; z < axes[2].Index.size(); ++z) {
        for (size_t y = 0; y < axes[1].Index.size(); ++y) {
            for (size_t x = 0; x < axes[0].Index.size(); ++x) {
                result.Index.push_back(axes[0].Index[x] + shape[0] * (axes[1].Index[y] + shape[1] * axes[2].Index[z]));
                result.Weight.push_back(axes[0].Weight[x] * axes[1].Weight[y] * axes[2].Weight[z]);
            }
        }
    }
    return result;
}

double Gather(const std::vector<double> &field, const DeltaStencil &stencil) {
    double result = 0.;
    for (size_t i = 0; i < stencil.Index.size(); ++i) result += stencil.Weight[i] * field[size_t(stencil.Index[i])];
    return result;
}

} // namespace immersed
