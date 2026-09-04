#pragma once

#include <array>
#include <vector>

namespace immersed {

using Point = std::array<double, 3>;

struct DeltaStencil {
    std::vector<int> Index;
    std::vector<double> Weight;
};

// Product-form Lagrange delta on a grid whose samples begin at origin and are
// offset by half a cell along the selected staggered velocity axis.
DeltaStencil LagrangeDelta(const Point &position, const Point &origin, double spacing, const std::array<int, 3> &shape, int order, int staggered_axis = -1);

double Gather(const std::vector<double> &field, const DeltaStencil &stencil);

} // namespace immersed
