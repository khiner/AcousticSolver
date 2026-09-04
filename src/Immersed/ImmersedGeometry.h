#pragma once

#include "ImmersedBoundary.h"

#include <vector>

namespace immersed {

// Equal-area golden-angle spiral patches. The total area is exact and each
// centroid lies on the sphere with its exact outward normal.
std::vector<Patch> SpherePatches(const Point &center, double radius, double target_area, const BoundaryImmittance &zv, const BoundaryImmittance &yp);

// Uniform square patches in the plane spanned by the orthonormal u and v axes.
std::vector<Patch> SquarePatches(const Point &center, const Point &u, const Point &v, double side, double target_area, const BoundaryImmittance &zv, const BoundaryImmittance &yp);

} // namespace immersed
