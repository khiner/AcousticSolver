#pragma once

#include "ImmersedDelta.h"
#include "ImmersedImmittance.h"

#include <utility>

namespace immersed {

struct BoundaryImmittance {
    RationalImmittance Value{RationalImmittance::Constant(0.)};
    bool Infinite{false};

    static BoundaryImmittance Finite(double value) { return Finite(RationalImmittance::Constant(value)); }
    static BoundaryImmittance Finite(RationalImmittance value) { return {std::move(value), false}; }
    static BoundaryImmittance Limit() { return {RationalImmittance::Constant(0.), true}; }
};

struct Patch {
    Point Center{};
    Point Normal{};
    double Area{0.};
    BoundaryImmittance Zv;
    BoundaryImmittance Yp;
};

} // namespace immersed
