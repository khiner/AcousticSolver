#pragma once

#include "PorousMaterial.h"

namespace Porous {

struct SlabResult {
    double Spacing{}, Frequency{}, Angle{}, Thickness{}, AirGap{};
    int Nodes{}, Steps{};
    double Seconds{}, Stationarity{}, SpatialResidual{}, Condition{};
    Complex Reflection{}, Transmission{}, ExpectedReflection{}, ExpectedTransmission{};
    double ReflectionError{}, TransmissionError{};
};

// A single transverse Neumann mode represents equal +angle/-angle plane waves,
// which have identical scattering at a planar isotropic slab. Existing mirror
// boundary stencils preserve that mode exactly, avoiding large finite apertures.
SlabResult RunSlab(double spacing, bool oblique, bool rigid_backing, bool gpu, double frequency, double width, double cycles);

} // namespace Porous
