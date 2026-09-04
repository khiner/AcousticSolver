#pragma once

#include "ImmersedDelta.h"

#include <complex>

namespace immersed {

// Transfer from volume velocity to pressure for a free-space monopole, using
// the same e^(i omega t) convention as the sphere expansion.
std::complex<double> FreeMonopoleTransfer(double angular_frequency, double sound_speed, double density, double distance);

// The free-space Green function reconstructed through its spherical series.
// This is an independent convention check for SphereTransfer.
std::complex<double> FreeSphereSeriesTransfer(double angular_frequency, double sound_speed, double density, const Point &source, const Point &receiver, int max_order = -1);

// Point-source transfer outside a sphere satisfying p + Z0 alpha v_n = 0.
// Infinite alpha is the rigid limit and alpha = 0 is pressure release.
std::complex<double> SphereTransfer(double angular_frequency, double sound_speed, double density, double radius, const Point &source, const Point &receiver, double alpha, int max_order = -1);

// Maximum modal boundary residual normalized by the larger term, over orders
// used by SphereTransfer. derivative=true checks the rigid radial derivative.
double SphereBoundaryResidual(double angular_frequency, double sound_speed, double radius, double source_radius, double alpha, bool derivative, int max_order = -1);

} // namespace immersed
