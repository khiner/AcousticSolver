#pragma once

#include <complex>
#include <span>
#include <vector>

namespace Porous {

using Complex = std::complex<double>;

// Smits (DAFx 2023), Eqs. (24)-(26), with s = i*2*pi*f.
struct Rational {
    double Constant{};
    struct Pole {
        double Rate, Residue;
    };
    std::vector<Pole> Poles;
    Complex Evaluate(Complex s) const;
};

struct Fluid {
    Complex Density;
    Complex BulkModulus;
};

struct Material {
    double FlowResistivity{};
    double MinFrequency{20};
    double MaxFrequency{4000};
    double MaxRelativeError{};
    Rational BulkModulus;
    Rational InverseDensity;
    Rational WaveSpeedSquared;
};

// Ambient constants and empirical coefficients match paper Eqs. (22)-(23).
Fluid AllardChampoux(double flow_resistivity, Complex s);
Material AirMaterial(double speed, double density = 1.2);
Material PrepareMaterial(double flow_resistivity, double min_frequency = 20, double max_frequency = 4000, double tolerance = 0.0032);

// Density is averaged across the link before inversion.
Rational PrepareLinkInverseDensity(const Material &a, const Material &b, double air_density = 1.2, double tolerance = 0.0032);

struct Layer {
    double Thickness{};
    double FlowResistivity{}; // zero denotes air
};

struct Scattering {
    Complex Reflection{};
    Complex Transmission{}; // pressure at rear face / incident pressure at front
    double Absorption{};
};

// Independent continuum transfer matrices. Air is present on both sides unless
// rigid_backing is true. The incidence angle is in radians; +i*omega*t convention.
Scattering LayerResponse(std::span<const Layer> layers, double frequency, double angle_radians = 0, bool rigid_backing = true, double air_speed = 343, double air_density = 1.2);

} // namespace Porous
