#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Independently implements Zheleznov & Bilbao (2026), arXiv:2608.06139.
namespace plate {

struct Config {
    // Zero cutoff selects the time integrator's stability limit automatically.
    double SampleRate{44100}, CutoffHz{}, Kappa{60}, Ratio{1.1};
    double Sigma0{1.3}, Sigma1{1e-4}, Lambda{1000}, Epsilon{1e-12};
    double Oversampling{1.5};
    bool Nonlinear{true};
    bool UseExact{true};
    int NormType{1};
};

using LossPoints = std::array<std::array<double, 2>, 2>; // {frequency Hz, decay time seconds}
struct Damping {
    double Sigma0{}, Sigma1{};
};
struct PhysicalParameters {
    double Lx{}, Ly{}, Density{}, Thickness{}, YoungModulus{}, PoissonRatio{};
    LossPoints Loss{};
};
struct PhysicalConversion {
    Config Settings;
    double LengthScale{}, ExcitationScale{}, OutputScale{};
};

// Preserves the upstream 6*ln(10) calibration convention.
Damping CalibrateDamping(const LossPoints &loss, double kappa);
// Coordinates in metres divide by LengthScale; force in newtons multiplies by
// ExcitationScale; scaled displacement multiplies by OutputScale to yield metres.
PhysicalConversion FromPhysicalParameters(const PhysicalParameters &physical, Config settings = {});

struct Prepared {
    Config Settings;
    std::size_t Mx{}, My{}, Nx{}, Ny{}, ActiveCount{};
    double Dt{}, TransformScale{};
    // Modal arrays use row-major Mx by My storage, including inactive zeros.
    std::vector<std::uint32_t> Active, AiryActive;
    std::vector<double> Kx, Ky, Omega2, Sigma, D;
    // Orthonormal transforms: modal index first, collocation index second.
    // Sx: Mx by Nx; Cx: (Mx+1) by Nx, and likewise for y.
    std::vector<double> Sx, Sy, Cx, Cy;
};

struct State {
    std::vector<double> Q, Previous;
    double Psi{};
    std::uint64_t StepIndex{};
};

struct Potential {
    double Value{};
    std::vector<double> Gradient;
    // Row-major (Mx+1) by (My+1), with constant and excluded modes zero.
    std::vector<double> Airy;
};

struct Diagnostics {
    // Energy, Psi, potential and drift at the half step BEFORE this update.
    double Energy{}, InputPower{}, DissipatedPower{}, Psi{}, HalfPotential{}, Drift{};
};

Prepared Prepare(const Config &config);
enum class Backend { Cpu,
                     MetalDense,
                     MetalFft };
// Nonlinear runs use Metal; the transform choice estimates relative work.
Backend SelectBackend(const Prepared &prepared);
Potential EvaluatePotential(const Prepared &prepared, std::span<const double> q, bool gradient = true);
State InitialState(const Prepared &prepared, double firstModeAmplitude = 0);
State InitialState(const Prepared &prepared, std::span<const double> q, std::span<const double> previous);
// Coordinates are in the unit-area domain [0,sqrt(Ratio)] x [0,1/sqrt(Ratio)].
void ValidatePoint(const Prepared &prepared, double x, double y);
std::vector<double> PointBasis(const Prepared &prepared, double x, double y);
double Pickup(const Prepared &prepared, const State &state, std::span<const double> basis);
// type=1 is a half cosine; type=2 is raised cosine. Any positive integer is valid.
// The support includes its endpoint, which matters for odd types.
double RaisedCosineStrike(double time, double start, double duration, double amplitude, int type = 2);
// Tiles exactly cycleSamples samples, matching upstream's sampled repetition.
double RepeatedCosineStrike(std::uint64_t sample, std::uint64_t cycleSamples, double sampleRate, double start, double duration, double amplitude, int type = 2);
double Energy(const Prepared &prepared, const State &state);
// Force is a full modal vector (e.g. PointBasis * RaisedCosineStrike).
// An empty span applies zero force. This is the explicit FP64 validation path.
Diagnostics Step(const Prepared &prepared, State &state, std::span<const double> modalForce = {});

} // namespace plate
