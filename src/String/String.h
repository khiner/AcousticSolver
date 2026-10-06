#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace stiffstring {

enum class Model { GeometricA,
                   GeometricB,
                   Cubic,
                   KirchhoffCarrier };
enum class Integrator { Sav,
                        Reference };
enum class InitialCondition { FirstMode,
                              RaisedCosine };
enum class Grid { Transverse,
                  Longitudinal };

struct Config {
    Model Equation{Model::GeometricB};
    Integrator Method{Integrator::Sav};
    InitialCondition Initial{InitialCondition::RaisedCosine};
    Grid Spacing{Grid::Longitudinal};
    bool Split{true}, Damped{false}, Nonlinear{true};
    double SampleRate{16 * 44100.0}, Duration{0.06}, Amplitude{0.002};
    double Density{8050}, Tension{75}, Radius{3.556e-4}, YoungModulus{174e9}, Length{1};
    double Shift{1000}, StabilityMargin{0.01};
    double LossFrequencyLow{0}, LossFrequencyHigh{1000}, T60Low{15}, T60High{10};
    // Zero selects the upstream grid; an explicit grid still recomputes its stable timestep.
    std::size_t Intervals{};
    double RaisedCosineWidth{0.25};
};

struct Prepared {
    Config Settings;
    std::size_t Intervals{}, Points{}, Components{}, SampleCount{};
    std::array<std::size_t, 2> Receiver{}; // zero-based interior indices
    double H{}, Dt{}, SampleRate{}, Area{}, Mass{}, Rigidity{}, EA{}, NonlinearModulus{};
    double Sigma0{}, Sigma1{}, LongitudinalTension{};
    double InverseMass{};
};

struct State {
    // Transverse interior nodes followed by longitudinal nodes for geometric models.
    std::vector<double> Q, Previous;
    double Psi{}, Loss{};
    std::uint64_t StepIndex{};
};

struct Diagnostics {
    double Kinetic{}, LinearPotential{}, NonlinearPotential{}, Dissipated{}, Energy{}, Psi{};
};

struct Potential {
    double Value{};
    std::vector<double> Gradient;
};

Config Preset(std::string_view name);
Prepared Prepare(const Config &config);
State InitialState(const Prepared &prepared);
Potential EvaluatePotential(const Prepared &prepared, std::span<const double> q, bool gradient = true);
// diagnostics=false omits energy observations for streaming convergence sweeps;
// state and accumulated damping loss still advance identically.
Diagnostics Step(const Prepared &prepared, State &state, bool diagnostics = true);
std::array<double, 2> Pickup(const Prepared &prepared, std::span<const double> q);

} // namespace stiffstring
