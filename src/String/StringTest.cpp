#include "String.h"
#include "StringGpu.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {
using namespace stiffstring;
constexpr const char *Variants[]{"ge-a-unsplit", "ge-a-split", "ge-b-split", "cubic-unsplit", "cubic-split", "kc-unsplit", "kc-split"};
int Checks{}, Failures{};

void Check(const std::string &name, double error, double tolerance) {
    ++Checks;
    const bool pass = std::isfinite(error) && error <= tolerance;
    Failures += !pass;
    std::cout << (pass ? "PASS " : "FAIL ") << name << ": " << error << " <= " << tolerance << '\n';
}

double Relative(std::span<const double> actual, std::span<const double> expected, double floor = 1e-14) {
    if (actual.size() != expected.size()) return std::numeric_limits<double>::infinity();
    double scale = floor, error = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        scale = std::max(scale, std::abs(expected[i]));
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    return error / scale;
}

Config Small(const char *name) {
    auto config = Preset(name);
    config.SampleRate = 2 * 44100;
    config.Duration = .002;
    config.Intervals = 32;
    config.Initial = InitialCondition::FirstMode;
    if (config.Equation == Model::GeometricA || config.Equation == Model::GeometricB) config.Spacing = Grid::Longitudinal;
    return config;
}

// Each formulation has a distinct potential; test its gradient and energy once.
void ModelChecks() {
    for (const auto *name : Variants) {
        auto config = Small(name);
        // Unsplit upstream damping is not energy balanced. Exercise damping on split models.
        config.Damped = config.Split;
        config.Shift = std::numeric_limits<double>::epsilon();
        const auto prepared = Prepare(config);
        auto state = InitialState(prepared);
        if (prepared.Components == 2)
            for (std::size_t i = 0; i < prepared.Points; ++i)
                state.Q[prepared.Points + i] = .00003 * std::sin(2 * std::numbers::pi * (i + 1) / prepared.Intervals);
        const auto potential = EvaluatePotential(prepared, state.Q);
        std::vector<double> finiteDifference(state.Q.size());
        for (std::size_t i = 0; i < state.Q.size(); ++i) {
            // Form B's large constant potential needs a larger finite-difference step.
            const double original = state.Q[i], increment = config.Equation == Model::GeometricB ? 1e-6 : 1e-8;
            state.Q[i] = original + increment;
            const double plus = EvaluatePotential(prepared, state.Q, false).Value;
            state.Q[i] = original - increment;
            const double minus = EvaluatePotential(prepared, state.Q, false).Value;
            state.Q[i] = original;
            finiteDifference[i] = (plus - minus) / (2 * increment);
        }
        Check(std::string(name) + "_gradient", Relative(potential.Gradient, finiteDifference), 2e-4);
        state = InitialState(prepared);
        const double energy = Step(prepared, state).Energy;
        double drift = 0;
        for (int i = 0; i < 127; ++i) drift = std::max(drift, std::abs(Step(prepared, state).Energy - energy));
        const double scale = config.Tension * config.Amplitude * config.Amplitude / config.Length;
        Check(std::string(name) + "_energy_balance", drift / scale, std::max(2e-6, 256 * std::numeric_limits<double>::epsilon() * std::abs(energy) / scale));
    }
}

void NonlinearConvergence() {
    const auto evolve = [](Config config, unsigned refinement) {
        auto prepared = Prepare(config);
        constexpr double duration = .02;
        const auto steps = unsigned(std::ceil(duration / prepared.Dt)) * refinement;
        prepared.Dt = duration / steps;
        prepared.InverseMass = 1 / (prepared.Mass * (1 + prepared.Sigma0 * prepared.Dt));
        auto state = InitialState(prepared);
        for (std::size_t i = 0; i < prepared.Points; ++i) {
            const double phase = std::numbers::pi * (i + 1) / prepared.Intervals;
            state.Previous[i] = config.Amplitude * (std::sin(phase) + .3 * std::sin(3 * phase));
        }
        std::fill(state.Previous.begin() + prepared.Points, state.Previous.end(), 0);
        // Consistent zero-velocity startup isolates integrator convergence from
        // the distinct historical startup expressions in the upstream scripts.
        auto full = prepared;
        full.Settings.Split = false;
        const auto potential = EvaluatePotential(full, state.Previous);
        for (std::size_t i = 0; i < state.Q.size(); ++i)
            state.Q[i] = state.Previous[i] - .5 * prepared.Dt * prepared.Dt * potential.Gradient[i] / (prepared.Mass * prepared.H);
        auto midpoint = state.Q;
        for (std::size_t i = 0; i < midpoint.size(); ++i) midpoint[i] = .5 * (state.Q[i] + state.Previous[i]);
        const double shift = config.Equation == Model::KirchhoffCarrier ? 0 : (config.Equation == Model::GeometricB ? config.Shift : 2 * config.Shift);
        state.Psi = std::sqrt(2 * EvaluatePotential(prepared, midpoint, false).Value + shift);
        for (unsigned i = 1; i < steps; ++i) Step(prepared, state);
        return state.Q;
    };
    for (const auto *name : {"ge-b-split", "cubic-split", "kc-split"}) {
        auto config = Small(name);
        config.Shift = std::numeric_limits<double>::epsilon();
        config.Method = Integrator::Reference;
        const auto reference = evolve(config, 128);
        config.Method = Integrator::Sav;
        double errors[3]{};
        for (unsigned level = 0; level < 3; ++level) {
            const auto actual = evolve(config, 8 << level);
            double square = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) square += std::pow(actual[i] - reference[i], 2);
            errors[level] = std::sqrt(square / actual.size()) / config.Amplitude;
        }
        Check(std::string(name) + "_nonlinear_second_order", std::max(errors[1] / errors[0], errors[2] / errors[1]), 1 / 3.4);
        Check(std::string(name) + "_nonlinear_refined_reference", errors[2], 3e-5);
    }
}

// One analytic mode oracle covers CPU stepping and the Metal high-rate regression.
void AnalyticMode(bool metal) {
    for (auto const method : {Integrator::Sav, Integrator::Reference}) {
        auto config = Small("kc-split");
        config.Nonlinear = false;
        config.Method = method;
        auto prepared = Prepare(config);
        if (metal) {
            prepared.Dt = 1 / (44100. * 1024);
            prepared.SampleRate = 1 / prepared.Dt;
            prepared.InverseMass = 1 / prepared.Mass;
        }
        const auto initial = InitialState(prepared);
        const double mu = 4 * std::pow(std::sin(std::numbers::pi / (2 * prepared.Intervals)), 2) / (prepared.H * prepared.H);
        const double omega = std::sqrt((config.Tension * mu + prepared.Rigidity * mu * mu) / prepared.Mass);
        // acos(1 - dt^2*omega^2/2) loses precision at MHz rates.
        const double phase = 2 * std::asin(.5 * prepared.Dt * omega);
        const double a = initial.Previous[prepared.Receiver[0]], first = initial.Q[prepared.Receiver[0]];
        const double b = ((first - a) + 2 * a * std::pow(std::sin(.5 * phase), 2)) / std::sin(phase);
        std::vector<double> samples;
        if (metal) samples = Gpu(prepared).Run(270000).Samples;
        else {
            auto state = initial;
            for (int i = 0; i < 128; ++i) {
                Step(prepared, state);
                samples.push_back(state.Q[prepared.Receiver[0]]);
            }
        }
        double maximum = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            const double angle = (i + 2) * phase;
            maximum = std::max(maximum, std::abs(samples[i] - a * std::cos(angle) - b * std::sin(angle)));
        }
        Check(std::string(metal ? "metal_high_rate" : "cpu_linear") + (method == Integrator::Sav ? "_sav" : "_reference"), maximum / std::abs(a), metal ? 1e-9 : 2e-10);
    }
}

void MetalChecks() {
    for (const auto *name : Variants)
        for (auto const method : {Integrator::Sav, Integrator::Reference}) {
            auto config = Small(name);
            config.Method = method;
            // Reference stepping has three distinct models; split/unsplit share that update.
            if (method == Integrator::Reference && (!config.Split || config.Equation == Model::GeometricA)) continue;
            if (method == Integrator::Reference && config.Equation == Model::Cubic) {
                // A non-power-of-two grid spans multiple worker strides in the tridiagonal solve.
                config.Intervals = 524;
                config.Damped = true;
            }
            const auto prepared = Prepare(config);
            auto expected = InitialState(prepared);
            for (int i = 0; i < 64; ++i) Step(prepared, expected);
            const std::string label = std::string(name) + (method == Integrator::Sav ? "_sav" : "_reference");
            Gpu gpu(prepared);
            const auto all = gpu.Run(64, true);
            const auto actual = gpu.GetState();
            Check(label + "_metal_cpu", std::max({Relative(actual.Q, expected.Q), std::abs(actual.Psi - expected.Psi) / std::max(1e-10, std::abs(expected.Psi)), std::abs(actual.Loss - expected.Loss) / std::max(1e-12, std::abs(expected.Loss))}), 2e-8);
            if (std::string_view(name) != "kc-split") continue;
            gpu.Reset(InitialState(prepared));
            auto first = gpu.Run(17, true), second = gpu.Run(47, true);
            first.States.insert(first.States.end(), second.States.begin(), second.States.end());
            Check(label + "_batching", Relative(first.States, all.States), 0);
            if (method == Integrator::Sav) {
                config.Amplitude = 0;
                gpu.Reset(InitialState(Prepare(config)));
                gpu.Run(8);
                Check("metal_silence", Relative(gpu.GetState().Q, std::vector<double>(expected.Q.size())), 0);
            }
        }
}
} // namespace

int main(int argc, char *const *argv) try {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--metal"))
        throw std::invalid_argument("Usage: StringTest [--metal]");
    ModelChecks();
    NonlinearConvergence();
    AnalyticMode(false);
    if (argc == 2) {
        MetalChecks();
        AnalyticMode(true);
    }
    std::cout << Checks << " checks, " << Failures << " failures\n";
    return Failures ? 1 : 0;
} catch (const std::exception &error) {
    std::cerr << "StringTest: " << error.what() << '\n';
    return 1;
}
