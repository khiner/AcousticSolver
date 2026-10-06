// Sections 2–4 of Russo, Ducceschi & Bilbao (2026), equations 14, 16 and 21.
// Standalone reproduction runner; see PaperOde.py for provenance and figures.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Pair {
    double U{}, V{};
};
Pair operator+(Pair a, Pair b) { return {a.U + b.U, a.V + b.V}; }
Pair operator-(Pair a, Pair b) { return {a.U - b.U, a.V - b.V}; }
Pair operator*(double a, Pair b) { return {a * b.U, a * b.V}; }
double Dot(Pair a, Pair b) { return a.U * b.U + a.V * b.V; }
enum class Scheme { Full,
                    A,
                    B };
const char *Name(Scheme scheme) { return scheme == Scheme::Full ? "V" : scheme == Scheme::A ? "RA" :
                                                                                              "RB"; }
struct Potential {
    double Value;
    Pair Gradient;
};
Potential Evaluate(Pair w, double alpha, Scheme scheme) {
    const double radius = std::sqrt(w.U * w.U + (1 + w.V) * (1 + w.V));
    // Algebraically equivalent to radius-1, with less cancellation near rest.
    const double extension = (w.U * w.U + w.V * (2 + w.V)) / (radius + 1);
    if (scheme == Scheme::B)
        return {(alpha - 1) * (.5 * w.U * w.U + .5 + w.V - extension), {(alpha - 1) * w.U * extension / radius, (alpha - 1) * (w.U * w.U / (radius + 1 + w.V)) / radius}};
    Potential result{.5 * (alpha - 1) * extension * extension, {(alpha - 1) * extension * w.U / radius, (alpha - 1) * extension * (1 + w.V) / radius}};
    if (scheme == Scheme::Full) {
        result.Value += .5 * Dot(w, w);
        result.Gradient = result.Gradient + w;
    }
    return result;
}
Pair Quadratic(Scheme scheme, double alpha) {
    return scheme == Scheme::Full ? Pair{} : Pair{1, scheme == Scheme::B ? alpha : 1};
}
double Psi(Pair w, double alpha, Scheme scheme, double epsilon) {
    return std::sqrt(2 * Evaluate(w, alpha, scheme).Value + epsilon);
}
double Energy(Pair previous, Pair current, Pair increment, double psi, double dt, Pair quadratic) {
    return .5 * Dot(increment, increment) / (dt * dt) + .5 * (quadratic.U * previous.U * current.U + quadratic.V * previous.V * current.V + psi * psi);
}

void CheckPotentialIdentities(double alpha) {
    for (const auto w : {Pair{.5, .2}, Pair{-.3, -.2}, Pair{.01, .002}}) {
        const auto full = Evaluate(w, alpha, Scheme::Full);
        const auto residual = Evaluate(w, alpha, Scheme::B);
        const double reconstructed = .5 * (w.U * w.U + alpha * w.V * w.V) + residual.Value - .5 * (alpha - 1);
        const auto reconstructedGradient = residual.Gradient + Pair{w.U, alpha * w.V};
        if (std::abs(full.Value - reconstructed) > 1e-10 * alpha ||
            std::abs(full.Gradient.U - reconstructedGradient.U) > 1e-10 * alpha ||
            std::abs(full.Gradient.V - reconstructedGradient.V) > 1e-10 * alpha)
            throw std::runtime_error("ODE potential decompositions disagree");
    }
}

struct Reference {
    std::map<std::uint64_t, Pair> States;
    double Energy{}, RelativeEnergyDrift{}, Seconds{}, MinV{.2};
};
Reference Verlet(double alpha, std::uint64_t rate, const std::vector<double> &times, int maxExponent) {
    Reference result;
    std::vector<std::uint64_t> indices;
    for (double const time : times) {
        const auto index = std::uint64_t(std::llround(time * rate));
        indices.push_back(index);
        for (int exponent = 1; exponent <= maxExponent; ++exponent)
            indices.push_back(index - rate / (2 * (10000ULL << exponent)));
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    Pair w{.5, .2}, velocity{}, gradient = Evaluate(w, alpha, Scheme::Full).Gradient;
    result.Energy = Evaluate(w, alpha, Scheme::Full).Value;
    const double dt = 1. / rate;
    const auto start = Clock::now();
    std::size_t target = 0;
    // Kick–drift–kick Störmer–Verlet avoids a repeated subtraction of positions.
    for (std::uint64_t n = 1; n <= indices.back(); ++n) {
        velocity = velocity - (.5 * dt) * gradient;
        w = w + dt * velocity;
        const auto potential = Evaluate(w, alpha, Scheme::Full);
        gradient = potential.Gradient;
        velocity = velocity - (.5 * dt) * gradient;
        const double energy = .5 * Dot(velocity, velocity) + potential.Value;
        result.RelativeEnergyDrift = std::max(result.RelativeEnergyDrift, std::abs(energy - result.Energy) / result.Energy);
        result.MinV = std::min(result.MinV, w.V);
        if (n == indices[target]) {
            result.States.emplace(n, w);
            if (++target == indices.size()) break;
        }
    }
    result.Seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

void Sav(std::ostream &output, double alpha, int exponent, Scheme scheme, double epsilon, std::uint64_t referenceRate, const Reference &reference, const std::vector<double> &times) {
    const std::uint64_t rate = 10000ULL << exponent;
    const double dt = 1. / rate, quarter = dt * dt / 4;
    const auto quadratic = Quadratic(scheme, alpha);
    Pair previous{.5, .2};
    Pair increment = (-.5 * dt * dt) * Evaluate(previous, alpha, Scheme::Full).Gradient;
    Pair current = previous + increment;
    double psi = Psi(.5 * (previous + current), alpha, scheme, epsilon);
    const double initialEnergy = Energy(previous, current, increment, psi, dt, quadratic);
    double drift = 0, minV = std::min(previous.V, current.V);
    const auto start = Clock::now();
    std::size_t target = 0;
    const auto finalStep = std::uint64_t(std::llround(times.back() * rate));
    for (std::uint64_t n = 1; n < finalStep; ++n) {
        const auto potential = Evaluate(current, alpha, scheme);
        const auto g = (1 / std::sqrt(2 * potential.Value + epsilon)) * potential.Gradient;
        // Solve equation 21 for the change in position increment. This avoids
        // rounding 2-k²K to nearly 2 and subtracting nearly equal positions.
        const Pair linear{quadratic.U * current.U, quadratic.V * current.V};
        const auto b = (-dt * dt) * (linear + (psi + .5 * Dot(g, increment)) * g);
        const auto change = b - (quarter * Dot(g, b) / (1 + quarter * Dot(g, g))) * g;
        psi += Dot(g, increment) + .5 * Dot(g, change);
        increment = increment + change;
        const auto next = current + increment;
        previous = current;
        current = next;
        const double energy = Energy(previous, current, increment, psi, dt, quadratic);
        if (!std::isfinite(energy) || !std::isfinite(current.U) || !std::isfinite(current.V))
            throw std::runtime_error("Nonfinite SAV trajectory");
        drift = std::max(drift, std::abs(energy - initialEnergy) / std::abs(initialEnergy));
        minV = std::min(minV, current.V);
        if (n + 1 == std::uint64_t(std::llround(times[target] * rate))) {
            const auto index = std::uint64_t(std::llround(times[target] * referenceRate));
            const auto expected = reference.States.at(index);
            const auto midpoint = reference.States.at(index - referenceRate / (2 * rate));
            const double expectedPsi = Psi(midpoint, alpha, scheme, epsilon);
            output << Name(scheme) << ',' << epsilon << ',' << exponent << ',' << rate << ',' << times[target]
                   << ',' << current.U << ',' << current.V << ',' << psi
                   << ',' << std::abs(current.U - expected.U) << ',' << std::abs(current.V - expected.V)
                   << ',' << std::abs(psi - expectedPsi) << ',' << initialEnergy << ',' << energy << ',' << drift
                   << ',' << expected.U << ',' << expected.V << ',' << expectedPsi << ',' << minV
                   << ',' << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
            if (++target == times.size()) break;
        }
    }
}
} // namespace

int main(int argc, char *const *argv) {
    try {
        std::filesystem::path output = "build/string-paper/ode";
        int maxExponent = 8, referenceExponent = 10;
        double duration = 5, alpha = 1e5;
        bool referenceOnly = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--reference-only") {
                referenceOnly = true;
                continue;
            }
            if (i + 1 >= argc) throw std::invalid_argument("Missing option value");
            if (argument == "--output") output = argv[++i];
            else if (argument == "--max-exponent") maxExponent = std::stoi(argv[++i]);
            else if (argument == "--reference-exponent") referenceExponent = std::stoi(argv[++i]);
            else if (argument == "--duration") duration = std::stod(argv[++i]);
            else if (argument == "--alpha") alpha = std::stod(argv[++i]);
            else throw std::invalid_argument("Unknown argument: " + argument);
        }
        if (maxExponent < 1 || maxExponent > 12 || referenceExponent <= maxExponent || referenceExponent > 16 || !std::isfinite(alpha) || alpha <= 1)
            throw std::invalid_argument("Invalid sample exponent or alpha");
        CheckPotentialIdentities(alpha);
        std::vector<double> times;
        for (double const time : {.05, .5, 5.})
            if (time <= duration) times.push_back(time);
        if (times.empty() || !std::isfinite(duration)) throw std::invalid_argument("Duration must include a published endpoint: .05, .5 or 5 s");
        std::filesystem::create_directories(output);
        const auto referenceRate = 10000ULL << referenceExponent;
        const auto reference = Verlet(alpha, referenceRate, times, maxExponent);
        std::ofstream metadata(output / "reference.json");
        metadata << std::setprecision(17) << "{\"sample_rate\":" << referenceRate << ",\"exponent\":" << referenceExponent
                 << ",\"alpha\":" << alpha << ",\"initial_energy\":" << reference.Energy << ",\"maximum_relative_energy_drift\":"
                 << reference.RelativeEnergyDrift << ",\"min_v\":" << reference.MinV << ",\"seconds\":" << reference.Seconds << "}\n";
        std::ofstream states(output / "reference.csv");
        states << std::setprecision(17) << "step,time,u,v\n";
        for (const auto &[index, state] : reference.States)
            states << index << ',' << double(index) / referenceRate << ',' << state.U << ',' << state.V << '\n';
        if (referenceOnly) return metadata && states ? 0 : 1;
        std::ofstream results(output / "results.csv");
        results << std::setprecision(17) << "scheme,epsilon,exponent,sample_rate,time,u,v,psi,u_error,v_error,psi_error,energy_initial,energy_final,energy_max_relative_drift,reference_u,reference_v,reference_psi,min_v,seconds\n";
        for (const auto scheme : {Scheme::Full, Scheme::A, Scheme::B})
            for (const double epsilon : {std::numeric_limits<double>::epsilon(), 1000.})
                for (int exponent = 1; exponent <= maxExponent; ++exponent) {
                    Sav(results, alpha, exponent, scheme, epsilon, referenceRate, reference, times);
                    std::cout << Name(scheme) << " epsilon=" << epsilon << " a=" << exponent << " complete\n";
                }
        if (!metadata || !states || !results) throw std::runtime_error("Failed to write ODE outputs");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "PaperOde: " << error.what() << '\n';
        return 1;
    }
}
