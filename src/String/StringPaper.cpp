#include "String.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace {
using namespace stiffstring;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
} // namespace

int main(int argc, char *const *argv) try {
    if (argc != 2) throw std::invalid_argument("Usage: StringPaper job.json");
    std::ifstream input(argv[1]);
    Json job;
    input >> job;
    auto c = Preset(job.at("variant").get<std::string>());
    const int exponent = job.at("exponent").get<int>();
    if (exponent < 0 || exponent > 10) throw std::invalid_argument("Paper exponent must lie in [0,10]");
    c.SampleRate = std::ldexp(44100., exponent);
    c.Duration = job.value("seconds", .06);
    const auto initial = job.value("initial", "mode"), method = job.value("method", "sav");
    if (initial != "mode" && initial != "raised-cosine") throw std::invalid_argument("Unknown initial condition");
    if (method != "sav" && method != "reference") throw std::invalid_argument("Unknown integrator");
    c.Initial = initial == "mode" ? InitialCondition::FirstMode : InitialCondition::RaisedCosine;
    c.Method = method == "sav" ? Integrator::Sav : Integrator::Reference;
    c.Shift = job.value("shift", std::numeric_limits<double>::epsilon());
    const auto grid = job.value("grid", "max");
    const double area = std::numbers::pi * c.Radius * c.Radius;
    const double mass = c.Density * area, rigidity = c.YoungModulus * std::numbers::pi * std::pow(c.Radius, 4) / 4;
    const double dt = 1 / c.SampleRate, tensionTerm = c.Tension / mass * dt * dt;
    const double hL = std::sqrt(c.YoungModulus / c.Density) * dt;
    const double hT = std::sqrt((tensionTerm + std::sqrt(tensionTerm * tensionTerm + 16 * rigidity / mass * dt * dt)) / 2);
    const bool geometric = c.Equation == Model::GeometricA || c.Equation == Model::GeometricB;
    if (grid != "max" && grid != "longitudinal" && grid != "transverse" && grid != "below") throw std::invalid_argument("Unknown paper grid");
    c.Spacing = geometric && (grid == "longitudinal" || (grid == "max" && hL >= hT)) ? Grid::Longitudinal : Grid::Transverse;
    auto p = Prepare(c);
    if (grid == "below") {
        // Retain the selected timestep while selecting a grid below its transverse bound.
        c.Intervals = 2 * std::size_t(std::ceil(c.Length / (.99 * hT) / 2));
        p = Prepare(c);
        p.Dt = dt;
        p.SampleRate = c.SampleRate;
        p.SampleCount = std::size_t(std::floor(c.Duration * p.SampleRate));
        p.InverseMass = 1 / p.Mass;
    }
    const auto steps = p.SampleCount;
    Json result{{"job", job}, {"intervals", p.Intervals}, {"dt", p.Dt}, {"sample_rate", p.SampleRate}, {"h", p.H}, {"requested_hL", hL}, {"requested_hT", hT}, {"selected_grid", c.Spacing == Grid::Longitudinal ? "longitudinal" : "transverse"}, {"steps", steps}, {"startup", "literal upstream"}, {"clock", "upstream grid adjustment; separate dt and floor(1/dt)"}};
    if (job.value("describe", false)) {
        std::cout << result.dump(2) << '\n';
        return 0;
    }
    Json observations = Json::object(), captures = Json::object();
    const auto targets = job.value("targets", Json::array());
    std::size_t targetIndex{};
    double previousTarget = p.Dt;
    std::vector<std::string> targetKeys;
    for (const auto &target : targets) {
        const double time = target.at("time").get<double>();
        const auto key = target.at("key").get<std::string>();
        if (!std::isfinite(time) || time < previousTarget || time > (steps + 1) * p.Dt || key.empty() || std::find(targetKeys.begin(), targetKeys.end(), key) != targetKeys.end())
            throw std::invalid_argument("Targets require unique keys, ordered finite times and complete simulation coverage");
        previousTarget = time;
        targetKeys.push_back(key);
    }
    auto const potentials = [&](std::span<const double> q) {
        Json values;
        const std::vector<std::string> variants = geometric ? std::vector<std::string>{"ge-a-unsplit", "ge-a-split", "ge-b-split"} : (c.Equation == Model::Cubic ? std::vector<std::string>{"cubic-split", "cubic-unsplit"} : std::vector<std::string>{"kc-split", "kc-unsplit"});
        for (const auto &variant : variants) {
            auto alternate = p;
            const auto settings = Preset(variant);
            alternate.Settings.Equation = settings.Equation;
            alternate.Settings.Split = settings.Split;
            alternate.LongitudinalTension = settings.Equation == Model::GeometricB ? p.EA : c.Tension;
            values[variant] = EvaluatePotential(alternate, q, false).Value;
        }
        return values;
    };
    auto state = InitialState(p);
    const auto start = Clock::now();
    std::size_t completed{};
    try {
        for (; completed < steps; ++completed) {
            Step(p, state, false);
            const auto index = state.StepIndex + 1;
            // Table 2 evaluates MATLAB Out(timeSamples), two entries before Out(end).
            if (index == p.SampleCount - 1) {
                auto pickup = Pickup(p, state.Q);
                observations["u"] = pickup[0];
                observations["v"] = pickup[1];
                observations["displacement_time"] = index * p.Dt;
            }
            // MATLAB Psi(timeSamples) is a half-step later than that displacement.
            if (index == p.SampleCount) {
                observations["psi"] = state.Psi;
                observations["psi_time"] = (index - .5) * p.Dt;
                auto midpoint = state.Q;
                for (std::size_t i = 0; i < midpoint.size(); ++i) midpoint[i] = .5 * (state.Q[i] + state.Previous[i]);
                observations["potentials"] = potentials(midpoint);
            }
            const double time = index * p.Dt;
            while (targetIndex < targets.size() && targets[targetIndex].at("time").get<double>() <= time) {
                const auto &target = targets[targetIndex++];
                const double targetTime = target.at("time").get<double>(), alpha = (targetTime - (time - p.Dt)) / p.Dt;
                if (alpha < -1e-7 || alpha > 1 + 1e-7) throw std::invalid_argument("Reference targets must be ordered and fall inside the simulation");
                auto interpolated = state.Q;
                for (std::size_t i = 0; i < interpolated.size(); ++i) interpolated[i] = (1 - alpha) * state.Previous[i] + alpha * state.Q[i];
                const auto pickup = Pickup(p, interpolated);
                captures[target.at("key").get<std::string>()] = Json{{"time", targetTime}, {"u", pickup[0]}, {"v", pickup[1]}, {"potentials", potentials(interpolated)}};
            }
        }
        result["status"] = "complete";
    } catch (const std::runtime_error &error) {
        result["status"] = "nonfinite";
        result["error"] = error.what();
    }
    if (result["status"] == "complete" && targetIndex != targets.size()) throw std::runtime_error("Incomplete reference target capture");
    result["completed_steps"] = completed;
    result["wall_seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
    result["paper_observation"] = observations;
    result["reference_captures"] = captures;
    std::cout << result.dump(2) << '\n';
    return 0;
} catch (const std::exception &error) {
    std::cerr << "StringPaper: " << error.what() << '\n';
    return 1;
}
