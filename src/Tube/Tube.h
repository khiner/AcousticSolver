#pragma once
#include <cstdint>
#include <json.hpp>
#include <vector>

namespace tube {
struct Model {
    uint32_t Elements{}, Order{}, Terms{}, Steps{}, Channels{};
    double Dt{};

    std::vector<double> PressureCoefficients, FlowCoefficients, Gradient, Ends, Input, Pickups, Positions;
    nlohmann::json Metadata;
};
struct Trace {
    std::vector<double> Samples, Energy, State;
    double Seconds{};
};
Model Prepare(const nlohmann::json &config);
Trace Run(const Model &model);
} // namespace tube
