#include "AudioFile.h"
#include "String.h"
#include "StringGpu.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <sys/resource.h>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace stiffstring;
constexpr auto Usage = "StringSolve config.json [--cpu | --metal] [--seconds N] [--output PREFIX] [--no-wav] [--states]";

Config Parse(const Json &j, Json &resolved) {
    if (!j.is_object()) throw std::invalid_argument("Configuration must be a JSON object");
    auto c = Preset(j.value("variant", "ge-b-split"));
    const auto method = j.value("integrator", "sav"), initial = j.value("initial_condition", c.Initial == InitialCondition::FirstMode ? "mode" : "raised-cosine");
    if (method != "sav" && method != "reference") throw std::invalid_argument("integrator must be sav or reference");
    if (initial != "mode" && initial != "raised-cosine") throw std::invalid_argument("initial_condition must be mode or raised-cosine");
    c.Method = method == "sav" ? Integrator::Sav : Integrator::Reference;
    c.Initial = initial == "mode" ? InitialCondition::FirstMode : InitialCondition::RaisedCosine;
    c.Spacing = j.value("longitudinal_grid", c.Spacing == Grid::Longitudinal) ? Grid::Longitudinal : Grid::Transverse;
    if (j.contains("oversampling") && j.contains("sample_rate")) throw std::invalid_argument("Choose oversampling or sample_rate");
    c.SampleRate = j.contains("oversampling") ? 44100 * j.at("oversampling").get<double>() : j.value("sample_rate", c.SampleRate);
    const auto parameter = [&](const char *key, auto &value) {
        value = j.value(key, value);
        resolved[key] = value;
    };
    parameter("seconds", c.Duration);
    parameter("initial_amplitude", c.Amplitude);
    parameter("damping", c.Damped);
    parameter("nonlinear", c.Nonlinear);
    parameter("potential_shift", c.Shift);
    parameter("density", c.Density);
    parameter("tension", c.Tension);
    parameter("radius", c.Radius);
    parameter("young_modulus", c.YoungModulus);
    parameter("length", c.Length);
    parameter("stability_margin", c.StabilityMargin);
    parameter("initial_width", c.RaisedCosineWidth);
    if (j.contains("intervals")) {
        if (!j["intervals"].is_number_integer() || j["intervals"].get<double>() < 0 || j["intervals"].get<double>() > 1000000)
            throw std::invalid_argument("intervals must be a nonnegative integer no larger than 1000000");
        c.Intervals = j["intervals"].get<size_t>();
    }
    if (j.contains("loss")) {
        const auto loss = j.at("loss").get<std::array<std::array<double, 2>, 2>>();
        c.LossFrequencyLow = loss[0][0];
        c.T60Low = loss[0][1];
        c.LossFrequencyHigh = loss[1][0];
        c.T60High = loss[1][1];
    }
    resolved.update({{"variant", j.value("variant", "ge-b-split")}, {"integrator", method}, {"initial_condition", initial}, {"sample_rate", c.SampleRate}, {"intervals", c.Intervals}, {"longitudinal_grid", c.Spacing == Grid::Longitudinal}, {"loss", {{c.LossFrequencyLow, c.T60Low}, {c.LossFrequencyHigh, c.T60High}}}});
    for (const auto &[key, value] : j.items())
        if (!resolved.contains(key) && key != "oversampling") throw std::invalid_argument("Unknown parameter: " + key);
    return c;
}

void Write(const std::string &name, const std::vector<double> &values) {
    if (!std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); })) throw std::runtime_error("Nonfinite output: " + name);
    std::ofstream stream(name, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(double)));
    if (!stream) throw std::runtime_error("Cannot write " + name);
}

// A 48 kHz listening copy, with a Blackman-windowed low-pass sinc for decimation.
// Raw displacement exports retain the original sample clock and physical scale.
double WriteWav(const std::string &path, const std::vector<double> &samples, size_t channels, double sourceRate) {
    constexpr double rate = 48000, pi = std::numbers::pi;
    const size_t frames = samples.size() / channels;
    const size_t count = size_t(std::ceil(frames * rate / sourceRate));
    if (count > size_t(std::numeric_limits<int>::max()) || count * channels > (UINT32_MAX - 100) / 4)
        throw std::invalid_argument("Listening WAV exceeds RIFF size limits; use --no-wav");
    std::vector<double> filtered(count * channels);
    const double ratio = sourceRate / rate, cutoff = .45 / std::max(1., ratio);
    const double filterRadius = std::ceil(16 * std::max(1., ratio));
    if (!std::isfinite(filterRadius) || filterRadius > std::numeric_limits<int>::max())
        throw std::invalid_argument("Sample rate exceeds WAV resampler limits; use --no-wav");
    const int radius = int(filterRadius);
    double peak = 0;
    for (size_t n = 0; n < count; ++n) {
        const double position = n * ratio;
        const auto center = int64_t(std::floor(position));
        std::array<double, 2> sum{};
        double norm = 0;
        for (int64_t i = std::max(int64_t(0), center - radius); i <= std::min(int64_t(frames - 1), center + radius); ++i) {
            const double delta = i - position, phase = 2 * pi * cutoff * delta;
            const double weight = (phase == 0 ? 2 * cutoff : std::sin(phase) / (pi * delta)) * (.42 + .5 * std::cos(pi * delta / radius) + .08 * std::cos(2 * pi * delta / radius));
            norm += weight;
            for (size_t c = 0; c < channels; ++c) sum[c] += weight * samples[size_t(i) * channels + c];
        }
        for (size_t c = 0; c < channels; ++c) {
            filtered[n * channels + c] = sum[c] / norm;
            peak = std::max(peak, std::abs(filtered[n * channels + c]));
        }
    }
    const double gain = peak ? .95 / std::max(peak, std::numeric_limits<double>::min()) : 1;
    AudioFile<float> audio;
    audio.setNumChannels(int(channels));
    audio.setNumSamplesPerChannel(int(count));
    audio.setSampleRate(uint32_t(rate));
    audio.setBitDepth(32);
    for (size_t n = 0; n < count; ++n)
        for (size_t c = 0; c < channels; ++c) audio.samples[c][n] = float(filtered[n * channels + c] * gain);
    if (!audio.save(path)) throw std::runtime_error("Cannot write " + path);
    return gain;
}

} // namespace

int main(int argc, char *const *argv) try {
    bool cpu = false, metal = false, wav = true, recordStates = false;
    double duration = -1;
    std::string filename, prefix;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << Usage << '\n';
            return 0;
        }
        if (arg == "--cpu") cpu = true;
        else if (arg == "--metal") metal = true;
        else if (arg == "--no-wav") wav = false;
        else if (arg == "--states") recordStates = true;
        else if (arg == "--output" && i + 1 < argc) prefix = argv[++i];
        else if (arg == "--seconds" && i + 1 < argc) {
            size_t used{};
            const std::string value = argv[++i];
            duration = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(duration) || duration <= 0) throw std::invalid_argument("--seconds must be positive and finite");
        } else if (!arg.starts_with('-') && filename.empty()) filename = arg;
        else throw std::invalid_argument("Unexpected argument: " + arg);
    }
    if (filename.empty()) throw std::invalid_argument(Usage);
    if (cpu && metal) throw std::invalid_argument("--cpu and --metal are mutually exclusive");
    const bool automatic = !cpu && !metal;
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot read " + filename);
    Json config;
    input >> config;
    if (duration > 0) config["seconds"] = duration;
    const auto setupStart = Clock::now();
    Json resolved;
    const auto p = Prepare(Parse(config, resolved));
    const size_t dofs = p.Points * p.Components, frames = p.SampleCount + 2;
    if (frames < p.SampleCount || frames > std::numeric_limits<size_t>::max() / (sizeof(double) * std::max(size_t(6), dofs)))
        throw std::invalid_argument("Requested output exceeds addressable size");
    auto state = InitialState(p);
    std::unique_ptr<Gpu> gpu;
    if (!cpu) gpu = std::make_unique<Gpu>(p);
    const double setupSeconds = std::chrono::duration<double>(Clock::now() - setupStart).count();
    std::vector<double> samples, diagnostics, states;
    samples.reserve(frames * p.Components);
    diagnostics.reserve(p.SampleCount * 6);
    if (recordStates) states.reserve(frames * dofs);
    const auto capture = [&](std::span<const double> q) {
        const auto pickup = Pickup(p, q);
        samples.insert(samples.end(), pickup.begin(), pickup.begin() + p.Components);
        if (recordStates) states.insert(states.end(), q.begin(), q.end());
    };
    capture(state.Previous);
    capture(state.Q);
    const auto started = Clock::now();
    for (size_t offset = 0; offset < p.SampleCount; offset += 256) {
        const auto count = uint32_t(std::min(size_t(256), p.SampleCount - offset));
        std::vector<Diagnostics> metrics;
        if (gpu) {
            auto trace = gpu->Run(count, recordStates);
            samples.insert(samples.end(), trace.Samples.begin(), trace.Samples.end());
            states.insert(states.end(), trace.States.begin(), trace.States.end());
            metrics = std::move(trace.Metrics);
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                metrics.push_back(Step(p, state));
                capture(state.Q);
            }
        }
        for (const auto &d : metrics) diagnostics.insert(diagnostics.end(), {d.Kinetic, d.LinearPotential, d.NonlinearPotential, d.Dissipated, d.Energy, d.Psi});
    }
    if (gpu) state = gpu->GetState();
    const double simulationSeconds = std::chrono::duration<double>(Clock::now() - started).count();
    if (prefix.empty()) prefix = "build/string/" + std::filesystem::path(filename).stem().string();
    const auto parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    Write(prefix + ".bin", samples);
    Write(prefix + ".diagnostics.bin", diagnostics);
    if (recordStates) Write(prefix + ".states.bin", states);
    Json metadata{{"config", config}, {"backend", cpu ? "CPU FP64" : "Metal paired-float"}, {"backend_selection", automatic ? "auto" : "explicit"}, {"paper", "https://doi.org/10.1007/s11071-026-12708-0"}, {"sample_count", frames}, {"steps", p.SampleCount}, {"channels", p.Components}, {"components", p.Components}, {"points", p.Points}, {"intervals", p.Intervals}, {"h", p.H}, {"dt", p.Dt}, {"sample_rate", p.SampleRate}, {"physical_sample_rate", 1 / p.Dt}, {"dtype", "little-endian float64"}, {"samples_layout", "frame, component; transverse then longitudinal displacement in metres"}, {"states_layout", "frame, component, interior node"}, {"states_recorded", recordStates}, {"initial_frames", 2}, {"diagnostic_columns", {"kinetic", "linear_potential", "nonlinear_potential", "dissipated", "energy", "psi"}}, {"diagnostic_alignment", "one row per completed update, starting at sample frame 2"}, {"setup_seconds", setupSeconds}, {"simulation_seconds", simulationSeconds}, {"real_time_factor", simulationSeconds / (p.SampleCount * p.Dt)}, {"final_state", {{"q", state.Q}, {"previous", state.Previous}, {"psi", state.Psi}, {"loss", state.Loss}, {"step_index", state.StepIndex}}}};
    metadata["resolved_config"] = std::move(resolved);
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("Cannot read process resource usage");
    metadata["peak_process_resident_bytes"] = usage.ru_maxrss;
    if (wav) {
        metadata["wav_gain"] = WriteWav(prefix + ".wav", samples, p.Components, 1 / p.Dt);
        metadata["wav_sample_rate"] = 48000;
    }
    std::ofstream output(prefix + ".json");
    output << metadata.dump(2) << '\n';
    if (!output) throw std::runtime_error("Cannot write metadata");
    std::cout << metadata["backend"] << ": " << p.Points << " interior nodes, " << frames << " frames, simulation " << simulationSeconds << " s\n"
              << prefix << ".json\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "StringSolve: " << e.what() << '\n';
    return 1;
}
