#include "Plate.h"
#include "PlateGpu.h"
#include "PlatePrecision.h"
#include "Profile.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr auto Usage = "Usage: PlateSolve config.json [--cpu | --metal | --fft] [--seconds N] [--output PREFIX] [--no-wav] [--states]";

void KnownKeys(const Json &object, std::initializer_list<const char *> keys) {
    if (!object.is_object()) throw std::invalid_argument("Expected a JSON object");
    for (const auto &[key, value] : object.items()) {
        (void)value;
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) throw std::invalid_argument("Unknown parameter: " + key);
    }
}

std::array<double, 2> Position(const Json &value) {
    if (!value.is_array() || value.size() != 2) throw std::invalid_argument("Position must contain two coordinates");
    return {value[0].get<double>(), value[1].get<double>()};
}

std::uint32_t Integer(const Json &value, const char *name, std::uint32_t maximum) {
    if (!value.is_number_integer() || value.get<double>() < 1 || value.get<double>() > maximum)
        throw std::invalid_argument(std::string(name) + " must be a positive integer in range");
    return value.get<std::uint32_t>();
}

void WriteDoubles(const std::string &name, const std::vector<double> &values) {
    if (!std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); })) throw std::runtime_error("Nonfinite output: " + name);
    std::ofstream file(name, std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(double)));
    if (!file) throw std::runtime_error("Cannot write " + name);
}

// Listening output has one shared gain across channels. Raw files preserve simulation scale.
double WriteWav(const std::string &name, const std::vector<double> &samples, std::size_t channels, double sampleRate) {
    if (samples.size() > (std::numeric_limits<std::uint32_t>::max() - 36u) / 2u) throw std::invalid_argument("WAV exceeds RIFF size limit");
    const auto bytes = std::uint32_t(samples.size() * 2);
    const auto rate = std::uint32_t(sampleRate);
    std::ofstream file(name, std::ios::binary);
    const auto word = [&file](std::uint32_t n, int count) {
        for (int i = 0; i < count; ++i) file.put(char((n >> (8 * i)) & 255));
    };
    file.write("RIFF", 4);
    word(36 + bytes, 4);
    file.write("WAVEfmt ", 8);
    word(16, 4);
    word(1, 2);
    word(std::uint32_t(channels), 2);
    word(rate, 4);
    word(std::uint32_t(rate * channels * 2), 4);
    word(std::uint32_t(channels * 2), 2);
    word(16, 2);
    file.write("data", 4);
    word(bytes, 4);
    double peak = 0;
    for (const double sample : samples) peak = std::max(peak, std::abs(sample));
    const double gain = peak > 0 ? 0.95 / std::max(peak, std::numeric_limits<double>::min()) : 1;
    for (const double sample : samples) word(std::uint16_t(std::int16_t(std::lround(sample * gain * 32767))), 2);
    if (!file) throw std::runtime_error("Cannot write " + name);
    return gain;
}

struct Strike {
    std::vector<double> Basis;
    double Amplitude{}, Start{}, Duration{};
    int Type{2};
};

int Run(int argc, char *const *argv) {
    bool cpu = false, fft = false, metal = false, wav = true, recordStates = false;
    double secondsOverride = -1;
    std::string configFile, prefix;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << Usage << '\n';
            return 0;
        }
        if (arg == "--cpu") cpu = true;
        else if (arg == "--fft") fft = true;
        else if (arg == "--metal") metal = true;
        else if (arg == "--no-wav") wav = false;
        else if (arg == "--states") recordStates = true;
        else if (arg == "--seconds" && i + 1 < argc) {
            std::size_t used = 0;
            const std::string value = argv[++i];
            secondsOverride = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(secondsOverride) || secondsOverride <= 0) throw std::invalid_argument("--seconds must be positive and finite");
        } else if (arg == "--output" && i + 1 < argc) prefix = argv[++i];
        else if (!arg.starts_with('-') && configFile.empty()) configFile = arg;
        else throw std::invalid_argument("Unexpected argument: " + arg);
    }
    if (configFile.empty()) throw std::invalid_argument(Usage);
    if (int(cpu) + int(fft) + int(metal) > 1) throw std::invalid_argument("--cpu, --metal and --fft are mutually exclusive");
    const bool automatic = !cpu && !fft && !metal;
    std::ifstream input(configFile);
    if (!input) throw std::runtime_error("Cannot open " + configFile);
    Json config;
    input >> config;
    KnownKeys(config, {"sample_rate", "cutoff_hz", "kappa", "ratio", "sigma0", "sigma1", "lambda", "epsilon", "oversampling", "nonlinear", "use_exact", "norm_type", "seconds", "repeat_count", "initial_amplitude", "strikes", "pickups", "physical", "excitation_scale", "output_scale"});
    plate::Config settings;
    settings.SampleRate = config.value("sample_rate", settings.SampleRate);
    settings.CutoffHz = config.contains("cutoff_hz") && !config["cutoff_hz"].is_null() ? config["cutoff_hz"].get<double>() : 0.;
    settings.Kappa = config.value("kappa", settings.Kappa);
    settings.Ratio = config.value("ratio", settings.Ratio);
    settings.Sigma0 = config.value("sigma0", settings.Sigma0);
    settings.Sigma1 = config.value("sigma1", settings.Sigma1);
    settings.Lambda = config.value("lambda", settings.Lambda);
    settings.Epsilon = config.value("epsilon", settings.Epsilon);
    settings.Oversampling = config.value("oversampling", settings.Oversampling);
    settings.Nonlinear = config.value("nonlinear", settings.Nonlinear);
    settings.UseExact = config.value("use_exact", settings.UseExact);
    settings.NormType = int(Integer(config.value("norm_type", Json(1)), "norm_type", 2));
    double lengthScale = 1, excitationScale = config.value("excitation_scale", 1.), outputScale = config.value("output_scale", 1.);
    std::array<double, 2> physicalExtent{};
    const bool physical = config.contains("physical");
    if (physical) {
        for (const char *key : {"kappa", "ratio", "sigma0", "sigma1"})
            if (config.contains(key)) throw std::invalid_argument(std::string("physical conflicts with scaled parameter: ") + key);
        const auto &material = config["physical"];
        KnownKeys(material, {"lx", "ly", "density", "thickness", "young_modulus", "poisson_ratio", "loss"});
        const auto &loss = material.at("loss");
        if (!loss.is_array() || loss.size() != 2 || !loss[0].is_array() || loss[0].size() != 2 || !loss[1].is_array() || loss[1].size() != 2)
            throw std::invalid_argument("physical.loss must contain two [frequency_hz, decay_seconds] pairs");
        const plate::PhysicalParameters parameters{
            material.at("lx").get<double>(), material.at("ly").get<double>(), material.at("density").get<double>(), material.at("thickness").get<double>(), material.at("young_modulus").get<double>(), material.at("poisson_ratio").get<double>(), {Position(loss[0]), Position(loss[1])}
        };
        const auto conversion = plate::FromPhysicalParameters(parameters, settings);
        physicalExtent = {parameters.Lx, parameters.Ly};
        settings = conversion.Settings;
        lengthScale = conversion.LengthScale;
        excitationScale *= conversion.ExcitationScale;
        outputScale *= conversion.OutputScale;
        if (!config.contains("pickups")) throw std::invalid_argument("physical configurations require pickup coordinates in metres");
    }
    if (!std::isfinite(excitationScale) || !std::isfinite(outputScale)) throw std::invalid_argument("Excitation and output scales must be finite");
    const auto scaledPosition = [&](const Json &value) {
        auto xy = Position(value);
        if (physical) {
            const std::array<double, 2> bounds{std::sqrt(settings.Ratio), 1 / std::sqrt(settings.Ratio)};
            for (std::size_t axis = 0; axis < 2; ++axis) {
                if (!std::isfinite(xy[axis]) || xy[axis] < 0 || xy[axis] > physicalExtent[axis]) throw std::invalid_argument("Physical point must lie within the plate dimensions");
                // Preserve a point exactly on an edge through the unit conversion.
                xy[axis] = xy[axis] == physicalExtent[axis] ? bounds[axis] : xy[axis] / lengthScale;
            }
        }
        return xy;
    };
    const double seconds = secondsOverride > 0 ? secondsOverride : config.value("seconds", 1.);
    if (!std::isfinite(seconds) || seconds <= 0) throw std::invalid_argument("seconds must be positive and finite");
    const auto setupStart = Clock::now();
    const auto prepared = plate::Prepare(settings);
    if (automatic) {
        const auto selected = plate::SelectBackend(prepared);
        cpu = selected == plate::Backend::Cpu;
        fft = selected == plate::Backend::MetalFft;
    }
    const double exactSteps = seconds * settings.SampleRate;
    if (!std::isfinite(exactSteps) || exactSteps < 1 || exactSteps > double(std::numeric_limits<std::uint32_t>::max())) throw std::invalid_argument("Duration must contain 1 to 2^32-1 samples");
    const auto cycleSteps = std::size_t(std::floor(exactSteps));
    const auto repeatCount = Integer(config.value("repeat_count", Json(1)), "repeat_count", std::numeric_limits<std::uint32_t>::max());
    if (repeatCount > std::numeric_limits<std::uint32_t>::max() / cycleSteps) throw std::invalid_argument("Repeated duration exceeds 2^32-1 samples");
    const auto steps = cycleSteps * repeatCount;
    const std::size_t modes = prepared.Mx * prepared.My;
    const double initialAmplitude = config.value("initial_amplitude", 0.);
    plate::State state;
    if (cpu) state = plate::InitialState(prepared, initialAmplitude);

    const Json pickupConfig = config.value("pickups", Json::array({{0.64, 0.79}}));
    if (!pickupConfig.is_array() || pickupConfig.empty()) throw std::invalid_argument("At least one pickup is required");
    std::vector<double> pickupWeights;
    std::vector<PlateReal> precisePickups;
    Json scaledPickups = Json::array(), scaledStrikes = Json::array();
    for (const auto &point : pickupConfig) {
        const auto xy = scaledPosition(point);
        if (cpu) {
            auto basis = plate::PointBasis(prepared, xy[0], xy[1]);
            for (auto &weight : basis) weight *= outputScale;
            pickupWeights.insert(pickupWeights.end(), basis.begin(), basis.end());
        } else {
            const auto basis = plate::PrecisePointBasis(prepared, xy[0], xy[1]);
            for (const auto &weight : basis) precisePickups.push_back(plate::Pack(weight * plate::Precise(outputScale)));
        }
        scaledPickups.push_back({xy[0], xy[1]});
    }
    const auto channels = pickupConfig.size();
    if (wav && (channels > 32767 || settings.SampleRate != std::floor(settings.SampleRate) || settings.SampleRate > 192000000 || std::uint64_t(settings.SampleRate) * channels * 2 > std::numeric_limits<std::uint32_t>::max())) throw std::invalid_argument("WAV requires an integer sample rate and valid channel/byte counts; use --no-wav for raw output");
    const Json strikeConfig = config.value("strikes", Json::array());
    if (!strikeConfig.is_array()) throw std::invalid_argument("strikes must be an array");
    std::vector<Strike> strikes;
    std::vector<plate::PreciseStrike> preciseStrikes;
    for (const auto &strike : strikeConfig) {
        KnownKeys(strike, {"position", "amplitude", "start", "duration", "type"});
        const auto xy = scaledPosition(strike.at("position"));
        const double amplitude = strike.at("amplitude").get<double>();
        const double start = strike.value("start", 0.);
        const double duration = strike.at("duration").get<double>();
        const int type = int(Integer(strike.value("type", Json(2)), "strike type", std::numeric_limits<int>::max()));
        if (!std::isfinite(amplitude) || !std::isfinite(start) || start < 0 || !std::isfinite(duration) || duration <= 0) throw std::invalid_argument("Invalid strike amplitude, start, or duration");
        if (!std::isfinite(amplitude * excitationScale)) throw std::invalid_argument("Scaled strike amplitude overflow");
        if (cpu) strikes.push_back({plate::PointBasis(prepared, xy[0], xy[1]), amplitude * excitationScale, start, duration, type});
        else preciseStrikes.push_back({plate::PrecisePointBasis(prepared, xy[0], xy[1]), plate::Precise(amplitude) * plate::Precise(excitationScale), plate::Precise(start), plate::Precise(duration), type});
        scaledStrikes.push_back({{"position", {xy[0], xy[1]}}, {"amplitude", amplitude}, {"start", start}, {"duration", duration}, {"type", type}});
    }
    config["sample_rate"] = settings.SampleRate;
    config["seconds"] = seconds;
    const Json resolved = {{"sample_rate", settings.SampleRate}, {"cutoff_hz", prepared.Settings.CutoffHz}, {"kappa", settings.Kappa}, {"ratio", settings.Ratio}, {"sigma0", settings.Sigma0}, {"sigma1", settings.Sigma1}, {"lambda", settings.Lambda}, {"epsilon", settings.Epsilon}, {"oversampling", settings.Oversampling}, {"nonlinear", settings.Nonlinear}, {"use_exact", settings.UseExact}, {"norm_type", settings.NormType}, {"seconds", seconds}, {"repeat_count", repeatCount}, {"initial_amplitude", initialAmplitude}, {"strikes", scaledStrikes}, {"pickups", scaledPickups}, {"excitation_scale", excitationScale}, {"output_scale", outputScale}};
    std::unique_ptr<plate::Gpu> gpu;
    if (!cpu) {
        gpu = std::make_unique<plate::Gpu>(prepared, fft ? plate::TransformBackend::Fft : plate::TransformBackend::Dense);
        if (initialAmplitude != 0) gpu->ResetInitial(initialAmplitude);
    }
    const char *backendName = cpu ? "CPU FP64" : (fft ? "Metal 256-bit FFT" : "Metal 256-bit");
    const double setupSeconds = std::chrono::duration<double>(Clock::now() - setupStart).count();
    std::vector<double> samples, diagnostics, states;
    samples.reserve(steps * channels);
    diagnostics.reserve(steps * 6);
    if (recordStates) states.reserve(steps * modes);
    const auto simulationStart = Clock::now();
    for (std::size_t offset = 0; offset < steps; offset += 1024) {
        const auto count = std::min<std::size_t>(1024, steps - offset);
        std::vector<double> forces;
        {
            const profile::Scope scope{"plate/force"};
            for (std::size_t i = 0; cpu && i < count; ++i) {
                for (const auto &strike : strikes) {
                    const double amplitude = plate::RepeatedCosineStrike(offset + i, cycleSteps, settings.SampleRate, strike.Start, strike.Duration, strike.Amplitude, strike.Type);
                    if (amplitude == 0) continue;
                    if (forces.empty()) forces.resize(count * modes, 0.);
                    for (std::size_t m = 0; m < modes; ++m) forces[i * modes + m] += strike.Basis[m] * amplitude;
                }
            }
        }
        std::vector<plate::Diagnostics> metrics;
        if (gpu) {
            const auto highForces = plate::PreciseForces(prepared, preciseStrikes, offset, count, cycleSteps);
            const auto trace = gpu->RunPrepared(highForces, precisePickups, std::uint32_t(count), recordStates);
            samples.insert(samples.end(), trace.Samples.begin(), trace.Samples.end());
            metrics = trace.Metrics;
            if (recordStates) states.insert(states.end(), trace.States.begin(), trace.States.end());
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                metrics.push_back(plate::Step(prepared, state, forces.empty() ? std::span<const double>{} : std::span(forces).subspan(i * modes, modes)));
                for (std::size_t channel = 0; channel < channels; ++channel) samples.push_back(plate::Pickup(prepared, state, std::span(pickupWeights).subspan(channel * modes, modes)));
                if (recordStates) states.insert(states.end(), state.Q.begin(), state.Q.end());
            }
        }
        for (const auto &d : metrics) diagnostics.insert(diagnostics.end(), {d.Energy, d.InputPower, d.DissipatedPower, d.Psi, d.HalfPotential, d.Drift});
    }
    if (gpu) state = gpu->GetState();
    const double simulationSeconds = std::chrono::duration<double>(Clock::now() - simulationStart).count();
    profile::Report();
    if (prefix.empty()) prefix = "build/plate/" + std::filesystem::path(configFile).stem().string();
    const auto parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    WriteDoubles(prefix + ".bin", samples);
    WriteDoubles(prefix + ".diagnostics.bin", diagnostics);
    if (recordStates) WriteDoubles(prefix + ".states.bin", states);
    Json metadata = {
        {"config", config}, {"backend_selection", automatic ? "auto" : "explicit"}, {"backend", backendName}, {"paper", "https://arxiv.org/abs/2608.06139"}, {"sample_rate", settings.SampleRate}, {"sample_count", steps}, {"channels", channels}, {"mx", prepared.Mx}, {"my", prepared.My}, {"nx", prepared.Nx}, {"ny", prepared.Ny}, {"active_modes", prepared.ActiveCount}, {"active_mask", prepared.Active}, {"dtype", "little-endian float64"}, {"samples_layout", "sample, pickup; displacement after update with output scaling"}, {"diagnostic_columns", {"energy", "input_power", "dissipated_power", "psi", "half_potential", "drift"}}, {"diagnostic_alignment", "energy, psi, potential, drift before update; powers centered across update"}, {"time_alignment", "drive[n] at n/fs; pickup/state[n] at (n+1)/fs; energy/psi[n] at (n-0.5)/fs"}, {"states_layout", recordStates ? "sample, mx, my; post-update q, inactive modes zero" : "not recorded"}, {"setup_seconds", setupSeconds}, {"simulation_seconds", simulationSeconds}, {"real_time_factor", simulationSeconds / (double(steps) / settings.SampleRate)}, {"final_state", {{"q", state.Q}, {"previous", state.Previous}, {"psi", state.Psi}, {"step_index", state.StepIndex}}}
    };
    metadata["resolved_config"] = resolved;
    metadata["cycle_sample_count"] = cycleSteps;
    metadata["repeat_count"] = repeatCount;
    metadata["rendered_seconds"] = double(steps) / settings.SampleRate;
    metadata["length_scale"] = lengthScale;
    metadata["output_units"] = physical ? "metres times user output_scale" : "scaled displacement times output_scale";
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("Cannot read process resource usage");
    metadata["peak_process_resident_bytes"] = usage.ru_maxrss;
    if (wav) metadata["wav_gain"] = WriteWav(prefix + ".wav", samples, channels, settings.SampleRate);
    std::ofstream output(prefix + ".json");
    output << metadata.dump(2) << '\n';
    if (!output) throw std::runtime_error("Cannot write metadata");
    std::cout << backendName << ": " << prepared.ActiveCount << " modes, " << prepared.Nx << 'x' << prepared.Ny << " grid, " << steps << " samples, setup " << setupSeconds << " s, simulation " << simulationSeconds << " s (" << metadata["real_time_factor"] << " x real time)\n"
              << prefix << ".json\n";
    return 0;
}
} // namespace

int main(int argc, char *const *argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "PlateSolve: " << error.what() << '\n';
        return 1;
    }
}
