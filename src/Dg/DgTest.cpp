#include "DgGpu.h"
#include "json.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
template<typename T> std::vector<T> Read(const fs::path &path, size_t count) {
    if (fs::file_size(path) != count * sizeof(T)) throw std::runtime_error("Wrong size: " + path.string());
    std::vector<T> data(count);
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T))) throw std::runtime_error("Cannot read: " + path.string());
    return data;
}

template<typename T> void Write(const fs::path &path, const std::vector<T> &data) {
    std::ofstream output(path, std::ios::binary);
    if (!output.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T))) throw std::runtime_error("Cannot write: " + path.string());
}

void Save(const fs::path &path, const json &value) {
    if (!(std::ofstream(path) << value.dump(2) << '\n')) throw std::runtime_error("Cannot write: " + path.string());
}

dg::Tables Load(const fs::path &path, const json &metadata) {
    dg::Tables result;
    auto &p = result.Params;
    p.Elements = metadata.at("elements");
    p.Nodes = metadata.at("nodes");
    p.FaceNodes = metadata.at("face_nodes");
    p.QuadraturePoints = metadata.at("quadrature_points");
    p.Receivers = metadata.at("receivers");
    p.SoundSpeed = metadata.at("sound_speed");
    const size_t e = p.Elements, n = p.Nodes, f = p.FaceNodes, q = p.QuadraturePoints, r = p.Receivers;
    if (!e || !n || !f || !q || !r || e * 10 * n * f > UINT32_MAX || e * 6 * n * n > UINT32_MAX || e * q > UINT32_MAX)
        throw std::runtime_error("Invalid DG dimensions");
    result.G = Read<float>(path / "G.bin", e * 6 * n * n);
    result.L = Read<float>(path / "L.bin", e * 10 * n * f);
    result.T = Read<float>(path / "T.bin", q * n);
    result.W = Read<float>(path / "W.bin", e * q);
    result.Initial = Read<float>(path / "initial.bin", e * n * 4);
    result.VmapM = Read<uint32_t>(path / "vmapM.bin", e * f);
    result.VmapP = Read<uint32_t>(path / "vmapP.bin", e * f);
    result.Boundary = Read<uint32_t>(path / "boundary.bin", e * f);
    result.ReceiverElements = Read<uint32_t>(path / "receiver_elements.bin", r);
    result.ReceiverWeights = Read<float>(path / "receiver_interp.bin", r * n);
    for (size_t i = 0; i < e * f; ++i)
        if (result.VmapM[i] >= e * n || result.VmapP[i] >= e * n || result.Boundary[i] > 1) throw std::runtime_error("Invalid DG face map");
    for (auto const element : result.ReceiverElements)
        if (element >= e) throw std::runtime_error("Invalid DG receiver element");
    for (const auto *values : {&result.G, &result.L, &result.T, &result.W, &result.Initial, &result.ReceiverWeights})
        for (float const value : *values)
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite DG input");
    for (float const weight : result.W)
        if (!(weight > 0)) throw std::runtime_error("Nonpositive DG mass weight");
    result.RkA = metadata.at("rka").get<std::array<float, 5>>();
    result.RkB = metadata.at("rkb").get<std::array<float, 5>>();
    return result;
}

double Norm(std::span<const float> values) {
    double sum = 0;
    for (double const value : values) sum += value * value;
    return std::sqrt(sum);
}

double Error(std::span<const float> values, std::span<const double> truth, double minimum_scale = 0) {
    double difference = 0, norm = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) throw std::runtime_error("Nonfinite Metal result");
        difference += std::pow(double(values[i]) - truth[i], 2);
        norm += truth[i] * truth[i];
    }
    return std::sqrt(difference) / std::max(std::sqrt(norm), minimum_scale);
}

struct Diagnostic {
    uint32_t Elements, Nodes;
    std::vector<double> H, Weights;
    double Volume;

    std::array<double, 2> Evaluate(std::span<const float> q) const {
        double energy = 0, mean = 0;
        for (size_t e = 0; e < Elements; ++e) {
            for (size_t i = 0; i < Nodes; ++i) {
                mean += Weights[e * Nodes + i] * q[(e * Nodes + i) * 4];
                for (size_t j = 0; j < Nodes; ++j) {
                    double dot = 0;
                    for (size_t c = 0; c < 4; ++c) dot += double(q[(e * Nodes + i) * 4 + c]) * q[(e * Nodes + j) * 4 + c];
                    energy += .5 * H[(e * Nodes + i) * Nodes + j] * dot;
                }
            }
        }
        if (!std::isfinite(energy) || !std::isfinite(mean)) throw std::runtime_error("Nonfinite DG diagnostic");
        return {energy, mean / Volume};
    }
};
} // namespace

int main(int argc, char *const *argv) {
    auto *pool = NS::AutoreleasePool::alloc()->init();
    try {
        const bool benchmark = argc == 7 && std::string(argv[6]) == "--benchmark";
        if (argc < 3 || (argc > 6 && !benchmark)) throw std::runtime_error("Usage: DgTest operator-directory output-directory [steps=1024] [dt=2^-18] [repeats=2] [--benchmark]");
        const fs::path input = argv[1], output = argv[2];
        const uint32_t steps = argc > 3 ? std::stoul(argv[3]) : 1024;
        const double dt = double(argc > 4 ? std::stof(argv[4]) : std::ldexp(1.f, -18));
        const uint32_t repeats = argc > 5 ? std::stoul(argv[5]) : 2;
        if (!steps || !repeats || !(dt > 0) || !std::isfinite(dt)) throw std::runtime_error("Positive run parameters required");
        fs::create_directories(output);
        for (const auto *name : {"result.json", "final_q.bin", "receivers.bin", "times.bin", "energy.json", "results.json", "operator-check.json", "provenance.json"}) fs::remove(output / name);
        for (const auto &entry : fs::directory_iterator(output)) {
            const auto name = entry.path().filename().string();
            if (!entry.is_directory() || entry.is_symlink() || !name.starts_with("repeat") || name.size() == 6 || name.find_first_not_of("0123456789", 6) != std::string::npos) continue;
            for (const auto *file : {"final_q.bin", "receivers.bin", "times.bin", "energy.json", "result.json"}) fs::remove(entry.path() / file);
            if (fs::is_empty(entry.path())) fs::remove(entry.path());
        }
        const auto metadata = json::parse(std::ifstream(input / "operator.json"));
        const auto tables = Load(input, metadata);
        dg::Gpu gpu(tables);
        const auto &p = tables.Params;
        const size_t count = size_t(p.Elements) * p.Nodes * 4;
        json checks;
        for (const std::string name : {"random", "initial", "constant"}) {
            const auto state = Read<float>(input / (name + "_state.bin"), count);
            const auto actual = gpu.ApplyRhs(state);
            const auto truth = Read<double>(input / (name + "_rhs.bin"), count);
            const double error = Error(actual, truth, Norm(state) * p.SoundSpeed);
            checks[name + "_rhs_scaled_l2"] = error;
            if (error > 2e-5) throw std::runtime_error("Metal RHS error: " + name + " " + std::to_string(error));
            if (name == "constant" && Norm(actual) != 0) throw std::runtime_error("Constant pressure must produce exactly zero Metal RHS");
        }
        const auto mass = gpu.ApplyMass(Read<float>(input / "mass_load.bin", count));
        const double mass_error = Error(mass, Read<double>(input / "mass_result.bin", count));
        checks["mass_relative_l2"] = mass_error;
        if (mass_error > 2e-5) throw std::runtime_error("Metal weight-adjusted mass error");
        Diagnostic diagnostic{p.Elements, p.Nodes, Read<double>(input / "modified_mass.bin", size_t(p.Elements) * p.Nodes * p.Nodes), Read<double>(input / "physical_mass_weights.bin", size_t(p.Elements) * p.Nodes), 0};
        diagnostic.Volume = std::accumulate(diagnostic.Weights.begin(), diagnostic.Weights.end(), 0.);
        const auto initial = diagnostic.Evaluate(tables.Initial);
        const double scale = std::sqrt(2 * initial[0] / diagnostic.Volume);
        auto &ctx = MetalContext::Get();
        json report{{"device", ctx.Device->name()->utf8String()}, {"precision", "float32"}, {"fast_math", false}, {"operator", fs::absolute(input).string()}, {"steps", steps}, {"dt", dt}, {"repeats", repeats}, {"output_scalar_bytes", 8}, {"sample_start", 0.}, {"pressure_offset", metadata.at("pressure_offset")}, {"operator_checks", checks}, {"diagnostic_interval_steps", benchmark ? steps : 64}, {"runs", json::array()}};
        std::vector<float> previous_state, previous_receivers;
        for (uint32_t repeat = 1; repeat <= repeats; ++repeat) {
            gpu.Reset(tables.Initial, steps);
            ctx.TakeBatchGpuSeconds();
            json history = json::array({{0., initial[0], initial[1]}});
            double maximum_growth = 0, maximum_mean = 0, gpu_seconds = 0;
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t step = 0; step < steps; ++step) {
                gpu.Step(step, float(dt));
                if ((step + 1) % 64 == 0 || step + 1 == steps) {
                    ctx.Drain();
                    gpu_seconds += ctx.TakeBatchGpuSeconds();
                    if (benchmark) continue;
                    const auto values = diagnostic.Evaluate(gpu.State());
                    maximum_growth = std::max(maximum_growth, values[0] / initial[0] - 1);
                    maximum_mean = std::max(maximum_mean, std::abs(values[1] - initial[1]) / scale);
                    history.push_back({(step + 1) * dt, values[0], values[1]});
                }
            }
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const auto state = gpu.State(), receiver = gpu.Receivers();
            if (benchmark) {
                const auto values = diagnostic.Evaluate(state);
                maximum_growth = std::max(0., values[0] / initial[0] - 1);
                maximum_mean = std::abs(values[1] - initial[1]) / scale;
                history.push_back({steps * dt, values[0], values[1]});
            }
            if (repeat > 1 && (std::memcmp(state.data(), previous_state.data(), state.size() * sizeof(float)) != 0 || std::memcmp(receiver.data(), previous_receivers.data(), receiver.size() * sizeof(float)) != 0))
                throw std::runtime_error("Metal trajectory repeat bytes differ");
            previous_state = state;
            previous_receivers = receiver;
            if (repeat == 1) report["energy"] = history;
            json result{{"wall_seconds_including_diagnostics", wall}, {"gpu_seconds", gpu_seconds}, {"final_to_initial_energy", history.back()[1].get<double>() / initial[0]}, {"maximum_energy_above_initial", maximum_growth}, {"maximum_mean_drift_over_initial_acoustic_rms", maximum_mean}};
            if (benchmark) {
                result.erase("wall_seconds_including_diagnostics");
                result["propagation_wall_seconds"] = wall;
            }
            report["runs"].push_back(result);
            std::cout << result.dump(2) << std::endl;
            if (maximum_growth > 1e-5 || maximum_mean > 1e-5) throw std::runtime_error("Metal energy or conservation gate failed");
        }
        const double offset = metadata.at("pressure_offset"), impedance = metadata.at("impedance");
        std::vector<double> final(count), record(previous_receivers.size());
        for (size_t e = 0; e < p.Elements; ++e)
            for (size_t n = 0; n < p.Nodes; ++n)
                for (size_t c = 0; c < 4; ++c)
                    final[(e * 4 + c) * p.Nodes + n] = c ? previous_state[(e * p.Nodes + n) * 4 + c] / impedance : double(previous_state[(e * p.Nodes + n) * 4]) + offset;
        for (size_t i = 0; i < record.size(); ++i) record[i] = double(previous_receivers[i]) + offset;
        Write(output / "final_q.bin", final);
        Write(output / "receivers.bin", record);
        report["repeatable_bytes"] = repeats > 1;
        report["passed"] = true;
        Save(output / "result.json", report);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        pool->release();
        return 1;
    }
    pool->release();
}
