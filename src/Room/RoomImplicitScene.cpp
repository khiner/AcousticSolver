#include "RoomImplicitScene.h"

#include "AabbTree.h"
#include "Parallel.h"
#include "Profile.h"
#include "RoomGpu.h"
#include "RoomScene.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#include "RoomImplicit.inc"

int SceneNode(const ImplicitVoxels &vox, const ImplicitBox &box, const Eigen::RowVector3d &p, const char *kind) {
    int q[3];
    for (int a = 0; a < 3; ++a) q[a] = int(std::lround(p[a]));
    if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= vox.N[0] || q[1] >= vox.N[1] || q[2] >= vox.N[2] || !vox.Filled[vox.At(q[0], q[1], q[2])])
        throw std::runtime_error(std::string(kind) + " does not snap to an interior room node in " + std::to_string(vox.N[0]) + "x" + std::to_string(vox.N[1]) + "x" + std::to_string(vox.N[2]) + " grid");
    const size_t node = box.Index(q[0], q[1], q[2]);
    if (node > size_t(std::numeric_limits<int>::max())) throw std::runtime_error("implicit room node index exceeds the Metal kernel's int32 range");
    return int(node);
}

std::vector<double> MaterialAdmittance(const ImplicitModel &model, const RoomImplicitSceneOptions &options) {
    if (options.Gamma < 0.) throw std::runtime_error("implicit room admittance cannot be negative");
    std::vector<double> gamma(model.Materials.size(), options.Gamma);
    for (const auto &[name, value] : options.MaterialGamma) {
        if (value < 0.) throw std::runtime_error("implicit room material admittance cannot be negative: " + name);
        const auto it = std::find(model.Materials.begin(), model.Materials.end(), name);
        if (it == model.Materials.end()) throw std::runtime_error("implicit room model has no material named: " + name);
        gamma[size_t(it - model.Materials.begin())] = value;
    }
    return gamma;
}
} // namespace

RoomImplicitSceneResult RenderImplicitRoomScene(const std::string &model_file, const RoomImplicitSceneOptions &options) {
    if (!(options.H > 0.) || !(options.C > 0.) || !(options.TimeStep > 0.) || !(options.Seconds > 0.) || !(options.PeakHz > 0.))
        throw std::runtime_error("implicit room h, c, dt, duration, and source peak must be positive");
    if (options.ProjectionEvery < 0) throw std::runtime_error("implicit room projection interval cannot be negative");

    constexpr int Margin = 2;
    const double start = Now();
    const ImplicitModel model = ImplicitLoadModel(model_file, options.H, Margin);
    const size_t nodes = size_t(model.N[0]) * size_t(model.N[1]) * size_t(model.N[2]);
    if (nodes > size_t(std::numeric_limits<int>::max())) throw std::runtime_error("implicit room grid exceeds the Metal kernels' int32 node-index range");
    std::printf("[implicit room] voxelising %s: %dx%dx%d = %.3fM nodes at %.3f mm\n", model_file.c_str(), model.N[0], model.N[1], model.N[2], double(nodes) / 1e6, 1e3 * options.H);
    std::fflush(stdout);
    const ImplicitMesh mesh(model.V, model.F);
    const ImplicitVoxels vox = ImplicitVoxelise(model, mesh);
    const double voxel_seconds = Now() - start;

    const double lambda = options.C * options.TimeStep / options.H;
    const double limit = ImplicitCourantLimit(Implicit1Pct);
    if (lambda > limit * 1.001) throw std::runtime_error("implicit room Courant number " + std::to_string(lambda) + " exceeds the IM1% limit " + std::to_string(limit));
    if (lambda > limit) std::printf("[implicit room] note: lambda %.6f is %.4f%% above the coefficients' unrounded limit %.6f\n", lambda, 100. * (lambda / limit - 1.), limit);

    ImplicitWall wall;
    wall.Gamma = options.Gamma;
    wall.GammaByMaterial = MaterialAdmittance(model, options);
    ImplicitBox box(Implicit1Pct, vox.N[0], vox.N[1], vox.N[2], true, lambda, std::move(wall), ImplicitShape::Voxels, 0., true, &vox);
    if (options.ProjectionEvery) box.SetMeanProjection(options.ProjectionEvery);

    const int steps = std::max(1, int(std::ceil(options.Seconds / options.TimeStep)));
    const double sigma = 1. / (2. * std::numbers::pi * options.PeakHz);
    const double pulse_midpoint = 6. * sigma;
    std::vector<float> signal(static_cast<size_t>(steps));
    for (int n = 0; n < steps; ++n) {
        const double q = (n * options.TimeStep - pulse_midpoint) / sigma;
        signal[size_t(n)] = float(-q * std::exp(-.5 * q * q));
    }

    const int source = SceneNode(vox, box, model.Source, "source");
    std::vector<int> receivers;
    receivers.reserve(std::max<size_t>(model.Receivers.size(), 1));
    if (model.Receivers.empty()) receivers.push_back(source);
    else
        for (const auto &receiver : model.Receivers) receivers.push_back(SceneNode(vox, box, receiver, "receiver"));
    box.SetIo({source}, signal, receivers, steps);

    const std::string projection = options.ProjectionEvery ? "stabilized every " + std::to_string(options.ProjectionEvery) : "off (paper)";
    std::printf("[implicit room] IM1%% lambda %.6f, %d steps at %.1f Hz, %d receivers, gamma %.3f, projection %s; voxelisation %.1fs\n", lambda, steps, 1. / options.TimeStep, int(receivers.size()), options.Gamma, projection.c_str(), voxel_seconds);
    std::printf("[implicit room] %zu inside nodes, %d wall nodes, %d exact boundary terms\n", box.InsideNodes, box.WallNodes, box.Codes);
    std::fflush(stdout);

    const double step_start = Now();
    const int report_every = std::max(1, steps / 20);
    for (int n = 0; n < steps; ++n) {
        box.StepIo(n);
        if ((n + 1) % report_every && n + 1 != steps) continue;
        const double elapsed = Now() - step_start;
        const double eta = elapsed * double(steps - n - 1) / double(n + 1);
        std::printf("[implicit room] step %d/%d, %.1fs elapsed, %.1fs remaining\n", n + 1, steps, elapsed, eta);
        std::fflush(stdout);
    }

    const float *out = box.Out.As<float>();
    RoomImplicitSceneResult result{vox.N[0], vox.N[1], vox.N[2], steps, int(receivers.size()), options.TimeStep, std::vector<double>(size_t(receivers.size()) * size_t(steps))};
    std::transform(out, out + result.Samples.size(), result.Samples.begin(), [](float x) { return double(x); });
    return result;
}

void RunImplicitRoomScene(const std::string &model_file, const RoomImplicitSceneOptions &options) {
    const RoomImplicitSceneResult result = RenderImplicitRoomScene(model_file, options);
    std::filesystem::create_directories("room");
    const std::filesystem::path stem = std::filesystem::path{"room"} / options.Output;
    std::ofstream{stem.string() + ".bin", std::ofstream::binary}.write(reinterpret_cast<const char *>(result.Samples.data()), std::streamsize(result.Samples.size() * sizeof(double)));

    nlohmann::json material_gamma = nlohmann::json::object();
    for (const auto &[name, gamma] : options.MaterialGamma) material_gamma[name] = gamma;
    nlohmann::json const meta{{"scheme", "IM1%"}, {"model", model_file}, {"grid", {result.Nx, result.Ny, result.Nz}}, {"h", options.H}, {"c", options.C}, {"dt", result.TimeStep}, {"srate", 1. / result.TimeStep}, {"samples", result.Steps}, {"receivers", result.Receivers}, {"source_peak_hz", options.PeakHz}, {"source_midpoint_seconds", 6. / (2. * std::numbers::pi * options.PeakHz)}, {"gamma", options.Gamma}, {"material_gamma", material_gamma}, {"projection_every", options.ProjectionEvery}, {"paper_method", options.ProjectionEvery == 0}};
    std::ofstream{stem.string() + ".json"} << meta.dump() << '\n';
    std::printf("[implicit room] wrote %s.bin (%d receivers x %d float64)\n", stem.string().c_str(), result.Receivers, result.Steps);
    profile::Report();
}
