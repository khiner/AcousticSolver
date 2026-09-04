#include "ImmersedScene.h"

#include "ImmersedGeometry.h"
#include "Profile.h"
#include "RadiationMesh.h"
#include "json.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>

namespace immersed {
namespace {
using json = nlohmann::json;

Point ReadPoint(const json &value, const char *name) {
    if (!value.is_array() || value.size() != 3) throw std::runtime_error(std::string{name} + " must have three coordinates");
    return {value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
}

Point Rotate(const Point &point, const Point &angles) {
    const double cx = std::cos(angles[0]), sx = std::sin(angles[0]);
    const double cy = std::cos(angles[1]), sy = std::sin(angles[1]);
    const double cz = std::cos(angles[2]), sz = std::sin(angles[2]);
    const Point x{point[0], cx * point[1] - sx * point[2], sx * point[1] + cx * point[2]};
    const Point y{cy * x[0] + sy * x[2], x[1], -sy * x[0] + cy * x[2]};
    return {cz * y[0] - sz * y[1], sz * y[0] + cz * y[1], y[2]};
}

RationalImmittance ReadFiniteImmittance(const json &value) {
    if (value.is_number()) return RationalImmittance::Constant(value.get<double>());
    if (!value.is_object()) throw std::runtime_error("A finite immittance must be a number or {a,b,c}");
    return RationalImmittance::Series(value.value("a", 0.), value.value("b", 0.), value.value("c", 0.));
}

struct SurfaceImmittances {
    BoundaryImmittance Zv;
    BoundaryImmittance Yp;
};

SurfaceImmittances ReadImmittances(const json &surface) {
    const json zv_value = surface.value("zv", json{0.});
    const json yp_value = surface.value("yp", json{0.});
    SurfaceImmittances result;
    if (zv_value.is_string() && zv_value.get<std::string>() == "rigid") result.Zv = BoundaryImmittance::Limit();
    else if (zv_value.is_string() && zv_value.get<std::string>() == "release") result.Zv = BoundaryImmittance::Finite(0.);
    else result.Zv = BoundaryImmittance::Finite(ReadFiniteImmittance(zv_value));

    if (yp_value.is_string()) {
        const std::string name = yp_value;
        if (name == "release") result.Yp = BoundaryImmittance::Limit();
        else if (name == "1/zv") {
            if (result.Zv.Infinite) throw std::runtime_error("A rigid impedance has no finite reciprocal admittance");
            result.Yp = BoundaryImmittance::Finite(result.Zv.Value.Reciprocal());
        } else throw std::runtime_error("Unknown pressure-side admittance " + name);
    } else result.Yp = BoundaryImmittance::Finite(ReadFiniteImmittance(yp_value));
    if (zv_value.is_string() && zv_value.get<std::string>() == "release" && !surface.contains("yp"))
        result.Yp = BoundaryImmittance::Limit();
    return result;
}

double TargetArea(const json &surface, double h) {
    if (surface.contains("target_area")) return surface["target_area"];
    const double patch_size = surface.value("patch_size", h);
    return patch_size * patch_size;
}

void Append(std::vector<Patch> &to, std::vector<Patch> from) { to.insert(to.end(), std::make_move_iterator(from.begin()), std::make_move_iterator(from.end())); }

void AppendObj(std::vector<Patch> &patches, const json &surface, const std::filesystem::path &base, double h, const SurfaceImmittances &immittance) {
    const Point offset = ReadPoint(surface.value("offset", json::array({0., 0., 0.})), "obj offset");
    const Point rotation = ReadPoint(surface.value("rotation", json::array({0., 0., 0.})), "obj rotation");
    const double scale = surface.value("scale", 1.);
    std::filesystem::path path = surface.at("file").get<std::string>();
    if (path.is_relative()) path = base / path;
    RadiationMesh mesh = RadiationMesh::FromObj(path.string(), Eigen::Vector3d::Zero());
    for (auto &vertex : mesh.Vertices) {
        const Point rotated = Rotate({scale * vertex[0], scale * vertex[1], scale * vertex[2]}, rotation);
        vertex = {rotated[0] + offset[0], rotated[1] + offset[1], rotated[2] + offset[2]};
    }
    mesh.Finalize();
    const double max_edge = surface.value("max_edge", 1.5 * h);
    mesh.Remesh(.6 * max_edge, max_edge);
    for (size_t i = 0; i < mesh.Centroids.size(); ++i) {
        const auto &c = mesh.Centroids[i];
        const auto &n = mesh.Normals[i];
        patches.push_back({{c[0], c[1], c[2]}, {n[0], n[1], n[2]}, mesh.Areas[i], immittance.Zv, immittance.Yp});
    }
}

double Gaussian(double time, double f60, double maximum) {
    const double sigma = std::sqrt(12. * std::numbers::ln10) / (2. * std::numbers::pi * f60);
    const double t0 = sigma * std::sqrt(2. * std::log(maximum / std::numeric_limits<float>::epsilon()));
    const double x = (time - t0) / sigma;
    return maximum * std::exp(-.5 * x * x);
}

} // namespace

Scene LoadScene(const std::string &config_file, double seconds, const std::string &output) {
    const profile::Scope scope{"immersed/scene_load"};
    std::ifstream input{config_file};
    if (!input) throw std::runtime_error("Failed to read immersed config: " + config_file);
    json config;
    input >> config;

    Scene scene;
    const auto source_path = std::filesystem::weakly_canonical(config_file);
    scene.SourceFile = std::filesystem::relative(source_path, ACOUSTIC_PROJECT_ROOT).generic_string();
    const json medium = config.value("medium", json::object());
    scene.GridSpec.C = medium.value("c", 344.);
    scene.GridSpec.Rho = medium.value("rho", 1.18);
    scene.GridSpec.PmlWidth = config.value("pml_width", 8);
    const auto &grid = config.at("grid");
    scene.GridSpec.Courant = grid.value("courant", .575);
    if (grid.contains("h")) scene.GridSpec.H = grid["h"];
    else if (grid.contains("sample_rate")) scene.GridSpec.H = scene.GridSpec.C / (scene.GridSpec.Courant * double(grid["sample_rate"]));
    else throw std::runtime_error("Immersed grid needs h or sample_rate");
    if (grid.contains("cells")) {
        const Point cells = ReadPoint(grid["cells"], "grid cells");
        scene.GridSpec.Nx = int(cells[0]);
        scene.GridSpec.Ny = int(cells[1]);
        scene.GridSpec.Nz = int(cells[2]);
    } else {
        const Point extent = ReadPoint(grid.at("extent"), "grid extent");
        scene.GridSpec.Nx = int(std::ceil(extent[0] / scene.GridSpec.H));
        scene.GridSpec.Ny = int(std::ceil(extent[1] / scene.GridSpec.H));
        scene.GridSpec.Nz = int(std::ceil(extent[2] / scene.GridSpec.H));
    }
    if (grid.contains("origin")) scene.GridSpec.Origin = ReadPoint(grid["origin"], "grid origin");
    else scene.GridSpec.Origin = {-.5 * (scene.GridSpec.Nx - 1) * scene.GridSpec.H, -.5 * (scene.GridSpec.Ny - 1) * scene.GridSpec.H, -.5 * (scene.GridSpec.Nz - 1) * scene.GridSpec.H};
    scene.GridSpec.Finalize();
    scene.SampleRate = 1. / scene.GridSpec.TimeStep;
    scene.InterpolationOrder = config.value("interpolation_order", 5);
    const double duration = seconds > 0. ? seconds : config.at("seconds").get<double>();
    if (!(duration > 0.)) throw std::runtime_error("Immersed scene duration must be positive");
    scene.Steps = std::max(1, int(std::ceil(duration / scene.GridSpec.TimeStep)));
    scene.Output = output.empty() ? config.value("output", std::filesystem::path{config_file}.stem().string()) : output;

    for (const auto &surface : config.value("surfaces", json::array())) {
        const auto immittance = ReadImmittances(surface);
        const std::string type = surface.at("type");
        if (type == "sphere") {
            Append(scene.Patches, SpherePatches(ReadPoint(surface.at("center"), "sphere center"), surface.at("radius"), TargetArea(surface, scene.GridSpec.H), immittance.Zv, immittance.Yp));
        } else if (type == "square") {
            const Point rotation = ReadPoint(surface.value("rotation", json::array({0., 0., 0.})), "square rotation");
            const Point u = Rotate({1., 0., 0.}, rotation);
            const Point v = Rotate({0., 1., 0.}, rotation);
            Append(scene.Patches, SquarePatches(ReadPoint(surface.at("center"), "square center"), u, v, surface.at("side"), TargetArea(surface, scene.GridSpec.H), immittance.Zv, immittance.Yp));
        } else if (type == "obj") AppendObj(scene.Patches, surface, std::filesystem::path{config_file}.parent_path(), scene.GridSpec.H, immittance);
        else throw std::runtime_error("Unknown immersed surface type " + type);
    }

    const auto &source = config.at("source");
    scene.Sources.push_back(ReadPoint(source.at("position"), "source position"));
    const double f60 = source.at("f60"), maximum = source.value("umax", 1e-3);
    scene.SourceSamples.resize(size_t(scene.Steps) + 1);
    for (int step = 0; step <= scene.Steps; ++step)
        scene.SourceSamples[size_t(step)] = float(Gaussian(step * scene.GridSpec.TimeStep, f60, maximum));
    for (const auto &receiver : config.at("receivers")) scene.Receivers.push_back(ReadPoint(receiver, "receiver"));
    if (scene.Receivers.empty()) throw std::runtime_error("Immersed scene needs at least one receiver");
    return scene;
}

std::vector<float> RenderScene(const Scene &scene) {
    Gpu gpu;
    gpu.Init(scene.GridSpec, scene.Sources, scene.Receivers, scene.Steps, scene.SourceSamples, scene.Patches, scene.InterpolationOrder);
    gpu.RunSteps(scene.Steps);
    const float *samples = gpu.Samples();
    return {samples, samples + size_t(scene.Steps) * scene.Receivers.size()};
}

void RunScene(const std::string &config_file, double seconds, const std::string &output) {
    const Scene scene = LoadScene(config_file, seconds, output);
    std::printf("[immersed] %s: %dx%dx%d = %.2fM cells, %zu patches, %d steps at %.1f Hz\n", scene.Output.c_str(), scene.GridSpec.Nx, scene.GridSpec.Ny, scene.GridSpec.Nz, double(scene.GridSpec.NumCells()) / 1e6, scene.Patches.size(), scene.Steps, scene.SampleRate);
    std::fflush(stdout);
    const auto samples = RenderScene(scene);
    const auto output_dir = std::filesystem::path{ACOUSTIC_PROJECT_ROOT} / "build/immersed";
    std::filesystem::create_directories(output_dir);
    const auto stem = output_dir / scene.Output;
    std::ofstream{stem.string() + ".bin", std::ios::binary}.write(
        reinterpret_cast<const char *>(samples.data()), std::streamsize(samples.size() * sizeof(float))
    );
    const json metadata{
        {"sample_rate", scene.SampleRate},
        {"samples", scene.Steps},
        {"receivers", scene.Receivers.size()},
        {"receiver_positions", scene.Receivers},
        {"grid", {{"cells", {scene.GridSpec.Nx, scene.GridSpec.Ny, scene.GridSpec.Nz}}, {"h", scene.GridSpec.H}}},
        {"scene", scene.SourceFile}
    };
    std::ofstream{stem.string() + ".json"} << metadata.dump() << '\n';
    std::printf("[immersed] wrote %s.bin (%d samples x %zu receivers, float32 interleaved)\n", stem.string().c_str(), scene.Steps, scene.Receivers.size());
    profile::Report();
}

} // namespace immersed
