#include "RoomScene.h"

#include "Profile.h"
#include "RoomGpu.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {
// Where the solver's receiver output goes, kept apart from the WaveBlender path's
// <output>.bin, which is a different solver at a different rate.
constexpr const char *OutputDir = "room";

template<typename T> std::vector<T> ReadBin(const std::filesystem::path &path) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in) throw std::runtime_error("Failed to read scene data: " + path.string());
    const auto bytes = size_t(in.tellg());
    if (bytes % sizeof(T) != 0) throw std::runtime_error("Scene data is not a whole number of elements: " + path.string());
    std::vector<T> values(bytes / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(values.data()), std::streamsize(bytes));
    return values;
}

// Node indices arrive as int64 (the reference's own width) and the kernels index in int32,
// which every grid that fits in memory stays inside.
std::vector<int> ReadIndices(const std::filesystem::path &path, int64_t nodes) {
    const auto wide = ReadBin<int64_t>(path);
    std::vector<int> narrow(wide.size());
    for (size_t i = 0; i < wide.size(); ++i) {
        if (wide[i] < 0 || wide[i] >= nodes) throw std::runtime_error("Scene node index outside the grid: " + path.string());
        narrow[i] = int(wide[i]);
    }
    return narrow;
}

// Normalises the source signals into the middle of the float range and returns the gain to
// re-apply to the receiver signals, as the reference engines do around a run. It moves only
// where the fp32 exponents sit, and it is what the goldens were produced under.
double ScaleSource(RoomScene &scene) {
    double max_in = 0.;
    for (double const v : scene.InSigs) max_in = std::max(max_in, std::fabs(v));
    if (max_in == 0.) return 1.;
    // Half way between the float exponent limits, rounded — REAL_MAX_EXP / REAL_MIN_EXP in
    // the reference, which is fp32 here.
    const double norm1 = std::pow(2., std::round(0.5 * 128 + 0.5 * -125));
    const double inv_infac = norm1 / max_in;
    for (double &v : scene.InSigs) v *= inv_infac;
    return 1. / inv_infac;
}
} // namespace

RoomScene LoadRoomScene(const std::string &config_file) {
    const profile::Scope scope{"room/scene_load"};
    const auto dir = std::filesystem::path{config_file}.parent_path();
    std::ifstream config_stream{config_file};
    if (!config_stream) throw std::runtime_error("Failed to read room config: " + config_file);
    json config;
    config_stream >> config;

    RoomScene scene;
    const std::string scheme = config["scheme"];
    if (scheme == "fcc") scene.Fcc = true;
    else if (scheme != "cart") throw std::runtime_error("Scene names a scheme that is neither cart nor fcc: " + scheme);
    const int neighbours = scene.Neighbours();

    const auto &grid = config["grid"];
    scene.Nx = grid["Nx"];
    scene.Ny = grid["Ny"];
    scene.Nz = grid["Nz"];
    scene.H = grid["h"];
    if (scene.Fcc && int(grid["Nyf"]) != scene.NyUnfolded()) throw std::runtime_error("The FCC grid's unfolded y extent is not twice its folded one less two");
    scene.C = config["air"]["c"];
    const auto &step = config["step"];
    scene.Ts = step["Ts"];
    scene.Srate = step["srate"];
    scene.L = step["l"];
    scene.L2 = step["l2"];
    scene.Nt = step["count"];
    const int64_t nodes = int64_t(scene.Nx) * scene.Ny * scene.Nz;

    const auto &boundary = config["boundary"];
    if (int(boundary["neighbours"]) != neighbours) throw std::runtime_error("Boundary adjacency does not carry the scheme's neighbour count");
    scene.BnIxyz = ReadIndices(dir / std::string(boundary["indices"]), nodes);
    scene.MatBn = ReadBin<int8_t>(dir / std::string(boundary["material"]));
    scene.SafBn = ReadBin<double>(dir / std::string(boundary["area"]));
    const auto adj = ReadBin<uint8_t>(dir / std::string(boundary["adjacency"]));
    const size_t nb = scene.BnIxyz.size();
    if (adj.size() != nb * size_t(neighbours) || scene.MatBn.size() != nb || scene.SafBn.size() != nb)
        throw std::runtime_error("Boundary sidecars disagree on the node count");
    scene.AdjBn.assign(nb, 0);
    for (size_t i = 0; i < nb; ++i) {
        for (int bit = 0; bit < neighbours; ++bit) {
            if (adj[i * size_t(neighbours) + bit]) scene.AdjBn[i] |= uint16_t(1u << bit);
        }
    }
    scene.NumLossy = int(std::count_if(scene.MatBn.begin(), scene.MatBn.end(), [](int8_t m) { return m >= 0; }));

    for (const auto &material : config["materials"]) scene.MatBranches.push_back(material["branches"]);
    if (scene.NumMaterials() > 0) scene.MatDef = ReadBin<double>(dir / std::string(config["material_def"]));
    size_t branches = 0;
    for (int const b : scene.MatBranches) {
        if (b < 1 || b > RoomMaxBranches) throw std::runtime_error("Material branch count outside the range the boundary state is sized for");
        branches += size_t(b);
    }
    if (scene.MatDef.size() != 3 * branches) throw std::runtime_error("Material definitions are not three coefficients a branch");
    for (int8_t const m : scene.MatBn) {
        if (m >= scene.NumMaterials()) throw std::runtime_error("Boundary node names a material the scene does not define");
    }

    const auto &source = config["source"];
    scene.InIxyz = ReadIndices(dir / std::string(source["indices"]), nodes);
    scene.InSigs = ReadBin<double>(dir / std::string(source["signals"]));
    if (scene.InSigs.size() != scene.InIxyz.size() * size_t(scene.Nt)) throw std::runtime_error("Source signals are not one row a corner");
    // The source scatter runs one thread a corner with no ordering, so two corners landing on
    // one node would race.
    auto sorted = scene.InIxyz;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) throw std::runtime_error("Source corners share a node");

    const auto &receivers = config["receivers"];
    scene.NumReceivers = receivers["count"];
    scene.CornersPerReceiver = receivers["corners"];
    scene.OutIxyz = ReadIndices(dir / std::string(receivers["indices"]), nodes);
    scene.OutAlpha = ReadBin<double>(dir / std::string(receivers["weights"]));
    if (int(scene.OutIxyz.size()) != scene.NumReceivers * scene.CornersPerReceiver || scene.OutAlpha.size() != scene.OutIxyz.size())
        throw std::runtime_error("Receiver sidecars disagree on the corner count");
    scene.Output = receivers["output"];
    return scene;
}

std::vector<double> RenderRoomScene(RoomScene &scene, int n_steps) {
    const double infac = ScaleSource(scene);
    RoomGpu gpu;
    gpu.Init(scene);
    gpu.RunSteps(n_steps);

    const float *samples = gpu.Samples();
    const int corners = scene.NumOutputs(), per = scene.CornersPerReceiver;
    std::vector<double> rows(size_t(scene.NumReceivers) * size_t(n_steps));
    for (int r = 0; r < scene.NumReceivers; ++r) {
        for (int n = 0; n < n_steps; ++n) {
            double acc = 0.;
            for (int c = 0; c < per; ++c) acc += scene.OutAlpha[size_t(r) * per + c] * double(samples[size_t(n) * corners + r * per + c]);
            rows[size_t(r) * n_steps + n] = acc * infac;
        }
    }
    if (!gpu.Timings().empty()) RoomGpu::ReportTimings(gpu.Timings());
    return rows;
}

void RunRoomScene(const std::string &config_file, double seconds) {
    RoomScene scene = LoadRoomScene(config_file);
    const int steps = seconds > 0. ? std::min(scene.Nt, std::max(1, int(std::ceil(seconds * scene.Srate)))) : scene.Nt;
    std::printf("[room] %s: %s %dx%dx%d = %.2fM nodes, %d boundary (%d lossy), %d steps at %.1f Hz\n", scene.Output.c_str(), scene.Fcc ? "FCC" : "cart", scene.Nx, scene.Ny, scene.Nz, double(scene.NumNodes()) / 1e6, int(scene.BnIxyz.size()), scene.NumLossy, steps, scene.Srate);
    std::fflush(stdout); // so a long render shows what it is running before it starts

    const auto rows = RenderRoomScene(scene, steps);

    std::filesystem::create_directories(OutputDir);
    const auto stem = std::filesystem::path{OutputDir} / scene.Output;
    std::ofstream{stem.string() + ".bin", std::ofstream::binary}.write(reinterpret_cast<const char *>(rows.data()), std::streamsize(rows.size() * sizeof(double)));
    std::ofstream{stem.string() + ".json"} << json{{"srate", scene.Srate}, {"receivers", scene.NumReceivers}, {"samples", steps}}.dump() << '\n';
    std::printf("[room] wrote %s.bin (%d receivers x %d float64)\n", stem.string().c_str(), scene.NumReceivers, steps);
    profile::Report();
}
