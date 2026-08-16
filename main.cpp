// Ported from WaveBlender (c) 2024 Kangrui Xue (main.cpp) — Metal port.
// Parses a JSON config file and runs the simulation.

#include "WaveBlender.h"

#include "Profile.h"
#include "json.hpp"

using json = nlohmann::json;

namespace {
void Run(const std::string &config_file) {
    std::ifstream config_stream{config_file};
    json config;
    config_stream >> config;

    // Parse global simulation parameters
    SimParams params;
    params.Nx = config["Nx"];
    params.Ny = config["Ny"];
    params.Nz = config["Nz"];
    params.Dx = config["cellsize"];

    params.FdtdSrate = config["FDTD_srate"];
    params.Dt = 1. / params.FdtdSrate;
    params.Ts = config["ts"];
    params.Tf = config["tf"];

    params.ShaderSrate = config["shader_srate"];
    params.BlendRate = config["blendrate"];

    params.Damping = config["damping"];

    WaveBlender solver{params};

    // Parse object parameters
    for (const auto &object_config : config["objects"]) {
        const std::vector<real> offset_values = object_config["offset"];
        const Eigen::Vector3<real> offset{offset_values[0], offset_values[1], offset_values[2]};

        const std::string shader = object_config["shader"];
        if (shader == "Monopole") {
            solver.AddObject<Monopole>(offset, params.BlendRate, params.ShaderSrate, params.Ts, object_config["meshFile"], real(object_config["freqHz"]), real(object_config["speed"]), params.C);
        } else if (shader == "Speaker") {
            solver.AddObject<Speaker>(offset, params.BlendRate, params.ShaderSrate, params.Ts, object_config["meshFile"], object_config["wavFile"], int(object_config["direction"]), object_config["animFile"]);
        } else if (shader == "Occluder") {
            solver.AddObject<Occluder>(offset, params.BlendRate, params.ShaderSrate, params.Ts, object_config["meshFile"], object_config["animFile"]);
        } else if (shader == "Bubbles") {
            solver.AddObject<Bubbles>(offset, params.BlendRate, params.ShaderSrate, params.Ts, object_config["bubFile"], object_config["meshDir"], params.Dx);
        } else if (shader == "Modal") {
            solver.AddObject<Modal>(offset, params.BlendRate, params.ShaderSrate, params.Ts, object_config["meshFile"], object_config["animFile"], object_config["dataPrefix"], object_config["material"]);
        } else if (shader == "Shell") {
            solver.AddObject<Shell>(offset, params.BlendRate, params.ShaderSrate, params.Ts, object_config["meshFile"], object_config["animDir"], object_config["accDir"], object_config["mapFile"]);
        } else if (shader == "Point") {
            solver.AddObject<PointSource>(offset, params.BlendRate, params.ShaderSrate, params.Ts, object_config["impulseFile"], params.Dx);
        } else if (shader == "Density") {
            solver.AddObject<Density>(offset, params.BlendRate, params.ShaderSrate, params.Ts, object_config["betaDir"]);
        } else {
            throw std::runtime_error("Invalid Shader: " + shader);
        }
    }

    // Parse listener parameters
    for (const auto &listener_config : config["listeners"]) {
        solver.AddListener(listener_config["format"], listener_config["position"], listener_config["output"]);
    }

    while (solver.RunBatch()) {}
    profile::Report();
}
} // namespace

// Usage: ./AcousticSolver [path to config file]
int main(int argc, char *argv[]) {
    const std::string config_file = argc > 1 ? argv[1] : "../Scenes/CupPhone/config.json";
    Run(config_file);
    return 0;
}
