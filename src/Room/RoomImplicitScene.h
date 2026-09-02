#pragma once

// Production runner for Smits & Bilbao's implicit scheme over PFFDTD model_export.json.

#include <numbers>
#include <string>
#include <unordered_map>
#include <vector>

struct RoomImplicitSceneOptions {
    // Fig. 6 / Table III IM1% configuration.
    double H{6.52e-3};
    double C{343.2};
    double TimeStep{15.96e-6};
    double Seconds{18e-3 + 6. / (2. * std::numbers::pi * 2000.)};
    double PeakHz{2000.};
    double Gamma{0.05};
    int ProjectionEvery{0}; // zero is the paper method; 1024 is the optional long-record closure
    std::string Output{"implicit-room"};

    // Per-material real-admittance overrides keyed by mats_hash name.
    std::unordered_map<std::string, double> MaterialGamma;
};

struct RoomImplicitSceneResult {
    int Nx{0}, Ny{0}, Nz{0};
    int Steps{0}, Receivers{0};
    double TimeStep{0.};
    std::vector<double> Samples; // receiver-major rows
};

RoomImplicitSceneResult RenderImplicitRoomScene(const std::string &model_file, const RoomImplicitSceneOptions &options);
void RunImplicitRoomScene(const std::string &model_file, const RoomImplicitSceneOptions &options);
