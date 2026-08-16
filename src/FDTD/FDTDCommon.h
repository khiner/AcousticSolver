#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (FDTDCommon.h). Global FDTD simulation parameters.

using real = float;

enum class BlendScheme {
    NoBlend,
    Smoothstep,
};

struct SimParams {
    int Nx{80}, Ny{80}, Nz{80}; // FDTD grid dimensions
    real Dx{0.005}; // cell size

    int FdtdSrate{120'000}; // FDTD sample rate
    double Dt{1. / 120'000}; // FDTD timestep size
    double Ts{0}, Tf{0}; // simulation start and end times

    int ShaderSrate{48'000}; // shader sample rate
    int BlendRate{100}; // blending rate (how often to rasterize geometry)
    BlendScheme Scheme{BlendScheme::Smoothstep};

    real C{343.2}; // speed of sound
    real Rho{1.204}; // density of acoustic medium

    real Damping{0}; // frequency-independent air damping (see reference TODO on boundary conditions)
};
