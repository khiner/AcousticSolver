#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (FDTDCommon.h). Global FDTD simulation parameters.

using REAL = float;

enum class BlendScheme {
    NoBlend,
    Smoothstep,
};

struct SimParams {
    int Nx{80}, Ny{80}, Nz{80}; // FDTD grid dimensions
    REAL Dx{0.005}; // cell size

    int FdtdSrate{120'000}; // FDTD sample rate
    double Dt{1. / 120'000}; // FDTD timestep size
    double Ts{0}, Tf{0}; // simulation start and end times

    int ShaderSrate{48'000}; // shader sample rate
    int BlendRate{100}; // blending rate (how often to rasterize geometry)
    BlendScheme Scheme{BlendScheme::Smoothstep};

    REAL C{343.2}; // speed of sound
    REAL Rho{1.204}; // density of acoustic medium

    REAL Damping{0}; // frequency-independent air damping (see reference TODO on boundary conditions)
};
