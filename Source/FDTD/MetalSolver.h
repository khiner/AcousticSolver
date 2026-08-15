#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (GPUSolver.h) — Metal port.
// Rasterization and object-level logic live in WaveBlender.h/cpp.
// Kernel implementations live in Kernels.metal.

#include "FDTDCommon.h"
#include "KernelParams.h"
#include "Listener.h"

// Base Metal FDTD acoustic wavesolver.
class FDTDSolver {
public:
    FDTDSolver(const SimParams &params);
    ~FDTDSolver();

    // Runs FDTD simulation for `NFdtdSamples` timesteps.
    void RunFdtd();

    // Adds a listener. `format` is "Mono" only for now, `position` is the listener's (x, y, z).
    void AddListener(const std::string &format, const std::vector<REAL> &position, const std::string &output_name = "output");

    // Writes pressure and beta z-slice (z = Nz/2) to file for debugging.
    void LogZSlice(const std::string &filetag);

protected:
    SimParams Params; // simulation parameters
    int Step{0}; // current simulation time step

    int GridSize{0}; // total number of cells (Nx * Ny * Nz)
    int NFdtdSamples{0}; // number of FDTD samples (timesteps) per batch

    // ----- Device (GPU) buffers -----
    // While P[cid] and Beta[cid] values are stored at cell centers, velocities are staggered on
    // cell faces. By convention, each cell index stores the velocities on its RIGHT, TOP, and BACK faces.
    GpuBuffer P; // pressure
    GpuBuffer Vx, Vy, Vz; // velocity components
    GpuBuffer Beta; // blending field
    GpuBuffer Cell; // rasterized cell states, i.e., which object each cell contains (0 for air)
    GpuBuffer Px, Py, Pz; // split pressure field (for PML)
    GpuBuffer PmlNp, PmlDp, PmlNv, PmlDv; // PML weights (separate pressure and velocity)

    // Map from boundary cell to shader velocity (or force) in ShaderData, and the linear array of
    // all shader samples for the current batch. For boundary points (b1, ..., bN) and times (0, ..., T),
    // ShaderData corresponds to:
    //   [ v_b1(0) ... v_b1(T) ]
    //   [   :            :    ]
    //   [ v_bN(0) ... v_bN(T) ]
    GpuBuffer ShaderMap;
    GpuBuffer ShaderData;
    int NShaderSamples{0}; // number of shader samples in time per batch
    int NShaderPoints{0}; // number of shader points (object boundary faces)
    int MaxNShaderPoints{0}; // maximum number of shader points allocated in memory so far

    // Precomputed constants
    REAL RhoCCDt, InvDx, InvRhoDt;

    // ----- Basic helper functions -----
    int Cid(int i, int j, int k) const { return (Params.Ny * Params.Nx) * k + Params.Nx * j + i; }
    int Cid(const Eigen::Vector3<REAL> &p) const {
        const int i = (p[0] / Params.Dx) + (Params.Nx - 1) / 2.;
        const int j = (p[1] / Params.Dx) + (Params.Ny - 1) / 2.;
        const int k = (p[2] / Params.Dx) + (Params.Nz - 1) / 2.;
        return Cid(i, j, k);
    }
    Eigen::Vector3<REAL> Pos(int i, int j, int k) const {
        return {(i - (Params.Nx - 1) / 2.f) * Params.Dx, (j - (Params.Ny - 1) / 2.f) * Params.Dx, (k - (Params.Nz - 1) / 2.f) * Params.Dx};
    }
    bool InBounds(int i, int j, int k) const { return i >= 0 && i < Params.Nx && j >= 0 && j < Params.Ny && k >= 0 && k < Params.Nz; }

private:
    REAL Damping{0}; // temporary solution in lieu of frequency-dependent boundary conditions

    std::vector<Listener *> Listeners;

    // Allocates memory and precomputes constants for PML.
    void InitializePml();
};
