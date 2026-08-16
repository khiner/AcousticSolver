#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (WaveBlender.h + GPUSolver.h) — Metal port.
//
// One merged solver: FDTD grid state and timestepping, rasterization, cavity detection,
// shader setup, fresh-cell extrapolation, and listeners.
//
// The batch loop is pipelined: the GPU executes batch N's FDTD steps while the CPU
// prepares batch N+1 (rasterization, shader ODE stepping, shader setup). The one hard
// sync point per batch is the fresh-cell velocity solve, which reads batch N's final
// velocities on the host. Buffers the CPU rewrites during the overlap window are
// double-buffered, and everything else is either GPU-written (ordered by the in-order
// queue) or written after the sync point.
//
// NOTE: Ordering of objects matters. Point sources MUST be placed at end; later objects
// will override previous rasterizations.

#include <deque>

#include "Shaders.h"

constexpr int CavityInterior{255};

// Mono-channel, stationary listener for single-point pressure output.
struct MonoListener {
    int Cid; // listener cell index
    std::ofstream Out;
};

class WaveBlender {
public:
    WaveBlender(const SimParams &params);

    // Runs a complete simulation batch: from rasterization to FDTD timestepping.
    // Returns false (after draining pending GPU work) once the simulation is complete.
    bool RunBatch();

    template<typename T, typename... Args> void AddObject(const Eigen::Vector3<REAL> &offset, Args &&...args) {
        Offsets.emplace_back(offset);
        Objects.emplace_back(std::in_place_type<T>, std::forward<Args>(args)...);
    }

    // Adds a listener. `format` is "Mono" only for now, `position` is the listener's (x, y, z).
    void AddListener(const std::string &format, const std::vector<REAL> &position, const std::string &output_name = "output");

    // Writes pressure and beta z-slice (z = Nz/2) to file for debugging.
    void LogZSlice(const std::string &filetag);

private:
    SimParams Params; // simulation parameters
    int Step{0}; // current simulation time step

    int GridSize{0}; // total number of cells (Nx * Ny * Nz)
    int NFdtdSamples{0}; // number of FDTD samples (timesteps) per batch
    int NShaderSamples{0}; // number of shader samples in time per batch
    int NShaderPoints{0}; // number of shader points (object boundary faces)
    int NRegularShaderPoints{0}; // leading shader points that are velocity blends (the rest are forces)
    int MaxNShaderPoints{0}; // maximum number of shader points allocated in memory so far

    // Precomputed constants
    REAL RhoCCDt{0}, InvDx{0}, InvRhoDt{0};
    REAL Damping{0}; // temporary solution in lieu of frequency-dependent boundary conditions

    std::deque<ObjectVariant> Objects; // deque: objects hold streams/trees and never move
    std::vector<Eigen::Vector3<REAL>> Offsets;
    std::vector<MonoListener> Listeners;
    bool ListenerPending{false}; // a batch's listener samples are on the GPU timeline, not yet written to file

    // ----- Device (GPU) buffers -----
    // While P[cid] and Beta[cid] values are stored at cell centers, velocities are staggered on
    // cell faces. By convention, each cell index stores the velocities on its RIGHT, TOP, and BACK faces.
    // P and the velocities ping-pong per fused step (read Cur(), write Other()), so the
    // current state is always Cur().
    GpuDouble P; // pressure
    GpuDouble Vx, Vy, Vz; // velocity components
    GpuBuffer Beta; // blending field
    GpuBuffer Px, Py, Pz; // split pressure field (for PML), also used as fresh-cell acceleration scratch
    GpuBuffer PmlNp, PmlDp, PmlNv, PmlDv; // PML weights (separate pressure and velocity)
    GpuBuffer ListenerCids, ListenerOut; // listener cell indices and per-batch sampled pressure

    // Rasterized cell states (which object each cell contains, 0 for air), the map from
    // boundary faces to shader points, and the shader sample data. All three are rewritten
    // by the CPU while the previous batch may still be in flight, hence double-buffered.
    // For boundary points (b1, ..., bN) and times (0, ..., T), ShaderData corresponds to:
    //   [ v_b1(0) ... v_b1(T) ]
    //   [   :            :    ]
    //   [ v_bN(0) ... v_bN(T) ]
    GpuDouble Cell, ShaderMap, ShaderData;

    // Cells changing solidity this batch, encoded (cid << 1) | new_solidity — the
    // per-step beta update set (see update_beta in Kernels.metal).
    GpuDouble BetaTransitions;
    int NBetaTransitions{0};
    std::vector<uint8_t> BetaSolid; // host-tracked beta state at batch start (exactly 0 or 1)

    std::vector<uint8_t> Cell1, Cell2; // rasterized cell states at batch endpoints t1 and t2 (values: 0 air, oid + 1, CavityInterior)

    // ----- Persistent scratch (sized once, reused per batch) -----
    std::vector<int> ShaderMapHost;
    std::vector<int> TransitionsHost;
    std::vector<uint8_t> FloodVisited; // flood-fill visited flags
    std::vector<int> FloodStack;
    std::vector<uint32_t> FaceStamp[3]; // epoch-stamped per-direction used-face sets (shader setup)
    uint32_t FaceEpoch{0};
    std::vector<int> FreshComponent; // per-cell fresh-cell component id (-1 when not fresh)
    std::vector<int> FaceCol[3]; // per-direction face column ids for the fresh-cell solve (-1 when unused)

    // ----- Basic helper functions -----
    int Cid(int i, int j, int k) const { return (Params.Ny * Params.Nx) * k + Params.Nx * j + i; }
    Eigen::Vector3<REAL> Pos(int i, int j, int k) const {
        return {(i - (Params.Nx - 1) / 2.f) * Params.Dx, (j - (Params.Ny - 1) / 2.f) * Params.Dx, (k - (Params.Nz - 1) / 2.f) * Params.Dx};
    }

    // ----- Rasterization -----
    void Rasterize(const bool (&keep_value)[256]);
    void SetupShaders();

    // ----- Per-batch overhead -----
    void DetectCavities();
    void FreshCellPressure();
    void FreshCellVelocity();
    void ShaderReInit();

    // ----- FDTD timestepping -----
    void RunFdtd(); // encodes all timesteps of the batch and flushes without waiting
    void InitializePml(); // allocates memory and precomputes constants for PML
    void InitializeListeners();
    void WritePendingListeners(); // writes the completed batch's listener samples to file
};
