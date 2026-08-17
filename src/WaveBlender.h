#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (WaveBlender.h + GPUSolver.h) — Metal port.
//
// One merged solver: FDTD grid state and timestepping, rasterization, cavity detection,
// shader setup, fresh-cell extrapolation, and listeners.
//
// The batch loop is pipelined: the GPU executes batch N's FDTD steps while the CPU
// prepares batch N+1 (rasterization, shader ODE stepping, shader setup, and the
// fresh-cell system factorization — its matrix depends only on cell geometry). Batch
// N+1's FDTD steps are encoded pre-sync too, into a deferred command buffer. The one
// hard sync point per batch is the fresh-cell velocity solve, which reads batch N's
// final velocities on the host (after the prologue kernels clear interior velocities);
// the deferred buffer commits right after those host writes land. Buffers the CPU
// rewrites during the overlap window are double-buffered, and everything else is either
// GPU-written (ordered by the in-order queue) or written after the sync point.
//
// NOTE: Ordering of objects matters. Point sources MUST be placed at end; later objects
// will override previous rasterizations.

#include "Shaders.h"

#include <algorithm>
#include <climits>
#include <deque>

constexpr int CavityInterior{255};

// Inclusive cell-index bounds of a grid region, used to confine per-batch CPU sweeps to
// the cells an object (or the cavity fill) can actually touch. Default-initialized empty.
struct CellBox {
    int Min[3]{INT_MAX, INT_MAX, INT_MAX};
    int Max[3]{-1, -1, -1};

    bool Empty() const { return Max[0] < Min[0] || Max[1] < Min[1] || Max[2] < Min[2]; }
    void Expand(const CellBox &o) {
        for (int d = 0; d < 3; ++d) {
            Min[d] = std::min(Min[d], o.Min[d]);
            Max[d] = std::max(Max[d], o.Max[d]);
        }
    }
    void Clamp(int nx, int ny, int nz) {
        const int n[3]{nx, ny, nz};
        for (int d = 0; d < 3; ++d) {
            Min[d] = std::max(Min[d], 0);
            Max[d] = std::min(Max[d], n[d] - 1);
        }
    }
};

// Calls fn(cid, i, j, k) for every cell in the box, in ascending cid order (several
// sweeps' float-rounding depends on visiting cells in this order).
template<typename F> void ForEachCell(const CellBox &box, int nx, int ny, F &&fn) {
    for (int k = box.Min[2]; k <= box.Max[2]; ++k) {
        for (int j = box.Min[1]; j <= box.Max[1]; ++j) {
            int cid = (ny * nx) * k + nx * j + box.Min[0];
            for (int i = box.Min[0]; i <= box.Max[0]; ++i, ++cid) fn(cid, i, j, k);
        }
    }
}

// Mono-channel, stationary listener for single-point pressure output.
struct MonoListener {
    int Cid;
    std::ofstream Out;
};

struct WaveBlender {
    WaveBlender(const SimParams &params);

    // Runs a complete simulation batch: from rasterization to FDTD timestepping.
    // Returns false (after draining pending GPU work) once the simulation is complete.
    bool RunBatch();

    template<typename T, typename... Args> void AddObject(const Eigen::Vector3<real> &offset, Args &&...args) {
        Offsets.emplace_back(offset);
        Objects.emplace_back(std::in_place_type<T>, std::forward<Args>(args)...);
    }

    // Adds a listener. `format` is "Mono" only for now, `position` is the listener's (x, y, z).
    void AddListener(const std::string &format, const std::vector<real> &position, const std::string &output_name = "output");

    // Writes pressure and beta z-slice (z = Nz/2) to file for debugging.
    void LogZSlice(const std::string &filetag);

private:
    SimParams Params;
    int Step{0}; // current simulation time step

    int GridSize{0}; // total number of cells (Nx * Ny * Nz)
    Dim3 FdtdTg; // threadgroup tile dims of the full-grid step kernels (per-scene, see ctor)
    Dim3 FdtdTiles; // threadgroup counts of the full-grid dispatch
    int NFdtdSamples{0}; // number of FDTD samples (timesteps) per batch
    int NShaderSamples{0}; // number of shader samples in time per batch
    int NShaderPoints{0}; // number of shader points (object boundary faces)
    int NRegularShaderPoints{0}; // leading shader points that are velocity blends (the rest are forces)
    int MaxNShaderPoints{0}; // maximum number of shader points allocated in memory so far
    bool FoldApply{false}; // fold the boundary application into the velocity writes

    // Chooses the boundary-application path per batch. Both are bit-exact, so probe batches
    // alternate them and compare measured GPU time — an in-run A/B, immune to the thermal
    // drift that makes cross-run comparison unreliable here.
    struct ApplyPathTuner {
        bool Choose(bool has_points, bool has_forces, int npoints); // the path for this batch (true: folded)
        void MarkInFlight(bool fold); // this batch is a probe: measure it at the next drain
        void RecordDrained(double gpu_seconds); // the last-marked probe batch just drained

    private:
        uint32_t Key{~0u}; // dispatch-structure key the probe state applies to
        int Locked{-1}; // -1 while probing, else the chosen path (0 plain, 1 folded)
        int LockedPoints{0}; // shader-point count when the lock was taken (drift re-opens probing)
        double BestSeconds[2]{}; // best measured batch GPU seconds per path
        int ProbeCount[2]{}; // completed probe batches per path
        int InFlightFold{-1}; // path of the in-flight probe batch (-1: none)
        bool SeenForces{false}; // sticky: any batch so far had force points
    };
    ApplyPathTuner PathTuner;

    // Reported under ACOUSTIC_PROFILE. The flagged-tile fraction is what the fold costs:
    // unflagged tiles read no face mask.
    struct FaceLoad {
        int PeakPoints{0};
        uint64_t Batches{0}, FlaggedTiles{0}, TotalTiles{0};
        void Report() const;
    };
    FaceLoad Faces;

    // Precomputed constants
    real RhoCcDt{0}, InvDx{0}, InvRhoDt{0};
    real Damping{0}; // temporary solution in lieu of frequency-dependent boundary conditions

    std::deque<ObjectVariant> Objects; // deque: objects hold streams/trees and never move
    std::vector<Eigen::Vector3<real>> Offsets;
    std::vector<MonoListener> Listeners;
    bool ListenerPending{false}; // a batch's listener samples are on the GPU timeline, not yet written to file

    // ----- Device (GPU) buffers -----
    // While P[cid] and cell states are stored at cell centers, velocities are staggered on
    // cell faces. By convention, each cell index stores the velocities on its RIGHT, TOP, and BACK faces.
    // P and the velocities ping-pong per fused step (read Cur(), write Other()), so the
    // current state is always Cur().
    GpuDouble P; // pressure
    GpuDouble Vx, Vy, Vz; // velocity components
    GpuBuffer Px, Py, Pz; // fresh-cell acceleration scratch (interior cells only)
    GpuBuffer PsiX, PsiY, PsiZ; // split pressure field (PML), shell-packed (PsiIndex in Kernels.metal)
    GpuBuffer PmlNp, PmlDp, PmlNv, PmlDv; // PML weights (separate pressure and velocity)
    GpuBuffer ListenerCids; // listener cell indices
    // Per-batch sampled pressure, double-buffered so a drained batch's samples can be
    // written to file after the next batch (which fills the other slot) is released.
    GpuDouble ListenerOut;
    const GpuBuffer *PendingListenerSlot{nullptr}; // slot holding the pending batch's samples

    // Per-cell blending state for the batch (the Cell* constants in KernelParams.h), the map from
    // boundary faces to shader points, and the shader sample data. All rewritten by the
    // CPU while the previous batch may still be in flight, hence double-buffered.
    // For boundary points (b1, ..., bN) and times (0, ..., T), ShaderData corresponds to:
    //   [ v_b1(0) ... v_b1(T) ]
    //   [   :            :    ]
    //   [ v_bN(0) ... v_bN(T) ]
    GpuDouble CellState, ShaderMap, ShaderData;

    // Per-cell lookup for the folded boundary application (layout in KernelParams.h)
    GpuDouble ShaderFaces;

    bool HadTransitions{false}; // last uploaded states include Rising/Falling cells

    std::vector<uint8_t> Cell1, Cell2; // rasterized cell states at batch endpoints t1 and t2 (values: 0 air, oid + 1, CavityInterior)

    // Cell bounds of each object's rasterization and of the cavity fill, for Cell2 (as
    // of the last rasterization) and Cell1 (the previous batch's snapshot). Every cell
    // holding an object's value lies within its box.
    std::vector<CellBox> ObjectBounds, PrevObjectBounds;
    CellBox CavityBounds, PrevCavityBounds;

    // ----- Persistent scratch (sized once, reused per batch) -----
    std::vector<int> ShaderMapHost;
    std::vector<int> FaceCells; // cells holding a boundary face this batch, in first-touch order
    std::vector<uint8_t> FloodVisited;
    std::vector<int> FloodStack;
    std::vector<uint32_t> FaceStamp[3]; // epoch-stamped per-direction used-face sets (shader setup)
    uint32_t FaceEpoch{0};
    std::vector<int> FreshComponent; // per-cell fresh-cell component id (-1 when not fresh)
    std::vector<int> FaceCol[3]; // per-direction face column ids for the fresh-cell solve (-1 when unused)

    // One fresh-cell component's least-squares system, factorized pre-sync (the matrix
    // depends only on Cell1/Cell2) while the GPU still runs the previous batch. The
    // post-sync solve builds the right-hand side from the drained velocities, solves
    // against the cached decomposition, and scatters.
    struct FreshComponentSolve {
        std::vector<int> Cells; // component cells, ascending cell order
        std::vector<int> Faces[3]; // per-direction faces, in column order
        int Counts[3]{}; // per-direction column counts
        Eigen::MatrixX<real> A; // zero-net-flux system (kept for the residual report)
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixX<real>> Cod;
    };
    std::vector<FreshComponentSolve> FreshSolves;
    size_t NFreshCells{0};
    real *FreshV[3]{}; // Vx/Vy/Vz host pointers captured at analysis time (before the encode ping-pong flips)

    int Cid(int i, int j, int k) const { return (Params.Ny * Params.Nx) * k + Params.Nx * j + i; }
    // Cell bounds of a vertex set's rasterization (padded by `pad` cells, grid-clamped)
    CellBox VertexBounds(const Eigen::MatrixX<real> &v, const Eigen::Vector3<real> &offset, int pad) const;
    // Union of every cell that can be solid in Cell2 (and, when `with_prev`, in Cell1)
    CellBox SolidBounds(bool with_prev) const;
    Eigen::Vector3<real> Pos(int i, int j, int k) const {
        return {(i - (Params.Nx - 1) / 2.f) * Params.Dx, (j - (Params.Ny - 1) / 2.f) * Params.Dx, (k - (Params.Nz - 1) / 2.f) * Params.Dx};
    }

    // ----- Rasterization -----
    void Rasterize(const bool (&keep_value)[256]);
    void SetupShaders();

    // ----- Per-batch overhead -----
    void DetectCavities();
    void FreshCellPressure();
    void AnalyzeFreshCells(); // pre-sync: find components, build + factorize their systems
    void SolveFreshCells(); // post-sync: right-hand sides from drained velocities, solve, scatter
    void ShaderReInit();

    // ----- FDTD timestepping -----
    void RunFdtd(); // encodes all timesteps of the batch (into the deferred command buffer)
    void InitializePml();
    void InitializeListeners();
    void WritePendingListeners();
};
