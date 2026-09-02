#pragma once

// GPU stepper for the room-acoustics solver, stepping the scheme documented in
// RoomKernels.metal over a RoomScene as the voxelizer left it.
//
// There is no per-step host round trip: the source signals and every table are uploaded once,
// receiver samples land in a device-side buffer for the whole run, and the two pressure levels
// rotate by swapping buffer indices.

#include "RoomParams.h"
#include "RoomScene.h"

#include "MetalContext.h"

namespace MTL {
class ComputePipelineState;
} // namespace MTL

// The interior update's coefficients, computed in double and narrowed to float once, as the
// reference does: A1 = 2 - Sl2*NN, A2 = f*l^2, Sl2 = (1 + eps)*f*l^2 with eps the fp32
// diagonal shift, NN the stencil's neighbour count and f the Laplacian factor — 1 and 6 on the
// Cartesian stencil, 1/4 and 12 on the FCC one. The scheme that steps is the one with *these*
// numbers in it, so its dispersion relation is written in terms of them and not the ideal l^2.
// Their rounding is worth a part in 10^5 at the lowest modes of a room-sized box.
struct RoomCoefficients {
    float A1, A2, Sl2, L;
};
RoomCoefficients RoomUpdateCoefficients(const RoomScene &);

// The impedance boundary's material coefficients, from the scene's (D, E, F) triples and its
// time step as the reference derives them: with Dh = D/Ts, Eh = E, Fh = F*Ts,
//
//   b = 1/(2 Dh + Eh + Fh/2),  bd = b (2 Dh - Eh - Fh/2),  bDh = b Dh,  bFh = b Fh
//
// and Beta[k] the sum of material k's b over its branches, accumulated in float because that
// sum is what the node update divides by. Quads is padded to RoomMaxBranches a material, so
// material k's branch m is Quads[k*RoomMaxBranches + m].
struct RoomMaterials {
    std::vector<RoomMatQuad> Quads;
    std::vector<float> Beta;
};
RoomMaterials RoomMaterialCoefficients(const RoomScene &);

// The boundary nodes that carry a material, in boundary-node order — the reference's bnl
// arrays. On the FCC grid the surface areas are rescaled by 1/(2 sqrt 2): the boundary term is
// S h / V, and an FCC node's cell has twice a Cartesian one's volume and a face diagonal's
// length for its edge.
struct RoomLossyNodes {
    std::vector<int> Ixyz;
    std::vector<float> Ssaf;
    std::vector<int8_t> Mat;
};
RoomLossyNodes RoomLossySubset(const RoomScene &);

struct RoomGpu {
    // Uploads the scene. The absorbing shell is not among the tables: the stencil kernels find
    // it from the node's position.
    void Init(const RoomScene &);

    void RunSteps(int n_steps);
    int Step() const { return StepN; }

    // Receiver-corner samples for the whole run, [step][corner]. Synchronizing.
    const float *Samples() const { return Out.As<float>(); }
    // The current level, u(StepN), over the whole grid. Synchronizing.
    const float *Level() const { return U[I1].As<float>(); }

    // Seeds both levels with `u` (one value a node) and returns the boundary state to rest
    // with them, which starts the scheme from rest at that displacement. For the rungs that
    // need an excitation injecting nothing after step 0.
    void Seed(const std::vector<float> &u);

    // The per-branch impedance state of the lossy boundary nodes, branch-major: branch m of
    // node nb is [m*Nbl + nb]. V is the branch velocity at the step just computed and G its
    // running trapezoidal integral, which together carry the wall's stored energy.
    // Synchronizing.
    const float *BranchV() const { return Vh1.As<float>(); }
    const float *BranchG() const { return Gh1.As<float>(); }
    int LossyNodes() const { return GP.Nbl; }

    // Discrete energy over the inclusive node box, summed in double on the host from the
    // per-node partials the GPU writes. The box has to be closed under the scheme's adjacency.
    // See RoomEnergy in RoomKernels.metal.
    double Energy(int x0, int x1, int y0, int y1, int z0, int z1);

    // Per-kernel GPU busy time, when ACOUSTIC_ROOM_KERNEL_TIMES is set: each dispatch gets
    // its own command buffer and is waited out. Serializing costs wall time, so this is a
    // diagnostic run, not a mode to measure the solver in.
    struct KernelTimes {
        const char *Name;
        double Seconds;
        uint64_t Count;
    };
    const std::vector<KernelTimes> &Timings() const { return KernelTiming; }
    static void ReportTimings(const std::vector<KernelTimes> &);

    // Destructive update-versus-stream roofline measurement.
    struct Bandwidth {
        double Update, Stream; // seconds a dispatch
        double Bytes; // per dispatch, the traffic neither can avoid
    };
    Bandwidth Roofline(int reps);

    // Measures the scattered boundary pass against a matching eight-stream ceiling.
    Bandwidth LossyRoofline(int reps);

private:
    RoomGridParams GP{};
    bool Fcc{false};
    int StepN{0};

    // Two pressure levels, rotating by index: I1 is u(n), I0 is u(n-1) and receives u(n+1).
    GpuBuffer U[2];
    int I0{0}, I1{1};

    // Cartesian masks are grid-wide; FCC masks are packed and reached through block rank.
    GpuBuffer BnMask, AdjBn;
    GpuBuffer InIxyz, InSigs;
    GpuBuffer OutIxyz, Out;
    GpuBuffer Params, EnergyOut;

    // Impedance boundary. Ub is a three-deep ring: the value two steps back is gone from the
    // grid by the time the branch update wants it, so the lossy nodes keep their own history.
    GpuBuffer BnlIxyz, SsafBnl, MatBnl, MatMb, MatBeta, MatQuads;
    GpuBuffer Ub[3], Vh1, Gh1;
    int I0b{0}, I1b{1}, I2b{2};

    MTL::ComputePipelineState *PsoAir{}, *PsoLossy{}, *PsoIo{}, *PsoEnergy{};

    Dim3 AirTiles{}, AirThreads{};
    double LossyBytes{0.};
    bool TimeKernels{false};
    std::vector<KernelTimes> KernelTiming;

    // Dispatches, and when timing is on isolates the dispatch in its own command buffer and
    // charges its GPU time to `name`.
    void Timed(int slot, const char *name, MTL::ComputePipelineState *, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice>, const void *params = nullptr, size_t params_size = 0);

    // Discards one warm-up round to avoid GPU clock-ramp bias.
    double TimePass(MTL::ComputePipelineState *, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice>, int reps);
};
