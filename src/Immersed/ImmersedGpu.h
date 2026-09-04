#pragma once

#include "ImmersedBoundary.h"
#include "ImmersedDelta.h"
#include "ImmersedParams.h"
#include "MetalContext.h"

#include <cstdint>
#include <vector>

namespace MTL {
class ComputePipelineState;
} // namespace MTL

namespace immersed {

struct Grid {
    int Nx{0}, Ny{0}, Nz{0};
    int PmlWidth{8};
    double H{0.};
    Point Origin{};
    double C{344.};
    double Rho{1.18};
    double Courant{0.575};
    double TimeStep{0.};
    std::vector<double> PmlNv, PmlDv, PmlNp, PmlDp;

    void Finalize();
    size_t NumCells() const { return size_t(Nx) * size_t(Ny) * size_t(Nz); }
};

class Gpu {
public:
    // source_samples is [step][source] and has NumSteps + 1 rows, since the
    // pressure forcing uses the average of adjacent source samples.
    void Init(const Grid &grid, const std::vector<Point> &sources, const std::vector<Point> &receivers, int num_steps, const std::vector<float> &source_samples, const std::vector<Patch> &patches = {}, int interpolation_order = 5, const std::vector<uint8_t> &solid = {});
    void RunSteps(int count);
    double RunTimedSteps(int count);

    int Step() const { return StepN; }
    int NumReceivers() const { return GP.NumReceivers; }
    const float *Samples() const { return ReceiverOut.As<float>(); }
    const float *Pressure() const { return P.As<float>(); }
    const float *VelocityX() const { return Vx.As<float>(); }
    const float *VelocityY() const { return Vy.As<float>(); }
    const float *VelocityZ() const { return Vz.As<float>(); }
    const ImmersedFilterState *VelocityStates() const { return ZvState.As<ImmersedFilterState>(); }
    void SeedPressure(const std::vector<float> &pressure);
    void SeedVelocity(const std::vector<float> &vx, const std::vector<float> &vy, const std::vector<float> &vz);
    double VelocityCondition() const { return VelocityConditionNumber; }
    double PressureCondition() const { return PressureConditionNumber; }

    struct Bandwidth {
        double Step;
        double Stream;
        double Bytes;
    };
    // Destructively compares the interior kernels with a matched field stream.
    Bandwidth Roofline(int repetitions);

private:
    Grid GridSpec;
    ImmersedGridParams GP{};
    int StepN{0};

    GpuBuffer P, Vx, Vy, Vz, Px, Py, Pz, Solid;
    GpuBuffer PmlNv, PmlDv, PmlNp, PmlDp;
    GpuBuffer SourceCells, SourceRefs, SourceSignal;
    GpuBuffer ReceiverBegin, ReceiverRefs, ReceiverOut;
    GpuBuffer ZvFilter, ZvState, YpFilter, YpState;
    GpuBuffer VelocityHistoryCells, VelocityHistoryRefs, PressureHistoryCells, PressureHistoryRefs;
    GpuBuffer VelocityPatchBegin, VelocityPatchRefs, PressurePatchBegin, PressurePatchRefs;
    GpuBuffer VelocityActiveBegin, VelocityActiveRefs;
    GpuBuffer PressureActiveBegin, PressureActiveRefs;
    GpuBuffer VelocityInverse, VelocityT, VelocitySolution, VelocityCorrectionCells, VelocityCorrectionRefs;
    GpuBuffer PressureInverse, PressureT, PressureY, PressureCorrectionCells, PressureCorrectionRefs;
    GpuBuffer Params;

    MTL::ComputePipelineState *PsoVelocity{}, *PsoPressure{}, *PsoSource{}, *PsoSample{};
    MTL::ComputePipelineState *PsoVelocityHistory{}, *PsoVelocityGather{}, *PsoVelocityScatter{},
        *PsoVelocityAdvance{};
    MTL::ComputePipelineState *PsoPressureHistory{}, *PsoPressureGather{}, *PsoPressureScatter{},
        *PsoPressureAdvance{}, *PsoDense{};
    MTL::ComputePipelineState *PsoStreamVelocity{}, *PsoStreamPressure{};
    Dim3 GridThreads{}, GridTiles{};
    double VelocityConditionNumber{1.}, PressureConditionNumber{1.};
};

} // namespace immersed
