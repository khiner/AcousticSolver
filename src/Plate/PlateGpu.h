#pragma once

#include "MetalContext.h"
#include "Plate.h"
#include "PlateParams.h"

#include <Foundation/NSSharedPtr.hpp>
#include <array>
#include <memory>

namespace plate {

class Fft;
enum class TransformBackend { Dense,
                              Fft };

struct GpuTrace {
    // Samples and optional post-update modal states are frame-major.
    std::vector<double> Samples, States;
    std::vector<Diagnostics> Metrics;
};

// 256-bit arithmetic throughout the GPU; FP64 readback is for exported results.
// State and all nonlinear transforms/reductions remain on the GPU between
// batches.
class Gpu {
public:
    explicit Gpu(const Prepared &, TransformBackend = TransformBackend::Dense);
    ~Gpu();
    Gpu(const Gpu &) = delete;
    Gpu &operator=(const Gpu &) = delete;
    void Reset(const State &);
    void ResetInitial(double amplitude = 0);
    State GetState();
    Potential EvaluatePotential(std::span<const double> q);
    // Empty forces means no forcing. Otherwise steps * Mx * My entries.
    // Pickup weights are receiver-major, with Mx * My entries per receiver.
    GpuTrace Run(std::span<const double> forces, std::span<const double> pickupWeights, std::uint32_t steps, bool recordStates = false);

    GpuTrace RunPrepared(std::span<const PlateReal> forces, std::span<const PlateReal> pickupWeights, std::uint32_t steps, bool recordStates = false);

private:
    void Dispatch(std::size_t kernel, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const PlateParams &params, std::size_t threadgroupBytes = 0);
    void Nonlinearity(const GpuBuffer &state, bool forceNonlinear = false);
    void FinishGradient(const PlateParams &);
    Prepared Tables;
    PlateParams BaseParams;
    std::unique_ptr<Fft> FastTransforms;
    std::uint64_t StepIndex{};
    GpuBuffer StateBuffer, EvaluationState, Coefficients, Active, AiryActive;
    GpuBuffer Sx, Sy, Cx, Cy, Ux, Uy, Px, Py, AiryCoefficients;
    GpuBuffer DerivativeX, Derivatives, ProjectionX, Airy;
    GpuBuffer Bracket, PhiBracket;
    GpuBuffer PhiX, ForceX, Gradient, Coupling, Scalars;
    GpuBuffer Forces, Pickups, Samples, Records, States;
    std::array<NS::SharedPtr<MTL::ComputePipelineState>, 8> Pipelines{};
};

} // namespace plate
