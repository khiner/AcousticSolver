#pragma once

#include "DgParams.h"
#include "MetalContext.h"

#include <array>
#include <span>
#include <vector>

namespace dg {

struct Tables {
    DgParams Params{};
    std::vector<float> G, L, T, W, Initial, ReceiverWeights;
    std::vector<uint32_t> VmapM, VmapP, Boundary, ReceiverElements;
    std::array<float, 5> RkA{}, RkB{};
};

// FP32 factored mass with compact weak loads. Setup is performed in FP64.
class Gpu {
public:
    explicit Gpu(const Tables &);
    ~Gpu();
    Gpu(const Gpu &) = delete;
    Gpu &operator=(const Gpu &) = delete;
    void Reset(std::span<const float> initial, uint32_t steps);
    void Step(uint32_t sample, float dt);
    std::vector<float> State();
    std::vector<float> Receivers();
    std::vector<float> ApplyRhs(std::span<const float>);
    std::vector<float> ApplyMass(std::span<const float>);

private:
    void WeakLoad();
    void Mass();
    DgParams Params;
    std::array<float, 5> RkA, RkB;
    GpuBuffer G, L, T, W, Vm, Vp, Boundary, FaceRows, ReceiverElements, ReceiverWeights;
    GpuBuffer Q, Residual, Load, Weighted, Rhs, Record;
    MTL::Library *Library{};
    MTL::ComputePipelineState *Weak{}, *Interpolate{}, *Project{}, *Sample{};
};
} // namespace dg
