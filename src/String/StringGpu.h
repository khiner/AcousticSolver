#pragma once
#include "MetalContext.h"
#include "String.h"
#include "StringParams.h"
#include <Foundation/NSSharedPtr.hpp>

namespace stiffstring {
struct GpuTrace {
    // Post-update receiver samples and optional states, both frame-major.
    std::vector<double> Samples, States;
    std::vector<Diagnostics> Metrics;
};

class Gpu {
public:
    explicit Gpu(const Prepared &prepared);
    ~Gpu();
    Gpu(const Gpu &) = delete;
    Gpu &operator=(const Gpu &) = delete;
    // Reconstructs the private displacement increment from Q - Previous.
    // Run partitioning preserves it exactly; an exported State is not a bit-exact GPU checkpoint.
    void Reset(const State &state);
    State GetState();
    GpuTrace Run(std::uint32_t steps, bool recordStates = false);

private:
    Prepared Tables;
    StringParams Params{};
    std::uint64_t StepIndex{};
    GpuBuffer StateBuffer, Scratch, Scalars, Samples, Records, States;
    NS::SharedPtr<MTL::ComputePipelineState> Pipeline;
};
} // namespace stiffstring
