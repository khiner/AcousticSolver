#pragma once

#include "MetalContext.h"
#include "Plate.h"
#include "PlateParams.h"

#include <Foundation/NSSharedPtr.hpp>
#include <array>

namespace plate {

// FFT-based orthonormal type-II sine/cosine transforms on the unchanged grid.
// Plans and chirp spectra are prepared once; every transform executes on Metal.
class Fft {
public:
    Fft(const Prepared &, MTL::Library *);
    ~Fft();
    Fft(const Fft &) = delete;
    Fft &operator=(const Fft &) = delete;
    void Dispatch(PlateFftStage, const GpuBuffer &input, GpuBuffer &output, GpuBuffer &derivatives, const GpuBuffer &coefficients, const GpuBuffer &active, const PlateParams &);

private:
    struct Plan {
        PlateFftParams Params{};
        GpuBuffer Table;
    };
    static void PreparePlan(Plan &, std::size_t nodes, std::size_t modes, double ratio, bool xAxis);
    std::array<Plan, 2> Plans;
    std::array<NS::SharedPtr<MTL::ComputePipelineState>, 8> Pipelines{};
    std::array<std::size_t, 8> SharedBytes{};
    GpuBuffer Scratch;
};

} // namespace plate
