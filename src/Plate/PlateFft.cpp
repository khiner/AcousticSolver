#include "PlateFft.h"
#include "PlateMetal.h"
#include "PlatePrecision.h"

#include <algorithm>
#include <bit>
#include <complex>
#include <stdexcept>

namespace plate {
namespace {
using Complex = std::complex<Precise>;
const char *Names[]{"PlateFftDerivativeX", "PlateFftDerivativeY", "PlateFftProjectX", "PlateFftAiryY", "PlateFftPhiX", "PlateFftPhiY", "PlateFftForceX", "PlateFftForceY"};

// Plan construction only. The native CPU solver remains independent of this FFT.
void Spectrum(std::vector<Complex> &a) {
    const auto n = a.size();
    const Precise pi = acos(Precise(-1));
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n / 2;
        for (; j & bit; bit /= 2) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t width = 2; width <= n; width *= 2)
        for (std::size_t base = 0; base < n; base += width)
            for (std::size_t j = 0; j < width / 2; ++j) {
                const Precise angle = -2 * pi * Precise(j) / Precise(width);
                const auto u = a[base + j], v = a[base + j + width / 2] * Complex(cos(angle), sin(angle));
                a[base + j] = u + v;
                a[base + j + width / 2] = u - v;
            }
}

std::size_t Lines(PlateFftStage stage, const PlateParams &p) {
    switch (stage) {
        case FftDerivativeX: return 2 * p.My;
        case FftDerivativeY: return 2 * p.Nx;
        case FftProjectX: return 2 * p.Ny;
        case FftAiryY: return 2 * (p.Mx + 1);
        case FftPhiX: return p.My + 1;
        case FftPhiY: return p.Nx;
        case FftForceX: return p.Ny;
        case FftForceY: return p.Mx;
    }
    throw std::invalid_argument("Invalid plate FFT stage");
}
} // namespace

void Fft::PreparePlan(Plan &plan, std::size_t nodes, std::size_t modes, double ratio, bool xAxis) {
    const Precise pi = acos(Precise(-1)), root = sqrt(Precise(ratio));
    const Precise wave = xAxis ? Precise(pi / root) : Precise(pi * root);
    auto &p = plan.Params;
    p.N = std::uint32_t(nodes);
    p.Length = std::uint32_t(std::has_single_bit(nodes) ? nodes : std::bit_ceil(2 * nodes - 1));
    p.Bits = std::countr_zero(p.Length);
    std::vector<Complex> table;
    for (std::size_t i = 0; i < p.Length / 2; ++i) {
        const Precise angle = -2 * pi * Precise(i) / p.Length;
        table.emplace_back(cos(angle), sin(angle));
    }
    p.Chirp = std::uint32_t(table.size());
    for (std::size_t i = 0; i < nodes; ++i) {
        const Precise angle = -pi * Precise((i * i) % (2 * nodes)) / Precise(nodes);
        table.emplace_back(cos(angle), sin(angle));
    }
    p.Phase = std::uint32_t(table.size());
    for (std::size_t i = 0; i < nodes; ++i) {
        const Precise angle = pi * Precise(i) / (2 * Precise(nodes));
        table.emplace_back(cos(angle), sin(angle));
    }
    p.Kernel = std::uint32_t(table.size());
    std::vector<Complex> kernel(p.Length);
    kernel[0] = 1;
    for (std::size_t i = 1; i < nodes && p.Length != nodes; ++i)
        kernel[i] = kernel[p.Length - i] = std::conj(table[p.Chirp + i]);
    Spectrum(kernel);
    table.insert(table.end(), kernel.begin(), kernel.end());
    p.Waves = std::uint32_t(table.size());
    for (std::size_t i = 0; i <= modes; ++i) {
        const Precise k = Precise(i) * wave;
        table.emplace_back(k, k * k);
    }
    p.AnalysisNorm = Pack(sqrt(Precise(2) / nodes));
    p.SynthesisNorm = Pack(1 / sqrt(Precise(2) * nodes));
    p.Sqrt2 = Pack(sqrt(Precise(2)));
    p.InvLength = Pack(Precise(1) / p.Length);
    std::vector<PlateReal> values;
    values.reserve(2 * table.size());
    for (const auto &z : table) {
        values.push_back(Pack(z.real()));
        values.push_back(Pack(z.imag()));
    }
    metal::Upload(plan.Table, values);
}

Fft::Fft(const Prepared &p, MTL::Library *library) {
    auto const &ctx = MetalContext::Get();
    PreparePlan(Plans[0], p.Nx, p.Mx, p.Settings.Ratio, true);
    PreparePlan(Plans[1], p.Ny, p.My, p.Settings.Ratio, false);
    std::size_t scratchBytes = 16;
    PlateParams dimensions{};
    dimensions.Mx = std::uint32_t(p.Mx);
    dimensions.My = std::uint32_t(p.My);
    dimensions.Nx = std::uint32_t(p.Nx);
    dimensions.Ny = std::uint32_t(p.Ny);
    for (std::uint32_t stage = 0; stage < Pipelines.size(); ++stage) {
        const auto channels = stage < 2 || stage == FftPhiX || stage == FftPhiY ? 3u : 1u;
        const auto bytes = std::size_t(Plans[stage % 2].Params.Length) * channels * 2 * sizeof(PlateReal);
        const bool shared = bytes <= ctx.Device->maxThreadgroupMemoryLength();
        SharedBytes[stage] = shared ? bytes : 0;
        if (!shared) scratchBytes = std::max(scratchBytes, bytes * Lines(PlateFftStage(stage), dimensions));
        auto const constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
        constants->setConstantValue(&stage, MTL::DataTypeUInt, NS::UInteger(0));
        constants->setConstantValue(&shared, MTL::DataTypeBool, NS::UInteger(1));
        Pipelines[stage] = metal::Pipeline(library, "PlateFftTransform", constants.get());
    }
    Scratch.Resize(scratchBytes);
}

Fft::~Fft() { MetalContext::Get().Drain(); }

void Fft::Dispatch(PlateFftStage stage, const GpuBuffer &input, GpuBuffer &output, GpuBuffer &derivatives, const GpuBuffer &coefficients, const GpuBuffer &active, const PlateParams &plate) {
    auto const &plan = Plans[stage % 2];
    auto params = plan.Params;
    params.Plate = plate;
    metal::Dispatch(Pipelines[stage].get(), Names[stage], {std::uint32_t(Lines(stage, plate))}, {256}, {&input, &output, &derivatives, &coefficients, &active, &plan.Table, &Scratch}, params, SharedBytes[stage]);
}

} // namespace plate
