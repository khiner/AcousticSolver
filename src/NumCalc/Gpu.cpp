#include "Backend.h"
#include "MetalContext.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace numcalc {
namespace {
struct PackedComplex {
    float Re, ReLow, Im, ImLow;
};
PackedComplex Pack(const Complex &z) {
    const float re = float(z.re()), im = float(z.im());
    if (!std::isfinite(re) || !std::isfinite(im)) throw std::runtime_error("NumCalc value exceeds Metal exponent range");
    return {re, float(z.re() - re), im, float(z.im() - im)};
}
struct Params {
    uint32_t Rows, Columns, Sparse, Add;
};
struct Packed {
    GpuBuffer Values, Offsets, Indices;
    Params Parameters;
    explicit Packed(const Matrix &a) : Parameters{uint32_t(a.Rows), uint32_t(a.Columns), a.Offsets != nullptr, 0} {
        if (a.Entries() > UINT32_MAX) throw std::runtime_error("Metal operator exceeds 32-bit indexing");
        Values.Resize(std::max<size_t>(1, a.Entries()) * sizeof(PackedComplex));
        auto *values = Values.As<PackedComplex>();
        for (size_t i = 0; i < a.Entries(); ++i) values[i] = Pack(a.Values[i]);
        Offsets.Resize(a.Offsets ? size_t(a.Rows + 1) * sizeof(int) : sizeof(int));
        Indices.Resize(std::max<size_t>(sizeof(int), a.Offsets ? a.Entries() * sizeof(int) : 0));
        if (a.Offsets) {
            Offsets.Upload(a.Offsets, size_t(a.Rows + 1) * sizeof(int));
            if (a.Entries()) Indices.Upload(a.Indices, a.Entries() * sizeof(int));
        }
    }
};

MTL::ComputePipelineState *Pipeline() {
    static auto const pipeline = [] {
        std::ifstream stream(ACOUSTIC_NUMCALC_MSL);
        if (!stream) throw std::runtime_error("Cannot read NumCalc Metal kernels");
        const std::string source{std::istreambuf_iterator<char>(stream), {}};
        auto const options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
        options->setFastMathEnabled(false);
        NS::Error *error{};
        auto const &context = MetalContext::Get();
        auto const library = NS::TransferPtr(context.Device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options.get(), &error));
        if (!library) throw std::runtime_error(error->localizedDescription()->utf8String());
        auto const function = NS::TransferPtr(library->newFunction(NS::String::string("Apply", NS::UTF8StringEncoding)));
        auto result = NS::TransferPtr(context.Device->newComputePipelineState(function.get(), &error));
        if (!result) throw std::runtime_error(error->localizedDescription()->utf8String());
        if (result->threadExecutionWidth() != 32) throw std::runtime_error("NumCalc requires 32-lane SIMD groups");
        return result;
    }();
    return pipeline.get();
}
} // namespace

struct Gpu::Impl {
    Packed Near;
    struct PackedLevel {
        Packed T, D, S;
        explicit PackedLevel(const Level &l) : T(l.T), D(l.D), S(l.S) {}
    };
    std::vector<std::unique_ptr<PackedLevel>> Levels;
    GpuBuffer X, Y, T, D;
    int Size;
    explicit Impl(const Operator &op) : Near(op.Near), Size(op.Near.Rows) {
        Pipeline();
        int nt = 1, nd = 1;
        for (const auto &l : op.Levels) {
            Levels.push_back(std::make_unique<PackedLevel>(l));
            nt = std::max(nt, l.T.Rows);
            nd = std::max(nd, l.D.Rows);
        }
        X.Resize(size_t(Size) * sizeof(PackedComplex));
        Y.Resize(size_t(Size) * sizeof(PackedComplex));
        T.Resize(size_t(nt) * sizeof(PackedComplex));
        D.Resize(size_t(nd) * sizeof(PackedComplex));
    }
    void Apply(const Packed &a, const GpuBuffer &x, const GpuBuffer &y, bool add = false) {
        auto parameters = a.Parameters;
        parameters.Add = add;
        MetalContext::Get().Dispatch(Pipeline(), {parameters.Rows, 1, 1}, {32, 1, 1}, {&a.Values, &a.Offsets, &a.Indices, &x, &y}, &parameters, sizeof(parameters));
    }
};

Gpu::Gpu(const Operator &op) : Data(std::make_unique<Impl>(op)) {}
Gpu::~Gpu() = default;
void Gpu::Apply(const Complex *x, Complex *y) {
    auto *input = Data->X.As<PackedComplex>();
    for (int i = 0; i < Data->Size; ++i) input[i] = Pack(x[i]);
    Data->Apply(Data->Near, Data->X, Data->Y);
    for (const auto &l : Data->Levels) {
        Data->Apply(l->T, Data->X, Data->T);
        Data->Apply(l->D, Data->T, Data->D);
        Data->Apply(l->S, Data->D, Data->Y, true);
    }
    const auto *output = Data->Y.As<PackedComplex>();
    for (int i = 0; i < Data->Size; ++i) y[i].set(double(output[i].Re) + output[i].ReLow, double(output[i].Im) + output[i].ImLow);
}
} // namespace numcalc
