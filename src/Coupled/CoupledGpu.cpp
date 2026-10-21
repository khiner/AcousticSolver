#include "Coupled.h"
#include "Metal/MetalContext.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <unordered_map>

namespace coupled {
namespace {
Real Pack(double x) {
    Real value{float(x), float(x - double(float(x)))};
    if (!std::isfinite(x) || !std::isfinite(value.Hi) || (x && !value.Hi))
        throw std::runtime_error("Coupled coefficient exceeds Metal float-pair range");
    return value;
}
std::string Read(const std::string &file) {
    std::ifstream input(file);
    if (!input) throw std::runtime_error("Cannot read " + file);
    return {std::istreambuf_iterator<char>(input), {}};
}
void Upload(GpuBuffer &b, const std::vector<double> &values) {
    b.Resize(std::max<size_t>(1, values.size()) * sizeof(Real));
    std::ranges::transform(values, static_cast<Real *>(b.RawData()), Pack);
}
void Download(std::vector<double> &output, const GpuBuffer &b, size_t offset, size_t count) {
    const auto *values = b.As<Real>() + offset;
    for (size_t i = 0; i < count; ++i) {
        const double value = double(values[i].Hi) + values[i].Lo;
        if (!std::isfinite(value)) throw std::runtime_error("Coupled Metal state became nonfinite");
        output.push_back(value);
    }
}
struct DevicePanel {
    PanelParams Params;
    GpuBuffer Interface, Mask;
};
} // namespace

Trace RunMetal(const Tables &t) {
    auto &context = MetalContext::Get();
    context.Drain();
    const auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    options->setFastMathEnabled(false);
    const std::string root = ACOUSTIC_PROJECT_ROOT;
    const std::string source = Read(root + "/src/String/StringReal.h") + "\n" +
        Read(root + "/src/Coupled/CoupledParams.h") + "\n" + Read(root + "/src/Coupled/CoupledKernels.metal");
    NS::Error *error{};
    auto const library = NS::TransferPtr(context.Device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options.get(), &error));
    if (!library) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "Coupled Metal compilation failed");
    std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> kernels;
    for (const char *name : {"CoupledPanel", "CoupledAir", "CoupledReverse", "CoupledForward", "CoupledInterpolate", "CoupledSample", "CoupledBoundary"}) {
        auto const function = NS::TransferPtr(library->newFunction(NS::String::string(name, NS::UTF8StringEncoding)));
        if (!function) throw std::runtime_error(std::string("Missing kernel ") + name);
        auto pipeline = NS::TransferPtr(context.Device->newComputePipelineState(function.get(), &error));
        if (!pipeline) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "Coupled pipeline creation failed");
        if (pipeline->maxTotalThreadsPerThreadgroup() < 256 || pipeline->threadExecutionWidth() != 32) throw std::runtime_error("Coupled solver requires 256 threads per group and 32 threads per SIMD group");
        kernels.emplace(name, std::move(pipeline));
    }
    const uint32_t cells = t.Nx * t.Ny * t.Nz, slice = t.Nx * t.Ny;
    const uint32_t channels = uint32_t(t.Receivers.size()), airReceivers = channels - uint32_t(t.Plates.size());
    GpuBuffer air[3], indices, values, coefficients, airIndex, receivers, output, forces, state, status, panelParams;
    for (auto &b : air) b.ResizeZeroed(size_t(cells) * sizeof(Real));
    std::vector<uint32_t> ix = t.Indices, receiverIndices;
    for (uint32_t i = 0; i < airReceivers; ++i) receiverIndices.push_back(t.Receivers[i].Index);
    receivers.Resize(receiverIndices.size() * sizeof(uint32_t));
    receivers.Upload(receiverIndices.data(), receivers.Capacity());
    Upload(values, t.Values);
    Upload(coefficients, t.AirCoefficients);
    airIndex.Resize(t.AirIndex.size());
    airIndex.Upload(t.AirIndex.data(), t.AirIndex.size());
    output.Resize(size_t(t.Steps) * channels * sizeof(Real));
    std::vector<DevicePanel> panels(t.Plates.size());
    std::vector<double> forceValues;
    Trace trace;
    uint32_t stateSize = 0, solves = 0;
    for (size_t index = 0; index < t.Plates.size(); ++index) {
        const auto &p = t.Plates[index];
        auto &gpu = panels[index];
        auto &s = gpu.Params;
        s.N = p.Size;
        s.StateOffset = stateSize;
        stateSize += 24 * p.Size + 1;
        s.Iterations = p.Iterations;
        s.Stride = p.Stride;
        s.Nonlinear = p.Nonlinear;
        s.Adjoint = p.Adjoint;
        s.MassInverse = p.MassInverse;
        s.MaxIterations = t.MaxIterations;
        // Limit dense matvec work per segment, retaining at least four iterations for large panels.
        s.IterationsPerDispatch = std::min(t.MaxIterations + 1, std::max(4u, 4u * 1024u * 1024u / (p.Size * p.Size)));
        s.Segments = p.Nonlinear ? t.MaxIterations / s.IterationsPerDispatch + 1 : 1;
        solves = std::max(solves, s.Iterations * s.Segments);
        s.Derivatives = p.Derivatives;
        s.Mask = p.Mask;
        s.Inverse = p.Inverse;
        if (p.Nonlinear) {
            std::vector<bool> coupled(p.Size);
            std::vector<double> columnSums(p.Size);
            double bound = 0;
            for (uint32_t row = 0; row < p.Size; ++row) {
                double rowSum = 0;
                for (uint32_t col = 0; col < p.Size; ++col) {
                    const double value = t.Values[p.Inverse + row * p.Size + col];
                    rowSum += std::abs(value);
                    columnSums[col] += std::abs(value);
                    if (row != col && value != 0) coupled[row] = coupled[col] = true;
                }
                bound = std::max(bound, rowSum);
            }
            bound = std::max(bound, *std::max_element(columnSums.begin(), columnSums.end()));
            // The larger absolute row/column sum bounds the spectral norm; double it for rounding margin.
            s.PreconditionBound = std::isfinite(float(2 * bound)) ? Pack(2 * bound) : Real{};
            // Store the coupled block first, followed by isolated diagonal entries, without dropping nonzero coefficients.
            std::vector<uint32_t> order;
            for (uint32_t row = 0; row < p.Size; ++row)
                if (coupled[row]) order.push_back(row);
            s.PreconditionSize = uint32_t(order.size());
            for (uint32_t row = 0; row < p.Size; ++row)
                if (!coupled[row]) order.push_back(row);
            s.PreconditionIndices = uint32_t(ix.size());
            ix.insert(ix.end(), order.begin(), order.end());
            auto *inverse = static_cast<Real *>(values.RawData()) + p.Inverse; // NOLINT(misc-const-correctness): pointee is written through incremented pointer
            for (uint32_t row = 0; row < s.PreconditionSize; ++row)
                for (uint32_t col = 0; col < s.PreconditionSize; ++col)
                    *inverse++ = Pack(t.Values[p.Inverse + order[row] * p.Size + order[col]]);
            for (uint32_t row = s.PreconditionSize; row < p.Size; ++row)
                *inverse++ = Pack(t.Values[p.Inverse + order[row] * p.Size + order[row]]);
        }
        s.WScale = Pack(p.WScale);
        s.FScale = Pack(p.FScale);
        s.InvH2 = Pack(p.InvH2);
        s.Tol2 = Pack(t.Tolerance * t.Tolerance);
        s.B = p.B;
        s.C = p.C;
        s.K = p.K;
        s.IJ = p.IJ;
        s.I = p.I;
        s.J = p.J;
        s.Channels = channels;
        s.Channel = airReceivers + uint32_t(index);
        s.Pickup = t.Receivers[s.Channel].Index;
        s.ForceIndices = uint32_t(ix.size());
        s.ForceOffset = uint32_t(forceValues.size());
        std::vector<Strike> strikes;
        for (const auto &strike : t.Strikes)
            if (strike.Plate == index) {
                strikes.push_back(strike);
                ix.push_back(strike.Index);
            }
        s.ForceCount = uint32_t(strikes.size());
        if (forceValues.size() + size_t(t.Steps) * strikes.size() >= UINT32_MAX) throw std::runtime_error("Strike data exceeds GPU indexing range");
        for (uint32_t step = 0; step < t.Steps; ++step)
            for (const auto &strike : strikes) {
                const double force = step >= strike.Start && step - strike.Start < strike.Length ?
                    p.Alpha * (.5 * strike.Amplitude * (1 - std::cos(2 * std::numbers::pi * (step - strike.Start) / strike.Length))) :
                    0;
                forceValues.push_back(force * (p.Adjoint ? t.Values[p.MassInverse + strike.Index] : 1));
            }
        gpu.Interface.ResizeZeroed(5 * size_t(slice) * sizeof(Real));
        Upload(gpu.Mask, p.AirMask);
        trace.BufferBytes += gpu.Interface.Capacity() + gpu.Mask.Capacity();
    }
    state.ResizeZeroed(size_t(stateSize) * sizeof(Real));
    status.ResizeZeroed(panels.size() * sizeof(uint32_t));
    panelParams.Resize(panels.size() * sizeof(PanelParams));
    for (size_t i = 0; i < panels.size(); ++i) panelParams.Upload(&panels[i].Params, sizeof(PanelParams), i * sizeof(PanelParams));
    Upload(forces, forceValues);
    indices.Resize(ix.size() * sizeof(uint32_t));
    indices.Upload(ix.data(), indices.Capacity());
    for (const GpuBuffer *b : {&air[0], &air[1], &air[2], &indices, &values, &coefficients, &airIndex, &receivers, &output, &forces, &state, &status, &panelParams}) trace.BufferBytes += b->Capacity();
    AirParams a{.N = cells, .Ny = t.Ny, .Slice = slice, .Channels = channels, .Receivers = airReceivers, .Reflecting = uint32_t(t.Reflecting), .Gamma2 = Pack(t.Gamma * t.Gamma), .Lambda2 = Pack(t.Lambda * t.Lambda), .QRate = Pack(t.Q * t.SampleRate), .Rate = Pack(t.PressureScale)};
    auto const dispatch = [&](const char *name, uint32_t count, std::initializer_list<GpuSlice> buffers, const auto &params) {
        context.Dispatch(kernels.at(name).get(), {(count + 255) / 256, 1, 1}, {256, 1, 1}, buffers, &params, sizeof(params));
    };
    auto const plateDispatch = [&](PanelDispatch params) {
        context.Dispatch(kernels.at("CoupledPanel").get(), {uint32_t(panels.size()), 1, 1}, {256, 1, 1}, {&indices, &values, &state, &forces, &output, &status, &panelParams}, &params, sizeof(params));
    };
    auto const checkStatus = [&] {
        const auto *errors = status.As<uint32_t>();
        for (size_t i = 0; i < panels.size(); ++i)
            if (errors[i]) throw std::runtime_error("Coupled Metal PCG failed to converge; increase solver_iterations or reduce excitation");
    };
    struct Drain {
        ~Drain() { MetalContext::Get().Drain(); }
    } drain;
    const auto start = std::chrono::steady_clock::now();
    int current = 0, previous = 1, older = 2;
    for (uint32_t step = 0; step < t.Steps; ++step) {
        a.Step = step;
        plateDispatch({0, step, 0});
        for (uint32_t solve = 0; solve < solves; ++solve) plateDispatch({1, step, solve});
        // Separated panel stencils allow all reverse updates before a single air-grid step.
        for (size_t i = 0; i < panels.size(); ++i) {
            auto const &p = panels[i];
            a.Lower = t.Plates[i].LowerSlice;
            dispatch("CoupledReverse", slice, {&indices, &values, GpuSlice(&state, size_t(p.Params.StateOffset) * sizeof(Real)), &p.Interface, &p.Mask, &air[previous]}, InterfaceParams{p.Params, a});
        }
        dispatch("CoupledAir", cells, {&air[current], &air[previous], &air[older], &coefficients, &airIndex}, a);
        for (size_t i = 0; i < panels.size(); ++i) {
            auto const &p = panels[i];
            a.Lower = t.Plates[i].LowerSlice;
            dispatch("CoupledForward", slice, {&air[current], &p.Interface}, a);
            dispatch("CoupledInterpolate", p.Params.N, {&indices, &values, GpuSlice(&state, size_t(p.Params.StateOffset) * sizeof(Real)), GpuSlice(&p.Interface, 4 * size_t(slice) * sizeof(Real))}, p.Params);
        }
        dispatch("CoupledSample", airReceivers, {&air[current], &air[older], &receivers, &output}, a);
        plateDispatch({2, step, 0});
        if (t.Reflecting) dispatch("CoupledBoundary", cells, {&air[current]}, a);
        std::swap(older, previous);
        std::swap(previous, current);
        // Keep eight steps per command buffer and at most four buffers in flight.
        if ((step + 1) % 8 == 0) context.Flush();
        if ((step + 1) % 32 == 0) {
            context.Drain();
            checkStatus();
        }
    }
    context.Drain();
    trace.Seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    checkStatus();
    Download(trace.Samples, output, 0, size_t(t.Steps) * channels);
    Download(trace.Air, air[previous], 0, cells);
    for (auto const &p : panels) Download(trace.Displacements, state, p.Params.StateOffset + p.Params.N, p.Params.N);
    return trace;
}
} // namespace coupled
