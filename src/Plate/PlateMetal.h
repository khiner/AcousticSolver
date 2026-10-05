#pragma once

#include "MetalContext.h"
#include "Profile.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace plate::metal {

inline std::string Read(const char *path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("Cannot read plate shader: ") + path);
    return {std::istreambuf_iterator<char>(input), {}};
}

inline NS::SharedPtr<MTL::Library> Compile(const std::string &source) {
    auto const options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    options->setFastMathEnabled(false);
    NS::Error *error{};
    auto library = NS::TransferPtr(MetalContext::Get().Device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options.get(), &error));
    if (!library) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "Plate shader compilation failed");
    return library;
}

inline NS::SharedPtr<MTL::ComputePipelineState> Pipeline(MTL::Library *library, const char *name, const MTL::FunctionConstantValues *constants = nullptr) {
    NS::Error *error{};
    auto *label = NS::String::string(name, NS::UTF8StringEncoding);
    auto const function = NS::TransferPtr(constants ? library->newFunction(label, constants, &error) : library->newFunction(label));
    if (!function) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "Missing plate kernel");
    auto pipeline = NS::TransferPtr(MetalContext::Get().Device->newComputePipelineState(function.get(), &error));
    if (!pipeline) throw std::runtime_error(error ? error->localizedDescription()->utf8String() : "Plate pipeline creation failed");
    if (pipeline->threadExecutionWidth() != 32 || pipeline->maxTotalThreadsPerThreadgroup() < 256)
        throw std::runtime_error("Plate kernels require 32-lane SIMD and 256-thread groups");
    return pipeline;
}

template<typename Values> void Upload(GpuBuffer &buffer, const Values &values) {
    using T = typename Values::value_type;
    buffer.Resize(std::max(sizeof(T), values.size() * sizeof(T)));
    if (!values.empty()) buffer.Upload(values.data(), values.size() * sizeof(T));
}

// Stage profiling deliberately drains each dispatch; use normal runs for throughput.
template<typename Params> void Dispatch(MTL::ComputePipelineState *pipeline, const char *name, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const Params &params, size_t sharedBytes = 0) {
    auto &ctx = MetalContext::Get();
    ctx.Dispatch(pipeline, blocks, threads, buffers, &params, sizeof(params), sharedBytes);
    static const bool stages = std::getenv("PLATE_PROFILE_STAGES") != nullptr;
    if (stages) {
        ctx.Drain();
        auto &entry = profile::Entries()[name];
        entry.Seconds += ctx.TakeBatchGpuSeconds();
        ++entry.Count;
    }
}
} // namespace plate::metal
