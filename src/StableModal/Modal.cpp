#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "Metal/FloatPair.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Params {
    uint32_t modes, hidden, frames, offset, steps;
    FloatPair dt, lambda, epsilon, scale;
};

FloatPair Pack(double x) {
    float const hi = float(x);
    if (!std::isfinite(x) || !std::isfinite(hi)) throw std::runtime_error("Nonfinite Metal input");
    return {hi, float(x - double(hi))};
}

struct Runtime {
    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL::CommandQueue> Queue;
    NS::SharedPtr<MTL::ComputePipelineState> Pipeline;
};
} // namespace

extern "C" void *modal_create(const char *source, char *error, size_t capacity) {
    auto const pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    try {
        auto runtime = std::make_unique<Runtime>();
        runtime->Device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        if (!runtime->Device) throw std::runtime_error("Metal device unavailable");
        auto const options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
        options->setMathMode(MTL::MathModeSafe);
        NS::Error *err{};
        auto const library = NS::TransferPtr(runtime->Device->newLibrary(NS::String::string(source, NS::UTF8StringEncoding), options.get(), &err));
        if (!library) throw std::runtime_error(err ? err->localizedDescription()->utf8String() : "Modal shader compilation failed");
        auto const function = NS::TransferPtr(library->newFunction(NS::String::string("modal", NS::UTF8StringEncoding)));
        if (!function) throw std::runtime_error("Missing modal Metal kernel");
        runtime->Pipeline = NS::TransferPtr(runtime->Device->newComputePipelineState(function.get(), &err));
        if (!runtime->Pipeline) throw std::runtime_error(err ? err->localizedDescription()->utf8String() : "Modal pipeline creation failed");
        if (runtime->Pipeline->maxTotalThreadsPerThreadgroup() < 256 || runtime->Pipeline->threadExecutionWidth() != 32)
            throw std::runtime_error("256-thread groups with 32-lane SIMD required");
        runtime->Queue = NS::TransferPtr(runtime->Device->newCommandQueue());
        if (!runtime->Queue) throw std::runtime_error("Metal command queue creation failed");
        return runtime.release();
    } catch (const std::exception &e) {
        snprintf(error, capacity, "%s", e.what());
        return nullptr;
    }
}

extern "C" void modal_destroy(void *handle) { delete static_cast<Runtime *>(handle); }

extern "C" int modal_render(void *handle, uint32_t modes, uint32_t hidden, uint32_t count, uint32_t frames, uint32_t chunk, double fs, double lambda, const double *weights, const double *physical, const double *excitation, const double *initial, double *output, double *seconds, char *error, size_t capacity) {
    auto const pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    try {
        if (!handle || modes != 75 || hidden != 76 || !count || !frames || !chunk || fs <= 0)
            throw std::runtime_error("Invalid modal dimensions or settings");
        auto &r = *static_cast<Runtime *>(handle);
        auto const buffer = [&](const double *data, size_t size) {
            auto result = NS::TransferPtr(r.Device->newBuffer(size * sizeof(FloatPair), MTL::ResourceStorageModeShared));
            if (!result) throw std::runtime_error("Metal buffer allocation failed");
            if (data) {
                auto *values = static_cast<FloatPair *>(result->contents());
                for (size_t i = 0; i < size; ++i) values[i] = Pack(data[i]);
            }
            return result;
        };
        size_t const dim = 2 * modes + 1;
        if (size_t(frames) > SIZE_MAX / count / dim / sizeof(FloatPair)) throw std::runtime_error("Output size overflow");
        std::vector<NS::SharedPtr<MTL::Buffer>> buffers = {buffer(weights, size_t(modes) * hidden), buffer(physical, size_t(count) * (3 * modes + 1)), buffer(excitation, size_t(count) * frames), buffer(nullptr, size_t(count) * frames * dim)};
        auto *values = static_cast<FloatPair *>(buffers[3]->contents());
        for (size_t b = 0; b < count; ++b)
            for (size_t i = 0; i < dim; ++i) values[b * frames * dim + i] = Pack(initial[b * dim + i]);
        Params p{modes, hidden, frames, 0, 0, Pack(1 / fs), Pack(lambda), Pack(1e-12), Pack(1. / hidden)};
        auto const start = std::chrono::steady_clock::now();
        double gpu = 0;
        for (uint32_t offset = 0; offset < frames - 1;) {
            auto *command = r.Queue->commandBuffer();
            if (!command) throw std::runtime_error("Metal command buffer creation failed");
            auto *encoder = command->computeCommandEncoder();
            if (!encoder) throw std::runtime_error("Metal command encoder creation failed");
            encoder->setComputePipelineState(r.Pipeline.get());
            for (size_t i = 0; i < buffers.size(); ++i) encoder->setBuffer(buffers[i].get(), 0, i);
            // Bound each submission to keep long offline trajectories responsive.
            uint32_t const end = std::min<uint32_t>(frames - 1, offset + 4096);
            while (offset < end) {
                p.offset = offset;
                p.steps = std::min(chunk, end - offset);
                encoder->setBytes(&p, sizeof(p), 4);
                encoder->dispatchThreadgroups(MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
                offset += p.steps;
            }
            encoder->endEncoding();
            command->commit();
            command->waitUntilCompleted();
            if (command->status() == MTL::CommandBufferStatusError)
                throw std::runtime_error(command->error() ? command->error()->localizedDescription()->utf8String() : "Modal command buffer failed");
            gpu += command->GPUEndTime() - command->GPUStartTime();
        }
        seconds[0] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        seconds[1] = gpu;
        for (size_t i = 0; i < size_t(count) * frames * dim; ++i) {
            output[i] = double(values[i].Hi) + double(values[i].Lo);
            if (!std::isfinite(output[i])) throw std::runtime_error("Nonfinite Metal output");
        }
        return 0;
    } catch (const std::exception &e) {
        snprintf(error, capacity, "%s", e.what());
        return 1;
    }
}
