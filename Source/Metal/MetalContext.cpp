// Metal port of the WaveBlender GPU runtime.
//
// This is the single translation unit that instantiates the metal-cpp implementation.
// Kernels are compiled at runtime from ACOUSTIC_MSL_DIR/KernelParams.h + Kernels.metal,
// with fast-math disabled so results stay comparable against the CUDA reference.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstring>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "MetalContext.h"

namespace {
std::string ReadFile(const std::string &path) {
    std::ifstream const in{path, std::ios::binary};
    if (!in.good()) throw std::runtime_error(std::format("Failed to read MSL source: {}", path));
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
} // namespace

MetalContext &MetalContext::Get() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() : Device(MTL::CreateSystemDefaultDevice()) {
    if (!Device) throw std::runtime_error("No Metal device available");
    Queue = Device->newCommandQueue();

    // Runtime-compile the kernel library. KernelParams.h is shared with the host,
    // so it is prepended in place of a #include (runtime compilation has no header search path).
    const auto source = ReadFile(std::string{ACOUSTIC_MSL_DIR} + "/KernelParams.h") + ReadFile(std::string{ACOUSTIC_MSL_DIR} + "/Kernels.metal");

    auto *options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(false); // nvcc-comparable IEEE behavior for validation

    NS::Error *error{nullptr};
    Library = Device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options, &error);
    options->release();
    if (!Library) throw std::runtime_error(std::format("MSL compilation failed:\n{}", error ? error->localizedDescription()->utf8String() : "unknown error"));
}

MTL::ComputePipelineState *MetalContext::Pipeline(const char *name) {
    if (auto it = Pipelines.find(name); it != Pipelines.end()) return it->second;

    auto *fn = Library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) throw std::runtime_error(std::format("Kernel not found: {}", name));

    NS::Error *error{nullptr};
    auto *pso = Device->newComputePipelineState(fn, &error);
    fn->release();
    if (!pso) throw std::runtime_error(std::format("Pipeline creation failed for {}:\n{}", name, error ? error->localizedDescription()->utf8String() : "unknown error"));
    Pipelines[name] = pso;
    return pso;
}

void MetalContext::EnsureEncoder() {
    if (!CmdBuf) {
        CmdBuf = Queue->commandBuffer();
        CmdBuf->retain();
    }
    if (!Encoder) {
        Encoder = CmdBuf->computeCommandEncoder(); // serial dispatch type: in-order execution
        Encoder->retain();
    }
}

void MetalContext::Dispatch(const char *kernel, Dim3 blocks, Dim3 threads, std::initializer_list<const GpuBuffer *> buffers, const void *params, size_t params_size) {
    if (blocks.x == 0 || blocks.y == 0 || blocks.z == 0) return;

    EnsureEncoder();
    Encoder->setComputePipelineState(Pipeline(kernel));

    uint32_t index = 0;
    for (const auto *buffer : buffers) Encoder->setBuffer(buffer->Handle(), 0, index++);
    if (params && params_size > 0) Encoder->setBytes(params, params_size, index);

    Encoder->dispatchThreadgroups(MTL::Size{blocks.x, blocks.y, blocks.z}, MTL::Size{threads.x, threads.y, threads.z});
}

void MetalContext::Sync() {
    if (Encoder) {
        Encoder->endEncoding();
        Encoder->release();
        Encoder = nullptr;
    }
    if (CmdBuf) {
        CmdBuf->commit();
        CmdBuf->waitUntilCompleted();
        CmdBuf->release();
        CmdBuf = nullptr;
    }
}

void GpuBuffer::Resize(size_t bytes) {
    if (bytes <= CapacityBytes) return;

    auto &ctx = MetalContext::Get();
    ctx.Sync(); // in-flight work may reference the old allocation
    if (Buf) Buf->release();
    Buf = ctx.Device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!Buf) throw std::runtime_error(std::format("MTL::Buffer allocation failed ({} bytes)", bytes));
    CapacityBytes = bytes;
}

void GpuBuffer::ResizeZeroed(size_t bytes) {
    Resize(bytes);
    Zero(CapacityBytes);
}

void GpuBuffer::Free() {
    if (!Buf) return;
    MetalContext::Get().Sync();
    Buf->release();
    Buf = nullptr;
    CapacityBytes = 0;
}

void *GpuBuffer::Data() const {
    MetalContext::Get().Sync();
    return Buf ? Buf->contents() : nullptr;
}

void GpuBuffer::Upload(const void *src, size_t bytes, size_t dst_offset_bytes) const {
    if (bytes == 0) return;
    std::memcpy(static_cast<char *>(Data()) + dst_offset_bytes, src, bytes);
}

void GpuBuffer::Zero(size_t bytes) const {
    if (bytes == 0) return;
    std::memset(Data(), 0, bytes);
}
