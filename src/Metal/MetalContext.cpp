// Metal port of the WaveBlender GPU runtime.
//
// This is the single translation unit that instantiates the metal-cpp implementation.
// Kernels are compiled at runtime from ACOUSTIC_MSL_DIR/KernelParams.h + Kernels.metal,
// with fast-math disabled so results stay comparable against the CUDA reference.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "KernelParams.h"
#include "MetalContext.h"
#include "Profile.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

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

    // Runtime-compile the kernel library (macOS caches this, so warm starts take ~1 ms).
    // KernelParams.h is shared with the host, so it is prepended in place of a #include
    // (runtime compilation has no header search path).
    const profile::Scope scope{"startup/msl_compile"};
    const auto source = ReadFile(std::string{ACOUSTIC_MSL_DIR} + "/KernelParams.h") + ReadFile(std::string{ACOUSTIC_MSL_DIR} + "/Kernels.metal");

    auto *options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(false); // nvcc-comparable IEEE behavior for validation

    NS::Error *error{nullptr};
    Library = Device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options, &error);
    options->release();
    if (!Library) throw std::runtime_error(std::format("MSL compilation failed:\n{}", error ? error->localizedDescription()->utf8String() : "unknown error"));
}

MTL::ComputePipelineState *MetalContext::Pipeline(const char *name, bool apply_faces) {
    const std::string key = apply_faces ? std::string{name} + "#faces" : std::string{name};
    if (auto it = Pipelines.find(key); it != Pipelines.end()) return it->second;

    const profile::Scope scope{"startup/pipeline_create"};
    auto *constants = MTL::FunctionConstantValues::alloc()->init();
    constants->setConstantValue(&apply_faces, MTL::DataTypeBool, NS::UInteger{ApplyFacesFcIndex});
    NS::Error *fn_error{nullptr};
    auto *fn = Library->newFunction(NS::String::string(name, NS::UTF8StringEncoding), constants, &fn_error);
    constants->release();
    if (!fn) throw std::runtime_error(std::format("Kernel not found: {}", name));

    NS::Error *error{nullptr};
    auto *pso = Device->newComputePipelineState(fn, &error);
    fn->release();
    if (!pso) throw std::runtime_error(std::format("Pipeline creation failed for {}:\n{}", name, error ? error->localizedDescription()->utf8String() : "unknown error"));
    Pipelines[key] = pso;
    return pso;
}

MTL::ComputeCommandEncoder *MetalContext::ActiveEncoder() {
    if (DeferredEncoder) return DeferredEncoder;
    if (!CmdBuf) {
        CmdBuf = Queue->commandBuffer();
        CmdBuf->retain();
    }
    if (!Encoder) {
        Encoder = CmdBuf->computeCommandEncoder(); // serial dispatch type: in-order execution
        Encoder->retain();
    }
    return Encoder;
}

void MetalContext::Dispatch(const char *kernel, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const void *params, size_t params_size) {
    if (blocks.x == 0 || blocks.y == 0 || blocks.z == 0) return;

    auto *encoder = ActiveEncoder();
    encoder->setComputePipelineState(Pipeline(kernel));

    uint32_t index = 0;
    for (const auto &slice : buffers) encoder->setBuffer(slice.Buffer->Handle(), slice.OffsetBytes, index++);
    if (params && params_size > 0) encoder->setBytes(params, params_size, index);

    encoder->dispatchThreadgroups(MTL::Size{blocks.x, blocks.y, blocks.z}, MTL::Size{threads.x, threads.y, threads.z});
}

void MetalContext::BeginDeferred() {
    if (!GateEvent) GateEvent = Device->newSharedEvent();
    DeferredCmdBuf = Queue->commandBuffer();
    DeferredCmdBuf->retain();
    DeferredCmdBuf->encodeWait(GateEvent, GateValue + 1); // released by SignalDeferred()
    DeferredEncoder = DeferredCmdBuf->computeCommandEncoder(); // serial dispatch type: in-order execution
    DeferredEncoder->retain();
}

void MetalContext::EndDeferred() {
    DeferredEncoder->endEncoding();
    DeferredEncoder->release();
    DeferredEncoder = nullptr;
}

void MetalContext::CommitDeferred() {
    if (!DeferredCmdBuf) return;
    DeferredCmdBuf->commit(); // scheduled now, stalled on the event
    GatedCmdBuf = DeferredCmdBuf; // keeps the +1 ref from BeginDeferred()
    DeferredCmdBuf = nullptr;
}

void MetalContext::SignalDeferred() {
    if (!GatedCmdBuf) return;
    GateEvent->setSignaledValue(++GateValue);
    // A gated buffer must never be Sync()'s wait target, so it joins the drain list only
    // once signaled
    Committed.emplace_back(GatedCmdBuf, true);
    GatedCmdBuf = nullptr;
}

void MetalContext::Flush() {
    if (Encoder) {
        Encoder->endEncoding();
        Encoder->release();
        Encoder = nullptr;
    }
    if (CmdBuf) {
        CmdBuf->commit();
        Committed.emplace_back(CmdBuf, false); // keeps the +1 ref from ActiveEncoder()
        CmdBuf = nullptr;
    }
}

void MetalContext::Sync() {
    Flush();
    if (Committed.empty()) return;
    {
        const profile::Scope scope{"gpu/wait"};
        // In-order queue: the last buffer completing implies all earlier ones completed.
        // Block only up to the second-to-last buffer, then spin out the last —
        // waitUntilCompleted's ~100 us kernel wake then overlaps the last (typically
        // small) buffer's execution.
        auto *last = Committed.back().first;
        if (Committed.size() > 1) Committed[Committed.size() - 2].first->waitUntilCompleted();
        for (int i = 0; i < 80000 && last->status() < MTL::CommandBufferStatusCompleted; ++i) {}
        if (last->status() < MTL::CommandBufferStatusCompleted) last->waitUntilCompleted();
    }
    for (const auto &[cb, deferred] : Committed) LastBatchSeconds = std::max(LastBatchSeconds, cb->GPUEndTime() - cb->GPUStartTime());
    if (profile::Enabled()) {
        auto &exec = profile::Entries()["gpu/exec"];
        auto &exec_fdtd = profile::Entries()["gpu/exec_fdtd"];
        auto &exec_prologue = profile::Entries()["gpu/exec_prologue"];
        for (const auto &[cb, deferred] : Committed) {
            const double seconds = cb->GPUEndTime() - cb->GPUStartTime();
            exec.Seconds += seconds;
            auto &split = deferred ? exec_fdtd : exec_prologue;
            split.Seconds += seconds;
            split.Count += 1;
        }
        exec.Count += Committed.size();

        // GPU idle between consecutive command buffers (commit order) — the batch loop's
        // pipeline bubbles, measurable independent of machine load. `idle_prologue` is pure
        // command-buffer handoff (committed long before), `idle_fdtd` the host sync window.
        auto &idle = profile::Entries()["gpu/idle"];
        auto &idle_fdtd = profile::Entries()["gpu/idle_fdtd"];
        auto &idle_prologue = profile::Entries()["gpu/idle_prologue"];
        for (const auto &[cb, deferred] : Committed) {
            auto &split = deferred ? idle_fdtd : idle_prologue;
            if (PrevGpuEndTime > 0. && cb->GPUStartTime() > PrevGpuEndTime) {
                idle.Seconds += cb->GPUStartTime() - PrevGpuEndTime;
                split.Seconds += cb->GPUStartTime() - PrevGpuEndTime;
            }
            PrevGpuEndTime = cb->GPUEndTime();
            idle.Count += 1;
            split.Count += 1;
        }
    }
    for (const auto &[cb, deferred] : Committed) cb->release();
    Committed.clear();
}

void GpuBuffer::Resize(size_t bytes) {
    if (bytes <= CapacityBytes) return;

    auto &ctx = MetalContext::Get();
    ctx.Sync(); // in-flight work may reference the old allocation
    if (Buf) {
        Buf->release();
        bytes = std::max(bytes, 2 * CapacityBytes); // geometric growth: each grow syncs the stream, so make them rare
    }
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

void *GpuBuffer::Contents() const {
    return Buf ? Buf->contents() : nullptr;
}

void *GpuBuffer::Data() const {
    MetalContext::Get().Sync();
    return Contents();
}

void GpuBuffer::Upload(const void *src, size_t bytes, size_t dst_offset_bytes) const {
    if (bytes == 0) return;
    std::memcpy(static_cast<char *>(Contents()) + dst_offset_bytes, src, bytes);
}

void GpuBuffer::Zero(size_t bytes) const {
    if (bytes == 0) return;
    std::memset(Contents(), 0, bytes);
}
