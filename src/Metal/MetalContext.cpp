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
#include "Radiation/RadiationParams.h" // RadFloorProbeFcIndex

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>

namespace {
// GPU busy time of a completed command buffer. A buffer that never reached the GPU reports
// GPUStartTime() as 0 while GPUEndTime() stays an absolute timestamp, so the naive
// difference is time since boot rather than a duration — one such buffer put 39,413
// seconds into a kernel's running total and ACOUSTIC_RAD_KERNEL_TIMES printed it as
// 13,257 ms/step beside neighbours reading 0.01. Dropping the sample is right: it is one
// dispatch of thousands, and an instrument that reported zero throughout would be obvious.
double GpuSeconds(const MTL::CommandBuffer *cb) {
    const double start = cb->GPUStartTime(), end = cb->GPUEndTime();
    return start > 0. && end > start ? end - start : 0.;
}

std::string ReadFile(const std::string &path) {
    std::ifstream const in{path, std::ios::binary};
    if (!in.good()) throw std::runtime_error(std::format("Failed to read MSL source: {}", path));
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Compiles one runtime library from concatenated MSL sources.
MTL::Library *CompileLibrary(MTL::Device *device, const std::string &source) {
    auto *options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(false); // nvcc-comparable IEEE behavior for validation

    NS::Error *error{nullptr};
    auto *library = device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options, &error);
    options->release();
    if (!library) throw std::runtime_error(std::format("MSL compilation failed:\n{}", error ? error->localizedDescription()->utf8String() : "unknown error"));
    return library;
}

void EnsureLibrary(MTL::Library *&library, MTL::Device *device, const char *profile_name, const char *directory, const char *params, const char *kernels) {
    if (library) return;
    const profile::Scope scope{profile_name};
    library = CompileLibrary(device, ReadFile(std::string{directory} + "/" + params) + ReadFile(std::string{directory} + "/" + kernels));
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
    Library = CompileLibrary(Device, ReadFile(std::string{ACOUSTIC_MSL_DIR} + "/KernelParams.h") + ReadFile(std::string{ACOUSTIC_MSL_DIR} + "/Kernels.metal"));
}

namespace {
// Specializes `name` out of `library` on one function constant, or on none when `value` is
// null. The two specializing libraries differ only in the constant's index and type.
MTL::ComputePipelineState *MakePipeline(MTL::Device *device, MTL::Library *library, const char *name, const void *value = nullptr, MTL::DataType type = MTL::DataTypeNone, int fc_index = 0) {
    const profile::Scope scope{"startup/pipeline_create"};
    const auto *fn_name = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function *fn{nullptr};
    if (value) {
        auto *constants = MTL::FunctionConstantValues::alloc()->init();
        constants->setConstantValue(value, type, NS::UInteger(fc_index));
        NS::Error *fn_error{nullptr};
        fn = library->newFunction(fn_name, constants, &fn_error);
        constants->release();
    } else {
        fn = library->newFunction(fn_name);
    }
    if (!fn) throw std::runtime_error(std::format("Kernel not found: {}", name));

    NS::Error *error{nullptr};
    auto *pso = device->newComputePipelineState(fn, &error);
    fn->release();
    if (!pso) throw std::runtime_error(std::format("Pipeline creation failed for {}:\n{}", name, error ? error->localizedDescription()->utf8String() : "unknown error"));
    return pso;
}
} // namespace

MTL::ComputePipelineState *MetalContext::RadiationPipeline(const char *name, int floor_probe) {
    // `floor_probe` picks a cut-down build of the convolution passes; 0 is the shipping path
    // and the rest are the fixed-cost probe documented in RadiationParams.h. Keyed into the
    // cache, so one process can hold several levels and dispatch them side by side.
    const std::string key = std::string{"rad/"} + name + (floor_probe ? "#probe" + std::to_string(floor_probe) : "");
    if (const auto it = Pipelines.find(key); it != Pipelines.end()) return it->second;
    EnsureLibrary(RadiationLib, Device, "startup/msl_compile_radiation", ACOUSTIC_RADIATION_MSL_DIR, "RadiationParams.h", "RadiationKernels.metal");
    return Pipelines[key] = MakePipeline(Device, RadiationLib, name, &floor_probe, MTL::DataTypeInt, RadFloorProbeFcIndex);
}

MTL::ComputePipelineState *MetalContext::RoomPipeline(const char *name) {
    const std::string key = std::string{"room/"} + name;
    if (const auto it = Pipelines.find(key); it != Pipelines.end()) return it->second;
    EnsureLibrary(RoomLib, Device, "startup/msl_compile_room", ACOUSTIC_ROOM_MSL_DIR, "RoomParams.h", "RoomKernels.metal");
    return Pipelines[key] = MakePipeline(Device, RoomLib, name);
}

MTL::ComputePipelineState *MetalContext::ImmersedPipeline(const char *name) {
    const std::string key = std::string{"immersed/"} + name;
    if (const auto it = Pipelines.find(key); it != Pipelines.end()) return it->second;
    EnsureLibrary(ImmersedLib, Device, "startup/msl_compile_immersed", ACOUSTIC_IMMERSED_MSL_DIR, "ImmersedParams.h", "ImmersedKernels.metal");
    return Pipelines[key] = MakePipeline(Device, ImmersedLib, name);
}

MTL::ComputePipelineState *MetalContext::Pipeline(const char *name, bool apply_faces) {
    const std::string key = apply_faces ? std::string{name} + "#faces" : std::string{name};
    if (const auto it = Pipelines.find(key); it != Pipelines.end()) return it->second;
    return Pipelines[key] = MakePipeline(Device, Library, name, &apply_faces, MTL::DataTypeBool, ApplyFacesFcIndex);
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
    Dispatch(Pipeline(kernel), blocks, threads, buffers, params, params_size);
}

void MetalContext::Dispatch(MTL::ComputePipelineState *pso, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const void *params, size_t params_size, size_t threadgroup_bytes) {
    if (blocks.x == 0 || blocks.y == 0 || blocks.z == 0) return;

    auto *encoder = ActiveEncoder();
    encoder->setComputePipelineState(pso);
    if (threadgroup_bytes > 0) encoder->setThreadgroupMemoryLength(threadgroup_bytes, 0);

    uint32_t index = 0;
    for (const auto &slice : buffers) encoder->setBuffer(slice.Buffer->Handle(), slice.OffsetBytes, index++);
    if (params && params_size > 0) encoder->setBytes(params, params_size, index);

    encoder->dispatchThreadgroups(MTL::Size{blocks.x, blocks.y, blocks.z}, MTL::Size{threads.x, threads.y, threads.z});
}

void MetalContext::BeginDeferred() {
    if (!GateEvent) {
        GateEvent = Device->newSharedEvent();
        // Unchecked, the nil reaches encodeWait as an opaque ObjC exception.
        if (!GateEvent) throw std::runtime_error("MTLSharedEvent creation failed");
    }
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
    // A gated buffer must never be the wait target of a Sync that runs before its release —
    // that would deadlock — so it joins the list only now, marked as the un-waited tail.
    // Releasing after the sync (the fresh-cell path) leaves the mark stale, which the next
    // commit clears.
    Committed.push_back({GatedCmdBuf, true});
    UnwaitedTail = 1;
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
        Committed.push_back({CmdBuf, false}); // keeps the +1 ref from ActiveEncoder()
        UnwaitedTail = 0; // in-order queue: waiting on this implies every earlier buffer finished
        CmdBuf = nullptr;
    }
}

void MetalContext::Sync() {
    Flush();
    // Everything before the un-waited tail. The in-order queue makes the last of these
    // completing imply all earlier ones did, so the tail keeps running past this call.
    const size_t n_wait = Committed.size() > UnwaitedTail ? Committed.size() - UnwaitedTail : 0;
    if (n_wait == 0) return;
    const auto waited = std::span{Committed}.first(n_wait);
    {
        const profile::Scope scope{"gpu/wait"};
        // Block only up to the second-to-last waited buffer, then spin out the last —
        // waitUntilCompleted's ~100 us kernel wake then overlaps the last (typically
        // small) buffer's execution.
        auto *last = waited.back().Buffer;
        if (n_wait > 1) waited[n_wait - 2].Buffer->waitUntilCompleted();
        for (int i = 0; i < 80000 && last->status() < MTL::CommandBufferStatusCompleted; ++i) {}
        if (last->status() < MTL::CommandBufferStatusCompleted) last->waitUntilCompleted();
    }
    for (const auto &[cb, deferred] : waited) LastBatchSeconds = std::max(LastBatchSeconds, GpuSeconds(cb));
    if (profile::Enabled()) {
        auto &exec = profile::Entries()["gpu/exec"];
        auto &exec_fdtd = profile::Entries()["gpu/exec_fdtd"];
        auto &exec_prologue = profile::Entries()["gpu/exec_prologue"];
        for (const auto &[cb, deferred] : waited) {
            const double seconds = GpuSeconds(cb);
            exec.Seconds += seconds;
            auto &split = deferred ? exec_fdtd : exec_prologue;
            split.Seconds += seconds;
            split.Count += 1;
        }
        exec.Count += n_wait;

        // GPU idle between consecutive command buffers (commit order) — the batch loop's
        // pipeline bubbles, measurable independent of machine load. `idle_prologue` is pure
        // command-buffer handoff (committed long before), `idle_fdtd` the host sync window.
        auto &idle = profile::Entries()["gpu/idle"];
        auto &idle_fdtd = profile::Entries()["gpu/idle_fdtd"];
        auto &idle_prologue = profile::Entries()["gpu/idle_prologue"];
        for (const auto &[cb, deferred] : waited) {
            auto &split = deferred ? idle_fdtd : idle_prologue;
            // Same guard: a buffer with no valid start time neither closes an idle gap
            // nor opens one, and letting its end time through would silently suppress
            // every gap after it.
            if (GpuSeconds(cb) > 0.) {
                if (PrevGpuEndTime > 0. && cb->GPUStartTime() > PrevGpuEndTime) {
                    idle.Seconds += cb->GPUStartTime() - PrevGpuEndTime;
                    split.Seconds += cb->GPUStartTime() - PrevGpuEndTime;
                }
                PrevGpuEndTime = cb->GPUEndTime();
            }
            idle.Count += 1;
            split.Count += 1;
        }
    }
    for (const auto &[cb, deferred] : waited) cb->release();
    Committed.erase(Committed.begin(), Committed.begin() + n_wait);
}

void MetalContext::Drain() {
    UnwaitedTail = 0;
    Sync();
}

void GpuBuffer::Resize(size_t bytes) {
    if (bytes <= CapacityBytes) return;

    auto &ctx = MetalContext::Get();
    ctx.Drain(); // in-flight work may reference the old allocation
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
    MetalContext::Get().Drain();
    Buf->release();
    Buf = nullptr;
    CapacityBytes = 0;
}

void *GpuBuffer::Contents() const {
    return Buf ? Buf->contents() : nullptr;
}

void *GpuBuffer::Data() const {
    MetalContext::Get().Drain();
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
