#pragma once

// Metal port of the WaveBlender GPU runtime (replaces the CUDA runtime + cuBLAS usage).
//
// The implicit command stream mirrors CUDA default-stream semantics: dispatches are
// asynchronous and execute in order, and any host access to a GpuBuffer synchronizes
// the stream first (the equivalent of a synchronous cudaMemcpy).

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>

namespace MTL {
class Device;
class CommandQueue;
class Library;
class ComputePipelineState;
class Buffer;
class CommandBuffer;
class ComputeCommandEncoder;
} // namespace MTL

struct Dim3 {
    uint32_t x{1}, y{1}, z{1};
};

class GpuBuffer;

class MetalContext {
public:
    static MetalContext &Get();

    // Encodes one compute dispatch. Threadgroup counts/sizes mirror CUDA <<<blocks, threads>>> exactly.
    // Buffers bind at indices 0..N-1 in order, `params` (optional) at index N via setBytes.
    // Skipped when any threadgroup count is 0, matching CUDA's no-op zero-block launches.
    void Dispatch(const char *kernel, Dim3 blocks, Dim3 threads, std::initializer_list<const GpuBuffer *> buffers, const void *params = nullptr, size_t params_size = 0);

    // Flushes pending dispatches and blocks until the GPU is idle. No-op when idle.
    void Sync();

    MTL::Device *Device{nullptr};
    MTL::CommandQueue *Queue{nullptr};
    MTL::Library *Library{nullptr};

private:
    MetalContext();

    MTL::ComputePipelineState *Pipeline(const char *name);
    void EnsureEncoder();

    MTL::CommandBuffer *CmdBuf{nullptr};
    MTL::ComputeCommandEncoder *Encoder{nullptr};
    std::unordered_map<std::string, MTL::ComputePipelineState *> Pipelines;
};

// Owning wrapper over a shared-storage MTL::Buffer. Grow-only, like the
// cudaFree + cudaMalloc reallocation pattern in the reference code.
class GpuBuffer {
public:
    GpuBuffer() = default;
    ~GpuBuffer() { Free(); }
    GpuBuffer(const GpuBuffer &) = delete;
    GpuBuffer &operator=(const GpuBuffer &) = delete;

    // Ensures capacity of at least `bytes`. New storage is uninitialized (like cudaMalloc).
    void Resize(size_t bytes);
    // Resize(), then zero-fill the whole allocation.
    void ResizeZeroed(size_t bytes);
    void Free();

    // Synchronizes the stream, then returns the host-visible pointer.
    void *Data() const;
    template<typename T> T *As() const { return static_cast<T *>(Data()); }

    // Synchronized host-to-buffer copy (the cudaMemcpyHostToDevice equivalent).
    void Upload(const void *src, size_t bytes, size_t dst_offset_bytes = 0) const;
    // Synchronized zero-fill of the first `bytes` bytes (the cudaMemset equivalent).
    void Zero(size_t bytes) const;

    size_t Capacity() const { return CapacityBytes; }
    MTL::Buffer *Handle() const { return Buf; }

private:
    MTL::Buffer *Buf{nullptr};
    size_t CapacityBytes{0};
};
