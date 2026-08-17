#pragma once

// Metal port of the WaveBlender GPU runtime (replaces the CUDA runtime + cuBLAS usage).
//
// The implicit command stream mirrors CUDA default-stream semantics: dispatches are
// asynchronous and execute in order (one serial encoder per command buffer, and command
// buffers on one queue complete in commit order).
//
// Synchronization discipline (this is what makes CPU/GPU pipelining safe):
//   - GpuBuffer::Data()/As<T>() synchronize the stream before returning a host pointer.
//     Use these for host reads of GPU results.
//   - Upload()/Zero() do NOT synchronize. The caller must guarantee no in-flight GPU work
//     reads the written range — either by writing after a sync point, or by writing to a
//     buffer (or buffer region) the in-flight batch does not reference.
//   - Flush() commits pending dispatches without waiting, so the GPU crunches one batch
//     while the CPU prepares the next. Sync() commits and waits.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MTL {
class Device;
class CommandQueue;
class Library;
class ComputePipelineState;
class Buffer;
class CommandBuffer;
class ComputeCommandEncoder;
class SharedEvent;
} // namespace MTL

struct Dim3 {
    uint32_t x{1}, y{1}, z{1};
};

struct GpuBuffer;

// A buffer binding with a byte offset, for dispatches that read a slice of a larger
// buffer (e.g. per-minibatch slots). Implicitly constructible from a GpuBuffer pointer.
struct GpuSlice {
    GpuSlice(const GpuBuffer *buffer) : Buffer(buffer) {}
    GpuSlice(const GpuBuffer *buffer, size_t offset_bytes) : Buffer(buffer), OffsetBytes(offset_bytes) {}

    const GpuBuffer *Buffer;
    size_t OffsetBytes{0};
};

struct MetalContext {
    static MetalContext &Get();

    // Encodes one compute dispatch. Threadgroup counts/sizes mirror CUDA <<<blocks, threads>>> exactly.
    // Buffers bind at indices 0..N-1 in order, `params` (optional) at index N via setBytes.
    // Skipped when any threadgroup count is 0, matching CUDA's no-op zero-block launches.
    void Dispatch(const char *kernel, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const void *params = nullptr, size_t params_size = 0);

    void Flush(); // No-op when nothing is pending
    void Sync(); // No-op when idle

    // Deferred command buffer: encode and commit GPU work whose host-written inputs are
    // not ready yet, gated on a shared event the host signals once they are. Between
    // BeginDeferred() and EndDeferred(), ActiveEncoder()/Dispatch() target the deferred
    // encoder. CommitDeferred() submits the buffer to the in-order queue (scheduled,
    // stalled on the event) and SignalDeferred() releases it. No buffer the deferred
    // work references may be resized between commit and signal.
    void BeginDeferred();
    void EndDeferred();
    void CommitDeferred(); // No-op when nothing is held
    void SignalDeferred(); // No-op when nothing is gated

    // GPU execution time of the longest command buffer drained since the last call
    // (a running max, so intermediate syncs during batch prep don't hide it).
    double TakeBatchGpuSeconds() { return std::exchange(LastBatchSeconds, 0.); }

    // The pipeline state for a kernel, compiled on first use. `fold_apply` selects the
    // FoldApply function-constant specialization (see Kernels.metal).
    MTL::ComputePipelineState *Pipeline(const char *name, bool fold_apply = false);

    // The active serial compute encoder, creating a command buffer/encoder if needed.
    // For hot encode loops that bypass Dispatch() and bind buffers persistently.
    MTL::ComputeCommandEncoder *ActiveEncoder();

    MTL::Device *Device{nullptr};
    MTL::CommandQueue *Queue{nullptr};
    MTL::Library *Library{nullptr};

private:
    MetalContext();

    MTL::CommandBuffer *CmdBuf{nullptr};
    MTL::ComputeCommandEncoder *Encoder{nullptr};
    MTL::CommandBuffer *DeferredCmdBuf{nullptr};
    MTL::ComputeCommandEncoder *DeferredEncoder{nullptr};
    MTL::CommandBuffer *GatedCmdBuf{nullptr}; // committed, stalled on GateEvent until SignalDeferred()
    MTL::SharedEvent *GateEvent{nullptr};
    uint64_t GateValue{0};
    // Committed, not yet waited on (in-order queue: waiting on the last waits on all).
    // The flag marks deferred (FDTD-batch) buffers, for the gpu/exec_* split instrument.
    std::vector<std::pair<MTL::CommandBuffer *, bool>> Committed;
    double LastBatchSeconds{0.};
    double PrevGpuEndTime{0.}; // GPU end timestamp of the last drained command buffer (for the gpu/idle instrument)
    std::unordered_map<std::string, MTL::ComputePipelineState *> Pipelines;
};

// Owning wrapper over a shared-storage MTL::Buffer. Grow-only, like the
// cudaFree + cudaMalloc reallocation pattern in the reference code.
struct GpuBuffer {
    GpuBuffer() = default;
    ~GpuBuffer() { Free(); }
    GpuBuffer(const GpuBuffer &) = delete;
    GpuBuffer &operator=(const GpuBuffer &) = delete;

    // Ensures capacity of at least `bytes`. New storage is uninitialized (like cudaMalloc).
    // Growing an existing allocation synchronizes first (in-flight work may reference it).
    void Resize(size_t bytes);
    // Resize(), then zero-fill the whole allocation.
    void ResizeZeroed(size_t bytes);
    void Free();

    // Synchronizes the stream, then returns the host-visible pointer. For host reads.
    void *Data() const;
    template<typename T> T *As() const { return static_cast<T *>(Data()); }

    // Host-visible pointer without synchronizing, for scatter writes (see the discipline
    // note above).
    void *RawData() const { return Contents(); }

    // Host-to-buffer copy without synchronizing (see the discipline note above).
    void Upload(const void *src, size_t bytes, size_t dst_offset_bytes = 0) const;
    // Zero-fill of the first `bytes` bytes without synchronizing.
    void Zero(size_t bytes) const;

    size_t Capacity() const { return CapacityBytes; }
    MTL::Buffer *Handle() const { return Buf; }

    // Exchanges the underlying allocations (handles only — no data copies). The caller
    // must guarantee no in-flight GPU work references either buffer.
    void Swap(GpuBuffer &o) {
        std::swap(Buf, o.Buf);
        std::swap(CapacityBytes, o.CapacityBytes);
    }

private:
    void *Contents() const; // host-visible pointer, no synchronization
    MTL::Buffer *Buf{nullptr};
    size_t CapacityBytes{0};
};

// Double-buffered GpuBuffer, for two hazard patterns:
//   - Per-batch CPU rewrites while the previous batch's command buffer may still read
//     the other slot: Flip() once per rewrite, then write into and bind Cur().
//   - Per-step GPU ping-pong (read Cur(), write Other(), then Flip()) where in-kernel
//     neighbor reads would race in-place writes.
struct GpuDouble {
    void Flip() { Index ^= 1; }
    GpuBuffer &Cur() { return Slots[Index]; }
    const GpuBuffer &Cur() const { return Slots[Index]; }
    GpuBuffer &Other() { return Slots[Index ^ 1]; }
    const GpuBuffer &Other() const { return Slots[Index ^ 1]; }

private:
    GpuBuffer Slots[2];
    int Index{0};
};
