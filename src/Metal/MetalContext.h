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
//     while the CPU prepares the next. Sync() commits and waits — for everything except
//     deferred work the caller deliberately released ahead of it; Drain() waits for that too.

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
    // As above, against an already-resolved pipeline (hot loops that cache theirs, and
    // kernels from the radiation library). `threadgroup_bytes`, when nonzero, sizes
    // threadgroup buffer 0 for kernels that take it dynamically, since a static array would
    // have to be declared at its worst case and cost occupancy for the unused remainder.
    void Dispatch(MTL::ComputePipelineState *, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const void *params = nullptr, size_t params_size = 0, size_t threadgroup_bytes = 0);

    void Flush(); // No-op when nothing is pending
    // Waits out everything committed, except deferred work whose gate was released before
    // this call (see SignalDeferred) — that work is running now and is waited on by the
    // next Sync. No-op when idle.
    void Sync();
    void Drain(); // Sync, including early-released deferred work

    // Deferred command buffer: encode and commit GPU work whose host-written inputs are
    // not ready yet, gated on a shared event the host signals once they are. Between
    // BeginDeferred() and EndDeferred(), ActiveEncoder()/Dispatch() target the deferred
    // encoder. CommitDeferred() submits the buffer to the in-order queue (scheduled,
    // stalled on the event) and SignalDeferred() releases it. No buffer the deferred
    // work references may be resized between commit and signal.
    // Releasing the gate *before* the sync (legal whenever the host has nothing left to
    // write into the deferred buffer's inputs) is what keeps the GPU busy: the release
    // lands a batch ahead of the GPU reaching that buffer, so the host round trip and the
    // signal-to-start latency both fall off the critical path. On Trumpet, whose batches
    // are mostly fresh-cell-free, that takes gpu/idle_fdtd from 1.92 s to 0.005 s.
    void BeginDeferred();
    void EndDeferred();
    void CommitDeferred(); // No-op when nothing is held
    void SignalDeferred(); // No-op when nothing is gated

    // GPU execution time of the longest command buffer drained since the last call
    // (a running max, so intermediate syncs during batch prep don't hide it).
    double TakeBatchGpuSeconds() { return std::exchange(LastBatchSeconds, 0.); }

    // The pipeline state for a kernel, compiled on first use. `apply_faces` selects the
    // ApplyFaces function-constant specialization (see Kernels.metal).
    MTL::ComputePipelineState *Pipeline(const char *name, bool apply_faces = false);

    // The pipeline for a kernel of the radiation solver's library (src/Radiation), which
    // is compiled on first use so WaveBlender runs never pay for it.
    MTL::ComputePipelineState *RadiationPipeline(const char *name, int floor_probe = 0);

    // The pipeline for a kernel of the room solver's library (src/Room), compiled on first
    // use like the radiation one. These kernels specialize on nothing.
    MTL::ComputePipelineState *RoomPipeline(const char *name);

    MTL::ComputePipelineState *ImmersedPipeline(const char *name);

    // The active serial compute encoder, creating a command buffer/encoder if needed.
    // For hot encode loops that bypass Dispatch() and bind buffers persistently.
    MTL::ComputeCommandEncoder *ActiveEncoder();

    MTL::Device *Device{nullptr};
    MTL::CommandQueue *Queue{nullptr};
    MTL::Library *Library{nullptr};
    MTL::Library *RadiationLib{nullptr}; // lazily compiled, see RadiationPipeline
    MTL::Library *RoomLib{nullptr}; // lazily compiled, see RoomPipeline
    MTL::Library *ImmersedLib{nullptr};

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
    struct Submission {
        MTL::CommandBuffer *Buffer;
        bool Deferred; // an FDTD-batch buffer, for the gpu/exec_* and gpu/idle_* splits
    };
    std::vector<Submission> Committed;
    // Trailing entries of Committed that the next Sync deliberately runs through, rather
    // than waits out: a deferred buffer released before the sync. Any later commit clears
    // it — waiting on a newer buffer implies the older one finished anyway.
    size_t UnwaitedTail{0};
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
