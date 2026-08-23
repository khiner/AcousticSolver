#pragma once

// GPU stepper for the SonicRadiation hybrid solver: the Metal port of the scheme documented
// in RadiationKernels.metal, consuming the CPU-built tables (RadiationTables) with their
// weights narrowed to block-scaled bytes and the history rings to float.
//
// There is no per-step host round trip. A batch's Neumann samples are uploaded up front
// and everything else — grid transport, PML, set-difference corrections, the element
// Dirichlet update, and listener output on both paths — stays on the GPU.

#include "RadiationTables.h"

#include "MetalContext.h"

// A table set in the shape the GPU reads it, which is not the shape the builder produces:
// four-lag-aligned block-scaled byte weights behind staged records, references renumbered
// onto the target-major slots the cell convolutions are stored in, and the lag-0
// double-layer terms lifted out of the table (the diagonal solve moves them to the left
// side). None of it touches Metal, so an epoch's set is packed by the worker that built it
// and binding costs only the uploads.
struct RadPackedTables {
    std::vector<RadStagedEntry> CellEntries, ElemEntries;
    std::vector<signed char> CellWeights, ElemWeights; // block-scaled bytes, as StagedWeights holds them
    std::vector<float> CellD0, CellD0Target; // source-major for the writing pass, target-major for the gathers
    std::vector<int> CellSrc, CellDst;
    std::vector<RadRef> CorrRefs, InterpRefs, ListenerRefs;
};

// Packs `tables` and the element-element table of `bem`. Pure CPU work, safe off the main
// thread.
RadPackedTables PackRadiation(const Tdbem &, const RadiationTables &);

namespace MTL {
class ComputePipelineState;
} // namespace MTL

struct RadiationGpu {
    int StepN{0}; // last completed step

    // Uploads the tables and the rest state. `bem` must already hold step-0 history
    // (phi(0) and g(0)) and the Neumann samples for step 1 in g(1).
    void Init(const RadiationGrid &, const Tdbem &, const RadiationTables &, const RadPackedTables &);

    // Enables listener output on both paths at the `n_points` points the tables' listener
    // stencils were built for: the grid path samples far-field state plus near potentials
    // trilinearly, the Eq. 5 path evaluates the retarded-potential representation directly
    // from `point_table` (dense over elements, from Tdbem::BuildPointTable at those points).
    void SetListeners(int n_points, const PairTable &point_table);

    // Runs `n_steps` steps. `g_rows` holds this batch's Neumann samples, row-major over
    // elements, for steps [StepN + 2, StepN + n_steps + 1] — one row past the batch,
    // because each step appends the *next* step's samples to the history rings.
    void RunSteps(int n_steps, const float *g_rows);

    // Applies a geometry epoch: re-bases the far-field state onto the new near-set
    // partition (see RebasePlan), then binds tables built for the new pose. The element
    // set and the history rings carry over unchanged.
    void ApplyEpoch(const RebasePlan &, const Tdbem &, const RadiationTables &, const RadPackedTables &);

    // Per-step listener samples of the batch just run, [step][listener]. Synchronizing.
    const float *GridSamples() const { return GridOut.As<float>(); }
    const float *PointSamples() const { return PointOut.As<float>(); }

    // Far-field grid state at the last completed step, for stability probes. Synchronizing.
    const float *FarField() const { return P[Cur].As<float>(); }
    // Surface Dirichlet history ring, element-major over HistLength() lag slots. Synchronizing.
    const float *Phi() const { return PhiRing.As<float>(); }
    int HistLength() const { return HistLen; }
    size_t NumCells() const { return Grid.NumCells(); }

    // Byte footprint of the device-side tables and state, for the build report.
    size_t Bytes() const { return TableBytes + StateBytes; }

    // Per-kernel GPU busy time, when ACOUSTIC_RAD_KERNEL_TIMES is set: each dispatch gets its
    // own command buffer and is waited out. Serializing costs wall time, so this is a
    // diagnostic run, not a mode to measure the solver in.
    struct KernelTimes {
        const char *Name;
        double Seconds;
        uint64_t Count;
    };
    const std::vector<KernelTimes> &Timings() const { return KernelTiming; }

private:
    RadiationGrid Grid;
    RadGridParams GP{};
    int NumElems{0}, HistLen{0};
    size_t TableBytes{0}, StateBytes{0};

    // Grid state: three pressure slots rotating per step (p(n-1), p(n), p(n+1)), the PML
    // shell's staggered velocities, and its split pressures.
    GpuBuffer P[3];
    int Cur{0}, Prev{1}, Next{2};
    GpuBuffer Vx, Vy, Vz, Px, Py, Pz;
    GpuBuffer PmlNv, PmlDv, PmlNp, PmlDp;

    // Weight tables and history (see RadPackedTables). CellD0 holds each cell-table entry's
    // lag-0 double-layer weight, which the repacking takes out of the table; CellDst maps an
    // entry to the target-major slot its convolution is stored in. CellD0Target and CellSrc
    // are the same lag-0 weights and their source elements in that target-major order, which
    // is what the listener path needs to complete a convolution at the step just solved.
    GpuBuffer CellEntries, CellWeights, CellD0, CellD0Target, CellSrc, CellDst, CellSrcBegin;
    GpuBuffer ElemEntries, ElemWeights, ElemSrcBegin, ElemBegin, ElemOrder;
    GpuBuffer PhiRing, GRing, GSrc;
    // Per-entry convolutions: float2 of (skip-current-phi at n + 1, completed at n) for the
    // cell table, plain floats at n + 1 for the element table. The cell results are stored
    // in target-major order, for the three passes that gather them; ConvCellX keeps the
    // skip-current-phi values in the writing pass's own source-major order, so completing
    // the previous step's convolution costs no second scatter.
    GpuBuffer ConvCell, ConvCellX, ConvElem;
    GpuBuffer CorrCell, CorrBegin, CorrRefs;
    GpuBuffer Interp, InterpRefs;
    GpuBuffer GridCorners, GridRefs;
    GpuBuffer EvalBegin, EvalOrder, EvalEntries, EvalWeights;
    GpuBuffer GridOut, PointOut, EvalPartials, Params;
    GpuBuffer RebaseCells, RebaseBegin, RebaseRefs, RebaseEntries, RebaseWeights;

    // Cached pipelines: the step loop encodes hundreds of thousands of dispatches.
    MTL::ComputePipelineState *PsoConvCells{}, *PsoConvElems{}, *PsoVelocity{}, *PsoPressure{}, *PsoCorrect{},
        *PsoDirichlet{}, *PsoSample{}, *PsoEval{}, *PsoEvalReduce{}, *PsoRebase{};

    Dim3 GridTiles{}, GridThreads{};
    size_t StageBytes{0}; // threadgroup staging for one source element's history windows
    bool TimeKernels{false};
    std::vector<KernelTimes> KernelTiming;
    // ACOUSTIC_RAD_FLOOR_PROBE (see Init): cut-down builds of the two convolution passes,
    // dispatched beside the real ones onto scratch.
    std::vector<int> FloorProbes;
    std::vector<MTL::ComputePipelineState *> ProbeConvCells, ProbeConvElems;
    GpuBuffer ProbeConvCell, ProbeConvCellX, ProbeConvElem;
    std::vector<std::string> ProbeNames; // KernelTimes keeps the pointer, so the strings live here
    // Dispatches, and when timing is on, isolates the dispatch in its own command buffer
    // and charges its GPU time to `name`. A negative slot dispatches untimed.
    void Timed(int slot, const char *name, MTL::ComputePipelineState *, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice>, const void *params = nullptr, size_t params_size = 0, size_t threadgroup_bytes = 0);

    // The two convolution passes. Their bindings have to agree between the real dispatch,
    // the seeding one BindTables runs and the floor probes' copies onto scratch, so all
    // three go through here.
    void ConvolveCells(int slot, const char *name, MTL::ComputePipelineState *, GpuBuffer &xs, GpuBuffer &conv, const RadStepParams &);
    void ConvolveElems(int slot, const char *name, MTL::ComputePipelineState *, GpuBuffer &conv, const RadStepParams &);

    // Uploads the geometry-dependent tables and reseeds the convolution pass on them.
    void BindTables(const Tdbem &, const RadiationTables &, const RadPackedTables &);

    // Uploads `bytes` from `data`, sizing the buffer with slack. Returns the byte size.
    size_t UploadSized(GpuBuffer &, const void *data, size_t bytes);
    // Uploads `values` narrowed to float and returns the byte size.
    size_t UploadFloats(GpuBuffer &, const std::vector<double> &);
    template<typename T> size_t UploadPod(GpuBuffer &, const std::vector<T> &);
};
