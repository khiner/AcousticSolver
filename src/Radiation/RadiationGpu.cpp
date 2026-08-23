#include "RadiationGpu.h"

#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <stdexcept>

namespace {
constexpr uint32_t Group = 256; // 1D threadgroup size of the flat per-entry/per-cell kernels
constexpr uint32_t Reduce = 128; // threadgroup size of the reducing kernels (matches their shared array)
constexpr uint32_t Stage = 256; // threadgroup size of the convolution passes, one group per source element
constexpr uint32_t SimdWidth = 32;

Dim3 Blocks1d(size_t n) { return {uint32_t((n + Group - 1) / Group), 1, 1}; }
// Blocks for a kernel that gives each work item one SIMD group rather than one thread.
Dim3 BlocksSimd(size_t n) { return {uint32_t((n + Group / SimdWidth - 1) / (Group / SimdWidth)), 1, 1}; }
} // namespace

// Table buffers are rebound once per geometry epoch, so this sizes with slack and zeroes
// only what the upload does not cover: a grow drains the whole stream, and consecutive
// epochs' tables differ by parts in ten thousand.
size_t RadiationGpu::UploadSized(GpuBuffer &buffer, const void *data, size_t bytes) {
    const size_t alloc = std::max<size_t>(4, bytes); // an empty table still needs a binding
    buffer.Resize(alloc + alloc / 8);
    // Spread the copy and the tail zeroing over the machine: a rebind is hundreds of
    // megabytes on the one thread feeding the GPU, bounded by a single core's memcpy rather
    // than by the device. Safe unsynchronized because the GPU is idle here — the batch before
    // an epoch swap is read back through GridSamples(), which syncs, before the swap.
    constexpr size_t Grain = 1 << 20; // below a megabyte the fan-out costs more than the copy
    auto *dst = static_cast<char *>(buffer.RawData());
    const auto *src = static_cast<const char *>(data);
    ParallelChunks(bytes, Grain, [&](size_t begin, size_t end) { std::memcpy(dst + begin, src + begin, end - begin); });
    const size_t tail = buffer.Capacity() - bytes;
    ParallelChunks(tail, Grain, [&](size_t begin, size_t end) { std::memset(dst + bytes + begin, 0, end - begin); });
    return alloc;
}

size_t RadiationGpu::UploadFloats(GpuBuffer &buffer, const std::vector<double> &values) {
    const std::vector<float> narrowed(values.begin(), values.end());
    return UploadSized(buffer, narrowed.data(), narrowed.size() * sizeof(float));
}

template<typename T> size_t RadiationGpu::UploadPod(GpuBuffer &buffer, const std::vector<T> &values) {
    return UploadSized(buffer, values.data(), values.size() * sizeof(T));
}

namespace {
// One float table repacked into the GPU's layout: four-lag-aligned windows of block-scaled
// bytes behind staged records. `d0` (cell table only) receives each entry's lag-0
// double-layer weight, which leaves the table itself.
void PackStaged(std::vector<RadStagedEntry> &staged, std::vector<signed char> &padded, const PairTable &table, std::vector<float> *d0) {
    const size_t n = table.Entries.size();
    staged.resize(n);
    size_t total = 0;
    for (size_t k = 0; k < n; ++k) {
        const RadEntry &e = table.Entries[k];
        int d_lo, d_len, v_lo, v_len;
        WidenQuad(e.DBegin, e.DLen, d_lo, d_len);
        WidenQuad(e.VBegin, e.VLen, v_lo, v_len);
        staged[k] = {int(total), (unsigned char)(d_lo / 4), (unsigned char)(d_len / 4), (unsigned char)(v_lo / 4), (unsigned char)(v_len / 4), 0, 0};
        total += 4 * (size_t(staged[k].DLenQ) + staged[k].VLenQ);
    }
    padded.assign(total, 0);
    if (d0) d0->assign(n, 0.f);
    ParallelFor(n, 4096, [&](size_t k) {
        const RadEntry &e = table.Entries[k];
        RadStagedEntry &p = staged[k];
        const float *w = table.Weights.data() + e.Off;
        QuantizeWindows(w, e.DBegin, e.DLen, e.VLen, padded.data() + p.Off + (e.DBegin - 4 * p.DBeginQ), padded.data() + p.Off + 4 * p.DLenQ + (e.VBegin - 4 * p.VBeginQ), d0 ? &(*d0)[k] : nullptr, p.DExp, p.VExp);
    });
}
} // namespace

RadPackedTables PackRadiation(const Tdbem &bem, const RadiationTables &tables) {
    RadPackedTables packed;
    // A table built staged is already in this layout and stays where it is: it is the
    // largest thing in the process, and the epoch after this one reads its weights back.
    if (!tables.CellElem.IsStaged()) PackStaged(packed.CellEntries, packed.CellWeights, tables.CellElem, &packed.CellD0);
    if (!bem.ElemElem.IsStaged()) PackStaged(packed.ElemEntries, packed.ElemWeights, bem.ElemElem, nullptr);
    const std::vector<float> &cell_d0 = tables.CellElem.IsStaged() ? tables.CellElem.Staged.D0 : packed.CellD0;

    // The cell convolutions are written by a pass that streams the table by source element
    // and read by three that gather it by target cell. Stored by source, a target's
    // references land one source-block apart and every one costs a cache line; stored by
    // target they are a contiguous ascending run of the cell's own row. So the results go in
    // target-major order and the writing pass scatters instead, which is one write per entry
    // against the several references each one is gathered by.
    const size_t n_cell = tables.CellElem.Entries.size();
    packed.CellDst.resize(n_cell);
    for (size_t p = 0; p < tables.CellElem.Order.size(); ++p) packed.CellDst[tables.CellElem.Order[p]] = int(p);
    const auto retargeted = [&](const std::vector<RadRef> &refs) {
        std::vector<RadRef> out(refs.size());
        ParallelFor(refs.size(), 4096, [&](size_t r) {
            out[r] = {packed.CellDst[refs[r].Packed & ~RadRefNegative] | (refs[r].Packed & RadRefNegative)};
        });
        return out;
    };
    packed.CorrRefs = retargeted(tables.CorrRefs);
    packed.ListenerRefs = retargeted(tables.ListenerRefs);
    packed.InterpRefs = retargeted(tables.InterpRefs);

    // Each cell entry's source element, which the listener path needs to complete a
    // convolution at the step just solved, and its lag-0 weight. Both are gathered by
    // reference, so both go in target-major order — the writing pass keeps its own
    // source-major copy of the lag-0 weights.
    packed.CellD0Target.resize(n_cell);
    packed.CellSrc.resize(n_cell);
    ParallelFor(n_cell, 4096, [&](size_t k) {
        packed.CellD0Target[packed.CellDst[k]] = cell_d0[k];
        packed.CellSrc[packed.CellDst[k]] = tables.CellElem.Entries[k].Src;
    });
    return packed;
}

void RadiationGpu::BindTables(const Tdbem &bem, const RadiationTables &tables, const RadPackedTables &packed) {
    GP.NumCorrCells = int(tables.CorrCell.size());
    const size_t n_cell_entries = tables.CellElem.Entries.size(), n_elem_entries = bem.ElemElem.Entries.size();

    TableBytes = 0;
    TableBytes += UploadFloats(PmlNv, Grid.PmlNv) + UploadFloats(PmlDv, Grid.PmlDv);
    TableBytes += UploadFloats(PmlNp, Grid.PmlNp) + UploadFloats(PmlDp, Grid.PmlDp);
    // Whichever form each table came in: built staged, its weights are already this and stay
    // where they are. Referenced one array at a time, because a reference to a whole
    // StagedWeights would have to bind a temporary in the other branch and copy gigabytes.
    const bool cell_staged = tables.CellElem.IsStaged(), elem_staged = bem.ElemElem.IsStaged();
    const auto &cell_entries = cell_staged ? tables.CellElem.Staged.Entries : packed.CellEntries;
    const auto &cell_weights = cell_staged ? tables.CellElem.Staged.Weights : packed.CellWeights;
    const auto &cell_d0 = cell_staged ? tables.CellElem.Staged.D0 : packed.CellD0;
    const auto &elem_entries = elem_staged ? bem.ElemElem.Staged.Entries : packed.ElemEntries;
    const auto &elem_weights = elem_staged ? bem.ElemElem.Staged.Weights : packed.ElemWeights;
    TableBytes += UploadPod(CellEntries, cell_entries) + UploadPod(CellWeights, cell_weights);
    TableBytes += UploadPod(ElemEntries, elem_entries) + UploadPod(ElemWeights, elem_weights);
    TableBytes += UploadPod(ElemBegin, bem.ElemElem.Begin) + UploadPod(ElemOrder, bem.ElemElem.Order);
    TableBytes += UploadPod(CellSrcBegin, tables.CellElem.SrcBegin) + UploadPod(ElemSrcBegin, bem.ElemElem.SrcBegin);
    TableBytes += UploadPod(CellDst, packed.CellDst);
    TableBytes += UploadPod(CorrCell, tables.CorrCell) + UploadPod(CorrBegin, tables.CorrBegin) + UploadPod(CorrRefs, packed.CorrRefs);
    TableBytes += UploadPod(GridCorners, tables.ListenerCorners) + UploadPod(GridRefs, packed.ListenerRefs);
    TableBytes += UploadPod(Interp, tables.Interp) + UploadPod(InterpRefs, packed.InterpRefs);
    TableBytes += UploadPod(CellD0, cell_d0) + UploadPod(CellD0Target, packed.CellD0Target) + UploadPod(CellSrc, packed.CellSrc);

    ConvCellX.ResizeZeroed(std::max<size_t>(4, n_cell_entries * sizeof(float)));
    ConvCell.ResizeZeroed(std::max<size_t>(8, n_cell_entries * 2 * sizeof(float)));
    ConvElem.ResizeZeroed(std::max<size_t>(4, n_elem_entries * sizeof(float)));
    if (!FloorProbes.empty()) { // scratch, so a probe dispatch never touches the real state
        ProbeConvCellX.ResizeZeroed(ConvCellX.Capacity());
        ProbeConvCell.ResizeZeroed(ConvCell.Capacity());
        ProbeConvElem.ResizeZeroed(ConvElem.Capacity());
    }
    Params.Upload(&GP, sizeof GP);

    // Seed the pass at the current step, so the next step's corrections find a completed
    // convolution there: the pass at N1 = StepN leaves the skip-current-phi value in x,
    // which the next step completes into y.
    const RadStepParams seed{StepN - 1, StepN, 0, 0, 0};
    ConvolveCells(-1, nullptr, PsoConvCells, ConvCellX, ConvCell, seed);
    MetalContext::Get().Sync();
}

void RadiationGpu::Init(const RadiationGrid &grid, const Tdbem &bem, const RadiationTables &tables, const RadPackedTables &packed) {
    Grid = grid;
    NumElems = bem.NumElems();
    HistLen = bem.HistLen;
    if (HistLen > RadMaxHistLen) throw std::runtime_error("History ring exceeds the staging limit");
    // The convolution passes read the staged history four lags at a time from an aligned
    // boundary, and the second window starts one ring plus RadStageTail into the buffer.
    if (HistLen % 4 != 0) throw std::runtime_error("History ring must be a multiple of four lags");

    auto &ctx = MetalContext::Get();
    // ACOUSTIC_RAD_FLOOR_PROBE is a comma-separated list of probe levels, e.g. "1,2,5". Each
    // gets its own build of the two convolution passes, dispatched every step beside the real
    // ones and timed on its own slot, which is what makes the legs comparable — one process,
    // one step, one thermal state. They write to scratch of the same shape, so the run stays
    // correct and can be scored like any other. See RadiationParams.h for what each level
    // removes.
    FloorProbes.clear();
    if (const char *env = std::getenv("ACOUSTIC_RAD_FLOOR_PROBE")) {
        for (const char *at = env; *at;) {
            char *end = nullptr;
            const long level = std::strtol(at, &end, 10);
            if (end == at) break;
            if (level > 0) FloorProbes.push_back(int(level));
            at = *end ? end + 1 : end;
        }
    }
    ProbeNames.clear();
    for (const int level : FloorProbes) {
        ProbeNames.push_back(std::format("conv_cells#{}", level));
        ProbeNames.push_back(std::format("conv_elems#{}", level));
    }
    PsoConvCells = ctx.RadiationPipeline("RadConvolveCells");
    PsoConvElems = ctx.RadiationPipeline("RadConvolveElems");
    ProbeConvCells.clear();
    ProbeConvElems.clear();
    for (const int level : FloorProbes) {
        ProbeConvCells.push_back(ctx.RadiationPipeline("RadConvolveCells", level));
        ProbeConvElems.push_back(ctx.RadiationPipeline("RadConvolveElems", level));
    }
    PsoVelocity = ctx.RadiationPipeline("RadVelocity");
    PsoPressure = ctx.RadiationPipeline("RadPressure");
    PsoCorrect = ctx.RadiationPipeline("RadCorrect");
    PsoDirichlet = ctx.RadiationPipeline("RadDirichlet");
    PsoSample = ctx.RadiationPipeline("RadSample");
    PsoEval = ctx.RadiationPipeline("RadEvalPoint");
    PsoEvalReduce = ctx.RadiationPipeline("RadEvalReduce");
    PsoRebase = ctx.RadiationPipeline("RadRebase");

    GP.Nx = Grid.Nx;
    GP.Ny = Grid.Ny;
    GP.Nz = Grid.Nz;
    GP.PmlWidth = Grid.PmlWidth;
    GP.NumElems = NumElems;
    GP.HistLen = HistLen;
    GP.NumListeners = 0;
    GP.S2 = float(std::pow(Grid.C * Grid.Tau / Grid.H, 2.));
    GP.InvDamp = float(1. / (1. + Grid.Damping));
    GP.InvRhoDtH = float(Grid.Tau / (Grid.Rho * Grid.H));
    GP.RhoCcDtH = float(Grid.Rho * Grid.C * Grid.C * Grid.Tau / Grid.H);
    GP.FilterFeedback = Grid.FilterFeedback ? 1 : 0;

    const size_t cells = Grid.NumCells(), cell_bytes = cells * sizeof(float);
    StateBytes = 0;
    for (auto *b : {&P[0], &P[1], &P[2], &Vx, &Vy, &Vz, &Px, &Py, &Pz}) {
        b->ResizeZeroed(cell_bytes);
        StateBytes += cell_bytes;
    }
    Cur = 0;
    Prev = 1;
    Next = 2;
    StepN = 0;
    TimeKernels = std::getenv("ACOUSTIC_RAD_KERNEL_TIMES") != nullptr;
    // Threadgroup staging for one source element's two history windows, plus their tails.
    StageBytes = size_t(HistLen + RadStageTail) * 2 * sizeof(float);

    // Rest state: phi(0) and the Neumann samples for steps 0 and 1, from the reference's
    // own initial solve.
    StateBytes += UploadFloats(PhiRing, bem.Phi()) + UploadFloats(GRing, bem.G());
    Params.ResizeZeroed(sizeof(RadGridParams));

    // Small tiles: the shell branch of the pressure kernel is the divergent one, and a
    // 32-wide row keeps its x-runs contiguous.
    GridThreads = {32, 4, 4};
    GridTiles = {(uint32_t(Grid.Nx) + 31) / 32, (uint32_t(Grid.Ny) + 3) / 4, (uint32_t(Grid.Nz) + 3) / 4};

    BindTables(bem, tables, packed);
    StateBytes += ConvCell.Capacity() + ConvCellX.Capacity() + ConvElem.Capacity();
}

void RadiationGpu::ApplyEpoch(const RebasePlan &plan, const Tdbem &bem, const RadiationTables &tables, const RadPackedTables &packed) {
    auto &ctx = MetalContext::Get();
    if (bem.HistLen != HistLen || bem.NumElems() != NumElems)
        throw std::runtime_error("A geometry epoch may not change the element set or the history ring");
    if (!plan.Empty()) {
        // Still on the old tables: the crossing potentials must be evaluated with the pose
        // the history was accumulated under.
        UploadPod(RebaseCells, plan.Cells);
        UploadPod(RebaseBegin, plan.Begin);
        UploadPod(RebaseRefs, plan.Refs);
        UploadPod(RebaseEntries, plan.Table.Entries);
        UploadPod(RebaseWeights, plan.Table.Weights);
        const RadStepParams step{StepN, StepN + 1, 0, 0, int(plan.Cells.size())};
        ctx.Dispatch(PsoRebase, Blocks1d(plan.Cells.size()), {Group, 1, 1}, {&P[Cur], &P[Prev], &RebaseCells, &RebaseBegin, &RebaseRefs, &RebaseEntries, &RebaseWeights, &PhiRing, &GRing, &Params}, &step, sizeof step);
        ctx.Sync();
    }
    BindTables(bem, tables, packed);
}

void RadiationGpu::SetListeners(int n_points, const PairTable &point_table) {
    // The rings are sized once and carried across geometry epochs, so a point table rebuilt
    // for a new pose has to still fit in them: a listener that ends up farther from the body
    // reaches longer lags, and one lag past the ring reads a wrapped slot rather than failing.
    if (point_table.MaxLagEnd() >= HistLen) throw std::runtime_error("Listener table reaches past the history ring");
    TableBytes += UploadPod(EvalBegin, point_table.Begin) + UploadPod(EvalOrder, point_table.Order);
    TableBytes += UploadPod(EvalEntries, point_table.Entries);
    TableBytes += UploadPod(EvalWeights, point_table.Weights);
    GP.NumListeners = n_points;
    // Split each point's entries into chunks of one per thread, so the pass fills the GPU
    // rather than the single threadgroup a per-point dispatch would give it.
    const int per_point = n_points > 0 ? int(point_table.Entries.size()) / n_points : 0;
    GP.EvalGroups = std::max(1, (per_point + int(Reduce) - 1) / int(Reduce));
    EvalPartials.ResizeZeroed(std::max<size_t>(4, size_t(n_points) * GP.EvalGroups * sizeof(float)));
    Params.Upload(&GP, sizeof GP);
}

void RadiationGpu::Timed(int slot, const char *name, MTL::ComputePipelineState *pso, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const void *params, size_t params_size, size_t threadgroup_bytes) {
    auto &ctx = MetalContext::Get();
    ctx.Dispatch(pso, blocks, threads, buffers, params, params_size, threadgroup_bytes);
    if (!TimeKernels || slot < 0) return;
    ctx.Sync();
    if (int(KernelTiming.size()) <= slot) KernelTiming.resize(slot + 1);
    KernelTiming[slot].Name = name;
    KernelTiming[slot].Seconds += ctx.TakeBatchGpuSeconds();
    KernelTiming[slot].Count += 1;
}

void RadiationGpu::ConvolveCells(int slot, const char *name, MTL::ComputePipelineState *pso, GpuBuffer &xs, GpuBuffer &conv, const RadStepParams &step) {
    Timed(slot, name, pso, {uint32_t(NumElems), 1, 1}, {Stage, 1, 1}, {&CellEntries, &CellWeights, &PhiRing, &GRing, &CellSrcBegin, &CellD0, &CellDst, &xs, &conv, &Params}, &step, sizeof step, StageBytes);
}

void RadiationGpu::ConvolveElems(int slot, const char *name, MTL::ComputePipelineState *pso, GpuBuffer &conv, const RadStepParams &step) {
    Timed(slot, name, pso, {uint32_t(NumElems), 1, 1}, {Stage, 1, 1}, {&ElemEntries, &ElemWeights, &PhiRing, &GRing, &ElemSrcBegin, &conv, &Params}, &step, sizeof step, StageBytes);
}

void RadiationGpu::RunSteps(int n_steps, const float *g_rows) {
    auto &ctx = MetalContext::Get();
    const size_t out_bytes = std::max<size_t>(4, size_t(n_steps) * std::max(GP.NumListeners, 1) * sizeof(float));
    GridOut.ResizeZeroed(out_bytes);
    PointOut.ResizeZeroed(out_bytes);
    GSrc.Resize(std::max<size_t>(4, size_t(n_steps) * NumElems * sizeof(float)));
    GSrc.Upload(g_rows, size_t(n_steps) * NumElems * sizeof(float));
    for (int local = 0; local < n_steps; ++local) {
        const RadStepParams step{StepN, StepN + 1, local, local, 0};
        // Cut-down builds first, onto scratch, so the real passes below overwrite nothing of
        // theirs and the step is unaffected by whether the probe is on.
        for (size_t i = 0; i < FloorProbes.size(); ++i) {
            ConvolveCells(int(9 + 2 * i), ProbeNames[2 * i].c_str(), ProbeConvCells[i], ProbeConvCellX, ProbeConvCell, step);
            ConvolveElems(int(10 + 2 * i), ProbeNames[2 * i + 1].c_str(), ProbeConvElems[i], ProbeConvElem, step);
        }
        // One pass over each weight table, feeding every gather below.
        ConvolveCells(0, "conv_cells", PsoConvCells, ConvCellX, ConvCell, step);
        ConvolveElems(1, "conv_elems", PsoConvElems, ConvElem, step);
        Timed(2, "velocity", PsoVelocity, GridTiles, GridThreads, {&P[Cur], &Vx, &Vy, &Vz, &PmlNv, &PmlDv, &Params});
        Timed(3, "pressure", PsoPressure, GridTiles, GridThreads, {&P[Cur], &P[Prev], &P[Next], &Vx, &Vy, &Vz, &Px, &Py, &Pz, &PmlNp, &PmlDp, &Params});
        Timed(4, "correct", PsoCorrect, BlocksSimd(GP.NumCorrCells), {Group, 1, 1}, {&P[Next], &CorrCell, &CorrBegin, &CorrRefs, &ConvCell, &Params}, &step, sizeof step);
        std::swap(Prev, Cur); // p(n) -> previous
        std::swap(Cur, Next); // p(n + 1) -> current (the old previous becomes scratch)
        Timed(5, "dirichlet", PsoDirichlet, {uint32_t(NumElems), 1, 1}, {Reduce, 1, 1}, {&P[Cur], &P[Prev], &Interp, &InterpRefs, &ConvCell, &ElemBegin, &ElemOrder, &ConvElem, &PhiRing, &GRing, &GSrc, &Params}, &step, sizeof step);
        if (GP.NumListeners > 0) {
            Timed(6, "sample", PsoSample, {uint32_t(GP.NumListeners), 1, 1}, {Reduce, 1, 1}, {&P[Cur], &GridCorners, &GridRefs, &ConvCell, &CellD0Target, &CellSrc, &PhiRing, &GridOut, &Params}, &step, sizeof step);
            Timed(7, "eval_point", PsoEval, {uint32_t(GP.EvalGroups), uint32_t(GP.NumListeners), 1}, {Reduce, 1, 1}, {&EvalBegin, &EvalOrder, &EvalEntries, &EvalWeights, &PhiRing, &GRing, &EvalPartials, &Params}, &step, sizeof step);
            Timed(8, "eval_reduce", PsoEvalReduce, Blocks1d(GP.NumListeners), {Group, 1, 1}, {&EvalPartials, &PointOut, &Params}, &step, sizeof step);
        }
        StepN = step.N1;
    }
    ctx.Flush();
}
