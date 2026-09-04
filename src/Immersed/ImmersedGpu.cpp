#include "ImmersedGpu.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace immersed {
namespace {
constexpr uint32_t Group = 256;

Dim3 Blocks1d(size_t count) { return {uint32_t((count + Group - 1) / Group), 1, 1}; }

template<typename T> void Upload(GpuBuffer &buffer, const std::vector<T> &values) {
    static_assert(std::is_trivially_copyable_v<T>);
    const size_t bytes = values.size() * sizeof(T);
    buffer.ResizeZeroed(std::max<size_t>(4, bytes));
    buffer.Upload(values.data(), bytes);
}

void Upload(GpuBuffer &buffer, const std::vector<double> &values) { Upload(buffer, std::vector<float>{values.begin(), values.end()}); }

void CheckInteriorSupport(const Grid &grid, const DeltaStencil &stencil) {
    for (const int index : stencil.Index) {
        const int x = index % grid.Nx;
        const int y = (index / grid.Nx) % grid.Ny;
        const int z = index / (grid.Nx * grid.Ny);
        if (x < grid.PmlWidth || x >= grid.Nx - grid.PmlWidth || y < grid.PmlWidth ||
            y >= grid.Ny - grid.PmlWidth || z < grid.PmlWidth || z >= grid.Nz - grid.PmlWidth)
            throw std::runtime_error("A source or receiver delta overlaps the PML");
    }
}

DeltaStencil InteriorDelta(const Grid &grid, const Point &position, int order, int staggered_axis = -1) {
    auto stencil = LagrangeDelta(position, grid.Origin, grid.H, {grid.Nx, grid.Ny, grid.Nz}, order, staggered_axis);
    CheckInteriorSupport(grid, stencil);
    return stencil;
}

ImmersedFilter PackFilter(const BoundaryImmittance &immittance, double time_step) {
    if (immittance.Infinite) return {};
    const TrapezoidImmittance filter(immittance.Value, time_step);
    const auto &b = filter.Feedforward();
    const auto &a = filter.Feedback();
    if (b.size() > 3 || a.size() > 3)
        throw std::runtime_error("GPU immersed immittances are limited to biquads");
    return {float(b[0]), b.size() > 1 ? float(b[1]) : 0.f, b.size() > 2 ? float(b[2]) : 0.f, a.size() > 1 ? float(a[1]) : 0.f, a.size() > 2 ? float(a[2]) : 0.f};
}

bool HasState(const ImmersedFilter &filter) { return filter.B0 != 0.f || filter.B1 != 0.f || filter.B2 != 0.f || filter.A1 != 0.f || filter.A2 != 0.f; }

using CellMap = std::map<int, std::vector<ImmersedWeightedIndex>>;

void PackCells(const CellMap &map, std::vector<ImmersedIoCell> &cells, std::vector<ImmersedWeightedIndex> &refs) {
    for (const auto &[cell, row] : map) {
        const int begin = int(refs.size());
        refs.insert(refs.end(), row.begin(), row.end());
        cells.push_back({cell, begin, int(refs.size())});
    }
}

double SparseDot(const DeltaStencil &a, const DeltaStencil &b) {
    size_t i = 0, j = 0;
    double result = 0.;
    while (i < a.Index.size() && j < b.Index.size()) {
        if (a.Index[i] < b.Index[j]) ++i;
        else if (b.Index[j] < a.Index[i]) ++j;
        else {
            result += a.Weight[i] * b.Weight[j];
            ++i;
            ++j;
        }
    }
    return result;
}

double MatrixInfinityNorm(const std::vector<double> &matrix, int n) {
    double norm = 0.;
    for (int row = 0; row < n; ++row) {
        double sum = 0.;
        for (int column = 0; column < n; ++column) sum += std::abs(matrix[size_t(column) * n + row]);
        norm = std::max(norm, sum);
    }
    return norm;
}

double InvertSpd(std::vector<double> &matrix, int n) {
    if (n == 0) return 1.;
    const double original_norm = MatrixInfinityNorm(matrix, n);
    const __LAPACK_int size = n;
    __LAPACK_int info = 0;
    dpotrf_("L", &size, matrix.data(), &size, &info);
    if (info != 0) throw std::runtime_error("Immersed patch Gram matrix is not positive definite");
    dpotri_("L", &size, matrix.data(), &size, &info);
    if (info != 0) throw std::runtime_error("Immersed patch Gram matrix inversion failed");
    for (int column = 0; column < n; ++column)
        for (int row = 0; row < column; ++row)
            matrix[size_t(column) * n + row] = matrix[size_t(row) * n + column];
    return original_norm * MatrixInfinityNorm(matrix, n);
}

struct SideTables {
    std::vector<ImmersedIoCell> HistoryCells, CorrectionCells;
    std::vector<ImmersedWeightedIndex> HistoryRefs, PatchRefs, ActiveRefs, CorrectionRefs;
    std::vector<int> PatchBegin{0}, ActiveBegin{0};
    std::vector<float> Inverse;
    double Condition{1.};
};

SideTables BuildSide(const Grid &grid, const std::vector<Patch> &patches, const std::vector<ImmersedFilter> &filters, bool velocity, int order) {
    SideTables result;
    CellMap history, correction;
    std::vector<DeltaStencil> active_stencils;
    std::vector<double> inverse_feedthrough;
    bool has_limit = false;
    const int cells = int(grid.NumCells());
    for (size_t patch = 0; patch < patches.size(); ++patch) {
        const Patch &p = patches[patch];
        DeltaStencil raw;
        if (velocity) {
            for (int axis = 0; axis < 3; ++axis) {
                const auto component = InteriorDelta(grid, p.Center, order, axis);
                for (size_t i = 0; i < component.Index.size(); ++i) {
                    const double weight = p.Normal[size_t(axis)] * component.Weight[i];
                    if (weight == 0.) continue;
                    raw.Index.push_back(axis * cells + component.Index[i]);
                    raw.Weight.push_back(weight);
                }
            }
        } else {
            raw = InteriorDelta(grid, p.Center, order);
        }
        const double surface = p.Area / (grid.H * grid.H);
        const bool has_state = HasState(filters[patch]);
        if (has_state) {
            for (size_t i = 0; i < raw.Index.size(); ++i) {
                result.PatchRefs.push_back({raw.Index[i], float(raw.Weight[i])});
                history[raw.Index[i]].push_back({int(patch), float(surface * raw.Weight[i])});
            }
        }
        result.PatchBegin.push_back(int(result.PatchRefs.size()));

        const BoundaryImmittance &immittance = velocity ? p.Zv : p.Yp;
        const double feedthrough = filters[patch].B0;
        if (!immittance.Infinite && feedthrough == 0.) continue;
        has_limit |= immittance.Infinite;
        inverse_feedthrough.push_back(immittance.Infinite ? 0. : 1. / feedthrough);
        const double scale = std::sqrt(.5 * grid.Courant * surface);
        DeltaStencil scaled = raw;
        for (double &weight : scaled.Weight) weight *= scale;
        for (size_t i = 0; i < scaled.Index.size(); ++i) {
            result.ActiveRefs.push_back({scaled.Index[i], float(scaled.Weight[i])});
            correction[scaled.Index[i]].push_back({int(active_stencils.size()), float(scaled.Weight[i])});
        }
        result.ActiveBegin.push_back(int(result.ActiveRefs.size()));
        active_stencils.push_back(std::move(scaled));
    }
    PackCells(history, result.HistoryCells, result.HistoryRefs);
    PackCells(correction, result.CorrectionCells, result.CorrectionRefs);

    const int n = int(active_stencils.size());
    std::vector<double> gram(size_t(n) * n, 0.);
    for (int column = 0; column < n; ++column) {
        for (int row = column; row < n; ++row) {
            const double value = SparseDot(active_stencils[size_t(row)], active_stencils[size_t(column)]) +
                (row == column ? inverse_feedthrough[size_t(row)] : 0.);
            gram[size_t(column) * n + row] = value;
            gram[size_t(row) * n + column] = value;
        }
    }
    result.Condition = InvertSpd(gram, n);
    if (has_limit && (!std::isfinite(result.Condition) || result.Condition > 1e8))
        throw std::runtime_error("Immersed limiting projection Gram matrix condition exceeds 1e8");
    result.Inverse.assign(gram.begin(), gram.end());
    return result;
}

} // namespace

void Grid::Finalize() {
    if (PmlWidth < 0) throw std::runtime_error("Immersed PML width cannot be negative");
    if (Nx <= 2 * PmlWidth || Ny <= 2 * PmlWidth || Nz <= 2 * PmlWidth)
        throw std::runtime_error("Immersed grid has no interior outside its PML");
    if (!(H > 0.) || !(C > 0.) || !(Rho > 0.)) throw std::runtime_error("Immersed grid medium and spacing must be positive");
    if (!(Courant > 0.) || Courant >= 1. / std::numbers::sqrt3)
        throw std::runtime_error("Immersed grid Courant number must be strictly below 1/sqrt(3)");
    TimeStep = Courant * H / C;

    const int max_dist = std::max({Nx, Ny, Nz}) / 2 + 2;
    const auto make_pml = [&](std::vector<double> &numerator, std::vector<double> &denominator, double offset) {
        numerator.assign(size_t(max_dist), 1.);
        denominator.assign(size_t(max_dist), 1.);
        for (int distance = 0; distance < PmlWidth; ++distance) {
            const double weight = .5 * std::pow((double(PmlWidth) - distance - offset) / PmlWidth, 2.);
            numerator[size_t(distance)] = 1. - weight;
            denominator[size_t(distance)] = 1. / (1. + weight);
        }
    };
    make_pml(PmlNv, PmlDv, 0.);
    make_pml(PmlNp, PmlDp, .5);
}

void Gpu::Init(const Grid &grid, const std::vector<Point> &sources, const std::vector<Point> &receivers, int num_steps, const std::vector<float> &source_samples, const std::vector<Patch> &patches, int interpolation_order, const std::vector<uint8_t> &solid) {
    GridSpec = grid;
    GridSpec.Finalize();
    if (num_steps <= 0) throw std::runtime_error("Immersed run must have at least one step");
    if (source_samples.size() != size_t(num_steps + 1) * sources.size())
        throw std::runtime_error("Immersed source signal does not have NumSteps + 1 rows");
    const int64_t cells = int64_t(GridSpec.Nx) * GridSpec.Ny * GridSpec.Nz;
    if (cells > 2147483647ll) throw std::runtime_error("Immersed grid exceeds 32-bit indexing");

    GP.Nx = GridSpec.Nx;
    GP.Ny = GridSpec.Ny;
    GP.Nz = GridSpec.Nz;
    GP.PmlWidth = GridSpec.PmlWidth;
    GP.NumSources = int(sources.size());
    GP.NumReceivers = int(receivers.size());
    GP.NumSteps = num_steps;
    GP.NumPatches = int(patches.size());
    GP.InvRhoDtH = float(GridSpec.TimeStep / (GridSpec.Rho * GridSpec.H));
    GP.RhoCcDtH = float(GridSpec.Rho * GridSpec.C * GridSpec.C * GridSpec.TimeStep / GridSpec.H);
    GP.SourceScale = float(GridSpec.Rho * GridSpec.C * GridSpec.C * GridSpec.TimeStep / (GridSpec.H * GridSpec.H * GridSpec.H));
    GP.Courant = float(GridSpec.Courant);

    const size_t cell_bytes = GridSpec.NumCells() * sizeof(float);
    for (GpuBuffer *buffer : {&P, &Vx, &Vy, &Vz, &Px, &Py, &Pz}) buffer->ResizeZeroed(cell_bytes);

    CellMap source_by_cell;
    for (size_t source = 0; source < sources.size(); ++source) {
        const auto stencil = InteriorDelta(GridSpec, sources[source], interpolation_order);
        for (size_t i = 0; i < stencil.Index.size(); ++i) {
            if (stencil.Weight[i] != 0.)
                source_by_cell[stencil.Index[i]].push_back({int(source), float(stencil.Weight[i])});
        }
    }
    std::vector<ImmersedIoCell> source_cells;
    std::vector<ImmersedWeightedIndex> source_refs;
    PackCells(source_by_cell, source_cells, source_refs);
    GP.NumSourceCells = int(source_cells.size());
    Upload(SourceCells, source_cells);
    Upload(SourceRefs, source_refs);
    Upload(SourceSignal, source_samples);

    std::vector<int> receiver_begin{0};
    std::vector<ImmersedWeightedIndex> receiver_refs;
    for (const Point &receiver : receivers) {
        const auto stencil = InteriorDelta(GridSpec, receiver, interpolation_order);
        for (size_t i = 0; i < stencil.Index.size(); ++i)
            receiver_refs.push_back({stencil.Index[i], float(stencil.Weight[i])});
        receiver_begin.push_back(int(receiver_refs.size()));
    }
    Upload(ReceiverBegin, receiver_begin);
    Upload(ReceiverRefs, receiver_refs);
    ReceiverOut.ResizeZeroed(std::max<size_t>(4, size_t(num_steps) * receivers.size() * sizeof(float)));

    std::vector<ImmersedFilter> zv_filters, yp_filters;
    zv_filters.reserve(patches.size());
    yp_filters.reserve(patches.size());
    for (const Patch &patch : patches) {
        if (!(patch.Area > 0.)) throw std::runtime_error("Immersed patch area must be positive");
        double normal2 = 0.;
        for (const double component : patch.Normal) normal2 += component * component;
        if (std::abs(normal2 - 1.) > 1e-6) throw std::runtime_error("Immersed patch normal must be unit length");
        if (patch.Zv.Infinite && patch.Yp.Infinite)
            throw std::runtime_error("An immersed patch cannot be both rigid and pressure release");
        zv_filters.push_back(PackFilter(patch.Zv, GridSpec.TimeStep));
        yp_filters.push_back(PackFilter(patch.Yp, GridSpec.TimeStep));
    }
    const SideTables velocity = BuildSide(GridSpec, patches, zv_filters, true, interpolation_order);
    const SideTables pressure = BuildSide(GridSpec, patches, yp_filters, false, interpolation_order);
    GP.NumVelocityActive = int(velocity.ActiveBegin.size()) - 1;
    GP.NumPressureActive = int(pressure.ActiveBegin.size()) - 1;
    GP.NumVelocityHistoryCells = int(velocity.HistoryCells.size());
    GP.NumPressureHistoryCells = int(pressure.HistoryCells.size());
    GP.NumVelocityCorrectionCells = int(velocity.CorrectionCells.size());
    GP.NumPressureCorrectionCells = int(pressure.CorrectionCells.size());
    GP.HasSolid = solid.empty() ? 0 : 1;
    VelocityConditionNumber = velocity.Condition;
    PressureConditionNumber = pressure.Condition;

    Upload(ZvFilter, zv_filters);
    Upload(YpFilter, yp_filters);
    const std::vector<ImmersedFilterState> zero_states(patches.size());
    Upload(ZvState, zero_states);
    Upload(YpState, zero_states);
    Upload(VelocityHistoryCells, velocity.HistoryCells);
    Upload(VelocityHistoryRefs, velocity.HistoryRefs);
    Upload(PressureHistoryCells, pressure.HistoryCells);
    Upload(PressureHistoryRefs, pressure.HistoryRefs);
    Upload(VelocityPatchBegin, velocity.PatchBegin);
    Upload(VelocityPatchRefs, velocity.PatchRefs);
    Upload(PressurePatchBegin, pressure.PatchBegin);
    Upload(PressurePatchRefs, pressure.PatchRefs);
    Upload(VelocityActiveBegin, velocity.ActiveBegin);
    Upload(VelocityActiveRefs, velocity.ActiveRefs);
    Upload(PressureActiveBegin, pressure.ActiveBegin);
    Upload(PressureActiveRefs, pressure.ActiveRefs);
    Upload(VelocityInverse, velocity.Inverse);
    Upload(PressureInverse, pressure.Inverse);
    Upload(VelocityCorrectionCells, velocity.CorrectionCells);
    Upload(VelocityCorrectionRefs, velocity.CorrectionRefs);
    Upload(PressureCorrectionCells, pressure.CorrectionCells);
    Upload(PressureCorrectionRefs, pressure.CorrectionRefs);
    VelocityT.ResizeZeroed(std::max<size_t>(4, size_t(GP.NumVelocityActive) * sizeof(float)));
    VelocitySolution.ResizeZeroed(std::max<size_t>(4, size_t(GP.NumVelocityActive) * sizeof(float)));
    PressureT.ResizeZeroed(std::max<size_t>(4, size_t(GP.NumPressureActive) * sizeof(float)));
    PressureY.ResizeZeroed(std::max<size_t>(4, size_t(GP.NumPressureActive) * sizeof(float)));

    Upload(PmlNv, GridSpec.PmlNv);
    Upload(PmlDv, GridSpec.PmlDv);
    Upload(PmlNp, GridSpec.PmlNp);
    Upload(PmlDp, GridSpec.PmlDp);
    if (!solid.empty() && solid.size() != GridSpec.NumCells())
        throw std::runtime_error("Immersed solid mask has the wrong size");
    if (solid.empty()) Solid.ResizeZeroed(std::max<size_t>(4, GridSpec.NumCells()));
    else Upload(Solid, solid);
    Params.ResizeZeroed(sizeof GP);
    Params.Upload(&GP, sizeof GP);

    auto &context = MetalContext::Get();
    PsoVelocity = context.ImmersedPipeline("ImmersedVelocity");
    PsoPressure = context.ImmersedPipeline("ImmersedPressure");
    PsoSource = context.ImmersedPipeline("ImmersedSource");
    PsoSample = context.ImmersedPipeline("ImmersedSample");
    PsoVelocityHistory = context.ImmersedPipeline("ImmersedVelocityHistory");
    PsoVelocityGather = context.ImmersedPipeline("ImmersedVelocityGather");
    PsoVelocityScatter = context.ImmersedPipeline("ImmersedVelocityScatter");
    PsoVelocityAdvance = context.ImmersedPipeline("ImmersedVelocityAdvance");
    PsoPressureHistory = context.ImmersedPipeline("ImmersedPressureHistory");
    PsoPressureGather = context.ImmersedPipeline("ImmersedPressureGather");
    PsoPressureScatter = context.ImmersedPipeline("ImmersedPressureScatter");
    PsoPressureAdvance = context.ImmersedPipeline("ImmersedPressureAdvance");
    PsoDense = context.ImmersedPipeline("ImmersedDense");
    PsoStreamVelocity = context.ImmersedPipeline("ImmersedVelocityStream");
    PsoStreamPressure = context.ImmersedPipeline("ImmersedPressureStream");
    GridThreads = {32, 4, 4};
    GridTiles = {(uint32_t(GridSpec.Nx) + GridThreads.x - 1) / GridThreads.x, (uint32_t(GridSpec.Ny) + GridThreads.y - 1) / GridThreads.y, (uint32_t(GridSpec.Nz) + GridThreads.z - 1) / GridThreads.z};
    StepN = 0;
}

void Gpu::RunSteps(int count) {
    if (count < 0 || StepN + count > GP.NumSteps) throw std::runtime_error("Immersed step count leaves the configured run");
    auto &context = MetalContext::Get();
    const ImmersedSolveParams velocity_solve{GP.NumVelocityActive};
    const ImmersedSolveParams pressure_solve{GP.NumPressureActive};
    for (int i = 0; i < count; ++i) {
        const ImmersedStepParams step{StepN};
        context.Dispatch(PsoVelocity, GridTiles, GridThreads, {&P, &Vx, &Vy, &Vz, &PmlNv, &PmlDv, &Solid, &Params});
        context.Dispatch(PsoVelocityHistory, Blocks1d(size_t(GP.NumVelocityHistoryCells)), {Group, 1, 1}, {&Vx, &Vy, &Vz, &ZvFilter, &ZvState, &VelocityHistoryCells, &VelocityHistoryRefs, &Params});
        context.Dispatch(PsoVelocityGather, Blocks1d(size_t(GP.NumVelocityActive)), {Group, 1, 1}, {&Vx, &Vy, &Vz, &VelocityActiveBegin, &VelocityActiveRefs, &VelocityT, &Params});
        context.Dispatch(PsoDense, Blocks1d(size_t(GP.NumVelocityActive)), {Group, 1, 1}, {&VelocityT, &VelocityInverse, &VelocitySolution}, &velocity_solve, sizeof velocity_solve);
        context.Dispatch(PsoVelocityScatter, Blocks1d(size_t(GP.NumVelocityCorrectionCells)), {Group, 1, 1}, {&Vx, &Vy, &Vz, &VelocityCorrectionCells, &VelocityCorrectionRefs, &VelocitySolution, &Params});
        context.Dispatch(PsoVelocityAdvance, Blocks1d(size_t(GP.NumPatches)), {Group, 1, 1}, {&Vx, &Vy, &Vz, &VelocityPatchBegin, &VelocityPatchRefs, &ZvFilter, &ZvState, &Params});
        context.Dispatch(PsoPressure, GridTiles, GridThreads, {&P, &Vx, &Vy, &Vz, &Px, &Py, &Pz, &PmlNp, &PmlDp, &Solid, &Params});
        context.Dispatch(PsoPressureHistory, Blocks1d(size_t(GP.NumPressureHistoryCells)), {Group, 1, 1}, {&P, &YpFilter, &YpState, &PressureHistoryCells, &PressureHistoryRefs, &Params});
        context.Dispatch(PsoSource, Blocks1d(size_t(GP.NumSourceCells)), {Group, 1, 1}, {&P, &SourceCells, &SourceRefs, &SourceSignal, &Params}, &step, sizeof step);
        context.Dispatch(PsoPressureGather, Blocks1d(size_t(GP.NumPressureActive)), {Group, 1, 1}, {&P, &PressureActiveBegin, &PressureActiveRefs, &PressureT, &Params});
        context.Dispatch(PsoDense, Blocks1d(size_t(GP.NumPressureActive)), {Group, 1, 1}, {&PressureT, &PressureInverse, &PressureY}, &pressure_solve, sizeof pressure_solve);
        context.Dispatch(PsoPressureScatter, Blocks1d(size_t(GP.NumPressureCorrectionCells)), {Group, 1, 1}, {&P, &PressureCorrectionCells, &PressureCorrectionRefs, &PressureY, &Params});
        context.Dispatch(PsoPressureAdvance, Blocks1d(size_t(GP.NumPatches)), {Group, 1, 1}, {&P, &PressurePatchBegin, &PressurePatchRefs, &YpFilter, &YpState, &Params});
        context.Dispatch(PsoSample, Blocks1d(size_t(GP.NumReceivers)), {Group, 1, 1}, {&P, &ReceiverBegin, &ReceiverRefs, &ReceiverOut, &Params}, &step, sizeof step);
        ++StepN;
    }
    context.Flush();
}

double Gpu::RunTimedSteps(int count) {
    if (count <= 0) throw std::runtime_error("Immersed timed step count must be positive");
    RunSteps(count);
    auto &context = MetalContext::Get();
    context.Sync();
    return context.TakeBatchGpuSeconds() / count;
}

void Gpu::SeedPressure(const std::vector<float> &pressure) {
    if (pressure.size() != GridSpec.NumCells()) throw std::runtime_error("Immersed pressure seed has the wrong size");
    MetalContext::Get().Drain();
    P.Upload(pressure.data(), pressure.size() * sizeof(float));
    Px.Zero(Px.Capacity());
    Py.Zero(Py.Capacity());
    Pz.Zero(Pz.Capacity());
}

void Gpu::SeedVelocity(const std::vector<float> &vx, const std::vector<float> &vy, const std::vector<float> &vz) {
    if (vx.size() != GridSpec.NumCells() || vy.size() != GridSpec.NumCells() || vz.size() != GridSpec.NumCells())
        throw std::runtime_error("Immersed velocity seed has the wrong size");
    MetalContext::Get().Drain();
    Vx.Upload(vx.data(), vx.size() * sizeof(float));
    Vy.Upload(vy.data(), vy.size() * sizeof(float));
    Vz.Upload(vz.data(), vz.size() * sizeof(float));
}

Gpu::Bandwidth Gpu::Roofline(int repetitions) {
    if (repetitions <= 0) throw std::runtime_error("Roofline repetition count must be positive");
    auto &context = MetalContext::Get();
    const size_t bytes = GridSpec.NumCells() * sizeof(float);

    const auto time_step = [&] {
        for (int i = 0; i < repetitions; ++i) {
            context.Dispatch(PsoVelocity, GridTiles, GridThreads, {&P, &Vx, &Vy, &Vz, &PmlNv, &PmlDv, &Solid, &Params});
            context.Dispatch(PsoPressure, GridTiles, GridThreads, {&P, &Vx, &Vy, &Vz, &Px, &Py, &Pz, &PmlNp, &PmlDp, &Solid, &Params});
        }
        context.Sync();
        return context.TakeBatchGpuSeconds() / repetitions;
    };
    const auto time_stream = [&] {
        for (int i = 0; i < repetitions; ++i) {
            context.Dispatch(PsoStreamVelocity, GridTiles, GridThreads, {&Vx, &Vy, &Vz, &Params});
            context.Dispatch(PsoStreamPressure, GridTiles, GridThreads, {&P, &Params});
        }
        context.Sync();
        return context.TakeBatchGpuSeconds() / repetitions;
    };

    std::vector<double> step_samples, stream_samples;
    for (int round = 0; round < 6; ++round) {
        const double step = time_step();
        const double stream = time_stream();
        if (round > 0) {
            step_samples.push_back(step);
            stream_samples.push_back(stream);
        }
    }
    std::ranges::sort(step_samples);
    std::ranges::sort(stream_samples);
    const double step_seconds = step_samples[step_samples.size() / 2];
    const double stream_seconds = stream_samples[stream_samples.size() / 2];
    return {step_seconds, stream_seconds, 8. * double(bytes)};
}

} // namespace immersed
