#include "RoomGpu.h"

#include "Profile.h"

#include <Metal/Metal.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace {
// The fp32 diagonal shift the reference applies to keep a single-precision run from
// accumulating a DC mode (fdtd_common.h's EPS). Its double-precision path uses 0 instead, and
// this solver is single precision throughout, so the shift is always on.
constexpr double Eps = 1.19209289e-07;

constexpr uint32_t Group = 256; // threads of the one-dimensional passes

Dim3 Blocks1d(int n) { return {(uint32_t(n) + Group - 1) / Group, 1, 1}; }

// The count RoomShellQ's position test picks out, against the closed form for the shell of a
// box. On the FCC grid it is the shell of the *unfolded* box, whose y extent is Nyf and half
// of whose nodes are off the sublattice, and the y-extreme layers 1 and Nyf-2 both fold onto
// layer 1 — so the folded shell is one node deep in y where it is two deep in x and z.
void CheckShellCount(int nx, int ny, int nz, bool fcc) {
    const int64_t nyf = fcc ? 2ll * (ny - 1) : ny;
    const int64_t closed_form = (2ll * (int64_t(nx) * nyf + int64_t(nx) * nz + nyf * nz) - 12ll * (nx + nyf + nz) + 56) / (fcc ? 2 : 1);
    const int64_t box = int64_t(nx - 2) * (ny - 2) * (nz - 2);
    const int64_t inside = int64_t(nx - 4) * (fcc ? ny - 3 : ny - 4) * (nz - 4);
    if (box - inside != closed_form) throw std::runtime_error("Engquist-Majda shell node count disagrees with the closed form");
}
} // namespace

RoomCoefficients RoomUpdateCoefficients(const RoomScene &scene) {
    // The FCC Laplacian averages twelve neighbours at a face diagonal's distance rather than
    // six at a cell's, which is a factor of four out front and twelve on the diagonal.
    const double lfac = scene.Fcc ? .25 : 1.;
    const double sl2 = (1. + Eps) * lfac * scene.L2;
    return {float(2. - sl2 * scene.Neighbours()), float(lfac * scene.L2), float(sl2), float(scene.L)};
}

RoomMaterials RoomMaterialCoefficients(const RoomScene &scene) {
    RoomMaterials materials;
    materials.Quads.assign(size_t(scene.NumMaterials()) * RoomMaxBranches, RoomMatQuad{0.f, 0.f, 0.f, 0.f});
    materials.Beta.assign(size_t(scene.NumMaterials()), 0.f);
    size_t row = 0;
    for (int k = 0; k < scene.NumMaterials(); ++k) {
        for (int m = 0; m < scene.MatBranches[size_t(k)]; ++m, ++row) {
            const double d = scene.MatDef[row * 3 + 0], e = scene.MatDef[row * 3 + 1], f = scene.MatDef[row * 3 + 2];
            const double dh = d / scene.Ts, eh = e, fh = f * scene.Ts;
            const double b = 1. / (2. * dh + eh + .5 * fh);
            if (!std::isfinite(b)) throw std::runtime_error("Material branch has no finite discrete admittance");
            auto &q = materials.Quads[size_t(k) * RoomMaxBranches + m];
            q.B = float(b);
            q.Bd = float(b * (2. * dh - eh - .5 * fh));
            q.BDh = float(b * dh);
            q.BFh = float(b * fh);
            // The reference accumulates beta in the working precision, not in double, and the
            // node update divides by 1 + lo2*S*beta — so the sum has to be the float one.
            materials.Beta[size_t(k)] += float(b);
        }
    }
    return materials;
}

RoomLossyNodes RoomLossySubset(const RoomScene &scene) {
    RoomLossyNodes lossy;
    // The reference narrows the FCC factor to the working precision before it multiplies, so
    // this does too.
    const double scale = scene.Fcc ? double(float(.5 / std::numbers::sqrt2)) : 1.;
    for (size_t i = 0; i < scene.MatBn.size(); ++i) {
        if (scene.MatBn[i] < 0) continue;
        lossy.Ixyz.push_back(scene.BnIxyz[i]);
        lossy.Ssaf.push_back(float(scale * scene.SafBn[i]));
        lossy.Mat.push_back(scene.MatBn[i]);
    }
    return lossy;
}

void RoomGpu::Init(const RoomScene &scene) {
    const profile::Scope scope{"room/init"};
    auto &ctx = MetalContext::Get();

    const int64_t nodes = int64_t(scene.Nx) * scene.Ny * scene.Nz;
    if (nodes > 2147483647ll) throw std::runtime_error("Room grid exceeds the 32-bit node indexing the kernels use");
    if (scene.Nx < 6 || scene.Ny < 6 || scene.Nz < 6) throw std::runtime_error("Room grid is too small for the halo mirror");

    Fcc = scene.Fcc;
    GP.Nx = scene.Nx;
    GP.Ny = scene.Ny;
    GP.Nz = scene.Nz;
    GP.Ns = scene.NumSources();
    GP.Nr = scene.NumOutputs();
    GP.Nt = scene.Nt;
    const auto coeffs = RoomUpdateCoefficients(scene);
    GP.A1 = coeffs.A1;
    GP.A2 = coeffs.A2;
    GP.Sl2 = coeffs.Sl2;
    GP.L = coeffs.L;
    GP.Lo2 = float(.5 * scene.L);

    CheckShellCount(scene.Nx, scene.Ny, scene.Nz, Fcc);

    for (auto &u : U) u.ResizeZeroed(size_t(nodes) * sizeof(float));
    I0 = 0;
    I1 = 1;
    StepN = 0;

    const auto upload = [&](GpuBuffer &buffer, const auto &values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        buffer.ResizeZeroed(std::max<size_t>(4, values.size() * sizeof(T)));
        buffer.Upload(values.data(), values.size() * sizeof(T));
    };

    // Cartesian masks are grid-wide; FCC masks are packed and reached through block rank.
    const int64_t nzny = int64_t(scene.Nz) * scene.Ny;
    const auto check_inside = [&](int ii) {
        const int64_t ix = ii / nzny, iy = (ii / scene.Nz) % scene.Ny, iz = ii % scene.Nz;
        if (ix < 1 || ix > scene.Nx - 2 || iy < 1 || iy > scene.Ny - 2 || iz < 1 || iz > scene.Nz - 2)
            throw std::runtime_error("A boundary node sits on the outer grid box, where the stencil has no mirror");
    };
    if (Fcc) {
        std::vector<RoomBnBlockEntry> blocks(size_t((nodes + RoomBnBlock - 1) / RoomBnBlock), RoomBnBlockEntry{});
        int previous = -1;
        for (int const ii : scene.BnIxyz) {
            check_inside(ii);
            // Block rank requires boundary nodes in grid order.
            if (ii <= previous) throw std::runtime_error("FCC boundary nodes are not in ascending index order, which the block ranks the interior pass reads assume");
            previous = ii;
            blocks[size_t(ii) / RoomBnBlock].Occupied |= 1u << (ii % RoomBnBlock);
        }
        uint32_t rank = 0;
        for (auto &block : blocks) {
            block.Rank = rank;
            rank += uint32_t(std::popcount(block.Occupied));
        }
        upload(BnMask, blocks);
        upload(AdjBn, scene.AdjBn);
    } else {
        const size_t node_count = size_t(nodes);
        std::vector<uint8_t> adj(node_count, uint8_t(RoomAirAdj));
        for (size_t i = 0; i < scene.BnIxyz.size(); ++i) {
            const int ii = scene.BnIxyz[i];
            check_inside(ii);
            if (scene.AdjBn[i] == RoomAirAdj) throw std::runtime_error("A boundary node keeps every neighbour, which is the value that marks ordinary air");
            adj[size_t(ii)] = uint8_t(scene.AdjBn[i]);
        }
        upload(BnMask, adj);
    }

    upload(InIxyz, scene.InIxyz);
    upload(OutIxyz, scene.OutIxyz);

    std::vector<float> in_sigs(scene.InSigs.size());
    for (size_t i = 0; i < in_sigs.size(); ++i) in_sigs[i] = float(scene.InSigs[i]);
    upload(InSigs, in_sigs);

    const auto lossy = RoomLossySubset(scene);
    const auto materials = RoomMaterialCoefficients(scene);
    GP.Nbl = int(lossy.Ixyz.size());
    // Fixed fields move 25 bytes; each LRC branch reads and writes 16 more.
    LossyBytes = 0.;
    for (const int8_t m : lossy.Mat) LossyBytes += 25. + 16. * double(scene.MatBranches[size_t(m)]);
    upload(BnlIxyz, lossy.Ixyz);
    upload(SsafBnl, lossy.Ssaf);
    upload(MatBnl, lossy.Mat);
    upload(MatMb, scene.MatBranches);
    upload(MatBeta, materials.Beta);
    upload(MatQuads, materials.Quads);

    const size_t branch_bytes = std::max<size_t>(4, size_t(GP.Nbl) * RoomMaxBranches * sizeof(float));
    Vh1.ResizeZeroed(branch_bytes);
    Gh1.ResizeZeroed(branch_bytes);
    for (auto &b : Ub) b.ResizeZeroed(std::max<size_t>(4, size_t(GP.Nbl) * sizeof(float)));
    I0b = 0;
    I1b = 1;
    I2b = 2;

    Out.ResizeZeroed(std::max<size_t>(4, size_t(GP.Nr) * size_t(GP.Nt) * sizeof(float)));

    Params.ResizeZeroed(sizeof(RoomGridParams));
    Params.Upload(&GP, sizeof GP);

    PsoAir = ctx.RoomPipeline(Fcc ? "RoomAirFcc" : "RoomAir");
    PsoLossy = ctx.RoomPipeline("RoomLossy");
    PsoIo = ctx.RoomPipeline("RoomIo");
    PsoEnergy = ctx.RoomPipeline(Fcc ? "RoomEnergyFcc" : "RoomEnergy");

    // z is the contiguous axis, so a 32-wide row of it keeps the stencil's reads sequential.
    AirThreads = {32, 4, 4};
    AirTiles = {(uint32_t(GP.Nz - 2) + AirThreads.x - 1) / AirThreads.x, (uint32_t(GP.Ny - 2) + AirThreads.y - 1) / AirThreads.y, (uint32_t(GP.Nx - 2) + AirThreads.z - 1) / AirThreads.z};

    TimeKernels = std::getenv("ACOUSTIC_ROOM_KERNEL_TIMES") != nullptr;
}

void RoomGpu::Timed(int slot, const char *name, MTL::ComputePipelineState *pso, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, const void *params, size_t params_size) {
    auto &ctx = MetalContext::Get();
    ctx.Dispatch(pso, blocks, threads, buffers, params, params_size);
    if (!TimeKernels) return;
    ctx.Sync();
    if (int(KernelTiming.size()) <= slot) KernelTiming.resize(slot + 1);
    KernelTiming[slot].Name = name;
    KernelTiming[slot].Seconds += ctx.TakeBatchGpuSeconds();
    KernelTiming[slot].Count += 1;
}

void RoomGpu::RunSteps(int n_steps) {
    static profile::Entry &phase = profile::Phase("room/steps");
    const profile::Scope scope{phase};
    auto &ctx = MetalContext::Get();

    for (int local = 0; local < n_steps; ++local) {
        if (StepN >= GP.Nt) throw std::runtime_error("Room step index ran past the scene's step count");
        const RoomStepParams step{StepN};

        if (Fcc) Timed(0, "air", PsoAir, AirTiles, AirThreads, {&U[I0], &U[I1], &BnMask, &AdjBn, &Params});
        else Timed(0, "air", PsoAir, AirTiles, AirThreads, {&U[I0], &U[I1], &BnMask, &Params});
        if (GP.Nbl > 0)
            Timed(1, "lossy", PsoLossy, Blocks1d(GP.Nbl), {Group, 1, 1}, {&U[I0], &Ub[I0b], &Ub[I2b], &Vh1, &Gh1, &BnlIxyz, &SsafBnl, &MatBnl, &MatMb, &MatBeta, &MatQuads, &Params});
        Timed(2, "io", PsoIo, Blocks1d(GP.Nr + GP.Ns), {Group, 1, 1}, {&U[I0], &Out, &U[I1], &OutIxyz, &InSigs, &InIxyz, &Params}, &step, sizeof step);

        std::swap(I0, I1); // u(n+1) becomes the current level, u(n) the one to overwrite
        // The boundary ring rotates the other way: what was just written becomes u1b, and the
        // oldest slot is the one the next step overwrites.
        const int oldest = I2b;
        I2b = I1b;
        I1b = I0b;
        I0b = oldest;
        ++StepN;
    }
    ctx.Flush();
}

void RoomGpu::Seed(const std::vector<float> &u) {
    const size_t bytes = u.size() * sizeof(float);
    if (bytes != size_t(GP.Nx) * GP.Ny * GP.Nz * sizeof(float)) throw std::runtime_error("Seed state is not one value a node");
    MetalContext::Get().Drain();
    for (const auto &level : U) level.Upload(u.data(), bytes);
    // At rest means at rest everywhere, walls included.
    Vh1.Zero(Vh1.Capacity());
    Gh1.Zero(Gh1.Capacity());
    for (const auto &b : Ub) b.Zero(b.Capacity());
}

double RoomGpu::Energy(int x0, int x1, int y0, int y1, int z0, int z1) {
    // The diagonal shift the interior update is actually carrying, read back out of A1
    // rather than recomputed from eps: A1 is the rounded 2 - 6 Sl2, and its rounding is worth
    // 17% of the shift itself. Boundary nodes take theirs from Sl2 instead, which differs in
    // the last bits, but they are a few percent of any box.
    const int neighbours = Fcc ? int(RoomFccNeighbours) : int(RoomNumNeighbours);
    const float diag = float((2. - double(GP.A1) - neighbours * double(GP.A2)) / neighbours);
    const RoomEnergyParams box{x0, x1, y0, y1, z0, z1, GP.A2, diag};
    const int64_t n = int64_t(x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
    if (n <= 0) throw std::runtime_error("Energy box is empty");
    EnergyOut.Resize(size_t(n) * sizeof(float));
    MetalContext::Get().Dispatch(PsoEnergy, Blocks1d(int(n)), {Group, 1, 1}, {&EnergyOut, &U[I1], &U[I0], &Params}, &box, sizeof box);

    const float *partials = EnergyOut.As<float>(); // synchronizes
    double total = 0.;
    for (int64_t i = 0; i < n; ++i) total += partials[i];
    return total;
}

double RoomGpu::TimePass(MTL::ComputePipelineState *pso, Dim3 blocks, Dim3 threads, std::initializer_list<GpuSlice> buffers, int reps) {
    auto &ctx = MetalContext::Get();
    for (int round = 0; round < 2; ++round) {
        for (int i = 0; i < reps; ++i) ctx.Dispatch(pso, blocks, threads, buffers);
        ctx.Sync();
        if (round == 0) ctx.TakeBatchGpuSeconds(); // the clock is still ramping through this one
    }
    return ctx.TakeBatchGpuSeconds() / reps;
}

RoomGpu::Bandwidth RoomGpu::Roofline(int reps) {
    auto *stream = MetalContext::Get().RoomPipeline("RoomStream");
    const double nodes = double(int64_t(GP.Nx - 2) * (GP.Ny - 2) * (GP.Nz - 2));
    const double update = Fcc ? TimePass(PsoAir, AirTiles, AirThreads, {&U[I0], &U[I1], &BnMask, &AdjBn, &Params}, reps) : TimePass(PsoAir, AirTiles, AirThreads, {&U[I0], &U[I1], &BnMask, &Params}, reps);
    return {update, TimePass(stream, AirTiles, AirThreads, {&U[I0], &U[I1], &Params}, reps), 12. * nodes};
}

RoomGpu::Bandwidth RoomGpu::LossyRoofline(int reps) {
    if (GP.Nbl == 0) return {0., 0., 0.};
    auto &ctx = MetalContext::Get();
    auto *stream = ctx.RoomPipeline("RoomLossyStream");
    const Dim3 blocks = Blocks1d(GP.Nbl), threads{Group, 1, 1};
    const double update = TimePass(PsoLossy, blocks, threads, {&U[I0], &Ub[I0b], &Ub[I2b], &Vh1, &Gh1, &BnlIxyz, &SsafBnl, &MatBnl, &MatMb, &MatBeta, &MatQuads, &Params}, reps);
    return {update, TimePass(stream, blocks, threads, {&U[I0], &Ub[I0b], &Ub[I2b], &Vh1, &Gh1, &BnlIxyz, &SsafBnl, &MatBnl, &MatMb, &Params}, reps), LossyBytes};
}

void RoomGpu::ReportTimings(const std::vector<KernelTimes> &timings) {
    double total = 0.;
    uint64_t steps = 0;
    for (const auto &k : timings) {
        total += k.Seconds;
        steps = std::max(steps, k.Count);
    }
    if (steps == 0) return;
    std::printf("[room kernels] %llu steps, %.3f ms/step total\n", (unsigned long long)steps, 1e3 * total / double(steps));
    for (const auto &k : timings) {
        if (!k.Name) continue;
        std::printf("  %-10s %8.4f ms/step  %5.1f%%  (%llu dispatches)\n", k.Name, 1e3 * k.Seconds / double(steps), 100. * k.Seconds / total, (unsigned long long)k.Count);
    }
}
