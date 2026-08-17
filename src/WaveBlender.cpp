// Ported from WaveBlender (c) 2024 Kangrui Xue (WaveBlender.cu + GPUSolver.cu) — Metal port.
// See WaveBlender.h for the batch pipeline and its synchronization contract.
//
// References:
//   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains

#include "WaveBlender.h"

#include "KernelParams.h"
#include "Parallel.h"
#include "Profile.h"

#include <Metal/Metal.hpp>

using REAL = real; // tribox.h is written against a `REAL` scalar its includer supplies
#include "tribox.h"

#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <string_view>

namespace {
// Whether a cell state is solid at batch end (Rising converges to solid, Falling to air)
bool EndsSolid(uint8_t state) { return state == CellSolid || state == CellRising; }
} // namespace

WaveBlender::WaveBlender(const SimParams &params)
    : Params(params), GridSize(Params.Nx * Params.Ny * Params.Nz),
      NFdtdSamples(Params.FdtdSrate / Params.BlendRate), NShaderSamples(Params.ShaderSrate / Params.BlendRate + 1),
      RhoCcDt(Params.Rho * Params.C * Params.C * Params.Dt), InvDx(1. / Params.Dx), InvRhoDt(1. / Params.Rho * Params.Dt),
      Damping(Params.Damping) {
    // Wide-x threadgroup tiles win on large grids, square ones on small (partial rows).
    FdtdTg = Params.Nx >= 88 ? Dim3{FdtdTgXWide, FdtdTgYWide, FdtdTgZWide} : Dim3{FdtdTgXSmall, FdtdTgYSmall, FdtdTgZSmall};
    FdtdTiles = {
        uint32_t(Params.Nx + FdtdTg.x - 1) / FdtdTg.x,
        uint32_t(Params.Ny + FdtdTg.y - 1) / FdtdTg.y,
        uint32_t(Params.Nz + FdtdTg.z - 1) / FdtdTg.z,
    };

    for (auto *field : {&P, &Vx, &Vy, &Vz}) { // both ping-pong slots
        field->Cur().ResizeZeroed(GridSize * sizeof(real));
        field->Other().ResizeZeroed(GridSize * sizeof(real));
    }

    Cell1.resize(GridSize);
    Cell2.resize(GridSize);

    // Both slots all-Air: Cur() doubles as the previous batch-end solidity source
    CellState.Cur().ResizeZeroed(GridSize * sizeof(uint8_t));
    CellState.Other().ResizeZeroed(GridSize * sizeof(uint8_t));

    // Bound but never read by batches with no boundary faces
    ShaderFaces.Cur().ResizeZeroed(16);
    ShaderFaces.Other().ResizeZeroed(16);

    for (auto &stamp : FaceStamp) stamp.assign(GridSize, 0);
    for (auto &col : FaceCol) col.assign(GridSize, -1);
    FreshComponent.assign(GridSize, -1);
    FloodVisited.assign(GridSize, 0);

    InitializePml();
}

// Same simple quadratic-ramp split-field PML as the reference (see its TODO on C-PML).
void WaveBlender::InitializePml() {
    Px.ResizeZeroed(GridSize * sizeof(real));
    Py.ResizeZeroed(GridSize * sizeof(real));
    Pz.ResizeZeroed(GridSize * sizeof(real));

    // Shell-packed split-field storage (the slab layout PsiIndex in Kernels.metal computes)
    const long w = PmlWidth;
    const long shell = 2 * Params.Nx * Params.Ny * w + 2 * Params.Nx * w * (Params.Nz - 2 * w) + 2 * w * (Params.Ny - 2 * w) * (Params.Nz - 2 * w);
    PsiX.ResizeZeroed(shell * sizeof(real));
    PsiY.ResizeZeroed(shell * sizeof(real));
    PsiZ.ResizeZeroed(shell * sizeof(real));

    const int max_half_grid_length = (std::max({Params.Nx, Params.Ny, Params.Nz}) + 1) / 2;

    // Velocity and pressure PML weights: numerator and denominator
    std::vector<real> pml_nv(max_half_grid_length + 1, 1.), pml_dv(max_half_grid_length + 1, 1. / (1. + Damping));
    std::vector<real> pml_np(max_half_grid_length, 1.), pml_dp(max_half_grid_length, 1.);

    for (int dist = 0; dist < PmlWidth; ++dist) {
        real weight = (real(PmlWidth) - dist) / real(PmlWidth); // velocity
        weight = 0.5 * weight * weight;
        pml_nv[dist] = 1. - weight;
        pml_dv[dist] = 1. / (1. + weight);

        weight = (real(PmlWidth) - dist - 0.5) / real(PmlWidth); // pressure
        weight = 0.5 * weight * weight;
        pml_np[dist] = 1. - weight;
        pml_dp[dist] = 1. / (1. + weight);
    }
    PmlNv.Resize((max_half_grid_length + 1) * sizeof(real));
    PmlDv.Resize((max_half_grid_length + 1) * sizeof(real));
    PmlNv.Upload(pml_nv.data(), (max_half_grid_length + 1) * sizeof(real));
    PmlDv.Upload(pml_dv.data(), (max_half_grid_length + 1) * sizeof(real));

    PmlNp.Resize(max_half_grid_length * sizeof(real));
    PmlDp.Resize(max_half_grid_length * sizeof(real));
    PmlNp.Upload(pml_np.data(), max_half_grid_length * sizeof(real));
    PmlDp.Upload(pml_dp.data(), max_half_grid_length * sizeof(real));
}

void WaveBlender::AddListener(const std::string &format, const std::vector<real> &position, const std::string &output_name) {
    const int i = (position[0] / Params.Dx) + (Params.Nx - 1) / 2.;
    const int j = (position[1] / Params.Dx) + (Params.Ny - 1) / 2.;
    const int k = (position[2] / Params.Dx) + (Params.Nz - 1) / 2.;

    if (format != "Mono") throw std::runtime_error(std::format("Invalid listener format: {}", format));
    Listeners.push_back({Cid(i, j, k), std::ofstream{output_name + ".bin", std::ofstream::binary}});
}

void WaveBlender::InitializeListeners() {
    std::vector<int> cids;
    cids.reserve(Listeners.size());
    for (const auto &listener : Listeners) cids.push_back(listener.Cid);

    ListenerCids.Resize(std::max<size_t>(1, cids.size()) * sizeof(int));
    ListenerCids.Upload(cids.data(), cids.size() * sizeof(int));
    ListenerOut.Cur().ResizeZeroed(std::max<size_t>(1, size_t(NFdtdSamples) * cids.size()) * sizeof(real));
    ListenerOut.Other().ResizeZeroed(std::max<size_t>(1, size_t(NFdtdSamples) * cids.size()) * sizeof(real));
}

void WaveBlender::WritePendingListeners() {
    if (!ListenerPending) return;
    ListenerPending = false;

    // The slot was drained at the last sync point, and the running batch fills the other
    // slot — so read without syncing (As<> would wait out the whole running batch).
    const auto *samples = static_cast<const real *>(PendingListenerSlot->RawData());
    const int n = Listeners.size();
    std::vector<real> deinterleaved(n > 1 ? NFdtdSamples : 0);
    for (int l = 0; l < n; ++l) {
        auto &listener = Listeners[l];
        if (n == 1) {
            listener.Out.write(reinterpret_cast<const char *>(samples), NFdtdSamples * sizeof(real));
        } else {
            for (int s = 0; s < NFdtdSamples; ++s) deinterleaved[s] = samples[size_t(s) * n + l];
            listener.Out.write(reinterpret_cast<const char *>(deinterleaved.data()), NFdtdSamples * sizeof(real));
        }
    }
}

void WaveBlender::LogZSlice(const std::string &filetag) {
    const int offset = Cid(0, 0, Params.Nz / 2); // z-slice
    std::ofstream logfile{filetag, std::ofstream::binary};

    const auto *p = P.Cur().As<real>(); // synchronizes the stream
    logfile.write(reinterpret_cast<const char *>(p + offset), Params.Nx * Params.Ny * sizeof(real));

    const auto *states = CellState.Cur().As<uint8_t>();
    std::vector<real> beta(Params.Nx * Params.Ny); // converged between batches: exactly 0 or 1
    for (int c = 0; c < int(beta.size()); ++c) beta[c] = EndsSolid(states[offset + c]) ? 1. : 0.;
    logfile.write(reinterpret_cast<const char *>(beta.data()), beta.size() * sizeof(real));
}

bool WaveBlender::RunBatch() {
    if (Step >= (Params.Tf - Params.Ts) * Params.FdtdSrate) {
        MetalContext::Get().Drain(); // drain the final batch
        WritePendingListeners();
        Faces.Report();
        return false;
    }
    if (Step == 0) InitializeListeners();

    Cell1 = Cell2;
    ObjectBounds.resize(Objects.size());
    PrevObjectBounds = ObjectBounds; // Cell1's bounds: the last rasterization's
    PrevCavityBounds = CavityBounds;

    // 1. Check if objects have moved; if not, we can skip step 2
    bool keep_value[256]{}; // keep_value[v]: cell value v belongs to an object unchanged this batch
    bool any_changed = false;
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        if (!Base(Objects[oid]).Changed) keep_value[oid + 1] = true;
        else any_changed = true;
    }
    // 2. Rasterize objects and setup shaders
    if (any_changed) {
        {
            const profile::Scope scope{"cpu/clear_cells"};
            // Only clear cells of objects that moved (plus the cavity fill, which is
            // never kept) — all within their previous rasterization bounds
            CellBox clear_box = PrevCavityBounds;
            for (int oid = 0; oid < int(Objects.size()); ++oid) {
                if (Base(Objects[oid]).Changed) clear_box.Expand(PrevObjectBounds[oid]);
            }
            ForEachCell(clear_box, Params.Nx, Params.Ny, [&](int cid, int, int, int) {
                if (!keep_value[Cell2[cid]]) Cell2[cid] = 0;
            });
        }
        {
            const profile::Scope scope{"cpu/rasterize"};
            Rasterize(keep_value);
        }
        {
            const profile::Scope scope{"cpu/detect_cavities"};
            DetectCavities();
        }

        if (Step == 0) { // batch-start solidity (pre point-source reversion, as in the reference)
            auto *states = static_cast<uint8_t *>(CellState.Cur().RawData()); // no GPU work in flight yet
            for (int cid = 0; cid < GridSize; ++cid) states[cid] = Cell2[cid] > 0 ? CellSolid : CellAir;
        }
        {
            const profile::Scope scope{"cpu/setup_shaders"};
            SetupShaders();
        }
        // Per-cell blending states for the batch, written into the spare CellState slot
        // (the in-flight batch reads Cur(), which also holds the previous batch-end
        // solidity). Cells whose solidity changed transition over the batch. Solid-now
        // and solid-before cells both lie within the current-or-previous bounds union,
        // so everything outside is Air.
        const auto *prev = static_cast<const uint8_t *>(CellState.Cur().RawData());
        auto *states = static_cast<uint8_t *>(CellState.Other().RawData());
        std::memset(states, CellAir, GridSize);
        HadTransitions = false;
        ForEachCell(SolidBounds(true), Params.Nx, Params.Ny, [&](int cid, int, int, int) {
            const bool solid = Cell2[cid] > 0;
            if (solid != EndsSolid(prev[cid])) {
                states[cid] = solid ? CellRising : CellFalling;
                HadTransitions = true;
            } else {
                states[cid] = solid ? CellSolid : CellAir;
            }
        });
        CellState.Flip();
    } else if (HadTransitions) {
        // Geometry unchanged: last batch's transitioning cells have converged
        const auto *prev = static_cast<const uint8_t *>(CellState.Cur().RawData());
        auto *states = static_cast<uint8_t *>(CellState.Other().RawData());
        std::memset(states, CellAir, GridSize);
        ForEachCell(SolidBounds(false), Params.Nx, Params.Ny, [&](int cid, int, int, int) {
            if (EndsSolid(prev[cid])) states[cid] = CellSolid;
        });
        CellState.Flip();
        HadTransitions = false;
    }
    // 3. Compute shader values at all sampled positions and times.
    // Overlaps with the previous batch's FDTD steps on the GPU: shader kernels encode
    // into this batch's prologue command buffer (in-order queue), and CPU writes go to
    // this batch's double-buffer slots.
    // (for now, assumes point sources are specified at the end of config file)
    {
        const profile::Scope scope{"cpu/shader_compute"};
        int global_bid = 0;
        for (auto &object : Objects) {
            Compute(object, ShaderData.Cur(), global_bid);
            if (Base(object).HasShader) global_bid += Base(object).NPoints;
        }
    }

    // 4. Additional "per-batch overhead"
    {
        const profile::Scope scope{"encode/fresh_cell_pressure"};
        FreshCellPressure();
    }
    // The prologue command buffer is complete: commit it now so the GPU picks it up the
    // moment the previous batch's FDTD steps drain, while the CPU keeps preparing.
    // ACOUSTIC_LAZY_COMMIT=1 defers both commits to the sync point, isolating the commit
    // policy for A/B on scenes where prologue GPU work and AMX-saturated CPU prep share
    // memory bandwidth.
    static const bool lazy_commit = std::getenv("ACOUSTIC_LAZY_COMMIT") != nullptr;
    auto &ctx = MetalContext::Get();
    if (!lazy_commit) ctx.Flush();

    // Factorize the fresh-cell systems (geometry-only) while the GPU still runs
    {
        const profile::Scope scope{"cpu/fresh_cell_analyze"};
        AnalyzeFreshCells();
    }

    // Per batch, not per rasterization: probing at coarser granularity lets the two paths run
    // in blocks at different GPU temperatures, which makes the comparison meaningless.
    // SetupShaders' face tables stay valid until the shader points change, so either path
    // can run on any batch.
    FoldApply = PathTuner.Choose(NShaderPoints > 0, NRegularShaderPoints != NShaderPoints, NShaderPoints);

    // 5. Encode the batch's FDTD update (shader re-init + all timesteps) into a deferred
    // command buffer, committed now but gated on a shared event signaled after the host
    // fresh-cell writes below — the GPU has it scheduled and restarts the moment they land.
    ctx.BeginDeferred();
    ShaderReInit();
    RunFdtd();
    ctx.EndDeferred();
    if (!lazy_commit) ctx.CommitDeferred();

    // A batch with no fresh cells has nothing for the host to write into the FDTD buffer's
    // inputs, so its gate opens now rather than after the sync. That takes the whole host
    // round trip — wake, solve, and the signal-to-start latency — off the GPU's critical
    // path: it picks the batch up straight off the prologue. The batch still runs through
    // the sync below, which is why Sync leaves released work alone. Half the batches of a
    // typical animated scene qualify (Trumpet 49%, TalkFan 50%, FillerUp all of them).
    const bool early_release = NFreshCells == 0 && !lazy_commit;
    if (early_release) ctx.SignalDeferred();

    // The one sync point per batch: the fresh-cell velocity solve reads the previous
    // batch's final velocities (and this batch's prologue kernels, which clear interior
    // velocities first) on the host.
    ctx.Sync();
    PathTuner.RecordDrained(ctx.TakeBatchGpuSeconds());
    if (!early_release) {
        // Host's share of the window the GPU waits out; the rest of gpu/idle_fdtd is the
        // host's wake latency plus the signal-to-start latency.
        const profile::Scope window{"cpu/sync_window"};
        {
            const profile::Scope scope{"cpu/fresh_cell_velocity"};
            SolveFreshCells();
        }
        if (lazy_commit) ctx.CommitDeferred(); // prologue was committed by the sync's flush
        ctx.SignalDeferred();
    }
    // Progress reporting, after the gate is open: it is per batch, so keeping it out of the
    // window the GPU waits on matters more than its cost. '\n' rather than std::endl for
    // the same reason — a flush per batch is not free at thousands of batches.
    std::cout << std::format("  # Fresh Cells = {}\nresidual: {}\n", NFreshCells, FreshResidual);

    // The released batch fills the other listener slot, so the drained batch's samples
    // can be written to file with the GPU already running.
    WritePendingListeners();
    ListenerPending = true;
    PendingListenerSlot = &ListenerOut.Cur();
    if (NShaderPoints > 0) PathTuner.MarkInFlight(FoldApply); // measure this batch at the next sync

    return true;
}

// Cell bounds of a vertex set's rasterization, using the raster paths' cell-index
// conversion, padded by `pad` cells and clamped to the grid.
CellBox WaveBlender::VertexBounds(const Eigen::MatrixX<real> &v, const Eigen::Vector3<real> &offset, int pad) const {
    CellBox box;
    if (v.rows() == 0) return box;
    Eigen::Vector3<real> min = v.colwise().minCoeff();
    min += offset;
    Eigen::Vector3<real> max = v.colwise().maxCoeff();
    max += offset;
    const int n[3]{Params.Nx, Params.Ny, Params.Nz};
    for (int d = 0; d < 3; ++d) {
        box.Min[d] = int(min[d] / Params.Dx + n[d] / 2.) - pad;
        box.Max[d] = int(max[d] / Params.Dx + n[d] / 2.) + pad;
    }
    box.Clamp(Params.Nx, Params.Ny, Params.Nz);
    return box;
}

// Union of every cell that can be solid in Cell2 (and, when `with_prev`, in Cell1)
CellBox WaveBlender::SolidBounds(bool with_prev) const {
    CellBox box = CavityBounds;
    for (const auto &b : ObjectBounds) box.Expand(b);
    if (with_prev) {
        box.Expand(PrevCavityBounds);
        for (const auto &b : PrevObjectBounds) box.Expand(b);
    }
    return box;
}

// Conservative CPU rasterizer based on triangle-box overlap test. Parallel over each
// object's triangles — safe because all of an object's threads write the same value.
void WaveBlender::Rasterize(const bool (&keep_value)[256]) {
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        const auto &base = Base(Objects[oid]);
        if (!base.Changed) continue;

        const auto &offset = Offsets[oid];
        const auto &v = base.V2;
        const auto &f = base.F;
        const uint8_t fill = oid + 1;
        ObjectBounds[oid] = VertexBounds(v, offset, base.Type == ShaderClass::Point ? 1 : 0);

        // Point source rasterization special handling
        if (base.Type == ShaderClass::Point) {
            for (int r = 0; r < v.rows(); ++r) {
                Eigen::Vector3<real> pt = v.row(r);
                pt += offset;

                const int i = int(pt[0] / Params.Dx + Params.Nx / 2.);
                const int j = int(pt[1] / Params.Dx + Params.Ny / 2.);
                const int k = int(pt[2] / Params.Dx + Params.Nz / 2.);

                if (const int cid = Cid(i, j, k); Cell2[cid] == 0) Cell2[cid] = fill;

                // Neighboring cells needed for point-to-grid
                const int neighbor_cids[6]{
                    Cid(i - 1, j, k), Cid(i + 1, j, k), // left, right
                    Cid(i, j - 1, k), Cid(i, j + 1, k), // down, up
                    Cid(i, j, k - 1), Cid(i, j, k + 1) // back, front
                };
                for (int n = 0; n < 6; n += 2) {
                    if (Cell2[neighbor_cids[n]] == 0) Cell2[neighbor_cids[n]] = fill;
                }
                // (we brute-force consider all 6 neighbors for good measure, though only 3 are needed)
            }
            continue;
        }
        // Density rasterization special handling
        if (base.Type == ShaderClass::Density) {
            for (int r = 0; r < v.rows(); ++r) {
                Eigen::Vector3<real> pt = v.row(r);
                pt += offset;

                const int i = int(pt[0] / Params.Dx + Params.Nx / 2.);
                const int j = int(pt[1] / Params.Dx + Params.Ny / 2.);
                const int k = int(pt[2] / Params.Dx + Params.Nz / 2.);

                const int cid = Cid(i, j, k);
                const int density = f(r, 0); // HACK to encode density in object's faces

                constexpr int Threshold{97};
                if (density >= Threshold && !keep_value[Cell2[cid]]) Cell2[cid] = fill;
            }
            continue;
        }
        // General triangle mesh rasterization
        ParallelFor(f.rows(), 32, [&](size_t r) {
            const Eigen::Vector3i face = f.row(r);
            Eigen::Matrix3<real> tri_v;
            tri_v.row(0) = v.row(face[0]);
            tri_v.row(1) = v.row(face[1]);
            tri_v.row(2) = v.row(face[2]);

            Eigen::Vector3<real> min = tri_v.colwise().minCoeff();
            min += offset;
            Eigen::Vector3<real> max = tri_v.colwise().maxCoeff();
            max += offset;

            const int min_i = std::max(int(min[0] / Params.Dx + Params.Nx / 2.), 0);
            const int max_i = std::min(int(max[0] / Params.Dx + Params.Nx / 2.), Params.Nx - 1);

            const int min_j = std::max(int(min[1] / Params.Dx + Params.Ny / 2.), 0);
            const int max_j = std::min(int(max[1] / Params.Dx + Params.Ny / 2.), Params.Ny - 1);

            const int min_k = std::max(int(min[2] / Params.Dx + Params.Nz / 2.), 0);
            const int max_k = std::min(int(max[2] / Params.Dx + Params.Nz / 2.), Params.Nz - 1);

            real boxhalfsize[3]{Params.Dx / 2.f, Params.Dx / 2.f, Params.Dx / 2.f};
            real triverts[3][3]{
                {tri_v(0, 0), tri_v(0, 1), tri_v(0, 2)},
                {tri_v(1, 0), tri_v(1, 1), tri_v(1, 2)},
                {tri_v(2, 0), tri_v(2, 1), tri_v(2, 2)},
            };
            for (int i = min_i; i <= max_i; ++i) {
                for (int j = min_j; j <= max_j; ++j) {
                    for (int k = min_k; k <= max_k; ++k) {
                        const Eigen::Vector3<real> p = Pos(i, j, k) - offset;
                        real boxcenter[3]{p[0], p[1], p[2]};
                        if (!triBoxOverlap(boxcenter, boxhalfsize, triverts)) continue;

                        const std::atomic_ref<uint8_t> cell{Cell2[Cid(i, j, k)]};
                        if (!keep_value[cell.load(std::memory_order_relaxed)]) cell.store(fill, std::memory_order_relaxed);
                    }
                }
            }
        });
    }
}

// Section 6.2.3 from [Xue et al. 2024]
void WaveBlender::DetectCavities() {
    // Determine bounding box
    int min_i = Params.Nx - 1, max_i = 0;
    int min_j = Params.Ny - 1, max_j = 0;
    int min_k = Params.Nz - 1, max_k = 0;
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        const auto &base = Base(Objects[oid]);
        if (base.Type == ShaderClass::Point) continue;

        const auto &v = base.V2;
        const auto &offset = Offsets[oid];

        Eigen::Vector3<real> min = v.colwise().minCoeff();
        min += offset;
        Eigen::Vector3<real> max = v.colwise().maxCoeff();
        max += offset;

        min_i = std::min(min_i, int(min[0] / Params.Dx + Params.Nx / 2.));
        max_i = std::max(max_i, int(max[0] / Params.Dx + Params.Nx / 2.));

        min_j = std::min(min_j, int(min[1] / Params.Dx + Params.Ny / 2.));
        max_j = std::max(max_j, int(max[1] / Params.Dx + Params.Ny / 2.));

        min_k = std::min(min_k, int(min[2] / Params.Dx + Params.Nz / 2.));
        max_k = std::max(max_k, int(max[2] / Params.Dx + Params.Nz / 2.));
    }
    min_i -= 1;
    min_j -= 1;
    min_k -= 1;
    max_i += 1;
    max_j += 1;
    max_k += 1;

    // passable_value[v]: a cell with value v does not block the flood fill
    bool passable_value[256]{};
    passable_value[0] = true;
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        if (Base(Objects[oid]).Type == ShaderClass::Point) passable_value[oid + 1] = true;
    }

    const auto in_box = [&](int cid) {
        const int i = cid % Params.Nx;
        const int j = (cid / Params.Nx) % Params.Ny;
        const int k = (cid / Params.Nx) / Params.Ny;
        // Enforce that the PML contains no objects to save time on flood fill
        return i >= min_i && i <= max_i && j >= min_j && j <= max_j && k >= min_k && k <= max_k;
    };

    // Flood fill to detect connected components. Visited marks from the previous fill
    // all lie within its bounds, so only that region needs re-clearing (row-wise).
    if (!CavityBounds.Empty()) {
        const size_t row_bytes = CavityBounds.Max[0] - CavityBounds.Min[0] + 1;
        for (int k = CavityBounds.Min[2]; k <= CavityBounds.Max[2]; ++k) {
            for (int j = CavityBounds.Min[1]; j <= CavityBounds.Max[1]; ++j) {
                std::memset(&FloodVisited[Cid(CavityBounds.Min[0], j, k)], 0, row_bytes);
            }
        }
    }
    CavityBounds = CellBox{{min_i, min_j, min_k}, {max_i, max_j, max_k}};
    CavityBounds.Clamp(Params.Nx, Params.Ny, Params.Nz);
    FloodStack.clear();
    if (const int seed = Cid(max_i, max_j, max_k); passable_value[Cell2[seed]]) {
        FloodVisited[seed] = 1;
        FloodStack.push_back(seed);
    }
    const int neighbor_offsets[6]{-1, 1, -Params.Nx, Params.Nx, -Params.Nx * Params.Ny, Params.Nx * Params.Ny};
    while (!FloodStack.empty()) {
        const int cid = FloodStack.back();
        FloodStack.pop_back();

        for (const int offset : neighbor_offsets) {
            const int ncid = cid + offset;
            if (!in_box(ncid) || FloodVisited[ncid] || !passable_value[Cell2[ncid]]) continue;
            FloodVisited[ncid] = 1;
            FloodStack.push_back(ncid);
        }
    }
    // Fill in cavities. The test is per cell and independent of its neighbors, so the
    // sweep runs in ascending cell order — x innermost, contiguous — rather than striding
    // Nx*Ny cells per step.
    ForEachCell(CavityBounds, Params.Nx, Params.Ny, [&](int cid, int, int, int) {
        if (!FloodVisited[cid] && passable_value[Cell2[cid]]) Cell2[cid] = CavityInterior;
    });
}

// Section 6.1 from [Xue et al. 2024]
void WaveBlender::SetupShaders() {
    ShaderMapHost.clear();

    // Boundary faces of regular shader objects
    FaceEpoch += 1;
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        auto &base = Base(Objects[oid]);
        if (!base.HasShader || base.Type == ShaderClass::Point) continue;

        const uint8_t fill = oid + 1;
        std::vector<Eigen::Vector3<real>> b_vec;
        std::vector<Eigen::Vector3<real>> bn_vec;
        std::vector<int> key_vec;
        int bid = 0;
        CellBox scan = ObjectBounds[oid]; // the object's cells at either batch endpoint
        scan.Expand(PrevObjectBounds[oid]);
        ForEachCell(scan, Params.Nx, Params.Ny, [&](int cid, int i, int j, int k) {
            if (Cell1[cid] != fill && Cell2[cid] != fill) return;

            if (i == 0 || i >= Params.Nx - 1 || j == 0 || j >= Params.Ny - 1 || k == 0 || k >= Params.Nz - 1) return;

            const Eigen::Vector3<real> p = Pos(i, j, k) - Offsets[oid]; // world-space to object-space

            const int neighbor_cids[6]{
                Cid(i - 1, j, k), Cid(i + 1, j, k), // left, right
                Cid(i, j - 1, k), Cid(i, j + 1, k), // down, up
                Cid(i, j, k - 1), Cid(i, j, k + 1) // back, front
            };
            for (int n = 0; n < 6; ++n) {
                if ((Cell1[cid] == fill && Cell1[neighbor_cids[n]] == 0) ||
                    (Cell2[cid] == fill && Cell2[neighbor_cids[n]] == 0)) {
                    const int d = (n % 2 == 0);
                    const real sign = (n % 2 == 0) ? -1. : 1.;
                    const int dir = n / 2;
                    const int face_cid = (dir == 0) ? Cid(i - d, j, k) : (dir == 1) ? Cid(i, j - d, k) :
                                                                                      Cid(i, j, k - d);

                    if (FaceStamp[dir][face_cid] == FaceEpoch) continue; // face already claimed
                    FaceStamp[dir][face_cid] = FaceEpoch;

                    ShaderMapHost.push_back(3 * face_cid + dir);
                    key_vec.push_back(3 * face_cid + dir);
                    bn_vec.emplace_back(sign * Eigen::Vector3<real>{dir == 0 ? 1.f : 0.f, dir == 1 ? 1.f : 0.f, dir == 2 ? 1.f : 0.f});
                    b_vec.emplace_back(p.cast<real>() + (0.5 * Params.Dx) * bn_vec[bid]);
                    bid += 1;
                }
            }
        });

        Eigen::MatrixX<real> b(b_vec.size(), 3);
        Eigen::MatrixX<real> bn(bn_vec.size(), 3);
        for (int row = 0; row < int(b_vec.size()); ++row) {
            b.row(row) = b_vec[row];
            bn.row(row) = bn_vec[row];
        }
        base.SetSamplePoints(b, bn, std::move(key_vec));
    }

    NRegularShaderPoints = ShaderMapHost.size();

    // Boundary faces of point sources (all 6 faces of each point cell — the cells
    // revert to air since we don't want to rasterize point sources)
    FaceEpoch += 1;
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        auto &base = Base(Objects[oid]);
        if (base.Type != ShaderClass::Point) continue;

        const uint8_t fill = oid + 1;
        std::vector<Eigen::Vector3<real>> b_vec;
        std::vector<Eigen::Vector3<real>> bn_vec;
        std::vector<int> key_vec;
        int bid = 0;
        ForEachCell(ObjectBounds[oid], Params.Nx, Params.Ny, [&](int cid, int i, int j, int k) {
            if (Cell2[cid] != fill) return;
            Cell2[cid] = 0;

            if (i == 0 || i >= Params.Nx - 1 || j == 0 || j >= Params.Ny - 1 || k == 0 || k >= Params.Nz - 1) return;

            const Eigen::Vector3<real> p = Pos(i, j, k) - Offsets[oid]; // world-space to object-space

            for (int n = 0; n < 6; ++n) {
                const int d = (n % 2 == 0);
                const real sign = (n % 2 == 0) ? -1. : 1.;
                const int dir = n / 2;
                const int face_cid = (dir == 0) ? Cid(i - d, j, k) : (dir == 1) ? Cid(i, j - d, k) :
                                                                                  Cid(i, j, k - d);

                if (FaceStamp[dir][face_cid] == FaceEpoch) continue; // face already claimed
                FaceStamp[dir][face_cid] = FaceEpoch;

                ShaderMapHost.push_back(-3 * face_cid - dir);
                key_vec.push_back(-3 * face_cid - dir);
                bn_vec.emplace_back(sign * Eigen::Vector3<real>{dir == 0 ? 1.f : 0.f, dir == 1 ? 1.f : 0.f, dir == 2 ? 1.f : 0.f});
                b_vec.emplace_back(p.cast<real>() + (0.5 * Params.Dx) * bn_vec[bid]);
                bid += 1;
            }
        });

        Eigen::MatrixX<real> b(b_vec.size(), 3);
        Eigen::MatrixX<real> bn(bn_vec.size(), 3);
        for (int row = 0; row < int(b_vec.size()); ++row) {
            b.row(row) = b_vec[row];
            bn.row(row) = bn_vec[row];
        }
        base.SetSamplePoints(b, bn, std::move(key_vec));
    }

    // Upload shader map into this batch's double-buffer slots
    NShaderPoints = ShaderMapHost.size();
    MaxNShaderPoints = std::max(MaxNShaderPoints, NShaderPoints);

    ShaderMap.Flip();
    ShaderMap.Cur().Resize(size_t(MaxNShaderPoints) * sizeof(int));
    ShaderMap.Cur().Upload(ShaderMapHost.data(), NShaderPoints * sizeof(int));

    ShaderData.Flip();
    ShaderData.Cur().Resize(size_t(MaxNShaderPoints) * NShaderSamples * sizeof(real));
    ShaderData.Cur().Zero(size_t(NShaderPoints) * NShaderSamples * sizeof(real));

    if (NShaderPoints == 0) return;

    // Per-cell face masks, packed row ids, and per-tile flags (layout in KernelParams.h).
    // Slot and row ids are read only where a mask bit is set, so only mask and tiles clear.
    const size_t slot_base = ShaderFacesSlotOffset(GridSize);
    const size_t tiles_base = ShaderFacesTilesOffset(GridSize);
    const int ntx = FdtdTiles.x, nty = FdtdTiles.y, ntz = FdtdTiles.z;
    const size_t n_tiles = size_t(ntx) * nty * ntz;
    const size_t bids_base = ShaderFacesBidsOffset(GridSize, n_tiles);

    ShaderFaces.Flip();
    ShaderFaces.Cur().Resize(bids_base + size_t(MaxNShaderPoints) * sizeof(int));
    auto *mask = static_cast<uint8_t *>(ShaderFaces.Cur().RawData());
    auto *slot = reinterpret_cast<int *>(mask + slot_base);
    auto *tiles = mask + tiles_base;
    auto *bids = reinterpret_cast<int *>(mask + bids_base);
    std::memset(mask, 0, slot_base);
    std::memset(tiles, 0, n_tiles);

    // Pass 1: mask bits, tile flags, and the face cells in first-touch order. Only the owning
    // tile needs flagging — a thread reads no mask but its own.
    FaceCells.clear();
    for (const int entry : ShaderMapHost) {
        const int face = entry < 0 ? -entry : entry;
        const int dir = face % 3, cid = face / 3;
        if (mask[cid] == 0) {
            FaceCells.push_back(cid);
            const int i = cid % Params.Nx, j = (cid / Params.Nx) % Params.Ny, k = (cid / Params.Nx) / Params.Ny;
            tiles[FdtdTileIndex(i / int(FdtdTg.x), j / int(FdtdTg.y), k / int(FdtdTg.z), ntx, nty)] = 1;
        }
        mask[cid] |= 1u << (dir + (entry < 0 ? 3 : 0)); // Blend planes 0-2, force planes 3-5
    }
    // Pass 2: pack each face cell's row ids contiguously, indexed by mask popcount
    int next_slot = 0;
    for (const int cid : FaceCells) {
        slot[cid] = next_slot;
        next_slot += std::popcount(mask[cid]);
    }
    for (int bid = 0; bid < NShaderPoints; ++bid) {
        const int entry = ShaderMapHost[bid];
        const int face = entry < 0 ? -entry : entry;
        const int dir = face % 3, cid = face / 3;
        const int plane = dir + (entry < 0 ? 3 : 0);
        bids[slot[cid] + std::popcount(uint8_t(mask[cid] & ((1u << plane) - 1u)))] = bid;
    }

    if (profile::Enabled()) {
        Faces.PeakPoints = std::max(Faces.PeakPoints, NShaderPoints);
        Faces.Batches += 1;
        Faces.TotalTiles += n_tiles;
        Faces.FlaggedTiles += std::count(tiles, tiles + n_tiles, uint8_t{1});
    }
}

bool WaveBlender::ApplyPathTuner::Choose(bool has_points, bool has_forces, int npoints) {
    constexpr int ProbesPerPath{4};

    // ACOUSTIC_APPLY_PATH=plain|folded pins the path, for whole-scene A/B of the two.
    static const char *const forced = std::getenv("ACOUSTIC_APPLY_PATH");
    if (forced) return has_points && std::string_view{forced} == "folded";

    SeenForces |= has_forces;
    const uint32_t key = uint32_t(has_points) | (SeenForces ? 2u : 0u);
    // Re-open probing when the boundary-application load has drifted far from where the
    // lock was measured (impulse counts can grow through a scene, shifting the balance).
    const bool points_drifted = Locked >= 0 && (npoints > 2 * LockedPoints || 2 * npoints < LockedPoints);
    if (key != Key || points_drifted) {
        Key = key;
        Locked = -1;
        BestSeconds[0] = BestSeconds[1] = std::numeric_limits<double>::max();
        ProbeCount[0] = ProbeCount[1] = 0;
        InFlightFold = -1;
    }
    if (!has_points) return false;
    if (Locked >= 0) return Locked == 1;

    if (ProbeCount[0] >= ProbesPerPath && ProbeCount[1] >= ProbesPerPath) {
        Locked = BestSeconds[1] < BestSeconds[0] ? 1 : 0;
        LockedPoints = npoints;
        if (profile::Enabled()) {
            std::cout << std::format("[apply-path] locked {} (plain {:.2f} ms, folded {:.2f} ms per batch)\n", Locked == 1 ? "folded" : "plain", BestSeconds[0] * 1e3, BestSeconds[1] * 1e3);
        }
        return Locked == 1;
    }
    return ProbeCount[1] < ProbeCount[0];
}

void WaveBlender::ApplyPathTuner::MarkInFlight(bool fold) {
    if (Locked < 0) InFlightFold = fold;
}

void WaveBlender::ApplyPathTuner::RecordDrained(double gpu_seconds) {
    if (InFlightFold < 0) return;
    BestSeconds[InFlightFold] = std::min(BestSeconds[InFlightFold], gpu_seconds);
    ProbeCount[InFlightFold] += 1;
    InFlightFold = -1;
}

void WaveBlender::FaceLoad::Report() const {
    if (!profile::Enabled() || Batches == 0) return;
    std::cout << std::format("[shader-faces] peak {} points, {:.1f}% of tiles flagged (mean {:.0f} of {:.0f})\n", PeakPoints, 100. * double(FlaggedTiles) / double(TotalTiles), double(FlaggedTiles) / double(Batches), double(TotalTiles) / double(Batches));
}

void WaveBlender::FreshCellPressure() {
    auto &ctx = MetalContext::Get();

    const Dim3 fdtd_threads{8, 8, 8};
    const Dim3 fdtd_blocks{uint32_t(Params.Nx + 7 - PmlWidth) / 8, uint32_t(Params.Ny + 7 - PmlWidth) / 8, uint32_t(Params.Nz + 7 - PmlWidth) / 8};

    const Dim3 shader_threads{32, 1, 1};
    const Dim3 shader_blocks{uint32_t(NShaderPoints + 31) / 32, 1, 1};

    const real incr = real(Params.ShaderSrate) / Params.FdtdSrate; // For now, assumes shader rate is same for all objects

    // Px/Py/Pz double as the fresh-cell acceleration buffers (the step kernels keep their
    // split pressure in the Psi buffers) -- just make sure to clear afterwards
    const PrepareFreshCellParams prep_params{NShaderSamples, NShaderPoints, float(1. / Params.Dt), incr};
    ctx.Dispatch("PrepareFreshCell", shader_blocks, shader_threads, {&Px, &Py, &Pz, &ShaderData.Cur(), &ShaderMap.Cur()}, &prep_params, sizeof(prep_params));

    const FreshCellPressureParams fc_params{Params.Nx, Params.Ny, Params.Nz, Params.Rho * Params.Dx};
    ctx.Dispatch("FreshCellPressure", fdtd_blocks, fdtd_threads, {&P.Cur(), &Px, &Py, &Pz, &CellState.Cur()}, &fc_params, sizeof(fc_params));

    const ClearSolidParams cs_params{Params.Nx, Params.Ny, Params.Nz};
    ctx.Dispatch("ClearSolid", fdtd_blocks, fdtd_threads, {&Vx.Cur(), &Vy.Cur(), &Vz.Cur(), &Px, &Py, &Pz, &CellState.Cur()}, &cs_params, sizeof(cs_params));
}

void WaveBlender::AnalyzeFreshCells() {
    FreshSolves.clear();
    FreshResidual = 0.;
    // Host pointers for the post-sync solve, captured before the FDTD encode's ping-pong
    // flips. ClearSolid (already encoded) writes these same slots on the GPU timeline.
    FreshV[0] = static_cast<real *>(Vx.Cur().RawData());
    FreshV[1] = static_cast<real *>(Vy.Cur().RawData());
    FreshV[2] = static_cast<real *>(Vz.Cur().RawData());

    // point_value[v]: cell value v belongs to a point source
    bool point_value[256]{};
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        if (Base(Objects[oid]).Type == ShaderClass::Point) point_value[oid + 1] = true;
    }

    // Determine fresh cells (based on Cell1 and Cell2 difference), in ascending cell
    // order. Nonzero Cell1 values all lie within the previous rasterization's bounds.
    std::vector<int> fresh_cids;
    CellBox prev_box = PrevCavityBounds;
    for (const auto &b : PrevObjectBounds) prev_box.Expand(b);
    ForEachCell(prev_box, Params.Nx, Params.Ny, [&](int cid, int, int, int) {
        if (Cell2[cid] == 0 && Cell1[cid] > 0 && Cell1[cid] != CavityInterior && !point_value[Cell1[cid]]) fresh_cids.push_back(cid);
    });
    NFreshCells = fresh_cids.size();
    if (fresh_cids.empty()) return;

    // Label connected components of fresh cells (6-adjacency). The least-squares system
    // is block-diagonal across components, so per-component solves yield the same
    // minimum-norm solution up to float rounding, at far lower cost and in parallel.
    const int neighbor_offsets[6]{-1, 1, -Params.Nx, Params.Nx, -Params.Nx * Params.Ny, Params.Nx * Params.Ny};
    for (const int cid : fresh_cids) FreshComponent[cid] = -2; // fresh, unlabeled

    int n_components = 0;
    std::vector<int> bfs;
    for (const int cid : fresh_cids) {
        if (FreshComponent[cid] != -2) continue;
        const int component = n_components++;
        FreshComponent[cid] = component;
        bfs.assign(1, cid);
        while (!bfs.empty()) {
            const int cur = bfs.back();
            bfs.pop_back();
            for (const int offset : neighbor_offsets) {
                if (const int ncid = cur + offset; FreshComponent[ncid] == -2) {
                    FreshComponent[ncid] = component;
                    bfs.push_back(ncid);
                }
            }
        }
    }
    // Gather each component's cells in ascending cell order (row order affects rounding).
    // ACOUSTIC_GLOBAL_LSTSQ=1 solves one global system instead, for exactness validation.
    static const bool global_lstsq = std::getenv("ACOUSTIC_GLOBAL_LSTSQ") != nullptr;
    if (global_lstsq) n_components = 1;
    FreshSolves.resize(n_components);
    if (n_components == 1) FreshSolves[0].Cells = std::move(fresh_cids);
    else
        for (const int cid : fresh_cids) FreshSolves[FreshComponent[cid]].Cells.push_back(cid);
    for (const auto &solve : FreshSolves)
        for (const int cid : solve.Cells) FreshComponent[cid] = -1; // reset scratch

    ParallelFor(n_components, 1, [&](size_t c) {
        auto &solve = FreshSolves[c];
        const auto &cells = solve.Cells;

        // Pass 1: assign a column to each interior face (a face whose Cell1 neighbor is
        // nonzero), in row-traversal order, grouped x then y then z.
        int (&counts)[3] = solve.Counts;
        std::vector<int>(&faces)[3] = solve.Faces;
        for (const int cid : cells) {
            const int i = cid % Params.Nx;
            const int j = (cid / Params.Nx) % Params.Ny;
            const int k = (cid / Params.Nx) / Params.Ny;
            const int neighbor_cids[6]{
                Cid(i - 1, j, k), Cid(i + 1, j, k), // left, right
                Cid(i, j - 1, k), Cid(i, j + 1, k), // down, up
                Cid(i, j, k - 1), Cid(i, j, k + 1) // back, front
            };
            for (int n = 0; n < 6; ++n) {
                if (Cell1[neighbor_cids[n]] == 0) continue;

                const int d = (n % 2 == 0);
                const int dir = n / 2;
                const int face_cid = (dir == 0) ? Cid(i - d, j, k) : (dir == 1) ? Cid(i, j - d, k) :
                                                                                  Cid(i, j, k - d);
                if (FaceCol[dir][face_cid] == -1) {
                    FaceCol[dir][face_cid] = counts[dir]++;
                    faces[dir].push_back(face_cid);
                }
            }
        }

        // Pass 2: build the least-squares system enforcing zero net flux per fresh cell,
        // and factorize it. The right-hand side needs the drained velocities, so it waits
        // for the post-sync solve.
        const int n_dof = counts[0] + counts[1] + counts[2];
        solve.A = Eigen::MatrixX<real>::Zero(cells.size(), n_dof);

        int row = 0;
        for (const int cid : cells) {
            const int i = cid % Params.Nx;
            const int j = (cid / Params.Nx) % Params.Ny;
            const int k = (cid / Params.Nx) / Params.Ny;

            for (int n = 0; n < 6; ++n) {
                const int d = (n % 2 == 0);
                const real sign = (n % 2 == 0) ? -1. : 1.;
                const int dir = n / 2;
                const int face_cid = (dir == 0) ? Cid(i - d, j, k) : (dir == 1) ? Cid(i, j - d, k) :
                                                                                  Cid(i, j, k - d);

                int col = FaceCol[dir][face_cid];
                if (col != -1) col += (dir >= 1 ? counts[0] : 0) + (dir >= 2 ? counts[1] : 0);
                if (col != -1) solve.A(row, col) = sign;
            }
            row += 1;
        }
        if (n_dof > 0) solve.Cod.compute(solve.A); // least squares
    });
}

void WaveBlender::SolveFreshCells() {
    if (FreshSolves.empty()) return;

    std::vector<double> residuals(FreshSolves.size(), 0.);
    ParallelFor(FreshSolves.size(), 1, [&](size_t c) {
        auto &solve = FreshSolves[c];
        const auto &cells = solve.Cells;
        const int (&counts)[3] = solve.Counts;
        const std::vector<int>(&faces)[3] = solve.Faces;
        const int n_dof = counts[0] + counts[1] + counts[2];

        // Right-hand side: boundary-face flux from the drained velocities, accumulated in
        // the same cell-and-face order as the one-pass build.
        Eigen::VectorX<real> g = Eigen::VectorX<real>::Zero(cells.size());
        int row = 0;
        for (const int cid : cells) {
            const int i = cid % Params.Nx;
            const int j = (cid / Params.Nx) % Params.Ny;
            const int k = (cid / Params.Nx) / Params.Ny;

            for (int n = 0; n < 6; ++n) {
                const int d = (n % 2 == 0);
                const real sign = (n % 2 == 0) ? -1. : 1.;
                const int dir = n / 2;
                const int face_cid = (dir == 0) ? Cid(i - d, j, k) : (dir == 1) ? Cid(i, j - d, k) :
                                                                                  Cid(i, j, k - d);
                if (FaceCol[dir][face_cid] == -1) g[row] -= sign * FreshV[dir][face_cid];
            }
            row += 1;
        }

        if (n_dof > 0) {
            const Eigen::VectorX<real> u = solve.Cod.solve(g);

            // Write fresh cell velocity solve results back to the shared buffers
            for (const int face_cid : faces[0]) FreshV[0][face_cid] = u[FaceCol[0][face_cid]];
            for (const int face_cid : faces[1]) FreshV[1][face_cid] = u[counts[0] + FaceCol[1][face_cid]];
            for (const int face_cid : faces[2]) FreshV[2][face_cid] = u[counts[0] + counts[1] + FaceCol[2][face_cid]];

            residuals[c] = (solve.A * u - g).squaredNorm();
        }
        // Reset column scratch (each face belongs to exactly one component)
        for (int dir = 0; dir < 3; ++dir) {
            for (const int face_cid : faces[dir]) FaceCol[dir][face_cid] = -1;
        }
    });

    double residual_sq = 0.;
    for (const double r : residuals) residual_sq += r;
    FreshResidual = std::sqrt(residual_sq);
}

// Section 6.2.2 from [Xue et al. 2024]
void WaveBlender::ShaderReInit() {
    const Dim3 shader_threads{32, 1, 1};
    const Dim3 shader_blocks{uint32_t(NShaderPoints + 31) / 32, 1, 1};

    const ShaderReInitParams params{NShaderSamples, NShaderPoints};
    MetalContext::Get().Dispatch("ShaderReInit", shader_blocks, shader_threads, {&Vx.Cur(), &Vy.Cur(), &Vz.Cur(), &ShaderData.Cur(), &ShaderMap.Cur()}, &params, sizeof(params));
}

// Encodes all timesteps of the batch into the deferred command buffer with a persistent
// argument table. Each step's sequence is [pressure, boundary application, velocity]. With
// beta derived in-kernel, the loop is one full-grid pass per step (the fused
// StepVelocityPressure kernel), book-ended by a StepPressure prologue and a StepVelocity tail:
//   P(0) | apply(0) | [V(0),P(1)] | ... | [V(N-2),P(N-1)] | V(N-1)
// The dispatch computing V(q) gets step q's params (tb, ss, and the listener slot for p(q)),
// plus, on the folded path, step q+1's — so only step 0 needs its own apply dispatch there.
// On the plain path every step's application is a standalone dispatch.
void WaveBlender::RunFdtd() {
    const profile::Scope scope{"encode/fdtd"};
    auto &ctx = MetalContext::Get();

    const MTL::Size fdtd_threads{FdtdTg.x, FdtdTg.y, FdtdTg.z};
    const MTL::Size fdtd_blocks{FdtdTiles.x, FdtdTiles.y, FdtdTiles.z};

    auto *pressure_pso = ctx.Pipeline("StepPressure");
    auto *fused_pso = ctx.Pipeline("StepVelocityPressure", FoldApply);
    auto *velocity_pso = ctx.Pipeline("StepVelocity");
    auto *shader_pso = NShaderPoints > 0 ? ctx.Pipeline("ApplyShader") : nullptr;

    ListenerOut.Flip(); // this batch's samples fill the spare slot (see WritePendingListeners)

    auto *encoder = ctx.ActiveEncoder();
    const GpuBuffer *const table[]{
        nullptr, &PsiX, &PsiY, &PsiZ, nullptr, nullptr, nullptr, &ShaderFaces.Cur(), &CellState.Cur(), &PmlNp, &PmlDp, &PmlNv, &PmlDv,
        &ShaderData.Cur(), &ShaderMap.Cur(), &ListenerCids, &ListenerOut.Cur()
    };
    for (uint32_t index = 1; index < std::size(table); ++index) {
        if (table[index]) encoder->setBuffer(table[index]->Handle(), 0, index);
    }

    const FdtdBatchParams batch_params{RhoCcDt, InvDx, InvRhoDt, Params.Nx, Params.Ny, Params.Nz, NShaderSamples, int(Listeners.size()), int(FdtdTiles.x), int(FdtdTiles.y), int(FdtdTiles.z)};
    encoder->setBytes(&batch_params, sizeof(batch_params), FdtdBatchParamsIndex);

    const auto bind_fields = [&] { // in slots 0, 4-6; out slots 21-24
        encoder->setBuffer(P.Cur().Handle(), 0, 0);
        encoder->setBuffer(Vx.Cur().Handle(), 0, 4);
        encoder->setBuffer(Vy.Cur().Handle(), 0, 5);
        encoder->setBuffer(Vz.Cur().Handle(), 0, 6);
        encoder->setBuffer(P.Other().Handle(), 0, 21);
        encoder->setBuffer(Vx.Other().Handle(), 0, 22);
        encoder->setBuffer(Vy.Other().Handle(), 0, 23);
        encoder->setBuffer(Vz.Other().Handle(), 0, 24);
    };
    const auto step_params = [&](int q) {
        const real t = (q + 1 == NFdtdSamples) ? 1. : real(q + 1) / NFdtdSamples; // normalized blending time (0, 1]
        const real ss = t * (NShaderSamples - 1); // shader sample index (fractional to support interpolation)
        const real tb = (Params.Scheme == BlendScheme::NoBlend && q + 1 != NFdtdSamples) ? 0. : t;
        return FdtdStepParams{tb, ss, q};
    };

    // Blend range then force range, so a dual-claimed face updates in a defined order. The
    // folded path needs this only for step 0.
    const auto apply_dispatches = [&](int q) {
        if (!shader_pso) return;
        const FdtdStepParams params = step_params(q);
        encoder->setBytes(&params, sizeof(params), FdtdStepParamsIndex);
        encoder->setComputePipelineState(shader_pso);
        for (const ApplyShaderRange range : {ApplyShaderRange{0, NRegularShaderPoints}, ApplyShaderRange{NRegularShaderPoints, NShaderPoints}}) {
            if (range.Start == range.End) continue;
            encoder->setBytes(&range, sizeof(range), FdtdApplyRangeIndex);
            encoder->dispatchThreadgroups(MTL::Size{uint32_t(range.End - range.Start + 31) / 32, 1, 1}, MTL::Size{32, 1, 1});
        }
    };

    // Prologue: P(0) (in-place, before step 0's boundary application)
    bind_fields();
    encoder->setComputePipelineState(pressure_pso);
    encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);
    apply_dispatches(0);

    for (int q = 1; q < NFdtdSamples; ++q) {
        if (!FoldApply && q > 1) apply_dispatches(q - 1);
        const FdtdStepParams cur = step_params(q - 1);
        encoder->setBytes(&cur, sizeof(cur), FdtdStepParamsIndex);
        if (FoldApply) {
            const FdtdStepParams next = step_params(q);
            encoder->setBytes(&next, sizeof(next), FdtdNextStepParamsIndex);
        }
        encoder->setComputePipelineState(fused_pso);
        encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);
        P.Flip();
        Vx.Flip();
        Vy.Flip();
        Vz.Flip();
        bind_fields();
    }

    // Tail: V(N-1) (in-place) and the final listener sample p(N-1)
    if (!FoldApply) apply_dispatches(NFdtdSamples - 1);
    const FdtdStepParams last = step_params(NFdtdSamples - 1);
    encoder->setBytes(&last, sizeof(last), FdtdStepParamsIndex);
    encoder->setComputePipelineState(velocity_pso);
    encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);

    Step += NFdtdSamples;
}
