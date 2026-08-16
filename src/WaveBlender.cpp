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
#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>

namespace {
// Whether a cell state is solid at batch end (Rising converges to solid, Falling to air)
bool EndsSolid(uint8_t state) { return state == CellSolid || state == CellRising; }
} // namespace

WaveBlender::WaveBlender(const SimParams &params)
    : Params(params), GridSize(Params.Nx * Params.Ny * Params.Nz),
      NFdtdSamples(Params.FdtdSrate / Params.BlendRate), NShaderSamples(Params.ShaderSrate / Params.BlendRate + 1),
      RhoCcDt(Params.Rho * Params.C * Params.C * Params.Dt), InvDx(1. / Params.Dx), InvRhoDt(1. / Params.Rho * Params.Dt),
      Damping(Params.Damping) {
    for (auto *field : {&P, &Vx, &Vy, &Vz}) { // both ping-pong slots
        field->Cur().ResizeZeroed(GridSize * sizeof(real));
        field->Other().ResizeZeroed(GridSize * sizeof(real));
    }

    Cell1.resize(GridSize);
    Cell2.resize(GridSize);

    // Both slots all-Air: Cur() doubles as the previous batch-end solidity source
    CellState.Cur().ResizeZeroed(GridSize * sizeof(uint8_t));
    CellState.Other().ResizeZeroed(GridSize * sizeof(uint8_t));

    // Valid (unread) allocations for plain-path batches, which bind but never touch these
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
    ListenerOut.ResizeZeroed(std::max<size_t>(1, size_t(NFdtdSamples) * cids.size()) * sizeof(real));
}

void WaveBlender::WritePendingListeners() {
    if (!ListenerPending) return;
    ListenerPending = false;

    const auto *samples = ListenerOut.As<real>(); // completed at this batch's sync point
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
        MetalContext::Get().Sync(); // drain the final batch
        WritePendingListeners();
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

    // The one sync point per batch: the fresh-cell velocity solve reads the previous
    // batch's final velocities (and this batch's prologue kernels, which clear interior
    // velocities first) on the host.
    MetalContext::Get().Sync();
    PathTuner.RecordDrained(MetalContext::Get().TakeBatchGpuSeconds());
    WritePendingListeners();
    {
        const profile::Scope scope{"cpu/fresh_cell_velocity"};
        FreshCellVelocity();
    }

    ShaderReInit(); // Shader velocity re-initialization

    // 5. Run FDTD update (encoded and committed, not waited on)
    RunFdtd();

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
    // Fill in cavities
    for (int i = min_i; i <= max_i; ++i) {
        for (int j = min_j; j <= max_j; ++j) {
            for (int k = min_k; k <= max_k; ++k) {
                const int cid = Cid(i, j, k);
                if (!FloodVisited[cid] && passable_value[Cell2[cid]]) Cell2[cid] = CavityInterior;
            }
        }
    }
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

    // Boundary-application path for this batch (see KernelParams.h)
    FoldApply = PathTuner.Choose(NShaderPoints > 0, NRegularShaderPoints != NShaderPoints);
    if (!FoldApply) return;

    // Per-cell face masks, shader-data row ids, and per-tile flags for the folded
    // boundary application (layout in KernelParams.h). Row-id planes are read only where
    // the mask bit is set, so only the mask and tile flags need clearing.
    const size_t bids_base = ShaderFacesBidsOffset(size_t(GridSize));
    const size_t tiles_base = ShaderFacesTilesOffset(size_t(GridSize));
    const int ntx = (Params.Nx + FdtdTgX - 1) / FdtdTgX;
    const int nty = (Params.Ny + FdtdTgY - 1) / FdtdTgY;
    const int ntz = (Params.Nz + FdtdTgZ - 1) / FdtdTgZ;
    ShaderFaces.Flip();
    ShaderFaces.Cur().Resize(tiles_base + size_t(ntx) * nty * ntz);
    auto *mask = static_cast<uint8_t *>(ShaderFaces.Cur().RawData());
    std::memset(mask, 0, bids_base);
    auto *bids = reinterpret_cast<int *>(mask + bids_base);
    auto *tiles = mask + tiles_base;
    std::memset(tiles, 0, size_t(ntx) * nty * ntz);
    const auto mark_tile = [&](int i, int j, int k) { tiles[FdtdTileIndex(i / FdtdTgX, j / FdtdTgY, k / FdtdTgZ, ntx, nty)] = 1; };
    for (int bid = 0; bid < NShaderPoints; ++bid) {
        const int entry = ShaderMapHost[bid];
        const bool is_force = entry < 0;
        const int face = is_force ? -entry : entry;
        const int dir = face % 3, cid = face / 3;
        const int plane = dir + (is_force ? 3 : 0);
        mask[cid] |= 1u << plane;
        bids[plane * size_t(GridSize) + cid] = bid;

        // Flag every tile whose threads read this cell's mask: its own, and the upper
        // neighbors that read it as their lower halo (shader faces are never at the
        // outermost grid layer)
        const int i = cid % Params.Nx, j = (cid / Params.Nx) % Params.Ny, k = (cid / Params.Nx) / Params.Ny;
        mark_tile(i, j, k);
        mark_tile(i + 1, j, k);
        mark_tile(i, j + 1, k);
        mark_tile(i, j, k + 1);
    }
}

void WaveBlender::FreshCellPressure() {
    auto &ctx = MetalContext::Get();

    const Dim3 fdtd_threads{8, 8, 8};
    const Dim3 fdtd_blocks{uint32_t(Params.Nx + 7 - PmlWidth) / 8, uint32_t(Params.Ny + 7 - PmlWidth) / 8, uint32_t(Params.Nz + 7 - PmlWidth) / 8};

    const Dim3 shader_threads{32, 1, 1};
    const Dim3 shader_blocks{uint32_t(NShaderPoints + 31) / 32, 1, 1};

    const real incr = real(Params.ShaderSrate) / Params.FdtdSrate; // For now, assumes shader rate is same for all objects

    // We use split-pressure buffers as acceleration buffers for convenience (since split-pressure only used in PML)
    // -- just make sure to clear afterwards
    const PrepareFreshCellParams prep_params{NShaderSamples, NShaderPoints, float(1. / Params.Dt), incr};
    ctx.Dispatch("PrepareFreshCell", shader_blocks, shader_threads, {&Px, &Py, &Pz, &ShaderData.Cur(), &ShaderMap.Cur()}, &prep_params, sizeof(prep_params));

    const FreshCellPressureParams fc_params{Params.Nx, Params.Ny, Params.Nz, Params.Rho * Params.Dx};
    ctx.Dispatch("FreshCellPressure", fdtd_blocks, fdtd_threads, {&P.Cur(), &Px, &Py, &Pz, &CellState.Cur()}, &fc_params, sizeof(fc_params));

    const ClearSolidParams cs_params{Params.Nx, Params.Ny, Params.Nz};
    ctx.Dispatch("ClearSolid", fdtd_blocks, fdtd_threads, {&Vx.Cur(), &Vy.Cur(), &Vz.Cur(), &Px, &Py, &Pz, &CellState.Cur()}, &cs_params, sizeof(cs_params));
}

void WaveBlender::FreshCellVelocity() {
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
    std::cout << "  # Fresh Cells = " << fresh_cids.size() << std::endl;
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
    std::vector<std::vector<int>> components(n_components);
    if (n_components == 1) components[0] = fresh_cids;
    else
        for (const int cid : fresh_cids) components[FreshComponent[cid]].push_back(cid);
    for (const int cid : fresh_cids) FreshComponent[cid] = -1; // reset scratch

    const real *const v_host[3]{Vx.Cur().As<real>(), Vy.Cur().As<real>(), Vz.Cur().As<real>()};
    real *const v_out[3]{Vx.Cur().As<real>(), Vy.Cur().As<real>(), Vz.Cur().As<real>()};

    std::vector<double> residuals(n_components, 0.);
    ParallelFor(n_components, 1, [&](size_t c) {
        const auto &cells = components[c];

        // Pass 1: assign a column to each interior face (a face whose Cell1 neighbor is
        // nonzero), in row-traversal order, grouped x then y then z.
        int counts[3]{};
        std::vector<int> faces[3];
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

        // Pass 2: build the least-squares system enforcing zero net flux per fresh cell
        const int n_dof = counts[0] + counts[1] + counts[2];
        Eigen::MatrixX<real> a = Eigen::MatrixX<real>::Zero(cells.size(), n_dof);
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

                int col = FaceCol[dir][face_cid];
                if (col != -1) col += (dir >= 1 ? counts[0] : 0) + (dir >= 2 ? counts[1] : 0);

                if (col != -1) a(row, col) = sign;
                else g[row] -= sign * v_host[dir][face_cid];
            }
            row += 1;
        }

        if (n_dof > 0) {
            const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixX<real>> qr{a}; // least squares
            const Eigen::VectorX<real> u = qr.solve(g);

            // Write fresh cell velocity solve results back to the shared buffers
            for (const int face_cid : faces[0]) v_out[0][face_cid] = u[FaceCol[0][face_cid]];
            for (const int face_cid : faces[1]) v_out[1][face_cid] = u[counts[0] + FaceCol[1][face_cid]];
            for (const int face_cid : faces[2]) v_out[2][face_cid] = u[counts[0] + counts[1] + FaceCol[2][face_cid]];

            residuals[c] = (a * u - g).squaredNorm();
        }
        // Reset column scratch (each face belongs to exactly one component)
        for (int dir = 0; dir < 3; ++dir) {
            for (const int face_cid : faces[dir]) FaceCol[dir][face_cid] = -1;
        }
    });

    double residual_sq = 0.;
    for (const double r : residuals) residual_sq += r;
    std::cout << "residual: " << std::sqrt(residual_sq) << std::endl;
}

// Section 6.2.2 from [Xue et al. 2024]
void WaveBlender::ShaderReInit() {
    const Dim3 shader_threads{32, 1, 1};
    const Dim3 shader_blocks{uint32_t(NShaderPoints + 31) / 32, 1, 1};

    const ShaderReInitParams params{NShaderSamples, NShaderPoints};
    MetalContext::Get().Dispatch("ShaderReInit", shader_blocks, shader_threads, {&Vx.Cur(), &Vy.Cur(), &Vz.Cur(), &ShaderData.Cur(), &ShaderMap.Cur()}, &params, sizeof(params));
}

bool WaveBlender::ApplyPathTuner::Choose(bool has_points, bool has_forces) {
    constexpr int ProbesPerPath{4};

    SeenForces |= has_forces;
    const uint32_t key = uint32_t(has_points) | (SeenForces ? 2u : 0u);
    if (key != Key) {
        Key = key;
        Locked = -1;
        BestSeconds[0] = BestSeconds[1] = std::numeric_limits<double>::max();
        ProbeCount[0] = ProbeCount[1] = 0;
        InFlightFold = -1;
    }
    if (!has_points) return false; // no boundary application either way
    if (Locked >= 0) return Locked == 1;

    if (ProbeCount[0] >= ProbesPerPath && ProbeCount[1] >= ProbesPerPath) {
        Locked = BestSeconds[1] < BestSeconds[0] ? 1 : 0;
        if (profile::Enabled()) {
            std::cout << std::format("[apply-path] locked {} (plain {:.1f} ms, folded {:.1f} ms per batch)\n", Locked == 1 ? "folded" : "plain", BestSeconds[0] * 1e3, BestSeconds[1] * 1e3);
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

// Encodes all timesteps of the batch into one command buffer with a persistent argument
// table, then commits without waiting — the GPU crunches while the CPU prepares the next
// batch. Each step's sequence is [pressure, boundary application, velocity]. With the
// boundary application folded in and beta derived in-kernel, the loop is one full-grid
// pass per step (the fused StepVelocityPressure kernel), book-ended by a StepPressure
// prologue and a StepVelocity tail:
//   P(0) | [V(0),P(1)] | ... | [V(N-2),P(N-1)] | V(N-1)
// The dispatch computing V(q) gets step q's params (tb, ss, and the listener slot for p(q)).
void WaveBlender::RunFdtd() {
    const profile::Scope scope{"encode/fdtd"};
    auto &ctx = MetalContext::Get();

    const MTL::Size fdtd_threads{FdtdTgX, FdtdTgY, FdtdTgZ};
    const MTL::Size fdtd_blocks{uint32_t(Params.Nx + FdtdTgX - 1) / FdtdTgX, uint32_t(Params.Ny + FdtdTgY - 1) / FdtdTgY, uint32_t(Params.Nz + FdtdTgZ - 1) / FdtdTgZ};

    auto *pressure_pso = ctx.Pipeline("StepPressure");
    auto *fused_pso = ctx.Pipeline("StepVelocityPressure", FoldApply);
    auto *velocity_pso = ctx.Pipeline("StepVelocity", FoldApply);
    auto *shader_pso = FoldApply ? nullptr : ctx.Pipeline("ApplyShader");

    // Standalone boundary application (plain path): blend range then force range, so a
    // dual-claimed face updates in a defined order.
    const MTL::Size shader_threads{32, 1, 1};
    const ApplyShaderRange shader_ranges[2]{{0, NRegularShaderPoints}, {NRegularShaderPoints, NShaderPoints}};

    auto *encoder = ctx.ActiveEncoder();
    const GpuBuffer *const table[]{
        nullptr, &Px, &Py, &Pz, nullptr, nullptr, nullptr, &ShaderFaces.Cur(), &CellState.Cur(), &PmlNp, &PmlDp, &PmlNv, &PmlDv,
        &ShaderData.Cur(), &ShaderMap.Cur(), &ListenerCids, &ListenerOut
    };
    for (uint32_t index = 1; index < std::size(table); ++index) {
        if (table[index]) encoder->setBuffer(table[index]->Handle(), 0, index);
    }

    const FdtdBatchParams batch_params{RhoCcDt, InvDx, InvRhoDt, Params.Nx, Params.Ny, Params.Nz, NShaderSamples, int(Listeners.size())};
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
    // The pending step's boundary application, as standalone dispatches (plain path)
    const auto apply_dispatches = [&] {
        if (FoldApply || NShaderPoints == 0) return;
        encoder->setComputePipelineState(shader_pso);
        for (const auto &range : shader_ranges) {
            if (range.Start == range.End) continue;
            encoder->setBytes(&range, sizeof(range), FdtdApplyRangeIndex);
            encoder->dispatchThreadgroups(MTL::Size{uint32_t(range.End - range.Start + 31) / 32, 1, 1}, shader_threads);
        }
    };
    // Step params of step q, consumed by the dispatch that computes V(q).
    const auto set_step_params = [&](int q) {
        const real t = (q + 1 == NFdtdSamples) ? 1. : real(q + 1) / NFdtdSamples; // normalized blending time (0, 1]
        const real ss = t * (NShaderSamples - 1); // shader sample index (fractional to support interpolation)
        const real tb = (Params.Scheme == BlendScheme::NoBlend && q + 1 != NFdtdSamples) ? 0. : t;
        const FdtdStepParams step_params{tb, ss, q};
        encoder->setBytes(&step_params, sizeof(step_params), FdtdStepParamsIndex);
    };

    // Prologue: P(0) (in-place, before step 0's boundary application)
    bind_fields();
    encoder->setComputePipelineState(pressure_pso);
    encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);

    for (int q = 1; q < NFdtdSamples; ++q) {
        set_step_params(q - 1);
        apply_dispatches();

        encoder->setComputePipelineState(fused_pso);
        encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);
        P.Flip();
        Vx.Flip();
        Vy.Flip();
        Vz.Flip();
        bind_fields();
    }

    // Tail: V(N-1) (in-place) and the final listener sample p(N-1)
    set_step_params(NFdtdSamples - 1);
    apply_dispatches();
    encoder->setComputePipelineState(velocity_pso);
    encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);

    Step += NFdtdSamples;
    ctx.Flush();
    ListenerPending = true;
    if (NShaderPoints > 0) PathTuner.MarkInFlight(FoldApply); // measure this batch at the next sync
}
