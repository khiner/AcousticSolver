// Ported from WaveBlender (c) 2024 Kangrui Xue (WaveBlender.cu + GPUSolver.cu) — Metal port.
// See WaveBlender.h for the batch pipeline and its synchronization contract.
//
// References:
//   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains

#include "WaveBlender.h"

#include "KernelParams.h"
#include "Parallel.h"
#include "Profile.h"
#include "tribox.h"

#include <Metal/Metal.hpp>

#include <atomic>
#include <cmath>
#include <format>
#include <iostream>

WaveBlender::WaveBlender(const SimParams &params)
    : Params(params), GridSize(Params.Nx * Params.Ny * Params.Nz),
      NFdtdSamples(Params.FdtdSrate / Params.BlendRate), NShaderSamples(Params.ShaderSrate / Params.BlendRate + 1),
      RhoCCDt(Params.Rho * Params.C * Params.C * Params.Dt), InvDx(1. / Params.Dx), InvRhoDt(1. / Params.Rho * Params.Dt),
      Damping(Params.Damping) {
    for (auto *field : {&P, &Vx, &Vy, &Vz}) { // both ping-pong slots
        field->Cur().ResizeZeroed(GridSize * sizeof(REAL));
        field->Other().ResizeZeroed(GridSize * sizeof(REAL));
    }
    Beta.ResizeZeroed(GridSize * sizeof(REAL));

    Cell1.resize(GridSize);
    Cell2.resize(GridSize);
    BetaSolid.assign(GridSize, 0);

    for (auto &stamp : FaceStamp) stamp.assign(GridSize, 0);
    for (auto &col : FaceCol) col.assign(GridSize, -1);
    FreshComponent.assign(GridSize, -1);

    InitializePml();
}

// Same simple quadratic-ramp split-field PML as the reference (see its TODO on C-PML).
void WaveBlender::InitializePml() {
    Px.ResizeZeroed(GridSize * sizeof(REAL));
    Py.ResizeZeroed(GridSize * sizeof(REAL));
    Pz.ResizeZeroed(GridSize * sizeof(REAL));

    const int max_half_grid_length = (std::max({Params.Nx, Params.Ny, Params.Nz}) + 1) / 2;

    // Velocity and pressure PML weights: numerator and denominator
    std::vector<REAL> pml_nv(max_half_grid_length + 1, 1.), pml_dv(max_half_grid_length + 1, 1. / (1. + Damping));
    std::vector<REAL> pml_np(max_half_grid_length, 1.), pml_dp(max_half_grid_length, 1.);

    for (int dist = 0; dist < PML_WIDTH; ++dist) {
        REAL weight = (REAL(PML_WIDTH) - dist) / PML_WIDTH; // velocity
        weight = 0.5 * weight * weight;
        pml_nv[dist] = 1. - weight;
        pml_dv[dist] = 1. / (1. + weight);

        weight = (REAL(PML_WIDTH) - dist - 0.5) / PML_WIDTH; // pressure
        weight = 0.5 * weight * weight;
        pml_np[dist] = 1. - weight;
        pml_dp[dist] = 1. / (1. + weight);
    }
    PmlNv.Resize((max_half_grid_length + 1) * sizeof(REAL));
    PmlDv.Resize((max_half_grid_length + 1) * sizeof(REAL));
    PmlNv.Upload(pml_nv.data(), (max_half_grid_length + 1) * sizeof(REAL));
    PmlDv.Upload(pml_dv.data(), (max_half_grid_length + 1) * sizeof(REAL));

    PmlNp.Resize(max_half_grid_length * sizeof(REAL));
    PmlDp.Resize(max_half_grid_length * sizeof(REAL));
    PmlNp.Upload(pml_np.data(), max_half_grid_length * sizeof(REAL));
    PmlDp.Upload(pml_dp.data(), max_half_grid_length * sizeof(REAL));
}

void WaveBlender::AddListener(const std::string &format, const std::vector<REAL> &position, const std::string &output_name) {
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
    ListenerOut.ResizeZeroed(std::max<size_t>(1, size_t(NFdtdSamples) * cids.size()) * sizeof(REAL));
}

void WaveBlender::WritePendingListeners() {
    if (!ListenerPending) return;
    ListenerPending = false;

    const auto *samples = ListenerOut.As<REAL>(); // completed at this batch's sync point
    const int n = Listeners.size();
    for (int l = 0; l < n; ++l) {
        auto &listener = Listeners[l];
        if (n == 1) {
            listener.Out.write(reinterpret_cast<const char *>(samples), NFdtdSamples * sizeof(REAL));
        } else {
            for (int s = 0; s < NFdtdSamples; ++s) {
                listener.Out.write(reinterpret_cast<const char *>(&samples[size_t(s) * n + l]), sizeof(REAL));
            }
        }
    }
}

void WaveBlender::LogZSlice(const std::string &filetag) {
    const int offset = Cid(0, 0, Params.Nz / 2); // z-slice
    std::ofstream logfile{filetag, std::ofstream::binary};

    const auto *p = P.Cur().As<REAL>(); // synchronizes the stream
    logfile.write(reinterpret_cast<const char *>(p + offset), Params.Nx * Params.Ny * sizeof(REAL));

    const auto *beta = Beta.As<REAL>();
    logfile.write(reinterpret_cast<const char *>(beta + offset), Params.Nx * Params.Ny * sizeof(REAL));
}

bool WaveBlender::RunBatch() {
    if (Step >= (Params.Tf - Params.Ts) * Params.FdtdSrate) {
        MetalContext::Get().Sync(); // drain the final batch
        WritePendingListeners();
        return false;
    }
    if (Step == 0) InitializeListeners();

    Cell1 = Cell2;

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
            for (int cid = 0; cid < GridSize; ++cid) { // Only clear cells of objects that moved
                if (!keep_value[Cell2[cid]]) Cell2[cid] = 0;
            }
        }
        {
            const profile::Scope scope{"cpu/rasterize"};
            Rasterize(keep_value);
        }
        {
            const profile::Scope scope{"cpu/detect_cavities"};
            DetectCavities(); // Runtime cavity detection
        }

        if (Step == 0) { // Initialize Beta
            std::vector<REAL> buf(GridSize);
            for (int cid = 0; cid < GridSize; ++cid) {
                BetaSolid[cid] = Cell2[cid] > 0;
                buf[cid] = BetaSolid[cid] ? 1. : 0.;
            }
            Beta.Upload(buf.data(), buf.size() * sizeof(REAL));
        }
        {
            const profile::Scope scope{"cpu/setup_shaders"};
            SetupShaders();
        }
        // Cells changing solidity this batch: the per-step beta update set (each batch
        // ends at tb = 1, so beta is exactly 0 or 1 at batch start).
        TransitionsHost.clear();
        for (int cid = 0; cid < GridSize; ++cid) {
            const uint8_t solid = Cell2[cid] > 0;
            if (solid != BetaSolid[cid]) {
                TransitionsHost.push_back((cid << 1) | solid);
                BetaSolid[cid] = solid;
            }
        }
        NBetaTransitions = TransitionsHost.size();
        if (NBetaTransitions > 0) {
            BetaTransitions.Flip();
            BetaTransitions.Cur().Resize(size_t(NBetaTransitions) * sizeof(int));
            BetaTransitions.Cur().Upload(TransitionsHost.data(), size_t(NBetaTransitions) * sizeof(int));
        }
    } else {
        NBetaTransitions = 0; // geometry unchanged: beta is fully converged
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
        FreshCellPressure(); // Fresh cell extrapolation
    }

    // The one sync point per batch: the fresh-cell velocity solve reads the previous
    // batch's final velocities (and this batch's prologue kernels, which clear interior
    // velocities first) on the host.
    MetalContext::Get().Sync();
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

        // Point source rasterization special handling
        // (for now, assumes point sources are specified at the end of config file)
        if (base.Type == ShaderClass::Point) {
            for (int r = 0; r < v.rows(); ++r) {
                Eigen::Vector3<REAL> pt = v.row(r);
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
                Eigen::Vector3<REAL> pt = v.row(r);
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
            Eigen::Matrix3<REAL> tri_v;
            tri_v.row(0) = v.row(face[0]);
            tri_v.row(1) = v.row(face[1]);
            tri_v.row(2) = v.row(face[2]);

            Eigen::Vector3<REAL> min = tri_v.colwise().minCoeff();
            min += offset;
            Eigen::Vector3<REAL> max = tri_v.colwise().maxCoeff();
            max += offset;

            const int min_i = std::max(int(min[0] / Params.Dx + Params.Nx / 2.), 0);
            const int max_i = std::min(int(max[0] / Params.Dx + Params.Nx / 2.), Params.Nx - 1);

            const int min_j = std::max(int(min[1] / Params.Dx + Params.Ny / 2.), 0);
            const int max_j = std::min(int(max[1] / Params.Dx + Params.Ny / 2.), Params.Ny - 1);

            const int min_k = std::max(int(min[2] / Params.Dx + Params.Nz / 2.), 0);
            const int max_k = std::min(int(max[2] / Params.Dx + Params.Nz / 2.), Params.Nz - 1);

            REAL boxhalfsize[3]{Params.Dx / 2.f, Params.Dx / 2.f, Params.Dx / 2.f};
            REAL triverts[3][3]{
                {tri_v(0, 0), tri_v(0, 1), tri_v(0, 2)},
                {tri_v(1, 0), tri_v(1, 1), tri_v(1, 2)},
                {tri_v(2, 0), tri_v(2, 1), tri_v(2, 2)},
            };
            for (int i = min_i; i <= max_i; ++i) {
                for (int j = min_j; j <= max_j; ++j) {
                    for (int k = min_k; k <= max_k; ++k) {
                        const Eigen::Vector3<REAL> p = Pos(i, j, k) - offset;
                        REAL boxcenter[3]{p[0], p[1], p[2]};
                        if (!triBoxOverlap(boxcenter, boxhalfsize, triverts)) continue;

                        const std::atomic_ref<uint8_t> cell{Cell2[Cid(i, j, k)]};
                        if (!keep_value[cell.load(std::memory_order_relaxed)]) cell.store(fill, std::memory_order_relaxed);
                    }
                }
            }
        }); // loop over triangles
    } // loop over objects
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

        Eigen::Vector3<REAL> min = v.colwise().minCoeff();
        min += offset;
        Eigen::Vector3<REAL> max = v.colwise().maxCoeff();
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

    // Flood fill to detect connected components
    FloodVisited.assign(GridSize, 0);
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
        std::vector<Eigen::Vector3<REAL>> b_vec;
        std::vector<Eigen::Vector3<REAL>> bn_vec;
        int bid = 0;
        for (int cid = 0; cid < GridSize; ++cid) {
            if (Cell1[cid] != fill && Cell2[cid] != fill) continue;

            const int i = cid % Params.Nx;
            const int j = (cid / Params.Nx) % Params.Ny;
            const int k = (cid / Params.Nx) / Params.Ny;

            if (i == 0 || i >= Params.Nx - 1 || j == 0 || j >= Params.Ny - 1 || k == 0 || k >= Params.Nz - 1) continue;

            const Eigen::Vector3<REAL> p = Pos(i, j, k) - Offsets[oid]; // world-space to object-space

            const int neighbor_cids[6]{
                Cid(i - 1, j, k), Cid(i + 1, j, k), // left, right
                Cid(i, j - 1, k), Cid(i, j + 1, k), // down, up
                Cid(i, j, k - 1), Cid(i, j, k + 1) // back, front
            };
            for (int n = 0; n < 6; ++n) {
                if ((Cell1[cid] == fill && Cell1[neighbor_cids[n]] == 0) ||
                    (Cell2[cid] == fill && Cell2[neighbor_cids[n]] == 0)) {
                    const int d = (n % 2 == 0);
                    const REAL sign = (n % 2 == 0) ? -1. : 1.;
                    const int dir = n / 2;
                    const int face_cid = (dir == 0) ? Cid(i - d, j, k) : (dir == 1) ? Cid(i, j - d, k) :
                                                                                      Cid(i, j, k - d);

                    if (FaceStamp[dir][face_cid] == FaceEpoch) continue; // face already claimed
                    FaceStamp[dir][face_cid] = FaceEpoch;

                    ShaderMapHost.push_back(3 * face_cid + dir);
                    bn_vec.emplace_back(sign * Eigen::Vector3<REAL>{dir == 0 ? 1.f : 0.f, dir == 1 ? 1.f : 0.f, dir == 2 ? 1.f : 0.f});
                    b_vec.emplace_back(p.cast<REAL>() + (0.5 * Params.Dx) * bn_vec[bid]);
                    bid += 1;
                }
            }
        } // loop over cells

        Eigen::MatrixX<REAL> b(b_vec.size(), 3);
        Eigen::MatrixX<REAL> bn(bn_vec.size(), 3);
        for (int row = 0; row < int(b_vec.size()); ++row) {
            b.row(row) = b_vec[row];
            bn.row(row) = bn_vec[row];
        }
        base.SetSamplePoints(b, bn);
    } // loop over objects

    NRegularShaderPoints = ShaderMapHost.size();

    // Boundary faces of point sources (all 6 faces of each point cell — the cells
    // revert to air since we don't want to rasterize point sources)
    FaceEpoch += 1;
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        auto &base = Base(Objects[oid]);
        if (base.Type != ShaderClass::Point) continue;

        const uint8_t fill = oid + 1;
        std::vector<Eigen::Vector3<REAL>> b_vec;
        std::vector<Eigen::Vector3<REAL>> bn_vec;
        int bid = 0;
        for (int cid = 0; cid < GridSize; ++cid) {
            if (Cell2[cid] != fill) continue;
            Cell2[cid] = 0;

            const int i = cid % Params.Nx;
            const int j = (cid / Params.Nx) % Params.Ny;
            const int k = (cid / Params.Nx) / Params.Ny;

            if (i == 0 || i >= Params.Nx - 1 || j == 0 || j >= Params.Ny - 1 || k == 0 || k >= Params.Nz - 1) continue;

            const Eigen::Vector3<REAL> p = Pos(i, j, k) - Offsets[oid]; // world-space to object-space

            for (int n = 0; n < 6; ++n) {
                const int d = (n % 2 == 0);
                const REAL sign = (n % 2 == 0) ? -1. : 1.;
                const int dir = n / 2;
                const int face_cid = (dir == 0) ? Cid(i - d, j, k) : (dir == 1) ? Cid(i, j - d, k) :
                                                                                  Cid(i, j, k - d);

                if (FaceStamp[dir][face_cid] == FaceEpoch) continue; // face already claimed
                FaceStamp[dir][face_cid] = FaceEpoch;

                ShaderMapHost.push_back(-3 * face_cid - dir);
                bn_vec.emplace_back(sign * Eigen::Vector3<REAL>{dir == 0 ? 1.f : 0.f, dir == 1 ? 1.f : 0.f, dir == 2 ? 1.f : 0.f});
                b_vec.emplace_back(p.cast<REAL>() + (0.5 * Params.Dx) * bn_vec[bid]);
                bid += 1;
            }
        } // loop over cells

        Eigen::MatrixX<REAL> b(b_vec.size(), 3);
        Eigen::MatrixX<REAL> bn(bn_vec.size(), 3);
        for (int row = 0; row < int(b_vec.size()); ++row) {
            b.row(row) = b_vec[row];
            bn.row(row) = bn_vec[row];
        }
        base.SetSamplePoints(b, bn);
    } // loop over objects

    // Upload raster and shader map into this batch's double-buffer slots
    Cell.Flip();
    Cell.Cur().Resize(GridSize * sizeof(uint8_t));
    Cell.Cur().Upload(Cell2.data(), GridSize * sizeof(uint8_t));

    NShaderPoints = ShaderMapHost.size();
    MaxNShaderPoints = std::max(MaxNShaderPoints, NShaderPoints);

    ShaderMap.Flip();
    ShaderMap.Cur().Resize(size_t(MaxNShaderPoints) * sizeof(int));
    ShaderMap.Cur().Upload(ShaderMapHost.data(), NShaderPoints * sizeof(int));

    ShaderData.Flip();
    ShaderData.Cur().Resize(size_t(MaxNShaderPoints) * NShaderSamples * sizeof(REAL));
    ShaderData.Cur().Zero(size_t(NShaderPoints) * NShaderSamples * sizeof(REAL));
}

void WaveBlender::FreshCellPressure() {
    auto &ctx = MetalContext::Get();

    const Dim3 fdtd_threads{8, 8, 8};
    const Dim3 fdtd_blocks{uint32_t(Params.Nx + 7 - PML_WIDTH) / 8, uint32_t(Params.Ny + 7 - PML_WIDTH) / 8, uint32_t(Params.Nz + 7 - PML_WIDTH) / 8};

    const Dim3 shader_threads{32, 1, 1};
    const Dim3 shader_blocks{uint32_t(NShaderPoints + 31) / 32, 1, 1};

    const REAL incr = REAL(Params.ShaderSrate) / Params.FdtdSrate; // For now, assumes shader rate is same for all objects

    // We use split-pressure buffers as acceleration buffers for convenience (since split-pressure only used in PML)
    // -- just make sure to clear afterwards
    const PrepareFreshCellParams prep_params{NShaderSamples, NShaderPoints, float(1. / Params.Dt), incr};
    ctx.Dispatch("prepare_fresh_cell", shader_blocks, shader_threads, {&Px, &Py, &Pz, &ShaderData.Cur(), &ShaderMap.Cur()}, &prep_params, sizeof(prep_params));

    const FreshCellPressureParams fc_params{Params.Nx, Params.Ny, Params.Nz, Params.Rho * Params.Dx};
    ctx.Dispatch("fresh_cell_pressure", fdtd_blocks, fdtd_threads, {&P.Cur(), &Px, &Py, &Pz, &Beta}, &fc_params, sizeof(fc_params));

    const ClearSolidParams cs_params{Params.Nx, Params.Ny, Params.Nz};
    ctx.Dispatch("clear_solid", fdtd_blocks, fdtd_threads, {&Vx.Cur(), &Vy.Cur(), &Vz.Cur(), &Px, &Py, &Pz, &Beta}, &cs_params, sizeof(cs_params));
}

void WaveBlender::FreshCellVelocity() {
    // point_value[v]: cell value v belongs to a point source
    bool point_value[256]{};
    for (int oid = 0; oid < int(Objects.size()); ++oid) {
        if (Base(Objects[oid]).Type == ShaderClass::Point) point_value[oid + 1] = true;
    }

    // Determine fresh cells (based on Cell1 and Cell2 difference), in ascending cell order
    std::vector<int> fresh_cids;
    for (int cid = 0; cid < GridSize; ++cid) {
        if (Cell2[cid] == 0 && Cell1[cid] > 0 && Cell1[cid] != CavityInterior && !point_value[Cell1[cid]]) fresh_cids.push_back(cid);
    }
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

    const auto *vx_host = Vx.Cur().As<REAL>();
    const auto *vy_host = Vy.Cur().As<REAL>();
    const auto *vz_host = Vz.Cur().As<REAL>();
    auto *vx_out = Vx.Cur().As<REAL>();
    auto *vy_out = Vy.Cur().As<REAL>();
    auto *vz_out = Vz.Cur().As<REAL>();

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
        Eigen::MatrixX<REAL> a = Eigen::MatrixX<REAL>::Zero(cells.size(), n_dof);
        Eigen::VectorX<REAL> g = Eigen::VectorX<REAL>::Zero(cells.size());

        int row = 0;
        for (const int cid : cells) {
            const int i = cid % Params.Nx;
            const int j = (cid / Params.Nx) % Params.Ny;
            const int k = (cid / Params.Nx) / Params.Ny;

            for (int n = 0; n < 6; ++n) {
                const int d = (n % 2 == 0);
                const REAL sign = (n % 2 == 0) ? -1. : 1.;
                const int dir = n / 2;
                const int face_cid = (dir == 0) ? Cid(i - d, j, k) : (dir == 1) ? Cid(i, j - d, k) :
                                                                                  Cid(i, j, k - d);

                int col = FaceCol[dir][face_cid];
                if (col != -1) col += (dir >= 1 ? counts[0] : 0) + (dir >= 2 ? counts[1] : 0);

                if (col != -1) a(row, col) = sign;
                else g[row] -= sign * ((dir == 0) ? vx_host[face_cid] : (dir == 1) ? vy_host[face_cid] :
                                                                                     vz_host[face_cid]);
            }
            row += 1;
        }

        if (n_dof > 0) {
            const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixX<REAL>> qr{a}; // least squares
            const Eigen::VectorX<REAL> u = qr.solve(g);

            // Write fresh cell velocity solve results back to the shared buffers
            for (const int face_cid : faces[0]) vx_out[face_cid] = u[FaceCol[0][face_cid]];
            for (const int face_cid : faces[1]) vy_out[face_cid] = u[counts[0] + FaceCol[1][face_cid]];
            for (const int face_cid : faces[2]) vz_out[face_cid] = u[counts[0] + counts[1] + FaceCol[2][face_cid]];

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
    MetalContext::Get().Dispatch("shader_reinit", shader_blocks, shader_threads, {&Vx.Cur(), &Vy.Cur(), &Vz.Cur(), &ShaderData.Cur(), &ShaderMap.Cur()}, &params, sizeof(params));
}

// Encodes all timesteps of the batch into one command buffer with a persistent argument
// table, then commits without waiting — the GPU crunches while the CPU prepares the next
// batch. The steady-state loop runs one full-grid pass per step (the fused
// step_velocity_pressure kernel) plus the small apply_shader and update_beta dispatches,
// book-ended by a step_pressure prologue and a step_velocity tail:
//   P(0) A(0) | [V(0),P(1)] beta(1) A(1) | ... | [V(N-2),P(N-1)] beta(N-1) A(N-1) | V(N-1)
void WaveBlender::RunFdtd() {
    const profile::Scope scope{"encode/fdtd"};
    auto &ctx = MetalContext::Get();

    const MTL::Size fdtd_threads{8, 8, 8};
    const MTL::Size fdtd_blocks{uint32_t(Params.Nx + 7) / 8, uint32_t(Params.Ny + 7) / 8, uint32_t(Params.Nz + 7) / 8};
    const MTL::Size shader_threads{32, 1, 1};

    // Velocity-blend points and force points dispatch separately, blend first, so a face
    // claimed by both updates in a defined order (see ApplyShaderRange in KernelParams.h).
    const ApplyShaderRange shader_ranges[2]{{0, NRegularShaderPoints}, {NRegularShaderPoints, NShaderPoints}};
    MTL::Size shader_blocks[2];
    for (int p = 0; p < 2; ++p) shader_blocks[p] = MTL::Size{uint32_t(shader_ranges[p].end - shader_ranges[p].start + 31) / 32, 1, 1};

    auto *pressure_pso = ctx.Pipeline("step_pressure");
    auto *shader_pso = ctx.Pipeline("apply_shader");
    auto *fused_pso = ctx.Pipeline("step_velocity_pressure");
    auto *beta_pso = ctx.Pipeline("update_beta");
    auto *velocity_pso = ctx.Pipeline("step_velocity");

    auto *encoder = ctx.ActiveEncoder();
    const GpuBuffer *const table[]{
        nullptr, &Px, &Py, &Pz, nullptr, nullptr, nullptr, &Beta, &Cell.Cur(), &PmlNp, &PmlDp, &PmlNv, &PmlDv,
        &ShaderData.Cur(), &ShaderMap.Cur(), &ListenerCids, &ListenerOut
    };
    for (uint32_t index = 1; index < std::size(table); ++index) {
        if (table[index]) encoder->setBuffer(table[index]->Handle(), 0, index);
    }
    encoder->setBuffer(BetaTransitions.Cur().Handle(), 0, 20);

    const FdtdBatchParams batch_params{RhoCCDt, InvDx, InvRhoDt, Params.Nx, Params.Ny, Params.Nz, NShaderSamples, int(Listeners.size()), NBetaTransitions};
    encoder->setBytes(&batch_params, sizeof(batch_params), FDTD_BATCH_PARAMS_INDEX);

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
    // The listener slot is the fused kernel's pre-update pressure sample p(q - 1).
    const auto set_step_params = [&](int q) {
        REAL t = (q + 1 == NFdtdSamples) ? 1. : REAL(q + 1) / NFdtdSamples; // normalized blending time (0, 1]
        const REAL ss = t * (NShaderSamples - 1); // shader sample index (fractional to support interpolation)
        if (Params.Scheme == BlendScheme::NoBlend) t = (q + 1 == NFdtdSamples) ? 1. : 0.;
        const FdtdStepParams step_params{t, ss, q - 1};
        encoder->setBytes(&step_params, sizeof(step_params), FDTD_STEP_PARAMS_INDEX);
    };
    const auto apply_shader_dispatches = [&] {
        if (NShaderPoints == 0) return;
        encoder->setComputePipelineState(shader_pso);
        for (int p = 0; p < 2; ++p) {
            if (shader_ranges[p].start == shader_ranges[p].end) continue;
            encoder->setBytes(&shader_ranges[p], sizeof(shader_ranges[p]), FDTD_APPLY_RANGE_INDEX);
            encoder->dispatchThreadgroups(shader_blocks[p], shader_threads);
        }
    };

    // Prologue: P(0) (in-place, with its own per-cell beta update) and A(0)
    bind_fields();
    set_step_params(0);
    encoder->setComputePipelineState(pressure_pso);
    encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);
    apply_shader_dispatches();

    for (int q = 1; q < NFdtdSamples; ++q) {
        set_step_params(q);

        encoder->setComputePipelineState(fused_pso);
        encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);
        P.Flip();
        Vx.Flip();
        Vy.Flip();
        Vz.Flip();
        bind_fields();

        if (NBetaTransitions > 0) {
            encoder->setComputePipelineState(beta_pso);
            encoder->dispatchThreadgroups(MTL::Size{uint32_t(NBetaTransitions + 31) / 32, 1, 1}, shader_threads);
        }
        apply_shader_dispatches();
    }

    // Tail: V(N-1) (in-place) and the final listener sample p(N-1)
    set_step_params(NFdtdSamples); // listener slot N-1; tb/ss unused by step_velocity
    encoder->setComputePipelineState(velocity_pso);
    encoder->dispatchThreadgroups(fdtd_blocks, fdtd_threads);

    Step += NFdtdSamples;
    ctx.Flush();
    ListenerPending = true;
}
