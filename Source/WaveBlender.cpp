// Ported from WaveBlender (c) 2024 Kangrui Xue (WaveBlender.cu) — Metal port.
//
// The CPU-side batch logic (rasterization, cavity detection, shader setup, fresh-cell
// velocity solve) is unchanged from the reference. CUDA kernel launches became Metal
// dispatches, and the per-element cudaMemcpys in the fresh-cell velocity solve became
// direct reads/writes of the shared buffers.
//
// References:
//   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains

#include "WaveBlender.h"

#include "tribox.h"
#include <algorithm>

bool WaveBlender::RunBatch() {
    if (Step >= (Params.Tf - Params.Ts) * Params.FdtdSrate) return false;

    Cell1 = Cell2;

    // 1. Check if objects have moved; if not, we can skip step 2
    std::set<int> unchanged_oids;
    bool any_changed = false;
    for (int oid = 0; oid < Objects.size(); ++oid) {
        if (!Objects[oid]->Changed) unchanged_oids.insert(oid);
        else any_changed = true;
    }
    // 2. Rasterize objects and setup shaders
    if (any_changed) {
        for (int cid = 0; cid < Cell2.size(); ++cid) { // Only clear cells of objects that moved
            if (unchanged_oids.count(Cell2[cid] - 1) == 0) Cell2[cid] = 0;
        }
        Rasterize(unchanged_oids);

        DetectCavities(); // Runtime cavity detection

        if (Step == 0) { // Initialize Beta
            std::vector<REAL> buf(GridSize);
            for (int cid = 0; cid < buf.size(); ++cid) buf[cid] = (Cell2[cid] > 0) ? 1. : 0.;
            Beta.Upload(buf.data(), buf.size() * sizeof(REAL));
        }
        SetupShaders();
    }
    // 3. Compute shader values at all sampled positions and times
    // (for now, assumes point sources are specified at the end of config file)
    int global_bid = 0;
    for (const auto &obj : Objects) {
        obj->Compute(ShaderData, global_bid);
        if (obj->HasShader) global_bid += obj->NPoints;
    }

    // 4. Additional "per-batch overhead"
    FreshCellPressure(); // Fresh cell extrapolation
    FreshCellVelocity();

    ShaderReInit(); // Shader velocity re-initialization

    // 5. Run FDTD update
    RunFdtd();

    return true;
}

// Conservative CPU rasterizer based on triangle-box overlap test.
void WaveBlender::Rasterize(const std::set<int> &unchanged_oids, bool log_raster) {
    for (int oid = 0; oid < Objects.size(); ++oid) {
        if (!Objects[oid]->Changed) continue;

        const auto &offset = Offsets[oid];
        const auto &v = Objects[oid]->V2;
        const auto &f = Objects[oid]->F;

        // Point source rasterization special handling
        // (for now, assumes point sources are specified at the end of config file)
        if (Objects[oid]->Type == ShaderClass::Point) {
            for (int r = 0; r < v.rows(); ++r) {
                Eigen::Vector3<REAL> pt = v.row(r);
                pt += offset;

                const int i = int(pt[0] / Params.Dx + Params.Nx / 2.);
                const int j = int(pt[1] / Params.Dx + Params.Ny / 2.);
                const int k = int(pt[2] / Params.Dx + Params.Nz / 2.);

                if (const int cid = Cid(i, j, k); Cell2[cid] == 0) Cell2[cid] = oid + 1;

                // Neighboring cells needed for point-to-grid
                const int neighbor_cids[6]{
                    Cid(i - 1, j, k), Cid(i + 1, j, k), // left, right
                    Cid(i, j - 1, k), Cid(i, j + 1, k), // down, up
                    Cid(i, j, k - 1), Cid(i, j, k + 1) // back, front
                };
                for (int n = 0; n < 6; n += 2) {
                    if (Cell2[neighbor_cids[n]] == 0) Cell2[neighbor_cids[n]] = oid + 1;
                }
                // (we brute-force consider all 6 neighbors for good measure, though only 3 are needed)
            }
            continue;
        }
        // Density rasterization special handling
        if (Objects[oid]->Type == ShaderClass::Density) {
            for (int r = 0; r < v.rows(); ++r) {
                Eigen::Vector3<REAL> pt = v.row(r);
                pt += offset;

                const int i = int(pt[0] / Params.Dx + Params.Nx / 2.);
                const int j = int(pt[1] / Params.Dx + Params.Ny / 2.);
                const int k = int(pt[2] / Params.Dx + Params.Nz / 2.);

                const int cid = Cid(i, j, k);
                const int density = f(r, 0); // HACK to encode density in object's faces

                constexpr int Threshold{97};
                if (density >= Threshold && unchanged_oids.count(Cell2[cid] - 1) == 0) Cell2[cid] = oid + 1;
            }
            continue;
        }
        // General triangle mesh rasterization
        for (int r = 0; r < f.rows(); ++r) {
            const Eigen::Vector3i face = f.row(r);
            Eigen::Matrix3<REAL> tri_v;
            tri_v.row(0) = v.row(face[0]);
            tri_v.row(1) = v.row(face[1]);
            tri_v.row(2) = v.row(face[2]);

            Eigen::Vector3<REAL> min = tri_v.colwise().minCoeff();
            min += offset;
            Eigen::Vector3<REAL> max = tri_v.colwise().maxCoeff();
            max += offset;

            const int min_i = int(min[0] / Params.Dx + Params.Nx / 2.);
            const int max_i = int(max[0] / Params.Dx + Params.Nx / 2.);

            const int min_j = int(min[1] / Params.Dx + Params.Ny / 2.);
            const int max_j = int(max[1] / Params.Dx + Params.Ny / 2.);

            const int min_k = int(min[2] / Params.Dx + Params.Nz / 2.);
            const int max_k = int(max[2] / Params.Dx + Params.Nz / 2.);

            for (int i = min_i; i <= max_i; ++i) {
                for (int j = min_j; j <= max_j; ++j) {
                    for (int k = min_k; k <= max_k; ++k) {
                        const Eigen::Vector3<REAL> p = Pos(i, j, k) - offset;

                        REAL boxcenter[3]{p[0], p[1], p[2]};
                        REAL boxhalfsize[3]{Params.Dx / 2.f, Params.Dx / 2.f, Params.Dx / 2.f};
                        REAL triverts[3][3]{
                            {tri_v(0, 0), tri_v(0, 1), tri_v(0, 2)},
                            {tri_v(1, 0), tri_v(1, 1), tri_v(1, 2)},
                            {tri_v(2, 0), tri_v(2, 1), tri_v(2, 2)},
                        };
                        if (triBoxOverlap(boxcenter, boxhalfsize, triverts)) {
                            if (i < 0 || i >= Params.Nx || j < 0 || j >= Params.Ny || k < 0 || k >= Params.Nz) continue;

                            if (const int cid = Cid(i, j, k); unchanged_oids.count(Cell2[cid] - 1) == 0) Cell2[cid] = oid + 1;
                        }
                    }
                }
            }
        } // loop over triangles
    } // loop over objects

    if (log_raster) {
        std::ofstream raster_file{"logs/raster-" + std::to_string(Step) + ".txt"};
        raster_file << Params.Nx << " " << Params.Ny << " " << Params.Nz << " " << Params.Dx << std::endl; // first line is metadata

        for (int cid = 0; cid < Cell2.size(); ++cid) {
            if (Cell2[cid] != 0 && Cell2[cid] != CavityInterior) {
                const int i = cid % Params.Nx;
                const int j = (cid / Params.Nx) % Params.Ny;
                const int k = (cid / Params.Nx) / Params.Ny;

                raster_file << Cell2[cid] << " " << i << " " << j << " " << k << std::endl;
            }
        }
    }
}

// Section 6.2.3 from [Xue et al. 2024]
void WaveBlender::DetectCavities() {
    std::vector<bool> connected(Cell2.size());
    std::queue<int> q;
    REAL tmp = 0.;

    // Determine bounding box
    int min_i = Params.Nx - 1, max_i = 0;
    int min_j = Params.Ny - 1, max_j = 0;
    int min_k = Params.Nz - 1, max_k = 0;
    for (int oid = 0; oid < Objects.size(); ++oid) {
        if (Objects[oid]->Type == ShaderClass::Point) continue;

        const auto &v = Objects[oid]->V2;
        const auto &offset = Offsets[oid];

        Eigen::Vector3<REAL> min = v.colwise().minCoeff();
        min += offset;
        Eigen::Vector3<REAL> max = v.colwise().maxCoeff();
        max += offset;

        tmp = int(min[0] / Params.Dx + Params.Nx / 2.);
        min_i = std::min<REAL>(tmp, min_i);
        tmp = int(max[0] / Params.Dx + Params.Nx / 2.);
        max_i = std::max<REAL>(tmp, max_i);

        tmp = int(min[1] / Params.Dx + Params.Ny / 2.);
        min_j = std::min<REAL>(tmp, min_j);
        tmp = int(max[1] / Params.Dx + Params.Ny / 2.);
        max_j = std::max<REAL>(tmp, max_j);

        tmp = int(min[2] / Params.Dx + Params.Nz / 2.);
        min_k = std::min<REAL>(tmp, min_k);
        tmp = int(max[2] / Params.Dx + Params.Nz / 2.);
        max_k = std::max<REAL>(tmp, max_k);
    }
    min_i -= 1;
    min_j -= 1;
    min_k -= 1;
    max_i += 1;
    max_j += 1;
    max_k += 1;

    // Flood fill to detect connected components
    q.push(Cid(max_i, max_j, max_k));
    while (!q.empty()) {
        const int cid = q.front();
        q.pop();

        const int i = cid % Params.Nx;
        const int j = (cid / Params.Nx) % Params.Ny;
        const int k = (cid / Params.Nx) / Params.Ny;

        // Enforce that the PML contains no objects to save time on flood fill
        if (i < min_i || i > max_i || j < min_j || j > max_j || k < min_k || k > max_k) continue;
        if (!connected[cid] && (Cell2[cid] == 0 || Objects[Cell2[cid] - 1]->Type == ShaderClass::Point)) {
            connected[cid] = true;
            q.push(Cid(i - 1, j, k));
            q.push(Cid(i + 1, j, k)); // left, right
            q.push(Cid(i, j - 1, k));
            q.push(Cid(i, j + 1, k)); // down, up
            q.push(Cid(i, j, k - 1));
            q.push(Cid(i, j, k + 1)); // back, front
        }
    }
    // Fill in cavities
    for (int i = min_i; i <= max_i; ++i) {
        for (int j = min_j; j <= max_j; ++j) {
            for (int k = min_k; k <= max_k; ++k) {
                const int cid = Cid(i, j, k);
                if (!connected[cid] && (Cell2[cid] == 0 || Objects[Cell2[cid] - 1]->Type == ShaderClass::Point)) {
                    Cell2[cid] = CavityInterior;
                }
            }
        }
    }
}

// Section 6.1 from [Xue et al. 2024]
void WaveBlender::SetupShaders() {
    std::vector<int> shader_map;
    std::set<int> used_vxids, used_vyids, used_vzids;
    for (int oid = 0; oid < Objects.size(); ++oid) {
        if (!Objects[oid]->HasShader || Objects[oid]->Type == ShaderClass::Point) continue;

        int bid = 0;
        std::vector<Eigen::Vector3<REAL>> b_vec;
        std::vector<Eigen::Vector3<REAL>> bn_vec;
        for (int cid = 0; cid < Cell2.size(); ++cid) {
            if (Cell1[cid] != oid + 1 && Cell2[cid] != oid + 1) continue;

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
                if ((Cell1[cid] == oid + 1 && Cell1[neighbor_cids[n]] == 0) ||
                    (Cell2[cid] == oid + 1 && Cell2[neighbor_cids[n]] == 0)) {
                    const int d = (n % 2 == 0);
                    const REAL sign = (n % 2 == 0) ? -1. : 1.;

                    if (n < 2 && used_vxids.count(Cid(i - d, j, k)) > 0) continue;
                    if (n >= 2 && n < 4 && used_vyids.count(Cid(i, j - d, k)) > 0) continue;
                    if (n >= 4 && n < 6 && used_vzids.count(Cid(i, j, k - d)) > 0) continue;

                    if (n < 2) {
                        shader_map.push_back(3 * Cid(i - d, j, k) + 0);
                        bn_vec.emplace_back(sign * Eigen::Vector3<REAL>{1., 0., 0.});
                        used_vxids.insert(Cid(i - d, j, k));
                    } else if (n < 4) {
                        shader_map.push_back(3 * Cid(i, j - d, k) + 1);
                        bn_vec.emplace_back(sign * Eigen::Vector3<REAL>{0., 1., 0.});
                        used_vyids.insert(Cid(i, j - d, k));
                    } else if (n < 6) {
                        shader_map.push_back(3 * Cid(i, j, k - d) + 2);
                        bn_vec.emplace_back(sign * Eigen::Vector3<REAL>{0., 0., 1.});
                        used_vzids.insert(Cid(i, j, k - d));
                    }
                    b_vec.emplace_back(p.cast<REAL>() + (0.5 * Params.Dx) * bn_vec[bid]);
                    bid += 1;
                }
            }
        } // loop over cells

        Eigen::MatrixX<REAL> b(b_vec.size(), 3);
        Eigen::MatrixX<REAL> bn(bn_vec.size(), 3);
        for (int row = 0; row < b_vec.size(); ++row) {
            b.row(row) = b_vec[row];
            bn.row(row) = bn_vec[row];
        }
        Objects[oid]->SetSamplePoints(b, bn);

    } // loop over objects

    used_vxids.clear(), used_vyids.clear(), used_vzids.clear();
    for (int oid = 0; oid < Objects.size(); ++oid) {
        if (Objects[oid]->Type != ShaderClass::Point) continue;

        int bid = 0;
        std::vector<Eigen::Vector3<REAL>> b_vec;
        std::vector<Eigen::Vector3<REAL>> bn_vec;
        for (int cid = 0; cid < Cell2.size(); ++cid) {
            if (Cell2[cid] != oid + 1) continue;
            Cell2[cid] = 0; // We don't want to rasterize point sources, so set cell back to 0

            const int i = cid % Params.Nx;
            const int j = (cid / Params.Nx) % Params.Ny;
            const int k = (cid / Params.Nx) / Params.Ny;

            if (i == 0 || i >= Params.Nx - 1 || j == 0 || j >= Params.Ny - 1 || k == 0 || k >= Params.Nz - 1) continue;

            const Eigen::Vector3<REAL> p = Pos(i, j, k) - Offsets[oid]; // world-space to object-space

            for (int n = 0; n < 6; ++n) {
                const int d = (n % 2 == 0);
                const REAL sign = (n % 2 == 0) ? -1. : 1.;

                if (n < 2 && used_vxids.count(Cid(i - d, j, k)) > 0) continue;
                if (n >= 2 && n < 4 && used_vyids.count(Cid(i, j - d, k)) > 0) continue;
                if (n >= 4 && n < 6 && used_vzids.count(Cid(i, j, k - d)) > 0) continue;

                if (n < 2) {
                    shader_map.push_back(-3 * Cid(i - d, j, k) - 0);
                    bn_vec.emplace_back(sign * Eigen::Vector3<REAL>{1., 0., 0.});
                    used_vxids.insert(Cid(i - d, j, k));
                } else if (n < 4) {
                    shader_map.push_back(-3 * Cid(i, j - d, k) - 1);
                    bn_vec.emplace_back(sign * Eigen::Vector3<REAL>{0., 1., 0.});
                    used_vyids.insert(Cid(i, j - d, k));
                } else if (n < 6) {
                    shader_map.push_back(-3 * Cid(i, j, k - d) - 2);
                    bn_vec.emplace_back(sign * Eigen::Vector3<REAL>{0., 0., 1.});
                    used_vzids.insert(Cid(i, j, k - d));
                }
                b_vec.emplace_back(p.cast<REAL>() + (0.5 * Params.Dx) * bn_vec[bid]);
                bid += 1;
            }
        } // loop over cells

        Eigen::MatrixX<REAL> b(b_vec.size(), 3);
        Eigen::MatrixX<REAL> bn(bn_vec.size(), 3);
        for (int row = 0; row < b_vec.size(); ++row) {
            b.row(row) = b_vec[row];
            bn.row(row) = bn_vec[row];
        }
        Objects[oid]->SetSamplePoints(b, bn);

    } // loop over objects

    // Set both raster and shader memory
    Cell.Upload(Cell2.data(), Cell2.size() * sizeof(int));

    NShaderPoints = shader_map.size();
    if (NShaderPoints > MaxNShaderPoints) { // Reallocate
        MaxNShaderPoints = NShaderPoints;

        ShaderData.Resize(size_t(NShaderPoints) * NShaderSamples * sizeof(REAL));
        ShaderMap.Resize(size_t(NShaderPoints) * sizeof(int));
    }
    ShaderData.Zero(size_t(NShaderPoints) * NShaderSamples * sizeof(REAL));
    ShaderMap.Upload(shader_map.data(), NShaderPoints * sizeof(int));
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
    ctx.Dispatch("prepare_fresh_cell", shader_blocks, shader_threads, {&Px, &Py, &Pz, &ShaderData, &ShaderMap}, &prep_params, sizeof(prep_params));

    const FreshCellPressureParams fc_params{Params.Nx, Params.Ny, Params.Nz, Params.Rho * Params.Dx};
    ctx.Dispatch("fresh_cell_pressure", fdtd_blocks, fdtd_threads, {&P, &Px, &Py, &Pz, &Beta}, &fc_params, sizeof(fc_params));

    const ClearSolidParams cs_params{Params.Nx, Params.Ny, Params.Nz};
    ctx.Dispatch("clear_solid", fdtd_blocks, fdtd_threads, {&Vx, &Vy, &Vz, &Px, &Py, &Pz, &Beta}, &cs_params, sizeof(cs_params));
}

void WaveBlender::FreshCellVelocity() {
    // Determine fresh cells (based on Cell1 and Cell2 difference)
    std::set<int> fresh_cids;
    for (int cid = 0; cid < Cell2.size(); ++cid) {
        if (Cell2[cid] == 0 && Cell1[cid] > 0 && Cell1[cid] != CavityInterior) {
            if (const int oid = Cell1[cid] - 1; Objects[oid]->Type != ShaderClass::Point) fresh_cids.insert(cid);
        }
    }
    std::cout << "  # Fresh Cells = " << fresh_cids.size() << std::endl;

    // Prepare velocity fresh cells: determine interior faces
    std::map<int, int> interior_vxids, interior_vyids, interior_vzids;
    int vxcount = 0, vycount = 0, vzcount = 0;
    for (const int cid : fresh_cids) {
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
            if ((n == 0 || n == 1) && interior_vxids.count(Cid(i - d, j, k)) == 0) { // x-direction
                interior_vxids.emplace(Cid(i - d, j, k), vxcount);
                vxcount += 1;
            } else if ((n == 2 || n == 3) && interior_vyids.count(Cid(i, j - d, k)) == 0) { // y-direction
                interior_vyids.emplace(Cid(i, j - d, k), vycount);
                vycount += 1;
            } else if ((n == 4 || n == 5) && interior_vzids.count(Cid(i, j, k - d)) == 0) { // z-direction
                interior_vzids.emplace(Cid(i, j, k - d), vzcount);
                vzcount += 1;
            }
        }
    }

    // Build least-squares matrix A and vector g, reading velocities directly from the shared buffers.
    // (The reference does per-element cudaMemcpys here — As<REAL>() synchronizes once instead.)
    const int n_dof = vxcount + vycount + vzcount;
    Eigen::MatrixX<REAL> a = Eigen::MatrixX<REAL>::Zero(fresh_cids.size(), n_dof);
    Eigen::VectorX<REAL> g = Eigen::VectorX<REAL>::Zero(fresh_cids.size());

    const auto *vx_host = Vx.As<REAL>();
    const auto *vy_host = Vy.As<REAL>();
    const auto *vz_host = Vz.As<REAL>();

    int row = 0;
    for (const int cid : fresh_cids) {
        const int i = cid % Params.Nx;
        const int j = (cid / Params.Nx) % Params.Ny;
        const int k = (cid / Params.Nx) / Params.Ny;

        for (int n = 0; n < 6; ++n) {
            const int d = (n % 2 == 0);
            const REAL sign = (n % 2 == 0) ? -1. : 1.;

            int col = -1;
            REAL vn = 0.;
            if (n < 2) { // x-direction
                const int vnid = Cid(i - d, j, k);
                if (interior_vxids.count(vnid) > 0) col = interior_vxids[vnid];
                vn = vx_host[vnid];
            } else if (n < 4) { // y-direction
                const int vnid = Cid(i, j - d, k);
                if (interior_vyids.count(vnid) > 0) col = vxcount + interior_vyids[vnid];
                vn = vy_host[vnid];
            } else if (n < 6) { // z-direction
                const int vnid = Cid(i, j, k - d);
                if (interior_vzids.count(vnid) > 0) col = vxcount + vycount + interior_vzids[vnid];
                vn = vz_host[vnid];
            }

            if (col != -1) a(row, col) = sign;
            else g[row] -= sign * vn;
        }
        row += 1;
    }

    if (!fresh_cids.empty() && n_dof > 0) {
        const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixX<REAL>> qr{a}; // least squares
        const Eigen::VectorX<REAL> u = qr.solve(g);

        // Write fresh cell velocity solve results back to the shared buffers
        auto *vx_out = Vx.As<REAL>();
        auto *vy_out = Vy.As<REAL>();
        auto *vz_out = Vz.As<REAL>();
        for (const auto &[vxid, col] : interior_vxids) vx_out[vxid] = u[col]; // x-direction
        for (const auto &[vyid, col] : interior_vyids) vy_out[vyid] = u[vxcount + col]; // y-direction
        for (const auto &[vzid, col] : interior_vzids) vz_out[vzid] = u[vxcount + vycount + col]; // z-direction

        std::cout << "residual: " << (a * u - g).norm() << std::endl;
    }
}

// Section 6.2.2 from [Xue et al. 2024]
void WaveBlender::ShaderReInit() {
    const Dim3 shader_threads{32, 1, 1};
    const Dim3 shader_blocks{uint32_t(NShaderPoints + 31) / 32, 1, 1};

    const ShaderReInitParams params{NShaderSamples, NShaderPoints};
    MetalContext::Get().Dispatch("shader_reinit", shader_blocks, shader_threads, {&Vx, &Vy, &Vz, &ShaderData, &ShaderMap}, &params, sizeof(params));
}
