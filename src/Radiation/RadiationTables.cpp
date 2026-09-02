#include "RadiationTables.h"

#include "Parallel.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <thread>
#include <unordered_map>

using Eigen::Vector3d, Eigen::Vector3i;

void RadBuildTimes::Restart() {
    static const bool on = std::getenv("ACOUSTIC_RAD_BUILD_TIMES") != nullptr;
    On = on;
    Phases.clear();
    Last = On ? Now() : 0.;
}

void RadBuildTimes::Mark(const char *phase) {
    if (!On) return;
    const double now = Now();
    Phases.emplace_back(phase, now - Last);
    Last = now;
}

void RadBuildTimes::Report() const {
    if (!On) return;
    double total = 0.;
    for (const auto &[phase, seconds] : Phases) total += seconds;
    std::printf("[radiation build]");
    for (const auto &[phase, seconds] : Phases) std::printf(" %s %.3f", phase, seconds);
    std::printf(" | total %.3f\n", total);
}

RadBuildTimes &BuildTimes() {
    static thread_local RadBuildTimes times;
    return times;
}

void RadiationGrid::Finalize() {
    Tau = CflFraction * H / (C * std::numbers::sqrt3);

    // Same quadratic-ramp weights as WaveBlender::InitializePml, with zero air damping.
    const int max_dist = std::max({Nx, Ny, Nz}) / 2 + 2;
    PmlNv.assign(max_dist, 1.);
    PmlDv.assign(max_dist, 1. / (1. + Damping));
    PmlNp.assign(max_dist, 1.);
    PmlDp.assign(max_dist, 1.);
    for (int dist = 0; dist < PmlWidth; ++dist) {
        double weight = 0.5 * std::pow((double(PmlWidth) - dist) / PmlWidth, 2.); // velocity
        PmlNv[dist] = 1. - weight;
        PmlDv[dist] = 1. / (1. + weight);
        weight = 0.5 * std::pow((double(PmlWidth) - dist - 0.5) / PmlWidth, 2.); // pressure
        PmlNp[dist] = 1. - weight;
        PmlDp[dist] = 1. / (1. + weight);
    }
}

RadiationGrid RadiationGrid::Cubic(int res, double domain, int pml_width) {
    RadiationGrid g;
    g.PmlWidth = pml_width;
    g.Nx = g.Ny = g.Nz = res + 2 * pml_width;
    g.H = domain / res;
    g.Origin = Vector3d::Constant(-0.5 * domain + (0.5 - pml_width) * g.H);
    g.Finalize();
    return g;
}

Vector3i RadiationGrid::CellOf(const Vector3d &x) const {
    Vector3i c;
    for (int d = 0; d < 3; ++d) c[d] = int(std::floor((x[d] - Origin[d]) / H + 0.5));
    return c;
}

namespace {
// The trilinear cell corners of `x` on `grid`, with their blending weights.
struct Corners {
    Vector3i Cell[8];
    double Gamma[8];
};
Corners CornersOf(const RadiationGrid &grid, const Vector3d &x) {
    int base[3];
    double frac[3];
    for (int d = 0; d < 3; ++d) {
        const double f = (x[d] - grid.Origin[d]) / grid.H;
        base[d] = int(std::floor(f));
        frac[d] = f - base[d];
    }
    Corners out;
    for (int c = 0; c < 8; ++c) {
        const int di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
        out.Cell[c] = {base[0] + di, base[1] + dj, base[2] + dk};
        out.Gamma[c] = (di ? frac[0] : 1. - frac[0]) * (dj ? frac[1] : 1. - frac[1]) * (dk ? frac[2] : 1. - frac[2]);
    }
    return out;
}
} // namespace

int RadiationTables::EntryOf(const RadiationGrid &grid, int cell, int elem) const {
    const int band = CellBand[cell];
    return band < 0 ? -1 : CellElem.Find(band, elem);
}

void RadiationTables::AssignCells(const RadiationGrid &grid, const RadiationMesh &mesh) {
    const int m = mesh.NumElems(), w = grid.PmlWidth;
    const int n[3]{grid.Nx, grid.Ny, grid.Nz};
    ElemCell.resize(m);
    for (int e = 0; e < m; ++e) {
        ElemCell[e] = grid.CellOf(mesh.Centroids[e]);
        for (int d = 0; d < 3; ++d) assert(ElemCell[e][d] >= w + grid.R2 + 2 && ElemCell[e][d] < n[d] - w - grid.R2 - 2);
    }
    ElemNear.assign(m, {});
    ParallelFor(m, 16, [&](size_t i) {
        for (int e = 0; e < m; ++e)
            if (RadiationGrid::Cheb(ElemCell[i], ElemCell[e]) <= grid.R2) ElemNear[i].push_back(e);
    });
}

void RadiationTables::Build(const RadiationGrid &grid, const CqmContour &contour, const RadiationMesh &mesh, const QuadPolicy &policy, const std::vector<Vector3d> &listeners, bool staged) {
    const int m = mesh.NumElems(), w = grid.PmlWidth;
    const size_t cells = grid.NumCells();
    const int n[3]{grid.Nx, grid.Ny, grid.Nz};

    // Candidate lists: every element within Chebyshev R2 + 1 of a cell. These are the
    // elements whose set membership the stencils below test — ints only, no weights.
    const int reach = grid.R2 + 1;
    std::vector<int> cand_row(cells, -1);
    std::vector<std::vector<int>> cand;
    for (int e = 0; e < m; ++e) {
        const auto &ec = ElemCell[e];
        for (int dk = -reach; dk <= reach; ++dk)
            for (int dj = -reach; dj <= reach; ++dj)
                for (int di = -reach; di <= reach; ++di) {
                    const int cid = grid.Cid(ec[0] + di, ec[1] + dj, ec[2] + dk);
                    if (cand_row[cid] < 0) {
                        cand_row[cid] = int(cand.size());
                        cand.emplace_back();
                    }
                    cand[cand_row[cid]].push_back(e);
                }
    }
    BuildTimes().Mark("candidates");
    auto candidates_of = [&](int cid) -> const std::vector<int> & {
        static const std::vector<int> Empty;
        return cand_row[cid] < 0 ? Empty : cand[cand_row[cid]];
    };

    // Pass 1: every stencil, recorded as (target cell, source element) pairs. Only these
    // pairs get weights; the dense neighbourhood they are drawn from would be several
    // times larger, because the set differences are shells.
    struct Pending {
        int Cell, Elem;
        bool Positive;
    };
    std::vector<Pending> corr_pending, interp_pending, listener_pending;
    CellBand.assign(cells, -1);
    std::vector<int> row_cells;

    // Laplacian correction refs (Eq. 12): for cell c and neighbor nb,
    // p_nb(F_c) = p_nb(F_nb) + p_nb(N_nb \ N_c) - p_nb(N_c \ N_nb), evaluated at nb.
    std::vector<uint8_t> candidate(cells, 0);
    for (int e = 0; e < m; ++e) {
        const auto &ec = ElemCell[e];
        const int r = grid.R1 + 1;
        for (int dk = -r; dk <= r; ++dk)
            for (int dj = -r; dj <= r; ++dj)
                for (int di = -r; di <= r; ++di) candidate[grid.Cid(ec[0] + di, ec[1] + dj, ec[2] + dk)] = 1;
    }
    // Counted then written, in parallel over the cells: what a cell owes depends on nothing
    // but its own neighbourhood.
    std::vector<int> corr_cells;
    for (size_t cid = 0; cid < cells; ++cid) {
        if (!candidate[cid]) continue;
        const Vector3i c = grid.CellCoords(int(cid));
        for (int d = 0; d < 3; ++d) assert(c[d] >= w && c[d] < n[d] - w); // corrections stay out of the shell
        corr_cells.push_back(int(cid));
    }
    // The elements of `cid`'s candidate list that lie in exactly one of two near sets,
    // tagged by membership in the second and targeted at `target`. Both the Laplacian
    // corrections and the interpolation re-basing below are that same difference, and both
    // walk it twice — once to count, with `out` null, once to write.
    const auto set_difference = [&](int cid, const Vector3i &a, int ra, const Vector3i &b, int rb, int target, Pending *out) {
        int found = 0;
        for (const int e : candidates_of(cid)) {
            const bool in_a = RadiationGrid::Cheb(ElemCell[e], a) <= ra, in_b = RadiationGrid::Cheb(ElemCell[e], b) <= rb;
            if (in_a == in_b) continue;
            if (out) out[found] = {target, e, in_b};
            ++found;
        }
        return found;
    };
    const auto cell_refs = [&](int cid, Pending *out) {
        const Vector3i c = grid.CellCoords(cid);
        int found = 0;
        for (int axis = 0; axis < 3; ++axis) {
            for (int dir = -1; dir <= 1; dir += 2) {
                Vector3i nb = c;
                nb[axis] += dir;
                // Elements in either near set are within R1 + 1 of c, hence candidates of c.
                found += set_difference(cid, c, grid.R1, nb, grid.R1, grid.Cid(nb), out ? out + found : nullptr);
            }
        }
        return found;
    };
    std::vector<int> corr_begin(corr_cells.size() + 1, 0);
    ParallelFor(corr_cells.size(), 8, [&](size_t k) { corr_begin[k + 1] = cell_refs(corr_cells[k], nullptr); });
    for (size_t k = 0; k < corr_cells.size(); ++k) corr_begin[k + 1] += corr_begin[k];
    corr_pending.resize(corr_begin.back());
    ParallelFor(corr_cells.size(), 8, [&](size_t k) { cell_refs(corr_cells[k], corr_pending.data() + corr_begin[k]); });

    CorrCell.clear();
    CorrRefs.clear();
    CorrBegin.assign(1, 0);
    for (size_t k = 0; k < corr_cells.size(); ++k) {
        if (corr_begin[k + 1] == corr_begin[k]) continue;
        CorrCell.push_back(corr_cells[k]);
        CorrBegin.push_back(corr_begin[k + 1]);
    }

    BuildTimes().Mark("corr");
    // Far-field interpolation onto element centroids (Eqs. 19-20): corner far values are
    // re-based from the corner's far set onto the element's (F_i) before blending. Counted
    // then written like the corrections above, over independent corners.
    Interp.assign(size_t(m) * 8, {});
    const auto corner_refs = [&](int i, const Vector3i &corner, int cid, Pending *out) {
        return set_difference(cid, ElemCell[i], grid.R2, corner, grid.R1, cid, out);
    };
    std::vector<int> corner_begin(size_t(m) * 8 + 1, 0);
    ParallelFor(m, 8, [&](size_t i) {
        const Corners corners = CornersOf(grid, mesh.Centroids[i]);
        for (int c = 0; c < 8; ++c) corner_begin[i * 8 + c + 1] = corner_refs(int(i), corners.Cell[c], grid.Cid(corners.Cell[c]), nullptr);
    });
    for (size_t k = 0; k < size_t(m) * 8; ++k) corner_begin[k + 1] += corner_begin[k];
    interp_pending.resize(corner_begin.back());
    ParallelFor(m, 8, [&](size_t i) {
        const Corners corners = CornersOf(grid, mesh.Centroids[i]);
        for (int c = 0; c < 8; ++c) {
            auto &ic = Interp[i * 8 + c];
            ic.Cell = grid.Cid(corners.Cell[c]);
            ic.Gamma = float(corners.Gamma[c]);
            ic.RefBegin = corner_begin[i * 8 + c];
            ic.RefEnd = corner_begin[i * 8 + c + 1];
            corner_refs(int(i), corners.Cell[c], ic.Cell, interp_pending.data() + ic.RefBegin);
        }
    });

    BuildTimes().Mark("interp");
    // Listener stencils: the corner cells' far-field state plus their own near elements.
    ListenerCorners.assign(listeners.size() * 8, {});
    for (size_t l = 0; l < listeners.size(); ++l) {
        const Corners corners = CornersOf(grid, listeners[l]);
        for (int c = 0; c < 8; ++c) {
            auto &ic = ListenerCorners[l * 8 + c];
            ic.Cell = grid.Cid(corners.Cell[c]);
            ic.Gamma = float(corners.Gamma[c]);
            ic.RefBegin = int(listener_pending.size());
            for (const int e : candidates_of(ic.Cell)) {
                if (RadiationGrid::Cheb(ElemCell[e], corners.Cell[c]) > grid.R1) continue;
                listener_pending.push_back({ic.Cell, e, true});
            }
            ic.RefEnd = int(listener_pending.size());
        }
    }

    BuildTimes().Mark("listeners");
    // The rows the stencils named, in the order they named them. Assigning a row is a scan
    // over the pending lists; filling one is a scatter, so only the scan stays serial.
    const std::array<const std::vector<Pending> *, 3> pending{&corr_pending, &interp_pending, &listener_pending};
    for (const auto *list : pending)
        for (const auto &p : *list)
            if (CellBand[p.Cell] < 0) {
                CellBand[p.Cell] = int(row_cells.size());
                row_cells.push_back(p.Cell);
            }
    const int n_rows = int(row_cells.size());
    std::vector<std::vector<int>> row_elems(n_rows);
    {
        const size_t n_parts = std::max<size_t>(1, std::thread::hardware_concurrency());
        // Each part takes the same slice of all three lists, counts what its pairs owe each
        // row, and then writes them. Rows are sorted below, so the order within one is free.
        const auto over_part = [&](size_t p, auto &&fn) {
            for (const auto *list : pending) {
                const size_t per_part = std::max<size_t>(1, (list->size() + n_parts - 1) / n_parts);
                for (size_t k = p * per_part, end = std::min(list->size(), (p + 1) * per_part); k < end; ++k) fn((*list)[k]);
            }
        };
        std::vector<int> counts(n_parts * n_rows, 0), cursor(n_parts * n_rows, 0);
        ParallelFor(n_parts, 1, [&](size_t p) {
            int *part = counts.data() + p * n_rows;
            over_part(p, [&](const Pending &q) { ++part[CellBand[q.Cell]]; });
        });
        ParallelFor(n_rows, 64, [&](size_t r) {
            int total = 0;
            for (size_t p = 0; p < n_parts; ++p) {
                cursor[p * n_rows + r] = total;
                total += counts[p * n_rows + r];
            }
            row_elems[r].resize(total);
        });
        ParallelFor(n_parts, 1, [&](size_t p) {
            int *slot = cursor.data() + p * n_rows;
            over_part(p, [&](const Pending &q) {
                const int r = CellBand[q.Cell];
                row_elems[r][slot[r]++] = q.Elem;
            });
        });
    }

    BuildTimes().Mark("rows");
    // Pass 2: weights for the pairs the stencils actually named, rows sorted by element so
    // ref resolution is a binary search.
    ParallelFor(n_rows, 8, [&](size_t r) {
        auto &row = row_elems[r];
        std::ranges::sort(row);
        row.erase(std::ranges::unique(row).begin(), row.end());
    });
    {
        PairTableBuilder builder{n_rows, staged};
        ParallelFor(n_rows, 1, [&](size_t r) {
            const Vector3d x = grid.CellCenter(grid.CellCoords(row_cells[r]));
            PairSeries series; // reused down the row, so its windows allocate once rather than per pair
            for (const int e : row_elems[r]) {
                ComputePairSeries(contour, x, mesh, e, false, policy, series);
                builder.Add(int(r), e, series);
            }
        });
        CellElem = builder.Finish(m);
    }

    BuildTimes().Mark("cell_weights");
    // Pass 3: resolve every stencil's pairs to entry indices.
    const auto resolve = [&](const std::vector<Pending> &pending, std::vector<RadRef> &out) {
        out.resize(pending.size());
        ParallelFor(pending.size(), 4096, [&](size_t k) {
            const int entry = EntryOf(grid, pending[k].Cell, pending[k].Elem);
            assert(entry >= 0);
            out[k] = {pending[k].Positive ? entry : entry | RadRefNegative};
        });
    };
    resolve(corr_pending, CorrRefs);
    resolve(interp_pending, InterpRefs);
    resolve(listener_pending, ListenerRefs);
    BuildTimes().Mark("resolve");
}

void BuildRadiation(const RadiationGrid &grid, const RadiationMesh &mesh, const QuadPolicy &policy, const std::vector<Vector3d> &listeners, Tdbem &bem, RadiationTables &tables, const PairTable *previous, bool staged) {
    BuildTimes().Restart();
    tables.AssignCells(grid, mesh);
    BuildTimes().Mark("assign_cells");
    // Furthest retardation either table reaches, for the contour's delay kernels: a cell
    // target sits within R2 + 1 cells of the element's cell, each centre within half a cell of
    // its own, and the quadrature samples anywhere inside the element. One spare cell of
    // margin, since overshooting only lengthens the kernel rows while undershooting drops
    // pairs back onto the per-pair transform.
    double extent = 0.;
    for (int e = 0; e < mesh.NumElems(); ++e)
        for (int v = 0; v < 3; ++v) extent = std::max(extent, (mesh.Vertices[mesh.Tris[e][v]] - mesh.Centroids[e]).norm());
    const double theta_max = ((grid.R2 + 2.5) * std::numbers::sqrt3 * grid.H + extent) / (grid.C * grid.Tau);
    bem.Init(mesh, grid.Tau, grid.C, tables.ElemNear, policy, theta_max, previous, staged);
    BuildTimes().Mark("elem_table");
    tables.Build(grid, *bem.Contour, bem.Mesh, policy, listeners, staged);
    bem.EnsureHistLen(std::max(bem.ElemElem.MaxLagEnd(), tables.CellElem.MaxLagEnd()), 0);
    BuildTimes().Report();
}

void SeedRestState(Tdbem &bem, double tau, const std::function<double(int, double)> &neumann) {
    for (int e = 0; e < bem.NumElems(); ++e) {
        bem.GAt(0, e) = neumann(e, 0.);
        bem.GAt(1, e) = neumann(e, tau);
    }
    bem.SolveDirichletDiag(0, nullptr);
}

RebasePlan PlanRebase(const RadiationGrid &grid, const CqmContour &contour, const RadiationMesh &old_mesh, const std::vector<Vector3i> &old_cells, const std::vector<Vector3i> &new_cells, const QuadPolicy &policy) {
    const int m = int(old_cells.size());
    // Only elements that changed cell can change any membership, and only within R1 of
    // either end of the move.
    std::unordered_map<int, std::vector<std::pair<int, bool>>> per_cell;
    for (int e = 0; e < m; ++e) {
        if (old_cells[e] == new_cells[e]) continue;
        std::vector<int> touched;
        for (const auto &centre : {old_cells[e], new_cells[e]}) {
            for (int dk = -grid.R1; dk <= grid.R1; ++dk)
                for (int dj = -grid.R1; dj <= grid.R1; ++dj)
                    for (int di = -grid.R1; di <= grid.R1; ++di)
                        touched.push_back(grid.Cid(centre[0] + di, centre[1] + dj, centre[2] + dk));
        }
        std::ranges::sort(touched);
        touched.erase(std::ranges::unique(touched).begin(), touched.end());
        for (const int cid : touched) {
            const Vector3i c = grid.CellCoords(cid);
            const bool in_old = RadiationGrid::Cheb(old_cells[e], c) <= grid.R1,
                       in_new = RadiationGrid::Cheb(new_cells[e], c) <= grid.R1;
            if (in_old == in_new) continue;
            per_cell[cid].emplace_back(e, in_old); // leaving the near set rejoins the far field
        }
    }

    RebasePlan plan;
    plan.Begin.push_back(0);
    std::vector<std::vector<std::pair<int, bool>>> rows;
    for (const auto &[cid, pairs] : per_cell) {
        plan.Cells.push_back(cid);
        rows.push_back(pairs);
        plan.Begin.push_back(plan.Begin.back() + int(pairs.size()));
    }
    PairTableBuilder builder{int(plan.Cells.size())};
    ParallelFor(plan.Cells.size(), 1, [&](size_t r) {
        const Vector3d x = grid.CellCenter(grid.CellCoords(plan.Cells[r]));
        PairSeries series;
        for (const auto &[e, positive] : rows[r]) {
            ComputePairSeries(contour, x, old_mesh, e, false, policy, series);
            builder.Add(int(r), e, series);
        }
    });
    plan.Table = builder.Finish(m);
    for (size_t r = 0; r < rows.size(); ++r) {
        for (const auto &[e, positive] : rows[r]) {
            const int entry = plan.Table.Find(int(r), e);
            plan.Refs.push_back({positive ? entry : entry | RadRefNegative});
        }
    }
    return plan;
}

double RadiationTables::CheckSetAlgebra(const RadiationGrid &grid, const Tdbem &bem, int n_cells, int step) const {
    double worst = 0.;
    const size_t stride = std::max<size_t>(1, CorrCell.size() / std::max(1, n_cells));
    for (size_t c = 0; c < CorrCell.size(); c += stride) {
        double table = 0.;
        for (int r = CorrBegin[c]; r < CorrBegin[c + 1]; ++r)
            table += RefSign(CorrRefs[r]) * bem.Convolve(CellElem, RefEntry(CorrRefs[r]), step, false);
        const Vector3i cc = grid.CellCoords(CorrCell[c]);
        double direct = 0.;
        for (int axis = 0; axis < 3; ++axis) {
            for (int dir = -1; dir <= 1; dir += 2) {
                Vector3i nb = cc;
                nb[axis] += dir;
                const int nb_cid = grid.Cid(nb);
                for (int e = 0; e < bem.NumElems(); ++e) {
                    const bool in_c = RadiationGrid::Cheb(ElemCell[e], cc) <= grid.R1,
                               in_nb = RadiationGrid::Cheb(ElemCell[e], nb) <= grid.R1;
                    if (in_c == in_nb) continue;
                    const int entry = EntryOf(grid, nb_cid, e);
                    assert(entry >= 0);
                    direct += (in_nb ? 1. : -1.) * bem.Convolve(CellElem, entry, step, false);
                }
            }
        }
        worst = std::max(worst, std::abs(table - direct) / std::max({std::abs(direct), std::abs(table), 1e-30}));
    }
    return worst;
}
